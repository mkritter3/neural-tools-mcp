"""
ARQ Worker for Processing Embedding Jobs

Provides resilient, async job processing for embedding requests with:
- Idempotency through Redis caching
- Dead letter queue integration for failures  
- Proper error handling and retry logic
- Context management for service dependencies
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any

from arq import Worker
from arq.connections import RedisSettings

logger = logging.getLogger(__name__)

async def process_embedding_job(ctx: Dict[str, Any], text: str, model: str = "nomic-v2") -> Dict[str, Any]:
    """
    ARQ worker function for processing embeddings with idempotency and error handling
    
    Args:
        ctx: ARQ context containing initialized services
        text: Text to embed
        model: Model identifier for embedding service
        
    Returns:
        Dict containing embedding result or error information
        
    Raises:
        Exception: Propagated to ARQ for retry handling
    """
    job_id = ctx.get('job_id', 'unknown')
    
    try:
        # Get services from context
        embedding_service = ctx['embedding_service']
        redis_cache = ctx['redis_cache']
        dlq_service = ctx['dlq_service']
        
        logger.info(f"Processing embedding job {job_id} for model {model} (text length: {len(text)})")
        
        # Check if result already exists (idempotency)
        result_key = f"l9:prod:neural_tools:job_result:{job_id}"
        cached_result = await redis_cache.get(result_key)
        if cached_result:
            logger.info(f"Job {job_id} already completed, returning cached result")
            return json.loads(cached_result)
        
        # Process embedding using the real service
        start_time = time.time()
        embedding = await embedding_service.get_embedding(text)
        processing_time = time.time() - start_time
        
        # Validate embedding result
        if not embedding or not isinstance(embedding, list):
            raise ValueError(f"Invalid embedding result: {type(embedding)}")
            
        # Store result with TTL
        result = {
            "status": "success", 
            "embedding": embedding, 
            "source": "worker",
            "job_id": job_id,
            "model": model,
            "processing_time_seconds": round(processing_time, 3),
            "completed_at": int(time.time())
        }
        
        # Cache with 1 hour TTL
        await redis_cache.setex(result_key, 3600, json.dumps(result))
        
        logger.info(f"Job {job_id} completed successfully in {processing_time:.3f}s")
        return result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        
        # Add to dead letter queue for analysis
        try:
            dlq_service = ctx.get('dlq_service')
            if dlq_service:
                await dlq_service.add_to_dlq(
                    job_data={
                        'job_id': job_id,
                        'text': text,
                        'model': model
                    },
                    error=e,
                    retry_count=ctx.get('job_try', 0)
                )
        except Exception as dlq_error:
            logger.error(f"Failed to add job {job_id} to DLQ: {dlq_error}")
        
        # Re-raise for ARQ retry logic
        raise

async def process_embedding_batch(ctx: Dict[str, Any], payload: List[Dict]) -> Dict[str, Any]:
    """
    ADR-0083: ARQ worker function for processing batch of chunks with full lifecycle management

    Args:
        ctx: ARQ context containing initialized services
        payload: List of {"chunk_id": str, "text": str} items

    Returns:
        Success response with processing stats
    """
    logger.info(f"🔄 Processing embedding batch of {len(payload)} chunks")

    # Get services from context
    nomic_service = ctx.get("nomic")
    neo4j_service = ctx.get("neo4j")

    if not nomic_service or not neo4j_service:
        raise RuntimeError("Required services not available in worker context")

    chunk_ids = [item["chunk_id"] for item in payload]
    texts = [item["text"] for item in payload]

    try:
        # Get embeddings from Nomic in single API call
        embeddings = await nomic_service.get_embeddings(texts)

        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings) if embeddings else 0}")

        # Update all chunks in single Neo4j transaction
        update_data = [
            {"chunk_id": cid, "embedding": emb}
            for cid, emb in zip(chunk_ids, embeddings)
        ]

        # Use Neo4j transaction to update all chunks atomically
        cypher = """
        UNWIND $batch as row
        MATCH (c:Chunk {id: row.chunk_id})
        SET c.embedding = row.embedding
        RETURN count(c) as updated_count
        """

        result = await neo4j_service.execute_cypher(cypher, {"batch": update_data})
        updated_count = result.get("data", [{}])[0].get("updated_count", 0)

        logger.info(f"✅ Successfully processed embeddings for {updated_count} chunks")

        return {
            "status": "success",
            "chunks_processed": len(chunk_ids),
            "embeddings_generated": len(embeddings),
            "chunks_updated": updated_count
        }

    except Exception as e:
        logger.error(f"❌ Batch embedding processing failed: {e}")
        raise

class EmbeddingWorker:
    """
    ARQ Worker manager for embedding processing
    
    Handles service initialization, context management, and graceful shutdown
    """
    
    def __init__(self, service_container):
        self.container = service_container
        self.initialized = False
        
    async def startup(self, ctx: Dict[str, Any]):
        """Initialize worker context with all required services"""
        try:
            logger.info("Initializing ARQ embedding worker context...")
            
            # Initialize service container if needed
            if not self.container.initialized:
                init_result = await self.container.initialize_all_services()
                if not init_result.get('success'):
                    raise RuntimeError(f"Service container initialization failed: {init_result}")
            
            # Set up context with initialized services
            ctx['embedding_service'] = self.container.nomic
            ctx['redis_cache'] = await self.container.get_redis_cache_client()
            ctx['dlq_service'] = await self.container.get_dlq_service()
            
            # Validate critical services
            if not ctx['embedding_service']:
                raise RuntimeError("Embedding service not available")
            if not ctx['redis_cache']:
                raise RuntimeError("Redis cache client not available")
            if not ctx['dlq_service']:
                raise RuntimeError("DLQ service not available")
                
            self.initialized = True
            logger.info("✅ ARQ embedding worker context initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ARQ worker context: {e}")
            raise
        
    async def shutdown(self, ctx: Dict[str, Any]):
        """Cleanup worker resources gracefully"""
        try:
            logger.info("Shutting down ARQ embedding worker...")
            
            # Close Redis connections if they have close methods
            redis_cache = ctx.get('redis_cache')
            if redis_cache and hasattr(redis_cache, 'close'):
                await redis_cache.close()
                
            # Clean up other services as needed
            dlq_service = ctx.get('dlq_service')
            if dlq_service and hasattr(dlq_service, 'close'):
                await dlq_service.close()
                
            self.initialized = False
            logger.info("✅ ARQ embedding worker shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during worker shutdown: {e}")

async def create_embedding_worker(project_name: str = "default") -> Worker:
    """
    Factory function to create configured ARQ worker
    
    Args:
        project_name: Project identifier for service container
        
    Returns:
        Configured ARQ Worker instance
    """
    # Import here to avoid circular dependencies
    from servers.services.service_container import ServiceContainer
    
    # Initialize service container
    container = ServiceContainer(project_name)
    worker_instance = EmbeddingWorker(container)
    
    # Redis settings for job queue
    redis_settings = RedisSettings(
        host=os.getenv('REDIS_QUEUE_HOST', 'localhost'),
        port=int(os.getenv('REDIS_QUEUE_PORT', 46380)),
        password=os.getenv('REDIS_QUEUE_PASSWORD', 'queue-secret-key'),
        database=0
    )
    
    # Create worker with production configuration
    worker = Worker(
        functions=[process_embedding_job, process_embedding_batch],
        redis_settings=redis_settings,
        on_startup=worker_instance.startup,
        on_shutdown=worker_instance.shutdown,
        max_jobs=int(os.getenv('ARQ_MAX_JOBS', 10)),  # Concurrency limit
        job_timeout=int(os.getenv('ARQ_JOB_TIMEOUT', 300)),  # 5 minute timeout
        keep_result=int(os.getenv('ARQ_KEEP_RESULT', 3600)),  # Keep results for 1 hour
        retry_jobs=True,  # Enable automatic retries
        max_tries=int(os.getenv('ARQ_MAX_TRIES', 3)),  # Maximum retry attempts
    )
    
    return worker

# Entry point for ARQ worker process
async def main():
    """ARQ worker process entry point"""
    try:
        project_name = os.getenv('PROJECT_NAME', 'default')
        logger.info(f"Starting ARQ embedding worker for project: {project_name}")
        
        worker = await create_embedding_worker(project_name)
        await worker.run()
        
    except KeyboardInterrupt:
        logger.info("ARQ worker interrupted by user")
    except Exception as e:
        logger.error(f"ARQ worker failed: {e}")
        raise

if __name__ == '__main__':
    # Configure logging for standalone worker
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())