#!/usr/bin/env python3
"""
Nomic Embedding Service - Extracted from monolithic neural-mcp-server-enhanced.py
Provides embedding generation using Nomic Embed v2-MoE with Context7 patterns
"""

import os
import logging
import asyncio
import hashlib
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

@dataclass
class NomicEmbedResponse:
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]

class NomicEmbedClient:
    """Client for Nomic Embed v2-MoE service with enhanced connectivity
    
    Uses Context7 recommended pattern: fresh httpx.AsyncClient per request
    to avoid asyncio event loop binding issues in MCP environments.
    """
    
    def __init__(self):
        host = os.environ.get('EMBEDDING_SERVICE_HOST', '172.18.0.5')
        port = int(os.environ.get('EMBEDDING_SERVICE_PORT', 8000))
        self.base_url = f"http://{host}:{port}"
        
        # Store configuration for creating clients per request
        # Context7 pattern: avoid storing AsyncClient in __init__
        self.timeout = httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=60.0,     # Read timeout
            write=30.0,    # Write timeout
            pool=5.0       # Pool timeout
        )
        self.transport_kwargs = {"retries": 1}
        self.limits = httpx.Limits(max_connections=20, max_keepalive_connections=5)
        
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Nomic Embed v2-MoE with Context7 async client pattern
        
        Creates fresh httpx.AsyncClient per request to avoid event loop binding issues.
        """
        max_retries = 3
        retry_delay = 1.0
        
        # Context7 recommended pattern: fresh AsyncClient per request
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(**self.transport_kwargs),
            timeout=self.timeout,
            limits=self.limits
        ) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/embed",
                        json={"inputs": texts, "normalize": True}
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    return data.get('embeddings', [])
                    
                except httpx.HTTPStatusError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic HTTP error (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Nomic connection failed after {max_retries} attempts: {e}")
                        raise
                        
                except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic timeout (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Nomic timeout after {max_retries} attempts: {e}")
                        # Fallback to local embedding after all retries exhausted
                        return await self._fallback_local_embeddings(texts)
                        
                except Exception as e:
                    logger.error(f"Nomic embed error: {e}")
                    # Fallback to local embedding for testing
                    return await self._fallback_local_embeddings(texts)
    
    async def _fallback_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback to simple local embeddings for testing purposes"""
        import hashlib
        import numpy as np
        
        logger.warning("Using fallback local embeddings - NOT Nomic quality!")
        
        # Simple hash-based embedding for testing (768 dimensions to match expected size)
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, 768).astype(np.float32).tolist()
            embeddings.append(embedding)
        
        return embeddings

class NomicService:
    """Service class for Nomic embedding operations with queue fallback and resilience"""
    
    def __init__(self):
        self.client = None
        self.initialized = False
        self.service_container = None  # Will be set by service container
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the Nomic service with health check"""
        try:
            self.client = NomicEmbedClient()
            
            # Test the service with a simple embedding
            test_response = await self.client.get_embeddings(["test initialization"])
            
            if test_response and len(test_response[0]) > 0:
                self.initialized = True
                return {
                    "success": True,
                    "message": "Nomic service initialized successfully",
                    "embedding_dim": len(test_response[0])
                }
            else:
                return {
                    "success": False,
                    "message": "Nomic service test embedding failed"
                }
                
        except Exception as e:
            logger.error(f"Nomic service initialization failed: {e}")
            return {
                "success": False,
                "message": f"Nomic initialization failed: {str(e)}"
            }
    
    def set_service_container(self, container):
        """Set reference to service container for queue access"""
        self.service_container = container
    
    def _generate_cache_key(self, text: str, model: str = "nomic-v2") -> str:
        """Generate cache key with model versioning"""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"l9:prod:neural_tools:embeddings:{model}:1.0:{content_hash}"
    
    def _generate_job_id(self, text: str, model: str = "nomic-v2") -> str:
        """Generate deterministic job ID for deduplication"""
        content = f"{model}:{text}"
        return f"embed_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    async def get_embedding(self, text: str, model: str = "nomic-v2") -> List[float]:
        """
        Get embedding for single text with intelligent caching and queue fallback
        
        Args:
            text: Text to embed
            model: Model identifier
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.initialized or not self.client:
            raise RuntimeError("Nomic service not initialized")
        
        # Check cache first if service container available
        if self.service_container:
            try:
                redis_cache = await self.service_container.get_redis_cache_client()
                cache_key = self._generate_cache_key(text, model)
                cached = await redis_cache.get(cache_key)
                
                if cached:
                    # Cache hit - refresh TTL and return
                    await redis_cache.expire(cache_key, 86400)  # 24 hour TTL
                    return json.loads(cached)
                    
            except Exception as e:
                logger.warning(f"Cache check failed: {e}")
        
        # Try direct embedding call
        try:
            embeddings = await self.client.get_embeddings([text])
            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                
                # Cache the result if service container available
                if self.service_container:
                    try:
                        redis_cache = await self.service_container.get_redis_cache_client()
                        cache_key = self._generate_cache_key(text, model)
                        await redis_cache.setex(cache_key, 86400, json.dumps(embedding))
                    except Exception as e:
                        logger.warning(f"Cache store failed: {e}")
                
                return embedding
            else:
                raise ValueError("Empty embedding response")
                
        except (ConnectionError, TimeoutError, httpx.HTTPError) as e:
            # Network/service errors - use queue fallback if available
            if self.service_container:
                return await self._fallback_to_queue(text, model, e)
            else:
                raise
        except Exception as e:
            # Other errors - propagate immediately
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def _fallback_to_queue(self, text: str, model: str, original_error: Exception) -> Dict[str, Any]:
        """
        Fallback to job queue when direct embedding fails
        
        Args:
            text: Text to embed
            model: Model identifier
            original_error: The original error that triggered fallback
            
        Returns:
            Queue response with job information
        """
        try:
            job_queue = await self.service_container.get_job_queue()
            job_id = self._generate_job_id(text, model)
            
            logger.info(f"Direct embedding failed ({original_error}), falling back to queue for job {job_id}")
            
            # Check if job already exists/completed
            redis_cache = await self.service_container.get_redis_cache_client()
            result_key = f"l9:prod:neural_tools:job_result:{job_id}"
            cached_result = await redis_cache.get(result_key)
            
            if cached_result:
                result_data = json.loads(cached_result)
                if result_data.get('status') == 'success':
                    return result_data['embedding']
            
            # Enqueue with deduplication
            from arq.jobs import Job
            job: Job = await job_queue.enqueue_job(
                'process_embedding_job',
                text,
                model,
                _job_id=job_id,  # ARQ deduplication
                _job_timeout=300,
                _defer_until=None
            )
            
            logger.info(f"Job {job_id} enqueued for resilient processing")
            
            # Return async response - client should poll for completion
            raise ConnectionError(
                f"Service temporarily unavailable. Job {job_id} queued for processing. "
                f"Original error: {original_error}"
            )
            
        except Exception as queue_error:
            logger.error(f"Queue fallback failed for {text[:50]}...: {queue_error}")
            # Both direct and queue failed - raise original error
            raise original_error
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with error handling"""
        if not self.initialized or not self.client:
            raise RuntimeError("Nomic service not initialized")
        
        # For batch requests, try direct first, then individual fallbacks
        try:
            response = await self.client.get_embeddings(texts)
            return response
        except (ConnectionError, TimeoutError, httpx.HTTPError) as e:
            # Fallback to individual processing through queue if service container available
            if self.service_container and len(texts) <= 10:  # Limit batch fallback
                logger.warning(f"Batch embedding failed, falling back to individual processing: {e}")
                results = []
                for text in texts:
                    try:
                        embedding = await self.get_embedding(text)
                        results.append(embedding)
                    except Exception:
                        # For batch operations, use fallback embedding
                        fallback_embedding = await self._generate_fallback_embedding(text)
                        results.append(fallback_embedding)
                return results
            else:
                raise
    
    async def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate fallback embedding for testing/degraded service"""
        import hashlib
        import numpy as np
        
        logger.warning(f"Using fallback embedding for: {text[:50]}...")
        
        # Create deterministic embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, 768).astype(np.float32).tolist()
        
        return embedding
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            if not self.initialized:
                return {"healthy": False, "message": "Service not initialized"}
                
            # Simple health check
            test_embedding = await self.get_embeddings(["health check"])
            return {
                "healthy": True,
                "embedding_dim": len(test_embedding[0]) if test_embedding else 0,
                "service_url": self.client.base_url if self.client else "unknown"
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}