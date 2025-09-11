"""
Async Preprocessing Pipeline with LLM Metadata Tagging
L9 2025 Architecture - ADR-0024 Implementation

This module implements a multi-stage async preprocessing pipeline that processes
files through metadata enrichment before embedding generation.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import redis.asyncio as aioredis
from dataclasses import dataclass, asdict

from ..services.service_container import ServiceContainer

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata extracted by Gemma LLM"""
    status: str  # active, archived, deprecated, experimental
    component_type: Optional[str] = None  # service, model, utility, config
    dependencies: List[str] = None
    complexity_score: Optional[float] = None
    last_modified: Optional[str] = None
    answers_questions: List[str] = None  # Questions this code can answer
    key_concepts: List[str] = None
    security_concerns: List[str] = None
    performance_notes: Optional[str] = None


class MetadataTaggerClient:
    """Client for Gemma 4B metadata tagging service"""
    
    def __init__(self, base_url: str = "http://localhost:48001"):
        self.base_url = base_url
        self.model = "gemma2:2b"  # Using 2B for faster processing
        
    async def tag_code(self, content: str, file_path: str) -> FileMetadata:
        """
        Tag code with metadata using Gemma LLM
        
        Following ADR-023 precise configuration:
        - Temperature: 0.1 for consistency
        - Top-p: 0.7 for factual accuracy
        - Top-k: 10 for limited creativity
        """
        import aiohttp
        
        # Check for archive/deprecated patterns first
        path_lower = file_path.lower()
        if '.archive/' in path_lower or '.deprecated/' in path_lower:
            # Fast path for obvious archived files
            return FileMetadata(
                status='archived' if '.archive/' in path_lower else 'deprecated',
                component_type='archived_code'
            )
        
        # Prepare prompt for Gemma
        prompt = self._create_tagging_prompt(content, file_path)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low for consistency
                            "top_p": 0.7,       # Narrow for factual accuracy
                            "top_k": 10,        # Limited creativity
                            "seed": 42,         # Reproducible outputs
                            "num_predict": 500  # Limit response length
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        metadata_json = json.loads(result.get('response', '{}'))
                        return self._parse_metadata(metadata_json)
                    else:
                        logger.error(f"Gemma tagging failed: {response.status}")
                        return FileMetadata(status='active')  # Default fallback
                        
        except Exception as e:
            logger.error(f"Error tagging {file_path}: {e}")
            return FileMetadata(status='active')  # Default fallback
    
    def _create_tagging_prompt(self, content: str, file_path: str) -> str:
        """Create precise prompt for metadata extraction"""
        # Limit content to first 2000 chars for efficiency
        content_preview = content[:2000] if len(content) > 2000 else content
        
        return f"""Analyze this code file and extract metadata. Return ONLY valid JSON.

File: {file_path}
Content:
```
{content_preview}
```

Extract and return this JSON structure:
{{
  "status": "active|archived|deprecated|experimental",
  "component_type": "service|model|utility|config|test|documentation",
  "dependencies": ["list", "of", "imports"],
  "complexity_score": 0.0 to 1.0,
  "answers_questions": ["what questions can this code answer?"],
  "key_concepts": ["main", "concepts"],
  "security_concerns": ["any", "security", "issues"],
  "performance_notes": "any performance considerations"
}}

Rules:
- status: "archived" if in .archive folder, "deprecated" if marked deprecated, else "active"
- component_type: best guess based on code structure
- dependencies: first 5 imports only
- complexity_score: 0.1=simple, 0.5=moderate, 0.9=complex
- answers_questions: max 3 questions
- Return ONLY the JSON, no explanation"""
    
    def _parse_metadata(self, metadata_json: dict) -> FileMetadata:
        """Parse JSON response into FileMetadata"""
        return FileMetadata(
            status=metadata_json.get('status', 'active'),
            component_type=metadata_json.get('component_type'),
            dependencies=metadata_json.get('dependencies', [])[:5],
            complexity_score=metadata_json.get('complexity_score'),
            answers_questions=metadata_json.get('answers_questions', [])[:3],
            key_concepts=metadata_json.get('key_concepts', [])[:5],
            security_concerns=metadata_json.get('security_concerns', []),
            performance_notes=metadata_json.get('performance_notes')
        )


class AsyncPreprocessingPipeline:
    """
    Orchestrates the complete preprocessing pipeline
    Following ADR-0024 architecture
    """
    
    def __init__(self, container: ServiceContainer):
        self.container = container
        self.metadata_tagger = MetadataTaggerClient(
            base_url="http://localhost:48001"
        )
        
        # Worker pools for each stage
        self.tagger_workers = 4  # Gemma can handle ~4 concurrent
        self.embedder_workers = 8  # Nomic is faster
        
        # Queue names
        self.RAW_QUEUE = "neural:queue:raw"
        self.TAGGED_QUEUE = "neural:queue:tagged"
        self.EMBED_QUEUE = "neural:queue:embed"
        
        # Metrics
        self.metrics = {
            'files_queued': 0,
            'files_tagged': 0,
            'files_embedded': 0,
            'files_stored': 0,
            'tagging_errors': 0,
            'embedding_errors': 0,
            'storage_errors': 0
        }
        
        # Redis connection will be initialized in start_workers
        self.redis_queue = None
        
    async def initialize_redis(self):
        """Initialize Redis connection for queues"""
        try:
            # Use Redis queue instance (port 46380)
            self.redis_queue = await aioredis.from_url(
                'redis://:queue-secret-key@localhost:46380',
                encoding='utf-8',
                decode_responses=True
            )
            logger.info("Redis queue connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def start_workers(self):
        """Start all async worker pools"""
        # Initialize Redis first
        await self.initialize_redis()
        
        tasks = []
        
        # Start metadata tagging workers
        for i in range(self.tagger_workers):
            tasks.append(asyncio.create_task(
                self.metadata_worker(f"tagger-{i}")
            ))
        
        # Start embedding workers
        for i in range(self.embedder_workers):
            tasks.append(asyncio.create_task(
                self.embedding_worker(f"embedder-{i}")
            ))
        
        # Start storage worker
        tasks.append(asyncio.create_task(self.storage_worker()))
        
        # Start metrics reporter
        tasks.append(asyncio.create_task(self.metrics_reporter()))
        
        await asyncio.gather(*tasks)
    
    async def queue_file(self, file_path: str, content: str, project_name: str, action: str = 'create'):
        """Queue a file for preprocessing"""
        file_data = {
            'path': str(file_path),
            'content': content,
            'action': action,
            'project': project_name,
            'queued_at': datetime.now(timezone.utc).isoformat()
        }
        
        await self.redis_queue.rpush(
            self.RAW_QUEUE,
            json.dumps(file_data)
        )
        
        self.metrics['files_queued'] += 1
        logger.debug(f"Queued {file_path} for preprocessing")
    
    async def metadata_worker(self, worker_id: str):
        """Process files from raw queue through metadata tagging"""
        logger.info(f"[{worker_id}] Starting metadata worker")
        
        while True:
            try:
                # Blocking pop with timeout
                item = await self.redis_queue.blpop(
                    self.RAW_QUEUE, 
                    timeout=5
                )
                
                if not item:
                    await asyncio.sleep(1)
                    continue
                
                _, data = item
                file_data = json.loads(data.decode('utf-8'))
                
                # Tag with Gemma
                metadata = await self.metadata_tagger.tag_code(
                    content=file_data['content'],
                    file_path=file_data['path']
                )
                
                # Enrich file data with metadata
                file_data['metadata'] = asdict(metadata)
                file_data['tagged_at'] = datetime.now(timezone.utc).isoformat()
                
                # Push to tagged queue
                await self.redis_queue.rpush(
                    self.TAGGED_QUEUE,
                    json.dumps(file_data)
                )
                
                self.metrics['files_tagged'] += 1
                logger.info(f"[{worker_id}] Tagged {file_data['path']} - status: {metadata.status}")
                
            except Exception as e:
                logger.error(f"[{worker_id}] Error in metadata tagging: {e}")
                self.metrics['tagging_errors'] += 1
                await asyncio.sleep(1)
    
    async def embedding_worker(self, worker_id: str):
        """Process tagged files through embedding generation"""
        logger.info(f"[{worker_id}] Starting embedding worker")
        
        while True:
            try:
                # Fetch from tagged queue
                item = await self.redis_queue.blpop(
                    self.TAGGED_QUEUE,
                    timeout=5
                )
                
                if not item:
                    await asyncio.sleep(1)
                    continue
                
                _, data = item
                file_data = json.loads(data.decode('utf-8'))
                
                # Skip archived/deprecated files from embedding
                if file_data['metadata'].get('status') in ['archived', 'deprecated']:
                    logger.info(f"[{worker_id}] Skipping {file_data['metadata']['status']} file: {file_data['path']}")
                    continue
                
                # Create composite content for embedding
                composite_text = self._create_composite_text(
                    content=file_data['content'],
                    metadata=file_data['metadata']
                )
                
                # Generate embedding using Nomic service
                embedding = await self.container.nomic_service.embed_text(
                    composite_text
                )
                
                # Prepare for storage
                file_data['embedding'] = embedding
                file_data['embedded_at'] = datetime.now(timezone.utc).isoformat()
                
                # Push to embedding queue
                await self.redis_queue.rpush(
                    self.EMBED_QUEUE,
                    json.dumps(file_data)
                )
                
                self.metrics['files_embedded'] += 1
                logger.info(f"[{worker_id}] Embedded {file_data['path']}")
                
            except Exception as e:
                logger.error(f"[{worker_id}] Error in embedding: {e}")
                self.metrics['embedding_errors'] += 1
                await asyncio.sleep(1)
    
    def _create_composite_text(self, content: str, metadata: dict) -> str:
        """
        Create composite text that includes metadata for richer embeddings
        Based on 2025 best practices for metadata-enriched embeddings
        """
        composite_parts = []
        
        # Add metadata context
        if metadata.get('status') == 'archived':
            composite_parts.append("[ARCHIVED CODE - NOT IN ACTIVE USE]")
        elif metadata.get('status') == 'deprecated':
            composite_parts.append("[DEPRECATED - MARKED FOR REMOVAL]")
        elif metadata.get('status') == 'experimental':
            composite_parts.append("[EXPERIMENTAL - SUBJECT TO CHANGE]")
        
        if metadata.get('component_type'):
            composite_parts.append(f"Type: {metadata['component_type']}")
        
        if metadata.get('dependencies'):
            deps = metadata['dependencies'][:5]  # Limit to top 5
            composite_parts.append(f"Dependencies: {', '.join(deps)}")
        
        # Add main content (truncated if too long)
        max_content_length = 6000  # Leave room for metadata
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        composite_parts.append(content)
        
        # Add extracted questions (helps with retrieval)
        if metadata.get('answers_questions'):
            questions = metadata['answers_questions'][:3]
            composite_parts.append("This code can answer: " + "; ".join(questions))
        
        # Add key concepts for better semantic matching
        if metadata.get('key_concepts'):
            concepts = metadata['key_concepts'][:5]
            composite_parts.append(f"Key concepts: {', '.join(concepts)}")
        
        return "\n\n".join(composite_parts)
    
    async def storage_worker(self):
        """Store embedded files in Qdrant"""
        logger.info("Starting storage worker")
        
        while True:
            try:
                # Fetch from embedding queue
                item = await self.redis_queue.blpop(
                    self.EMBED_QUEUE,
                    timeout=5
                )
                
                if not item:
                    await asyncio.sleep(1)
                    continue
                
                _, data = item
                file_data = json.loads(data.decode('utf-8'))
                
                # Generate unique ID for the point
                point_id = self._generate_point_id(file_data['path'])
                
                # Store in Qdrant with metadata as payload
                await self.container.qdrant_service.upsert(
                    collection_name=f"project_{file_data['project']}_embeddings",
                    points=[{
                        'id': point_id,
                        'vector': file_data['embedding'],
                        'payload': {
                            'content': file_data['content'][:10000],  # Limit content size
                            'file_path': file_data['path'],
                            'metadata': file_data['metadata'],
                            'indexed_at': datetime.now(timezone.utc).isoformat()
                        }
                    }]
                )
                
                self.metrics['files_stored'] += 1
                logger.info(f"Stored {file_data['path']} in Qdrant")
                
            except Exception as e:
                logger.error(f"Error in storage: {e}")
                self.metrics['storage_errors'] += 1
                await asyncio.sleep(1)
    
    def _generate_point_id(self, file_path: str) -> str:
        """Generate unique ID for Qdrant point"""
        # Use hash of file path for consistent IDs
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()
        # Convert to integer for Qdrant (use first 15 hex chars to fit in int64)
        return str(int(path_hash[:15], 16))
    
    async def metrics_reporter(self):
        """Report pipeline metrics periodically"""
        while True:
            await asyncio.sleep(30)  # Report every 30 seconds
            
            # Get queue depths
            try:
                raw_depth = await self.redis_queue.llen(self.RAW_QUEUE)
                tagged_depth = await self.redis_queue.llen(self.TAGGED_QUEUE)
                embed_depth = await self.redis_queue.llen(self.EMBED_QUEUE)
                
                logger.info(
                    f"Pipeline Metrics - "
                    f"Queues[raw:{raw_depth}, tagged:{tagged_depth}, embed:{embed_depth}] "
                    f"Processed[queued:{self.metrics['files_queued']}, "
                    f"tagged:{self.metrics['files_tagged']}, "
                    f"embedded:{self.metrics['files_embedded']}, "
                    f"stored:{self.metrics['files_stored']}] "
                    f"Errors[tag:{self.metrics['tagging_errors']}, "
                    f"embed:{self.metrics['embedding_errors']}, "
                    f"store:{self.metrics['storage_errors']}]"
                )
            except Exception as e:
                logger.error(f"Error reporting metrics: {e}")
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and metrics"""
        return {
            'queue_depths': {
                'raw': await self.redis_queue.llen(self.RAW_QUEUE),
                'tagged': await self.redis_queue.llen(self.TAGGED_QUEUE),
                'embed': await self.redis_queue.llen(self.EMBED_QUEUE)
            },
            'metrics': self.metrics,
            'workers': {
                'taggers': self.tagger_workers,
                'embedders': self.embedder_workers
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of pipeline"""
        logger.info("Shutting down preprocessing pipeline")
        if self.redis_queue:
            self.redis_queue.close()
            await self.redis_queue.wait_closed()