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
    """Metadata extracted by pattern-based analysis"""
    status: str  # active, archived, deprecated, experimental
    component_type: Optional[str] = None  # service, model, utility, config
    dependencies: List[str] = None
    complexity_score: Optional[float] = None
    last_modified: Optional[str] = None
    answers_questions: List[str] = None  # Questions this code can answer
    key_concepts: List[str] = None
    security_concerns: List[str] = None
    performance_notes: Optional[str] = None
    
    # Optimal metadata additions (Grok 4 analysis)
    public_api: List[str] = None  # Non-underscore prefixed exports
    has_type_hints: bool = False  # Type annotations present
    todo_count: int = 0  # TODO/FIXME/HACK/XXX count
    has_io_operations: bool = False  # File/network/DB operations
    is_async_heavy: bool = False  # >50% async functions
    line_count: int = 0  # Total lines in file


class MetadataTaggerClient:
    """Client for LLM metadata tagging service with fallback"""
    
    def __init__(self, base_url: str = "http://localhost:48001"):
        self.base_url = base_url
        self.model = "qwen3:0.6b"  # Fast lightweight model
        self.use_llm = False  # Temporarily disabled due to performance issues
        
    async def tag_code(self, content: str, file_path: str) -> FileMetadata:
        """
        Tag code with metadata using pattern-based analysis
        Fallback to simple heuristics due to LLM performance issues
        """
        # Check for archive/deprecated patterns first
        path_lower = file_path.lower()
        if '.archive/' in path_lower or '.deprecated/' in path_lower:
            return FileMetadata(
                status='archived' if '.archive/' in path_lower else 'deprecated',
                component_type='archived_code'
            )
        
        # Use LLM if enabled (currently disabled for performance)
        if self.use_llm:
            return await self._tag_with_llm(content, file_path)
        
        # Fast pattern-based tagging
        return self._tag_with_patterns(content, file_path)
    
    def _tag_with_patterns(self, content: str, file_path: str) -> FileMetadata:
        """Fast pattern-based metadata extraction with optimal fields"""
        import re
        import os
        from pathlib import Path
        
        # Get file stats
        file_stats = None
        try:
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                from datetime import datetime, timezone
                file_stats = {
                    'created': datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    'size_bytes': stat.st_size
                }
        except:
            pass
        
        # Determine status
        status = 'active'
        if 'deprecated' in content.lower()[:500]:
            status = 'deprecated'
        elif 'experimental' in content.lower()[:500]:
            status = 'experimental'
        elif '.test.' in file_path or '_test.' in file_path:
            status = 'test'
        
        # Determine component type and check for entrypoint
        component_type = 'utility'
        is_entrypoint = False
        
        # Check for main/entrypoint patterns
        if '__main__' in content:
            is_entrypoint = True
        if 'def main(' in content or 'async def main(' in content:
            is_entrypoint = True
        if file_path.endswith(('main.py', 'app.py', '__main__.py', 'cli.py', 'server.py')):
            is_entrypoint = True
            
        if 'class ' in content[:1000]:
            if 'Service' in content[:1000]:
                component_type = 'service'
            elif 'Model' in content[:1000]:
                component_type = 'model'
            elif 'Server' in content[:1000]:
                component_type = 'server'
            else:
                component_type = 'class'
        elif is_entrypoint:
            component_type = 'entrypoint'
        elif '.config.' in file_path or 'config/' in file_path:
            component_type = 'config'
        elif '.test.' in file_path or 'test/' in file_path:
            component_type = 'test'
        elif '.md' in file_path:
            component_type = 'documentation'
        
        # Extract imports/dependencies (enhanced)
        import_pattern = r'^(?:from|import)\s+([\w\.]+)'
        imports = re.findall(import_pattern, content, re.MULTILINE)
        # Group by package (first part before .)
        unique_imports = []
        seen_packages = set()
        for imp in imports:
            package = imp.split('.')[0]
            if package not in seen_packages and package not in ['__future__', 'typing']:
                unique_imports.append(imp)
                seen_packages.add(package)
        dependencies = unique_imports[:10]  # Top 10 unique imports
        
        # Extract public API (non-underscore prefixed exports)
        exported_functions = re.findall(r'^(?:async\s+)?def\s+(\w+)\(', content, re.MULTILINE)
        exported_classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        all_exports = exported_functions + exported_classes
        public_api = [name for name in all_exports if not name.startswith('_')][:10]
        
        # Calculate line count and function metrics
        line_count = content.count('\n') + 1
        classes = len(re.findall(r'^class\s+', content, re.MULTILINE))
        functions = len(re.findall(r'^def\s+', content, re.MULTILINE))
        async_functions = len(re.findall(r'^async\s+def\s+', content, re.MULTILINE))
        total_functions = functions + async_functions
        try_blocks = len(re.findall(r'^\s*try:', content, re.MULTILINE))
        decorators = len(re.findall(r'^@\w+', content, re.MULTILINE))
        
        # Check for type hints
        has_type_hints = bool(re.search(r'def\s+\w+\([^)]*:[^)]*\)', content) or 
                              re.search(r'->\s*[\w\[\]]+', content) or
                              'from typing import' in content)
        
        # Count TODOs/FIXMEs/HACKs/XXXs
        todo_pattern = r'#.*(TODO|FIXME|HACK|XXX|WIP|REFACTOR)'
        todo_count = len(re.findall(todo_pattern, content, re.IGNORECASE))
        
        # Detect I/O operations
        io_patterns = [
            r'\bopen\s*\(',  # File operations
            r'\.read\s*\(', r'\.write\s*\(',  # File I/O
            r'\brequests\.',  # HTTP requests
            r'\baiohttp\.',  # Async HTTP
            r'\bcursor\.',  # Database cursors
            r'\.execute\s*\(',  # SQL execution
            r'\bredis\.',  # Redis operations
            r'\bsocket\.',  # Socket operations
            r'\bfile\s*=',  # File parameters
            r'\bwith\s+open\s*\(',  # Context manager file ops
        ]
        has_io_operations = any(re.search(pattern, content) for pattern in io_patterns)
        
        # Check if async-heavy (>50% async functions)
        is_async_heavy = False
        if total_functions > 0:
            async_ratio = async_functions / total_functions
            is_async_heavy = async_ratio > 0.5
        
        complexity_score = min(1.0, (
            (line_count / 500) * 0.15 +        # Lines of code
            (classes / 5) * 0.25 +             # Number of classes
            (total_functions / 20) * 0.25 +    # Number of functions
            (async_functions / 10) * 0.15 +    # Async complexity
            (try_blocks / 10) * 0.1 +          # Error handling
            (decorators / 15) * 0.1             # Decorator usage
        ))
        
        # Extract key concepts (enhanced)
        key_concepts = []
        concept_patterns = {
            'async': r'\basync\b',
            'graphrag': r'graphrag',
            'embeddings': r'embedd?ing',
            'neural': r'neural',
            'mcp': r'\bmcp\b',
            'api': r'\bapi\b',
            'database': r'(database|db|sql|neo4j|qdrant)',
            'cache': r'(cache|redis)',
            'queue': r'(queue|worker|task)',
            'auth': r'(auth|jwt|oauth|token)',
            'websocket': r'websocket',
            'rest': r'(rest|http|endpoint)',
        }
        
        for concept, pattern in concept_patterns.items():
            if re.search(pattern, content.lower()):
                key_concepts.append(concept)
        
        # Identify what questions this code answers (enhanced)
        answers_questions = []
        question_patterns = {
            'How to generate embeddings?': r'embedd?ing',
            'How to search data?': r'search',
            'How to index content?': r'index',
            'How to handle authentication?': r'(auth|jwt|oauth)',
            'How to cache data?': r'cache',
            'How to process queues?': r'(queue|worker)',
            'How to connect to database?': r'(connect|database|neo4j|qdrant)',
            'How to handle errors?': r'(try|except|error|exception)',
            'How to run the application?': r'__main__|def main',
        }
        
        for question, pattern in question_patterns.items():
            if re.search(pattern, content.lower()):
                answers_questions.append(question)
        
        # Build enhanced metadata with optimal fields
        metadata = FileMetadata(
            status=status,
            component_type=component_type,
            dependencies=dependencies,
            complexity_score=round(complexity_score, 2),
            answers_questions=answers_questions[:5],
            key_concepts=key_concepts[:8],
            # New optimal fields
            public_api=public_api,
            has_type_hints=has_type_hints,
            todo_count=todo_count,
            has_io_operations=has_io_operations,
            is_async_heavy=is_async_heavy,
            line_count=line_count
        )
        
        # Add extra fields if we have them
        if file_stats:
            metadata.last_modified = file_stats['modified']
        if is_entrypoint:
            if not metadata.key_concepts:
                metadata.key_concepts = []
            metadata.key_concepts.append('entrypoint')
        if public_api:
            metadata.performance_notes = f"Public API: {', '.join(public_api[:5])}"
            
        return metadata
    
    async def _tag_with_llm(self, content: str, file_path: str) -> FileMetadata:
        """LLM-based tagging (currently disabled for performance)"""
        import aiohttp
        
        # Simplified prompt without JSON formatting requirement
        prompt = f"""Analyze this code and describe it briefly:
{content[:1000]}

What type of component is this? (service, model, utility, config, test)
Is it active, deprecated, or experimental?
What are the main imports?
How complex is it (0.0 to 1.0)?"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.7,
                            "top_k": 10,
                            "num_predict": 200
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=10)  # Short timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Parse natural language response
                        response_text = result.get('response', '')
                        return self._parse_natural_response(response_text)
                    else:
                        logger.warning(f"LLM tagging failed: {response.status}")
                        return self._tag_with_patterns(content, file_path)
                        
        except Exception as e:
            logger.warning(f"LLM timeout, using patterns: {e}")
            return self._tag_with_patterns(content, file_path)
    
    def _parse_natural_response(self, response: str) -> FileMetadata:
        """Parse natural language LLM response"""
        response_lower = response.lower()
        
        # Extract component type
        component_type = 'utility'
        for comp_type in ['service', 'model', 'config', 'test']:
            if comp_type in response_lower:
                component_type = comp_type
                break
        
        # Extract status
        status = 'active'
        if 'deprecated' in response_lower:
            status = 'deprecated'
        elif 'experimental' in response_lower:
            status = 'experimental'
        
        # Extract complexity
        import re
        complexity_match = re.search(r'0\.\d+|1\.0', response)
        complexity_score = float(complexity_match.group()) if complexity_match else 0.5
        
        return FileMetadata(
            status=status,
            component_type=component_type,
            complexity_score=complexity_score
        )
    
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