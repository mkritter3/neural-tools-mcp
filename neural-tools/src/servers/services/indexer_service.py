#!/usr/bin/env python3
"""
L9 Automatic Incremental Indexer Service
Monitors file changes and maintains vector/graph indices in real-time
Uses sidecar pattern with supervisord for non-blocking background processing
"""

import os
import sys
import time
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque
import json

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Timer, Lock
from qdrant_client import models

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from servers.services.service_container import ServiceContainer

# Configure logging to stderr for Docker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class DebouncedEventHandler(FileSystemEventHandler):
    """
    Enhanced event handler with debouncing to prevent event storms
    Crucial for handling bulk operations like git pull or IDE saves
    """
    
    def __init__(self, indexer, debounce_interval=2.0):
        self.indexer = indexer
        self.debounce_interval = debounce_interval
        self._lock = Lock()
        self._events_queue = {}  # path -> event_type
        self._timer = None
    
    def dispatch(self, event):
        """Override dispatch to implement debouncing"""
        # Ignore directory events
        if event.is_directory:
            return
        
        # Filter based on indexer's should_index
        if not self.indexer.should_index(event.src_path):
            return
        
        with self._lock:
            # Handle different event types
            if event.event_type == 'moved':
                # For moves, track both source (delete) and dest (create)
                self._events_queue[event.src_path] = 'delete'
                if hasattr(event, 'dest_path'):
                    self._events_queue[event.dest_path] = 'create'
            else:
                # For other events, track the latest event type
                # Priority: delete > create > modify
                current = self._events_queue.get(event.src_path)
                if current != 'delete':  # Don't override delete
                    self._events_queue[event.src_path] = event.event_type
            
            # Reset timer
            if self._timer:
                self._timer.cancel()
            
            self._timer = Timer(self.debounce_interval, self._process_events)
            self._timer.start()
    
    def _process_events(self):
        """Process accumulated events after debounce period"""
        with self._lock:
            if not self._events_queue:
                return
            
            events_to_process = self._events_queue.copy()
            self._events_queue.clear()
        
        logger.info(f"Processing batch of {len(events_to_process)} file events after debounce")
        
        # Track event storms (>5 events in one batch indicates a storm)
        if len(events_to_process) > 5:
            self.indexer.metrics['event_storms_handled'] += 1
        
        # Process events asynchronously
        for path, event_type in events_to_process.items():
            try:
                # Map watchdog events to our indexer actions
                if event_type == 'deleted':
                    asyncio.create_task(self.indexer._queue_change(path, 'delete'))
                elif event_type == 'created':
                    asyncio.create_task(self.indexer._queue_change(path, 'create'))
                elif event_type == 'modified':
                    asyncio.create_task(self.indexer._queue_change(path, 'update'))
                
            except Exception as e:
                logger.error(f"Error queuing {event_type} for {path}: {e}")

class IncrementalIndexer(FileSystemEventHandler):
    """
    Handles file system events and performs incremental indexing
    Integrates with Neo4j GraphRAG, Qdrant vectors, and Nomic embeddings
    """
    
    def __init__(self, project_path: str, project_name: str = "default"):
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.collection_prefix = f"project_{project_name}_"
        
        # Service container for accessing Neo4j, Qdrant, Nomic
        self.container = None
        self.services_initialized = False
        
        # File tracking with deduplication
        self.file_hashes: Dict[str, str] = {}
        self.file_cooldowns: Dict[str, datetime] = {}
        self.cooldown_seconds = 3  # Prevent rapid re-indexing
        
        # Bounded queue for memory management
        self.pending_queue: asyncio.Queue = None  # Will init with asyncio
        self.max_queue_size = 1000
        self.queue_warning_threshold = 800
        
        # Batch processing settings
        self.batch_size = 10
        self.batch_timeout = 5.0  # seconds
        self.last_batch_time = time.time()
        
        # File patterns configuration
        self.watch_patterns = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go',
            '.rs', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php',
            '.md', '.yaml', '.yml', '.json', '.toml', '.txt', '.sql'
        }
        
        self.ignore_patterns = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            'dist', 'build', '.next', 'target', '.pytest_cache',
            '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.log',
            '.env', '.env.local', '.env.production', 'secrets',
            '*.key', '*.pem', '*.crt', '.neural-tools'  # Add our own data dir
        }
        
        # File size limits
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
        # Performance metrics
        self.metrics = {
            'files_indexed': 0,
            'files_skipped': 0,
            'errors': 0,
            'last_index_time': None,
            'queue_depth': 0,
            'service_failures': {'neo4j': 0, 'qdrant': 0, 'nomic': 0},
            # GraphRAG metrics
            'chunks_created': 0,
            'cross_references_created': 0,
            'avg_index_time_ms': 0,
            'total_index_time_ms': 0,
            'dedup_hits': 0,  # Files skipped due to unchanged hash
            'event_storms_handled': 0  # Debouncing events
        }
        
        # Graceful degradation flags
        self.degraded_mode = {
            'neo4j': False,
            'qdrant': False,
            'nomic': False
        }
        
    async def initialize_services(self):
        """Initialize service connections with retry logic"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing services (attempt {attempt + 1}/{max_retries})")
                
                self.container = ServiceContainer(self.project_name)
                result = await self.container.initialize_all_services()
                
                # Check individual service health
                if self.container.neo4j:
                    self.degraded_mode['neo4j'] = False
                else:
                    logger.warning("Neo4j unavailable - continuing without graph indexing")
                    self.degraded_mode['neo4j'] = True
                    
                if self.container.qdrant:
                    self.degraded_mode['qdrant'] = False
                    await self._ensure_collections()
                else:
                    logger.warning("Qdrant unavailable - continuing without vector search")
                    self.degraded_mode['qdrant'] = True
                    
                if self.container.nomic:
                    self.degraded_mode['nomic'] = False
                else:
                    logger.warning("Nomic unavailable - continuing without semantic embeddings")
                    self.degraded_mode['nomic'] = True
                
                self.services_initialized = True
                logger.info(f"Services initialized: {result.get('overall_health', 'degraded')}")
                return True
                
            except Exception as e:
                logger.error(f"Service initialization failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("All initialization attempts failed - running in degraded mode")
                    self.services_initialized = True  # Continue anyway
                    return False
    
    async def _ensure_collections(self):
        """Ensure required Qdrant collections exist"""
        if self.degraded_mode['qdrant']:
            return
            
        code_collection = f"{self.collection_prefix}code"
        
        try:
            await self.container.qdrant.get_collection_info(code_collection)
        except:
            # Create collection with hybrid search config
            await self.container.qdrant.create_collection(
                collection_name=code_collection,
                vector_size=768,  # Nomic v2 dimension
                enable_sparse=True
            )
            logger.info(f"Created collection: {code_collection}")
    
    def should_index(self, file_path: str) -> bool:
        """Check if file should be indexed with security filters"""
        path = Path(file_path)
        
        # Check file exists and is not too large
        if not path.exists() or not path.is_file():
            return False
            
        if path.stat().st_size > self.max_file_size:
            logger.debug(f"Skipping large file: {file_path}")
            self.metrics['files_skipped'] += 1
            return False
        
        # Check if it's a file type we care about
        if not any(str(path).endswith(ext) for ext in self.watch_patterns):
            return False
        
        # Security: Check ignore patterns (includes .env, secrets, etc)
        for pattern in self.ignore_patterns:
            if pattern in str(path):
                return False
        
        # Check cooldown to prevent rapid re-indexing
        if file_path in self.file_cooldowns:
            last_update = self.file_cooldowns[file_path]
            if datetime.now() - last_update < timedelta(seconds=self.cooldown_seconds):
                logger.debug(f"File in cooldown: {file_path}")
                return False
        
        return True
    
    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get hash of file contents for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.debug(f"Could not hash {file_path}: {e}")
            return None
    
    async def index_file(self, file_path: str, action: str = "update"):
        """
        Index a single file with graceful degradation and performance tracking
        Integrates with Neo4j, Qdrant, and Nomic based on availability
        """
        if not self.should_index(file_path):
            return
        
        start_time = time.time()
        
        try:
            # Check if file actually changed
            current_hash = self.get_file_hash(file_path)
            if current_hash == self.file_hashes.get(file_path):
                logger.debug(f"File unchanged, skipping: {file_path}")
                self.metrics['dedup_hits'] += 1
                return
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                return
            
            # Update cooldown
            self.file_cooldowns[file_path] = datetime.now()
            
            path = Path(file_path)
            relative_path = path.relative_to(self.project_path)
            
            # Chunk large files
            chunks = self._chunk_content(content, str(relative_path))
            
            # Process with available services
            success = False
            
            # 1. Semantic indexing with Qdrant + Nomic
            if not self.degraded_mode['qdrant']:
                if not self.degraded_mode['nomic']:
                    # Full semantic indexing
                    await self._index_semantic(file_path, relative_path, chunks)
                    success = True
                else:
                    # Keyword-only indexing (no embeddings)
                    await self._index_keywords(file_path, relative_path, chunks)
                    success = True
            
            # 2. Graph indexing with Neo4j
            if not self.degraded_mode['neo4j']:
                await self._index_graph(file_path, relative_path, content)
                success = True
            
            if success:
                # Update hash tracking
                self.file_hashes[file_path] = current_hash
                self.metrics['files_indexed'] += 1
                self.metrics['chunks_created'] += len(chunks)
                self.metrics['last_index_time'] = datetime.now().isoformat()
                
                # Track timing metrics
                elapsed_ms = (time.time() - start_time) * 1000
                self.metrics['total_index_time_ms'] += elapsed_ms
                self.metrics['avg_index_time_ms'] = (
                    self.metrics['total_index_time_ms'] / self.metrics['files_indexed']
                )
                
                logger.info(f"Indexed {file_path}: {len(chunks)} chunks in {elapsed_ms:.1f}ms")
            else:
                logger.warning(f"No services available to index {file_path}")
                self.metrics['files_skipped'] += 1
                
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            self.metrics['errors'] += 1
    
    def _chunk_content(self, content: str, file_path: str) -> List[Dict]:
        """Intelligently chunk content for indexing"""
        max_chunk_lines = 100  # Smaller chunks for better precision
        lines = content.split('\n')
        chunks = []
        
        # For code files, try to chunk at logical boundaries
        if file_path.endswith(('.py', '.js', '.ts', '.java', '.go')):
            # Simple heuristic: break at function/class definitions
            current_chunk = []
            current_start = 1
            
            for i, line in enumerate(lines, 1):
                # Detect function/class boundaries (simplified)
                if any(line.strip().startswith(kw) for kw in ['def ', 'class ', 'function ', 'const ', 'export ']):
                    if current_chunk and len(current_chunk) > 10:
                        chunks.append({
                            'text': '\n'.join(current_chunk),
                            'start_line': current_start,
                            'end_line': i - 1
                        })
                        current_chunk = [line]
                        current_start = i
                    else:
                        current_chunk.append(line)
                else:
                    current_chunk.append(line)
                    
                # Force break at max size
                if len(current_chunk) >= max_chunk_lines:
                    chunks.append({
                        'text': '\n'.join(current_chunk),
                        'start_line': current_start,
                        'end_line': i
                    })
                    current_chunk = []
                    current_start = i + 1
            
            # Add remaining
            if current_chunk:
                chunks.append({
                    'text': '\n'.join(current_chunk),
                    'start_line': current_start,
                    'end_line': len(lines)
                })
        else:
            # Simple line-based chunking for other files
            for i in range(0, len(lines), max_chunk_lines):
                chunk_lines = lines[i:i + max_chunk_lines]
                if '\n'.join(chunk_lines).strip():
                    chunks.append({
                        'text': '\n'.join(chunk_lines),
                        'start_line': i + 1,
                        'end_line': min(i + max_chunk_lines, len(lines))
                    })
        
        return chunks if chunks else [{'text': content, 'start_line': 1, 'end_line': len(lines)}]
    
    async def _index_semantic(self, file_path: str, relative_path: Path, chunks: List[Dict]):
        """Index with full semantic embeddings and GraphRAG cross-references"""
        try:
            # Prepare texts for embedding
            texts = [f"File: {relative_path}\n{chunk['text'][:500]}" for chunk in chunks]
            
            # Get embeddings in batch
            embeddings = await self.container.nomic.get_embeddings(texts)
            
            if not embeddings:
                logger.warning(f"No embeddings generated for {file_path}")
                return
            
            # Prepare points for Qdrant with Neo4j cross-references
            points = []
            neo4j_chunk_ids = []  # Track for Neo4j batch update
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate deterministic shared ID using SHA256
                chunk_id_str = f"{file_path}:{chunk['start_line']}:{chunk['end_line']}"
                chunk_id_hash = hashlib.sha256(chunk_id_str.encode()).hexdigest()
                # Convert to numeric ID for Qdrant (use first 16 hex chars as int)
                chunk_id = int(chunk_id_hash[:16], 16)
                
                # Create sparse vector from keywords
                sparse_indices, sparse_values = self._extract_keywords(chunk['text'])
                
                point = {
                    'id': chunk_id,
                    'vector': embedding,
                    'sparse_vector': {
                        'indices': sparse_indices,
                        'values': sparse_values
                    },
                    'payload': {
                        'file_path': str(relative_path),
                        'full_path': file_path,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'start_line': chunk['start_line'],
                        'end_line': chunk['end_line'],
                        'content': chunk['text'][:1000],
                        'file_type': relative_path.suffix,
                        'indexed_at': datetime.now().isoformat(),
                        'project': self.project_name,
                        # GraphRAG: Store Neo4j node ID reference
                        'neo4j_chunk_id': chunk_id_hash,
                        'chunk_hash': chunk_id_hash
                    }
                }
                points.append(point)
                neo4j_chunk_ids.append({
                    'id': chunk_id_hash,
                    'qdrant_id': chunk_id,
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'chunk_type': self._detect_chunk_type(chunk['text']),
                    'content': chunk['text']  # Add content to be stored in Neo4j
                })
            
            # Upsert to Qdrant
            collection_name = f"{self.collection_prefix}code"
            await self.container.qdrant.upsert_points(collection_name, points)
            
            # Update Neo4j with chunk references (if available)
            if not self.degraded_mode['neo4j'] and neo4j_chunk_ids:
                await self._update_neo4j_chunks(file_path, relative_path, neo4j_chunk_ids)
                self.metrics['cross_references_created'] += len(neo4j_chunk_ids)
            
        except Exception as e:
            logger.error(f"Semantic indexing failed for {file_path}: {e}")
            self.metrics['service_failures']['nomic'] += 1
            if self.metrics['service_failures']['nomic'] > 10:
                logger.warning("Nomic failures exceeded threshold - entering degraded mode")
                self.degraded_mode['nomic'] = True
    
    async def _index_keywords(self, file_path: str, relative_path: Path, chunks: List[Dict]):
        """Fallback: Index with keywords only (no embeddings)"""
        try:
            # Create keyword-only entries
            points = []
            for i, chunk in enumerate(chunks):
                point_id = abs(hash(f"{file_path}_{i}_{datetime.now()}")) % (10**15)
                
                # Extract keywords for sparse vector
                sparse_indices, sparse_values = self._extract_keywords(chunk['text'])
                
                # Use zero vector for dense (required by schema)
                zero_vector = [0.0] * 768
                
                point = {
                    'id': point_id,
                    'vector': zero_vector,
                    'sparse_vector': {
                        'indices': sparse_indices,
                        'values': sparse_values
                    },
                    'payload': {
                        'file_path': str(relative_path),
                        'full_path': file_path,
                        'chunk_index': i,
                        'content': chunk['text'][:1000],
                        'indexed_at': datetime.now().isoformat(),
                        'index_type': 'keyword_only'
                    }
                }
                points.append(point)
            
            collection_name = f"{self.collection_prefix}code"
            await self.container.qdrant.upsert_points(collection_name, points)
            
        except Exception as e:
            logger.error(f"Keyword indexing failed for {file_path}: {e}")
            self.metrics['service_failures']['qdrant'] += 1
    
    def _extract_keywords(self, text: str, max_keywords: int = 100) -> Tuple[List[int], List[float]]:
        """Extract keywords for sparse vector representation"""
        # Simple TF-based keyword extraction
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            # Filter short words and common stop words
            if len(word) > 2 and word not in {'the', 'and', 'for', 'with', 'from', 'this', 'that'}:
                word = ''.join(c for c in word if c.isalnum())  # Clean punctuation
                if word:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Convert to sparse vector format
        vocab_size = 10000
        sparse_indices = []
        sparse_values = []
        
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]:
            idx = hash(word) % vocab_size
            sparse_indices.append(idx)
            sparse_values.append(float(freq))
        
        return sparse_indices, sparse_values
    
    async def _index_graph(self, file_path: str, relative_path: Path, content: str):
        """Index code relationships in Neo4j graph with GraphRAG support"""
        try:
            # Extract basic metadata
            file_type = relative_path.suffix
            
            # Generate file content hash for deduplication
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Create or update file node with content hash
            cypher = """
            MERGE (f:File {path: $path})
            SET f.name = $name,
                f.type = $type,
                f.size = $size,
                f.content_hash = $content_hash,
                f.indexed_at = datetime(),
                f.project = $project
            RETURN f.content_hash AS existing_hash
            """
            
            result = await self.container.neo4j.execute_cypher(cypher, {
                'path': str(relative_path),
                'name': relative_path.name,
                'type': file_type,
                'size': len(content),
                'content_hash': file_hash,
                'project': self.project_name
            })
            
            # Check if query was successful and content unchanged (for deduplication)
            if result.get('status') == 'success' and result.get('result') and len(result['result']) > 0 and result['result'][0].get('existing_hash') == file_hash:
                logger.debug(f"File content unchanged in Neo4j: {file_path}")
                # Still need to ensure chunks exist if Qdrant was updated
            
            # Parse imports/dependencies (simplified - could use tree-sitter for accuracy)
            if file_type in ['.py', '.js', '.ts']:
                imports = self._extract_imports(content, file_type)
                
                for imp in imports:
                    # Create import relationship
                    cypher = """
                    MATCH (f:File {path: $from_path})
                    MERGE (m:Module {name: $module})
                    MERGE (f)-[:IMPORTS]->(m)
                    """
                    result = await self.container.neo4j.execute_cypher(cypher, {
                        'from_path': str(relative_path),
                        'module': imp
                    })
                    if result.get('status') != 'success':
                        logger.warning(f"Failed to create import relationship: {result.get('message')}")
            
        except Exception as e:
            logger.error(f"Graph indexing failed for {file_path}: {e}")
            self.metrics['service_failures']['neo4j'] += 1
            if self.metrics['service_failures']['neo4j'] > 10:
                logger.warning("Neo4j failures exceeded threshold - entering degraded mode")
                self.degraded_mode['neo4j'] = True
    
    def _detect_chunk_type(self, text: str) -> str:
        """Detect the type of code chunk (function, class, import, etc.)"""
        lines = text.strip().split('\n')
        if not lines:
            return 'unknown'
        
        first_line = lines[0].strip()
        
        # Python patterns
        if first_line.startswith('def '):
            return 'function'
        elif first_line.startswith('class '):
            return 'class'
        elif first_line.startswith('import ') or first_line.startswith('from '):
            return 'import'
        # JavaScript/TypeScript patterns
        elif 'function ' in first_line or 'const ' in first_line or 'let ' in first_line:
            return 'function'
        elif 'class ' in first_line:
            return 'class'
        elif 'export ' in first_line:
            return 'export'
        # General patterns
        elif first_line.startswith('#') or first_line.startswith('//'):
            return 'comment'
        else:
            return 'code'
    
    async def _update_neo4j_chunks(self, file_path: str, relative_path: Path, chunk_ids: List[Dict]):
        """Create CodeChunk nodes in Neo4j with Qdrant cross-references"""
        try:
            # Batch create/update chunk nodes
            for chunk_info in chunk_ids:
                cypher = """
                MERGE (c:CodeChunk {id: $chunk_id})
                SET c.qdrant_id = $qdrant_id,
                    c.file_path = $file_path,
                    c.start_line = $start_line,
                    c.end_line = $end_line,
                    c.type = $chunk_type,
                    c.content = $content,
                    c.indexed_at = datetime(),
                    c.project = $project
                WITH c
                MATCH (f:File {path: $file_path})
                MERGE (c)-[:PART_OF]->(f)
                """
                
                # Debug: Log content being stored
                content_to_store = chunk_info.get('content', '')
                logger.debug(f"Storing chunk with content length: {len(content_to_store)}, first 50 chars: {content_to_store[:50]}")
                
                result = await self.container.neo4j.execute_cypher(cypher, {
                    'chunk_id': chunk_info['id'],
                    'qdrant_id': chunk_info['qdrant_id'],
                    'file_path': str(relative_path),
                    'start_line': chunk_info['start_line'],
                    'end_line': chunk_info['end_line'],
                    'chunk_type': chunk_info['chunk_type'],
                    'content': content_to_store,  # Include content in parameters
                    'project': self.project_name
                })
                
                # Check if query executed successfully
                logger.debug(f"Neo4j execute result: {result}")
                if result.get('status') != 'success':
                    logger.error(f"Failed to update chunk in Neo4j: {result.get('message', 'Unknown error')}")
                    raise ValueError(f"Neo4j chunk update failed: {result.get('message')}")
            
            logger.debug(f"Created/updated {len(chunk_ids)} chunk nodes in Neo4j")
            
        except Exception as e:
            logger.error(f"Failed to update Neo4j chunks: {e}")
            # Don't mark as degraded for chunk updates - file node is more important
    
    def _extract_imports(self, content: str, file_type: str) -> List[str]:
        """Extract import statements (simplified)"""
        imports = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if file_type == '.py':
                if line.startswith('import ') or line.startswith('from '):
                    # Extract module name
                    parts = line.split()
                    if parts[0] == 'import' and len(parts) > 1:
                        imports.append(parts[1].split('.')[0])
                    elif parts[0] == 'from' and len(parts) > 1:
                        imports.append(parts[1].split('.')[0])
                        
            elif file_type in ['.js', '.ts']:
                if 'import ' in line or 'require(' in line:
                    # Simplified extraction
                    if 'from ' in line:
                        module = line.split('from ')[-1].strip(' ;\'"')
                        imports.append(module)
                    elif 'require(' in line:
                        start = line.find('require(') + 8
                        end = line.find(')', start)
                        if end > start:
                            module = line[start:end].strip(' \'"')
                            imports.append(module)
        
        return imports[:20]  # Limit to prevent explosion
    
    async def remove_file_from_index(self, file_path: str):
        """Remove file from all indices when deleted (GraphRAG-aware)"""
        try:
            path = Path(file_path)
            relative_path = path.relative_to(self.project_path)
            
            # First, get chunk IDs from Neo4j to ensure complete removal
            chunk_ids_to_remove = []
            if not self.degraded_mode['neo4j']:
                # Get all chunks for this file
                cypher = """
                MATCH (c:CodeChunk)-[:PART_OF]->(f:File {path: $path})
                RETURN c.qdrant_id AS qdrant_id, c.id AS chunk_id
                """
                result = await self.container.neo4j.execute_cypher(cypher, {
                    'path': str(relative_path)
                })
                
                if result:
                    chunk_ids_to_remove = [(r['qdrant_id'], r['chunk_id']) for r in result if r.get('qdrant_id')]
            
            # Remove from Qdrant
            if not self.degraded_mode['qdrant']:
                collection_name = f"{self.collection_prefix}code"
                
                if chunk_ids_to_remove:
                    # Remove specific chunks by ID (more precise)
                    qdrant_ids = [qid for qid, _ in chunk_ids_to_remove if qid]
                    if qdrant_ids:
                        await self.container.qdrant.delete_points(
                            collection_name,
                            points_selector=models.PointIdsList(points=qdrant_ids)
                        )
                        logger.info(f"Removed {len(qdrant_ids)} chunks from Qdrant for {file_path}")
                else:
                    # Fallback: Delete by file path filter
                    await self.container.qdrant.delete_points(
                        collection_name,
                        filter={
                            'must': [
                                {'key': 'full_path', 'match': {'value': file_path}}
                            ]
                        }
                    )
                    logger.info(f"Removed {file_path} from Qdrant")
            
            # Remove from Neo4j (including chunks and file)
            if not self.degraded_mode['neo4j']:
                # Delete chunks and file in one transaction
                cypher = """
                MATCH (f:File {path: $path})
                OPTIONAL MATCH (c:CodeChunk)-[:PART_OF]->(f)
                DETACH DELETE c, f
                """
                await self.container.neo4j.execute_cypher(cypher, {
                    'path': str(relative_path)
                })
                logger.info(f"Removed {file_path} and its chunks from Neo4j")
            
            # Remove from tracking
            self.file_hashes.pop(file_path, None)
            self.file_cooldowns.pop(file_path, None)
            
        except Exception as e:
            logger.error(f"Failed to remove {file_path} from index: {e}")
            self.metrics['errors'] += 1
    
    async def process_queue(self):
        """Process pending file changes from queue"""
        batch = []
        
        try:
            # Collect batch of changes
            while len(batch) < self.batch_size:
                try:
                    file_event = await asyncio.wait_for(
                        self.pending_queue.get(),
                        timeout=0.1
                    )
                    batch.append(file_event)
                except asyncio.TimeoutError:
                    break
            
            if not batch:
                return
            
            logger.info(f"Processing batch of {len(batch)} file changes")
            
            for file_path, event_type in batch:
                if event_type == 'delete':
                    await self.remove_file_from_index(file_path)
                else:
                    await self.index_file(file_path, event_type)
            
            # Update metrics
            self.metrics['queue_depth'] = self.pending_queue.qsize()
            
            # Warn if queue is getting full
            if self.metrics['queue_depth'] > self.queue_warning_threshold:
                logger.warning(f"Queue depth high: {self.metrics['queue_depth']}/{self.max_queue_size}")
                
        except Exception as e:
            logger.error(f"Error processing queue: {e}")
            self.metrics['errors'] += 1
    
    # Watchdog event handlers
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory and self.should_index(event.src_path):
            asyncio.create_task(self._queue_change(event.src_path, 'create'))
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and self.should_index(event.src_path):
            asyncio.create_task(self._queue_change(event.src_path, 'update'))
    
    def on_deleted(self, event):
        """Handle file deletion"""
        if not event.is_directory:
            asyncio.create_task(self._queue_change(event.src_path, 'delete'))
    
    async def _queue_change(self, file_path: str, event_type: str):
        """Add file change to processing queue"""
        try:
            await self.pending_queue.put((file_path, event_type))
        except asyncio.QueueFull:
            logger.error(f"Queue full! Dropping event: {file_path} ({event_type})")
            self.metrics['files_skipped'] += 1
    
    async def initial_index(self):
        """Perform initial indexing of existing files"""
        logger.info(f"Starting initial index of {self.project_path}")
        
        # Check if we've already indexed
        if not self.degraded_mode['qdrant']:
            try:
                collection_name = f"{self.collection_prefix}code"
                info = await self.container.qdrant.get_collection_info(collection_name)
                if info.get('points_count', 0) > 0:
                    logger.info("Collection already has data - skipping initial index")
                    return
            except:
                pass
        
        # Find all files to index
        files_to_index = []
        for ext in self.watch_patterns:
            files_to_index.extend(self.project_path.rglob(f"*{ext}"))
        
        # Filter out ignored paths
        files_to_index = [f for f in files_to_index if self.should_index(str(f))]
        
        logger.info(f"Found {len(files_to_index)} files to index")
        
        # Index in batches
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i:i + self.batch_size]
            for file_path in batch:
                await self.index_file(str(file_path), "initial")
            await asyncio.sleep(0.1)  # Small delay between batches
        
        logger.info(f"Initial indexing complete: {self.metrics['files_indexed']} files")
    
    def get_metrics(self) -> Dict:
        """Get current metrics for monitoring with GraphRAG insights"""
        return {
            **self.metrics,
            'degraded_services': [k for k, v in self.degraded_mode.items() if v],
            'healthy_services': [k for k, v in self.degraded_mode.items() if not v],
            # GraphRAG health indicators
            'graphrag_enabled': (
                not self.degraded_mode['neo4j'] and 
                not self.degraded_mode['qdrant']
            ),
            'cross_reference_ratio': (
                self.metrics['cross_references_created'] / max(self.metrics['chunks_created'], 1)
                if self.metrics['chunks_created'] > 0 else 0
            ),
            'dedup_efficiency': (
                self.metrics['dedup_hits'] / max(self.metrics['files_indexed'] + self.metrics['dedup_hits'], 1)
                if (self.metrics['files_indexed'] + self.metrics['dedup_hits']) > 0 else 0
            )
        }

async def run_indexer(project_path: str, project_name: str = "default", initial_index: bool = False):
    """Main entry point for indexer service"""
    logger.info(f"Starting L9 Incremental Indexer for {project_path}")
    
    # Create indexer
    indexer = IncrementalIndexer(project_path, project_name)
    
    # Initialize async queue
    indexer.pending_queue = asyncio.Queue(maxsize=indexer.max_queue_size)
    
    # Initialize services
    await indexer.initialize_services()
    
    # Perform initial index if requested
    if initial_index:
        await indexer.initial_index()
    
    # Set up file watcher with debouncing
    observer = Observer()
    debounced_handler = DebouncedEventHandler(indexer, debounce_interval=2.0)
    observer.schedule(debounced_handler, project_path, recursive=True)
    observer.start()
    
    logger.info(f"Watching {project_path} for changes...")
    
    try:
        # Main processing loop
        while True:
            await indexer.process_queue()
            await asyncio.sleep(1)  # Check queue every second
            
            # Periodic metrics logging
            if int(time.time()) % 60 == 0:
                metrics = indexer.get_metrics()
                logger.info(f"Indexer metrics: {json.dumps(metrics, indent=2)}")
                
    except KeyboardInterrupt:
        logger.info("Shutting down indexer...")
        observer.stop()
    except Exception as e:
        logger.error(f"Indexer error: {e}")
        observer.stop()
        raise
    
    observer.join()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L9 Incremental Indexer Service")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--project-name", default="default", help="Project name")
    parser.add_argument("--initial-index", action="store_true", help="Perform initial indexing")
    
    args = parser.parse_args()
    
    asyncio.run(run_indexer(
        project_path=args.project_path,
        project_name=args.project_name,
        initial_index=args.initial_index
    ))