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
from typing import Dict, Set, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from collections import deque
import json

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Timer, Lock
from qdrant_client import models

# Add services directory to path for imports
services_dir = Path(__file__).parent
sys.path.insert(0, str(services_dir))

from service_container import ServiceContainer
from collection_config import get_collection_manager, CollectionType

# Configure logging to stderr for Docker
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to: {log_level}")

# Import code parser for structure extraction (ADR 0017)
STRUCTURE_EXTRACTION_ENABLED = os.getenv("STRUCTURE_EXTRACTION_ENABLED", "true").lower() == "true"
if STRUCTURE_EXTRACTION_ENABLED:
    try:
        from code_parser import CodeParser
        logger.info("Code structure extraction enabled (ADR 0017: GraphRAG)")
    except ImportError as e:
        logger.warning(f"Code parser not available: {e}")
        STRUCTURE_EXTRACTION_ENABLED = False

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
        
        # Store event loop reference for cross-thread task submission
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
    
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
                # Use run_coroutine_threadsafe since Timer runs in a separate thread
                if self._loop and self._loop.is_running():
                    if event_type == 'deleted':
                        asyncio.run_coroutine_threadsafe(
                            self.indexer._queue_change(path, 'delete'), 
                            self._loop
                        )
                    elif event_type == 'created':
                        asyncio.run_coroutine_threadsafe(
                            self.indexer._queue_change(path, 'create'), 
                            self._loop
                        )
                    elif event_type == 'modified':
                        asyncio.run_coroutine_threadsafe(
                            self.indexer._queue_change(path, 'update'), 
                            self._loop
                        )
                else:
                    logger.error(f"Event loop not available for {event_type} on {path}")
                
            except Exception as e:
                logger.error(f"Error queuing {event_type} for {path}: {e}")

class IncrementalIndexer(FileSystemEventHandler):
    """
    Handles file system events and performs incremental indexing
    Integrates with Neo4j GraphRAG, Qdrant vectors, and Nomic embeddings
    """
    
    def __init__(self, project_path: str, project_name: str = "default", container: ServiceContainer = None):
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.collection_manager = get_collection_manager(project_name)
        # Keep legacy collection_prefix for backward compatibility
        self.collection_prefix = f"project_{project_name}_"
        
        # Service container for accessing Neo4j, Qdrant, Nomic
        self.container = container
        self.services_initialized = False
        
        # Store event loop for cross-thread task submission (Python 2025 asyncio standard)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running during init - will be set later
            self._loop = None
        
        # Initialize code parser for structure extraction (ADR 0017)
        self.code_parser = None
        self.tree_sitter_extractor = None  # Always initialize this attribute
        
        if STRUCTURE_EXTRACTION_ENABLED:
            try:
                self.code_parser = CodeParser()
                logger.info("Code parser initialized for GraphRAG structure extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize tree-sitter extractor: {e}")
                self.code_parser = None
        
        # Observer for file system watching (will be set externally)
        self.observer = None
        
        # File tracking with deduplication
        self.file_hashes: Dict[str, str] = {}
        self.file_cooldowns: Dict[str, datetime] = {}
        self.cooldown_seconds = 3  # Prevent rapid re-indexing
        
        # Bounded queue for memory management
        self.pending_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
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
            'event_storms_handled': 0,  # Debouncing events
            # Error tracking
            'error_categories': {},
            'critical_errors': 0,  # Errors that prevent indexing
            'warning_count': 0,  # Non-critical issues
            'recovery_attempts': 0,  # Service reconnection attempts
            'last_error_time': None,
            'error_rate_per_minute': 0.0
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
                
                # Create container if not provided
                if self.container is None:
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
                    
                    # Capture actual embedding dimension for dynamic alignment
                    try:
                        health = await self.container.nomic.health_check()
                        embedding_dim = health.get('embedding_dim')
                        if embedding_dim and embedding_dim != self.collection_manager.embedding_dimension:
                            logger.info(f"Detected embedding dimension: {embedding_dim}, reinitializing collection manager")
                            # Reinitialize collection manager with correct dimension
                            from collection_config import CollectionManager
                            self.collection_manager = CollectionManager(
                                self.project_name, 
                                embedding_dimension=embedding_dim
                            )
                            logger.info(f"Collection manager updated with embedding dimension: {embedding_dim}")
                    except Exception as e:
                        logger.warning(f"Could not detect embedding dimension: {e}")
                        
                else:
                    logger.warning("Nomic unavailable - continuing without semantic embeddings")
                    self.degraded_mode['nomic'] = True
                
                self.services_initialized = True
                logger.info(f"Services initialized: {'healthy' if result else 'degraded'}")
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
        """Ensure required Qdrant collections exist using battle-tested patterns"""
        if self.degraded_mode['qdrant']:
            return
        
        # Use centralized collection management
        success = await self.collection_manager.ensure_collection_exists(
            CollectionType.CODE, 
            self.container.qdrant
        )
        
        if not success:
            logger.error("Failed to ensure code collection exists")
            self.degraded_mode['qdrant'] = True
    
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
            
            # 3. Fallback: Basic file tracking when all services are degraded
            if not success and self.degraded_mode['neo4j'] and self.degraded_mode['qdrant']:
                # At minimum, track that we've seen the file
                logger.debug(f"All services degraded - basic tracking for: {file_path}")
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
            error_category = self._categorize_error(e, file_path)
            logger.error(f"Failed to index {file_path} ({error_category}): {e}")
            self.metrics['errors'] += 1
            
            # Use comprehensive error handling
            self._handle_error_metrics(error_category)
            
            # Attempt service recovery for service-related errors
            service_errors = {
                'neo4j': 'neo4j',
                'qdrant': 'qdrant', 
                'embedding': 'nomic',
                'embedding_generation': 'nomic',
                'connectivity': None  # Could be any service
            }
            
            if error_category in service_errors and service_errors[error_category]:
                service_name = service_errors[error_category]
                if self.degraded_mode[service_name]:
                    # Try recovery if service is degraded and enough time has passed
                    await self._attempt_service_recovery(service_name)
    
    def _categorize_error(self, error: Exception, file_path: str) -> str:
        """Categorize errors for better monitoring and debugging"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # File system errors
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return 'filesystem'
        if isinstance(error, UnicodeDecodeError):
            return 'encoding'
        if 'no space left' in error_msg or 'disk full' in error_msg:
            return 'disk_space'
            
        # Network/service errors
        if 'connection' in error_msg or 'timeout' in error_msg:
            return 'connectivity'
        if 'authentication' in error_msg or 'auth' in error_msg:
            return 'authentication'
        if 'neo4j' in error_msg:
            return 'neo4j'
        if 'qdrant' in error_msg:
            return 'qdrant'
        if 'embedding' in error_msg or 'nomic' in error_msg:
            return 'embedding'
            
        # Content processing errors
        if isinstance(error, (ValueError, TypeError)) and 'chunk' in error_msg:
            return 'chunking'
        if 'embedding' in error_msg:
            return 'embedding_generation'
        if 'cypher' in error_msg or 'query' in error_msg:
            return 'query_execution'
            
        # File-specific categorization
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in ['.py', '.js', '.ts'] and 'syntax' in error_msg:
                return 'syntax_error'
            if Path(file_path).stat().st_size > self.max_file_size:
                return 'file_too_large'
                
        # Memory/resource errors
        if isinstance(error, MemoryError) or 'memory' in error_msg:
            return 'memory'
        if 'queue full' in error_msg:
            return 'queue_overflow'
            
        # Default classification
        return f'unknown_{error_type.lower()}'
    
    def _is_critical_error(self, error_category: str) -> bool:
        """Determine if an error category is critical (prevents indexing)"""
        critical_categories = {
            'filesystem', 'disk_space', 'memory', 'queue_overflow',
            'authentication', 'file_too_large'
        }
        return error_category in critical_categories
    
    def _handle_error_metrics(self, error_category: str):
        """Update error metrics with categorization and criticality"""
        # Update error categories
        if 'error_categories' not in self.metrics:
            self.metrics['error_categories'] = {}
        self.metrics['error_categories'][error_category] = self.metrics['error_categories'].get(error_category, 0) + 1
        
        # Track critical vs warning errors
        if self._is_critical_error(error_category):
            self.metrics['critical_errors'] += 1
        else:
            self.metrics['warning_count'] += 1
        
        # Update timestamp and calculate error rate
        self.metrics['last_error_time'] = datetime.now().isoformat()
        self._update_error_rate()
    
    def _update_error_rate(self):
        """Calculate errors per minute for monitoring"""
        if self.metrics['total_index_time_ms'] > 0:
            time_minutes = self.metrics['total_index_time_ms'] / (1000 * 60)
            self.metrics['error_rate_per_minute'] = self.metrics['errors'] / max(time_minutes, 1)
    
    async def _attempt_service_recovery(self, service_name: str):
        """Attempt to recover a failed service"""
        try:
            self.metrics['recovery_attempts'] += 1
            logger.info(f"Attempting recovery of {service_name} service")
            
            if service_name == 'neo4j':
                # Reinitialize Neo4j connection
                result = await self.container.neo4j.initialize()
                if result.get('success'):
                    self.degraded_mode['neo4j'] = False
                    logger.info("Neo4j service recovered successfully")
                    return True
                    
            elif service_name == 'qdrant':
                # Reinitialize Qdrant connection
                result = await self.container.qdrant.health_check()
                if result.get('healthy'):
                    self.degraded_mode['qdrant'] = False
                    await self._ensure_collections()
                    logger.info("Qdrant service recovered successfully")
                    return True
                    
            elif service_name == 'nomic':
                # Test Nomic embedding service
                test_result = await self.container.nomic.get_embeddings(['test'])
                if test_result:
                    self.degraded_mode['nomic'] = False
                    logger.info("Nomic embedding service recovered successfully")
                    return True
                    
        except Exception as e:
            logger.warning(f"Recovery attempt failed for {service_name}: {e}")
        
        return False
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary for monitoring"""
        total_operations = self.metrics['files_indexed'] + self.metrics['files_skipped'] + self.metrics['errors']
        
        summary = {
            'error_overview': {
                'total_errors': self.metrics['errors'],
                'critical_errors': self.metrics['critical_errors'],
                'warnings': self.metrics['warning_count'],
                'error_rate': self.metrics['error_rate_per_minute'],
                'last_error': self.metrics['last_error_time']
            },
            'error_categories': self.metrics.get('error_categories', {}),
            'service_health': {
                'degraded_services': [k for k, v in self.degraded_mode.items() if v],
                'service_failures': self.metrics['service_failures'],
                'recovery_attempts': self.metrics['recovery_attempts']
            },
            'operational_health': {
                'success_rate': (self.metrics['files_indexed'] / max(total_operations, 1)) * 100,
                'queue_health': 'healthy' if self.metrics['queue_depth'] < self.queue_warning_threshold else 'warning',
                'dedup_efficiency': (self.metrics['dedup_hits'] / max(self.metrics['files_indexed'] + self.metrics['dedup_hits'], 1)) * 100
            }
        }
        
        return summary
    
    def get_health_status(self) -> str:
        """Get overall system health status"""
        # Critical: Any critical errors or all services degraded
        if (self.metrics['critical_errors'] > 0 or 
            all(self.degraded_mode.values())):
            return 'critical'
        
        # Warning: Some services degraded or high error rate
        elif (any(self.degraded_mode.values()) or 
              self.metrics['error_rate_per_minute'] > 5.0 or
              self.metrics['queue_depth'] > self.queue_warning_threshold):
            return 'warning'
        
        # Healthy: All services operational, low error rate
        else:
            return 'healthy'
    
    def _chunk_content(self, content: str, file_path: str) -> List[Dict]:
        """Battle-tested semantic chunking based on LlamaIndex patterns"""
        chunks = []
        
        # Battle-tested parameters from LlamaIndex research
        CHUNK_SIZE = 512  # tokens (optimal for semantic search)
        CHUNK_OVERLAP = 50  # 10% overlap for context preservation
        MIN_CHUNK_SIZE = 100  # minimum viable chunk
        
        # Estimate tokens (rough: ~4 chars per token)
        approx_tokens = len(content) // 4
        
        if approx_tokens <= CHUNK_SIZE:
            # Single chunk for small content
            return [{
                'text': content,
                'start_line': 1,
                'end_line': len(content.split('\n')),
                'chunk_type': 'single',
                'tokens_estimate': approx_tokens
            }]
        
        # Semantic chunking for code files
        if file_path.endswith(('.py', '.js', '.ts', '.java', '.go', '.cpp', '.c', '.h')):
            chunks = self._semantic_code_chunking(content, file_path, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            # Hierarchical chunking for documentation/text
            chunks = self._hierarchical_text_chunking(content, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Ensure minimum chunk quality
        quality_chunks = []
        for chunk in chunks:
            if len(chunk['text'].strip()) >= MIN_CHUNK_SIZE:
                chunk['file_path'] = file_path
                chunk['chunk_id'] = f"{file_path}:{chunk['start_line']}-{chunk['end_line']}"
                quality_chunks.append(chunk)
        
        logger.debug(f"Chunked {file_path}: {len(quality_chunks)} chunks from {approx_tokens} tokens")
        return quality_chunks
    
    def _semantic_code_chunking(self, content: str, file_path: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Semantic chunking for code following LlamaIndex patterns"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_start = 1
        current_tokens = 0
        
        # Enhanced code boundary detection
        boundary_keywords = {
            '.py': ['def ', 'class ', 'async def ', '@', 'if __name__'],
            '.js': ['function ', 'const ', 'let ', 'var ', 'class ', 'export ', 'import '],
            '.ts': ['function ', 'const ', 'let ', 'var ', 'class ', 'export ', 'import ', 'interface ', 'type '],
            '.java': ['public class ', 'private class ', 'public interface ', 'public ', 'private '],
            '.go': ['func ', 'type ', 'var ', 'const ', 'package '],
            '.cpp': ['class ', 'struct ', 'namespace ', 'template ', 'void ', 'int ', 'bool '],
            '.c': ['struct ', 'void ', 'int ', 'bool ', 'static ', 'extern '],
            '.h': ['struct ', 'void ', 'int ', 'bool ', 'typedef ', '#define ']
        }
        
        ext = next((k for k in boundary_keywords.keys() if file_path.endswith(k)), '.py')
        keywords = boundary_keywords[ext]
        
        for i, line in enumerate(lines, 1):
            line_tokens = len(line) // 4  # rough token estimate
            
            # Check for logical boundary
            is_boundary = any(line.strip().startswith(kw) for kw in keywords)
            
            # Chunk decision: size limit OR semantic boundary with sufficient content
            should_chunk = (current_tokens + line_tokens > chunk_size) or \
                          (is_boundary and current_tokens > chunk_size // 2)
            
            if should_chunk and current_chunk:
                # Create chunk with overlap handling
                chunks.append({
                    'text': '\n'.join(current_chunk),
                    'start_line': current_start,
                    'end_line': i - 1,
                    'chunk_type': 'semantic_code',
                    'tokens_estimate': current_tokens,
                    'boundary_type': 'semantic' if is_boundary else 'size'
                })
                
                # Handle overlap: keep last few lines for context
                overlap_lines = min(overlap // 10, len(current_chunk) // 4, 5)
                if overlap_lines > 0:
                    current_chunk = current_chunk[-overlap_lines:] + [line]
                    current_start = i - overlap_lines
                    current_tokens = sum(len(l) // 4 for l in current_chunk)
                else:
                    current_chunk = [line]
                    current_start = i
                    current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Handle final chunk
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'start_line': current_start,
                'end_line': len(lines),
                'chunk_type': 'semantic_code',
                'tokens_estimate': current_tokens,
                'boundary_type': 'final'
            })
        
        return chunks
    
    def _hierarchical_text_chunking(self, content: str, chunk_size: int, overlap: int) -> List[Dict]:
        """Hierarchical chunking for text/docs following LlamaIndex patterns"""
        # Split by sentences/paragraphs for better semantic coherence
        import re
        
        # Enhanced sentence splitting that preserves code blocks
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)
        chunks = []
        current_chunk = []
        current_start_pos = 0
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence) // 4
            
            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_line': self._estimate_line_number(content, current_start_pos),
                    'end_line': self._estimate_line_number(content, current_start_pos + len(chunk_text)),
                    'chunk_type': 'hierarchical_text',
                    'tokens_estimate': current_tokens,
                    'boundary_type': 'sentence'
                })
                
                # Handle overlap
                overlap_sentences = min(overlap // 20, len(current_chunk) // 2)
                if overlap_sentences > 0:
                    current_chunk = current_chunk[-overlap_sentences:] + [sentence]
                    current_start_pos += len(' '.join(current_chunk[:-overlap_sentences-1])) + 1
                    current_tokens = sum(len(s) // 4 for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_start_pos += len(' '.join(current_chunk[:-1])) + 1
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_line': self._estimate_line_number(content, current_start_pos),
                'end_line': self._estimate_line_number(content, current_start_pos + len(chunk_text)),
                'chunk_type': 'hierarchical_text',
                'tokens_estimate': current_tokens,
                'boundary_type': 'final'
            })
        
        return chunks
    
    def _estimate_line_number(self, content: str, char_position: int) -> int:
        """Estimate line number from character position"""
        if char_position >= len(content):
            return len(content.split('\n'))
        return content[:char_position].count('\n') + 1
    
    async def _index_semantic(self, file_path: str, relative_path: Path, chunks: List[Dict]):
        """Index with full semantic embeddings and GraphRAG cross-references"""
        try:
            # Extract symbols if tree-sitter is available
            symbols_data = None
            if self.tree_sitter_extractor and file_path.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                logger.debug(f"Attempting symbol extraction for {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    symbols_result = await self.tree_sitter_extractor.extract_symbols_from_file(
                        file_path, content, timeout=5.0
                    )
                    
                    if symbols_result and not symbols_result.get('error'):
                        symbols_data = symbols_result.get('symbols', [])
                        logger.info(f"✓ Extracted {len(symbols_data)} symbols from {file_path}")
                        # Log first few symbols for verification
                        for symbol in symbols_data[:3]:
                            logger.debug(f"  - {symbol['type']}: {symbol['name']}")
                    else:
                        logger.debug(f"No symbols extracted from {file_path}: {symbols_result.get('error', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Symbol extraction failed for {file_path}: {e}")
            else:
                if not self.tree_sitter_extractor:
                    logger.debug("Tree-sitter extractor not available")
                elif not file_path.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                    logger.debug(f"Skipping symbol extraction for non-supported file: {file_path}")
            
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
                # Convert to numeric ID for Qdrant (use first 15 hex chars to stay within Neo4j int64 range)
                # Neo4j max int: 9223372036854775807 (19 digits)
                # 15 hex chars = max 1152921504606846975 (19 digits, always safe)
                chunk_id = int(chunk_id_hash[:15], 16)
                
                # Create sparse vector from keywords
                sparse_indices, sparse_values = self._extract_keywords(chunk['text'])
                
                # Find symbols that overlap with this chunk
                chunk_symbols = []
                if symbols_data:
                    for symbol in symbols_data:
                        # Check if symbol is within this chunk's line range
                        if (symbol['start_line'] >= chunk['start_line'] and 
                            symbol['start_line'] <= chunk['end_line']):
                            chunk_symbols.append({
                                'type': symbol['type'],
                                'name': symbol['name'],
                                'qualified_name': symbol.get('qualified_name', symbol['name']),
                                'line': symbol['start_line'],
                                'language': symbol.get('language', 'unknown')
                            })
                
                payload = {
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
                
                # Add symbol information if available
                if chunk_symbols:
                    payload['symbols'] = chunk_symbols
                    payload['symbol_count'] = len(chunk_symbols)
                    payload['has_symbols'] = True
                    # Add primary symbol for better search
                    if chunk_symbols:
                        payload['primary_symbol'] = chunk_symbols[0]['qualified_name']
                else:
                    payload['has_symbols'] = False
                    payload['symbol_count'] = 0
                
                # Create PointStruct-compatible dict (no sparse_vector field)
                point = {
                    'id': chunk_id,
                    'vector': embedding,  # Use unnamed default vector
                    'payload': payload
                }
                # Note: sparse vectors would need separate handling if needed
                points.append(point)
                neo4j_chunk_ids.append({
                    'id': chunk_id_hash,
                    'qdrant_id': chunk_id,
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'chunk_type': self._detect_chunk_type(chunk['text']),
                    'content': chunk['text']  # Add content to be stored in Neo4j
                })
            
            # Upsert to Qdrant using proper collection management
            collection_name = self.collection_manager.get_collection_name(CollectionType.CODE)
            await self.container.qdrant.upsert_points(collection_name, points)
            
            # Update Neo4j with chunk references (if available)
            if not self.degraded_mode['neo4j'] and neo4j_chunk_ids:
                await self._update_neo4j_chunks(file_path, relative_path, neo4j_chunk_ids, symbols_data)
                self.metrics['cross_references_created'] += len(neo4j_chunk_ids)
            
        except Exception as e:
            error_category = self._categorize_error(e, file_path)
            logger.error(f"Semantic indexing failed for {file_path} ({error_category}): {e}")
            self.metrics['service_failures']['nomic'] += 1
            self._handle_error_metrics(error_category)
            
            # Enter degraded mode after threshold failures
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
            
            collection_name = self.collection_manager.get_collection_name(CollectionType.CODE)
            await self.container.qdrant.upsert_points(collection_name, points)
            
        except Exception as e:
            error_category = self._categorize_error(e, file_path)
            logger.error(f"Keyword indexing failed for {file_path} ({error_category}): {e}")
            self.metrics['service_failures']['qdrant'] += 1
            self._handle_error_metrics(error_category)
            
            # Enter degraded mode after threshold failures
            if self.metrics['service_failures']['qdrant'] > 10:
                logger.warning("Qdrant failures exceeded threshold - entering degraded mode")
                self.degraded_mode['qdrant'] = True
    
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
            # ADR-0029: Use composite key (project, path) for multi-project isolation
            cypher = """
            MERGE (f:File {path: $path, project: $project})
            SET f.name = $name,
                f.type = $type,
                f.size = $size,
                f.content_hash = $content_hash,
                f.indexed_at = datetime()
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
            
            # Extract structured code information using CodeParser (ADR 0017)
            if self.code_parser and file_type in self.code_parser.supported_extensions:
                structure = self.code_parser.extract_structure(content, file_type)
                
                # Create Function nodes
                for func in structure.get('functions', []):
                    await self.container.neo4j.execute_cypher("""
                        MERGE (f:Function {name: $name, file_path: $file_path, project: $project})
                        SET f.signature = $signature,
                            f.start_line = $start_line,
                            f.end_line = $end_line
                        WITH f
                        MATCH (file:File {path: $file_path, project: $project})
                        MERGE (f)-[:DEFINED_IN]->(file)
                    """, {
                        'name': func['name'],
                        'file_path': str(relative_path),
                        'signature': func.get('signature', ''),
                        'start_line': func.get('start_line', 0),
                        'end_line': func.get('end_line', 0),
                        'project': self.project_name
                    })
                
                # Create Class nodes
                for cls in structure.get('classes', []):
                    await self.container.neo4j.execute_cypher("""
                        MERGE (c:Class {name: $name, file_path: $file_path, project: $project})
                        SET c.start_line = $start_line,
                            c.end_line = $end_line
                        WITH c
                        MATCH (file:File {path: $file_path, project: $project})
                        MERGE (c)-[:DEFINED_IN]->(file)
                    """, {
                        'name': cls['name'],
                        'file_path': str(relative_path),
                        'start_line': cls.get('start_line', 0),
                        'end_line': cls.get('end_line', 0),
                        'project': self.project_name
                    })
                
                # Create CALLS relationships
                for call in structure.get('calls', []):
                    # Only create if both functions exist
                    await self.container.neo4j.execute_cypher("""
                        MATCH (caller:Function {file_path: $file_path, project: $project})
                        WHERE caller.start_line <= $call_line AND caller.end_line >= $call_line
                        MATCH (callee:Function {name: $callee_name, project: $project})
                        MERGE (caller)-[:CALLS]->(callee)
                    """, {
                        'file_path': str(relative_path),
                        'call_line': call.get('line', 0),
                        'callee_name': call.get('name', ''),
                        'project': self.project_name
                    })
                
                # Process imports with better extraction
                imports = structure.get('imports', [])
                for imp in imports:
                    # Create import relationships
                    cypher = """
                    MATCH (f:File {path: $from_path, project: $project})
                    MERGE (m:Module {name: $module, project: $project})
                    MERGE (f)-[:IMPORTS]->(m)
                    """
                    result = await self.container.neo4j.execute_cypher(cypher, {
                        'from_path': str(relative_path),
                        'module': imp,
                        'project': self.project_name
                    })
                    if result.get('status') != 'success':
                        logger.warning(f"Failed to create import relationship: {result.get('message')}")
            
        except Exception as e:
            error_category = self._categorize_error(e, file_path)
            logger.error(f"Graph indexing failed for {file_path} ({error_category}): {e}")
            self.metrics['service_failures']['neo4j'] += 1
            self._handle_error_metrics(error_category)
            
            # Enter degraded mode after threshold failures
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
    
    async def _update_neo4j_chunks(self, file_path: str, relative_path: Path, chunk_ids: List[Dict], symbols_data: Optional[List[Dict]] = None):
        """Create CodeChunk nodes in Neo4j with Qdrant cross-references and symbol information"""
        try:
            # Batch create/update chunk nodes
            for chunk_info in chunk_ids:
                # Use centralized collection management
                collection_name = self.collection_manager.get_collection_name(CollectionType.CODE)
                
                cypher = """
                MERGE (c:CodeChunk {id: $chunk_id, project: $project})
                SET c.qdrant_id = $qdrant_id,
                    c.file_path = $file_path,
                    c.start_line = $start_line,
                    c.end_line = $end_line,
                    c.type = $chunk_type,
                    c.content = $content,
                    c.collection_name = $collection_name,
                    c.indexed_at = datetime()
                WITH c
                MATCH (f:File {path: $file_path, project: $project})
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
                    'collection_name': collection_name,  # Add collection name parameter
                    'project': self.project_name
                })
                
                # Check if query executed successfully
                logger.debug(f"Neo4j execute result: {result}")
                if result.get('status') != 'success':
                    logger.error(f"Failed to update chunk in Neo4j: {result.get('message', 'Unknown error')}")
                    raise ValueError(f"Neo4j chunk update failed: {result.get('message')}")
            
            logger.debug(f"Created/updated {len(chunk_ids)} chunk nodes in Neo4j")
            
            # Create symbol nodes if we have symbol data
            if symbols_data and not self.degraded_mode['neo4j']:
                for symbol in symbols_data:
                    # Create nodes for classes and functions
                    if symbol['type'] in ['class', 'function', 'interface', 'method']:
                        cypher = """
                        MERGE (s:Symbol {qualified_name: $qualified_name})
                        SET s.name = $name,
                            s.type = $type,
                            s.file_path = $file_path,
                            s.start_line = $start_line,
                            s.end_line = $end_line,
                            s.language = $language,
                            s.docstring = $docstring,
                            s.indexed_at = datetime(),
                            s.project = $project
                        WITH s
                        MATCH (f:File {path: $file_path})
                        MERGE (f)-[:CONTAINS_SYMBOL]->(s)
                        """
                        
                        result = await self.container.neo4j.execute_write(cypher, {
                            'qualified_name': symbol.get('qualified_name', symbol['name']),
                            'name': symbol['name'],
                            'type': symbol['type'],
                            'file_path': str(relative_path),
                            'start_line': symbol['start_line'],
                            'end_line': symbol['end_line'],
                            'language': symbol.get('language', 'unknown'),
                            'docstring': symbol.get('docstring', ''),
                            'project': self.project_name
                        })
                        
                        # Create parent-child relationships for methods
                        if symbol['type'] == 'method' and symbol.get('parent_class'):
                            parent_cypher = """
                            MATCH (parent:Symbol {qualified_name: $parent_name})
                            MATCH (child:Symbol {qualified_name: $child_name})
                            MERGE (parent)-[:HAS_METHOD]->(child)
                            """
                            await self.container.neo4j.execute_write(parent_cypher, {
                                'parent_name': symbol['parent_class'],
                                'child_name': symbol.get('qualified_name', symbol['name'])
                            })
                
                logger.info(f"Created/updated {len(symbols_data)} symbol nodes in Neo4j")
            
        except Exception as e:
            error_category = self._categorize_error(e, "")
            logger.error(f"Failed to update Neo4j chunks ({error_category}): {e}")
            self._handle_error_metrics(error_category)
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
                collection_name = self.collection_manager.get_collection_name(CollectionType.CODE)
                
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
                    # Fallback: Delete by file path filter (use Qdrant models)
                    file_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key='full_path',
                                match=models.MatchValue(value=file_path)
                            )
                        ]
                    )
                    await self.container.qdrant.delete_points(
                        collection_name,
                        filter_conditions=file_filter
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
        # Initialize queue if not already done
        if self.pending_queue is None:
            self.pending_queue = asyncio.Queue(maxsize=self.max_queue_size)
            logger.warning("Queue was not initialized in process_queue, creating it now")
            return  # Nothing to process yet
        
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
            # Use run_coroutine_threadsafe for cross-thread task submission (Python 2025 standard)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._queue_change(event.src_path, 'create'), 
                    self._loop
                )
            else:
                logger.error(f"Event loop not available for file creation: {event.src_path}")
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and self.should_index(event.src_path):
            # Use run_coroutine_threadsafe for cross-thread task submission (Python 2025 standard)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._queue_change(event.src_path, 'update'), 
                    self._loop
                )
            else:
                logger.error(f"Event loop not available for file modification: {event.src_path}")
    
    def on_deleted(self, event):
        """Handle file deletion"""
        if not event.is_directory:
            # Use run_coroutine_threadsafe for cross-thread task submission (Python 2025 standard)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._queue_change(event.src_path, 'delete'), 
                    self._loop
                )
            else:
                logger.error(f"Event loop not available for file deletion: {event.src_path}")
    
    async def _queue_change(self, file_path: str, event_type: str):
        """Add file change to processing queue"""
        try:
            # Initialize queue if not already done
            if self.pending_queue is None:
                self.pending_queue = asyncio.Queue(maxsize=self.max_queue_size)
                logger.warning("Queue was not initialized, creating it now")
            
            await self.pending_queue.put((file_path, event_type))
        except asyncio.QueueFull:
            logger.error(f"Queue full! Dropping event: {file_path} ({event_type})")
            self.metrics['files_skipped'] += 1
    
    async def initial_index(self):
        """Perform initial indexing of existing files"""
        logger.info(f"Starting initial index of {self.project_path}")
        
        # Check if we've already indexed
        logger.debug(f"Degraded mode status: {self.degraded_mode}")
        if not self.degraded_mode['qdrant']:
            try:
                logger.debug("Checking if collection already has data...")
                collection_name = self.collection_manager.get_collection_name(CollectionType.CODE)
                info = await self.container.qdrant.get_collection_info(collection_name)
                # Only skip if collection exists AND has points
                if info.get('exists', False) and info.get('points_count', 0) > 0:
                    logger.info(f"Collection already has data ({info.get('points_count')} points) - skipping initial index")
                    return
                else:
                    logger.debug(f"Collection status: exists={info.get('exists')}, points={info.get('points_count', 0)}")
            except Exception as e:
                logger.debug(f"Error checking collection (will proceed with indexing): {e}")
                # If we can't check, proceed with indexing
                pass
        
        # Find all files to index
        logger.debug("Starting file discovery...")
        files_to_index = []
        for ext in self.watch_patterns:
            logger.debug(f"Searching for files with extension: {ext}")
            matching_files = list(self.project_path.rglob(f"*{ext}"))
            logger.debug(f"Found {len(matching_files)} files with {ext}")
            files_to_index.extend(matching_files)
        
        logger.debug(f"Total files found before filtering: {len(files_to_index)}")
        
        # Filter out ignored paths
        files_to_index = [f for f in files_to_index if self.should_index(str(f))]
        
        logger.info(f"Found {len(files_to_index)} files to index after filtering")
        
        # Index in batches
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i:i + self.batch_size]
            for file_path in batch:
                await self.index_file(str(file_path), "initial")
            await asyncio.sleep(0.1)  # Small delay between batches
        
        logger.info(f"Initial indexing complete: {self.metrics['files_indexed']} files")
    
    async def start_monitoring(self):
        """Start file system monitoring for continuous indexing"""
        try:
            # Initialize async queue if not already initialized
            if self.pending_queue is None:
                self.pending_queue = asyncio.Queue(maxsize=self.max_queue_size)
                logger.info(f"Initialized pending queue with max size {self.max_queue_size}")
            
            # Load any existing state
            await self._load_persistent_state()
            
            # Set up file watcher with debouncing
            from watchdog.observers import Observer
            self.observer = Observer()
            debounced_handler = DebouncedEventHandler(self, debounce_interval=2.0)
            self.observer.schedule(debounced_handler, str(self.project_path), recursive=True)
            self.observer.start()
            
            logger.info(f"File system monitoring started for {self.project_path}")
            
            # Start queue processing loop
            await self._queue_processing_loop()
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}", exc_info=True)
            raise
    
    async def _queue_processing_loop(self):
        """Continuous queue processing loop with periodic state saving"""
        logger.info("Starting queue processing loop...")
        
        last_state_save = time.time()
        state_save_interval = 300  # Save state every 5 minutes
        
        while True:
            try:
                # Process queue items
                await self.process_queue()
                
                # Periodic state saving
                current_time = time.time()
                if current_time - last_state_save > state_save_interval:
                    try:
                        await self._save_persistent_state()
                        last_state_save = current_time
                        logger.debug("Periodic state save completed")
                    except Exception as e:
                        logger.warning(f"Periodic state save failed: {e}")
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info("Queue processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in queue processing loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Back off on errors
    
    async def shutdown(self):
        """Graceful shutdown procedure for container environments"""
        logger.info("Starting graceful indexer shutdown...")
        
        try:
            # Stop accepting new file events (observer should be stopped externally)
            if hasattr(self, 'observer') and self.observer:
                logger.info("Stopping file system observer...")
                self.observer.stop()
                self.observer.join(timeout=5.0)
            
            # Process any remaining items in queue
            if self.pending_queue and not self.pending_queue.empty():
                logger.info(f"Processing {self.pending_queue.qsize()} remaining items in queue...")
                # Process up to 50 items during shutdown to avoid hanging
                processed = 0
                while not self.pending_queue.empty() and processed < 50:
                    try:
                        await asyncio.wait_for(self.process_queue(), timeout=2.0)
                        processed += 1
                    except asyncio.TimeoutError:
                        logger.warning("Timeout processing queue item during shutdown")
                        break
                    except Exception as e:
                        logger.error(f"Error processing queue item during shutdown: {e}")
                        break
            
            # Save any pending state
            await self._save_persistent_state()
            
            # Close service connections
            if self.container:
                logger.info("Closing service connections...")
                if hasattr(self.container, 'close'):
                    await self.container.close()
                elif hasattr(self.container, 'neo4j') and self.container.neo4j:
                    try:
                        self.container.neo4j.close()
                    except Exception as e:
                        logger.warning(f"Error closing Neo4j connection: {e}")
            
            logger.info("Graceful shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}", exc_info=True)
            
    async def _save_persistent_state(self):
        """Save indexer state for recovery after restart with atomic writes and backup"""
        try:
            state_dir = Path("/app/state")
            state_dir.mkdir(parents=True, exist_ok=True)
            
            state_file = state_dir / "indexer_state.json"
            temp_file = state_dir / "indexer_state_temp.json"
            backup_file = state_dir / "indexer_state_backup.json"
            
            state = {
                "file_hashes": self.file_hashes,
                "metrics": self.metrics,
                "project_name": self.project_name,
                "last_save": datetime.now().isoformat(),
                "version": "1.0"  # For future schema migrations
            }
            
            # Atomic write: write to temp file first, then rename
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
                f.flush()  # Ensure data is written to disk
                
            # Create backup before replacing current state
            if state_file.exists():
                import shutil
                shutil.copy2(state_file, backup_file)
            
            # Atomically replace the state file
            temp_file.replace(state_file)
                
            logger.debug(f"Saved indexer state to {state_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save persistent state: {e}")
            # Clean up temp file if it exists
            try:
                temp_file = Path("/app/state/indexer_state_temp.json")
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
    
    async def _load_persistent_state(self):
        """Load indexer state from previous run with corruption detection"""
        try:
            state_file = Path("/app/state/indexer_state.json")
            backup_file = Path("/app/state/indexer_state_backup.json")
            
            if not state_file.exists() and not backup_file.exists():
                logger.info("No persistent state found, starting fresh")
                return
                
            # Try to load primary state file
            state = None
            state_source = "primary"
            
            try:
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    
                    # Validate state structure
                    if not isinstance(state, dict) or "file_hashes" not in state:
                        raise ValueError("Invalid state file structure")
                        
                    logger.info(f"Loaded primary state file")
                else:
                    raise FileNotFoundError("Primary state file missing")
                    
            except Exception as e:
                logger.warning(f"Primary state file corrupted or missing: {e}")
                
                # Try backup file
                if backup_file.exists():
                    try:
                        with open(backup_file, 'r') as f:
                            state = json.load(f)
                        
                        if not isinstance(state, dict) or "file_hashes" not in state:
                            raise ValueError("Invalid backup state structure")
                            
                        state_source = "backup"
                        logger.info("Loaded backup state file")
                    except Exception as backup_error:
                        logger.error(f"Backup state file also corrupted: {backup_error}")
                        logger.info("Starting fresh - all previous state lost")
                        return
            
            if state:
                # Restore state
                self.file_hashes = state.get("file_hashes", {})
                
                # Merge metrics, keeping counters from saved state  
                saved_metrics = state.get("metrics", {})
                self.metrics.update({
                    k: v for k, v in saved_metrics.items() 
                    if k in ["files_indexed", "chunks_created", "cross_references_created"]
                })
                
                # Validate loaded data
                if not isinstance(self.file_hashes, dict):
                    logger.warning("Invalid file_hashes in state, resetting")
                    self.file_hashes = {}
                
                logger.info(f"Loaded persistent state from {state_source}: {len(self.file_hashes)} file hashes, {self.metrics['files_indexed']} files indexed")
                
                # Create backup of current state for next time
                if state_source == "primary":
                    await self._create_backup_state()
            
        except Exception as e:
            logger.error(f"Critical error loading persistent state: {e}")
            logger.info("Starting fresh due to state loading failure")
    
    async def _create_backup_state(self):
        """Create backup of current state"""
        try:
            state_file = Path("/app/state/indexer_state.json")
            backup_file = Path("/app/state/indexer_state_backup.json")
            
            if state_file.exists():
                import shutil
                shutil.copy2(state_file, backup_file)
                logger.debug("State backup created")
        except Exception as e:
            logger.warning(f"Failed to create state backup: {e}")

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
            ),
            # Enhanced error reporting
            'health_status': self.get_health_status(),
            'error_summary': self.get_error_summary()
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
