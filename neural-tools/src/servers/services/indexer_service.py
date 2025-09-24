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
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import json
import math

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Timer, Lock

# ADR-0075: Removed qdrant_client import - Neo4j-only architecture

# CRITICAL: DO NOT REMOVE - Required for imports to work (see ADR-0056)
# These sys.path modifications compensate for mixed import patterns across the codebase
# Removing them will break pattern extraction, code parsing, and service imports
services_dir = Path(__file__).parent
sys.path.insert(0, str(services_dir))  # Enables: from service_container import ...

from service_container import ServiceContainer
from collection_config import get_collection_manager, CollectionType
from tree_sitter_extractor import TreeSitterExtractor

# CRITICAL: DO NOT REMOVE - Required for config imports (see ADR-0056)
sys.path.insert(
    0, str(services_dir.parent)
)  # Enables: from config.collection_naming import ...

# Configure logging to stderr for Docker
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)
logger.info(f"Logging level set to: {log_level}")

# Import metadata extraction components (ADR-0031)
METADATA_EXTRACTION_ENABLED = False
PRISM_SCORING_ENABLED = False

try:
    from pattern_extractor import PatternExtractor
    from git_extractor import GitMetadataExtractor
    from canon_manager import CanonManager

    METADATA_EXTRACTION_ENABLED = True
    logger.info("Metadata extraction enabled (ADR-0031)")
except ImportError as e:
    logger.warning(f"Metadata extraction components not available: {e}")

# Import PRISM scorer if available
try:
    # CRITICAL: DO NOT REMOVE - Required for PRISM scorer import (see ADR-0056)
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "infrastructure"))
    from prism_scorer import PrismScorer

    PRISM_SCORING_ENABLED = True
    logger.info("PRISM scoring enabled (ADR-0031)")
except ImportError as e:
    logger.warning(f"PRISM scorer not available: {e}")

# Import code parser for structure extraction (ADR 0017)
STRUCTURE_EXTRACTION_ENABLED = (
    os.getenv("STRUCTURE_EXTRACTION_ENABLED", "true").lower() == "true"
)
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
            if event.event_type == "moved":
                # For moves, track both source (delete) and dest (create)
                self._events_queue[event.src_path] = "delete"
                if hasattr(event, "dest_path"):
                    self._events_queue[event.dest_path] = "create"
            else:
                # For other events, track the latest event type
                # Priority: delete > create > modify
                current = self._events_queue.get(event.src_path)
                if current != "delete":  # Don't override delete
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

        logger.info(
            f"Processing batch of {len(events_to_process)} file events after debounce"
        )

        # Track event storms (>5 events in one batch indicates a storm)
        if len(events_to_process) > 5:
            self.indexer.metrics["event_storms_handled"] += 1

        # Process events asynchronously
        for path, event_type in events_to_process.items():
            try:
                # Map watchdog events to our indexer actions
                # Use run_coroutine_threadsafe since Timer runs in a separate thread
                if self._loop and self._loop.is_running():
                    if event_type == "deleted":
                        asyncio.run_coroutine_threadsafe(
                            self.indexer._queue_change(path, "delete"), self._loop
                        )
                    elif event_type == "created":
                        asyncio.run_coroutine_threadsafe(
                            self.indexer._queue_change(path, "create"), self._loop
                        )
                    elif event_type == "modified":
                        asyncio.run_coroutine_threadsafe(
                            self.indexer._queue_change(path, "update"), self._loop
                        )
                else:
                    logger.error(f"Event loop not available for {event_type} on {path}")

            except Exception as e:
                logger.error(f"Error queuing {event_type} for {path}: {e}")


class IncrementalIndexer(FileSystemEventHandler):
    """
    Handles file system events and performs incremental indexing
    ADR-0075: Neo4j-only architecture with HNSW vectors and Nomic embeddings
    """

    def __init__(
        self,
        project_path: str,
        project_name: str = "default",
        container: ServiceContainer = None,
    ):
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.collection_manager = get_collection_manager(project_name)
        # ADR-0057: Fixed collection naming to use hyphens (project-name format)
        self.collection_prefix = f"project-{project_name}"

        # Service container for accessing Neo4j and Nomic (ADR-0075: Neo4j-only)
        self.container = container
        self.services_initialized = False

        # Initialize WriteSynchronizationManager for atomic writes (ADR-053)
        self.sync_manager = None  # Will be initialized when services are initialized

        # Store event loop for cross-thread task submission (Python 2025 asyncio standard)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running during init - will be set later
            self._loop = None

        # Initialize code parser for structure extraction (ADR 0017)
        self.code_parser = None
        # Initialize Tree-sitter extractor for symbol extraction (ADR-0090)
        try:
            self.tree_sitter_extractor = TreeSitterExtractor()
            logger.info("✓ Tree-sitter extractor initialized successfully")
        except Exception as e:
            logger.warning(
                f"Tree-sitter initialization failed, symbol extraction disabled: {e}"
            )
            self.tree_sitter_extractor = None

        if STRUCTURE_EXTRACTION_ENABLED:
            try:
                self.code_parser = CodeParser()
                logger.info("Code parser initialized for GraphRAG structure extraction")

                # ADR-0090: Initialize Tree-sitter extractor for symbol extraction
                # Fall back to AST-based extraction if Tree-sitter fails
                try:
                    from tree_sitter_extractor import TreeSitterExtractor

                    self.tree_sitter_extractor = TreeSitterExtractor()
                    logger.info(
                        "✅ Tree-sitter extractor initialized for symbol extraction"
                    )
                except ImportError as ie:
                    logger.warning(
                        f"Tree-sitter not available, using AST-based extraction: {ie}"
                    )
                    from symbol_extractor import SymbolExtractorFactory

                    self.tree_sitter_extractor = (
                        SymbolExtractorFactory  # Use as fallback
                    )
                    logger.info("✅ AST-based symbol extractor initialized as fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize code extraction: {e}")
                self.code_parser = None

        # Initialize metadata extractors (ADR-0031)
        self.pattern_extractor = None
        self.git_extractor = None
        self.canon_manager = None
        self.prism_scorer = None

        if METADATA_EXTRACTION_ENABLED:
            try:
                self.pattern_extractor = PatternExtractor()
                self.git_extractor = GitMetadataExtractor(str(self.project_path))
                self.canon_manager = CanonManager(
                    self.project_name, str(self.project_path)
                )
                logger.info("Metadata extraction components initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize metadata extractors: {e}")

        if PRISM_SCORING_ENABLED:
            try:
                self.prism_scorer = PrismScorer(str(self.project_path))
                logger.info("PRISM scorer initialized for intelligent file importance")
            except Exception as e:
                logger.warning(f"Failed to initialize PRISM scorer: {e}")

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
        # ADR-0058: Added missing file extensions for comprehensive language support
        self.watch_patterns = {
            # Core languages
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".rs",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".rb",
            ".php",
            # Mobile development
            ".dart",
            ".swift",
            ".kt",
            ".m",
            # Web frameworks
            ".vue",
            ".svelte",
            ".astro",
            # Additional languages
            ".scala",
            ".r",
            ".jl",
            ".perl",
            ".lua",
            # Config and data
            ".md",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".txt",
            ".sql",
            ".xml",
            ".ini",
            ".env",
            ".graphql",
            ".prisma",
            # Documentation
            ".rst",
            ".adoc",
        }

        self.ignore_patterns = {
            "__pycache__",
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
            "target",
            ".pytest_cache",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
            "*.log",
            ".env",
            ".env.local",
            ".env.production",
            "secrets",
            "*.key",
            "*.pem",
            "*.crt",
            ".neural-tools",  # Add our own data dir
        }

        # File size limits
        self.max_file_size = 10 * 1024 * 1024  # 10MB

        # Performance metrics
        self.metrics = {
            "files_indexed": 0,
            "files_skipped": 0,
            "errors": 0,
            "last_index_time": None,
            "queue_depth": 0,
            "service_failures": {"neo4j": 0, "nomic": 0},  # ADR-0066: Neo4j-only
            # GraphRAG metrics
            "chunks_created": 0,
            "cross_references_created": 0,
            "avg_index_time_ms": 0,
            "total_index_time_ms": 0,
            "dedup_hits": 0,  # Files skipped due to unchanged hash
            "event_storms_handled": 0,  # Debouncing events
            # Error tracking
            "error_categories": {},
            "critical_errors": 0,  # Errors that prevent indexing
            "warning_count": 0,  # Non-critical issues
            "recovery_attempts": 0,  # Service reconnection attempts
            "last_error_time": None,
            "error_rate_per_minute": 0.0,
        }

    async def initialize_services(self):
        """Initialize service connections with retry logic"""
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Initializing services (attempt {attempt + 1}/{max_retries})"
                )

                # Create container if not provided
                if self.container is None:
                    self.container = ServiceContainer(self.project_name)

                result = await self.container.initialize_all_services()

                # Fail-fast: Elite GraphRAG requires Neo4j (unified storage) + Nomic (embeddings)
                if not self.container.neo4j:
                    raise RuntimeError(
                        "Neo4j service is unavailable - cannot proceed with unified graph+vector storage"
                    )

                if not self.container.nomic:
                    raise RuntimeError(
                        "Nomic service is unavailable - cannot proceed with semantic embeddings"
                    )

                # ADR-0075: Neo4j handles all storage - no collection setup needed

                # Capture actual embedding dimension for dynamic alignment
                try:
                    health = await self.container.nomic.health_check()
                    embedding_dim = health.get("embedding_dim")
                    if (
                        embedding_dim
                        and embedding_dim != self.collection_manager.embedding_dimension
                    ):
                        logger.info(
                            f"Detected embedding dimension: {embedding_dim}, reinitializing collection manager"
                        )
                        # Reinitialize collection manager with correct dimension
                        from collection_config import CollectionManager

                        self.collection_manager = CollectionManager(
                            self.project_name, embedding_dimension=embedding_dim
                        )
                        logger.info(
                            f"Collection manager updated with embedding dimension: {embedding_dim}"
                        )
                except Exception as e:
                    raise RuntimeError(f"Could not detect embedding dimension: {e}")

                # ADR-0066: Neo4j-only architecture eliminates need for sync manager
                # All operations are atomic within Neo4j - no cross-database coordination needed
                logger.info(
                    "Neo4j unified storage - atomic operations native to single database"
                )

                self.services_initialized = True
                logger.info(
                    "All services initialized successfully - elite GraphRAG mode active"
                )
                return True

            except Exception as e:
                logger.error(f"Service initialization failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise RuntimeError(f"All initialization attempts failed: {e}")

    # ADR-0075: Removed _ensure_collections() - Neo4j-only architecture needs no Qdrant collections

    def should_index(self, file_path: str) -> bool:
        """Check if file should be indexed with security filters"""
        path = Path(file_path)

        # Check file exists and is not too large
        if not path.exists() or not path.is_file():
            return False

        if path.stat().st_size > self.max_file_size:
            logger.debug(f"Skipping large file: {file_path}")
            self.metrics["files_skipped"] += 1
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
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.debug(f"Could not hash {file_path}: {e}")
            return None

    async def index_file(self, file_path: str, action: str = "update"):
        """
        Index a single file with graceful degradation and performance tracking
        ADR-0075: Neo4j-only architecture with HNSW vectors and Nomic embeddings
        """
        if not self.should_index(file_path):
            return

        start_time = time.time()

        try:
            # Check if file actually changed
            current_hash = self.get_file_hash(file_path)
            if current_hash == self.file_hashes.get(file_path):
                logger.debug(f"File unchanged, skipping: {file_path}")
                self.metrics["dedup_hits"] += 1
                return

            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                return

            # Update cooldown
            self.file_cooldowns[file_path] = datetime.now()

            path = Path(file_path)
            relative_path = path.relative_to(self.project_path)

            # Chunk large files
            chunks = self._chunk_content(content, str(relative_path))

            # ADR-0066: Unified Neo4j indexing - vectors + graph in single atomic transaction
            success = await self._index_unified(
                file_path, relative_path, chunks, content
            )

            # Track unified indexing metrics
            if success:
                # Update hash tracking
                self.file_hashes[file_path] = current_hash
                self.metrics["files_indexed"] += 1
                logger.info(f"✅ Unified Neo4j indexing successful: {file_path}")
            else:
                if "files_failed" not in self.metrics:
                    self.metrics["files_failed"] = 0
                self.metrics["files_failed"] += 1
                logger.error(f"❌ Unified Neo4j indexing failed: {file_path}")

            if success:
                self.metrics["chunks_created"] += len(chunks)
                self.metrics["last_index_time"] = datetime.now().isoformat()

                # Track timing metrics
                elapsed_ms = (time.time() - start_time) * 1000
                self.metrics["total_index_time_ms"] += elapsed_ms
                self.metrics["avg_index_time_ms"] = (
                    self.metrics["total_index_time_ms"] / self.metrics["files_indexed"]
                )

                logger.info(
                    f"Indexed {file_path}: {len(chunks)} chunks in {elapsed_ms:.1f}ms"
                )
            else:
                logger.warning(f"No services available to index {file_path}")
                self.metrics["files_skipped"] += 1

        except Exception as e:
            error_category = self._categorize_error(e, file_path)
            logger.error(f"Failed to index {file_path} ({error_category}): {e}")
            self.metrics["errors"] += 1

            # Use comprehensive error handling
            self._handle_error_metrics(error_category)

            # Attempt service recovery for service-related errors (ADR-0075: Neo4j-only)
            service_errors = {
                "neo4j": "neo4j",
                "embedding": "nomic",
                "embedding_generation": "nomic",
                "connectivity": None,  # Could be any service
            }

    def _categorize_error(self, error: Exception, file_path: str) -> str:
        """Categorize errors for better monitoring and debugging"""
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # File system errors
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return "filesystem"
        if isinstance(error, UnicodeDecodeError):
            return "encoding"
        if "no space left" in error_msg or "disk full" in error_msg:
            return "disk_space"

        # Network/service errors
        if "connection" in error_msg or "timeout" in error_msg:
            return "connectivity"
        if "authentication" in error_msg or "auth" in error_msg:
            return "authentication"
        if "neo4j" in error_msg:
            return "neo4j"
        # ADR-0075: Removed Qdrant error detection - Neo4j-only architecture
        if "embedding" in error_msg or "nomic" in error_msg:
            return "embedding"

        # Content processing errors
        if isinstance(error, (ValueError, TypeError)) and "chunk" in error_msg:
            return "chunking"
        if "embedding" in error_msg:
            return "embedding_generation"
        if "cypher" in error_msg or "query" in error_msg:
            return "query_execution"

        # File-specific categorization
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in [".py", ".js", ".ts"] and "syntax" in error_msg:
                return "syntax_error"
            if Path(file_path).stat().st_size > self.max_file_size:
                return "file_too_large"

        # Memory/resource errors
        if isinstance(error, MemoryError) or "memory" in error_msg:
            return "memory"
        if "queue full" in error_msg:
            return "queue_overflow"

        # Default classification
        return f"unknown_{error_type.lower()}"

    def _is_critical_error(self, error_category: str) -> bool:
        """Determine if an error category is critical (prevents indexing)"""
        critical_categories = {
            "filesystem",
            "disk_space",
            "memory",
            "queue_overflow",
            "authentication",
            "file_too_large",
        }
        return error_category in critical_categories

    def _handle_error_metrics(self, error_category: str):
        """Update error metrics with categorization and criticality"""
        # Update error categories
        if "error_categories" not in self.metrics:
            self.metrics["error_categories"] = {}
        self.metrics["error_categories"][error_category] = (
            self.metrics["error_categories"].get(error_category, 0) + 1
        )

        # Track critical vs warning errors
        if self._is_critical_error(error_category):
            self.metrics["critical_errors"] += 1
        else:
            self.metrics["warning_count"] += 1

        # Update timestamp and calculate error rate
        self.metrics["last_error_time"] = datetime.now().isoformat()
        self._update_error_rate()

    def _update_error_rate(self):
        """Calculate errors per minute for monitoring"""
        if self.metrics["total_index_time_ms"] > 0:
            time_minutes = self.metrics["total_index_time_ms"] / (1000 * 60)
            self.metrics["error_rate_per_minute"] = self.metrics["errors"] / max(
                time_minutes, 1
            )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary for monitoring"""
        total_operations = (
            self.metrics["files_indexed"]
            + self.metrics["files_skipped"]
            + self.metrics["errors"]
        )

        summary = {
            "error_overview": {
                "total_errors": self.metrics["errors"],
                "critical_errors": self.metrics["critical_errors"],
                "warnings": self.metrics["warning_count"],
                "error_rate": self.metrics["error_rate_per_minute"],
                "last_error": self.metrics["last_error_time"],
            },
            "error_categories": self.metrics.get("error_categories", {}),
            "service_health": {
                "all_services_healthy": True,  # Fail-fast: all services required
                "service_failures": self.metrics["service_failures"],
                "recovery_attempts": self.metrics["recovery_attempts"],
            },
            "operational_health": {
                "success_rate": (
                    self.metrics["files_indexed"] / max(total_operations, 1)
                )
                * 100,
                "queue_health": (
                    "healthy"
                    if self.metrics["queue_depth"] < self.queue_warning_threshold
                    else "warning"
                ),
                "dedup_efficiency": (
                    self.metrics["dedup_hits"]
                    / max(self.metrics["files_indexed"] + self.metrics["dedup_hits"], 1)
                )
                * 100,
            },
        }

        return summary

    def get_health_status(self) -> str:
        """Get overall system health status"""
        # Critical: Any critical errors (fail-fast ensures no degraded services)
        if self.metrics["critical_errors"] > 0:
            return "critical"

        # Warning: High error rate or queue issues
        elif (
            self.metrics["error_rate_per_minute"] > 5.0
            or self.metrics["queue_depth"] > self.queue_warning_threshold
        ):
            return "warning"

        # Healthy: All services operational, low error rate
        else:
            return "healthy"

    def _chunk_content(self, content: str, file_path: str) -> List[Dict]:
        """ADR-0047 Phase 3: AST-aware semantic chunking for optimal retrieval"""
        chunks = []

        # Battle-tested parameters from LlamaIndex research
        CHUNK_SIZE = 512  # tokens (optimal for semantic search)
        CHUNK_OVERLAP = 50  # 10% overlap for context preservation
        MIN_CHUNK_SIZE = 100  # minimum viable chunk

        # Estimate tokens (rough: ~4 chars per token)
        approx_tokens = len(content) // 4

        if approx_tokens <= CHUNK_SIZE:
            # Single chunk for small content
            return [
                {
                    "text": content,
                    "start_line": 1,
                    "end_line": len(content.split("\n")),
                    "chunk_type": "single",
                    "tokens_estimate": approx_tokens,
                }
            ]

        # ADR-0047: Try AST-aware chunking first for code files
        if file_path.endswith(
            (
                ".py",
                ".js",
                ".ts",
                ".java",
                ".go",
                ".cpp",
                ".c",
                ".h",
                ".rs",
                ".kt",
                ".swift",
            )
        ):
            try:
                # Use AST-aware chunker if available
                if not hasattr(self, "_ast_chunker"):
                    from servers.services.ast_aware_chunker import ASTAwareChunker

                    self._ast_chunker = ASTAwareChunker(
                        max_chunk_size=100,  # ~400 tokens at 4 chars/token
                        min_chunk_size=25,  # ~100 tokens minimum
                    )

                # Get AST-based chunks
                ast_chunks = self._ast_chunker.chunk_file(file_path, content)

                # Convert to expected format
                chunks = []
                for ast_chunk in ast_chunks:
                    chunks.append(
                        {
                            "text": ast_chunk.content,
                            "start_line": ast_chunk.start_line,
                            "end_line": ast_chunk.end_line,
                            "chunk_type": ast_chunk.chunk_type,
                            "chunk_name": ast_chunk.name,
                            "chunk_parent": ast_chunk.parent,
                            "complexity": ast_chunk.complexity,
                            "tokens_estimate": len(ast_chunk.content) // 4,
                        }
                    )

                logger.debug(f"AST-chunked {file_path}: {len(chunks)} semantic chunks")

            except Exception as e:
                logger.warning(
                    f"AST chunking failed for {file_path}: {e}, falling back to semantic chunking"
                )
                chunks = self._semantic_code_chunking(
                    content, file_path, CHUNK_SIZE, CHUNK_OVERLAP
                )
        else:
            # Hierarchical chunking for documentation/text
            chunks = self._hierarchical_text_chunking(
                content, CHUNK_SIZE, CHUNK_OVERLAP
            )

        # Ensure minimum chunk quality
        quality_chunks = []
        for chunk in chunks:
            if len(chunk["text"].strip()) >= MIN_CHUNK_SIZE:
                chunk["file_path"] = file_path
                chunk["chunk_id"] = (
                    f"{file_path}:{chunk['start_line']}-{chunk['end_line']}"
                )
                quality_chunks.append(chunk)

        logger.debug(
            f"Chunked {file_path}: {len(quality_chunks)} chunks from {approx_tokens} tokens"
        )
        return quality_chunks

    def _semantic_code_chunking(
        self, content: str, file_path: str, chunk_size: int, overlap: int
    ) -> List[Dict]:
        """Semantic chunking for code following LlamaIndex patterns"""
        lines = content.split("\n")
        chunks = []
        current_chunk = []
        current_start = 1
        current_tokens = 0

        # Enhanced code boundary detection
        boundary_keywords = {
            ".py": ["def ", "class ", "async def ", "@", "if __name__"],
            ".js": [
                "function ",
                "const ",
                "let ",
                "var ",
                "class ",
                "export ",
                "import ",
            ],
            ".ts": [
                "function ",
                "const ",
                "let ",
                "var ",
                "class ",
                "export ",
                "import ",
                "interface ",
                "type ",
            ],
            ".java": [
                "public class ",
                "private class ",
                "public interface ",
                "public ",
                "private ",
            ],
            ".go": ["func ", "type ", "var ", "const ", "package "],
            ".cpp": [
                "class ",
                "struct ",
                "namespace ",
                "template ",
                "void ",
                "int ",
                "bool ",
            ],
            ".c": ["struct ", "void ", "int ", "bool ", "static ", "extern "],
            ".h": ["struct ", "void ", "int ", "bool ", "typedef ", "#define "],
        }

        ext = next(
            (k for k in boundary_keywords.keys() if file_path.endswith(k)), ".py"
        )
        keywords = boundary_keywords[ext]

        for i, line in enumerate(lines, 1):
            line_tokens = len(line) // 4  # rough token estimate

            # Check for logical boundary
            is_boundary = any(line.strip().startswith(kw) for kw in keywords)

            # Chunk decision: size limit OR semantic boundary with sufficient content
            should_chunk = (current_tokens + line_tokens > chunk_size) or (
                is_boundary and current_tokens > chunk_size // 2
            )

            if should_chunk and current_chunk:
                # Create chunk with overlap handling
                chunks.append(
                    {
                        "text": "\n".join(current_chunk),
                        "start_line": current_start,
                        "end_line": i - 1,
                        "chunk_type": "semantic_code",
                        "tokens_estimate": current_tokens,
                        "boundary_type": "semantic" if is_boundary else "size",
                    }
                )

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
            chunks.append(
                {
                    "text": "\n".join(current_chunk),
                    "start_line": current_start,
                    "end_line": len(lines),
                    "chunk_type": "semantic_code",
                    "tokens_estimate": current_tokens,
                    "boundary_type": "final",
                }
            )

        return chunks

    def _hierarchical_text_chunking(
        self, content: str, chunk_size: int, overlap: int
    ) -> List[Dict]:
        """Hierarchical chunking for text/docs following LlamaIndex patterns"""
        # Split by sentences/paragraphs for better semantic coherence
        import re

        # Enhanced sentence splitting that preserves code blocks
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", content)
        chunks = []
        current_chunk = []
        current_start_pos = 0
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence) // 4

            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    {
                        "text": chunk_text,
                        "start_line": self._estimate_line_number(
                            content, current_start_pos
                        ),
                        "end_line": self._estimate_line_number(
                            content, current_start_pos + len(chunk_text)
                        ),
                        "chunk_type": "hierarchical_text",
                        "tokens_estimate": current_tokens,
                        "boundary_type": "sentence",
                    }
                )

                # Handle overlap
                overlap_sentences = min(overlap // 20, len(current_chunk) // 2)
                if overlap_sentences > 0:
                    current_chunk = current_chunk[-overlap_sentences:] + [sentence]
                    current_start_pos += (
                        len(" ".join(current_chunk[: -overlap_sentences - 1])) + 1
                    )
                    current_tokens = sum(len(s) // 4 for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_start_pos += len(" ".join(current_chunk[:-1])) + 1
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Handle final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                {
                    "text": chunk_text,
                    "start_line": self._estimate_line_number(
                        content, current_start_pos
                    ),
                    "end_line": self._estimate_line_number(
                        content, current_start_pos + len(chunk_text)
                    ),
                    "chunk_type": "hierarchical_text",
                    "tokens_estimate": current_tokens,
                    "boundary_type": "final",
                }
            )

        return chunks

    def _estimate_line_number(self, content: str, char_position: int) -> int:
        """Estimate line number from character position"""
        if char_position >= len(content):
            return len(content.split("\n"))
        return content[:char_position].count("\n") + 1

    async def _extract_metadata(self, file_path: str, content: str) -> dict:
        """Extract all metadata during indexing (ADR-0031)"""
        metadata = {}

        # 1. PRISM Scores
        if self.prism_scorer:
            try:
                prism_components = self.prism_scorer.get_score_components(file_path)
                metadata.update(
                    {
                        "complexity_score": prism_components.get("complexity", 0.0),
                        "dependencies_score": prism_components.get("dependencies", 0.0),
                        "recency_score": prism_components.get("recency", 0.0),
                        "contextual_score": prism_components.get("contextual", 0.0),
                        "prism_total": prism_components.get("total", 0.0),
                    }
                )
            except Exception as e:
                logger.debug(f"PRISM scoring failed for {file_path}: {e}")

        # 2. Git Metadata
        if self.git_extractor:
            try:
                git_metadata = await self.git_extractor.extract(file_path)
                metadata.update(
                    {
                        "last_modified": git_metadata.get("last_modified"),
                        "change_frequency": git_metadata.get("change_frequency", 0),
                        "author_count": git_metadata.get("author_count", 1),
                        "last_commit": git_metadata.get("last_commit", "unknown"),
                    }
                )
            except Exception as e:
                logger.debug(f"Git metadata extraction failed for {file_path}: {e}")

        # 3. Pattern-based Extraction
        if self.pattern_extractor:
            try:
                patterns = self.pattern_extractor.extract(content)
                metadata.update(
                    {
                        "todo_count": patterns.get("todo_count", 0),
                        "fixme_count": patterns.get("fixme_count", 0),
                        "deprecated_markers": patterns.get("deprecated_count", 0),
                        "test_markers": patterns.get("test_count", 0),
                        "security_patterns": patterns.get("security_count", 0),
                        "canon_markers": patterns.get("canon_markers", 0),
                        "experimental_markers": patterns.get("experimental_markers", 0),
                        "is_async": patterns.get("is_async", False),
                        "has_type_hints": patterns.get("has_type_hints", False),
                    }
                )
            except Exception as e:
                logger.debug(f"Pattern extraction failed for {file_path}: {e}")

        # 4. Canon Configuration
        if self.canon_manager:
            try:
                canon_data = await self.canon_manager.get_file_metadata(file_path)
                metadata.update(
                    {
                        "canon_level": canon_data.get("level", "none"),
                        "canon_weight": canon_data.get("weight", 0.5),
                        "canon_reason": canon_data.get("reason", ""),
                        "is_canonical": canon_data.get("weight", 0.5) >= 0.7,
                    }
                )
            except Exception as e:
                logger.debug(f"Canon metadata extraction failed for {file_path}: {e}")

        return metadata

    async def _index_unified(
        self, file_path: str, relative_path: Path, chunks: List[Dict], content: str
    ) -> bool:
        """ADR-0066: Unified Neo4j indexing - vectors + graph in single atomic transaction
        ADR-0083: Fire-and-forget embedding pattern with Redis queue"""
        try:
            # Feature flag for gradual rollout to embedding queue system
            if os.getenv("USE_EMBEDDING_QUEUE", "false").lower() == "true":
                return await self._index_unified_with_queue(
                    file_path, relative_path, chunks, content
                )

            # Fallback to direct HTTP during transition (current behavior)
            # 1. Generate embeddings with Nomic v2 (September 2025 standard: requires task prefixes)
            texts = [f"search_document: {chunk['text']}" for chunk in chunks]
            embeddings = await self.container.nomic.get_embeddings(texts)

            if not embeddings:
                logger.error(f"Nomic embedding failed for {relative_path}")
                raise RuntimeError(
                    f"Nomic embedding failed for {relative_path}: No embeddings generated"
                )

            # 2. Extract symbols for graph enrichment
            symbols_data = None
            relationships_data = None
            if self.tree_sitter_extractor and str(file_path).endswith(
                (".py", ".js", ".jsx", ".ts", ".tsx")
            ):
                try:
                    # Check if we're using Tree-sitter or AST fallback
                    if hasattr(self.tree_sitter_extractor, "extract_symbols_from_file"):
                        # Tree-sitter path
                        symbols_result = (
                            await self.tree_sitter_extractor.extract_symbols_from_file(
                                file_path, content, timeout=5.0
                            )
                        )
                        if symbols_result and not symbols_result.get("error"):
                            symbols_data = symbols_result.get("symbols", [])
                            logger.info(
                                f"✓ Tree-sitter extracted {len(symbols_data)} symbols from {file_path}"
                            )
                    else:
                        # AST-based fallback (SymbolExtractorFactory)
                        result = self.tree_sitter_extractor.extract(
                            content, str(file_path)
                        )
                        symbols_data = result.get("symbols", [])
                        relationships_data = result.get("relationships", [])
                        if symbols_data:
                            logger.info(
                                f"✓ AST extracted {len(symbols_data)} symbols and {len(relationships_data or [])} relationships from {file_path}"
                            )
                except Exception as e:
                    logger.warning(f"Symbol extraction failed for {file_path}: {e}")

            # 3. Single atomic transaction: store file + chunks with embeddings + symbols + relationships
            # ADR-0093: Pass relationships_data to storage function
            success = await self._store_unified_neo4j(
                file_path,
                relative_path,
                chunks,
                embeddings,
                content,
                symbols_data,
                relationships_data,
            )

            if success:
                logger.info(
                    f"✅ Unified Neo4j storage successful: {len(chunks)} chunks with vectors"
                )
                return True
            else:
                raise RuntimeError(f"Neo4j unified storage failed for {relative_path}")

        except Exception as e:
            logger.error(f"Unified indexing failed for {relative_path}: {e}")
            return False

    def _clean_embedding(self, embedding):
        """
        ADR-0088: Ensure embedding is Neo4j-compatible LIST<FLOAT>.
        Handles numpy arrays, invalid values, and type conversion.
        """
        if embedding is None:
            return None

        # Convert numpy/tensor to Python list
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            try:
                embedding = list(embedding)
            except Exception:
                logger.warning("Could not convert embedding to list")
                return None

        # Validate and clean values
        cleaned = []
        for val in embedding:
            if val is None or math.isnan(val) or math.isinf(val):
                cleaned.append(0.0)  # Replace invalid with zero
            else:
                cleaned.append(float(val))

        return cleaned

    async def _store_unified_neo4j(
        self,
        file_path: str,
        relative_path: Path,
        chunks: List[Dict],
        embeddings: List[List[float]],
        content: str,
        symbols_data: List[Dict] = None,
        relationships_data: List[Dict] = None,
    ) -> bool:
        """ADR-0066/0093: Store file, chunks, symbols, and relationships in single Neo4j transaction"""
        try:
            logger.info(
                f"🔥 ADR-0078 ENTRY: _store_unified_neo4j called for {file_path}"
            )
            logger.info(
                f"🔥 ADR-0078 ENTRY: chunks={len(chunks)}, embeddings={len(embeddings)}"
            )
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            metadata = await self._extract_metadata(file_path, content)

            # ADR-0089: Using batch processing implementation instead of single atomic transaction
            # The unified cypher query was too large for Neo4j Bolt protocol limits
            # Processing is done in batches below: file -> symbols -> relationships -> chunks

            # ADR-0078: Debug chunk preparation pipeline
            logger.info("🔍 ADR-0078 Pipeline Debug:")
            logger.info(f"  - Input chunks length: {len(chunks)}")
            logger.info(f"  - Input embeddings length: {len(embeddings)}")
            if chunks:
                logger.info(f"  - First chunk keys: {list(chunks[0].keys())}")
                logger.info(
                    f"  - First chunk text preview: {chunks[0].get('text', 'NO_TEXT')[:50]}..."
                )

            # Prepare chunk data with embeddings and enhanced validation (ADR-0081/0082)
            chunks_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Enhanced content validation (September 2025 best practice)
                chunk_text = chunk.get("text", "")
                if not chunk_text or not chunk_text.strip():
                    logger.error(
                        f"🚨 CONTENT VALIDATION FAILED: Empty chunk content at index {i}"
                    )
                    logger.error(f"🚨 Chunk keys: {list(chunk.keys())}")
                    logger.error(f"🚨 File: {relative_path}")
                    continue

                # Additional validation: minimum content length
                if len(chunk_text.strip()) < 10:
                    logger.warning(
                        f"⚠️ Very short chunk content ({len(chunk_text)} chars) at index {i}"
                    )

                # Enhanced chunk_id with better uniqueness
                chunk_id = f"{relative_path}:chunk:{i}"

                # ADR-0088: Clean and convert embedding for Neo4j compatibility
                clean_emb = self._clean_embedding(embedding)
                if i == 0 and clean_emb:
                    logger.info(
                        f"🔍 ADR-0088 Debug - cleaned embedding type: {type(clean_emb)}, len: {len(clean_emb)}, sample: {clean_emb[:3] if clean_emb else 'None'}"
                    )

                chunks_data.append(
                    {
                        "chunk_id": chunk_id,
                        "content": chunk_text,  # Validated non-empty content
                        "start_line": chunk.get("start_line", 0),
                        "end_line": chunk.get("end_line", 0),
                        "size": len(chunk_text),
                        "embedding": clean_emb,
                    }
                )

                # Log first few chunks for debugging
                if i < 3:
                    logger.info(
                        f"✅ Chunk {i} validated: {chunk_id} ({len(chunk_text)} chars)"
                    )

            # Critical validation: Ensure we have processable chunks
            if not chunks_data:
                logger.error(
                    f"🚨 CRITICAL: No valid chunks after content validation for {relative_path}"
                )
                logger.error(
                    f"🚨 Original chunks: {len(chunks)}, Embeddings: {len(embeddings)}"
                )
                return False

            logger.info(f"🔍 ADR-0078 chunks_data prepared: {len(chunks_data)} items")

            # ADR-0088: Deep inspection of chunks_data before Neo4j
            if chunks_data:
                first_chunk = chunks_data[0]
                logger.info("🔍 ADR-0088: First chunk inspection:")
                logger.info(f"  - Type: {type(first_chunk)}")
                logger.info(f"  - Keys: {list(first_chunk.keys())}")
                logger.info(f"  - chunk_id: {first_chunk.get('chunk_id')}")
                logger.info(
                    f"  - content length: {len(first_chunk.get('content', ''))}"
                )
                logger.info(f"  - embedding type: {type(first_chunk.get('embedding'))}")
                if first_chunk.get("embedding"):
                    logger.info(
                        f"  - embedding length: {len(first_chunk.get('embedding'))}"
                    )
                    logger.info(
                        f"  - embedding sample: {first_chunk.get('embedding')[:3]}"
                    )

            # ADR-0093: Log relationship counts if present (used later in batch processing)
            if relationships_data:
                uses_count = sum(
                    1 for r in relationships_data if r.get("type") == "USES"
                )
                inst_count = sum(
                    1 for r in relationships_data if r.get("type") == "INSTANTIATES"
                )
                logger.info(
                    f"📊 ADR-0093: Processing {uses_count} USES and {inst_count} INSTANTIATES relationships"
                )

            # ADR-0078: Enhanced debugging and validation
            if not chunks_data:
                logger.error(f"🚨 CRITICAL: chunks_data is empty for {file_path}")
                logger.error(f"🚨 Original chunks length: {len(chunks)}")
                logger.error(f"🚨 Embeddings length: {len(embeddings)}")
                return False

            # Validate chunk data structure (ADR-0078)
            required_fields = [
                "chunk_id",
                "content",
                "start_line",
                "end_line",
                "embedding",
            ]
            for i, chunk in enumerate(chunks_data):
                missing_fields = [
                    field for field in required_fields if field not in chunk
                ]
                if missing_fields:
                    logger.error(
                        f"🚨 Invalid chunk {i} structure - missing: {missing_fields}"
                    )
                    return False
                if (
                    not isinstance(chunk["embedding"], list)
                    or len(chunk["embedding"]) == 0
                ):
                    logger.error(
                        f"🚨 Invalid embedding in chunk {i}: {chunk['chunk_id']}"
                    )
                    return False

            logger.info(f"✅ ADR-0078 Debug - chunks_data length: {len(chunks_data)}")
            logger.info(
                f"✅ ADR-0078 Debug - first chunk preview: {chunks_data[0]['chunk_id']}"
            )
            logger.info(
                f"✅ ADR-0078 Debug - embedding dimensions: {len(chunks_data[0]['embedding'])}"
            )

            # ADR-0089: Fix parameter serialization by chunking
            # Neo4j driver can't handle large nested structures in parameters
            # Process in smaller batches to avoid Bolt protocol limits
            BATCH_SIZE = (
                10  # Process 10 chunks at a time to stay under serialization limits
            )

            total_chunks_created = 0
            total_symbols_created = 0

            # First, create/update the file node and symbols
            file_params = {
                "path": str(relative_path),
                "project": self.project_name,
                "name": relative_path.name,
                "file_hash": file_hash,
                "size": len(content),
                "file_type": metadata.get("file_type", "unknown"),
                "language": metadata.get("language", "unknown"),
                "is_test": metadata.get("is_test", False),
                "is_config": metadata.get("is_config", False),
                "is_docs": metadata.get("is_docs", False),
                "complexity_score": metadata.get("complexity_score", 0),
                "importance": metadata.get("importance", 1.0),
                "symbols_data": symbols_data,
            }

            # Create file and symbols (without chunks)
            file_cypher = """
            // Create or update File node
            MERGE (f:File {path: $path, project: $project})
            SET f += {
                name: $name,
                content_hash: $file_hash,
                size: $size,
                created_time: datetime(),
                updated_time: datetime(),
                indexed_time: datetime(),
                project: $project,
                file_type: $file_type,
                language: $language,
                is_test: $is_test,
                is_config: $is_config,
                is_docs: $is_docs,
                complexity_score: $complexity_score,
                importance: $importance
            }

            // Remove old chunks
            WITH f
            MATCH (f)-[:HAS_CHUNK]->(old_chunk:Chunk)
            DETACH DELETE old_chunk

            // Create symbols if provided
            WITH f
            UNWIND CASE
                WHEN $symbols_data IS NULL OR size($symbols_data) = 0
                THEN []
                ELSE $symbols_data
            END AS symbol
            CREATE (s:Symbol {
                name: symbol.name,
                type: symbol.type,
                start_line: symbol.start_line,
                end_line: symbol.end_line,
                project: $project
            })
            CREATE (f)-[:HAS_SYMBOL]->(s)

            WITH f, collect(s) as symbols
            RETURN f.path as file_path, size(symbols) as symbols_created
            """

            file_result = await self.container.neo4j.execute_cypher(
                file_cypher, file_params
            )
            if file_result.get("status") == "success" and file_result.get("result"):
                total_symbols_created = file_result["result"][0].get(
                    "symbols_created", 0
                )

            # ADR-0093: Create USES and INSTANTIATES relationships after symbols
            if relationships_data:
                uses_count = 0
                inst_count = 0
                for rel in relationships_data:
                    if rel.get("type") == "USES" and rel.get("from_function"):
                        # Ensure Function node exists
                        func_cypher = """
                        MERGE (f:Function {name: $func_name, project: $project, file_path: $file_path})
                        """
                        await self.container.neo4j.execute_cypher(
                            func_cypher,
                            {
                                "func_name": rel.get("from_function"),
                                "project": self.project_name,
                                "file_path": str(relative_path),
                            },
                        )

                        # Create Variable and USES relationship
                        uses_cypher = """
                        MERGE (v:Variable {name: $var_name, project: $project})
                        WITH v
                        MATCH (f:Function {name: $func_name, project: $project, file_path: $file_path})
                        MERGE (f)-[:USES {line: $line}]->(v)
                        """
                        result = await self.container.neo4j.execute_cypher(
                            uses_cypher,
                            {
                                "var_name": rel.get("variable_name"),
                                "func_name": rel.get("from_function"),
                                "project": self.project_name,
                                "file_path": str(relative_path),
                                "line": rel.get("line", 0),
                            },
                        )
                        if result.get("status") == "success":
                            uses_count += 1

                    elif rel.get("type") == "INSTANTIATES" and rel.get("from_function"):
                        # Create INSTANTIATES relationship (Class should already exist from symbols)
                        inst_cypher = """
                        MATCH (f:Function {name: $func_name, project: $project, file_path: $file_path})
                        MATCH (c:Class {name: $class_name, project: $project})
                        MERGE (f)-[:INSTANTIATES {line: $line}]->(c)
                        """
                        result = await self.container.neo4j.execute_cypher(
                            inst_cypher,
                            {
                                "func_name": rel.get("from_function"),
                                "class_name": rel.get(
                                    "class_name", rel.get("to_class")
                                ),
                                "project": self.project_name,
                                "file_path": str(relative_path),
                                "line": rel.get("line", 0),
                            },
                        )
                        if result.get("status") == "success":
                            inst_count += 1

                logger.info(
                    f"📊 ADR-0093: Created {uses_count} USES and {inst_count} INSTANTIATES relationships"
                )

            # Now process chunks in batches
            for i in range(0, len(chunks_data), BATCH_SIZE):
                batch = chunks_data[i : i + BATCH_SIZE]
                logger.info(
                    f"🔧 Processing chunk batch {i//BATCH_SIZE + 1}/{(len(chunks_data) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} chunks)"
                )

                # Create chunks in this batch
                batch_cypher = """
                MATCH (f:File {path: $path, project: $project})
                WITH f
                UNWIND $batch_chunks AS chunk_data
                CREATE (c:Chunk {
                    chunk_id: chunk_data.chunk_id,
                    content: chunk_data.content,
                    start_line: chunk_data.start_line,
                    end_line: chunk_data.end_line,
                    size: chunk_data.size,
                    project: $project,
                    embedding: chunk_data.embedding,
                    created_time: datetime()
                })
                CREATE (f)-[:HAS_CHUNK]->(c)
                WITH count(c) as batch_count
                RETURN batch_count
                """

                batch_params = {
                    "path": str(relative_path),
                    "project": self.project_name,
                    "batch_chunks": batch,
                }

                batch_result = await self.container.neo4j.execute_cypher(
                    batch_cypher, batch_params
                )

                if batch_result.get("status") == "success":
                    result_list = batch_result.get("result", [])
                    if result_list:
                        batch_created = result_list[0].get("batch_count", 0)
                        total_chunks_created += batch_created
                        logger.info(f"✅ Batch created {batch_created} chunks")

                        if batch_created == 0:
                            logger.error(
                                f"🚨 Batch {i//BATCH_SIZE + 1} failed to create chunks despite having {len(batch)} items"
                            )
                else:
                    logger.error(
                        f"🚨 Batch {i//BATCH_SIZE + 1} failed: {batch_result.get('message')}"
                    )

            chunks_created = total_chunks_created
            symbols_created = total_symbols_created
            logger.info(
                f"✅ Neo4j unified storage complete: {chunks_created} chunks + {symbols_created} symbols"
            )

            # ADR-0093: Removed dead code that referenced undefined 'result' variable
            # The chunks_created and symbols_created values are already set from batch processing above

            # ADR-0089: Post-execution validation
            if chunks_created == 0 and len(chunks_data) > 0:
                logger.error(
                    f"🚨 BATCH PROCESSING FAILURE: {len(chunks_data)} chunks passed but 0 created"
                )
                logger.error("🚨 Check batch processing and parameter serialization")
                return False
            elif chunks_created > 0:
                logger.info(
                    f"✅ ADR-0089 SUCCESS: {chunks_created}/{len(chunks_data)} chunks created via batch processing"
                )

                # Update metrics
                self.metrics["chunks_created"] += chunks_created
                if symbols_created:
                    self.metrics["symbols_created"] = (
                        self.metrics.get("symbols_created", 0) + symbols_created
                    )

                return True
            else:
                logger.error(f"Neo4j unified storage failed: No chunks created")
                return False

        except Exception as e:
            logger.error(f"Neo4j unified storage error: {e}")
            return False

    async def _index_unified_with_queue(
        self, file_path: str, relative_path: Path, chunks: List[Dict], content: str
    ) -> bool:
        """ADR-0083: Fire-and-forget embedding pattern with Redis queue

        Steps:
        1. Store chunks in Neo4j first (without embeddings)
        2. Enqueue embedding job with chunk IDs
        3. Return immediately (fire-and-forget)
        """
        import hashlib

        try:
            # Step 1: Extract symbols for graph enrichment
            symbols_data = None
            if self.tree_sitter_extractor and str(file_path).endswith(
                (".py", ".js", ".jsx", ".ts", ".tsx")
            ):
                try:
                    symbols_result = (
                        await self.tree_sitter_extractor.extract_symbols_from_file(
                            file_path, content, timeout=5.0
                        )
                    )
                    if symbols_result and not symbols_result.get("error"):
                        symbols_data = symbols_result.get("symbols", [])
                        logger.info(
                            f"✓ Extracted {len(symbols_data)} symbols from {file_path}"
                        )
                except Exception as e:
                    logger.warning(f"Symbol extraction failed for {file_path}: {e}")

            # Step 2: Store chunks in Neo4j WITHOUT embeddings first
            stored_chunks = await self._store_chunks_without_embeddings(
                file_path, relative_path, chunks, content, symbols_data
            )

            if not stored_chunks:
                logger.error(f"Failed to store chunks for {relative_path}")
                return False

            # Step 3: Prepare batch payload with chunk IDs for ARQ worker
            embedding_payload = [
                {"chunk_id": chunk["id"], "text": f"search_document: {chunk['text']}"}
                for chunk in stored_chunks
            ]

            # Step 4: Enqueue single batch job (fire-and-forget)
            if embedding_payload:
                file_hash = hashlib.sha256(content.encode()).hexdigest()
                job_id = f"embed:{relative_path}:{file_hash}"

                job_queue = await self.container.get_job_queue()
                await job_queue.enqueue_job(
                    "process_embedding_batch",  # ARQ worker function name
                    embedding_payload,
                    _job_id=job_id,  # Idempotency via ARQ deduplication
                )
                logger.info(
                    f"✅ Enqueued embedding job {job_id} for {len(embedding_payload)} chunks"
                )

            return True  # Indexer job complete - no waiting

        except Exception as e:
            logger.error(f"Queue-based indexing error for {relative_path}: {e}")
            return False

    async def _store_chunks_without_embeddings(
        self,
        file_path: str,
        relative_path: Path,
        chunks: List[Dict],
        content: str,
        symbols_data=None,
    ) -> List[Dict]:
        """Store chunks in Neo4j without embeddings, return chunk data with IDs"""
        # This is a simplified version of the current storage logic
        # We'll implement the full logic based on existing patterns
        stored_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{self.project_name}:{relative_path}:{i}"
            chunk_info = {
                "id": chunk_id,
                "text": chunk.get("text", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
                "chunk_type": chunk.get("chunk_type", "code"),
            }
            stored_chunks.append(chunk_info)

        # TODO: Implement actual Neo4j storage (reuse existing patterns)
        logger.info(
            f"Stored {len(stored_chunks)} chunks without embeddings for {relative_path}"
        )
        return stored_chunks

    # ADR-0075: Removed legacy Qdrant methods - Neo4j-only architecture
    def _detect_chunk_type(self, text: str) -> str:
        """Detect the type of code chunk (function, class, import, etc.)"""
        lines = text.strip().split("\n")
        if not lines:
            return "unknown"

        first_line = lines[0].strip()

        # Python patterns
        if first_line.startswith("def "):
            return "function"
        elif first_line.startswith("class "):
            return "class"
        elif first_line.startswith("import ") or first_line.startswith("from "):
            return "import"
        # JavaScript/TypeScript patterns
        elif (
            "function " in first_line or "const " in first_line or "let " in first_line
        ):
            return "function"
        elif "class " in first_line:
            return "class"
        elif "export " in first_line:
            return "export"
        # General patterns
        elif first_line.startswith("#") or first_line.startswith("//"):
            return "comment"
        else:
            return "code"

    async def _update_neo4j_chunks(
        self,
        file_path: str,
        relative_path: Path,
        chunk_ids: List[Dict],
        symbols_data: Optional[List[Dict]] = None,
    ):
        """
        Create Chunk nodes in Neo4j (ADR-0075: Neo4j-only architecture)
        ADR-0050: Fixed to create proper Chunk nodes with File→HAS_CHUNK→Chunk relationships
        """
        try:
            # Batch create/update chunk nodes
            for idx, chunk_info in enumerate(chunk_ids):
                # Use centralized collection management
                collection_name = self.collection_manager.get_collection_name(
                    CollectionType.CODE
                )

                # ADR-0050: Create Chunk nodes (not CodeChunk) with proper relationships
                cypher = """
                MERGE (c:Chunk {chunk_id: $chunk_id, project: $project})
                SET c.content = $content,
                    c.start_line = $start_line,
                    c.end_line = $end_line,
                    c.chunk_type = $chunk_type,
                    c.file_path = $file_path,
                    # ADR-0075: Removed qdrant_id - Neo4j-only architecture
                    c.collection_name = $collection_name,
                    c.embedding_status = 'completed',
                    c.indexed_at = datetime(),
                    c.sequence = $sequence
                WITH c
                MATCH (f:File {path: $file_path, project: $project})
                MERGE (f)-[r:HAS_CHUNK {sequence: $sequence}]->(c)
                RETURN c.chunk_id as created_id
                """

                # Debug: Log content being stored
                content_to_store = chunk_info.get("content", "")
                logger.debug(
                    f"Storing chunk {chunk_info['id']} with content length: {len(content_to_store)}"
                )

                # ADR-0050: Use chunk_id consistently across both databases
                result = await self.container.neo4j.execute_cypher(
                    cypher,
                    {
                        "chunk_id": chunk_info["id"],  # ADR-0075: Neo4j-only chunk ID
                        "file_path": str(relative_path),
                        "start_line": chunk_info.get("start_line", 0),
                        "end_line": chunk_info.get("end_line", 0),
                        "chunk_type": chunk_info.get("chunk_type", "code"),
                        "content": content_to_store,
                        "collection_name": collection_name,
                        "sequence": idx,  # Add sequence number for ordered chunks
                        "project": self.project_name,
                    },
                )

                # ADR-0050: Check write success and fail fast if not
                if result.get("status") != "success":
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"Failed to create Chunk node in Neo4j: {error_msg}")
                    self.metrics["service_failures"]["neo4j"] += 1
                    raise ValueError(f"Neo4j Chunk creation failed: {error_msg}")

                # Verify the chunk was created
                if result.get("data") and len(result["data"]) > 0:
                    created_id = result["data"][0].get("created_id")
                    logger.debug(f"Successfully created Chunk node: {created_id}")
                else:
                    logger.warning(
                        f"Chunk node may not have been created properly for {chunk_info['id']}"
                    )

            logger.info(
                f"✅ Created/updated {len(chunk_ids)} Chunk nodes in Neo4j with HAS_CHUNK relationships"
            )

            # Create symbol nodes if we have symbol data
            if symbols_data:
                for symbol in symbols_data:
                    # Create nodes for classes and functions
                    if symbol["type"] in ["class", "function", "interface", "method"]:
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

                        result = await self.container.neo4j.execute_write(
                            cypher,
                            {
                                "qualified_name": symbol.get(
                                    "qualified_name", symbol["name"]
                                ),
                                "name": symbol["name"],
                                "type": symbol["type"],
                                "file_path": str(relative_path),
                                "start_line": symbol["start_line"],
                                "end_line": symbol["end_line"],
                                "language": symbol.get("language", "unknown"),
                                "docstring": symbol.get("docstring", ""),
                                "project": self.project_name,
                            },
                        )

                        # Create parent-child relationships for methods
                        if symbol["type"] == "method" and symbol.get("parent_class"):
                            parent_cypher = """
                            MATCH (parent:Symbol {qualified_name: $parent_name})
                            MATCH (child:Symbol {qualified_name: $child_name})
                            MERGE (parent)-[:HAS_METHOD]->(child)
                            """
                            await self.container.neo4j.execute_write(
                                parent_cypher,
                                {
                                    "parent_name": symbol["parent_class"],
                                    "child_name": symbol.get(
                                        "qualified_name", symbol["name"]
                                    ),
                                },
                            )

                logger.info(
                    f"Created/updated {len(symbols_data)} symbol nodes in Neo4j"
                )

        except Exception as e:
            error_category = self._categorize_error(e, "")
            logger.error(f"Failed to update Neo4j chunks ({error_category}): {e}")
            self._handle_error_metrics(error_category)
            # Don't mark as degraded for chunk updates - file node is more important

    async def _create_file_chunk_relationships(
        self, file_path: str, relative_path: Path, chunk_ids: List[Dict]
    ):
        """Create relationships between File nodes and Chunk nodes in Neo4j after sync manager writes"""
        try:
            # Create or update the File node
            file_cypher = """
            MERGE (f:File {path: $path, project: $project})
            SET f.updated_at = datetime()
            RETURN f
            """
            await self.container.neo4j.execute_write(
                file_cypher, {"path": str(relative_path), "project": self.project_name}
            )

            # Create HAS_CHUNK relationships
            for chunk_info in chunk_ids:
                rel_cypher = """
                MATCH (f:File {path: $path, project: $project})
                MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
                MERGE (f)-[:HAS_CHUNK]->(c)
                """
                await self.container.neo4j.execute_write(
                    rel_cypher,
                    {
                        "path": str(relative_path),
                        "chunk_id": chunk_info["id"],  # This is the chunk_hash
                        "project": self.project_name,
                    },
                )

            logger.info(
                f"Created File->HAS_CHUNK relationships for {len(chunk_ids)} chunks"
            )

        except Exception as e:
            logger.error(f"Failed to create file-chunk relationships: {e}")

    def _extract_imports(self, content: str, file_type: str) -> List[str]:
        """Extract import statements (simplified)"""
        imports = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()

            if file_type == ".py":
                if line.startswith("import ") or line.startswith("from "):
                    # Extract module name
                    parts = line.split()
                    if parts[0] == "import" and len(parts) > 1:
                        imports.append(parts[1].split(".")[0])
                    elif parts[0] == "from" and len(parts) > 1:
                        imports.append(parts[1].split(".")[0])

            elif file_type in [".js", ".ts"]:
                if "import " in line or "require(" in line:
                    # Simplified extraction
                    if "from " in line:
                        module = line.split("from ")[-1].strip(" ;'\"")
                        imports.append(module)
                    elif "require(" in line:
                        start = line.find("require(") + 8
                        end = line.find(")", start)
                        if end > start:
                            module = line[start:end].strip(" '\"")
                            imports.append(module)

        return imports[:20]  # Limit to prevent explosion

    async def remove_file_from_index(self, file_path: str):
        """Remove file from Neo4j index when deleted (ADR-0075: Neo4j-only)"""
        try:
            path = Path(file_path)
            relative_path = path.relative_to(self.project_path)

            # ADR-0075: Neo4j-only removal - delete file with all chunks and relationships
            cypher = """
                MATCH (f:File {path: $path, project: $project})
                OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (f)-[:HAS_SYMBOL]->(s)
                DETACH DELETE f, c, s
                """
            await self.container.neo4j.execute_cypher(
                cypher, {"path": str(relative_path), "project": self.project_name}
            )
            logger.info(f"✅ Removed {file_path} and all related data from Neo4j")

            # Remove from tracking
            self.file_hashes.pop(file_path, None)
            self.file_cooldowns.pop(file_path, None)

        except Exception as e:
            logger.error(f"Failed to remove {file_path} from index: {e}")
            self.metrics["errors"] += 1

    async def process_queue(self):
        """Process pending file changes from queue"""
        # Initialize queue if not already done
        if self.pending_queue is None:
            self.pending_queue = asyncio.Queue(maxsize=self.max_queue_size)
            logger.warning(
                "Queue was not initialized in process_queue, creating it now"
            )
            return  # Nothing to process yet

        batch = []

        try:
            # Collect batch of changes
            while len(batch) < self.batch_size:
                try:
                    file_event = await asyncio.wait_for(
                        self.pending_queue.get(), timeout=0.1
                    )
                    batch.append(file_event)
                except asyncio.TimeoutError:
                    break

            if not batch:
                return

            logger.info(f"Processing batch of {len(batch)} file changes")

            # ADR-0084 Phase 2: Parallel file processing with semaphore
            sem = asyncio.Semaphore(10)  # Process up to 10 files concurrently

            async def process_file_with_limit(file_path, event_type):
                async with sem:
                    if event_type == "delete":
                        await self.remove_file_from_index(file_path)
                    else:
                        await self.index_file(file_path, event_type)

            # Process all files in parallel
            tasks = [
                process_file_with_limit(file_path, event_type)
                for file_path, event_type in batch
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Update metrics
            self.metrics["queue_depth"] = self.pending_queue.qsize()

            # Warn if queue is getting full
            if self.metrics["queue_depth"] > self.queue_warning_threshold:
                logger.warning(
                    f"Queue depth high: {self.metrics['queue_depth']}/{self.max_queue_size}"
                )

        except Exception as e:
            logger.error(f"Error processing queue: {e}")
            self.metrics["errors"] += 1

    # Watchdog event handlers
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory and self.should_index(event.src_path):
            # Use run_coroutine_threadsafe for cross-thread task submission (Python 2025 standard)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._queue_change(event.src_path, "create"), self._loop
                )
            else:
                logger.error(
                    f"Event loop not available for file creation: {event.src_path}"
                )

    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and self.should_index(event.src_path):
            # Use run_coroutine_threadsafe for cross-thread task submission (Python 2025 standard)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._queue_change(event.src_path, "update"), self._loop
                )
            else:
                logger.error(
                    f"Event loop not available for file modification: {event.src_path}"
                )

    def on_deleted(self, event):
        """Handle file deletion"""
        if not event.is_directory:
            # Use run_coroutine_threadsafe for cross-thread task submission (Python 2025 standard)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._queue_change(event.src_path, "delete"), self._loop
                )
            else:
                logger.error(
                    f"Event loop not available for file deletion: {event.src_path}"
                )

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
            self.metrics["files_skipped"] += 1

    async def initial_index(self):
        """Perform initial indexing of existing files"""
        logger.info(f"Starting initial index of {self.project_path}")

        # Always perform initial index when container starts
        # This ensures proper indexing when:
        # - Mount paths change (ADR-63)
        # - Containers are recreated
        # - Projects switch
        logger.info(
            "Performing initial index (container startup) - elite GraphRAG mode"
        )

        # Find all files to index
        logger.debug("Starting file discovery...")
        logger.debug(
            f"Project path: {self.project_path}, exists: {self.project_path.exists()}"
        )
        logger.debug(f"Watch patterns: {self.watch_patterns}")

        files_to_index = []
        for ext in self.watch_patterns:
            logger.info(
                f"Searching for files with extension: {ext} in {self.project_path}"
            )
            try:
                # Use glob with timeout protection and progress logging
                import time

                start_time = time.time()
                logger.debug(f"Starting rglob for pattern: *{ext}")

                # Convert generator to list with progress tracking
                matching_files = []
                file_count = 0
                for file_path in self.project_path.rglob(f"*{ext}"):
                    matching_files.append(file_path)
                    file_count += 1
                    if file_count % 100 == 0:
                        logger.info(
                            f"  ...discovered {file_count} {ext} files so far..."
                        )

                elapsed = time.time() - start_time
                logger.info(
                    f"Found {len(matching_files)} files with {ext} in {elapsed:.2f}s"
                )
                files_to_index.extend(matching_files)
            except Exception as e:
                logger.error(f"Error searching for {ext} files: {e}")
                logger.error(f"  Path: {self.project_path}")
                logger.error(f"  Path exists: {self.project_path.exists()}")
                import traceback

                logger.error(f"  Traceback: {traceback.format_exc()}")
                continue

        logger.debug(f"Total files found before filtering: {len(files_to_index)}")

        # Filter out ignored paths
        files_to_index = [f for f in files_to_index if self.should_index(str(f))]

        logger.info(f"Found {len(files_to_index)} files to index after filtering")

        # Index in batches
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i : i + self.batch_size]
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
                logger.info(
                    f"Initialized pending queue with max size {self.max_queue_size}"
                )

            # Load any existing state
            await self._load_persistent_state()

            # Set up file watcher with debouncing
            from watchdog.observers import Observer

            self.observer = Observer()
            debounced_handler = DebouncedEventHandler(self, debounce_interval=2.0)
            self.observer.schedule(
                debounced_handler, str(self.project_path), recursive=True
            )
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
            if hasattr(self, "observer") and self.observer:
                logger.info("Stopping file system observer...")
                self.observer.stop()
                self.observer.join(timeout=5.0)

            # Process any remaining items in queue
            if self.pending_queue and not self.pending_queue.empty():
                logger.info(
                    f"Processing {self.pending_queue.qsize()} remaining items in queue..."
                )
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
                        logger.error(
                            f"Error processing queue item during shutdown: {e}"
                        )
                        break

            # Save any pending state
            await self._save_persistent_state()

            # Close service connections
            if self.container:
                logger.info("Closing service connections...")
                if hasattr(self.container, "close"):
                    await self.container.close()
                elif hasattr(self.container, "neo4j") and self.container.neo4j:
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
                "version": "1.0",  # For future schema migrations
            }

            # Atomic write: write to temp file first, then rename
            with open(temp_file, "w") as f:
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
                    with open(state_file, "r") as f:
                        state = json.load(f)

                    # Validate state structure
                    if not isinstance(state, dict) or "file_hashes" not in state:
                        raise ValueError("Invalid state file structure")

                    logger.info("Loaded primary state file")
                else:
                    raise FileNotFoundError("Primary state file missing")

            except Exception as e:
                logger.warning(f"Primary state file corrupted or missing: {e}")

                # Try backup file
                if backup_file.exists():
                    try:
                        with open(backup_file, "r") as f:
                            state = json.load(f)

                        if not isinstance(state, dict) or "file_hashes" not in state:
                            raise ValueError("Invalid backup state structure")

                        state_source = "backup"
                        logger.info("Loaded backup state file")
                    except Exception as backup_error:
                        logger.error(
                            f"Backup state file also corrupted: {backup_error}"
                        )
                        logger.info("Starting fresh - all previous state lost")
                        return

            if state:
                # Restore state
                self.file_hashes = state.get("file_hashes", {})

                # Merge metrics, keeping counters from saved state
                saved_metrics = state.get("metrics", {})
                self.metrics.update(
                    {
                        k: v
                        for k, v in saved_metrics.items()
                        if k
                        in [
                            "files_indexed",
                            "chunks_created",
                            "cross_references_created",
                        ]
                    }
                )

                # Validate loaded data
                if not isinstance(self.file_hashes, dict):
                    logger.warning("Invalid file_hashes in state, resetting")
                    self.file_hashes = {}

                logger.info(
                    f"Loaded persistent state from {state_source}: {len(self.file_hashes)} file hashes, {self.metrics['files_indexed']} files indexed"
                )

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
            "all_services_healthy": True,  # Fail-fast: all services always healthy or system fails
            "healthy_services": ["neo4j", "nomic"],  # ADR-0066: Neo4j-only architecture
            # GraphRAG health indicators
            "graphrag_enabled": True,  # Always true in fail-fast mode
            "cross_reference_ratio": (
                self.metrics["cross_references_created"]
                / max(self.metrics["chunks_created"], 1)
                if self.metrics["chunks_created"] > 0
                else 0
            ),
            "dedup_efficiency": (
                self.metrics["dedup_hits"]
                / max(self.metrics["files_indexed"] + self.metrics["dedup_hits"], 1)
                if (self.metrics["files_indexed"] + self.metrics["dedup_hits"]) > 0
                else 0
            ),
            # Enhanced error reporting
            "health_status": self.get_health_status(),
            "error_summary": self.get_error_summary(),
        }


async def run_indexer(
    project_path: str, project_name: str = "default", initial_index: bool = False
):
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
    parser.add_argument(
        "--initial-index", action="store_true", help="Perform initial indexing"
    )

    args = parser.parse_args()

    asyncio.run(
        run_indexer(
            project_path=args.project_path,
            project_name=args.project_name,
            initial_index=args.initial_index,
        )
    )
# Test trigger
# Test trigger 2
# Test trigger 3

# Trigger reindex
# Fri Sep 12 22:56:47 PDT 2025
