"""
Queue-Based Indexer Service
L9 2025 Architecture - ADR-0024 Implementation

Modified version of the indexer that queues files for async preprocessing
instead of directly processing them.
"""

import asyncio
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from watchdog.events import FileSystemEventHandler

from .async_preprocessing_pipeline import AsyncPreprocessingPipeline
from .exclusion_manager import ExclusionManager
from .service_container import ServiceContainer

logger = logging.getLogger(__name__)


class QueueBasedIndexer(FileSystemEventHandler):
    """
    Modified indexer that queues files for async preprocessing
    Integrates with the multi-stage pipeline for metadata tagging before embedding
    """
    
    def __init__(
        self, 
        project_path: str, 
        project_name: str = "default",
        container: ServiceContainer = None
    ):
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.container = container
        
        # Initialize exclusion manager
        self.exclusion_manager = ExclusionManager(str(self.project_path))
        
        # Initialize async preprocessing pipeline
        self.preprocessing_pipeline = AsyncPreprocessingPipeline(container)
        
        # File tracking
        self.file_hashes: Dict[str, str] = {}
        self.file_cooldowns: Dict[str, datetime] = {}
        self.cooldown_seconds = 3
        
        # Metrics
        self.metrics = {
            'files_queued': 0,
            'files_skipped_excluded': 0,
            'files_skipped_unchanged': 0,
            'files_skipped_cooldown': 0,
            'queue_errors': 0,
            'last_queue_time': None
        }
        
        # File patterns
        self.watch_patterns = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go',
            '.rs', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php',
            '.md', '.yaml', '.yml', '.json', '.toml', '.txt', '.sql'
        }
        
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
        # Observer for file system watching
        self.observer = None
        
        # Track if pipeline is running
        self.pipeline_task = None
        
    async def initialize(self):
        """Initialize the queue-based indexer"""
        logger.info(f"Initializing queue-based indexer for {self.project_name}")
        
        # Create default .graphragignore if needed
        self.exclusion_manager.create_default_graphragignore()
        
        # Start the preprocessing pipeline workers
        if not self.pipeline_task:
            self.pipeline_task = asyncio.create_task(
                self.preprocessing_pipeline.start_workers()
            )
            logger.info("Started async preprocessing pipeline workers")
        
        # Log exclusion statistics
        stats = self.exclusion_manager.get_statistics()
        logger.info(f"Exclusion manager initialized with {stats['total_rules']} rules")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down queue-based indexer")
        
        # Cancel pipeline task
        if self.pipeline_task:
            self.pipeline_task.cancel()
            try:
                await self.pipeline_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown pipeline
        await self.preprocessing_pipeline.shutdown()
    
    def should_index_file(self, file_path: str) -> bool:
        """Check if file should be indexed"""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists() or not path.is_file():
            return False
        
        # Check extension
        if path.suffix not in self.watch_patterns:
            return False
        
        # Check file size
        try:
            if path.stat().st_size > self.max_file_size:
                logger.debug(f"File too large: {file_path}")
                return False
        except Exception:
            return False
        
        # Check exclusion patterns
        if self.exclusion_manager.should_exclude(str(file_path)):
            self.metrics['files_skipped_excluded'] += 1
            logger.debug(f"File excluded by .graphragignore: {file_path}")
            return False
        
        # Check cooldown
        if file_path in self.file_cooldowns:
            time_since_last = datetime.now() - self.file_cooldowns[file_path]
            if time_since_last.total_seconds() < self.cooldown_seconds:
                self.metrics['files_skipped_cooldown'] += 1
                logger.debug(f"File in cooldown: {file_path}")
                return False
        
        return True
    
    def _compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute hash of file contents"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return None
    
    async def queue_file(self, file_path: str, action: str = 'create'):
        """Queue a file for preprocessing"""
        if not self.should_index_file(file_path):
            return
        
        # Check if content has changed
        current_hash = self._compute_file_hash(file_path)
        if not current_hash:
            return
        
        if file_path in self.file_hashes:
            if self.file_hashes[file_path] == current_hash:
                self.metrics['files_skipped_unchanged'] += 1
                logger.debug(f"File unchanged: {file_path}")
                return
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Queue for preprocessing
            await self.preprocessing_pipeline.queue_file(
                file_path=file_path,
                content=content,
                project_name=self.project_name,
                action=action
            )
            
            # Update tracking
            self.file_hashes[file_path] = current_hash
            self.file_cooldowns[file_path] = datetime.now()
            self.metrics['files_queued'] += 1
            self.metrics['last_queue_time'] = datetime.now().isoformat()
            
            logger.info(f"Queued {file_path} for preprocessing")
            
        except Exception as e:
            logger.error(f"Error queuing {file_path}: {e}")
            self.metrics['queue_errors'] += 1
    
    async def initial_index(self):
        """Queue all existing files for initial indexing"""
        logger.info(f"Starting initial indexing of {self.project_path}")
        
        files_queued = 0
        files_skipped = 0
        
        for pattern in self.watch_patterns:
            for file_path in self.project_path.rglob(f"*{pattern}"):
                if self.should_index_file(str(file_path)):
                    await self.queue_file(str(file_path), 'create')
                    files_queued += 1
                    
                    # Yield control periodically
                    if files_queued % 50 == 0:
                        await asyncio.sleep(0.1)
                        logger.info(f"Initial indexing progress: {files_queued} files queued")
                else:
                    files_skipped += 1
        
        logger.info(
            f"Initial indexing complete: {files_queued} files queued, "
            f"{files_skipped} files skipped"
        )
    
    # Watchdog event handlers
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            asyncio.create_task(self.queue_file(event.src_path, 'create'))
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            asyncio.create_task(self.queue_file(event.src_path, 'update'))
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            file_path = event.src_path
            if file_path in self.file_hashes:
                del self.file_hashes[file_path]
            if file_path in self.file_cooldowns:
                del self.file_cooldowns[file_path]
            logger.info(f"File deleted: {file_path}")
    
    def on_moved(self, event):
        """Handle file move events"""
        if not event.is_directory:
            # Remove old path
            if event.src_path in self.file_hashes:
                del self.file_hashes[event.src_path]
            if event.src_path in self.file_cooldowns:
                del self.file_cooldowns[event.src_path]
            
            # Queue new path
            asyncio.create_task(self.queue_file(event.dest_path, 'create'))
    
    async def get_status(self) -> Dict:
        """Get indexer status and metrics"""
        pipeline_status = await self.preprocessing_pipeline.get_queue_status()
        exclusion_stats = self.exclusion_manager.get_statistics()
        
        return {
            'indexer_metrics': self.metrics,
            'pipeline_status': pipeline_status,
            'exclusion_stats': exclusion_stats,
            'files_tracked': len(self.file_hashes),
            'project_path': str(self.project_path),
            'project_name': self.project_name
        }
    
    async def reindex_file(self, file_path: str):
        """Force reindex a specific file"""
        # Remove from hash cache to force reindexing
        if file_path in self.file_hashes:
            del self.file_hashes[file_path]
        
        await self.queue_file(file_path, 'update')
    
    async def reindex_directory(self, directory: str):
        """Force reindex all files in a directory"""
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory not found: {directory}")
            return
        
        files_requeued = 0
        for pattern in self.watch_patterns:
            for file_path in dir_path.rglob(f"*{pattern}"):
                # Remove from hash cache
                file_str = str(file_path)
                if file_str in self.file_hashes:
                    del self.file_hashes[file_str]
                
                if self.should_index_file(file_str):
                    await self.queue_file(file_str, 'update')
                    files_requeued += 1
        
        logger.info(f"Requeued {files_requeued} files from {directory}")