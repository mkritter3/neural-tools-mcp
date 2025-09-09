#!/usr/bin/env python3
"""
Enhanced File Watcher for Selective Reprocessing - Phase 1.1
Implements debounced file monitoring with selective chunk updating
"""

import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, Set, Optional, Callable, Awaitable, List
import time
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent

logger = logging.getLogger(__name__)

class EnhancedFileWatcher:
    """Enhanced file watcher with debouncing and selective reprocessing"""
    
    def __init__(
        self,
        watch_path: str,
        callback: Callable[[str], Awaitable[None]],
        debounce_delay: float = 0.5,
        patterns: Optional[List[str]] = None
    ):
        """
        Initialize enhanced file watcher
        
        Args:
            watch_path: Directory path to watch
            callback: Async callback function to call on file changes
            debounce_delay: Delay in seconds for debouncing rapid changes
            patterns: List of file patterns to watch (e.g., ['*.py', '*.txt'])
        """
        self.watch_path = Path(watch_path)
        self.callback = callback
        self.debounce_delay = debounce_delay
        self.patterns = patterns or ['*']
        
        # File state tracking
        self.file_hashes: Dict[str, str] = {}
        self.pending_files: Dict[str, float] = {}  # file_path -> last_change_time
        self.debounce_tasks: Dict[str, asyncio.Task] = {}
        
        # Observer setup
        self.observer: Optional[Observer] = None
        self.event_handler: Optional[DebouncedEventHandler] = None
        self.is_running = False
        
        logger.info(f"Enhanced file watcher initialized for {watch_path}")
    
    def _matches_pattern(self, file_path: str) -> bool:
        """Check if file matches any of the watch patterns"""
        path = Path(file_path)
        for pattern in self.patterns:
            if pattern == '*' or path.match(pattern):
                return True
        return False
    
    def _compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute SHA256 hash of file contents"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except (OSError, IOError) as e:
            logger.debug(f"Could not hash {file_path}: {e}")
            return None
    
    async def _debounced_file_callback(self, file_path: str):
        """Debounced callback that processes file changes after delay"""
        try:
            # Wait for debounce delay
            await asyncio.sleep(self.debounce_delay)
            
            # Check if file was changed again during debounce
            if file_path in self.pending_files:
                last_change = self.pending_files[file_path]
                if time.time() - last_change < self.debounce_delay:
                    # File was changed again, this task will be cancelled
                    return
            
            # Remove from pending and process
            self.pending_files.pop(file_path, None)
            
            # Check if file actually changed (selective reprocessing)
            if Path(file_path).exists():
                current_hash = self._compute_file_hash(file_path)
                cached_hash = self.file_hashes.get(file_path)
                
                if current_hash != cached_hash:
                    # File content actually changed
                    logger.debug(f"File content changed: {file_path}")
                    self.file_hashes[file_path] = current_hash
                    await self.callback(file_path)
                else:
                    logger.debug(f"File timestamp changed but content unchanged: {file_path}")
            else:
                # File was deleted
                logger.debug(f"File deleted: {file_path}")
                self.file_hashes.pop(file_path, None)
                await self.callback(file_path)
            
        except asyncio.CancelledError:
            logger.debug(f"Debounced callback cancelled for {file_path}")
        except Exception as e:
            logger.error(f"Error in debounced callback for {file_path}: {e}")
        finally:
            # Clean up task reference
            self.debounce_tasks.pop(file_path, None)
    
    def _schedule_file_processing(self, file_path: str):
        """Schedule debounced processing for a file"""
        if not self._matches_pattern(file_path):
            return
        
        # Update last change time
        self.pending_files[file_path] = time.time()
        
        # Cancel existing task if any
        existing_task = self.debounce_tasks.get(file_path)
        if existing_task and not existing_task.done():
            existing_task.cancel()
        
        # Schedule new debounced task (handle thread-safe creation)
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._debounced_file_callback(file_path))
            self.debounce_tasks[file_path] = task
        except RuntimeError:
            # No event loop running, schedule task differently
            logger.debug(f"No event loop available, scheduling task via call_soon_threadsafe for {file_path}")
            # Store the file path for later processing
            if not hasattr(self, '_loop'):
                logger.warning("No event loop reference available for threadsafe scheduling")
                return
            
            def schedule_task():
                task = self._loop.create_task(self._debounced_file_callback(file_path))
                self.debounce_tasks[file_path] = task
            
            self._loop.call_soon_threadsafe(schedule_task)
        
        logger.debug(f"Scheduled debounced processing for {file_path}")
    
    async def start(self):
        """Start the file watcher"""
        if self.is_running:
            logger.warning("File watcher is already running")
            return
        
        try:
            # Store reference to current event loop for thread-safe operations
            self._loop = asyncio.get_running_loop()
            
            self.event_handler = DebouncedEventHandler(self._schedule_file_processing)
            self.observer = Observer()
            self.observer.schedule(self.event_handler, str(self.watch_path), recursive=True)
            self.observer.start()
            
            self.is_running = True
            logger.info(f"Started watching {self.watch_path}")
            
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            raise
    
    async def stop(self):
        """Stop the file watcher and cleanup"""
        if not self.is_running:
            return
        
        try:
            # Stop observer
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None
            
            # Cancel all pending debounce tasks
            for task in self.debounce_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.debounce_tasks:
                await asyncio.gather(*self.debounce_tasks.values(), return_exceptions=True)
            
            self.debounce_tasks.clear()
            self.pending_files.clear()
            self.is_running = False
            
            logger.info("File watcher stopped")
            
        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}")


class DebouncedEventHandler(FileSystemEventHandler):
    """File system event handler with debouncing"""
    
    def __init__(self, schedule_callback: Callable[[str], None]):
        """
        Initialize event handler
        
        Args:
            schedule_callback: Function to call to schedule file processing
        """
        super().__init__()
        self.schedule_callback = schedule_callback
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            self.schedule_callback(event.src_path)
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            self.schedule_callback(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events"""
        if not event.is_directory:
            self.schedule_callback(event.src_path)
    
    def on_moved(self, event):
        """Handle file move/rename events"""
        if not event.is_directory:
            # Handle as deletion of old path and creation of new path
            self.schedule_callback(event.src_path)
            self.schedule_callback(event.dest_path)


# Compatibility alias for the existing file_watcher.py
class NeuralIndexerEnhanced:
    """Enhanced wrapper around existing NeuralIndexer with selective reprocessing"""
    
    def __init__(self, neural_indexer):
        """Initialize with existing NeuralIndexer instance"""
        self.neural_indexer = neural_indexer
        self.file_watcher: Optional[EnhancedFileWatcher] = None
    
    async def start_enhanced_watching(self, debounce_delay: float = 0.5):
        """Start enhanced file watching with selective reprocessing"""
        
        async def enhanced_callback(file_path: str):
            """Enhanced callback that uses selective reprocessing"""
            try:
                if Path(file_path).exists():
                    # File exists, index it (selective reprocessing built into index_file)
                    await self.neural_indexer.index_file(file_path)
                else:
                    # File was deleted, remove from index
                    await self.neural_indexer.remove_file_from_index(file_path)
            except Exception as e:
                logger.error(f"Enhanced callback error for {file_path}: {e}")
        
        self.file_watcher = EnhancedFileWatcher(
            watch_path=str(self.neural_indexer.project_path),
            callback=enhanced_callback,
            debounce_delay=debounce_delay,
            patterns=['*.py', '*.js', '*.ts', '*.md', '*.txt', '*.json', '*.yaml']
        )
        
        await self.file_watcher.start()
        logger.info("Enhanced file watching started")
    
    async def stop_enhanced_watching(self):
        """Stop enhanced file watching"""
        if self.file_watcher:
            await self.file_watcher.stop()
            self.file_watcher = None
            logger.info("Enhanced file watching stopped")


if __name__ == "__main__":
    # Example usage
    async def example_callback(file_path: str):
        print(f"File changed: {file_path}")
    
    async def main():
        watcher = EnhancedFileWatcher(
            watch_path=".",
            callback=example_callback,
            debounce_delay=0.5,
            patterns=['*.py', '*.txt']
        )
        
        try:
            await watcher.start()
            print("Watching for changes... (Ctrl+C to stop)")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            await watcher.stop()
    
    asyncio.run(main())