#!/usr/bin/env python3
"""
Debounced File Watcher for Neural Search
Monitors project directories for changes and triggers re-indexing
Based on Gemini's recommended patterns with proper debouncing
"""

import asyncio
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Callable, Set, Optional, List, Any
from collections import defaultdict
import json

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class SelectiveReprocessingCache:
    """
    Cache for tracking file hashes and chunks to enable selective reprocessing
    Only reprocesses chunks that have actually changed
    """
    
    def __init__(self):
        self.file_hashes: Dict[str, str] = {}
        self.file_chunks: Dict[str, List[Dict[str, Any]]] = {}
        self.lock = threading.Lock()
    
    def compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute MD5 hash of file contents"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return None
    
    def has_file_changed(self, file_path: str) -> bool:
        """Check if file has changed since last processing"""
        current_hash = self.compute_file_hash(file_path)
        if not current_hash:
            return False
        
        with self.lock:
            previous_hash = self.file_hashes.get(file_path)
            return current_hash != previous_hash
    
    def update_file_hash(self, file_path: str, chunks: List[Dict[str, Any]]) -> None:
        """Update file hash and chunks cache"""
        current_hash = self.compute_file_hash(file_path)
        if current_hash:
            with self.lock:
                self.file_hashes[file_path] = current_hash
                self.file_chunks[file_path] = chunks
    
    def get_cached_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """Get previously cached chunks for a file"""
        with self.lock:
            return self.file_chunks.get(file_path, [])
    
    def remove_file(self, file_path: str) -> None:
        """Remove file from cache when deleted"""
        with self.lock:
            self.file_hashes.pop(file_path, None)
            self.file_chunks.pop(file_path, None)
    
    def diff_chunks(self, old_chunks: List[Dict], new_chunks: List[Dict]) -> List[int]:
        """
        Compare chunks and return indices of changed chunks
        Uses content hash comparison for efficiency
        """
        changed_indices = []
        
        # Create hash maps for quick lookup
        old_chunk_hashes = {}
        for i, chunk in enumerate(old_chunks):
            content = chunk.get('content', '')
            chunk_hash = hashlib.md5(content.encode()).hexdigest()
            old_chunk_hashes[i] = chunk_hash
        
        # Compare new chunks
        for i, new_chunk in enumerate(new_chunks):
            content = new_chunk.get('content', '')
            new_hash = hashlib.md5(content.encode()).hexdigest()
            
            # If chunk index exists in old chunks, compare hashes
            if i < len(old_chunks):
                old_hash = old_chunk_hashes.get(i, '')
                if new_hash != old_hash:
                    changed_indices.append(i)
            else:
                # New chunk added
                changed_indices.append(i)
        
        # If old chunks were longer, mark them as changed (deletions)
        if len(old_chunks) > len(new_chunks):
            for i in range(len(new_chunks), len(old_chunks)):
                changed_indices.append(i)
        
        return changed_indices

class DebouncedEventHandler(FileSystemEventHandler):
    """
    File system event handler with debouncing to prevent thrashing
    Batches rapid file changes and only triggers action after quiet period
    """
    
    def __init__(self, callback: Callable[[str, str, Dict[str, Any]], None], debounce_interval: float = 1.5):
        """
        Initialize debounced event handler with selective reprocessing
        
        Args:
            callback: Function to call with (event_type, file_path, metadata) after debounce
            debounce_interval: Seconds to wait for quiet period before triggering
        """
        super().__init__()
        self.callback = callback
        self.debounce_interval = debounce_interval
        self.timers: Dict[str, threading.Timer] = {}
        self.lock = threading.Lock()
        self.cache = SelectiveReprocessingCache()
        
        # File extensions to monitor
        self.monitored_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java', '.cpp', '.c', '.h'}
        
        # Directories to ignore
        self.ignore_dirs = {
            '__pycache__', '.venv', 'venv', 'node_modules', '.git', 
            '.pytest_cache', '.mypy_cache', 'build', 'dist', 'target'
        }
        
        logger.info(f"Initialized file watcher with {debounce_interval}s debounce interval")
    
    def should_monitor_file(self, file_path: str) -> bool:
        """Check if file should be monitored based on extension and location"""
        path = Path(file_path)
        
        # Check extension
        if path.suffix not in self.monitored_extensions:
            return False
        
        # Check if in ignored directory
        for part in path.parts:
            if part in self.ignore_dirs:
                return False
        
        return True
    
    def dispatch(self, event):
        """Handle filesystem events with debouncing"""
        # Skip directory events
        if event.is_directory:
            return
        
        # Handle different event types
        if event.event_type == 'moved':
            # For moved files, handle both source deletion and destination creation
            if hasattr(event, 'src_path') and self.should_monitor_file(event.src_path):
                self._schedule_callback('deleted', event.src_path)
            
            if hasattr(event, 'dest_path') and self.should_monitor_file(event.dest_path):
                self._schedule_callback('created', event.dest_path)
                
        elif event.event_type in ('created', 'modified', 'deleted'):
            file_path = event.src_path
            
            if self.should_monitor_file(file_path):
                self._schedule_callback(event.event_type, file_path)
    
    def _schedule_callback(self, event_type: str, file_path: str):
        """Schedule callback with debouncing"""
        with self.lock:
            # Cancel existing timer for this file
            if file_path in self.timers:
                self.timers[file_path].cancel()
            
            # For deletions, trigger immediately without debounce
            if event_type == 'deleted':
                self._fire_callback(event_type, file_path)
                return
            
            # Schedule new timer
            timer = threading.Timer(
                self.debounce_interval,
                self._fire_callback,
                args=[event_type, file_path]
            )
            timer.start()
            self.timers[file_path] = timer
            
            logger.debug(f"Scheduled {event_type} event for {file_path}")
    
    def _fire_callback(self, event_type: str, file_path: str):
        """Fire the callback with selective reprocessing metadata"""
        with self.lock:
            self.timers.pop(file_path, None)
        
        logger.info(f"File {event_type}: {file_path}")
        
        try:
            # Prepare metadata for selective reprocessing
            metadata = {
                'selective_reprocessing': True,
                'requires_full_reindex': False,
                'changed_chunks': [],
                'file_changed': True
            }
            
            if event_type == 'deleted':
                # For deletions, always remove from cache
                self.cache.remove_file(file_path)
                metadata['requires_full_reindex'] = False  # Just deletion
                
            elif event_type in ['created', 'modified']:
                # Check if file actually changed
                if not self.cache.has_file_changed(file_path):
                    logger.debug(f"File {file_path} hash unchanged, skipping reprocessing")
                    metadata['file_changed'] = False
                    return  # Skip callback for unchanged files
                
                # For existing files with changes, determine what changed
                if event_type == 'modified':
                    old_chunks = self.cache.get_cached_chunks(file_path)
                    if old_chunks:
                        # This would normally require chunking the file to compare
                        # For now, mark as needing selective update
                        metadata['requires_selective_update'] = True
                        metadata['has_cached_chunks'] = True
                        logger.info(f"File {file_path} modified - will use selective reprocessing")
                    else:
                        # No cached chunks, treat as new file
                        metadata['requires_full_reindex'] = True
                        logger.info(f"File {file_path} modified but no cache - full reindex needed")
                else:  # created
                    metadata['requires_full_reindex'] = True
                    logger.info(f"File {file_path} created - full indexing needed")
            
            self.callback(event_type, file_path, metadata)
            
        except Exception as e:
            logger.error(f"Callback failed for {file_path}: {e}")

class ProjectWatcher:
    """
    Manages file watching for multiple projects
    Coordinates with API server for re-indexing
    """
    
    def __init__(self, reindex_callback: Callable[[str, str, Dict[str, Any]], None]):
        """
        Initialize project watcher with selective reprocessing support
        
        Args:
            reindex_callback: Function to call with (project_name, file_path, metadata) for re-indexing
        """
        self.reindex_callback = reindex_callback
        self.observers: Dict[str, Observer] = {}
        self.project_paths: Dict[str, str] = {}  # project_name -> project_path
        self.lock = threading.Lock()
        
        logger.info("Initialized project watcher")
    
    def start_watching_project(self, project_name: str, project_path: str) -> bool:
        """
        Start watching a project directory
        
        Args:
            project_name: Unique project identifier
            project_path: Absolute path to project directory
            
        Returns:
            True if watching started successfully
        """
        try:
            project_path = str(Path(project_path).resolve())
            
            if not Path(project_path).exists():
                logger.error(f"Project path does not exist: {project_path}")
                return False
            
            with self.lock:
                # Stop existing watcher if any
                if project_name in self.observers:
                    self.stop_watching_project(project_name)
                
                # Create event handler with selective reprocessing support
                def handle_file_change(event_type: str, file_path: str, metadata: Dict[str, Any]):
                    self.reindex_callback(project_name, file_path, metadata)
                
                handler = DebouncedEventHandler(handle_file_change)
                
                # Create and start observer
                observer = Observer()
                observer.schedule(handler, project_path, recursive=True)
                observer.start()
                
                # Store references
                self.observers[project_name] = observer
                self.project_paths[project_name] = project_path
                
                logger.info(f"Started watching project '{project_name}' at {project_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start watching project {project_name}: {e}")
            return False
    
    def stop_watching_project(self, project_name: str) -> bool:
        """
        Stop watching a project directory
        
        Args:
            project_name: Project identifier
            
        Returns:
            True if stopped successfully
        """
        try:
            with self.lock:
                observer = self.observers.pop(project_name, None)
                self.project_paths.pop(project_name, None)
                
                if observer:
                    observer.stop()
                    observer.join(timeout=5)  # Wait up to 5 seconds
                    logger.info(f"Stopped watching project '{project_name}'")
                    return True
                else:
                    logger.warning(f"Project '{project_name}' was not being watched")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to stop watching project {project_name}: {e}")
            return False
    
    def stop_all(self):
        """Stop all project watchers"""
        with self.lock:
            project_names = list(self.observers.keys())
            
        for project_name in project_names:
            self.stop_watching_project(project_name)
            
        logger.info("Stopped all project watchers")
    
    def get_watched_projects(self) -> Dict[str, str]:
        """Get dict of currently watched projects {name: path}"""
        with self.lock:
            return self.project_paths.copy()
    
    def shutdown(self):
        """Stop all watchers and cleanup"""
        logger.info("Shutting down project watcher...")
        
        with self.lock:
            project_names = list(self.observers.keys())
        
        for project_name in project_names:
            self.stop_watching_project(project_name)
        
        logger.info("Project watcher shutdown complete")

# Async wrapper for integration with FastAPI
class AsyncProjectWatcher:
    """
    Async wrapper around ProjectWatcher for FastAPI integration
    Manages file watching in background thread
    """
    
    def __init__(self):
        self.watcher: Optional[ProjectWatcher] = None
        self.watcher_thread: Optional[threading.Thread] = None
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.pending_reindex_jobs: Dict[str, float] = {}  # file_path -> timestamp
        
    def initialize(self, reindex_callback: Callable[[str, str, Dict[str, Any]], None]):
        """Initialize the watcher with reindex callback supporting selective reprocessing"""
        self.event_loop = asyncio.get_event_loop()
        
        # Wrapper to handle async callback with metadata
        def async_reindex_wrapper(project_name: str, file_path: str, metadata: Dict[str, Any]):
            if self.event_loop:
                # Schedule callback in the event loop
                asyncio.create_task(
                    self._handle_reindex_async(reindex_callback, project_name, file_path, metadata)
                )
        
        self.watcher = ProjectWatcher(async_reindex_wrapper)
        logger.info("Async project watcher initialized")
    
    async def _handle_reindex_async(self, callback, project_name: str, file_path: str, metadata: Dict[str, Any]):
        """Handle reindex callback asynchronously with selective reprocessing metadata"""
        try:
            await callback(project_name, file_path, metadata)
        except Exception as e:
            logger.error(f"Async reindex callback failed: {e}")
    
    async def start_watching_project(self, project_name: str, project_path: str) -> bool:
        """Start watching a project (async interface)"""
        if not self.watcher:
            logger.error("Watcher not initialized")
            return False
        
        # Run the synchronous method in a thread to avoid blocking the event loop
        return await asyncio.to_thread(
            self.watcher.start_watching_project, project_name, project_path
        )
    
    async def stop_watching_project(self, project_name: str) -> bool:
        """Stop watching a project (async interface)"""
        if not self.watcher:
            return False
        
        # Run the synchronous method in a thread to avoid blocking the event loop
        return await asyncio.to_thread(
            self.watcher.stop_watching_project, project_name
        )
    
    async def get_watched_projects(self) -> Dict[str, str]:
        """Get watched projects (async interface)"""
        if not self.watcher:
            return {}
        
        return self.watcher.get_watched_projects()
    
    async def stop_all(self):
        """Stop all project watchers (async interface)"""
        if not self.watcher:
            return
            
        # Run the synchronous method in a thread to avoid blocking the event loop
        await asyncio.to_thread(self.watcher.stop_all)
    
    def shutdown(self):
        """Shutdown the watcher"""
        if self.watcher:
            self.watcher.shutdown()

# Example usage and testing
if __name__ == "__main__":
    import tempfile
    
    def test_watcher():
        """Test the file watcher with a temporary directory"""
        print("🔍 Testing file watcher...")
        
        def mock_callback(event_type: str, file_path: str, metadata: Dict[str, Any]):
            file_name = Path(file_path).name
            selective = "SELECTIVE" if metadata.get('requires_selective_update') else "FULL"
            changed = "CHANGED" if metadata.get('file_changed', True) else "UNCHANGED"
            print(f"   📝 {event_type.upper()}: {file_name} [{selective}, {changed}]")
            
            if not metadata.get('file_changed', True):
                print(f"      ⏭️  Skipping unchanged file")
            elif metadata.get('requires_selective_update'):
                print(f"      🎯 Using selective reprocessing")
            elif metadata.get('requires_full_reindex'):
                print(f"      🔄 Full reindexing required")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create handler
            handler = DebouncedEventHandler(mock_callback, debounce_interval=1.0)
            
            # Create observer
            observer = Observer()
            observer.schedule(handler, str(tmp_path), recursive=True)
            observer.start()
            
            try:
                print(f"   👀 Watching: {tmp_path}")
                
                # Create test file
                test_file = tmp_path / "test.py"
                test_file.write_text("print('hello')")
                print("   ✏️  Created test.py")
                
                time.sleep(2)  # Wait for debounce
                
                # Modify file multiple times quickly
                for i in range(3):
                    test_file.write_text(f"print('hello {i}')")
                    time.sleep(0.1)
                
                print("   ✏️  Modified test.py 3 times quickly")
                time.sleep(2)  # Wait for debounce
                
                # Delete file
                test_file.unlink()
                print("   🗑️  Deleted test.py")
                
                time.sleep(1)
                
            finally:
                observer.stop()
                observer.join()
        
        print("✅ File watcher test completed")
    
    test_watcher()