#!/usr/bin/env python3
"""
L9 Multi-Project Automatic Incremental Indexer Service
Monitors multiple projects under /workspace/project-*/ and maintains isolated indices
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Set
from watchdog.observers import Observer
from threading import Thread
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.indexer_service import AutomaticIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class MultiProjectIndexerManager:
    """Manages multiple project indexers with automatic project discovery"""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.project_indexers: Dict[str, AutomaticIndexer] = {}
        self.observers: Dict[str, Observer] = {}
        self.project_threads: Dict[str, Thread] = {}
        self.running = False
        
    def discover_projects(self) -> Set[str]:
        """Discover project directories under workspace"""
        projects = set()
        
        if not self.workspace_root.exists():
            logger.warning(f"Workspace root does not exist: {self.workspace_root}")
            return projects
            
        # Look for project-* directories and direct subdirectories
        for item in self.workspace_root.iterdir():
            if item.is_dir():
                # Skip hidden directories and common non-project directories
                if item.name.startswith('.') or item.name in ['tmp', 'cache', 'logs']:
                    continue
                    
                project_name = item.name
                projects.add(project_name)
                logger.info(f"Discovered project: {project_name}")
                
        return projects
    
    async def start_project_indexer(self, project_name: str):
        """Start indexer for a specific project"""
        project_path = self.workspace_root / project_name
        
        if not project_path.exists():
            logger.error(f"Project path does not exist: {project_path}")
            return
            
        try:
            logger.info(f"🚀 Starting indexer for project: {project_name}")
            
            # Create project-specific indexer
            indexer = AutomaticIndexer(project_name=project_name)
            
            # Initialize services
            await indexer.initialize_services()
            
            # Store indexer
            self.project_indexers[project_name] = indexer
            
            # Start file monitoring in a thread
            def run_project_indexer():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(indexer.run(
                        project_path=str(project_path),
                        initial_index=True
                    ))
                except Exception as e:
                    logger.error(f"Project {project_name} indexer failed: {e}")
                finally:
                    loop.close()
            
            thread = Thread(target=run_project_indexer, daemon=True)
            thread.start()
            self.project_threads[project_name] = thread
            
            logger.info(f"✅ Project indexer started: {project_name}")
            
        except Exception as e:
            logger.error(f"Failed to start indexer for {project_name}: {e}")
    
    def stop_project_indexer(self, project_name: str):
        """Stop indexer for a specific project"""
        if project_name in self.project_indexers:
            try:
                indexer = self.project_indexers[project_name]
                # The indexer runs in its own thread, so we just remove it
                del self.project_indexers[project_name]
                
                if project_name in self.project_threads:
                    # Note: Thread will end when indexer completes
                    del self.project_threads[project_name]
                    
                logger.info(f"🛑 Stopped indexer for project: {project_name}")
                
            except Exception as e:
                logger.error(f"Failed to stop indexer for {project_name}: {e}")
    
    async def start_all_projects(self):
        """Discover and start indexers for all projects"""
        self.running = True
        projects = self.discover_projects()
        
        if not projects:
            logger.warning("No projects discovered. Monitoring for new projects...")
            # Still create a default project if workspace has files
            if any(self.workspace_root.glob('*.py')):
                logger.info("Found Python files in workspace root, creating 'default' project")
                projects.add('default')
        
        # Start indexers for discovered projects
        for project_name in projects:
            await self.start_project_indexer(project_name)
        
        # Monitor for new project directories
        await self.monitor_workspace()
    
    async def monitor_workspace(self):
        """Monitor workspace for new/removed projects"""
        known_projects = set(self.project_indexers.keys())
        
        while self.running:
            try:
                current_projects = self.discover_projects()
                
                # Start indexers for new projects
                new_projects = current_projects - known_projects
                for project in new_projects:
                    await self.start_project_indexer(project)
                
                # Stop indexers for removed projects
                removed_projects = known_projects - current_projects
                for project in removed_projects:
                    self.stop_project_indexer(project)
                
                known_projects = current_projects
                
            except Exception as e:
                logger.error(f"Error monitoring workspace: {e}")
            
            # Check every 30 seconds for new/removed projects
            await asyncio.sleep(30)
    
    def stop_all(self):
        """Stop all project indexers"""
        self.running = False
        for project_name in list(self.project_indexers.keys()):
            self.stop_project_indexer(project_name)

async def run_multi_project_indexer(workspace_root: str = "/workspace"):
    """Main entry point for multi-project indexer"""
    manager = MultiProjectIndexerManager(workspace_root)
    
    try:
        logger.info("🌟 Starting L9 Multi-Project Indexer Manager")
        await manager.start_all_projects()
        
    except KeyboardInterrupt:
        logger.info("Shutting down multi-project indexer...")
        manager.stop_all()
    except Exception as e:
        logger.error(f"Multi-project indexer failed: {e}")
        manager.stop_all()
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L9 Multi-Project Indexer Manager")
    parser.add_argument("--workspace-root", default="/workspace", 
                       help="Root directory containing project subdirectories")
    
    args = parser.parse_args()
    
    asyncio.run(run_multi_project_indexer(args.workspace_root))