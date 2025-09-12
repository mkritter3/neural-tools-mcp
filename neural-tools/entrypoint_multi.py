#!/usr/bin/env python3
"""
Multi-Project Neural Indexer Entry Point
Supports both single-project (backward compatible) and multi-project modes
"""
import signal
import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s","file":"%(filename)s","line":"%(lineno)d"}',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("multi-indexer-entrypoint")

# Add source to path
sys.path.insert(0, '/app/src')

from servers.services.multi_project_indexer import MultiProjectIndexer
from servers.services.indexer_service import IncrementalIndexer
from servers.services.service_container import ServiceContainer


class IndexerRunner:
    """Manages the indexer lifecycle"""
    
    def __init__(self):
        self.indexer = None
        self.container = None
        self.mode = os.getenv('INDEXER_MODE', 'single')  # single or multi-project
        self.shutdown_event = asyncio.Event()
        
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Starting graceful shutdown...")
        
        if self.indexer:
            try:
                await self.indexer.stop()
            except Exception as e:
                logger.error(f"Error during indexer shutdown: {e}")
        
        self.shutdown_event.set()
        logger.info("Graceful shutdown complete")
    
    async def run(self):
        """Main entry point"""
        logger.info(f"Starting Neural Indexer in {self.mode} mode...")
        
        # Parse command line arguments
        args = sys.argv[1:]
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        try:
            # Initialize shared container (without async neo4j for now)
            self.container = ServiceContainer()
            # Skip async initialization that causes event loop issues
            # await self.container.initialize()
            
            if self.mode == 'multi-project':
                await self.run_multi_project(args)
            else:
                await self.run_single_project(args)
            
            # Keep running until shutdown
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Fatal error in indexer: {e}", exc_info=True)
            sys.exit(1)
    
    async def run_multi_project(self, args):
        """Run in multi-project mode"""
        logger.info("Starting Multi-Project Indexer...")
        
        # Parse base workspace path
        base_workspace = args[0] if args else os.getenv('BASE_WORKSPACE', '/workspace')
        
        # Create multi-project indexer
        self.indexer = MultiProjectIndexer(
            base_path=base_workspace,
            container=self.container
        )
        
        # Check for specific project arguments
        specific_projects = []
        i = 0
        while i < len(args):
            if args[i] == '--project' and i + 2 < len(args):
                project_name = args[i + 1]
                project_path = args[i + 2]
                specific_projects.append((project_name, project_path))
                i += 3
            else:
                i += 1
        
        # Start indexer
        await self.indexer.start()
        
        # Add specific projects if provided
        for project_name, project_path in specific_projects:
            await self.indexer.add_project(project_name, project_path)
        
        logger.info("Multi-Project Indexer started successfully")
    
    async def run_single_project(self, args):
        """Run in single-project mode (backward compatible)"""
        logger.info("Starting Single-Project Indexer...")
        
        # Parse arguments (backward compatible)
        project_path = args[0] if args else "/workspace"
        project_name = os.getenv('PROJECT_NAME', 'default')
        
        # Parse additional arguments
        i = 1
        while i < len(args):
            if args[i] == "--project-name" and i + 1 < len(args):
                project_name = args[i + 1]
                i += 2
            else:
                i += 1
        
        logger.info(f"Configuration: path={project_path}, project={project_name}")
        
        # Create single project indexer
        from servers.services.indexer_service import IncrementalIndexer
        
        self.indexer = IncrementalIndexer(
            project_path=project_path,
            project_name=project_name,
            container=self.container
        )
        
        # Initial index if requested
        if os.getenv('INITIAL_INDEX', 'true').lower() == 'true':
            logger.info("Starting initial index...")
            await self.indexer.initial_index()
        
        # Start watching
        await self.indexer.start_watching()
        
        logger.info("Single-Project Indexer started successfully")


async def main():
    """Main entry point"""
    runner = IndexerRunner()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())