#!/usr/bin/env python3
"""
Production-ready neural indexer entry point with signal handling and health endpoint
"""
import signal
import asyncio
import logging
import sys
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional
import uvicorn
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response


# Configure structured logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s","file":"%(filename)s","line":"%(lineno)d"}',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("indexer-entrypoint")

# Prometheus metrics
files_processed = Counter('indexer_files_processed_total', 'Total files processed')
chunks_created = Counter('indexer_chunks_created_total', 'Total chunks created')
processing_duration = Histogram('indexer_processing_duration_seconds', 'File processing time (s)', 
                               buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float('inf')))
queue_depth = Gauge('indexer_queue_depth', 'Current queue depth')
degraded_mode = Gauge('indexer_degraded_mode', 'Degraded mode indicator (1=degraded)')
health_status = Gauge('indexer_health_status', 'Health status (1=healthy, 0=unhealthy)')
error_count = Counter('indexer_errors_total', 'Total errors encountered', ['error_type'])
service_health = Gauge('indexer_service_health', 'Service health status', ['service'])
indexer_uptime = Counter('indexer_uptime_seconds', 'Indexer uptime in seconds')

# Track uptime
import time
_start_time = time.time()

def update_uptime():
    indexer_uptime._value._value = time.time() - _start_time

# Health endpoint
app = FastAPI(title="Neural Indexer Sidecar")

@app.get('/health')
async def health():
    """Fast health check for container orchestration"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get('/metrics')
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get('/status')
async def status():
    """Detailed status for operational monitoring"""
    base_status = {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "queue_depth": 0,
        "files_processed": 0,
        "degraded_mode": False
    }
    
    # Get real metrics from indexer if available
    if hasattr(app.state, 'indexer_runner') and app.state.indexer_runner.indexer:
        try:
            indexer_metrics = app.state.indexer_runner.indexer.get_metrics()
            base_status.update({
                "queue_depth": indexer_metrics.get('queue_depth', 0),
                "files_processed": indexer_metrics.get('files_indexed', 0),
                "degraded_mode": len(indexer_metrics.get('degraded_services', [])) > 0,
                "degraded_services": indexer_metrics.get('degraded_services', []),
                "healthy_services": indexer_metrics.get('healthy_services', []),
                "chunks_created": indexer_metrics.get('chunks_created', 0),
                "errors": indexer_metrics.get('errors', 0)
            })
            
            # Update Prometheus metrics
            queue_depth.set(base_status['queue_depth'])
            files_processed._value._value = base_status['files_processed']
            chunks_created._value._value = base_status['chunks_created']
            degraded_mode.set(1 if base_status['degraded_mode'] else 0)
            
            # Update service health metrics
            for service in ['neo4j', 'qdrant', 'nomic']:
                is_healthy = service in base_status.get('healthy_services', [])
                service_health.labels(service=service).set(1 if is_healthy else 0)
            
            # Update uptime
            update_uptime()
            
        except Exception as e:
            logger.warning(f"Error getting indexer metrics: {e}")
    
    return base_status

@app.post('/reindex-path')
async def reindex_path(path: str):
    """Trigger reindexing of a specific path"""
    if not path:
        return {"error": "Path parameter is required"}
    
    # This will be populated by the indexer service when runner is set up
    # For now, return a placeholder response
    if hasattr(app.state, 'indexer_runner') and app.state.indexer_runner.indexer:
        try:
            # Queue the path for reindexing
            await app.state.indexer_runner.indexer._queue_change(path, 'update')
            return {
                "status": "success",
                "message": f"Reindex request queued for: {path}",
                "enqueued": path
            }
        except Exception as e:
            logger.error(f"Error queuing reindex request: {e}")
            return {"error": f"Failed to queue reindex: {str(e)}"}
    else:
        return {"error": "Indexer service not available"}


class IndexerRunner:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.indexer = None
        self.health_server = None
        self.health_task = None
        
    async def handle_shutdown(self, sig):
        """Handle graceful shutdown signals"""
        logger.info(f"Received signal {sig}, starting graceful shutdown...")
        
        # Set health status to unhealthy
        health_status.set(0)
        
        # Stop the indexer service gracefully
        if self.indexer and hasattr(self.indexer, 'shutdown'):
            logger.info("Shutting down indexer service...")
            try:
                await self.indexer.shutdown()
                logger.info("Indexer service shutdown complete")
            except Exception as e:
                logger.error(f"Error during indexer shutdown: {e}")
        
        # Stop health server
        if self.health_server:
            logger.info("Shutting down health server...")
            self.health_server.should_exit = True
            if self.health_task:
                self.health_task.cancel()
        
        self.shutdown_event.set()
        logger.info("Graceful shutdown complete")
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"Signal {sig} received, scheduling shutdown...")
            asyncio.create_task(self.handle_shutdown(sig))
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
    async def start_health_server(self):
        """Start health/metrics server in background"""
        config = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=8080, 
            log_level="warning",  # Reduce health server noise
            access_log=False
        )
        self.health_server = uvicorn.Server(config)
        
        # Run health server in background task
        self.health_task = asyncio.create_task(self.health_server.serve())
        logger.info("Health server started on port 8080 (external: 48080)")
        
    async def run_indexer(self, project_path: str, project_name: str, initial_index: bool):
        """Run the neural indexer service"""
        try:
            # Import and start indexer service
            sys.path.insert(0, '/app/src')
            from servers.services.indexer_service import IncrementalIndexer
            from servers.services.service_container import ServiceContainer
            
            logger.info(f"Starting neural indexer for project: {project_name} at {project_path}")

            # Initialize service container with real services - MUST pass project name!
            container = ServiceContainer(project_name)

            # Create and configure indexer with proper parameters
            self.indexer = IncrementalIndexer(
                project_path=project_path,  # Pass as string, not Path
                project_name=project_name,
                container=container
            )
            
            # Initialize async queue for docker sidecar runner
            if self.indexer.pending_queue is None:
                self.indexer.pending_queue = asyncio.Queue(maxsize=self.indexer.max_queue_size)
                logger.info(f"Initialized pending queue in docker runner with max size {self.indexer.max_queue_size}")
            
            # Set health status to healthy
            health_status.set(1)
            
            # Initialize services if needed
            if hasattr(self.indexer, 'initialize_services'):
                logger.info("Initializing indexer services...")
                await self.indexer.initialize_services()
            
            # Perform initial indexing if requested
            if initial_index:
                logger.info("Performing initial project indexing...")
                if hasattr(self.indexer, 'initial_index'):
                    await self.indexer.initial_index()
                elif hasattr(self.indexer, 'index_project'):
                    await self.indexer.index_project()
                else:
                    logger.warning("No initial indexing method found on indexer")
                logger.info("Initial indexing complete")
            
            # Start continuous monitoring
            logger.info("Starting continuous file monitoring...")
            if hasattr(self.indexer, 'start_monitoring'):
                await self.indexer.start_monitoring()
            elif hasattr(self.indexer, 'start_watching'):
                await self.indexer.start_watching()
            else:
                # Start basic file monitoring manually
                from watchdog.observers import Observer
                from servers.services.indexer_service import DebouncedEventHandler
                
                observer = Observer()
                event_handler = DebouncedEventHandler(self.indexer)
                observer.schedule(event_handler, project_path, recursive=True)
                observer.start()
                self.indexer.observer = observer
                
                logger.info("File monitoring started with watchdog observer")
                
                # Keep the service running
                try:
                    while not self.shutdown_event.is_set():
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Monitoring task cancelled")
                finally:
                    observer.stop()
                    observer.join()
            
        except Exception as e:
            logger.error(f"Error in indexer service: {e}", exc_info=True)
            health_status.set(0)
            degraded_mode.set(1)
            raise
            
    async def run(self):
        """Main entry point for the indexer container."""
        logger.info("Starting Neural Indexer Sidecar Container")
        
        # ADR-0037: Environment Variable Configuration Priority
        # 1. PRIORITY: Environment Variables (explicit configuration)
        project_name = os.getenv("PROJECT_NAME")
        project_path_env = os.getenv("PROJECT_PATH")
        
        if project_name and project_path_env:
            project_path = Path(project_path_env)
            if not project_path.exists() or not project_path.is_dir():
                logger.error(f"FATAL: PROJECT_PATH does not exist or is not a directory: {project_path_env}")
                sys.exit(1)
            logger.info(f"✅ [ADR-0037] Using explicit configuration: PROJECT_NAME='{project_name}', PROJECT_PATH='{project_path_env}'")
        else:
            # 2. FALLBACK: Auto-detection (only if env vars missing)  
            logger.warning("⚠️ [ADR-0037] Environment variables missing. Falling back to auto-detection.")
            logger.warning("⚠️ [ADR-0037] For production, set PROJECT_NAME and PROJECT_PATH explicitly.")
            
            workspace_path = Path("/workspace")
            
            # Look for project markers instead of directory ordering
            if (workspace_path / ".git").exists() or (workspace_path / "pyproject.toml").exists() or (workspace_path / "package.json").exists():
                # Use workspace root if it has project markers
                project_path = workspace_path
                project_name = project_name or "default"
                logger.info(f"🔍 [ADR-0037] Auto-detected project root with markers: {project_path}")
            else:
                # Legacy fallback: Find subdirectories (improved logic)
                project_dirs = [d for d in workspace_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
                
                if not project_dirs:
                    logger.error("FATAL: No project directory found inside /workspace.")
                    logger.error("Set PROJECT_NAME and PROJECT_PATH explicitly, or mount project with markers (.git, pyproject.toml, etc.)")
                    sys.exit(1)
                
                if len(project_dirs) > 1:
                    logger.warning(f"Multiple directories found in /workspace: {[d.name for d in project_dirs]}.")
                    logger.warning(f"Using the first one found: {project_dirs[0].name}")
                    logger.warning("For reliable behavior, set PROJECT_NAME and PROJECT_PATH environment variables.")
                
                project_path = project_dirs[0]
                project_name = project_name or project_path.name
            
            logger.info(f"🔍 [ADR-0037] Auto-detected: PROJECT_NAME='{project_name}', PROJECT_PATH='{project_path}'")
        
        initial_index = os.getenv('INITIAL_INDEX', 'true').lower() == 'true'
        
        logger.info(f"Configuration: path={project_path}, project={project_name}, initial_index={initial_index}")
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Start health server
        await self.start_health_server()
        
        # Make this runner accessible to FastAPI endpoints
        app.state.indexer_runner = self
        
        try:
            # Run indexer service with the detected project
            await self.run_indexer(str(project_path), project_name, initial_index)
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Fatal error in indexer runner: {e}", exc_info=True)
            health_status.set(0)
            sys.exit(1)
        
        logger.info("Neural Indexer Sidecar Container stopped")


async def main():
    """Async main entry point"""
    runner = IndexerRunner()
    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, exiting...")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)