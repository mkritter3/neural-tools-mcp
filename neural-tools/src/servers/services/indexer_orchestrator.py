#!/usr/bin/env python3
"""
IndexerOrchestrator - Multi-Container Indexer Lifecycle Management
Implements ADR-0030 for per-project indexer container orchestration

This orchestrator manages one indexer container per project, ensuring:
- Complete project isolation (filesystem, process, resource)
- Automatic lifecycle management (spawn, idle shutdown, cleanup)
- Resource limits and concurrency control
- Security hardening (read-only mounts, non-root, path validation)

Author: L9 Engineering Team
Date: 2025-09-12
"""

import asyncio
import docker
import os
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class IndexerOrchestrator:
    """
    Dedicated orchestrator for indexer container lifecycle management
    Implements ADR-0030 multi-container architecture
    Updated with ADR-0044 for dependency injection and container discovery
    """

    def __init__(self, context_manager=None, max_concurrent: int = 8):
        """
        Initialize the orchestrator with dependency injection (ADR-0044)

        Args:
            context_manager: Injected ProjectContextManager instance
            max_concurrent: Maximum number of concurrent indexer containers
        """
        # ADR-0044: Accept injected context manager
        self.context_manager = context_manager
        self.docker_client = None
        self.discovery_service = None  # Will be initialized with docker client
        self.active_indexers: Dict[str, dict] = {}  # project -> {container_id, last_activity, port}
        self.max_concurrent = max_concurrent
        
        # Port allocation for indexer HTTP APIs
        self.port_base = 48100  # Start from 48100 for project indexers
        self.allocated_ports = set()  # Track allocated ports
        
        # Resource limits per ADR-0030 and expert recommendations
        self.resource_limits = {
            'mem_limit': '512m',
            'cpu_quota': 50000,  # 0.5 CPU
            'cpu_period': 100000
        }
        
        # Idle timeout - containers stop after 1 hour of inactivity
        self.idle_timeout = timedelta(hours=1)
        self._cleanup_task = None
        self._lock = asyncio.Lock()  # Thread safety for container operations
        
        # Security: Whitelist of allowed mount paths (can be configured)
        self.allowed_mount_prefixes = [
            '/Users',
            '/home',
            '/var/workspace',
            '/opt/projects'
        ]
        
    async def initialize(self):
        """
        Initialize Docker client and start background tasks
        """
        try:
            self.docker_client = docker.from_env()
            # Verify Docker is accessible
            self.docker_client.ping()

            # ADR-0044: Initialize container discovery service
            from servers.services.container_discovery import ContainerDiscoveryService
            self.discovery_service = ContainerDiscoveryService(self.docker_client)

            logger.info("IndexerOrchestrator initialized with container discovery")

            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._idle_cleanup_loop())

        except docker.errors.DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def _allocate_port(self) -> int:
        """Allocate a unique port for an indexer container"""
        port = self.port_base
        while port in self.allocated_ports:
            port += 1
            if port > self.port_base + 100:  # Max 100 indexers
                raise ValueError("No available ports for indexer")
        self.allocated_ports.add(port)
        return port
    
    def _release_port(self, port: int):
        """Release a port back to the pool"""
        self.allocated_ports.discard(port)
    
    def get_indexer_port(self, project_name: str) -> Optional[int]:
        """Get the port for a project's indexer if running"""
        if project_name in self.active_indexers:
            return self.active_indexers[project_name].get('port')
        return None
            
    async def ensure_indexer(self, project_name: str, project_path: str) -> str:
        """
        Ensure indexer is running for project, with resource limits
        Updated with ADR-0044 to use ContainerDiscoveryService

        Args:
            project_name: Name of the project (used for container naming)
            project_path: Absolute path to project directory

        Returns:
            Container ID of the running indexer

        Raises:
            ValueError: If project path is invalid or insecure
            docker.errors.DockerException: If container operations fail
        """
        async with self._lock:
            # ADR-0044: Use discovery service to find or create container
            if self.discovery_service and self.context_manager:
                logger.info(f"🔍 Using discovery service for project: {project_name}")

                # Update context manager's current project path if needed
                if not self.context_manager.current_project_path:
                    self.context_manager.current_project_path = Path(project_path)

                # Use discovery service to get or create container
                container_info = await self.discovery_service.get_or_create_container(
                    project_name,
                    self.context_manager
                )

                # Update active indexers tracking
                self.active_indexers[project_name] = {
                    'container_id': container_info['container_id'],
                    'last_activity': datetime.now(),
                    'project_path': project_path,
                    'port': container_info['port']
                }

                logger.info(f"✅ Container ready: {container_info['container_name']} on port {container_info['port']}")
                return container_info['container_id']

            # Fallback to original logic if no discovery service
            logger.warning("⚠️ No discovery service, using legacy container creation")

            # ADR-0048: Idempotent container management - remove any existing indexer first
            try:
                existing_containers = self.docker_client.containers.list(
                    all=True,  # Include stopped containers
                    filters={'name': f'indexer-{project_name}'}
                )
                for container in existing_containers:
                    logger.info(f"[ADR-0048] Removing existing indexer for idempotency: {container.id[:12]}")
                    try:
                        # Graceful stop first if running
                        if container.status == 'running':
                            container.stop(timeout=10)
                    except:
                        pass  # Container might already be stopped
                    container.remove(force=True)
                    # Clean up tracking if present
                    if project_name in self.active_indexers:
                        port = self.active_indexers[project_name].get('port')
                        if port:
                            self._release_port(port)
                        del self.active_indexers[project_name]
            except docker.errors.NotFound:
                pass  # No existing container
            except Exception as e:
                logger.warning(f"[ADR-0048] Error during cleanup: {e}")

            # Check if already running (this should now always be false due to cleanup above)
            if project_name in self.active_indexers:
                self.active_indexers[project_name]['last_activity'] = datetime.now()
                container_id = self.active_indexers[project_name]['container_id']

                # Verify container is still running
                try:
                    container = self.docker_client.containers.get(container_id)
                    if container.status == 'running':
                        logger.debug(f"Indexer for {project_name} already running: {container_id[:12]}")
                        return container_id
                except docker.errors.NotFound:
                    logger.warning(f"Container {container_id[:12]} not found, will recreate")
                    del self.active_indexers[project_name]
                    
            # Check concurrency limit
            if len(self.active_indexers) >= self.max_concurrent:
                await self._stop_least_recently_used()
            
            # Validate path security (prevent traversal attacks)
            project_path = os.path.abspath(project_path)
            if not os.path.exists(project_path):
                raise ValueError(f"Project path does not exist: {project_path}")
                
            if not os.path.isdir(project_path):
                raise ValueError(f"Project path is not a directory: {project_path}")
                
            # Security: Validate against whitelist
            if not any(project_path.startswith(prefix) for prefix in self.allowed_mount_prefixes):
                logger.warning(f"Project path outside whitelist: {project_path}")
                # For now, log warning but allow (can be made stricter)
                
            # Spawn container with security hardening
            try:
                # Allocate a unique port for this indexer
                indexer_port = self._allocate_port()
                logger.info(f"Starting indexer container for project: {project_name} on port {indexer_port}")
                
                container = self.docker_client.containers.run(
                    image='l9-neural-indexer:production',
                    name=f'indexer-{project_name}',
                    environment={
                        'PROJECT_NAME': project_name,
                        'PROJECT_PATH': '/workspace',  # ADR-0048: ALWAYS use container path, never host path
                        # Use host.docker.internal for container->host communication
                        'NEO4J_URI': 'bolt://host.docker.internal:47687',
                        'NEO4J_USERNAME': 'neo4j',
                        'NEO4J_PASSWORD': 'graphrag-password',
                        'QDRANT_HOST': 'host.docker.internal',
                        'QDRANT_PORT': '46333',
                        'REDIS_CACHE_HOST': 'host.docker.internal',
                        'REDIS_CACHE_PORT': '46379',
                        'REDIS_QUEUE_HOST': 'host.docker.internal',
                        'REDIS_QUEUE_PORT': '46380',
                        'EMBEDDING_SERVICE_HOST': 'host.docker.internal',
                        'EMBEDDING_SERVICE_PORT': '48000',
                        # Performance tuning
                        'BATCH_SIZE': '10',
                        'DEBOUNCE_INTERVAL': '2.0',
                        'MAX_QUEUE_SIZE': '1000',
                        'EMBED_DIM': '768',
                        'STRUCTURE_EXTRACTION_ENABLED': 'true',
                        'INITIAL_INDEX': 'true',
                        'LOG_LEVEL': 'INFO',
                        # File watching optimization for Docker
                        'WATCHDOG_FORCE_POLLING': '1'
                    },
                    volumes={
                        project_path: {'bind': '/workspace', 'mode': 'ro'}  # Read-only mount
                    },
                    ports={
                        '8080/tcp': indexer_port  # Map container port 8080 to allocated host port
                    },
                    network='l9-graphrag-network',
                    detach=True,
                    auto_remove=True,  # Clean up on stop
                    **self.resource_limits,
                    security_opt=['no-new-privileges'],
                    user='1000:1000'  # Non-root user
                )
                
                self.active_indexers[project_name] = {
                    'container_id': container.id,
                    'last_activity': datetime.now(),
                    'project_path': project_path,
                    'port': indexer_port
                }
                
                logger.info(f"Indexer container started for {project_name}: {container.id[:12]} on port {indexer_port}")
                return container.id
                
            except docker.errors.ImageNotFound:
                logger.error("Indexer image 'l9-neural-indexer:production' not found. Please build it first.")
                raise
            except docker.errors.APIError as e:
                logger.error(f"Docker API error while starting container: {e}")
                raise
                
    async def _idle_cleanup_loop(self):
        """
        Background task to stop idle indexers
        Runs every 5 minutes to check for containers idle > 1 hour
        """
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                async with self._lock:
                    now = datetime.now()
                    to_remove = []
                    
                    for project, info in self.active_indexers.items():
                        if now - info['last_activity'] > self.idle_timeout:
                            logger.info(f"Indexer for {project} idle for > 1 hour, stopping")
                            to_remove.append(project)
                            
                    for project in to_remove:
                        await self._stop_indexer_internal(project)
                        
            except Exception as e:
                logger.error(f"Error in idle cleanup loop: {e}")
                # Continue running despite errors
                
    async def stop_indexer(self, project_name: str):
        """
        Gracefully stop an indexer
        
        Args:
            project_name: Name of the project whose indexer to stop
        """
        async with self._lock:
            await self._stop_indexer_internal(project_name)
            
    async def _stop_indexer_internal(self, project_name: str):
        """
        Internal method to stop indexer (assumes lock is held)
        """
        if project_name in self.active_indexers:
            try:
                container_id = self.active_indexers[project_name]['container_id']
                port = self.active_indexers[project_name].get('port')
                container = self.docker_client.containers.get(container_id)
                
                logger.info(f"Stopping indexer container for {project_name}: {container_id[:12]}")
                container.stop(timeout=10)
                
                # Release the allocated port
                if port:
                    self._release_port(port)
                    logger.debug(f"Released port {port} from {project_name}")
                
            except docker.errors.NotFound:
                logger.warning(f"Container already stopped for {project_name}")
            except Exception as e:
                logger.error(f"Error stopping container for {project_name}: {e}")
            finally:
                # Release port if not already done
                if project_name in self.active_indexers:
                    port = self.active_indexers[project_name].get('port')
                    if port:
                        self._release_port(port)
                del self.active_indexers[project_name]
                
    async def _stop_least_recently_used(self):
        """
        Stop the least recently used indexer to make room for a new one
        """
        if not self.active_indexers:
            return
            
        # Find LRU project
        lru_project = min(
            self.active_indexers.items(),
            key=lambda x: x[1]['last_activity']
        )[0]
        
        logger.info(f"Stopping LRU indexer: {lru_project} to make room")
        await self._stop_indexer_internal(lru_project)
        
    async def stop_all(self):
        """
        Stop all running indexers (useful for shutdown)
        """
        async with self._lock:
            projects = list(self.active_indexers.keys())
            for project in projects:
                await self._stop_indexer_internal(project)
                
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current orchestrator status
        
        Returns:
            Status dictionary with active indexers and resource usage
        """
        status = {
            'active_indexers': len(self.active_indexers),
            'max_concurrent': self.max_concurrent,
            'indexers': {}
        }
        
        for project, info in self.active_indexers.items():
            try:
                container = self.docker_client.containers.get(info['container_id'])
                stats = container.stats(stream=False)
                
                status['indexers'][project] = {
                    'container_id': info['container_id'][:12],
                    'status': container.status,
                    'last_activity': info['last_activity'].isoformat(),
                    'idle_minutes': (datetime.now() - info['last_activity']).total_seconds() / 60,
                    'project_path': info['project_path'],
                    'memory_usage_mb': stats.get('memory_stats', {}).get('usage', 0) / (1024 * 1024),
                    'cpu_percent': self._calculate_cpu_percent(stats)
                }
            except docker.errors.NotFound:
                status['indexers'][project] = {
                    'container_id': info['container_id'][:12],
                    'status': 'not_found',
                    'last_activity': info['last_activity'].isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting status for {project}: {e}")
                
        return status
        
    def _calculate_cpu_percent(self, stats: dict) -> float:
        """
        Calculate CPU percentage from Docker stats
        
        Args:
            stats: Docker container stats
            
        Returns:
            CPU usage percentage
        """
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
                return round(cpu_percent, 2)
        except (KeyError, ZeroDivisionError):
            pass
            
        return 0.0
        
    async def cleanup(self):
        """
        Cleanup method for graceful shutdown
        """
        logger.info("IndexerOrchestrator cleanup initiated")
        
        # Cancel background task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            
        # Stop all indexers
        await self.stop_all()
        
        # Close Docker client
        if self.docker_client:
            self.docker_client.close()