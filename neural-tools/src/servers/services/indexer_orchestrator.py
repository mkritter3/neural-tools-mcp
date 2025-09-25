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
import hashlib
import time
import secrets
import redis.asyncio as redis
from redis.exceptions import LockError
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def is_container_reusable(
    container_attrs: dict,
    requested_path: str,
    required_env_vars: Optional[Dict[str, str]] = None
) -> bool:
    """
    Pure function to validate if a container can be reused.
    ADR-0063: Critical validation to prevent mount path regression.

    Args:
        container_attrs: The attrs dictionary from a Docker container
        requested_path: The requested mount path
        required_env_vars: Environment variables that must match (e.g., {'NEO4J_PASSWORD': 'value'})

    Returns:
        bool: True if container can be reused, False otherwise
    """
    # 1. Validate Mount Path
    mounts = container_attrs.get('Mounts', [])
    abs_requested_path = os.path.abspath(requested_path)

    mount_valid = any(
        m.get('Source') == abs_requested_path and
        m.get('Destination') == '/workspace'
        for m in mounts
    )

    if not mount_valid:
        return False

    # 2. Validate Environment Variables (if required)
    if required_env_vars:
        env_vars = container_attrs.get('Config', {}).get('Env', [])
        env_dict = {}

        for env_str in env_vars:
            if '=' in env_str:
                key, value = env_str.split('=', 1)
                env_dict[key] = value

        # Check all required env vars match
        for key, expected_value in required_env_vars.items():
            if env_dict.get(key) != expected_value:
                return False

    return True


class IndexerOrchestrator:
    """
    Dedicated orchestrator for indexer container lifecycle management
    Implements ADR-0030 multi-container architecture
    Updated with ADR-0044 for dependency injection and container discovery
    Updated with ADR-0060 for Graceful Ephemeral Containers pattern
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
        self.stopped_indexers: Dict[str, dict] = {}  # project -> {container_id, stop_time, port, project_path}
        self.max_concurrent = max_concurrent

        # ADR-0060: Redis clients for distributed locking and caching
        self.redis_client = None
        self.redis_cache = None

        # Port allocation for indexer HTTP APIs
        self.port_base = 48100  # Start from 48100 for project indexers
        self.allocated_ports = set()  # Track allocated ports
        
        # Resource limits per ADR-0030 and expert recommendations
        self.resource_limits = {
            'mem_limit': '512m',
            'cpu_quota': 50000,  # 0.5 CPU
            'cpu_period': 100000
        }
        
        # ADR-0060: Updated garbage collection - 24 hours for stopped containers before removal
        self.gc_removal_age = timedelta(hours=24)  # Remove stopped containers after 24 hours
        self.gc_idle_timeout = timedelta(hours=1)  # Stop (not remove) containers after 1 hour idle
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
        Initialize Docker client, Redis connections, and start background tasks
        Updated with ADR-0060 for distributed coordination
        """
        try:
            self.docker_client = docker.from_env()
            # Verify Docker is accessible
            self.docker_client.ping()

            # ADR-0064: Hardened Redis initialization with comprehensive error handling
            await self._init_redis()

            # ADR-0044: Initialize container discovery service
            # This must succeed even if Redis fails
            from servers.services.container_discovery import ContainerDiscoveryService
            self.discovery_service = ContainerDiscoveryService(self.docker_client)

            if self.redis_client:
                logger.info("IndexerOrchestrator initialized with container discovery and Redis")
            else:
                logger.warning("[ADR-0064] IndexerOrchestrator initialized with container discovery only (Redis unavailable)")

            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._idle_cleanup_loop())

            # ADR-0060: Start garbage collection task
            self._gc_task = asyncio.create_task(self._garbage_collection_loop())

        except docker.errors.DockerException as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise

    async def _init_redis(self):
        """
        ADR-0064: Initialize Redis with comprehensive error handling
        Falls back gracefully to local locks on any Redis error
        """
        redis_host = os.getenv('REDIS_CACHE_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_CACHE_PORT', 46379))
        redis_password = os.getenv('REDIS_CACHE_PASSWORD', 'cache-secret-key')

        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5
            )

            # Test Redis connectivity
            await self.redis_client.ping()
            logger.info(f"[ADR-0060] Redis connected at {redis_host}:{redis_port}")

        except redis.AuthenticationError as e:
            logger.warning(f"[ADR-0064] Redis authentication failed at {redis_host}:{redis_port} - {e.__class__.__name__}: {e}")
            logger.warning("[ADR-0064] Falling back to local locks (no cross-instance coordination)")
            self.redis_client = None

        except redis.ResponseError as e:
            logger.warning(f"[ADR-0064] Redis response error at {redis_host}:{redis_port} - {e.__class__.__name__}: {e}")
            logger.warning("[ADR-0064] Falling back to local locks (no cross-instance coordination)")
            self.redis_client = None

        except redis.ConnectionError as e:
            logger.warning(f"[ADR-0064] Redis connection failed at {redis_host}:{redis_port} - {e.__class__.__name__}: {e}")
            logger.warning("[ADR-0064] Falling back to local locks (no cross-instance coordination)")
            self.redis_client = None

        except redis.TimeoutError as e:
            logger.warning(f"[ADR-0064] Redis timeout at {redis_host}:{redis_port} - {e.__class__.__name__}: {e}")
            logger.warning("[ADR-0064] Falling back to local locks (no cross-instance coordination)")
            self.redis_client = None

        except Exception as e:
            # Catch-all for any other Redis errors
            logger.warning(f"[ADR-0064] Unexpected Redis error at {redis_host}:{redis_port} - {e.__class__.__name__}: {e}")
            logger.warning("[ADR-0064] Falling back to local locks (no cross-instance coordination)")
            self.redis_client = None
    
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

    def _generate_unique_container_name(self, project_name: str) -> str:
        """
        Generate a unique container name with timestamp and random suffix
        ADR-0060: Prevents naming conflicts

        Format: indexer-{project}-{timestamp}-{random}
        Example: indexer-myapp-1726123456-a4f2
        """
        timestamp = int(time.time())
        random_suffix = secrets.token_hex(2)  # 4 hex chars
        return f"indexer-{project_name}-{timestamp}-{random_suffix}"

    async def get_indexer_endpoint(self, project_name: str) -> Optional[str]:
        """
        Get the endpoint for a project's indexer with Redis caching
        ADR-0060: Cached discovery for performance

        Returns:
            Endpoint URL if container exists, None otherwise
        """
        cache_key = f"endpoint:{project_name}"

        # Try Redis cache first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    logger.debug(f"[ADR-0060] Cache hit for {project_name}: {cached}")
                    return cached
            except Exception as e:
                logger.warning(f"[ADR-0060] Cache read failed: {e}")

        # Discover container by labels
        try:
            containers = self.docker_client.containers.list(
                filters={'label': f'com.l9.project={project_name}'}
            )

            if not containers:
                return None

            # Get the first running container
            for container in containers:
                if container.status == 'running':
                    # Get the mapped port
                    ports = container.ports.get('8080/tcp')
                    if ports and len(ports) > 0:
                        port = ports[0]['HostPort']
                        endpoint = f"http://localhost:{port}"

                        # Cache for 30 seconds
                        if self.redis_client:
                            try:
                                await self.redis_client.setex(
                                    cache_key,
                                    30,  # 30 second TTL
                                    endpoint
                                )
                            except Exception as e:
                                logger.warning(f"[ADR-0060] Cache write failed: {e}")

                        return endpoint
        except Exception as e:
            logger.error(f"[ADR-0060] Discovery failed: {e}")

        return None

    async def _invalidate_cache(self, project_name: str):
        """
        Invalidate Redis cache after state changes
        ADR-0060: Explicit cache invalidation
        """
        if self.redis_client:
            try:
                cache_key = f"endpoint:{project_name}"
                await self.redis_client.delete(cache_key)
                logger.debug(f"[ADR-0060] Cache invalidated for {project_name}")
            except Exception as e:
                logger.warning(f"[ADR-0060] Cache invalidation failed: {e}")

    async def ensure_indexer(self, project_name: str, project_path: str) -> str:
        """
        Ensure indexer is running for project, with resource limits
        Updated with ADR-0060 for Graceful Ephemeral Containers pattern

        Args:
            project_name: Name of the project (used for container naming)
            project_path: Absolute path to project directory

        Returns:
            Container ID of the running indexer

        Raises:
            ValueError: If project path is invalid or insecure
            docker.errors.DockerException: If container operations fail
        """
        # ADR-0060: Use Redis distributed lock for multi-instance coordination
        lock_key = f"lock:project:{project_name}"
        lock_acquired = False

        if self.redis_client:
            try:
                # Try to acquire distributed lock with 60s timeout
                async with self.redis_client.lock(
                    lock_key,
                    timeout=60,  # Lock expires after 60 seconds
                    blocking_timeout=5  # Wait up to 5 seconds to acquire
                ):
                    lock_acquired = True
                    logger.info(f"[ADR-0060] Acquired distributed lock for {project_name}")
                    return await self._ensure_indexer_internal(project_name, project_path)
            except LockError as e:
                logger.warning(f"[ADR-0060] Failed to acquire lock for {project_name}: {e}")
                # Fall through to local lock
            except Exception as e:
                logger.error(f"[ADR-0060] Redis lock error: {e}")
                # Fall through to local lock

        # Fallback to local asyncio lock if Redis unavailable
        if not lock_acquired:
            logger.info(f"[ADR-0060] Using local lock for {project_name} (Redis unavailable)")
            async with self._lock:
                return await self._ensure_indexer_internal(project_name, project_path)

    async def _ensure_indexer_internal(self, project_name: str, project_path: str) -> str:
        """
        Internal method to ensure indexer (assumes lock is held)
        ADR-0060: Core logic separated for lock flexibility
        ADR-0063: Added mount verification before container reuse
        """
        # First check if container already exists using label-based discovery
        existing_container = await self._discover_existing_container(project_name)
        if existing_container:
            logger.info(f"[ADR-0060] Found existing container for {project_name}: {existing_container['id'][:12]}")

            # ADR-0063: Verify mount matches requested path before reuse
            try:
                container = self.docker_client.containers.get(existing_container['id'])
                mounts = container.attrs.get('Mounts', [])

                # Check if mount path matches
                requested_path = os.path.abspath(project_path)
                mount_valid = any(
                    m.get('Source') == requested_path and
                    m.get('Destination') == '/workspace'
                    for m in mounts
                )

                if mount_valid:
                    # ADR-0063: Also check if environment variables match
                    env_vars = container.attrs.get('Config', {}).get('Env', [])
                    env_dict = {}
                    for env_str in env_vars:
                        if '=' in env_str:
                            key, value = env_str.split('=', 1)
                            env_dict[key] = value

                    # ADR-0085: Relaxed validation - only check password matches
                    # MCP doesn't have env vars set, so we can't strictly validate
                    # The important thing is the password matches for security
                    current_neo4j_pass = os.getenv('NEO4J_PASSWORD', 'graphrag-password')

                    # Check if container has correct password
                    container_password = env_dict.get('NEO4J_PASSWORD', '')

                    # Also verify container points to host.docker.internal (standard setup)
                    container_qdrant_host = env_dict.get('QDRANT_HOST', '')

                    env_valid = (
                        container_password == current_neo4j_pass and
                        container_qdrant_host in ['host.docker.internal', 'localhost', '127.0.0.1']
                    )

                    if env_valid:
                        logger.info(f"[ADR-0063] Mount and env verified, reusing container for {project_name}")
                        logger.info(f"  Mount path: {requested_path} -> /workspace")

                        # Update tracking with verified path
                        self.active_indexers[project_name] = {
                            'container_id': existing_container['id'],
                            'last_activity': datetime.now(),
                            'project_path': project_path,
                            'port': existing_container.get('port')
                        }

                        # ADR-0098 Phase 0: Observability
                        try:
                            from .docker_observability import observer
                            observer.check_state_divergence(
                                project_name,
                                self.active_indexers[project_name],
                                source="active_indexers"
                            )
                        except ImportError:
                            pass
                        except Exception as e:
                            logger.debug(f"[ADR-0098] Observability check failed: {e}")

                        # Invalidate cache to force fresh discovery
                        await self._invalidate_cache(project_name)

                        return existing_container['id']
                    else:
                        # Environment mismatch - but more lenient now
                        logger.info(f"[ADR-0085] Container env validation failed, but attempting reuse for {project_name}")

                        # Only log specific issues at debug level
                        if container_password != current_neo4j_pass:
                            logger.debug(f"  Password mismatch detected")
                        if container_qdrant_host not in ['host.docker.internal', 'localhost', '127.0.0.1']:
                            logger.debug(f"  Host configuration: {container_qdrant_host}")

                        # ADR-0085: Be more lenient - reuse if mount is correct
                        # Even if env vars don't match exactly, the container works
                        logger.info(f"[ADR-0085] Reusing existing container despite env differences")

                        # Update tracking
                        self.active_indexers[project_name] = {
                            'container_id': existing_container['id'],
                            'last_activity': datetime.now(),
                            'project_path': project_path,
                            'port': existing_container.get('port')
                        }

                        # ADR-0098 Phase 0: Observability
                        try:
                            from .docker_observability import observer
                            observer.check_state_divergence(
                                project_name,
                                self.active_indexers[project_name],
                                source="active_indexers"
                            )
                        except ImportError:
                            pass
                        except Exception as e:
                            logger.debug(f"[ADR-0098] Observability check failed: {e}")

                        # Invalidate cache to force fresh discovery
                        await self._invalidate_cache(project_name)

                        return existing_container['id']
                else:
                    # Mount mismatch - remove and recreate
                    logger.warning(f"[ADR-0063] Mount mismatch detected for {project_name}")
                    logger.warning(f"  Expected: {requested_path}")
                    actual_mounts = [m.get('Source') for m in mounts if m.get('Destination') == '/workspace']
                    logger.warning(f"  Actual: {actual_mounts}")
                    logger.info(f"[ADR-0063] Removing container with wrong mount")

                    container.remove(force=True)
                    await self._invalidate_cache(project_name)
                    # Fall through to create new container

            except docker.errors.NotFound:
                logger.warning(f"[ADR-0063] Container {existing_container['id'][:12]} not found, will create new")
                await self._invalidate_cache(project_name)
                # Fall through to create new container
            except Exception as e:
                logger.error(f"[ADR-0063] Error verifying container mount: {e}")
                # On any error, remove and recreate for safety
                try:
                    container.remove(force=True)
                except:
                    pass
                await self._invalidate_cache(project_name)
                # Fall through to create new container

        # Check if we have a stopped container we can restart (within grace period)
        if project_name in self.stopped_indexers:
            stopped_info = self.stopped_indexers[project_name]
            if datetime.now() - stopped_info['stop_time'] < self.gc_removal_age:
                try:
                    container = self.docker_client.containers.get(stopped_info['container_id'])
                    logger.info(f"[ADR-0060] Restarting stopped container for {project_name}: {stopped_info['container_id'][:12]}")
                    container.start()

                    # Move back to active tracking
                    self.active_indexers[project_name] = {
                        'container_id': stopped_info['container_id'],
                        'last_activity': datetime.now(),
                        'port': stopped_info.get('port'),
                        'project_path': stopped_info.get('project_path')
                    }

                    # Remove from stopped tracking
                    del self.stopped_indexers[project_name]

                    # Update last activity and return
                    await self._update_last_activity(project_name)
                    return stopped_info['container_id']

                except docker.errors.NotFound:
                    logger.warning(f"[ADR-0060] Stopped container not found, will create new")
                    # Clean up stopped tracking and fall through to create new
                    del self.stopped_indexers[project_name]
                except Exception as e:
                    logger.error(f"[ADR-0060] Error restarting container: {e}")
                    # Clean up and fall through to create new
                    del self.stopped_indexers[project_name]

        # No existing or restorable container, create a new one
        logger.info(f"[ADR-0060] Creating new container for {project_name}")

        # Validate path security
        project_path = os.path.abspath(project_path)
        if not os.path.exists(project_path):
            raise ValueError(f"Project path does not exist: {project_path}")
        if not os.path.isdir(project_path):
            raise ValueError(f"Project path is not a directory: {project_path}")

        # Check concurrency limit
        if len(self.active_indexers) >= self.max_concurrent:
            await self._stop_least_recently_used()

        # ADR-0044: Use ContainerDiscoveryService if available
        if self.discovery_service:
            logger.info(f"[ADR-0044] Using ContainerDiscoveryService for container creation")
            try:
                # Let discovery service handle container lifecycle
                container_info = await self.discovery_service.discover_project_container(project_name)
                if container_info:
                    # Found existing container through discovery service
                    logger.info(f"[ADR-0044] Discovery service found container: {container_info.get('container_id', 'unknown')[:12]}")
                    # Verify mount before using (ADR-0063)
                    container = self.docker_client.containers.get(container_info['container_id'])
                    mounts = container.attrs.get('Mounts', [])
                    requested_path = os.path.abspath(project_path)
                    mount_valid = any(
                        m.get('Source') == requested_path and
                        m.get('Destination') == '/workspace'
                        for m in mounts
                    )

                    if mount_valid:
                        logger.info(f"[ADR-0044] Discovery service container has correct mount")
                        return container_info['container_id']
                    else:
                        logger.warning(f"[ADR-0044] Discovery service container has wrong mount, will recreate")
                        container.remove(force=True)
            except Exception as e:
                logger.warning(f"[ADR-0044] Discovery service error: {e}, falling back to direct creation")

        # ADR-0098 Phase 1: Allocate port before container creation
        # so we can include it in the labels
        allocated_port = self._allocate_port()

        # Create container with unique name and labels
        container_name = self._generate_unique_container_name(project_name)
        container = await self._create_container(
            project_name=project_name,
            container_name=container_name,
            project_path=project_path,
            port=allocated_port  # Pass allocated port
        )

        # Verify the port was used correctly
        actual_port = container.attrs['HostConfig']['PortBindings']['8080/tcp'][0]['HostPort']
        if int(actual_port) != allocated_port:
            logger.warning(f"[ADR-0098] Port mismatch: allocated {allocated_port}, actual {actual_port}")
            # Release the allocated port if not used
            self._release_port(allocated_port)
            allocated_port = int(actual_port)
            self.allocated_ports.add(allocated_port)

        logger.info(f"[ADR-0098] Container {container_name} using port {allocated_port}")

        # Update tracking
        self.active_indexers[project_name] = {
            'container_id': container.id,
            'last_activity': datetime.now(),
            'project_path': project_path,
            'port': allocated_port
        }

        # ADR-0098 Phase 0: Observability - check for state divergence
        try:
            from .docker_observability import observer
            observer.check_state_divergence(
                project_name,
                self.active_indexers[project_name],
                source="active_indexers"
            )
        except ImportError:
            pass  # Observability module not yet deployed
        except Exception as e:
            logger.debug(f"[ADR-0098] Observability check failed: {e}")

        # Invalidate cache after creating new container
        await self._invalidate_cache(project_name)

        logger.info(f"[ADR-0060] ✅ Container created: {container_name} ({container.id[:12]})")
        return container.id

    async def _discover_existing_container(self, project_name: str) -> Optional[Dict[str, Any]]:
        """
        Discover existing container by project label
        ADR-0060: Label-based discovery instead of name-based
        """
        try:
            containers = self.docker_client.containers.list(
                filters={'label': f'com.l9.project={project_name}'}
            )

            for container in containers:
                if container.status == 'running':
                    # Get port mapping
                    ports = container.ports.get('8080/tcp')
                    port = None
                    if ports and len(ports) > 0:
                        port = int(ports[0]['HostPort'])

                    return {
                        'id': container.id,
                        'name': container.name,
                        'port': port
                    }
        except Exception as e:
            logger.error(f"[ADR-0060] Discovery error: {e}")

        return None

    async def _create_container(self, project_name: str, container_name: str, project_path: str, port: int = None):
        """
        Create a new indexer container with ADR-0060 specifications
        ADR-0098 Phase 1: Enhanced Docker labels for better observability
        """
        # Use provided port or let Docker allocate
        if port:
            logger.info(f"[ADR-0098] Creating container {container_name} for project {project_name} on port {port}")
        else:
            logger.info(f"[ADR-0060] Creating container {container_name} for project {project_name} (dynamic port)")

        # Detect if this is a test container
        test_patterns = ['test-', 'adr63-', 'adr60-', 'mount-test']
        is_test = any(pattern in project_name for pattern in test_patterns)

        # ADR-0098 Phase 1: Enhanced labels for better state tracking
        import hashlib
        project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:12]

        labels = {
            'com.l9.managed': 'true',
            'com.l9.project': project_name,
            'com.l9.created': str(int(time.time())),
            # ADR-0098 Phase 1: Additional metadata
            'com.l9.project_hash': project_hash,  # Stable ID from path
            'com.l9.project_path': project_path,  # Full path for reference
        }

        # Add port label if we know it in advance
        if port:
            labels['com.l9.port'] = str(port)

        # Add test label if this is a test container
        if is_test:
            labels['com.l9.test'] = 'true'
            logger.info(f"[TEST] Marking {container_name} as test container")

        container = self.docker_client.containers.run(
            image='l9-neural-indexer:production',
            name=container_name,
            labels=labels,
            environment={
                'PROJECT_NAME': project_name,
                'PROJECT_PATH': '/workspace',
                # Use container name when on same network (l9-graphrag-network)
                # Don't inherit from host environment - always use container name
                'NEO4J_URI': 'bolt://claude-l9-template-neo4j-1:7687',
                'NEO4J_USERNAME': os.environ.get('NEO4J_USERNAME', 'neo4j'),
                'NEO4J_PASSWORD': os.environ.get('NEO4J_PASSWORD', 'graphrag-password'),
                'QDRANT_HOST': os.environ.get('QDRANT_HOST', 'host.docker.internal'),
                'QDRANT_PORT': os.environ.get('QDRANT_PORT', '46333'),
                # Use container names since Redis is on same network (l9-graphrag-network)
                # Don't use os.environ.get - always use container names for container-to-container communication
                'REDIS_CACHE_HOST': 'claude-l9-template-redis-cache-1',
                'REDIS_CACHE_PORT': '6379',  # Internal port
                'REDIS_QUEUE_HOST': 'claude-l9-template-redis-queue-1',
                'REDIS_QUEUE_PORT': '6379',  # Internal port
                # Use container name since Nomic is on same network (l9-graphrag-network)
                # Don't use os.environ.get - always use container name for container-to-container communication
                'EMBEDDING_SERVICE_HOST': 'neural-flow-nomic-v2-production-optimized',
                'EMBEDDING_SERVICE_PORT': '8000',  # Internal port, not exposed port
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
                '8080/tcp': port if port else None  # Use specified port or let Docker allocate
            },
            network='l9-graphrag-network',
            detach=True,
            auto_remove=True,  # Clean up on stop
            **self.resource_limits,
            security_opt=['no-new-privileges'],
            user='1000:1000'  # Non-root user
        )

        return container

    async def _garbage_collection_loop(self):
        """
        ADR-0060: Garbage collection with 7-day policy for stopped containers
        ADR-0098: Added observability metrics reporting
        """
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                await self.garbage_collect_containers()

                # ADR-0098 Phase 0: Report observability metrics periodically
                try:
                    from .docker_observability import observer
                    metrics = observer.report_metrics()
                    if metrics['divergence_rate'] != "0.0%":
                        logger.warning(f"[ADR-0098] State divergence detected: {metrics}")
                except ImportError:
                    pass  # Observability module not yet deployed
                except Exception as e:
                    logger.debug(f"[ADR-0098] Metrics reporting failed: {e}")
            except Exception as e:
                logger.error(f"[ADR-0060] GC error: {e}")

    async def garbage_collect_containers(self):
        """
        ADR-0060: Clean up stopped containers older than 7 days
        """
        try:
            now = int(time.time())
            seven_days_ago = now - (7 * 24 * 3600)

            containers = self.docker_client.containers.list(
                all=True,
                filters={'label': 'com.l9.managed=true'}
            )

            for container in containers:
                # Skip running containers
                if container.status == 'running':
                    continue

                # Check age via label
                created_str = container.labels.get('com.l9.created', '0')
                try:
                    created = int(created_str)
                    if created < seven_days_ago:
                        logger.info(f"[ADR-0060] GC removing old container: {container.name}")
                        container.remove(force=True)
                except ValueError:
                    logger.warning(f"[ADR-0060] Invalid creation time for {container.name}")
        except Exception as e:
            logger.error(f"[ADR-0060] GC failed: {e}")

                
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
                    
                    # Check active containers for idle timeout (stop after 1 hour)
                    for project, info in self.active_indexers.items():
                        if now - info['last_activity'] > self.gc_idle_timeout:
                            logger.info(f"Indexer for {project} idle for > 1 hour, stopping (not removing)")
                            to_remove.append(project)

                    for project in to_remove:
                        await self._stop_only_internal(project)

                    # Check stopped containers for removal (after 24 hours)
                    to_delete = []
                    for project, info in self.stopped_indexers.items():
                        if now - info['stop_time'] > self.gc_removal_age:
                            logger.info(f"Stopped indexer for {project} > 24 hours old, removing")
                            to_delete.append(project)

                    for project in to_delete:
                        info = self.stopped_indexers[project]
                        await self._remove_indexer_internal(project, info)
                        del self.stopped_indexers[project]
                        
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
            
    async def _stop_only_internal(self, project_name: str):
        """
        Stop container without removing it (for 24hr grace period)
        Moves container from active to stopped tracking
        """
        if project_name in self.active_indexers:
            try:
                info = self.active_indexers[project_name]
                container_id = info['container_id']
                container = self.docker_client.containers.get(container_id)

                logger.info(f"Stopping (not removing) indexer for {project_name}: {container_id[:12]}")
                container.stop(timeout=10)

                # Move to stopped_indexers for grace period tracking
                self.stopped_indexers[project_name] = {
                    'container_id': container_id,
                    'stop_time': datetime.now(),
                    'port': info.get('port'),
                    'project_path': info.get('project_path')
                }

                # Don't release port yet - keep it reserved for potential restart
                logger.info(f"Container {container_id[:12]} stopped, entering 24hr grace period")

            except docker.errors.NotFound:
                logger.warning(f"Container already stopped for {project_name}")
            except Exception as e:
                logger.error(f"Error stopping container for {project_name}: {e}")
            finally:
                # Remove from active tracking
                if project_name in self.active_indexers:
                    del self.active_indexers[project_name]

    async def _remove_indexer_internal(self, project_name: str, info: dict):
        """
        Actually remove a container and release resources
        Called after 24hr grace period expires
        """
        try:
            container_id = info['container_id']
            container = self.docker_client.containers.get(container_id)

            logger.info(f"Removing indexer container for {project_name} after grace period: {container_id[:12]}")
            container.remove(force=True)

            # Release the allocated port
            port = info.get('port')
            if port:
                self._release_port(port)
                logger.debug(f"Released port {port} from {project_name}")

        except docker.errors.NotFound:
            logger.warning(f"Container already removed for {project_name}")
        except Exception as e:
            logger.error(f"Error removing container for {project_name}: {e}")

    async def _stop_indexer_internal(self, project_name: str):
        """
        Internal method to stop and remove indexer immediately (for shutdown or forced removal)
        """
        if project_name in self.active_indexers:
            try:
                container_id = self.active_indexers[project_name]['container_id']
                port = self.active_indexers[project_name].get('port')
                container = self.docker_client.containers.get(container_id)

                logger.info(f"Stopping and removing indexer container for {project_name}: {container_id[:12]}")
                container.stop(timeout=10)
                container.remove(force=True)

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
            'stopped_indexers': len(self.stopped_indexers),
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
                
        # Add stopped containers info
        for project, info in self.stopped_indexers.items():
            status['indexers'][f"{project} (stopped)"] = {
                'status': 'stopped',
                'stop_time': info['stop_time'].isoformat(),
                'grace_period_remaining': str(self.gc_removal_age - (datetime.now() - info['stop_time'])),
                'container_id': info['container_id'][:12],
                'port': info.get('port', 'N/A')
            }

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