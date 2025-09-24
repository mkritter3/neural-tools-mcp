#!/usr/bin/env python3
"""
Container Orchestrator - ADR-0100 Implementation
Elite container orchestration with proactive lifecycle management

Author: L9 Engineering Team
Date: September 24, 2025
"""

import asyncio
import atexit
import hashlib
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import docker
from docker.models.containers import Container
import redis

logger = logging.getLogger(__name__)


class ContainerOrchestrator:
    """
    Comprehensive container orchestration system.
    Implements ADR-0100 with proactive initialization, health monitoring,
    and session-aware lifecycle management.
    """

    def __init__(self):
        self.docker_client = docker.from_env()
        self.project_path = None
        self.project_hash = None
        self.session_id = None
        self.accepting_requests = True
        self.health_monitor_task = None
        self.redis_client = None  # For session counting

        # Container configuration
        self.essential_containers = [
            {
                "name": "neo4j",
                "image": "neo4j:5.22.0",
                "port": 47687,
                "healthcheck": {
                    "test": ["CMD", "cypher-shell", "-u", "neo4j", "-p", "graphrag-password", "RETURN 1"],
                    "interval": 30000000000,  # 30s in nanoseconds
                    "timeout": 10000000000,   # 10s
                    "retries": 3,
                    "start_period": 60000000000  # 60s
                }
            },
            {
                "name": "redis-cache",
                "image": "redis:7-alpine",
                "port": 46379,
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": 10000000000,
                    "timeout": 3000000000,
                    "retries": 3
                }
            },
            {
                "name": "redis-queue",
                "image": "redis:7-alpine",
                "port": 46380,
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": 10000000000,
                    "timeout": 3000000000,
                    "retries": 3
                }
            },
            {
                "name": "nomic",
                "image": "nomic:production",
                "port": 48000,
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:48000/health"],
                    "interval": 30000000000,
                    "timeout": 5000000000,
                    "retries": 3
                }
            }
        ]

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        atexit.register(self._cleanup)

    async def initialize(self) -> Dict[str, Any]:
        """
        Proactive initialization when MCP server starts.
        Starts all essential containers before any tool calls.
        """
        logger.info("🚀 Container Orchestrator initializing...")

        # Check Docker daemon is accessible
        try:
            self.docker_client.ping()
        except Exception as e:
            logger.error(f"❌ Docker daemon not accessible: {e}")
            raise RuntimeError("Docker daemon must be running for neural-tools to work")

        # 1. Detect project using CLAUDE_PROJECT_DIR
        self.project_path = os.getenv("CLAUDE_PROJECT_DIR")
        if not self.project_path:
            # Fallback to current working directory
            self.project_path = os.getcwd()

        self.project_hash = self._hash_project(self.project_path)
        self.session_id = f"mcp-{int(time.time())}-{os.getpid()}"

        logger.info(f"📁 Project: {self.project_path}")
        logger.info(f"🔑 Session: {self.session_id}")

        # Initialize Redis for session counting
        try:
            # Try without password first (common for local dev)
            self.redis_client = redis.Redis(
                host='localhost',
                port=46379,
                decode_responses=True,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for session management")
        except redis.AuthenticationError:
            # Try with password if auth required
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=46379,
                    password='cache-password',
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.redis_client.ping()
                logger.info("✅ Redis connected with auth for session management")
            except Exception as e:
                logger.warning(f"⚠️ Redis not available for session counting: {e}")
                self.redis_client = None
        except Exception as e:
            logger.warning(f"⚠️ Redis not available for session counting: {e}")
            self.redis_client = None

        # 2. Start essential containers proactively
        results = {}
        for container_config in self.essential_containers:
            try:
                container = await self._start_container(container_config)
                results[container_config["name"]] = {
                    "status": "started",
                    "container_id": container.id[:12],
                    "port": container_config["port"]
                }
                logger.info(f"✅ Started {container_config['name']} on port {container_config['port']}")
            except Exception as e:
                logger.error(f"❌ Failed to start {container_config['name']}: {e}")
                results[container_config["name"]] = {
                    "status": "failed",
                    "error": str(e)
                }

        # 3. Wait for health checks
        await self._wait_for_healthy(timeout=30)

        # 4. Start project indexer
        indexer_result = await self._start_indexer()
        results["indexer"] = indexer_result

        # 5. Register session
        await self._register_session()

        # 6. Start health monitor
        self.health_monitor_task = asyncio.create_task(self._monitor_health())

        logger.info("✅ Container Orchestrator initialized successfully")
        return {
            "status": "initialized",
            "project": self.project_path,
            "session": self.session_id,
            "containers": results
        }

    async def _start_container(self, config: Dict) -> Container:
        """Start a single container with configuration."""
        # First check if container is already running on the expected port
        # This handles docker-compose managed containers
        try:
            # Look for any container using the expected port
            all_containers = self.docker_client.containers.list(all=True)
            for container in all_containers:
                # Check if this container is using our port
                ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                for port_key, port_bindings in ports.items():
                    if port_bindings:
                        for binding in port_bindings:
                            if binding.get('HostPort') == str(config['port']):
                                logger.info(f"Found existing container {container.name} on port {config['port']}")
                                # Update labels if possible (Docker doesn't allow label updates on running containers)
                                # Just track it in our session
                                return container
        except Exception as e:
            logger.warning(f"Error checking existing containers: {e}")

        # Try exact name match
        container_name = f"{config['name']}-{self.project_hash}"
        try:
            existing = self.docker_client.containers.get(container_name)
            if existing.status == "running":
                logger.info(f"Container {container_name} already running")
                return existing
            else:
                logger.info(f"Restarting container {container_name}")
                existing.restart()
                return existing
        except docker.errors.NotFound:
            pass

        # Create new container only if port is free
        labels = {
            "com.l9.managed": "true",
            "com.l9.service": config["name"],
            "com.l9.project_hash": self.project_hash,
            "com.l9.project_path": self.project_path,
            "com.l9.session_count": "1",
            f"com.l9.session.{self.session_id}": str(time.time()),
            "com.l9.created": str(int(time.time()))
        }

        # Prepare port bindings
        port_bindings = {}
        if "port" in config:
            internal_port = self._get_internal_port(config["name"])
            port_bindings[f"{internal_port}/tcp"] = config["port"]

        # Start container with restart policy for auto-recovery
        try:
            container = self.docker_client.containers.run(
                config["image"],
                name=container_name,
                labels=labels,
                ports=port_bindings,
                environment=self._get_environment(config["name"]),
                healthcheck=config.get("healthcheck"),
                restart_policy={"Name": "unless-stopped"},  # Docker handles crashes
                detach=True,
                remove=False,
                network_mode="bridge"
            )
            return container
        except docker.errors.APIError as e:
            # Handle container name conflict (race condition)
            if e.response and e.response.status_code == 409 and "is already in use by container" in str(e):
                logger.info(f"Container {container_name} was created by another session. Re-fetching.")
                return self.docker_client.containers.get(container_name)

            # Handle port already allocated - find and adopt the real container
            if "port is already allocated" in str(e):
                logger.warning(f"Port {config['port']} is allocated. Finding and adopting existing container.")
                # Find the real container using the port
                all_containers = self.docker_client.containers.list()
                for container in all_containers:
                    ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                    for port_bindings in ports.values():
                        if port_bindings:
                            for binding in port_bindings:
                                if binding.get('HostPort') == str(config['port']):
                                    logger.info(f"Adopted container {container.name} using port {config['port']}.")
                                    return container  # Return the REAL container object

                # If we can't find it, something is wrong
                logger.error(f"Port {config['port']} is allocated, but could not find the container using it.")
                raise e  # Re-raise the original exception
            raise

    def _get_internal_port(self, service: str) -> int:
        """Get the internal port for a service."""
        ports = {
            "neo4j": 7687,
            "redis-cache": 6379,
            "redis-queue": 6379,
            "nomic": 48000,
            "indexer": 48100
        }
        return ports.get(service, 8080)

    def _get_environment(self, service: str) -> Dict[str, str]:
        """Get environment variables for a service."""
        if service == "neo4j":
            return {
                "NEO4J_AUTH": "neo4j/graphrag-password",
                "NEO4J_PLUGINS": '["apoc", "graph-data-science"]',
                "NEO4J_dbms_memory_heap_initial__size": "512m",
                "NEO4J_dbms_memory_heap_max__size": "2g"
            }
        elif service == "redis-cache":
            return {
                "REDIS_PASSWORD": "cache-password"
            }
        elif service == "redis-queue":
            return {
                "REDIS_PASSWORD": "queue-password"
            }
        elif service == "nomic":
            return {
                "EMBEDDING_MODEL": "nomic-embed-text-v2",
                "MAX_BATCH_SIZE": "100"
            }
        return {}

    async def _start_indexer(self) -> Dict:
        """Start the project-specific indexer."""
        from .indexer_orchestrator import IndexerOrchestrator

        try:
            orchestrator = IndexerOrchestrator()
            await orchestrator.initialize()  # Initialize Docker client and Redis
            port = await orchestrator.ensure_indexer(
                Path(self.project_path).name,
                self.project_path
            )

            # Update session labels on indexer container
            container_name = f"indexer-{Path(self.project_path).name}"
            try:
                container = self.docker_client.containers.get(container_name)
                labels = container.labels.copy()
                labels[f"com.l9.session.{self.session_id}"] = str(time.time())
                count = int(labels.get("com.l9.session_count", 0))
                labels["com.l9.session_count"] = str(count + 1)
                # Note: Docker doesn't support updating labels, we'll track this separately
            except docker.errors.NotFound:
                logger.warning(f"Indexer container {container_name} not found")

            return {
                "status": "started",
                "port": port,
                "project": Path(self.project_path).name
            }
        except Exception as e:
            logger.error(f"Failed to start indexer: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def _wait_for_healthy(self, timeout: int = 30):
        """Wait for all containers to become healthy."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_healthy = True

            for config in self.essential_containers:
                container_name = f"{config['name']}-{self.project_hash}"
                try:
                    container = self.docker_client.containers.get(container_name)
                    container.reload()

                    # Check health status
                    health = container.attrs.get("State", {}).get("Health", {})
                    if health.get("Status") != "healthy":
                        all_healthy = False
                        logger.debug(f"Waiting for {container_name}: {health.get('Status', 'unknown')}")
                except docker.errors.NotFound:
                    all_healthy = False
                    logger.debug(f"Container {container_name} not found")

            if all_healthy:
                logger.info("✅ All containers healthy")
                return

            await asyncio.sleep(1)

        logger.warning(f"⚠️ Some containers not healthy after {timeout}s")

    async def _register_session(self):
        """Register this MCP session using Redis."""
        logger.info(f"📝 Registering session {self.session_id}")

        if self.redis_client:
            try:
                # Increment session count for this project
                key = f"project:{self.project_hash}:sessions"
                self.redis_client.sadd(key, self.session_id)
                count = self.redis_client.scard(key)
                logger.info(f"✅ Session registered. Active sessions for project: {count}")

                # Set expiry in case of ungraceful shutdown (24 hours)
                self.redis_client.expire(key, 86400)
            except Exception as e:
                logger.error(f"Failed to register session in Redis: {e}")
        else:
            logger.warning("Redis not available, session counting disabled")

    async def _monitor_health(self):
        """Background task for health monitoring."""
        logger.info("🏥 Health monitor started")
        check_interval = 30  # seconds

        while self.accepting_requests:
            try:
                await asyncio.sleep(check_interval)

                containers = self.docker_client.containers.list(
                    filters={"label": f"com.l9.project_hash={self.project_hash}"}
                )

                for container in containers:
                    container.reload()
                    health = container.attrs.get("State", {}).get("Health", {})
                    status = health.get("Status", "unknown")

                    if status == "unhealthy":
                        logger.warning(f"⚠️ Container {container.name} is unhealthy")
                        await self._recover_container(container)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

        logger.info("🏥 Health monitor stopped")

    async def _recover_container(self, container: Container):
        """Attempt to recover an unhealthy container."""
        logger.info(f"🔧 Attempting recovery for {container.name}")

        try:
            # Try restart first
            container.restart()
            await asyncio.sleep(10)

            # Check if healthy after restart
            container.reload()
            health = container.attrs.get("State", {}).get("Health", {})

            if health.get("Status") == "healthy":
                logger.info(f"✅ Recovery successful for {container.name}")
            else:
                logger.error(f"❌ Recovery failed for {container.name}")
        except Exception as e:
            logger.error(f"Recovery error for {container.name}: {e}")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"📛 Received signal {signum}, initiating graceful shutdown")

        # Stop accepting new requests
        self.accepting_requests = False

        # Cancel health monitor
        if self.health_monitor_task:
            self.health_monitor_task.cancel()

        # Schedule unregister_session on the event loop (safe from signal handler)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self._unregister_session())
        else:
            # If no loop is running, we can use asyncio.run
            asyncio.run(self._unregister_session())

        sys.exit(0)

    async def _unregister_session(self):
        """Unregister this MCP session and stop containers if needed."""
        logger.info(f"📝 Unregistering session {self.session_id}")

        if self.redis_client:
            try:
                # Remove this session from the set
                key = f"project:{self.project_hash}:sessions"
                self.redis_client.srem(key, self.session_id)
                remaining = self.redis_client.scard(key)

                logger.info(f"Session unregistered. Remaining sessions: {remaining}")

                # If no more sessions, stop containers
                if remaining == 0:
                    logger.info("🛑 Last session ended, stopping containers...")
                    await self._stop_project_containers()
            except Exception as e:
                logger.error(f"Failed to unregister session: {e}")
        else:
            logger.warning("Redis not available, cannot track sessions")

    async def _stop_project_containers(self):
        """Stop all containers for this project."""
        containers = self.docker_client.containers.list(
            filters={"label": f"com.l9.project_hash={self.project_hash}"}
        )

        for container in containers:
            try:
                logger.info(f"Stopping container {container.name}")
                container.stop(timeout=10)
            except Exception as e:
                logger.error(f"Failed to stop {container.name}: {e}")

    def _cleanup(self):
        """Cleanup on exit."""
        logger.info("🧹 Cleaning up...")

    def _hash_project(self, project_path: str) -> str:
        """Generate stable hash for project path."""
        return hashlib.sha256(project_path.encode()).hexdigest()[:12]


# Singleton instance
_orchestrator = None


async def get_container_orchestrator() -> ContainerOrchestrator:
    """Get or create the singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ContainerOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator