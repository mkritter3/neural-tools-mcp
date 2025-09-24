#!/usr/bin/env python3
"""
Project Orchestrator Service - ADR-0097
Multi-project indexer lifecycle management with Redis state registry

This module provides context-aware orchestration for multiple projects,
allowing Claude to seamlessly work across different directories while
preserving resource efficiency and preventing conflicts.
"""

import os
import json
import time
import hashlib
import asyncio
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
from pathlib import Path

import docker
from docker.models.containers import Container
import aioredis

logger = logging.getLogger(__name__)

# Feature flag for multi-project mode
MCP_MULTI_PROJECT_MODE = os.getenv("MCP_MULTI_PROJECT_MODE", "false").lower() == "true"

# Configuration
IDLE_THRESHOLD_SECONDS = int(os.getenv("MCP_IDLE_THRESHOLD", "3600"))  # 1 hour default
JANITOR_INTERVAL_SECONDS = int(os.getenv("MCP_JANITOR_INTERVAL", "300"))  # 5 minutes
MAX_CONCURRENT_PROJECTS = int(os.getenv("MCP_MAX_PROJECTS", "10"))
DEFAULT_MEMORY_LIMIT = os.getenv("MCP_MEMORY_LIMIT", "2g")
DEFAULT_CPU_SHARES = int(os.getenv("MCP_CPU_SHARES", "512"))

class ProjectOrchestrator:
    """Manages indexer containers across multiple projects"""

    def __init__(self, redis_client: aioredis.Redis, docker_client: docker.DockerClient):
        self.redis = redis_client
        self.docker = docker_client
        self.registry_key = "mcp:project_registry"
        self.lock_prefix = "mcp:create:"
        self._janitor_task: Optional[asyncio.Task] = None

    def get_project_id(self, project_dir: str) -> str:
        """Generate deterministic project ID from directory path"""
        normalized_path = os.path.abspath(os.path.normpath(project_dir))
        # Use SHA1 hash for consistent, filesystem-agnostic ID
        return hashlib.sha1(normalized_path.encode('utf-8')).hexdigest()[:12]

    async def get_or_create_indexer(self, project_dir: str) -> Dict[str, Any]:
        """
        Get existing or create new indexer for project.
        Returns: {"port": int, "status": "existing|created", "container_id": str}
        """
        if not MCP_MULTI_PROJECT_MODE:
            # Legacy single-project mode
            return await self._legacy_start_indexer(project_dir)

        project_id = self.get_project_id(project_dir)
        logger.info(f"🔍 Getting indexer for project {project_id} at {project_dir}")

        # Check registry for existing container
        project_info = await self._get_project_info(project_id)
        if project_info and await self._is_container_running(project_info['container_id']):
            await self._update_heartbeat(project_id)
            logger.info(f"✅ Found existing indexer on port {project_info['port']}")
            return {
                "port": project_info['port'],
                "status": "existing",
                "container_id": project_info['container_id']
            }

        # Need to create new container
        return await self._create_indexer(project_id, project_dir)

    async def _get_project_info(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project info from Redis registry"""
        try:
            info_json = await self.redis.hget(self.registry_key, project_id)
            if info_json:
                return json.loads(info_json)
        except Exception as e:
            logger.warning(f"Failed to get project info: {e}")
        return None

    async def _is_container_running(self, container_id: str) -> bool:
        """Check if container is actually running"""
        try:
            container = self.docker.containers.get(container_id)
            return container.status == "running"
        except docker.errors.NotFound:
            return False
        except Exception as e:
            logger.warning(f"Error checking container {container_id}: {e}")
            return False

    async def _update_heartbeat(self, project_id: str):
        """Update last heartbeat timestamp for project"""
        try:
            info = await self._get_project_info(project_id)
            if info:
                info['last_heartbeat'] = time.time()
                await self.redis.hset(self.registry_key, project_id, json.dumps(info))
        except Exception as e:
            logger.warning(f"Failed to update heartbeat: {e}")

    async def _create_indexer(self, project_id: str, project_dir: str) -> Dict[str, Any]:
        """Create new indexer container with distributed lock"""
        lock_key = f"{self.lock_prefix}{project_id}"

        # Use Redis distributed lock to prevent races
        async with self.redis.lock(lock_key, timeout=30):
            # Double-check in case another process created it
            project_info = await self._get_project_info(project_id)
            if project_info and await self._is_container_running(project_info['container_id']):
                return {
                    "port": project_info['port'],
                    "status": "existing",
                    "container_id": project_info['container_id']
                }

            # Check we're not exceeding max concurrent projects
            all_projects = await self.redis.hlen(self.registry_key)
            if all_projects >= MAX_CONCURRENT_PROJECTS:
                # Trigger immediate cleanup of idle containers
                await self._cleanup_idle_containers()
                # Re-check
                all_projects = await self.redis.hlen(self.registry_key)
                if all_projects >= MAX_CONCURRENT_PROJECTS:
                    raise RuntimeError(f"Max concurrent projects ({MAX_CONCURRENT_PROJECTS}) reached")

            # Find available port
            port = await self._find_free_port()

            # Start container
            container_name = f"mcp-indexer-{project_id}"
            logger.info(f"🚀 Starting new indexer container {container_name} on port {port}")

            try:
                # Remove any existing container with same name (cleanup)
                try:
                    old_container = self.docker.containers.get(container_name)
                    old_container.remove(force=True)
                    logger.info(f"Removed stale container {container_name}")
                except docker.errors.NotFound:
                    pass

                # Start new container
                container = self.docker.containers.run(
                    image="indexer:production",
                    name=container_name,
                    ports={'48100/tcp': port},
                    environment={
                        'PROJECT_PATH': project_dir,
                        'PROJECT_ID': project_id,
                        'PROJECT_NAME': Path(project_dir).name,
                        # Pass through Neo4j config
                        'NEO4J_URI': os.getenv('NEO4J_URI', 'bolt://localhost:47687'),
                        'NEO4J_USERNAME': os.getenv('NEO4J_USERNAME', 'neo4j'),
                        'NEO4J_PASSWORD': os.getenv('NEO4J_PASSWORD', 'graphrag-password'),
                    },
                    labels={
                        'com.l9.project': project_id,
                        'com.l9.managed': 'true',
                        'com.l9.type': 'indexer',
                        'com.l9.project_path': project_dir
                    },
                    mem_limit=DEFAULT_MEMORY_LIMIT,
                    cpu_shares=DEFAULT_CPU_SHARES,
                    detach=True,
                    remove=False  # Don't auto-remove, we'll manage lifecycle
                )

                # Register in Redis
                project_info = {
                    "container_id": container.id,
                    "container_name": container_name,
                    "port": port,
                    "status": "running",
                    "last_heartbeat": time.time(),
                    "project_path": project_dir,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                await self.redis.hset(self.registry_key, project_id, json.dumps(project_info))

                logger.info(f"✅ Successfully started indexer for {project_id}")
                return {
                    "port": port,
                    "status": "created",
                    "container_id": container.id
                }

            except Exception as e:
                logger.error(f"Failed to start container: {e}")
                raise

    async def _find_free_port(self, start_port: int = 48100) -> int:
        """Find an available port for the indexer"""
        # Get all registered ports from Redis
        registered_ports = set()
        try:
            all_projects = await self.redis.hgetall(self.registry_key)
            for info_json in all_projects.values():
                info = json.loads(info_json)
                registered_ports.add(info['port'])
        except Exception as e:
            logger.warning(f"Error getting registered ports: {e}")

        # Find first available port
        for port in range(start_port, start_port + 100):
            if port not in registered_ports:
                # Double-check with Docker
                try:
                    # Check if any container is using this port
                    containers = self.docker.containers.list(
                        filters={'publish': f'{port}/tcp'}
                    )
                    if not containers:
                        return port
                except Exception:
                    pass

        raise RuntimeError("No free ports available")

    async def _cleanup_idle_containers(self):
        """Clean up containers that have been idle too long"""
        now = time.time()
        logger.info("🧹 Running cleanup of idle containers")

        try:
            all_projects = await self.redis.hgetall(self.registry_key)
            for project_id, info_json in all_projects.items():
                try:
                    info = json.loads(info_json)
                    idle_time = now - info['last_heartbeat']

                    if idle_time > IDLE_THRESHOLD_SECONDS:
                        logger.info(f"Cleaning up idle container for {project_id} (idle {idle_time:.0f}s)")

                        # Stop and remove container
                        try:
                            container = self.docker.containers.get(info['container_id'])
                            container.stop(timeout=10)
                            container.remove()
                        except docker.errors.NotFound:
                            pass
                        except Exception as e:
                            logger.warning(f"Error stopping container: {e}")

                        # Remove from registry
                        await self.redis.hdel(self.registry_key, project_id)

                except Exception as e:
                    logger.warning(f"Error processing cleanup for {project_id}: {e}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def start_janitor(self):
        """Start background cleanup task"""
        if self._janitor_task:
            return

        async def janitor_loop():
            while True:
                try:
                    await asyncio.sleep(JANITOR_INTERVAL_SECONDS)
                    await self._cleanup_idle_containers()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Janitor error: {e}")

        self._janitor_task = asyncio.create_task(janitor_loop())
        logger.info("🧹 Started janitor task")

    async def stop_janitor(self):
        """Stop background cleanup task"""
        if self._janitor_task:
            self._janitor_task.cancel()
            try:
                await self._janitor_task
            except asyncio.CancelledError:
                pass
            self._janitor_task = None
            logger.info("🛑 Stopped janitor task")

    async def shutdown(self):
        """Graceful shutdown - stop all managed containers"""
        logger.info("Shutting down project orchestrator")

        # Stop janitor
        await self.stop_janitor()

        if not MCP_MULTI_PROJECT_MODE:
            return

        # Stop all managed containers
        try:
            all_projects = await self.redis.hgetall(self.registry_key)
            for project_id, info_json in all_projects.items():
                try:
                    info = json.loads(info_json)
                    container = self.docker.containers.get(info['container_id'])
                    logger.info(f"Stopping container for {project_id}")
                    container.stop(timeout=10)
                    container.remove()
                except Exception as e:
                    logger.warning(f"Error stopping container: {e}")

            # Clear registry
            await self.redis.delete(self.registry_key)

        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def _legacy_start_indexer(self, project_dir: str) -> Dict[str, Any]:
        """Legacy single-project indexer logic (preserves existing behavior)"""
        # This is a placeholder - implement your existing logic here
        # For now, return a default response
        logger.info("Running in legacy single-project mode")
        return {
            "port": 48100,
            "status": "legacy",
            "container_id": "legacy-mode"
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status and metrics"""
        status = {
            "mode": "multi-project" if MCP_MULTI_PROJECT_MODE else "legacy",
            "projects": {}
        }

        if MCP_MULTI_PROJECT_MODE:
            try:
                all_projects = await self.redis.hgetall(self.registry_key)
                for project_id, info_json in all_projects.items():
                    info = json.loads(info_json)
                    status["projects"][project_id] = {
                        "port": info['port'],
                        "status": info['status'],
                        "path": info['project_path'],
                        "idle_seconds": int(time.time() - info['last_heartbeat'])
                    }
            except Exception as e:
                status["error"] = str(e)

        return status