#!/usr/bin/env python3
"""
Container Discovery Service - Implements ADR-0044
Discovers and manages Docker containers for projects.

This service solves the problem of blindly creating new containers
when existing ones are already running. It provides discovery by name
and port, intelligent reuse, and port allocation tracking.

Author: L9 Engineering Team
Date: 2025-09-13
"""

import logging
from typing import Optional, Dict, Any
import docker

logger = logging.getLogger(__name__)


class ContainerDiscoveryService:
    """
    Service for discovering and managing Docker containers.
    Implements ADR-0044 for robust container orchestration.

    Key Features:
    - Discovers existing containers by project name
    - Tracks port allocations
    - Intelligent container reuse
    - Port range management (48100-48199)
    """

    def __init__(self, docker_client=None):
        """
        Initialize with Docker client

        Args:
            docker_client: Docker client instance (creates new if None)
        """
        self.docker = docker_client or docker.from_env()
        logger.info("🔍 ContainerDiscoveryService initialized")

    async def discover_project_container(self, project_name: str) -> Optional[Dict[str, Any]]:
        """
        Find existing container for project.

        Args:
            project_name: Name of the project to find container for

        Returns:
            Dict with container info if found, None otherwise
        """
        try:
            containers = self.docker.containers.list(all=True)

            # First try exact match
            exact_match = None
            partial_matches = []

            for container in containers:
                container_name = container.name
                # Skip containers without names (shouldn't happen but defensive coding)
                if not container_name:
                    continue

                if container_name == f"indexer-{project_name}":
                    exact_match = container
                    break  # Exact match is best, stop here
                elif f"indexer-{project_name}" in container_name:
                    partial_matches.append(container)

            # Use exact match if found, otherwise warn about partial matches
            if exact_match:
                container = exact_match
                logger.info(f"🐳 Found existing container (exact match): {container.name}")
            elif partial_matches:
                # Log warning about ambiguous matches
                logger.warning(f"⚠️ Found {len(partial_matches)} partial matches for indexer-{project_name}: {[c.name for c in partial_matches]}")
                # Skip partial matches to avoid confusion
                return None
            else:
                # No matches found
                return None

            # Now process the found container
            if container:
                # Ensure we have the container name
                container_name = container.name
                if not container_name:
                    logger.warning(f"⚠️ Container {container.id[:12]} has no name, skipping")
                    return None

                # Get port mapping
                ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                host_port = None

                # Look for 8080/tcp mapping
                if '8080/tcp' in ports and ports['8080/tcp']:
                    host_port = int(ports['8080/tcp'][0]['HostPort'])

                # Also check HostConfig for port bindings
                if not host_port:
                    port_bindings = container.attrs.get('HostConfig', {}).get('PortBindings', {})
                    if '8080/tcp' in port_bindings and port_bindings['8080/tcp']:
                        host_port = int(port_bindings['8080/tcp'][0]['HostPort'])

                if host_port:
                        logger.info(f"✅ Container {container_name} is on port {host_port}, status: {container.status}")
                        return {
                            'container': container,
                            'container_id': container.id,
                            'container_name': container_name,
                            'port': host_port,
                            'status': container.status
                        }
                else:
                    logger.warning(f"⚠️ Container {container_name} found but no port mapping")
                    return None

        except Exception as e:
            logger.error(f"❌ Container discovery failed: {e}")

        return None

    async def get_or_create_container(self, project_name: str,
                                     context_manager) -> Dict[str, Any]:
        """
        Get existing container or create new one with port tracking.

        Args:
            project_name: Name of the project
            context_manager: ProjectContextManager for registry updates

        Returns:
            Dict with container info
        """
        # First, try to discover existing container
        existing = await self.discover_project_container(project_name)
        if existing:
            if existing['status'] == 'running':
                # Update registry with discovered port
                context_manager.container_registry[project_name] = existing['port']
                await context_manager._persist_registry()
                logger.info(f"♻️ Reusing existing container on port {existing['port']}")
                return existing
            else:
                # Container exists but not running, start it
                try:
                    container = existing['container']
                    container.start()
                    logger.info(f"▶️ Started existing container {existing['container_name']}")
                    # Update registry
                    context_manager.container_registry[project_name] = existing['port']
                    await context_manager._persist_registry()
                    return {
                        'container': container,
                        'container_id': container.id,
                        'container_name': existing['container_name'],
                        'port': existing['port'],
                        'status': 'running'
                    }
                except Exception as e:
                    logger.error(f"Failed to start container: {e}")
                    # Fall through to create new

        # No existing container, allocate new port
        used_ports = set(context_manager.container_registry.values())
        new_port = None

        # Find available port in range 48100-48199
        for port in range(48100, 48200):
            if port not in used_ports:
                new_port = port
                break

        if not new_port:
            raise RuntimeError("❌ No available ports in range 48100-48199")

        logger.info(f"🚀 Creating new container for {project_name} on port {new_port}")

        try:
            # Create new container with proper configuration
            container = self.docker.containers.run(
                image='l9-neural-indexer:production',  # Use production tag (ADR-0038)
                name=f'indexer-{project_name}',
                ports={'8080/tcp': new_port},
                environment={
                    'PROJECT_NAME': project_name,
                    'PROJECT_PATH': '/workspace',  # ADR-0048: Always use container path
                    # ADR-0037 compliant environment variables
                    'NEO4J_URI': 'bolt://host.docker.internal:47687',
                    'NEO4J_PASSWORD': 'graphrag-password',
                    'QDRANT_HOST': 'host.docker.internal',
                    'QDRANT_PORT': '46333',
                    'EMBEDDING_SERVICE_HOST': 'host.docker.internal',
                    'EMBEDDING_SERVICE_PORT': '48000'
                },
                volumes={
                    str(context_manager.current_project_path): {
                        'bind': '/workspace',
                        'mode': 'ro'
                    }
                },
                detach=True,
                remove=False,  # Keep container for reuse
                network='l9-graphrag-network'  # Use correct network
            )

            # Update registry with new container port
            context_manager.container_registry[project_name] = new_port
            await context_manager._persist_registry()

            logger.info(f"✅ Created container {container.name} on port {new_port}")

            return {
                'container': container,
                'container_id': container.id,
                'container_name': container.name,
                'port': new_port,
                'status': 'running'
            }

        except Exception as e:
            logger.error(f"❌ Failed to create container: {e}")
            raise

    async def stop_project_container(self, project_name: str) -> bool:
        """
        Stop container for a project.

        Args:
            project_name: Name of the project

        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            existing = await self.discover_project_container(project_name)
            if existing and existing['status'] == 'running':
                container = existing['container']
                container.stop(timeout=10)
                logger.info(f"⏹️ Stopped container {existing['container_name']}")
                return True
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")

        return False

    async def cleanup_stale_containers(self, active_projects: list) -> int:
        """
        Clean up containers for projects that are no longer active.

        Args:
            active_projects: List of currently active project names

        Returns:
            Number of containers cleaned up
        """
        cleaned = 0
        try:
            containers = self.docker.containers.list(all=True)

            for container in containers:
                if container.name.startswith('indexer-'):
                    # Extract project name from container name
                    project_name = container.name.replace('indexer-', '')

                    if project_name not in active_projects:
                        logger.info(f"🧹 Cleaning up stale container: {container.name}")
                        try:
                            container.stop(timeout=5)
                            container.remove()
                            cleaned += 1
                        except:
                            pass  # Container might already be stopped/removed

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

        return cleaned