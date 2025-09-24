#!/usr/bin/env python3
"""
Docker State Helper - ADR-0097 Gradual Migration
Query Docker directly for container state (single source of truth)
This is a NON-BREAKING addition that works alongside existing code.

Author: L9 Engineering Team
Date: September 2025
"""

import os
import hashlib
import docker
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)


class DockerStateHelper:
    """
    Helper class to query Docker directly for container state.
    ADR-0097: Docker as single source of truth - gradual migration path.
    """

    def __init__(self):
        """Initialize Docker client"""
        self.docker_client = docker.from_env()

    def get_project_hash(self, project_path: str) -> str:
        """
        Generate deterministic project hash from path.
        ADR-0097: Stable project identification.
        """
        normalized_path = os.path.abspath(os.path.normpath(project_path))
        return hashlib.sha256(normalized_path.encode()).hexdigest()

    def find_container_by_project(self, project_path: str) -> Optional[Dict[str, Any]]:
        """
        Find container for a project by querying Docker directly.
        Returns container info or None if not found.

        This replaces checking active_indexers dict or container_registry.
        """
        project_hash = self.get_project_hash(project_path)

        try:
            # Query Docker for containers with our labels
            containers = self.docker_client.containers.list(
                filters={
                    'label': [
                        'com.l9.managed=true',
                        f'com.l9.project_hash={project_hash}'
                    ],
                    'status': 'running'
                }
            )

            if containers:
                container = containers[0]  # Should only be one per project

                # Extract port from container
                ports = container.ports.get('8080/tcp')
                port = None
                if ports and len(ports) > 0:
                    port = int(ports[0]['HostPort'])

                return {
                    'container_id': container.id,
                    'container_name': container.name,
                    'port': port,
                    'project_path': container.labels.get('com.l9.project_path', project_path),
                    'project_name': container.labels.get('com.l9.project'),
                    'created': container.labels.get('com.l9.created'),
                    'status': container.status
                }
        except Exception as e:
            logger.error(f"[ADR-0097] Error querying Docker: {e}")

        return None

    def find_container_by_name(self, project_name: str) -> Optional[Dict[str, Any]]:
        """
        Find container by project name (backward compatibility).
        Uses the existing com.l9.project label.
        """
        try:
            containers = self.docker_client.containers.list(
                filters={
                    'label': f'com.l9.project={project_name}',
                    'status': 'running'
                }
            )

            if containers:
                container = containers[0]

                # Extract port from container
                ports = container.ports.get('8080/tcp')
                port = None
                if ports and len(ports) > 0:
                    port = int(ports[0]['HostPort'])

                return {
                    'container_id': container.id,
                    'container_name': container.name,
                    'port': port,
                    'project_name': project_name,
                    'created': container.labels.get('com.l9.created'),
                    'status': container.status
                }
        except Exception as e:
            logger.error(f"[ADR-0097] Error querying Docker: {e}")

        return None

    def list_all_managed_containers(self) -> List[Dict[str, Any]]:
        """
        List all L9-managed containers by querying Docker.
        Replaces iterating over active_indexers or container_registry.
        """
        managed_containers = []

        try:
            containers = self.docker_client.containers.list(
                filters={'label': 'com.l9.managed=true'},
                all=True  # Include stopped containers
            )

            for container in containers:
                # Extract port from container
                ports = container.ports.get('8080/tcp', [])
                port = None
                if ports and len(ports) > 0:
                    port = int(ports[0]['HostPort'])

                managed_containers.append({
                    'container_id': container.id,
                    'container_name': container.name,
                    'port': port,
                    'project_name': container.labels.get('com.l9.project'),
                    'project_path': container.labels.get('com.l9.project_path'),
                    'created': container.labels.get('com.l9.created'),
                    'status': container.status
                })
        except Exception as e:
            logger.error(f"[ADR-0097] Error listing containers: {e}")

        return managed_containers

    def get_used_ports(self) -> List[int]:
        """
        Get all ports currently in use by L9 containers.
        Queries Docker directly instead of maintaining a separate list.
        """
        used_ports = []

        try:
            containers = self.docker_client.containers.list(
                filters={'label': 'com.l9.managed=true'},
                all=False  # Only running containers
            )

            for container in containers:
                ports = container.ports.get('8080/tcp')
                if ports and len(ports) > 0:
                    port = int(ports[0]['HostPort'])
                    used_ports.append(port)
        except Exception as e:
            logger.error(f"[ADR-0097] Error getting used ports: {e}")

        return used_ports


# Singleton instance for easy import
docker_state = DockerStateHelper()


# Example usage functions showing gradual migration
async def ensure_indexer_with_docker_truth(project_path: str) -> Dict[str, Any]:
    """
    Example of using Docker as truth instead of checking multiple state stores.
    This could gradually replace the existing ensure_indexer logic.
    """
    # Query Docker directly - single source of truth
    container_info = docker_state.find_container_by_project(project_path)

    if container_info:
        logger.info(f"[ADR-0097] Found existing container via Docker query")
        return {
            'port': container_info['port'],
            'status': 'existing',
            'container_id': container_info['container_id']
        }

    # No container found, would create new one here
    logger.info(f"[ADR-0097] No container found, would create new one")
    return {'status': 'not_found'}


if __name__ == "__main__":
    # Test the helper
    import asyncio

    async def test():
        """Test Docker state queries"""
        # List all managed containers
        containers = docker_state.list_all_managed_containers()
        print(f"Found {len(containers)} L9-managed containers:")
        for c in containers:
            print(f"  - {c['project_name']}: {c['status']} (port {c['port']})")

        # Test finding by project path
        test_path = "/Users/mkr/local-coding/claude-l9-template"
        result = docker_state.find_container_by_project(test_path)
        if result:
            print(f"\nFound container for {test_path}:")
            print(f"  Port: {result['port']}")
            print(f"  Status: {result['status']}")
        else:
            print(f"\nNo container found for {test_path}")

        # Get used ports
        ports = docker_state.get_used_ports()
        print(f"\nPorts in use: {ports}")

    asyncio.run(test())