#!/usr/bin/env python3
"""
Docker State Observability - Phase 0 of ADR-0097 Migration
Logs state divergence between Docker and internal dicts WITHOUT changing behavior.
This is a read-only, non-breaking addition for understanding the problem.

Author: L9 Engineering Team
Date: September 2025
"""

import logging
import docker
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class DockerStateObserver:
    """
    Phase 0: Observability-only helper to detect state divergence.
    Does NOT change any behavior - only logs differences.
    """

    def __init__(self):
        self.docker_client = docker.from_env()
        self.divergence_count = 0
        self.total_checks = 0

    def check_state_divergence(
        self,
        project_name: str,
        dict_state: Dict[str, Any],
        source: str = "unknown"
    ) -> None:
        """
        Compare dict state with Docker state and log any differences.
        This is READ-ONLY - does not change any state.

        Args:
            project_name: Name of the project
            dict_state: State from active_indexers or container_registry
            source: Which dict this came from (for logging)
        """
        self.total_checks += 1

        try:
            # Query Docker for this project
            containers = self.docker_client.containers.list(
                filters={'label': f'com.l9.project={project_name}'}
            )

            if not containers and dict_state:
                # Dict thinks container exists, Docker doesn't
                self.divergence_count += 1
                logger.warning(
                    f"[DIVERGENCE] {source} has container for '{project_name}' "
                    f"but Docker doesn't. Dict state: {dict_state}"
                )
                return

            if containers and not dict_state:
                # Docker has container, dict doesn't
                self.divergence_count += 1
                container = containers[0]
                logger.warning(
                    f"[DIVERGENCE] Docker has container '{container.name}' "
                    f"for '{project_name}' but {source} doesn't know about it"
                )
                return

            if containers and dict_state:
                # Both have state - compare details
                container = containers[0]
                docker_state = self._extract_docker_state(container)

                # Compare container IDs
                dict_id = dict_state.get('container_id', '')
                if dict_id and dict_id != container.id:
                    self.divergence_count += 1
                    logger.warning(
                        f"[DIVERGENCE] Container ID mismatch for '{project_name}': "
                        f"{source}={dict_id[:12]}, Docker={container.id[:12]}"
                    )

                # Compare ports
                dict_port = dict_state.get('port')
                docker_port = docker_state.get('port')
                if dict_port and docker_port and dict_port != docker_port:
                    self.divergence_count += 1
                    logger.warning(
                        f"[DIVERGENCE] Port mismatch for '{project_name}': "
                        f"{source}={dict_port}, Docker={docker_port}"
                    )

                # Log successful match (debug level)
                if self.divergence_count == 0:
                    logger.debug(
                        f"[MATCH] State consistent for '{project_name}' "
                        f"between {source} and Docker"
                    )

        except Exception as e:
            logger.error(f"[OBSERVER] Error checking state: {e}")

    def _extract_docker_state(self, container) -> Dict[str, Any]:
        """Extract state info from Docker container"""
        # Get port from container
        ports = container.ports.get('8080/tcp', [])
        port = None
        if ports and len(ports) > 0:
            port = int(ports[0]['HostPort'])

        return {
            'container_id': container.id,
            'name': container.name,
            'port': port,
            'status': container.status,
            'labels': container.labels
        }

    def report_metrics(self) -> Dict[str, Any]:
        """
        Report observability metrics.
        Call this periodically to understand divergence rate.
        """
        if self.total_checks == 0:
            divergence_rate = 0
        else:
            divergence_rate = (self.divergence_count / self.total_checks) * 100

        metrics = {
            'total_checks': self.total_checks,
            'divergence_count': self.divergence_count,
            'divergence_rate': f"{divergence_rate:.1f}%"
        }

        logger.info(
            f"[METRICS] State divergence: {self.divergence_count}/{self.total_checks} "
            f"({divergence_rate:.1f}% divergence rate)"
        )

        return metrics


# Global singleton for easy integration
observer = DockerStateObserver()


# Integration examples - add these to existing code WITHOUT changing behavior
def observe_active_indexers(active_indexers: Dict[str, Any]) -> None:
    """
    Add to IndexerOrchestrator to observe active_indexers state.
    Call this periodically or after state changes.
    """
    for project_name, state in active_indexers.items():
        observer.check_state_divergence(
            project_name,
            state,
            source="active_indexers"
        )


def observe_container_registry(container_registry: Dict[str, Any]) -> None:
    """
    Add to ProjectContextManager to observe container_registry state.
    Call this periodically or after state changes.
    """
    for project_name, port in container_registry.items():
        # Convert simple port to dict format
        state = {'port': port} if isinstance(port, int) else port
        observer.check_state_divergence(
            project_name,
            state,
            source="container_registry"
        )


# Example integration points (add to existing code):
"""
# In IndexerOrchestrator.ensure_indexer():
from .docker_observability import observer
# After updating active_indexers:
observer.check_state_divergence(project_name, self.active_indexers[project_name], "active_indexers")

# In ProjectContextManager.set_project():
from .docker_observability import observer
# After updating container_registry:
observer.check_state_divergence(project_name, {'port': port}, "container_registry")

# Periodic reporting (add to garbage collection loop):
from .docker_observability import observer
observer.report_metrics()
"""