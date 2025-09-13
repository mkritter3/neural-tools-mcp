#!/usr/bin/env python3
"""
Project Detection Service for MCP
Solves the fundamental problem of MCP servers not knowing the user's actual working directory

This service provides multiple strategies to detect the user's actual project:
1. Explicit environment variable (PROJECT_PATH)
2. Registry persistence (last known project)
3. Container discovery (find active containers)
4. Fallback to set_project_context tool

Author: L9 Engineering Team
Date: 2025-09-13
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import docker

logger = logging.getLogger(__name__)


class ProjectDetector:
    """
    Detects the user's actual project context despite MCP isolation
    """

    def __init__(self):
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except:
            logger.warning("Docker not available for container discovery")

    async def detect_user_project(self) -> Tuple[str, Path]:
        """
        Detect the user's actual project using multiple strategies

        Returns:
            Tuple of (project_name, project_path)
        """

        # Strategy 1: Check for explicit PROJECT_PATH environment variable
        if 'PROJECT_PATH' in os.environ:
            project_path = Path(os.environ['PROJECT_PATH'])
            if project_path.exists():
                project_name = self._extract_project_name(project_path)
                logger.info(f"✅ Detected project from PROJECT_PATH: {project_name} at {project_path}")
                return project_name, project_path

        # Strategy 2: Check for PROJECT_NAME environment variable
        if 'PROJECT_NAME' in os.environ:
            project_name = os.environ['PROJECT_NAME']
            # Try to find the project path from registry
            registry_path = Path.home() / ".claude" / "project_registry.json"
            if registry_path.exists():
                try:
                    with open(registry_path) as f:
                        data = json.load(f)
                        if project_name in data.get("projects", {}):
                            project_path = Path(data["projects"][project_name])
                            if project_path.exists():
                                logger.info(f"✅ Detected project from PROJECT_NAME: {project_name} at {project_path}")
                                return project_name, project_path
                except:
                    pass

        # Strategy 3: Check for active Docker containers (indicates active project)
        if self.docker_client:
            active_project = self._detect_from_containers()
            if active_project:
                project_name, project_path = active_project
                logger.info(f"✅ Detected project from active containers: {project_name}")
                return project_name, project_path

        # Strategy 4: Check registry for last active project
        registry_path = Path.home() / ".claude" / "project_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path) as f:
                    data = json.load(f)
                    if "active_project" in data and data["active_project"]:
                        project_name = data["active_project"]
                        if "active_project_path" in data and data["active_project_path"]:
                            project_path = Path(data["active_project_path"])
                            if project_path.exists():
                                logger.info(f"✅ Using last active project from registry: {project_name}")
                                return project_name, project_path
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

        # Strategy 5: Fallback - return None to indicate manual setting needed
        logger.warning("⚠️ Could not detect project. User must call set_project_context()")
        return None, None

    def _extract_project_name(self, project_path: Path) -> str:
        """Extract project name from path"""
        # Check for package.json
        package_json = project_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    if "name" in data:
                        return self._sanitize_name(data["name"])
            except:
                pass

        # Check for pyproject.toml
        pyproject = project_path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomli
                with open(pyproject, 'rb') as f:
                    data = tomli.load(f)
                    if "project" in data and "name" in data["project"]:
                        return self._sanitize_name(data["project"]["name"])
            except:
                pass

        # Fallback to directory name
        return self._sanitize_name(project_path.name)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize project name for use as identifier"""
        import re
        # Convert to lowercase and replace non-alphanumeric with hyphens
        sanitized = re.sub(r'[^a-z0-9]+', '-', name.lower())
        # Remove leading/trailing hyphens
        sanitized = sanitized.strip('-')
        return sanitized or "default"

    def _detect_from_containers(self) -> Optional[Tuple[str, Path]]:
        """Detect project from running Docker containers"""
        try:
            containers = self.docker_client.containers.list(all=True)

            # Look for indexer containers (they indicate active projects)
            for container in containers:
                if container.name.startswith("indexer-") and container.status == "running":
                    # Extract project name from container name
                    project_name = container.name.replace("indexer-", "")

                    # Try to get mount path from container
                    mounts = container.attrs.get("Mounts", [])
                    for mount in mounts:
                        if mount.get("Destination") == "/workspace":
                            source = mount.get("Source")
                            if source:
                                project_path = Path(source)
                                if project_path.exists():
                                    return project_name, project_path

                    # If no mount found, check registry
                    registry_path = Path.home() / ".claude" / "project_registry.json"
                    if registry_path.exists():
                        try:
                            with open(registry_path) as f:
                                data = json.load(f)
                                if project_name in data.get("projects", {}):
                                    project_path = Path(data["projects"][project_name])
                                    if project_path.exists():
                                        return project_name, project_path
                        except:
                            pass
        except Exception as e:
            logger.error(f"Container detection failed: {e}")

        return None


# Global detector instance
project_detector = ProjectDetector()


async def get_user_project() -> Tuple[Optional[str], Optional[Path]]:
    """
    Get the user's actual project, not the MCP server's directory

    Returns:
        Tuple of (project_name, project_path) or (None, None) if detection fails
    """
    return await project_detector.detect_user_project()