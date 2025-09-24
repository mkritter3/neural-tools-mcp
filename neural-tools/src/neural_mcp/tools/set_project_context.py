#!/usr/bin/env python3
"""
Set Project Context MCP Tool
Implements ADR-0033 and ADR-0099 for explicit project setting

This tool solves the critical problem where global MCP servers cannot
detect which project the user is working on, causing all projects to be
incorrectly identified as "claude-l9-template".

Author: L9 Engineering Team
Date: 2025-09-24
"""

from typing import Dict, Any
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Tool configuration for MCP auto-discovery
TOOL_CONFIG = {
    "name": "set_project_context",
    "description": """Set or auto-detect the active project for indexing and search.
Use this when neural-tools reports the wrong project name.
Can explicitly set a path or auto-detect from environment.""",
    "inputSchema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to project directory (optional - will auto-detect if not provided)"
            }
        },
        "required": []  # Path is optional - can auto-detect
    }
}


async def execute(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set the project context explicitly or auto-detect from environment

    This solves the problem where MCP cannot detect which project
    you're working on when running from global location.

    Args:
        arguments: Dict with optional 'path' key

    Returns:
        Dict with project info and status
    """
    from servers.services.project_context_manager import get_project_context_manager

    try:
        # Get the singleton manager
        manager = await get_project_context_manager()

        # Check if path was provided
        project_path = arguments.get("path")

        if project_path:
            # Explicit path provided
            logger.info(f"🎯 Setting explicit project path: {project_path}")
            result = await manager.set_project(project_path)
            result["source"] = "explicit"

        else:
            # Auto-detect from environment or current context

            # First check CLAUDE_PROJECT_DIR (highest priority)
            if claude_dir := os.getenv("CLAUDE_PROJECT_DIR"):
                logger.info(f"🎯 Using CLAUDE_PROJECT_DIR: {claude_dir}")
                result = await manager.set_project(claude_dir)
                result["source"] = "CLAUDE_PROJECT_DIR"

            # Check legacy PROJECT_PATH
            elif project_env := os.getenv("PROJECT_PATH"):
                logger.info(f"📍 Using PROJECT_PATH: {project_env}")
                result = await manager.set_project(project_env)
                result["source"] = "PROJECT_PATH"

            # Try to detect from current directory (if not in MCP location)
            else:
                current_dir = Path.cwd()
                if "mcp-servers" not in str(current_dir) and ".claude" not in str(current_dir):
                    logger.info(f"📂 Using current directory: {current_dir}")
                    result = await manager.set_project(str(current_dir))
                    result["source"] = "current_directory"
                else:
                    # Can't auto-detect - need explicit path
                    return {
                        "error": "Cannot auto-detect project from MCP server location",
                        "message": "Please provide explicit path or set CLAUDE_PROJECT_DIR environment variable",
                        "current_location": str(current_dir),
                        "suggestion": "Use: set_project_context with path parameter"
                    }

        # Ensure indexer is running for this project
        if manager.container:
            try:
                await manager.container.ensure_indexer_running()
                result["indexer"] = "started"
            except Exception as e:
                logger.warning(f"Could not start indexer: {e}")
                result["indexer"] = f"failed: {e}"

        logger.info(f"✅ Project context set: {result.get('project')} at {result.get('path')}")
        return result

    except ValueError as e:
        return {
            "error": "Invalid path",
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Failed to set project context: {e}")
        return {
            "error": "Failed to set project context",
            "message": str(e)
        }


# Helper function to list known projects
async def list_projects() -> Dict[str, Any]:
    """
    List all known projects from registry

    Returns:
        Dict with list of projects and their paths
    """
    from servers.services.project_context_manager import get_project_context_manager

    manager = await get_project_context_manager()

    projects = []
    for name, path in manager.project_registry.items():
        last_activity = manager.last_activity.get(name)
        projects.append({
            "name": name,
            "path": str(path),
            "last_activity": last_activity.isoformat() if last_activity else None,
            "is_current": name == manager.current_project
        })

    return {
        "projects": projects,
        "current": manager.current_project,
        "total": len(projects)
    }