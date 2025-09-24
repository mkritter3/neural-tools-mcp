#!/usr/bin/env python3
"""
List Projects MCP Tool
Companion to set_project_context - shows all known projects

Author: L9 Engineering Team
Date: 2025-09-24
"""

from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Tool configuration for MCP auto-discovery
TOOL_CONFIG = {
    "name": "list_projects",
    "description": "List all known projects and show which is currently active",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "required": []
    }
}


async def execute(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all known projects from registry

    Returns:
        Dict with list of projects and their paths
    """
    from servers.services.project_context_manager import get_project_context_manager

    try:
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

        # Sort by last activity (most recent first)
        projects.sort(key=lambda p: p["last_activity"] or "", reverse=True)

        result = {
            "projects": projects,
            "current_project": manager.current_project,
            "current_path": str(manager.current_project_path) if manager.current_project_path else None,
            "total": len(projects)
        }

        # Add detection info
        import os
        if claude_dir := os.getenv("CLAUDE_PROJECT_DIR"):
            result["claude_project_dir"] = claude_dir
            result["detection_mode"] = "environment"
        else:
            result["detection_mode"] = "registry"

        return result

    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        return {
            "error": "Failed to list projects",
            "message": str(e)
        }