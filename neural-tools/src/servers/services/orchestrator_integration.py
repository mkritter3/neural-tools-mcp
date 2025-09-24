#!/usr/bin/env python3
"""
Integration Helper for Project Orchestrator - ADR-0097
INTEGRATES WITH EXISTING ProjectContextManager and IndexerOrchestrator

This module shows how ADR-97 integrates with the existing system:
- Uses existing ProjectContextManager for project detection
- Extends existing IndexerOrchestrator for multi-project support
- No parallel or duplicate implementations
"""

import os
import logging
from typing import Dict, Any, Optional

from .project_context_manager import get_project_context_manager
from .indexer_orchestrator import IndexerOrchestrator
from .service_container import ServiceContainer

logger = logging.getLogger(__name__)

# Global orchestrator instance (singleton)
_orchestrator: Optional[IndexerOrchestrator] = None


async def get_orchestrator() -> IndexerOrchestrator:
    """
    Get or create the global orchestrator instance.
    INTEGRATES WITH EXISTING SYSTEM - not a parallel implementation!
    """
    global _orchestrator

    if _orchestrator is None:
        # Get the existing ProjectContextManager singleton
        context_manager = await get_project_context_manager()

        # Create IndexerOrchestrator with injected context manager (ADR-0044 pattern)
        _orchestrator = IndexerOrchestrator(context_manager=context_manager)

        # Initialize the orchestrator
        await _orchestrator.initialize()

        # Check if multi-project mode is enabled
        if os.getenv("MCP_MULTI_PROJECT_MODE", "false").lower() == "true":
            logger.info("✅ IndexerOrchestrator initialized in multi-project mode")
        else:
            logger.info("✅ IndexerOrchestrator initialized in legacy mode")

    return _orchestrator


async def ensure_indexer_for_project() -> Dict[str, Any]:
    """
    Ensure indexer is running for the current project.
    USES EXISTING ProjectContextManager for project detection!

    Returns:
        Dict with indexer info: {"port": int, "status": str, "container_id": str}
    """
    # Get the existing ProjectContextManager
    context_manager = await get_project_context_manager()

    # Use $CLAUDE_PROJECT_DIR if available, otherwise use context manager's detection
    project_dir = os.getenv("CLAUDE_PROJECT_DIR")
    if project_dir:
        # Update context manager with Claude's project directory
        await context_manager.set_project(project_dir)
    else:
        # Use context manager's current project
        project_info = await context_manager.get_current_project()
        project_dir = project_info.get("path")
        if not project_dir:
            # Last fallback
            project_dir = os.getcwd()
            logger.warning(f"No project context, using cwd: {project_dir}")

    # Get orchestrator (which already has the context manager)
    orchestrator = await get_orchestrator()

    # The existing IndexerOrchestrator handles container lifecycle
    # We just need to ensure it's using the right project
    service_container = ServiceContainer(context_manager=context_manager)
    await service_container.initialize()

    # Use the existing ensure_indexer_running method
    indexer_info = await service_container.ensure_indexer_running(project_dir)

    logger.info(f"📍 Indexer ready for project at {project_dir}")
    logger.info(f"   Port: {indexer_info['port']}, Status: {indexer_info['status']}")

    return indexer_info


def get_indexer_url(port: int) -> str:
    """
    Get the URL for the indexer service.

    Args:
        port: The port number where indexer is running

    Returns:
        Full URL for indexer API
    """
    # Use localhost for host-based MCP
    # If MCP moves to container, this would use host.docker.internal
    host = "localhost"
    return f"http://{host}:{port}"


# Example integration for MCP tool
async def example_mcp_tool_integration(arguments: dict):
    """
    Example showing how to integrate orchestrator into an MCP tool.

    This pattern should be used in each MCP tool's execute() function.
    """
    try:
        # Step 1: Ensure indexer is running for current project
        indexer_info = await ensure_indexer_for_project()

        # Step 2: Get indexer URL
        indexer_url = get_indexer_url(indexer_info['port'])

        # Step 3: Use indexer for tool operation
        # ... your tool logic here ...
        logger.info(f"Using indexer at {indexer_url}")

        # Step 4: Return tool response
        return {"status": "success", "indexer_url": indexer_url}

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return {"status": "error", "message": str(e)}


# Cleanup function for graceful shutdown
async def cleanup_orchestrator():
    """Clean up orchestrator resources on shutdown"""
    global _orchestrator

    if _orchestrator:
        logger.info("Cleaning up orchestrator...")
        await _orchestrator.shutdown()
        _orchestrator = None


# Integration template for existing MCP tools
INTEGRATION_TEMPLATE = """
# Add to the top of your MCP tool's execute() function:

from servers.services.orchestrator_integration import ensure_indexer_for_project, get_indexer_url

async def execute(arguments: dict) -> List[types.TextContent]:
    '''Your existing tool execute function'''

    # Ensure indexer is running for current project
    indexer_info = await ensure_indexer_for_project()
    indexer_url = get_indexer_url(indexer_info['port'])

    # Now use indexer_url instead of hardcoded URL
    # Example: response = await httpx.get(f"{indexer_url}/api/search")

    # ... rest of your tool logic ...
"""

if __name__ == "__main__":
    # Test the integration
    import asyncio

    async def test():
        """Test orchestrator integration"""
        # Set test environment
        os.environ["CLAUDE_PROJECT_DIR"] = "/tmp/test-project"

        # Test getting indexer
        info = await ensure_indexer_for_project()
        print(f"Indexer info: {info}")

        # Test example integration
        result = await example_mcp_tool_integration({})
        print(f"Integration result: {result}")

        # Cleanup
        await cleanup_orchestrator()

    asyncio.run(test())