#!/usr/bin/env python3
"""
Shared Orchestrator Instance
Ensures only one IndexerOrchestrator exists to prevent conflicts

Author: L9 Engineering Team
Date: September 24, 2025
"""

import os
import logging
from typing import Optional
from pathlib import Path
from .indexer_orchestrator import IndexerOrchestrator

logger = logging.getLogger(__name__)

# Global singleton instance
_shared_indexer_orchestrator: Optional[IndexerOrchestrator] = None
# Cache the resolved project path to ensure consistency
_resolved_project_path: Optional[str] = None


def get_consistent_project_path() -> str:
    """
    Get consistent project path across all components.
    Priority:
    1. CLAUDE_PROJECT_DIR (set by Claude Code)
    2. Cached resolved path (from first resolution)
    3. Auto-detect from markers (.git, package.json, etc.)

    This ensures ContainerOrchestrator and ServiceContainer
    use the same path, preventing mount mismatches.
    """
    global _resolved_project_path

    # If we've already resolved it, use cached value for consistency
    if _resolved_project_path:
        return _resolved_project_path

    # Priority 1: CLAUDE_PROJECT_DIR (most reliable)
    claude_dir = os.getenv("CLAUDE_PROJECT_DIR")
    if claude_dir and os.path.exists(claude_dir):
        _resolved_project_path = claude_dir
        logger.info(f"Using CLAUDE_PROJECT_DIR for consistency: {claude_dir}")
        return claude_dir

    # Priority 2: Auto-detect from markers
    current = os.path.abspath(os.getcwd())
    markers = [".git", "pyproject.toml", "package.json", "requirements.txt", "setup.py", ".claude"]

    while current != "/":
        for marker in markers:
            if os.path.exists(os.path.join(current, marker)):
                _resolved_project_path = current
                logger.info(f"Auto-detected project root at {current} (marker: {marker})")
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # Fallback to cwd
    _resolved_project_path = os.getcwd()
    logger.warning(f"No project markers found, using cwd: {_resolved_project_path}")
    return _resolved_project_path


async def get_shared_indexer_orchestrator(context_manager=None) -> IndexerOrchestrator:
    """
    Get or create the shared IndexerOrchestrator instance.

    This ensures only one orchestrator exists across:
    - ContainerOrchestrator (proactive startup)
    - ServiceContainer (tool requests)

    Prevents container termination due to conflicting orchestrators.
    """
    global _shared_indexer_orchestrator

    if _shared_indexer_orchestrator is None:
        logger.info("Creating shared IndexerOrchestrator instance")

        if context_manager:
            _shared_indexer_orchestrator = IndexerOrchestrator(
                context_manager=context_manager
            )
        else:
            _shared_indexer_orchestrator = IndexerOrchestrator()

        await _shared_indexer_orchestrator.initialize()
        logger.info("✅ Shared IndexerOrchestrator initialized")
    else:
        logger.debug("Reusing existing IndexerOrchestrator instance")

    return _shared_indexer_orchestrator


def reset_shared_orchestrator():
    """Reset the shared orchestrator (mainly for testing)"""
    global _shared_indexer_orchestrator
    _shared_indexer_orchestrator = None
    logger.debug("Shared IndexerOrchestrator reset")