#!/usr/bin/env python3
"""
Ensure Containers Ready
Helper function for tools that need containers to be ready

Author: L9 Engineering Team
Date: September 24, 2025
"""

import logging
from typing import Optional
from .container_status import get_container_status

logger = logging.getLogger(__name__)


async def ensure_containers_ready(timeout: float = 10) -> bool:
    """
    Ensure containers are ready before proceeding.

    This is called by tools that need containers. It will:
    1. Return immediately if containers are already ready
    2. Wait up to 'timeout' seconds if still initializing
    3. Return False if initialization failed or timed out

    Args:
        timeout: Maximum seconds to wait (default 10)

    Returns:
        True if containers are ready, False otherwise
    """
    status = get_container_status()

    # If already ready, return immediately
    if status.ready:
        return True

    # If initialization hasn't started, this is on-demand mode
    if status.initialization_started is None:
        logger.info("Containers not initialized proactively, starting on-demand...")
        # Could trigger on-demand startup here if needed
        return False

    # If failed, don't wait
    if status.initialization_error:
        logger.warning(f"Container initialization failed: {status.initialization_error}")
        return False

    # Wait for initialization to complete
    logger.info(f"Waiting up to {timeout}s for containers to be ready...")
    ready = await status.wait_ready(timeout=timeout)

    if ready:
        logger.info("✅ Containers are ready")
    else:
        logger.warning(f"⚠️ Containers not ready after {timeout}s")

    return ready