#!/usr/bin/env python3
"""
Container Status Tracker
Allows tools to check if containers are ready without blocking

Author: L9 Engineering Team
Date: September 24, 2025
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ContainerStatus:
    """Singleton to track container initialization status"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.orchestrator = None
            self.initialization_started = None
            self.initialization_completed = None
            self.initialization_error = None
            self.ready = False
            self._ready_event = asyncio.Event()
            self._initialized = True

    def set_orchestrator(self, orchestrator):
        """Store reference to orchestrator once initialized"""
        self.orchestrator = orchestrator
        self.ready = True
        self.initialization_completed = datetime.now()
        self._ready_event.set()
        logger.info("✅ Container status: READY")

    def set_error(self, error: Exception):
        """Mark initialization as failed"""
        self.initialization_error = str(error)
        self.ready = False
        logger.error(f"❌ Container status: FAILED - {error}")

    async def wait_ready(self, timeout: float = 30) -> bool:
        """
        Wait for containers to be ready.
        Returns True if ready, False if timeout.
        """
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current initialization status"""
        status = {
            "ready": self.ready,
            "started_at": self.initialization_started.isoformat() if self.initialization_started else None,
            "completed_at": self.initialization_completed.isoformat() if self.initialization_completed else None,
            "error": self.initialization_error
        }

        if self.initialization_started and self.initialization_completed:
            duration = (self.initialization_completed - self.initialization_started).total_seconds()
            status["initialization_time_seconds"] = duration

        return status


# Global instance
_container_status = ContainerStatus()


def get_container_status() -> ContainerStatus:
    """Get the singleton container status tracker"""
    return _container_status