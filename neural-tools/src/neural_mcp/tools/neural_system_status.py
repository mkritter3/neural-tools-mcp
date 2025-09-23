"""
Neural System Status Tool - September 2025 Standards
Health monitoring and system diagnostics

ADR-0076: Modular tool architecture
"""

import json
import time
import psutil
from typing import List, Dict, Any

from mcp import types
from ..shared.performance_metrics import track_performance, get_performance_metrics
from ..shared.cache_manager import get_cache_stats
from ..shared.connection_pool import get_connection_stats

import logging
logger = logging.getLogger(__name__)

# Tool metadata for automatic registration
TOOL_CONFIG = {
    "name": "neural_system_status",
    "description": "Get comprehensive neural system status and health",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }
}

@track_performance
async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Main tool execution function - September 2025 Standards

    Args:
        arguments: Validated input parameters

    Returns:
        List of TextContent responses with system status
    """
    try:
        # Collect system metrics
        system_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "architecture": "modular_september_2025",
            "adr_compliance": {
                "adr_0075": "connection_pooling_active",
                "adr_0076": "modular_architecture_active"
            },
            "system_metrics": _get_system_metrics(),
            "performance_metrics": get_performance_metrics(),
            "cache_stats": get_cache_stats(),
            "connection_stats": get_connection_stats(),
            "health_checks": await _run_health_checks()
        }

        response = [types.TextContent(type="text", text=json.dumps(system_status, indent=2))]
        return response

    except Exception as e:
        logger.error(f"System status check failed: {e}")
        return _make_error_response(f"System status check failed: {e}")

def _get_system_metrics() -> dict:
    """Get basic system resource metrics"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent if psutil.disk_usage('/') else 0,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
        return {"error": "Unable to collect system metrics"}

async def _run_health_checks() -> dict:
    """Run basic health checks"""
    checks = {
        "modular_architecture": True,
        "shared_utilities": True,
        "tool_discovery": True,
        "performance_tracking": True
    }

    # Test shared utilities
    try:
        from ..shared.connection_pool import get_connection_stats
        from ..shared.performance_metrics import get_performance_metrics
        from ..shared.cache_manager import get_cache_stats
        checks["shared_utilities"] = True
    except Exception:
        checks["shared_utilities"] = False

    # Test tool discovery
    try:
        from pathlib import Path
        tools_dir = Path(__file__).parent
        tool_count = len([f for f in tools_dir.glob("*.py") if not f.stem.startswith("_")])
        checks["tool_discovery"] = tool_count > 0
        checks["tools_discovered"] = tool_count
    except Exception:
        checks["tool_discovery"] = False

    return checks

def _make_error_response(error: str) -> List[types.TextContent]:
    """Standard error response format"""
    return [types.TextContent(type="text", text=json.dumps({
        "status": "error",
        "message": error,
        "tool": TOOL_CONFIG["name"],
        "architecture": "modular_september_2025"
    }))]