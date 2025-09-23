"""
Performance Metrics - September 2025 Standards
Track tool performance and caching efficiency
"""

import time
import logging
from functools import wraps
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global performance metrics
_performance_metrics: Dict[str, Any] = {
    "query_count": 0,
    "total_query_time": 0.0,
    "avg_query_time": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
    "service_creation_count": 0,
    "tool_executions": {},
    "errors": 0
}

def track_performance(func):
    """Decorator to track tool performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = getattr(func, '__module__', 'unknown').split('.')[-1]
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # Update metrics
            update_performance_metrics(duration_ms, tool_name)

            return result

        except Exception as e:
            _performance_metrics["errors"] += 1
            logger.error(f"Tool {tool_name} failed: {e}")
            raise

    return wrapper

def update_performance_metrics(duration_ms: float, tool_name: str = "unknown"):
    """Update performance metrics for monitoring"""
    _performance_metrics["query_count"] += 1
    _performance_metrics["total_query_time"] += duration_ms
    _performance_metrics["avg_query_time"] = (
        _performance_metrics["total_query_time"] / _performance_metrics["query_count"]
    )

    # Track per-tool metrics
    if tool_name not in _performance_metrics["tool_executions"]:
        _performance_metrics["tool_executions"][tool_name] = {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0
        }

    tool_metrics = _performance_metrics["tool_executions"][tool_name]
    tool_metrics["count"] += 1
    tool_metrics["total_time"] += duration_ms
    tool_metrics["avg_time"] = tool_metrics["total_time"] / tool_metrics["count"]

def increment_cache_hit():
    """Increment cache hit counter"""
    _performance_metrics["cache_hits"] += 1

def increment_cache_miss():
    """Increment cache miss counter"""
    _performance_metrics["cache_misses"] += 1

def get_performance_metrics() -> dict:
    """Get current performance metrics"""
    cache_total = _performance_metrics["cache_hits"] + _performance_metrics["cache_misses"]
    cache_hit_rate = (_performance_metrics["cache_hits"] / cache_total * 100) if cache_total > 0 else 0

    return {
        **_performance_metrics,
        "cache_hit_rate_percent": round(cache_hit_rate, 2)
    }

def reset_metrics():
    """Reset all performance metrics"""
    global _performance_metrics
    _performance_metrics = {
        "query_count": 0,
        "total_query_time": 0.0,
        "avg_query_time": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "service_creation_count": 0,
        "tool_executions": {},
        "errors": 0
    }