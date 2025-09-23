"""
Cache Manager - September 2025 Standards
Simple in-memory caching with TTL support
"""

import time
import logging
from typing import Any, Optional, Dict
from .performance_metrics import increment_cache_hit, increment_cache_miss

logger = logging.getLogger(__name__)

# Global cache store
_cache_store: Dict[str, Dict[str, Any]] = {}
_default_ttl = 3600  # 1 hour default TTL

def cache_result(cache_key: str, result: Any, ttl: int = None):
    """Cache result with optional TTL"""
    if ttl is None:
        ttl = _default_ttl

    _cache_store[cache_key] = {
        'data': result,
        'timestamp': time.time(),
        'ttl': ttl
    }

    # Simple cache cleanup - keep only last 100 entries
    if len(_cache_store) > 100:
        # Remove oldest entries
        oldest_keys = sorted(_cache_store.keys(),
                           key=lambda k: _cache_store[k]['timestamp'])[:20]
        for key in oldest_keys:
            del _cache_store[key]

def get_cached_result(cache_key: str) -> Optional[Any]:
    """Get cached result if available and not expired"""
    if cache_key not in _cache_store:
        increment_cache_miss()
        return None

    cached = _cache_store[cache_key]
    current_time = time.time()

    # Check if expired
    if current_time - cached['timestamp'] > cached['ttl']:
        del _cache_store[cache_key]
        increment_cache_miss()
        return None

    increment_cache_hit()
    return cached['data']

def invalidate_cache(pattern: str = None):
    """Invalidate cache entries matching pattern"""
    if pattern is None:
        _cache_store.clear()
        logger.info("Cache cleared completely")
        return

    keys_to_remove = [k for k in _cache_store.keys() if pattern in k]
    for key in keys_to_remove:
        del _cache_store[key]

    logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching '{pattern}'")

def get_cache_stats() -> dict:
    """Get cache statistics"""
    current_time = time.time()
    expired_count = sum(1 for cached in _cache_store.values()
                       if current_time - cached['timestamp'] > cached['ttl'])

    return {
        'total_entries': len(_cache_store),
        'expired_entries': expired_count,
        'active_entries': len(_cache_store) - expired_count,
        'memory_usage_kb': sum(len(str(cached['data'])) for cached in _cache_store.values()) / 1024
    }

def cleanup_expired_cache():
    """Remove expired cache entries"""
    current_time = time.time()
    expired_keys = [k for k, cached in _cache_store.items()
                   if current_time - cached['timestamp'] > cached['ttl']]

    for key in expired_keys:
        del _cache_store[key]

    if expired_keys:
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")