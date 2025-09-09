#!/usr/bin/env python3
"""
Simple Caching Layer for Neural Tools
Provides transparent caching with TTL support and hit/miss metrics
Following the roadmap Phase 1.2 specifications
"""

import asyncio
import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Optional, Callable, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SimpleCache:
    """
    In-memory cache with TTL support and basic eviction
    Thread-safe and async-compatible
    """
    
    def __init__(self, ttl: int = 300, max_size: int = 1000):
        """
        Initialize cache
        
        Args:
            ttl: Time to live in seconds (default 5 minutes)
            max_size: Maximum number of entries before eviction
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self._access_times: Dict[str, float] = {}  # LRU tracking
        self._lock = asyncio.Lock()
        
        # Metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_entries': 0,
            'total_gets': 0,
            'total_sets': 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            self.metrics['total_gets'] += 1
            current_time = time.time()
            
            if key in self._cache:
                value, expiry_time = self._cache[key]
                
                if current_time < expiry_time:
                    # Cache hit
                    self._access_times[key] = current_time
                    self.metrics['hits'] += 1
                    return value
                else:
                    # Expired entry
                    self._remove_key(key)
                    self.metrics['expired_entries'] += 1
            
            # Cache miss
            self.metrics['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL"""
        async with self._lock:
            self.metrics['total_sets'] += 1
            current_time = time.time()
            expiry_time = current_time + self.ttl
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict_lru()
            
            self._cache[key] = (value, expiry_time)
            self._access_times[key] = current_time
    
    async def get_or_compute(self, key: str, compute_func: Callable) -> Any:
        """Get value from cache or compute if not present"""
        # First try to get from cache
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        # Cache miss - compute value
        try:
            if asyncio.iscoroutinefunction(compute_func):
                computed_value = await compute_func()
            else:
                computed_value = compute_func()
            
            # Store in cache
            await self.set(key, computed_value)
            return computed_value
        except Exception as e:
            # Don't cache exceptions
            logger.error(f"Failed to compute value for key {key}: {e}")
            raise
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                self._remove_key(key)
                return True
            return False
    
    async def clear_async(self) -> None:
        """Clear all cache entries (async version)"""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            # Reset metrics when clearing
            for key in self.metrics:
                self.metrics[key] = 0
    
    def clear(self) -> None:
        """Synchronous clear all cache entries"""
        self._cache.clear()
        self._access_times.clear()
        # Reset metrics when clearing
        for key in self.metrics:
            self.metrics[key] = 0
    
    def delete(self, key: str) -> bool:
        """Synchronous delete key from cache"""
        if key in self._cache:
            self._remove_key(key)
            return True
        return False
    
    def _remove_key(self, key: str) -> None:
        """Remove key from both cache and access tracking"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._access_times:
            return
        
        # Find LRU key
        lru_key = min(self._access_times, key=self._access_times.get)
        self._remove_key(lru_key)
        self.metrics['evictions'] += 1
        
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries and return count removed"""
        async with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, expiry_time) in self._cache.items():
                if current_time >= expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_key(key)
            
            if expired_keys:
                self.metrics['expired_entries'] += len(expired_keys)
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.metrics['hits'] + self.metrics['misses']
        hit_rate = (self.metrics['hits'] / max(total_requests, 1)) * 100
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_rate_percent': hit_rate,
            'ttl_seconds': self.ttl,
            **self.metrics
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics (alias for get_stats with additional metrics)"""
        total_requests = self.metrics['hits'] + self.metrics['misses']
        hit_rate = self.metrics['hits'] / max(total_requests, 1)
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.metrics['hits'],
            'misses': self.metrics['misses'],
            'hit_rate': hit_rate,
            'evictions': self.metrics['evictions'],
            'expired_entries': self.metrics['expired_entries'],
            'total_gets': self.metrics['total_gets'],
            'total_sets': self.metrics['total_sets']
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters"""
        for key in self.metrics:
            self.metrics[key] = 0

# Global cache instance
_global_cache = SimpleCache(ttl=300, max_size=1000)

def cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key from args/kwargs
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                if args:
                    key_parts.extend(str(arg) for arg in args)
                if kwargs:
                    sorted_kwargs = sorted(kwargs.items())
                    key_parts.extend(f"{k}={v}" for k, v in sorted_kwargs)
                
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = await _global_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache HIT for {func.__name__}: {cache_key}")
                return cached_result
            
            # Compute result
            logger.debug(f"Cache MISS for {func.__name__}: {cache_key}")
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Store in cache
            await _global_cache.set(cache_key, result)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we still need to run cache operations in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop, run without caching
                logger.warning(f"No event loop for caching {func.__name__}, running uncached")
                return func(*args, **kwargs)
            
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class CachedHybridRetriever:
    """
    Wrapper around HybridRetriever that adds caching
    Follows the roadmap specification for transparent caching
    """
    
    def __init__(self, hybrid_retriever, cache_ttl: int = 300):
        """
        Initialize cached retriever
        
        Args:
            hybrid_retriever: Instance of HybridRetriever to wrap
            cache_ttl: Cache TTL in seconds
        """
        self.retriever = hybrid_retriever
        self.cache = SimpleCache(ttl=cache_ttl, max_size=500)  # Smaller cache for search results
    
    def _generate_search_key(self, query: str, limit: int, include_graph_context: bool, max_hops: int) -> str:
        """Generate cache key for search query"""
        key_data = {
            'query': query.strip().lower(),
            'limit': limit,
            'include_graph_context': include_graph_context,
            'max_hops': max_hops,
            'project': self.retriever.container.project_name
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def find_similar_with_context(
        self,
        query: str,
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Cached version of find_similar_with_context
        Transparently caches results for identical queries
        """
        cache_key = self._generate_search_key(query, limit, include_graph_context, max_hops)
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache HIT for search query: {query[:50]}...")
            return cached_result
        
        # Cache miss - compute result
        logger.debug(f"Cache MISS for search query: {query[:50]}...")
        start_time = time.time()
        
        result = await self.retriever.find_similar_with_context(
            query=query,
            limit=limit,
            include_graph_context=include_graph_context,
            max_hops=max_hops
        )
        
        compute_time = (time.time() - start_time) * 1000
        logger.info(f"Search computed in {compute_time:.1f}ms, caching result")
        
        # Store in cache
        await self.cache.set(cache_key, result)
        
        return result
    
    async def invalidate_file_cache(self, file_path: str) -> int:
        """
        Invalidate cache entries that might be affected by file changes
        Returns number of entries invalidated
        """
        # For now, clear entire cache on any file change
        # In production, this would be more sophisticated
        stats_before = self.cache.get_stats()
        await self.cache.clear()
        
        entries_cleared = stats_before['size']
        if entries_cleared > 0:
            logger.info(f"Invalidated {entries_cleared} cache entries due to file change: {file_path}")
        
        return entries_cleared
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

# Background cache cleanup task
async def start_cache_cleanup_task(cache: SimpleCache, interval: int = 300):
    """
    Background task to clean up expired cache entries
    
    Args:
        cache: Cache instance to clean
        interval: Cleanup interval in seconds (default 5 minutes)
    """
    logger.info(f"Starting cache cleanup task (interval: {interval}s)")
    
    while True:
        try:
            await asyncio.sleep(interval)
            removed_count = await cache.cleanup_expired()
            
            if removed_count > 0:
                logger.debug(f"Cache cleanup removed {removed_count} expired entries")
                
        except asyncio.CancelledError:
            logger.info("Cache cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Cache cleanup task error: {e}")

# Cache key generation utilities
def query_cache_key(query: str, **params) -> str:
    """Generate cache key for search queries"""
    key_data = {'query': query.strip().lower(), **params}
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

def file_cache_key(file_path: str, **params) -> str:
    """Generate cache key for file-based operations"""
    key_data = {'file_path': str(file_path), **params}
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()

# Example usage and testing
async def test_cache():
    """Test the caching system"""
    print("🧪 Testing caching system...")
    
    cache = SimpleCache(ttl=2, max_size=3)  # Short TTL for testing
    
    # Test basic operations
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    
    # Test hits
    result1 = await cache.get("key1")
    result2 = await cache.get("key2")
    print(f"   ✅ Cache hits: {result1}, {result2}")
    
    # Test miss
    result3 = await cache.get("key3")
    print(f"   ❌ Cache miss: {result3}")
    
    # Test eviction
    await cache.set("key3", "value3")
    await cache.set("key4", "value4")  # Should trigger eviction
    
    # Test expiration
    print("   ⏱️  Waiting for expiration...")
    await asyncio.sleep(3)
    
    expired = await cache.get("key1")
    print(f"   ⏰ Expired entry: {expired}")
    
    # Test cleanup
    removed = await cache.cleanup_expired()
    print(f"   🧹 Cleanup removed: {removed} entries")
    
    # Print stats
    stats = cache.get_stats()
    print(f"   📊 Stats: {stats}")
    print("✅ Cache test completed")

async def test_cached_decorator():
    """Test the @cached decorator"""
    print("🎯 Testing @cached decorator...")
    
    call_count = 0
    
    @cached(ttl=5)
    async def expensive_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # Simulate work
        return x * 2
    
    # First call - should compute
    start_time = time.time()
    result1 = await expensive_function(5)
    time1 = (time.time() - start_time) * 1000
    print(f"   🔥 First call: {result1} (took {time1:.1f}ms, calls: {call_count})")
    
    # Second call - should be cached
    start_time = time.time()
    result2 = await expensive_function(5)
    time2 = (time.time() - start_time) * 1000
    print(f"   ⚡ Second call: {result2} (took {time2:.1f}ms, calls: {call_count})")
    
    # Should be much faster and same call count
    assert time2 < time1 / 2, "Second call should be much faster"
    assert call_count == 1, "Function should only be called once"
    
    print("✅ Decorator test passed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def run_tests():
        await test_cache()
        await test_cached_decorator()
    
    asyncio.run(run_tests())