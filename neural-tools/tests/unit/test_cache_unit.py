#!/usr/bin/env python3
"""
Phase 1.2 Unit Tests - Simple Caching Layer
Tests TTL-based caching with hit/miss metrics and LRU eviction
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock
import weakref
import gc

import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.simple_cache import SimpleCache, cached

class TestSimpleCache:
    """Test suite for Simple Cache implementation"""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing"""
        return SimpleCache(ttl=60, max_size=100)
    
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_value(self, cache):
        """Test that cache hit returns cached value without recomputation"""
        # First call - compute
        compute_called = False
        
        async def compute():
            nonlocal compute_called
            compute_called = True
            return {"result": "data"}
        
        result1 = await cache.get_or_compute("key1", compute)
        assert compute_called
        assert result1 == {"result": "data"}
        
        # Second call - from cache
        compute_called = False
        result2 = await cache.get_or_compute("key1", compute)
        assert not compute_called  # Should not recompute
        assert result2 == {"result": "data"}
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL"""
        cache = SimpleCache(ttl=0.1)  # 100ms TTL
        
        compute_count = 0
        async def compute():
            nonlocal compute_count
            compute_count += 1
            return {"v": compute_count}
        
        result1 = await cache.get_or_compute("key", compute)
        assert result1 == {"v": 1}
        assert compute_count == 1
        
        await asyncio.sleep(0.2)  # Wait for expiration
        
        # Should recompute after TTL
        result2 = await cache.get_or_compute("key", compute)
        assert result2 == {"v": 2}  # New value, not cached
        assert compute_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_miss_computes_value(self, cache):
        """Test that cache miss triggers computation"""
        async def compute():
            return {"computed": True}
        
        # Should compute on first access
        result = await cache.get_or_compute("new_key", compute)
        assert result == {"computed": True}
        
        # Verify it was cached
        cached_result = await cache.get_or_compute("new_key", AsyncMock())
        assert cached_result == {"computed": True}
    
    @pytest.mark.asyncio
    async def test_cache_metrics_tracking(self, cache):
        """Test that cache tracks hit/miss metrics"""
        async def compute(value):
            return {"value": value}
        
        # Generate some cache hits and misses
        await cache.get_or_compute("key1", lambda: compute("a"))  # Miss
        await cache.get_or_compute("key2", lambda: compute("b"))  # Miss
        await cache.get_or_compute("key1", lambda: compute("c"))  # Hit
        await cache.get_or_compute("key2", lambda: compute("d"))  # Hit
        await cache.get_or_compute("key3", lambda: compute("e"))  # Miss
        
        metrics = cache.get_metrics()
        assert metrics["hits"] == 2
        assert metrics["misses"] == 3
        assert metrics["hit_rate"] == 2/5  # 40%
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = SimpleCache(max_size=3, ttl=60)  # Small cache
        
        # Fill cache
        await cache.get_or_compute("key1", lambda: {"v": 1})
        await cache.get_or_compute("key2", lambda: {"v": 2})
        await cache.get_or_compute("key3", lambda: {"v": 3})
        
        # Access key1 to make it recently used
        await cache.get_or_compute("key1", lambda: {"v": "should not compute"})
        
        # Add new key - should evict key2 (least recently used)
        await cache.get_or_compute("key4", lambda: {"v": 4})
        
        # Verify key2 was evicted
        compute_called = False
        async def check_compute():
            nonlocal compute_called
            compute_called = True
            return {"v": "recomputed"}
        
        await cache.get_or_compute("key2", check_compute)
        assert compute_called  # Should have been evicted and recomputed
        
        # key1 should still be cached
        compute_called = False
        await cache.get_or_compute("key1", check_compute)
        assert not compute_called  # Should still be cached
    
    @pytest.mark.asyncio
    async def test_concurrent_access_same_key(self, cache):
        """Test concurrent access to same key doesn't duplicate computation"""
        compute_count = 0
        computation_event = asyncio.Event()
        
        async def slow_compute():
            nonlocal compute_count
            compute_count += 1
            await computation_event.wait()  # Wait for signal
            return {"computed": compute_count}
        
        # Start multiple concurrent computations
        tasks = [
            asyncio.create_task(cache.get_or_compute("concurrent_key", slow_compute))
            for _ in range(5)
        ]
        
        await asyncio.sleep(0.1)  # Let all tasks start
        computation_event.set()  # Allow computation to complete
        
        results = await asyncio.gather(*tasks)
        
        # Should only compute once despite concurrent access
        assert compute_count == 1
        assert all(result == {"computed": 1} for result in results)
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, cache):
        """Test cache clear functionality"""
        await cache.get_or_compute("key1", lambda: {"v": 1})
        await cache.get_or_compute("key2", lambda: {"v": 2})
        
        # Verify items are cached
        initial_metrics = cache.get_metrics()
        assert initial_metrics["size"] == 2
        
        cache.clear()
        
        # Verify cache is empty
        metrics = cache.get_metrics()
        assert metrics["size"] == 0
        assert metrics["hits"] == 0
        assert metrics["misses"] == 0
    
    @pytest.mark.asyncio
    async def test_cache_delete_specific_key(self, cache):
        """Test deleting specific cache keys"""
        await cache.get_or_compute("key1", lambda: {"v": 1})
        await cache.get_or_compute("key2", lambda: {"v": 2})
        
        cache.delete("key1")
        
        # key1 should be deleted
        compute_called = False
        async def check_compute():
            nonlocal compute_called
            compute_called = True
            return {"v": "recomputed"}
        
        await cache.get_or_compute("key1", check_compute)
        assert compute_called
        
        # key2 should still be cached
        compute_called = False
        await cache.get_or_compute("key2", check_compute)
        assert not compute_called
    
    @pytest.mark.asyncio
    async def test_cache_hit_latency_performance(self, cache):
        """Test that cache hits are fast (< 5ms requirement)"""
        await cache.get_or_compute("perf_key", lambda: {"data": "test"})
        
        # Measure cache hit latency
        hit_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            await cache.get_or_compute("perf_key", lambda: {"should": "not_compute"})
            hit_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            hit_times.append(hit_time)
        
        avg_hit_time = sum(hit_times) / len(hit_times)
        max_hit_time = max(hit_times)
        
        print(f"Cache hit times - Avg: {avg_hit_time:.2f}ms, Max: {max_hit_time:.2f}ms")
        
        # EXIT CRITERIA 1.2.1: Cache hit latency < 5ms
        assert avg_hit_time < 5.0, f"Average hit time {avg_hit_time:.2f}ms exceeds 5ms requirement"
        assert max_hit_time < 10.0, f"Max hit time {max_hit_time:.2f}ms exceeds reasonable threshold"
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test no memory leaks after 1000 cache operations"""
        cache = SimpleCache(max_size=50, ttl=60)  # Smaller cache for testing
        
        # Track objects that should be garbage collected
        weak_refs = []
        
        # Perform 1000 cache operations
        for i in range(1000):
            key = f"key_{i % 100}"  # Reuse keys to trigger evictions
            
            # Create object with weak reference
            obj = {"data": f"value_{i}", "large_data": "x" * 1000}
            weak_refs.append(weakref.ref(obj))
            
            await cache.get_or_compute(key, lambda o=obj: o)
            
            # Force garbage collection every 100 operations
            if i % 100 == 0:
                gc.collect()
        
        # Force final garbage collection
        cache.clear()
        gc.collect()
        
        # Check that most objects were garbage collected
        alive_refs = sum(1 for ref in weak_refs if ref() is not None)
        total_refs = len(weak_refs)
        
        print(f"Memory leak test - Alive references: {alive_refs}/{total_refs}")
        
        # Should have collected most references (allow some margin for test environment)
        assert alive_refs < total_refs * 0.1, f"Too many objects still alive: {alive_refs}/{total_refs}"


class TestCachedDecorator:
    """Test the @cached decorator"""
    
    @pytest.mark.asyncio
    async def test_cached_decorator_basic(self):
        """Test basic functionality of @cached decorator"""
        call_count = 0
        
        @cached(ttl=60)
        async def expensive_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result_{arg}_{call_count}"
        
        # First call
        result1 = await expensive_function("test")
        assert call_count == 1
        assert "result_test_1" in result1
        
        # Second call - should be cached
        result2 = await expensive_function("test")
        assert call_count == 1  # Should not increment
        assert result1 == result2
        
        # Different argument - should compute
        result3 = await expensive_function("other")
        assert call_count == 2
        assert "result_other_2" in result3
    
    @pytest.mark.asyncio
    async def test_cached_decorator_ttl_expiration(self):
        """Test TTL expiration with decorator"""
        call_count = 0
        
        @cached(ttl=0.1)  # 100ms TTL
        async def fast_expire_function(arg):
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"
        
        result1 = await fast_expire_function("key")
        assert call_count == 1
        
        await asyncio.sleep(0.2)  # Wait for expiration
        
        result2 = await fast_expire_function("key")
        assert call_count == 2
        assert result1 != result2
    
    @pytest.mark.asyncio
    async def test_cached_decorator_with_exceptions(self):
        """Test decorator behavior when function raises exceptions"""
        call_count = 0
        
        @cached(ttl=60)
        async def failing_function(should_fail):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Intentional failure")
            return f"success_{call_count}"
        
        # Exception should not be cached
        with pytest.raises(ValueError):
            await failing_function(True)
        assert call_count == 1
        
        # Another call with same args should try again
        with pytest.raises(ValueError):
            await failing_function(True)
        assert call_count == 2
        
        # Successful call should be cached
        result1 = await failing_function(False)
        assert call_count == 3
        
        result2 = await failing_function(False)
        assert call_count == 3  # Should be cached
        assert result1 == result2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])