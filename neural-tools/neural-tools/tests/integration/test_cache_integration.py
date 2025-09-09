#!/usr/bin/env python3
"""
Phase 1.2 Integration Tests - Caching with Neural Tools Integration
Tests caching integration with search endpoints and real workloads
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch
import httpx

import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.simple_cache import SimpleCache, cached

class TestCacheIntegration:
    """Integration tests for caching with neural tools pipeline"""
    
    @pytest.fixture
    def search_cache(self):
        """Cache for search operations"""
        return SimpleCache(ttl=300, max_size=1000)  # 5 minute TTL
    
    @pytest.fixture
    def embedding_cache(self):
        """Cache for embedding operations"""
        return SimpleCache(ttl=1800, max_size=500)  # 30 minute TTL
    
    @pytest.mark.asyncio
    async def test_search_result_caching(self, search_cache):
        """Test caching of search results"""
        
        # Mock search function that's expensive to compute
        search_call_count = 0
        
        async def perform_search(query: str):
            nonlocal search_call_count
            search_call_count += 1
            # Simulate expensive search operation
            await asyncio.sleep(0.1)
            return {
                "results": [
                    {"id": 1, "content": f"Result for {query}", "score": 0.95},
                    {"id": 2, "content": f"Another result for {query}", "score": 0.87}
                ],
                "total": 2,
                "query": query,
                "computed_at": time.time()
            }
        
        query = "test search query"
        
        # First search - should compute
        start_time = time.time()
        async def compute_func1():
            return await perform_search(query)
        result1 = await search_cache.get_or_compute(f"search:{query}", compute_func1)
        first_search_time = time.time() - start_time
        
        assert search_call_count == 1
        assert result1["query"] == query
        assert result1["total"] == 2
        
        # Second search - should be cached
        start_time = time.time()
        async def compute_func2():
            return await perform_search(query)
        result2 = await search_cache.get_or_compute(f"search:{query}", compute_func2)
        cached_search_time = time.time() - start_time
        
        assert search_call_count == 1  # Should not increment
        assert result1 == result2
        
        # Cached should be significantly faster
        print(f"First search: {first_search_time*1000:.1f}ms, Cached: {cached_search_time*1000:.1f}ms")
        assert cached_search_time < first_search_time * 0.1  # Cached should be 10x faster
    
    @pytest.mark.asyncio
    async def test_embedding_caching_integration(self, embedding_cache):
        """Test caching of embedding computations"""
        
        embedding_call_count = 0
        
        async def compute_embeddings(texts):
            nonlocal embedding_call_count
            embedding_call_count += 1
            # Simulate expensive embedding computation
            await asyncio.sleep(0.05)
            return {
                "embeddings": [[0.1, 0.2, 0.3] for _ in texts],
                "model": "nomic-v2",
                "texts": texts,
                "computed_at": time.time()
            }
        
        texts = ["Hello world", "Test document"]
        cache_key = f"embeddings:{hash(str(texts))}"
        
        # First embedding - should compute
        async def embed_func1():
            return await compute_embeddings(texts)
        result1 = await embedding_cache.get_or_compute(cache_key, embed_func1)
        assert embedding_call_count == 1
        assert len(result1["embeddings"]) == 2
        
        # Second embedding with same texts - should be cached
        async def embed_func2():
            return await compute_embeddings(texts)
        result2 = await embedding_cache.get_or_compute(cache_key, embed_func2)
        assert embedding_call_count == 1  # Should not increment
        assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_realistic_workload(self, search_cache):
        """Test cache hit rate > 60% in realistic workload"""
        
        search_queries = [
            "machine learning",
            "python programming", 
            "data science",
            "machine learning",  # Repeat
            "neural networks",
            "python programming",  # Repeat
            "machine learning",  # Repeat
            "deep learning",
            "python programming",  # Repeat
            "data science",  # Repeat
            "machine learning",  # Repeat
            "neural networks",  # Repeat
            "python programming",  # Repeat
            "machine learning",  # Repeat
            "data science",  # Repeat
            "python programming",  # Repeat
            "machine learning",  # Repeat
            "neural networks",  # Repeat
            "python programming",  # Repeat
            "machine learning",  # Repeat
        ]
        
        async def mock_search(query):
            await asyncio.sleep(0.01)  # Small delay to simulate work
            return {"query": query, "results": [f"result for {query}"]}
        
        # Execute search workload
        for query in search_queries:
            async def search_func(q=query):
                return await mock_search(q)
            await search_cache.get_or_compute(f"search:{query}", search_func)
        
        # Check hit rate
        metrics = search_cache.get_metrics()
        hit_rate = metrics["hit_rate"]
        
        print(f"Cache metrics: {metrics}")
        print(f"Hit rate: {hit_rate:.1%}")
        
        # EXIT CRITERIA 1.2.2: Cache hit rate > 60% in realistic workload
        assert hit_rate > 0.6, f"Hit rate {hit_rate:.1%} is below 60% requirement"
    
    @pytest.mark.asyncio
    async def test_multilevel_caching(self):
        """Test multiple cache levels (L1: memory, L2: longer TTL)"""
        
        # L1 cache - fast, short TTL
        l1_cache = SimpleCache(ttl=5, max_size=50)
        
        # L2 cache - larger, longer TTL  
        l2_cache = SimpleCache(ttl=60, max_size=200)
        
        computation_count = 0
        
        async def expensive_computation(key):
            nonlocal computation_count
            computation_count += 1
            await asyncio.sleep(0.1)
            return {"key": key, "computed": computation_count, "timestamp": time.time()}
        
        async def multilevel_get(key):
            # Try L1 first
            try:
                async def l1_compute():
                    async def l2_compute():
                        return await expensive_computation(key)
                    return await l2_cache.get_or_compute(key, l2_compute)
                return await l1_cache.get_or_compute(key, l1_compute)
            except Exception:
                # Fallback to L2 directly
                async def l2_fallback():
                    return await expensive_computation(key)
                return await l2_cache.get_or_compute(key, l2_fallback)
        
        # First access - should compute
        result1 = await multilevel_get("test_key")
        assert computation_count == 1
        
        # Second access - should hit L1
        result2 = await multilevel_get("test_key")
        assert computation_count == 1
        assert result1 == result2
        
        # Clear L1, should hit L2
        l1_cache.clear()
        result3 = await multilevel_get("test_key")
        assert computation_count == 1  # Should hit L2
        assert result1 == result3
    
    @pytest.mark.asyncio
    async def test_cache_warming(self, search_cache):
        """Test cache warming functionality"""
        
        # Popular queries that should be pre-cached
        popular_queries = [
            "python tutorial",
            "machine learning basics", 
            "data science guide",
            "programming tips",
            "AI fundamentals"
        ]
        
        async def search_function(query):
            return {"query": query, "results": [f"cached result for {query}"]}
        
        # Warm the cache
        warming_start = time.time()
        warming_tasks = []
        for query in popular_queries:
            async def warm_func(q=query):
                return await search_function(q)
            warming_tasks.append(search_cache.get_or_compute(f"search:{query}", warm_func))
        await asyncio.gather(*warming_tasks)
        warming_time = time.time() - warming_start
        
        # Test cache effectiveness after warming
        access_start = time.time()
        for query in popular_queries:
            def none_func():
                return None
            result = await search_cache.get_or_compute(f"search:{query}", none_func)
            assert result["query"] == query
        access_time = time.time() - access_start
        
        print(f"Cache warming: {warming_time:.4f}s, Post-warming access: {access_time:.4f}s")
        
        # Post-warming access should be very fast
        # Since both times are very small, just check that access was cached (very fast)
        assert access_time < 0.01, f"Cache access time {access_time:.4f}s should be very fast"
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_patterns(self, search_cache):
        """Test different cache invalidation patterns"""
        
        async def data_function(version):
            return {"data": f"content_v{version}", "version": version}
        
        # Initial data
        async def data_func1():
            return await data_function(1)
        result1 = await search_cache.get_or_compute("data", data_func1)
        assert result1["version"] == 1
        
        # Simulate data update - invalidate specific key
        search_cache.delete("data")
        async def data_func2():
            return await data_function(2)
        result2 = await search_cache.get_or_compute("data", data_func2)
        assert result2["version"] == 2
        
        # Test pattern-based invalidation (simulate with manual deletions)
        def john_func():
            return {"user": "john"}
        def dark_func():
            return {"theme": "dark"}
        def jane_func():
            return {"user": "jane"}
        await search_cache.get_or_compute("user:123:profile", john_func)
        await search_cache.get_or_compute("user:123:settings", dark_func)
        await search_cache.get_or_compute("user:456:profile", jane_func)
        
        # Invalidate all user:123:* entries
        keys_to_delete = [key for key in search_cache._cache.keys() if key.startswith("user:123:")]
        for key in keys_to_delete:
            search_cache.delete(key)
        
        # user:123 entries should be invalidated, user:456 should remain
        recompute_count = 0
        async def check_recompute():
            nonlocal recompute_count
            recompute_count += 1
            return {"recomputed": True}
        
        await search_cache.get_or_compute("user:123:profile", check_recompute)
        assert recompute_count == 1  # Should recompute
        
        await search_cache.get_or_compute("user:456:profile", check_recompute)
        assert recompute_count == 1  # Should still be cached
    
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self):
        """Test cache performance under concurrent load"""
        cache = SimpleCache(ttl=60, max_size=1000)
        
        async def worker_task(worker_id, num_operations):
            """Worker that performs cache operations"""
            for i in range(num_operations):
                key = f"worker_{worker_id}_item_{i % 10}"  # Reuse some keys
                def worker_func():
                    return {"worker": worker_id, "item": i, "data": "x" * 100}
                result = await cache.get_or_compute(key, worker_func)
                assert "worker" in result
        
        # Run concurrent workers
        num_workers = 10
        operations_per_worker = 50
        
        start_time = time.time()
        tasks = [
            asyncio.create_task(worker_task(worker_id, operations_per_worker))
            for worker_id in range(num_workers)
        ]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze performance
        metrics = cache.get_metrics()
        total_operations = num_workers * operations_per_worker
        ops_per_second = total_operations / total_time
        
        print(f"Concurrent performance:")
        print(f"Total operations: {total_operations}")
        print(f"Total time: {total_time:.2f}s") 
        print(f"Operations/second: {ops_per_second:.1f}")
        print(f"Cache metrics: {metrics}")
        
        # Performance requirements
        assert ops_per_second > 1000, f"Performance {ops_per_second:.1f} ops/sec is too low"
        assert metrics["hit_rate"] > 0.5, f"Hit rate {metrics['hit_rate']:.1%} too low under load"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])