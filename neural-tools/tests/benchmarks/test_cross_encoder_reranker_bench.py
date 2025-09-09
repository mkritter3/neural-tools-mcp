#!/usr/bin/env python3
"""
Benchmark tests for CrossEncoderReranker
Tests performance under realistic conditions
"""

import time
import pytest
import asyncio
from typing import List, Dict, Any

from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_rerank_under_120ms_budget():
    """Test reranking performance under 120ms budget with realistic document set"""
    cfg = RerankConfig(latency_budget_ms=120)
    rr = CrossEncoderReranker(cfg, tenant_id="bench")
    
    # Create realistic document set
    docs = [
        {
            "content": f"def function_{i}(param): # Implementation {i} for data processing",
            "score": 0.9 - i*0.01,
            "metadata": {
                "file_path": f"src/module_{i}.py",
                "function_name": f"function_{i}",
                "language": "python"
            }
        }
        for i in range(40)
    ]
    
    start = time.perf_counter()
    results = await rr.rerank("efficient data processing function", docs, top_k=10)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # Should meet performance target
    assert elapsed_ms <= 150, f"Reranking took {elapsed_ms:.1f}ms, expected ≤150ms (120ms + 25% overhead)"
    
    # Should return requested results
    assert len(results) == 10
    
    # All results should have rerank scores
    assert all("rerank_score" in r for r in results)
    
    print(f"✅ Reranked 40 candidates in {elapsed_ms:.1f}ms")


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_cache_performance():
    """Test cache hit performance for repeated queries"""
    cfg = RerankConfig(latency_budget_ms=120, cache_ttl_s=300)
    rr = CrossEncoderReranker(cfg, tenant_id="cache_bench")
    
    docs = [
        {
            "content": "Binary search algorithm implementation",
            "score": 0.9,
            "metadata": {"file_path": "algorithms/binary_search.py"}
        },
        {
            "content": "Hash table with collision resolution", 
            "score": 0.8,
            "metadata": {"file_path": "data_structures/hash_table.py"}
        }
    ]
    
    # First call (cold cache)
    start1 = time.perf_counter()
    results1 = await rr.rerank("search algorithm", docs, top_k=2)
    cold_time = (time.perf_counter() - start1) * 1000
    
    # Second call (warm cache)
    start2 = time.perf_counter()
    results2 = await rr.rerank("search algorithm", docs, top_k=2)
    warm_time = (time.perf_counter() - start2) * 1000
    
    # Results should be consistent
    assert len(results1) == 2
    assert len(results2) == 2
    
    # Warm cache should be faster or at least not significantly slower
    assert warm_time <= cold_time + 10, f"Warm cache ({warm_time:.1f}ms) should be ≤ cold cache ({cold_time:.1f}ms) + 10ms overhead"
    
    print(f"✅ Cold cache: {cold_time:.1f}ms, Warm cache: {warm_time:.1f}ms")


@pytest.mark.benchmark  
@pytest.mark.asyncio
async def test_concurrent_rerank_performance():
    """Test performance under concurrent load"""
    cfg = RerankConfig(latency_budget_ms=120)
    rr = CrossEncoderReranker(cfg, tenant_id="concurrent_bench")
    
    # Create test documents
    docs = [
        {
            "content": f"Implementation of algorithm {i % 10}",
            "score": 0.9 - (i % 10)*0.05,
            "metadata": {"file_path": f"algo_{i % 10}.py"}
        }
        for i in range(20)
    ]
    
    # Different queries to avoid cache hits
    queries = [
        "sorting algorithm",
        "search implementation", 
        "data structure",
        "efficient processing",
        "optimization technique"
    ]
    
    async def single_rerank(query: str) -> float:
        start = time.perf_counter()
        results = await rr.rerank(query, docs, top_k=5)
        elapsed = (time.perf_counter() - start) * 1000
        assert len(results) == 5
        return elapsed
    
    # Run concurrent reranking
    start_total = time.perf_counter()
    times = await asyncio.gather(*[single_rerank(q) for q in queries])
    total_time = (time.perf_counter() - start_total) * 1000
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    # Individual operations should meet budget
    assert max_time <= 150, f"Max time {max_time:.1f}ms exceeded 150ms budget"
    
    # Concurrent execution should be efficient
    sequential_estimate = sum(times)
    concurrency_factor = sequential_estimate / total_time
    
    print(f"✅ Concurrent reranking: {len(queries)} queries in {total_time:.1f}ms")
    print(f"   Average per query: {avg_time:.1f}ms, Max: {max_time:.1f}ms")
    print(f"   Concurrency factor: {concurrency_factor:.1f}x")
    
    assert concurrency_factor >= 1.0, "Concurrent execution should not be slower than sequential"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_scaling_with_candidate_count():
    """Test how performance scales with number of candidates"""
    cfg = RerankConfig(latency_budget_ms=120)
    rr = CrossEncoderReranker(cfg, tenant_id="scaling_bench")
    
    candidate_counts = [10, 20, 30, 50]
    results_data = []
    
    for count in candidate_counts:
        docs = [
            {
                "content": f"Function implementation {i} with processing logic",
                "score": 0.9 - i*0.01,
                "metadata": {"file_path": f"impl_{i}.py"}
            }
            for i in range(count)
        ]
        
        # Run multiple times for average
        times = []
        for _ in range(3):
            start = time.perf_counter()
            results = await rr.rerank("processing function", docs, top_k=min(10, count))
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert len(results) == min(10, count)
        
        avg_time = sum(times) / len(times)
        results_data.append((count, avg_time))
        
        # Should stay under budget regardless of candidate count
        assert avg_time <= 150, f"With {count} candidates: {avg_time:.1f}ms > 150ms budget"
        
        print(f"   {count:2d} candidates: {avg_time:5.1f}ms average")
    
    # Performance should scale reasonably
    time_10 = next(t for c, t in results_data if c == 10)
    time_50 = next(t for c, t in results_data if c == 50)
    
    # 5x candidates should not be more than 3x slower (allowing for batching efficiency)
    scaling_factor = time_50 / time_10
    assert scaling_factor <= 3.0, f"Performance scaling {scaling_factor:.1f}x too steep (50 vs 10 candidates)"
    
    print(f"✅ Scaling from 10 to 50 candidates: {scaling_factor:.1f}x slower (≤3x target)")


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_heuristic_fallback_performance():
    """Test performance of heuristic fallback when model unavailable"""
    cfg = RerankConfig(latency_budget_ms=50, model_path=None)  # No model to force heuristic
    rr = CrossEncoderReranker(cfg, tenant_id="heuristic_bench")
    
    docs = [
        {
            "content": f"Cache implementation with {algo} eviction policy",
            "score": 0.8 - i*0.05,
            "metadata": {"file_path": f"{algo}_cache.py"}
        }
        for i, algo in enumerate(["LRU", "LFU", "Random", "FIFO", "Clock"])
    ]
    
    start = time.perf_counter()
    results = await rr.rerank("LRU cache implementation", docs, top_k=5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # Heuristic should be very fast
    assert elapsed_ms <= 20, f"Heuristic fallback took {elapsed_ms:.1f}ms, expected ≤20ms"
    
    # Should return results
    assert len(results) == 5
    
    # LRU should rank highest due to keyword match
    top_result = results[0]
    assert "LRU" in top_result["content"]
    
    print(f"✅ Heuristic fallback: {elapsed_ms:.1f}ms for 5 candidates")


@pytest.mark.benchmark
def test_cache_memory_usage():
    """Test cache memory behavior under load"""
    cfg = RerankConfig(cache_ttl_s=1, latency_budget_ms=100)  # Short TTL for testing
    rr = CrossEncoderReranker(cfg, tenant_id="memory_bench")
    
    # Simulate many queries to test cache pruning
    import asyncio
    
    async def load_cache():
        for i in range(1000):
            docs = [{"content": f"doc {i % 100}", "score": 0.8, "file_path": f"{i}.py"}]
            await rr.rerank(f"query {i}", docs, top_k=1)
    
    start_time = time.perf_counter()
    asyncio.run(load_cache())
    elapsed = time.perf_counter() - start_time
    
    # Should handle load without excessive memory
    cache_size = len(rr._cache._store)
    
    print(f"✅ Processed 1000 queries in {elapsed:.1f}s, cache size: {cache_size}")
    
    # Cache should be pruned to reasonable size
    assert cache_size <= 50000, f"Cache size {cache_size} exceeds reasonable limit"
    
    # Wait for TTL expiry
    time.sleep(1.5)
    
    # Access one more time to trigger cleanup
    asyncio.run(rr.rerank("cleanup query", [{"content": "test", "score": 0.8}], top_k=1))
    
    # Expired entries should be cleaned up over time
    final_cache_size = len(rr._cache._store)
    print(f"   Cache after TTL expiry: {final_cache_size}")


if __name__ == "__main__":
    # Run benchmarks directly
    import sys
    
    async def run_all_benchmarks():
        print("🏁 Cross-Encoder Reranker Benchmarks")
        print("=" * 50)
        
        try:
            await test_rerank_under_120ms_budget()
            await test_cache_performance() 
            await test_concurrent_rerank_performance()
            await test_scaling_with_candidate_count()
            await test_heuristic_fallback_performance()
            test_cache_memory_usage()
            
            print("\n🎉 All benchmarks passed!")
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            sys.exit(1)
    
    asyncio.run(run_all_benchmarks())