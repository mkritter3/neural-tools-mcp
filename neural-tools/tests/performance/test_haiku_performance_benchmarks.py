#!/usr/bin/env python3
"""
Performance Benchmarks for HaikuReRanker (Production Validation)

Comprehensive performance testing with real API measurements:
- Latency percentiles (p50, p95, p99)
- Throughput under load
- Cache hit rate optimization
- Memory usage profiling
- Concurrent request handling
- Production scenario simulation

Run with: pytest tests/performance/ -v --benchmark-only
"""

import asyncio
import os
import pytest
import time
import statistics
import psutil
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.haiku_reranker import HaikuReRanker
from infrastructure.enhanced_hybrid_retriever import EnhancedHybridRetriever


class TestHaikuPerformanceBenchmarks:
    """Production performance validation"""
    
    @pytest.fixture
    def api_key(self):
        """Get API key from environment"""
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            pytest.skip("ANTHROPIC_API_KEY required for performance tests")
        return key
    
    @pytest.fixture
    def performance_reranker(self, api_key):
        """Optimized reranker for performance testing"""
        return HaikuReRanker(
            api_key=api_key,
            cache_ttl=600,  # 10 minutes for better hit rate
            max_cache_size=5000  # Larger cache for throughput
        )
    
    @pytest.fixture
    def large_result_set(self):
        """Large result set for performance testing"""
        results = []
        code_templates = [
            "def function_{}(x): return x * 2",
            "class Class{}:\n    def method(self): pass",
            "async def async_function_{}(): await asyncio.sleep(0.1)",
            "for i in range({}):\n    print(i)",
            "if condition_{}:\n    return True\nelse:\n    return False"
        ]
        
        for i in range(50):
            template = code_templates[i % len(code_templates)]
            content = template.format(i)
            results.append({
                'content': content,
                'score': 0.9 - (i * 0.01),  # Decreasing scores
                'metadata': {
                    'file_path': f'test_file_{i}.py',
                    'language': 'python',
                    'function_count': 1
                }
            })
        
        return results
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_latency_percentiles(self, performance_reranker, large_result_set):
        """Measure latency percentiles with real API"""
        query = "efficient Python implementation"
        latencies = []
        
        # Run 20 requests to measure latency distribution
        for i in range(20):
            # Vary the query slightly to avoid cache hits
            test_query = f"{query} variant {i}"
            test_results = large_result_set[:10]  # Limit for faster testing
            
            start_time = time.perf_counter()
            await performance_reranker.rerank_simple(
                query=test_query,
                results=test_results,
                max_results=5
            )
            latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate percentiles
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = max(latencies)  # With 20 samples, max is roughly p99
        
        print(f"\nLatency Percentiles (Real API):")
        print(f"P50: {p50:.1f}ms")
        print(f"P95: {p95:.1f}ms") 
        print(f"P99: {p99:.1f}ms")
        
        # Production targets
        assert p50 < 800, f"P50 latency {p50:.1f}ms exceeds 800ms target"
        assert p95 < 2000, f"P95 latency {p95:.1f}ms exceeds 2000ms target"
        assert p99 < 3000, f"P99 latency {p99:.1f}ms exceeds 3000ms target"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_cache_optimization_performance(self, performance_reranker, large_result_set):
        """Test cache hit rate optimization"""
        queries = [
            "algorithm implementation",
            "data structure design", 
            "algorithm implementation",  # Repeat for cache hit
            "performance optimization",
            "data structure design",    # Repeat for cache hit
        ]
        
        results_subset = large_result_set[:8]
        total_time = 0
        
        for i, query in enumerate(queries):
            start_time = time.perf_counter()
            await performance_reranker.rerank_simple(
                query=query,
                results=results_subset,
                max_results=5
            )
            elapsed = time.perf_counter() - start_time
            total_time += elapsed
            
            print(f"Query {i+1}: {elapsed*1000:.1f}ms")
        
        # Check cache performance
        stats = performance_reranker.get_stats()
        cache_hit_rate = stats['cache_hit_rate']
        
        print(f"Cache hit rate: {cache_hit_rate:.1f}%")
        print(f"Total time: {total_time*1000:.1f}ms")
        print(f"Average time per query: {total_time*1000/len(queries):.1f}ms")
        
        # Should have at least 30% cache hit rate with repeated queries
        assert cache_hit_rate >= 30, f"Cache hit rate {cache_hit_rate:.1f}% too low"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, performance_reranker, large_result_set):
        """Test performance under concurrent load"""
        query_base = "concurrent test query"
        concurrency_levels = [1, 3, 5]
        results_subset = large_result_set[:6]  # Smaller set for faster testing
        
        for concurrency in concurrency_levels:
            # Create concurrent tasks
            tasks = []
            for i in range(concurrency):
                task = performance_reranker.rerank_simple(
                    query=f"{query_base} {i}",
                    results=results_subset,
                    max_results=3
                )
                tasks.append(task)
            
            # Measure concurrent execution time
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - start_time
            
            # All tasks should complete successfully
            assert len(results) == concurrency
            for result in results:
                assert isinstance(result, list)
                assert len(result) <= 3
            
            avg_latency = (elapsed / concurrency) * 1000
            print(f"Concurrency {concurrency}: {elapsed*1000:.1f}ms total, {avg_latency:.1f}ms avg")
            
            # Performance shouldn't degrade significantly under moderate concurrency
            if concurrency <= 3:
                assert avg_latency < 1500, f"Average latency {avg_latency:.1f}ms too high"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_memory_usage_profiling(self, performance_reranker, large_result_set):
        """Profile memory usage during operations"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        query = "memory usage test"
        
        # Run multiple operations and measure memory
        for i in range(10):
            await performance_reranker.rerank_simple(
                query=f"{query} {i}",
                results=large_result_set[:15],
                max_results=8
            )
            
            if i % 3 == 0:  # Sample memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                print(f"After {i+1} operations: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory
        
        print(f"Total memory increase: {total_increase:.1f}MB")
        
        # Should not have excessive memory growth
        assert total_increase < 50, f"Memory increase {total_increase:.1f}MB excessive"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, performance_reranker):
        """Measure sustained throughput"""
        query = "throughput test"
        simple_results = [
            {"content": f"Simple result {i}", "score": 0.8 - i*0.1}
            for i in range(5)
        ]
        
        # Measure requests per second
        duration = 10  # seconds
        request_count = 0
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < duration:
            await performance_reranker.rerank_simple(
                query=f"{query} {request_count}",
                results=simple_results,
                max_results=3
            )
            request_count += 1
        
        elapsed = time.perf_counter() - start_time
        throughput = request_count / elapsed
        
        print(f"Sustained throughput: {throughput:.2f} requests/second")
        print(f"Total requests in {duration}s: {request_count}")
        
        # Should maintain reasonable throughput
        assert throughput >= 0.5, f"Throughput {throughput:.2f} req/s too low"
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_production_scenario_simulation(self, api_key, large_result_set):
        """Simulate realistic production usage patterns"""
        # Create enhanced retriever (production setup)
        from unittest.mock import MagicMock
        
        # Mock the hybrid retriever dependency
        mock_hybrid = MagicMock()
        mock_hybrid.find_similar_with_context = AsyncMock(return_value=large_result_set[:10])
        
        enhanced_retriever = EnhancedHybridRetriever(
            hybrid_retriever=mock_hybrid,
            anthropic_api_key=api_key,
            enable_reranking=True,
            rerank_threshold=3
        )
        
        # Simulate realistic query patterns
        production_queries = [
            "find authentication middleware implementation",
            "database connection pooling example", 
            "error handling best practices",
            "async task processing pattern",
            "caching strategy implementation",
            "find authentication middleware implementation",  # Repeat for cache
        ]
        
        total_time = 0
        successful_requests = 0
        
        for query in production_queries:
            try:
                start_time = time.perf_counter()
                results = await enhanced_retriever.find_similar_with_context(
                    query=query,
                    limit=5,
                    include_graph_context=True,
                    enable_reranking=True
                )
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                successful_requests += 1
                
                assert len(results) <= 5
                print(f"Query: {query[:30]}... -> {len(results)} results in {elapsed*1000:.1f}ms")
                
            except Exception as e:
                print(f"Query failed: {e}")
        
        # Performance metrics
        avg_response_time = (total_time / successful_requests) * 1000 if successful_requests > 0 else 0
        success_rate = (successful_requests / len(production_queries)) * 100
        
        print(f"\nProduction Simulation Results:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Response Time: {avg_response_time:.1f}ms")
        print(f"Total Time: {total_time:.2f}s")
        
        # Production targets
        assert success_rate >= 95, f"Success rate {success_rate:.1f}% below 95% target"
        assert avg_response_time < 1000, f"Avg response time {avg_response_time:.1f}ms exceeds 1s"


class AsyncMock:
    """Simple async mock for testing"""
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __call__(self, *args, **kwargs):
        return self.return_value


@pytest.mark.benchmark
class TestProductionScalability:
    """Scalability tests for production deployment"""
    
    @pytest.mark.asyncio
    async def test_cache_scalability(self):
        """Test cache performance with large datasets"""
        reranker = HaikuReRanker(
            api_key=None,  # Use mock for fast testing
            cache_ttl=300,
            max_cache_size=1000
        )
        
        # Generate many unique queries to test cache limits
        queries = [f"test query {i}" for i in range(1200)]  # Exceeds cache size
        results = [{"content": "test content", "score": 0.8}]
        
        # Fill cache beyond limit
        for i, query in enumerate(queries):
            await reranker.rerank_simple(query, results, max_results=1)
            
            if i % 200 == 0:
                stats = reranker.get_stats()
                print(f"After {i} queries - Cache size: {stats['cache_size']}")
        
        # Cache should not exceed max size
        final_stats = reranker.get_stats()
        assert final_stats['cache_size'] <= 1000, "Cache size exceeded limit"
        
        print(f"Final cache size: {final_stats['cache_size']}")
    
    @pytest.mark.asyncio
    async def test_error_recovery_resilience(self):
        """Test system resilience under error conditions"""
        # Test with intermittent failures
        reranker = HaikuReRanker(api_key="invalid_key_for_testing")
        
        error_count = 0
        success_count = 0
        
        for i in range(10):
            try:
                result = await reranker.rerank_simple(
                    query=f"test query {i}",
                    results=[{"content": "test", "score": 0.8}],
                    max_results=1
                )
                
                # Should always return a result (fallback)
                assert len(result) == 1
                success_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"Error in iteration {i}: {e}")
        
        print(f"Success rate under errors: {success_count}/{success_count + error_count}")
        
        # System should be resilient with fallbacks
        assert success_count >= 8, "Too many failures - check error handling"


if __name__ == "__main__":
    # Run with: python -m pytest tests/performance/test_haiku_performance_benchmarks.py -v -m benchmark
    import sys
    sys.exit(pytest.main([__file__, "-v", "-m", "benchmark"]))