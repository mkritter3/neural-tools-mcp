#!/usr/bin/env python3
"""
Real API Integration Tests for HaikuReRanker (Production Readiness)

These tests use actual Anthropic API calls to validate:
- Real API performance and latency
- Contract compliance with Anthropic's API
- Actual error handling scenarios
- Production-ready caching behavior
- Rate limiting and throttling responses

Requires ANTHROPIC_API_KEY environment variable for real tests.
"""

import asyncio
import os
import pytest
import time
from typing import List, Dict, Any

from src.infrastructure.haiku_reranker import (
    HaikuReRanker, 
    SearchResult, 
    ReRankingRequest,
    rerank_search_results
)


class TestHaikuReRankerRealAPI:
    """Real API integration tests for production readiness"""
    
    @pytest.fixture
    def api_key(self):
        """Get API key from environment"""
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            pytest.skip("ANTHROPIC_API_KEY not set - skipping real API tests")
        return key
    
    @pytest.fixture
    def real_reranker(self, api_key):
        """Create reranker with real API key"""
        return HaikuReRanker(api_key=api_key, cache_ttl=60, max_cache_size=100)
    
    @pytest.fixture
    def sample_search_results(self):
        """Real-world search results for testing"""
        return [
            {
                'content': '''
def calculate_fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number using dynamic programming"""
    if n <= 1:
        return n
    
    prev, curr = 0, 1
    for i in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr
                '''.strip(),
                'score': 0.85,
                'metadata': {'file_path': 'algorithms/fibonacci.py', 'language': 'python'}
            },
            {
                'content': '''
class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)
                '''.strip(),
                'score': 0.78,
                'metadata': {'file_path': 'data_structures/binary_tree.py', 'language': 'python'}
            },
            {
                'content': '''
async def process_batch(items: List[str]) -> List[Dict]:
    """Process items in parallel batches"""
    results = []
    semaphore = asyncio.Semaphore(10)  # Limit concurrency
    
    async def process_item(item):
        async with semaphore:
            # Simulate processing
            await asyncio.sleep(0.1)
            return {"item": item, "processed": True}
    
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
                '''.strip(),
                'score': 0.72,
                'metadata': {'file_path': 'async/batch_processor.py', 'language': 'python'}
            },
            {
                'content': '''
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
                '''.strip(),
                'score': 0.69,
                'metadata': {'file_path': 'algorithms/sorting.py', 'language': 'python'}
            }
        ]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_basic_reranking(self, real_reranker, sample_search_results):
        """Test basic re-ranking with real API"""
        query = "efficient algorithm implementation"
        
        start_time = time.perf_counter()
        result = await real_reranker.rerank_simple(
            query=query,
            results=sample_search_results,
            max_results=3
        )
        elapsed_time = time.perf_counter() - start_time
        
        # Validate results
        assert len(result) == 3
        assert all('reranked' in r.get('metadata', {}) for r in result)
        assert all('ranking_confidence' in r.get('metadata', {}) for r in result)
        
        # Performance validation - real API should be under 2 seconds
        assert elapsed_time < 2.0, f"Real API took {elapsed_time:.2f}s, expected < 2s"
        
        # Confidence should be reasonable for real API
        confidence = result[0]['metadata']['ranking_confidence']
        assert 0.3 <= confidence <= 1.0, f"Unexpected confidence: {confidence}"
        
        print(f"Real API reranking took {elapsed_time*1000:.1f}ms with confidence {confidence}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_performance_target(self, real_reranker, sample_search_results):
        """Test real API meets sub-2s performance target"""
        query = "data structure implementation"
        
        # Run multiple iterations to get average performance
        times = []
        for _ in range(3):
            start_time = time.perf_counter()
            await real_reranker.rerank_simple(
                query=query,
                results=sample_search_results[:2],  # Smaller batch for speed
                max_results=2
            )
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Real API performance targets (more lenient than mock)
        assert avg_time < 1.5, f"Average time {avg_time:.2f}s exceeds 1.5s target"
        assert max_time < 2.0, f"Max time {max_time:.2f}s exceeds 2s target"
        
        print(f"Real API performance - Avg: {avg_time*1000:.1f}ms, Max: {max_time*1000:.1f}ms")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_caching_behavior(self, real_reranker, sample_search_results):
        """Test caching works with real API"""
        query = "algorithm efficiency"
        
        # First call - should hit real API
        start_time = time.perf_counter()
        result1 = await real_reranker.rerank_simple(query, sample_search_results)
        first_call_time = time.perf_counter() - start_time
        
        # Second identical call - should hit cache
        start_time = time.perf_counter()
        result2 = await real_reranker.rerank_simple(query, sample_search_results)
        second_call_time = time.perf_counter() - start_time
        
        # Validate caching worked
        assert second_call_time < first_call_time / 2, "Cache should be much faster"
        assert len(result1) == len(result2)
        
        # Check cache stats
        stats = real_reranker.get_stats()
        assert stats['cache_hits'] >= 1
        assert stats['cache_hit_rate'] > 0
        
        print(f"Cache performance - First: {first_call_time*1000:.1f}ms, Cached: {second_call_time*1000:.1f}ms")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_error_handling(self, api_key):
        """Test error handling with real API"""
        # Test with invalid API key
        bad_reranker = HaikuReRanker(api_key="invalid_key_12345")
        
        result = await bad_reranker.rerank_simple(
            query="test query",
            results=[{"content": "test content", "score": 0.8}]
        )
        
        # Should fall back gracefully
        assert len(result) == 1
        assert result[0]['content'] == "test content"
        assert 'processing_time' in result[0]
        
        # Test with malformed results
        good_reranker = HaikuReRanker(api_key=api_key)
        result = await good_reranker.rerank_simple(
            query="test query",
            results=[{"malformed": "data"}]  # Missing content/score
        )
        
        # Should handle gracefully
        assert isinstance(result, list)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_contract_validation(self, real_reranker, sample_search_results):
        """Validate API contract compliance"""
        query = "python programming patterns"
        
        # Create detailed request
        search_results = [
            SearchResult(
                content=r['content'],
                score=r['score'],
                metadata=r['metadata'],
                source='vector',
                original_rank=i+1
            ) for i, r in enumerate(sample_search_results)
        ]
        
        request = ReRankingRequest(
            query=query,
            results=search_results,
            context="Looking for high-quality Python code examples",
            max_results=3
        )
        
        result = await real_reranker.rerank(request)
        
        # Validate response contract
        assert hasattr(result, 'results')
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'model_used')
        assert hasattr(result, 'confidence_score')
        assert hasattr(result, 'cache_hit')
        
        assert result.model_used == "claude-3-haiku"  # Real API model
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.processing_time, float)
        assert result.processing_time > 0
        
        # Validate result metadata
        for ranked_result in result.results:
            assert hasattr(ranked_result, 'metadata')
            assert 'reranked' in ranked_result.metadata
            assert 'ranking_confidence' in ranked_result.metadata
            assert 'original_rank' in ranked_result.metadata
            assert 'new_rank' in ranked_result.metadata
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_api_batch_processing(self, real_reranker):
        """Test batch processing with real API"""
        queries = [
            "fast sorting algorithm",
            "recursive data structure",
            "async programming pattern"
        ]
        
        results_sets = [
            [{"content": f"Content for query {i+1}", "score": 0.8}]
            for i in range(len(queries))
        ]
        
        start_time = time.perf_counter()
        # Note: This would need to be implemented in EnhancedHybridRetriever
        # For now, test sequential processing
        batch_results = []
        for query, results in zip(queries, results_sets):
            result = await real_reranker.rerank_simple(query, results)
            batch_results.append(result)
        
        elapsed = time.perf_counter() - start_time
        
        assert len(batch_results) == len(queries)
        # Should complete in reasonable time even for sequential processing
        assert elapsed < 5.0, f"Batch processing took {elapsed:.2f}s"
    
    @pytest.mark.integration
    @pytest.mark.asyncio 
    async def test_real_api_rate_limiting_resilience(self, real_reranker, sample_search_results):
        """Test behavior under potential rate limiting"""
        # Send multiple rapid requests to test rate limiting handling
        tasks = []
        for i in range(5):
            task = real_reranker.rerank_simple(
                query=f"test query {i}",
                results=sample_search_results[:2],
                max_results=2
            )
            tasks.append(task)
        
        # All should complete without throwing exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Validate all completed (some may be cached, some may fail gracefully)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Request {i} failed with exception: {result}")
            assert isinstance(result, list)
            assert len(result) <= 2


@pytest.mark.integration
class TestProductionReadiness:
    """Tests for production deployment readiness"""
    
    def test_environment_configuration(self):
        """Test proper environment setup"""
        # These would be set in production environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if api_key:
            # Validate key format (basic check)
            assert api_key.startswith("sk-") or len(api_key) > 20
            
        # Test configuration validation
        reranker = HaikuReRanker(api_key=api_key)
        assert reranker.cache_ttl > 0
        assert reranker.max_cache_size > 0
        assert reranker.base_url == "https://api.anthropic.com"
    
    @pytest.mark.asyncio
    async def test_fallback_behavior(self):
        """Test graceful degradation when API unavailable"""
        # Test with no API key (should use mock)
        reranker = HaikuReRanker(api_key=None)
        assert reranker.mock_mode is True
        
        result = await reranker.rerank_simple(
            query="test",
            results=[{"content": "test content", "score": 0.8}]
        )
        
        assert len(result) == 1
        assert result[0]['content'] == "test content"
        
        # Test with invalid base URL (should fallback)
        reranker = HaikuReRanker(
            api_key="test_key",
            base_url="https://invalid-url-that-doesnt-exist.com"
        )
        
        result = await reranker.rerank_simple(
            query="test", 
            results=[{"content": "test content", "score": 0.8}]
        )
        
        # Should fallback gracefully
        assert len(result) == 1


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_haiku_reranker_real_api.py -v -m integration
    import sys
    sys.exit(pytest.main([__file__, "-v", "-m", "integration"]))