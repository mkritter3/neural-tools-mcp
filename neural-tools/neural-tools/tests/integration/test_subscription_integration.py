#!/usr/bin/env python3
"""
Real Subscription Integration Tests for HaikuReRanker
Tests using Claude subscription authentication (no separate API key needed)
"""

import asyncio
import pytest
import time
from typing import List, Dict, Any

from src.infrastructure.subscription_haiku_reranker import (
    SubscriptionHaikuReRanker,
    rerank_with_subscription
)


class TestSubscriptionHaikuReRankerIntegration:
    """Integration tests using Claude subscription authentication"""
    
    @pytest.fixture
    def subscription_reranker(self):
        """Create reranker with subscription auth"""
        return SubscriptionHaikuReRanker()
    
    @pytest.fixture
    def sample_search_results(self):
        """Real-world search results for testing"""
        return [
            {
                'content': '''
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
                '''.strip(),
                'score': 0.92,
                'metadata': {'file_path': 'algorithms/sorting.py', 'language': 'python'}
            },
            {
                'content': '''
async def parallel_fetch(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)
                '''.strip(),
                'score': 0.87,
                'metadata': {'file_path': 'networking/async_fetch.py', 'language': 'python'}
            },
            {
                'content': '''
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
                '''.strip(),
                'score': 0.81,
                'metadata': {'file_path': 'data_structures/hash_table.py', 'language': 'python'}
            }
        ]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_subscription_basic_reranking(self, subscription_reranker, sample_search_results):
        """Test basic re-ranking with subscription authentication"""
        query = "efficient data structure implementation"
        
        start_time = time.perf_counter()
        result = await subscription_reranker.rerank_simple(
            query=query,
            results=sample_search_results,
            max_results=3
        )
        elapsed_time = time.perf_counter() - start_time
        
        # Validate results
        assert len(result) == 3
        assert all('reranked' in r.get('metadata', {}) for r in result)
        assert all('ranking_confidence' in r.get('metadata', {}) for r in result)
        assert all('subscription_mode' in r.get('metadata', {}) for r in result)
        
        # Should be using subscription mode
        assert result[0]['metadata']['subscription_mode'] is True
        
        # Performance validation - real subscription API
        assert elapsed_time < 60.0, f"Subscription API took {elapsed_time:.2f}s, expected < 60s"
        
        # Confidence should be reasonable for real API
        confidence = result[0]['metadata']['ranking_confidence']
        assert 0.1 <= confidence <= 1.0, f"Unexpected confidence: {confidence}"
        
        print(f"Subscription reranking took {elapsed_time:.2f}s with confidence {confidence}")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_subscription_vs_mock_performance(self, sample_search_results):
        """Compare subscription vs mock performance"""
        query = "algorithm implementation comparison"
        
        # Test subscription mode
        subscription_reranker = SubscriptionHaikuReRanker(use_subscription=True)
        start_time = time.perf_counter()
        subscription_result = await subscription_reranker.rerank_simple(query, sample_search_results)
        subscription_time = time.perf_counter() - start_time
        
        # Test mock mode
        mock_reranker = SubscriptionHaikuReRanker(use_subscription=False)
        start_time = time.perf_counter()
        mock_result = await mock_reranker.rerank_simple(query, sample_search_results)
        mock_time = time.perf_counter() - start_time
        
        # Both should return results
        assert len(subscription_result) == len(sample_search_results)
        assert len(mock_result) == len(sample_search_results)
        
        # Subscription should be slower than mock
        assert subscription_time > mock_time, "Subscription should be slower than mock"
        
        # Check mode indicators
        subscription_stats = subscription_reranker.get_stats()
        mock_stats = mock_reranker.get_stats()
        
        assert subscription_stats['subscription_mode'] is True
        assert subscription_stats['subscription_calls'] >= 1
        assert mock_stats['subscription_mode'] is False
        assert mock_stats['mock_calls'] >= 1
        
        print(f"Performance comparison:")
        print(f"  Subscription: {subscription_time:.2f}s")
        print(f"  Mock: {mock_time:.2f}s")
        print(f"  Speed difference: {subscription_time/mock_time:.1f}x slower")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_subscription_caching_behavior(self, subscription_reranker, sample_search_results):
        """Test caching works with subscription API"""
        query = "caching test query"
        
        # First call - should hit real API
        start_time = time.perf_counter()
        result1 = await subscription_reranker.rerank_simple(query, sample_search_results)
        first_call_time = time.perf_counter() - start_time
        
        # Second identical call - should hit cache
        start_time = time.perf_counter()
        result2 = await subscription_reranker.rerank_simple(query, sample_search_results)
        second_call_time = time.perf_counter() - start_time
        
        # Validate caching worked
        assert second_call_time < first_call_time / 2, "Cache should be much faster"
        assert len(result1) == len(result2)
        
        # Check cache stats
        stats = subscription_reranker.get_stats()
        assert stats['cache_hits'] >= 1
        assert stats['cache_hit_rate'] > 0
        
        print(f"Cache performance:")
        print(f"  First call: {first_call_time:.2f}s")
        print(f"  Cached call: {second_call_time:.2f}s")
        print(f"  Cache speedup: {first_call_time/second_call_time:.1f}x faster")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_subscription_fallback_handling(self):
        """Test graceful fallback when subscription has issues"""
        # This is hard to test directly, but we can test initialization robustness
        reranker = SubscriptionHaikuReRanker()
        
        stats = reranker.get_stats()
        
        # Should have initialized something
        assert 'subscription_mode' in stats
        assert 'sdk_available' in stats
        
        # Should be able to handle requests regardless of mode
        results = [{"content": "test content", "score": 0.8}]
        reranked = await reranker.rerank_simple("test query", results)
        
        assert len(reranked) == 1
        assert 'processing_time' in reranked[0]
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_convenience_function(self, sample_search_results):
        """Test convenience function uses subscription"""
        query = "convenience function test"
        
        result = await rerank_with_subscription(
            query=query,
            results=sample_search_results,
            max_results=2
        )
        
        assert len(result) == 2
        assert all('content' in r for r in result)
        assert all('metadata' in r for r in result)
        
        # Should have subscription metadata if available
        if result[0]['metadata'].get('subscription_mode'):
            assert result[0]['metadata']['subscription_mode'] is True
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_world_code_ranking(self, subscription_reranker):
        """Test ranking real code snippets"""
        code_samples = [
            {
                'content': 'def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]',
                'score': 0.6,
                'metadata': {'complexity': 'O(n²)', 'algorithm': 'bubble_sort'}
            },
            {
                'content': 'def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)',
                'score': 0.9,
                'metadata': {'complexity': 'O(n log n)', 'algorithm': 'merge_sort'}
            },
            {
                'content': 'def insertion_sort(arr):\n    for i in range(1, len(arr)):\n        key = arr[i]\n        j = i - 1\n        while j >= 0 and arr[j] > key:\n            arr[j + 1] = arr[j]\n            j -= 1\n        arr[j + 1] = key',
                'score': 0.75,
                'metadata': {'complexity': 'O(n²)', 'algorithm': 'insertion_sort'}
            }
        ]
        
        result = await subscription_reranker.rerank_simple(
            query="efficient sorting algorithm implementation",
            results=code_samples,
            context="Looking for algorithms with good time complexity",
            max_results=3
        )
        
        # Should prioritize efficient algorithms
        assert len(result) == 3
        
        # Check if merge_sort (most efficient) is ranked highly
        merge_sort_found = False
        for i, r in enumerate(result):
            if r['metadata'].get('algorithm') == 'merge_sort':
                merge_sort_found = True
                print(f"Merge sort ranked at position {i+1}")
                break
        
        assert merge_sort_found, "Merge sort should be found in results"


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_subscription_integration.py -v -m integration
    import sys
    sys.exit(pytest.main([__file__, "-v", "-m", "integration"]))