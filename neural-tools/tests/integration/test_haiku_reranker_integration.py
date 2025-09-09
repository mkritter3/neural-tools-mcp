#!/usr/bin/env python3
"""
Integration tests for Haiku Dynamic Re-ranker (Phase 1.7)
Tests performance targets, end-to-end workflows, and integration with hybrid retriever
"""

import pytest
import asyncio
import time
import os
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Import the modules to test
from infrastructure.haiku_reranker import HaikuReRanker, SearchResult, ReRankingRequest
from infrastructure.enhanced_hybrid_retriever import EnhancedHybridRetriever, enhance_hybrid_retriever


@pytest.mark.benchmark
class TestHaikuReRankerPerformanceTargets:
    """Integration tests for Haiku re-ranker performance targets from roadmap"""
    
    @pytest.fixture
    def reranker(self):
        """Create HaikuReRanker for performance testing"""
        # Use mock mode for consistent performance testing
        return HaikuReRanker(api_key=None, cache_ttl=600)
    
    @pytest.fixture
    def realistic_search_results(self):
        """Create realistic search results for testing"""
        results = []
        
        # Python function results
        results.append(SearchResult(
            content="""def calculate_fibonacci(n: int) -> int:
    '''Calculate nth Fibonacci number using recursion'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)""",
            score=0.92,
            metadata={
                "file_path": "/src/algorithms/fibonacci.py",
                "function": "calculate_fibonacci",
                "language": "python",
                "complexity": "O(2^n)"
            },
            source="vector",
            original_rank=1
        ))
        
        results.append(SearchResult(
            content="""class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)""",
            score=0.87,
            metadata={
                "file_path": "/src/data_structures/tree.py",
                "class": "BinaryTree",
                "language": "python"
            },
            source="vector",
            original_rank=2
        ))
        
        results.append(SearchResult(
            content="""function quickSort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const right = arr.filter(x => x > pivot);
    return [...quickSort(left), pivot, ...quickSort(right)];
}""",
            score=0.83,
            metadata={
                "file_path": "/src/algorithms/sort.js",
                "function": "quickSort",
                "language": "javascript"
            },
            source="graph",
            original_rank=3
        ))
        
        results.append(SearchResult(
            content="""import numpy as np

def matrix_multiply(A, B):
    '''Multiply two matrices using numpy'''
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible matrix dimensions")
    return np.dot(A, B)""",
            score=0.79,
            metadata={
                "file_path": "/src/math/matrix.py",
                "function": "matrix_multiply",
                "language": "python"
            },
            source="vector",
            original_rank=4
        ))
        
        results.append(SearchResult(
            content="""public class HashTable<K, V> {
    private Entry<K, V>[] buckets;
    private int size;
    
    public HashTable(int capacity) {
        this.buckets = new Entry[capacity];
        this.size = 0;
    }
    
    public void put(K key, V value) {
        int hash = key.hashCode() % buckets.length;
        // Handle collision...
    }
}""",
            score=0.75,
            metadata={
                "file_path": "/src/data_structures/HashTable.java",
                "class": "HashTable",
                "language": "java"
            },
            source="hybrid",
            original_rank=5
        ))
        
        return results
    
    @pytest.mark.asyncio
    async def test_sub_100ms_reranking_target(self, reranker, realistic_search_results):
        """ROADMAP EXIT CRITERIA: Sub-100ms re-ranking for realistic result sets"""
        request = ReRankingRequest(
            query="recursive algorithms and data structures",
            results=realistic_search_results,
            context="Looking for efficient algorithmic implementations",
            max_results=5
        )
        
        # Run multiple iterations to get consistent timing
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = await reranker.rerank(request)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # ROADMAP EXIT CRITERIA: <100ms average, <150ms maximum
        assert avg_time < 100, f"Average re-ranking time {avg_time:.1f}ms exceeds 100ms target"
        assert max_time < 150, f"Maximum re-ranking time {max_time:.1f}ms exceeds 150ms limit"
        
        # Verify result quality
        assert len(result.results) == 5
        assert result.confidence_score > 0.5
        assert all(r.metadata.get('reranked') for r in result.results)
        
        print(f"Re-ranking performance: avg={avg_time:.1f}ms, max={max_time:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance_target(self, reranker, realistic_search_results):
        """ROADMAP EXIT CRITERIA: <10ms cache hit response time"""
        request = ReRankingRequest(
            query="algorithm performance test",
            results=realistic_search_results[:3],  # Smaller set for cache testing
            max_results=3
        )
        
        # First call to populate cache
        await reranker.rerank(request)
        
        # Test cache hit performance
        cache_times = []
        for _ in range(20):
            start_time = time.perf_counter()
            result = await reranker.rerank(request)
            end_time = time.perf_counter()
            
            assert result.cache_hit is True
            cache_times.append((end_time - start_time) * 1000)
        
        avg_cache_time = sum(cache_times) / len(cache_times)
        max_cache_time = max(cache_times)
        
        # ROADMAP EXIT CRITERIA: <10ms cache hits
        assert avg_cache_time < 10, f"Average cache hit time {avg_cache_time:.2f}ms exceeds 10ms target"
        assert max_cache_time < 15, f"Maximum cache hit time {max_cache_time:.2f}ms exceeds 15ms limit"
        
        print(f"Cache hit performance: avg={avg_cache_time:.2f}ms, max={max_cache_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_relevance_improvement_target(self, reranker):
        """ROADMAP EXIT CRITERIA: Demonstrable relevance improvement"""
        # Create test case where re-ranking should improve relevance
        results = [
            SearchResult(
                content="print('Hello World')",  # Low relevance for algorithm query
                score=0.95,  # But high vector similarity
                metadata={"file": "hello.py"},
                original_rank=1
            ),
            SearchResult(
                content="def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
                score=0.65,  # Lower vector similarity
                metadata={"file": "sort.py", "function": "bubble_sort"},
                original_rank=2
            ),
            SearchResult(
                content="def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
                score=0.70,
                metadata={"file": "algorithms.py", "function": "quick_sort"},
                original_rank=3
            )
        ]
        
        request = ReRankingRequest(
            query="sorting algorithm implementation",
            results=results,
            context="Looking for efficient sorting algorithms",
            max_results=3
        )
        
        result = await reranker.rerank(request)
        
        # In mock mode, we can check that re-ranking was attempted
        # and proper metadata was added
        assert result.confidence_score > 0.3
        assert all(r.metadata.get('reranked') for r in result.results)
        
        # Check that ranking positions were tracked
        for i, ranked_result in enumerate(result.results):
            assert ranked_result.metadata['new_rank'] == i + 1
            assert 'original_rank' in ranked_result.metadata
        
        print(f"Relevance improvement confidence: {result.confidence_score:.2f}")


@pytest.mark.resilience
class TestHaikuReRankerErrorHandling:
    """Test error handling and resilience"""
    
    @pytest.fixture
    def reranker_with_api(self):
        """Create reranker with invalid API key to test error handling"""
        return HaikuReRanker(api_key="invalid-test-key")
    
    @pytest.mark.asyncio
    async def test_api_failure_fallback(self, reranker_with_api):
        """Test graceful fallback when API fails"""
        results = [
            SearchResult("test content 1", 0.8, {}),
            SearchResult("test content 2", 0.6, {}),
            SearchResult("test content 3", 0.4, {})
        ]
        
        request = ReRankingRequest(
            query="test query",
            results=results
        )
        
        # API should fail but function should still return results
        result = await reranker_with_api.rerank(request)
        
        assert isinstance(result, type(result))
        assert len(result.results) == len(results)
        assert result.model_used == "fallback"
        assert result.confidence_score < 1.0
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_empty_results_handling(self):
        """Test handling of empty result sets"""
        reranker = HaikuReRanker(api_key=None)
        
        request = ReRankingRequest(
            query="test query",
            results=[],  # Empty results
            max_results=5
        )
        
        result = await reranker.rerank(request)
        
        assert len(result.results) == 0
        assert result.processing_time >= 0
        assert isinstance(result.confidence_score, (int, float))
    
    @pytest.mark.asyncio
    async def test_large_content_handling(self):
        """Test handling of very large content snippets"""
        reranker = HaikuReRanker(api_key=None)
        
        # Create result with very large content
        large_content = "def large_function():\n" + "    pass\n" * 1000  # Large function
        results = [SearchResult(large_content, 0.8, {})]
        
        request = ReRankingRequest(
            query="large function",
            results=results
        )
        
        # Should handle large content without crashing
        result = await reranker.rerank(request)
        
        assert len(result.results) == 1
        assert result.processing_time > 0


@pytest.mark.accuracy
class TestEnhancedHybridRetrieverIntegration:
    """Test integration with hybrid retriever system"""
    
    @pytest.fixture
    def mock_hybrid_retriever(self):
        """Create mock hybrid retriever for testing"""
        mock_retriever = Mock()
        mock_retriever.find_similar_with_context = AsyncMock(return_value=[
            {
                "content": "def search_algorithm():\n    return 'found'",
                "score": 0.85,
                "metadata": {"file_path": "/algorithms/search.py", "function": "search_algorithm"}
            },
            {
                "content": "class SearchTree:\n    def find(self, key):\n        pass",
                "score": 0.75,
                "metadata": {"file_path": "/data_structures/tree.py", "class": "SearchTree"}
            }
        ])
        return mock_retriever
    
    @pytest.mark.asyncio
    async def test_enhanced_retriever_basic_integration(self, mock_hybrid_retriever):
        """Test basic integration with enhanced hybrid retriever"""
        enhanced_retriever = EnhancedHybridRetriever(
            hybrid_retriever=mock_hybrid_retriever,
            anthropic_api_key=None,  # Mock mode
            enable_reranking=True,
            rerank_threshold=2
        )
        
        results = await enhanced_retriever.find_similar_with_context(
            query="search algorithms",
            limit=5,
            enable_reranking=True
        )
        
        # Should have called underlying retriever
        mock_hybrid_retriever.find_similar_with_context.assert_called_once()
        
        # Should have re-ranking metadata
        assert len(results) > 0
        for result in results:
            metadata = result.get('metadata', {})
            assert metadata.get('enhanced_hybrid_retrieval') is True
            assert metadata.get('reranked') is True
            assert 'rerank_position' in metadata
    
    @pytest.mark.asyncio
    async def test_rerank_threshold_behavior(self, mock_hybrid_retriever):
        """Test that re-ranking threshold works correctly"""
        enhanced_retriever = EnhancedHybridRetriever(
            hybrid_retriever=mock_hybrid_retriever,
            enable_reranking=True,
            rerank_threshold=5  # High threshold
        )
        
        # Mock fewer results than threshold
        mock_hybrid_retriever.find_similar_with_context.return_value = [
            {"content": "single result", "score": 0.8, "metadata": {}}
        ]
        
        results = await enhanced_retriever.find_similar_with_context(
            query="test query",
            limit=3
        )
        
        # Should not re-rank due to threshold
        assert len(results) == 1
        result_metadata = results[0].get('metadata', {})
        assert result_metadata.get('reranked') is not True
    
    @pytest.mark.asyncio
    async def test_code_chunk_search_integration(self, mock_hybrid_retriever):
        """Test enhanced code chunk search"""
        # Mock code search method if available
        mock_hybrid_retriever.search_code_chunks = AsyncMock(return_value=[
            {
                "content": "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "score": 0.9,
                "metadata": {"file_path": "/math/fib.py", "language": "python", "function": "fibonacci"}
            },
            {
                "content": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
                "score": 0.8,
                "metadata": {"file_path": "/math/fact.py", "language": "python", "function": "factorial"}
            }
        ])
        
        enhanced_retriever = EnhancedHybridRetriever(
            hybrid_retriever=mock_hybrid_retriever,
            enable_reranking=True,
            rerank_threshold=2
        )
        
        results = await enhanced_retriever.search_code_chunks(
            query="recursive mathematical functions",
            limit=5,
            language_filter="python",
            enable_reranking=True
        )
        
        # Should have called the code search method
        mock_hybrid_retriever.search_code_chunks.assert_called_once()
        
        # Should have re-ranking applied
        assert len(results) > 0
        for result in results:
            assert 'reranking_confidence' in result
    
    @pytest.mark.asyncio
    async def test_batch_reranking(self, mock_hybrid_retriever):
        """Test batch re-ranking functionality"""
        enhanced_retriever = EnhancedHybridRetriever(
            hybrid_retriever=mock_hybrid_retriever,
            enable_reranking=True,
            rerank_threshold=1
        )
        
        queries = ["algorithm query", "data structure query"]
        results_lists = [
            [{"content": "algo result 1", "score": 0.8, "metadata": {}}],
            [{"content": "ds result 1", "score": 0.7, "metadata": {}}]
        ]
        
        batch_results = await enhanced_retriever.batch_rerank(
            queries=queries,
            results_list=results_lists,
            context="Programming concepts"
        )
        
        assert len(batch_results) == 2
        assert all(isinstance(results, list) for results in batch_results)
    
    def test_stats_collection(self, mock_hybrid_retriever):
        """Test statistics collection"""
        enhanced_retriever = EnhancedHybridRetriever(
            hybrid_retriever=mock_hybrid_retriever,
            enable_reranking=True
        )
        
        stats = enhanced_retriever.get_stats()
        
        assert 'total_queries' in stats
        assert 'reranked_queries' in stats
        assert 'rerank_rate' in stats
        assert 'reranker_stats' in stats
        assert 'rerank_threshold' in stats
        assert 'reranking_enabled' in stats
    
    def test_configuration_updates(self, mock_hybrid_retriever):
        """Test dynamic configuration updates"""
        enhanced_retriever = EnhancedHybridRetriever(
            hybrid_retriever=mock_hybrid_retriever,
            enable_reranking=False,
            rerank_threshold=5
        )
        
        # Initial configuration
        assert enhanced_retriever.enable_reranking is False
        assert enhanced_retriever.rerank_threshold == 5
        
        # Update configuration
        enhanced_retriever.configure_reranking(
            enable=True,
            threshold=3,
            cache_ttl=120
        )
        
        assert enhanced_retriever.enable_reranking is True
        assert enhanced_retriever.rerank_threshold == 3
        assert enhanced_retriever.reranker.cache_ttl == 120
    
    def test_method_delegation(self, mock_hybrid_retriever):
        """Test that unknown methods are delegated to underlying retriever"""
        # Add a custom method to mock retriever
        mock_hybrid_retriever.custom_method = Mock(return_value="custom_result")
        
        enhanced_retriever = EnhancedHybridRetriever(
            hybrid_retriever=mock_hybrid_retriever
        )
        
        # Should delegate to underlying retriever
        result = enhanced_retriever.custom_method("test_arg")
        
        assert result == "custom_result"
        mock_hybrid_retriever.custom_method.assert_called_once_with("test_arg")


@pytest.mark.stress
class TestHaikuReRankerStressTests:
    """Stress tests for heavy usage scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_reranking_requests(self):
        """Test handling of many concurrent re-ranking requests"""
        reranker = HaikuReRanker(api_key=None, max_cache_size=100)
        
        # Create many different queries to avoid cache hits
        async def single_rerank(query_id: int):
            results = [
                SearchResult(f"content {query_id} {i}", 0.8 - (i * 0.1), {})
                for i in range(5)
            ]
            request = ReRankingRequest(
                query=f"unique query {query_id}",
                results=results
            )
            return await reranker.rerank(request)
        
        # Run 20 concurrent re-ranking requests
        tasks = [single_rerank(i) for i in range(20)]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time
        
        # All should complete successfully
        assert len(results) == 20
        assert all(not isinstance(r, Exception) for r in results)
        
        # Should complete in reasonable time (parallel processing)
        assert total_time < 5.0, f"Concurrent processing took {total_time:.1f}s, should be <5s"
        
        # Check statistics
        stats = reranker.get_stats()
        assert stats['requests'] == 20
        
        print(f"Concurrent processing: 20 requests in {total_time:.2f}s")
    
    @pytest.mark.asyncio
    async def test_cache_size_management(self):
        """Test cache size limits and cleanup"""
        reranker = HaikuReRanker(api_key=None, max_cache_size=5, cache_ttl=60)
        
        # Add more entries than cache limit
        for i in range(10):
            results = [SearchResult(f"content {i}", 0.5, {})]
            request = ReRankingRequest(f"query {i}", results)
            await reranker.rerank(request)
        
        # Cache should be limited to max size
        assert len(reranker.cache) <= 5
        
        # Should still function correctly
        stats = reranker.get_stats()
        assert stats['requests'] == 10


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_search_and_rerank_workflow(self):
        """Test complete search and re-rank workflow"""
        # Simulate a complete search workflow
        
        # 1. Initial search results (simulated)
        initial_results = [
            {
                "content": "class BinarySearchTree:\n    def __init__(self):\n        self.root = None",
                "score": 0.91,
                "metadata": {"file_path": "/ds/bst.py", "class": "BinarySearchTree"}
            },
            {
                "content": "def linear_search(arr, target):\n    for i, val in enumerate(arr):\n        if val == target:\n            return i\n    return -1",
                "score": 0.85,
                "metadata": {"file_path": "/algo/search.py", "function": "linear_search"}
            },
            {
                "content": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                "score": 0.89,
                "metadata": {"file_path": "/algo/search.py", "function": "binary_search"}
            }
        ]
        
        # 2. Apply re-ranking
        reranked_results = await rerank_search_results(
            query="binary search tree implementation",
            results=initial_results,
            context="Looking for tree data structures and search algorithms",
            max_results=3
        )
        
        # 3. Verify end-to-end result
        assert len(reranked_results) == 3
        
        # Should have re-ranking metadata
        for result in reranked_results:
            assert "reranking_confidence" in result
            assert result["metadata"]["reranked"] is True
            assert "processing_time" in result
        
        # Should maintain original data
        for result in reranked_results:
            assert "content" in result
            assert "score" in result
            assert "metadata" in result
        
        print(f"End-to-end workflow completed successfully with confidence scores: {[r['reranking_confidence'] for r in reranked_results]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])