#!/usr/bin/env python3
"""
Unit tests for Haiku Dynamic Re-ranker (Phase 1.7)
Tests fast re-ranking, caching, and performance optimization
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Import the modules to test
from src.infrastructure.haiku_reranker import (
    HaikuReRanker, SearchResult, ReRankingRequest, ReRankingResult,
    rerank_search_results, get_global_reranker
)


class TestSearchResult:
    """Unit tests for SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test SearchResult creation with all fields"""
        result = SearchResult(
            content="def test_function():\n    return True",
            score=0.85,
            metadata={"file": "test.py", "line": 10},
            source="vector",
            original_rank=1
        )
        
        assert result.content == "def test_function():\n    return True"
        assert result.score == 0.85
        assert result.metadata["file"] == "test.py"
        assert result.source == "vector"
        assert result.original_rank == 1
    
    def test_search_result_defaults(self):
        """Test SearchResult with default values"""
        result = SearchResult(
            content="sample content",
            score=0.5,
            metadata={}
        )
        
        assert result.source == "vector"
        assert result.original_rank == 0


class TestReRankingRequest:
    """Unit tests for ReRankingRequest dataclass"""
    
    def test_reranking_request_creation(self):
        """Test ReRankingRequest creation"""
        results = [
            SearchResult("content1", 0.8, {}),
            SearchResult("content2", 0.6, {})
        ]
        
        request = ReRankingRequest(
            query="test query",
            results=results,
            context="test context",
            max_results=5
        )
        
        assert request.query == "test query"
        assert len(request.results) == 2
        assert request.context == "test context"
        assert request.max_results == 5
    
    def test_reranking_request_defaults(self):
        """Test ReRankingRequest with default values"""
        results = [SearchResult("content", 0.7, {})]
        request = ReRankingRequest(query="query", results=results)
        
        assert request.context is None
        assert request.max_results == 10
        assert request.relevance_threshold == 0.1


class TestHaikuReRanker:
    """Unit tests for HaikuReRanker class"""
    
    @pytest.fixture
    def mock_reranker(self):
        """Create HaikuReRanker in mock mode"""
        return HaikuReRanker(api_key=None)  # None triggers mock mode
    
    @pytest.fixture
    def sample_results(self):
        """Create sample search results for testing"""
        return [
            SearchResult(
                content="def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
                score=0.9,
                metadata={"file": "math.py", "function": "fibonacci"},
                source="vector",
                original_rank=1
            ),
            SearchResult(
                content="class Calculator:\n    def add(self, a, b):\n        return a + b",
                score=0.7,
                metadata={"file": "calc.py", "class": "Calculator"},
                source="vector",
                original_rank=2
            ),
            SearchResult(
                content="import math\nprint('Hello, World!')",
                score=0.5,
                metadata={"file": "hello.py"},
                source="graph",
                original_rank=3
            )
        ]
    
    def test_initialization_mock_mode(self):
        """Test HaikuReRanker initialization in mock mode"""
        reranker = HaikuReRanker(api_key=None)
        
        assert reranker.mock_mode is True
        assert reranker.api_key is None
        assert reranker.cache == {}
        assert reranker.stats['requests'] == 0
        assert reranker.cache_ttl == 300
    
    def test_initialization_api_mode(self):
        """Test HaikuReRanker initialization with API key"""
        reranker = HaikuReRanker(api_key="test-key")
        
        assert reranker.mock_mode is False
        assert reranker.api_key == "test-key"
        assert reranker.base_url == "https://api.anthropic.com"
    
    def test_cache_key_generation(self, mock_reranker, sample_results):
        """Test cache key generation"""
        request = ReRankingRequest(
            query="test query",
            results=sample_results,
            context="test context"
        )
        
        key1 = mock_reranker._generate_cache_key(request)
        key2 = mock_reranker._generate_cache_key(request)
        
        # Same request should generate same key
        assert key1 == key2
        assert len(key1) == 16  # Truncated hash
        
        # Different query should generate different key
        request2 = ReRankingRequest(
            query="different query",
            results=sample_results,
            context="test context"
        )
        key3 = mock_reranker._generate_cache_key(request2)
        assert key1 != key3
    
    def test_cache_operations(self, mock_reranker, sample_results):
        """Test cache storage and retrieval"""
        request = ReRankingRequest(
            query="test query",
            results=sample_results
        )
        
        # Should not be cached initially
        cached = mock_reranker._check_cache(request)
        assert cached is None
        
        # Create and cache a result
        result = ReRankingResult(
            results=sample_results,
            processing_time=0.1,
            model_used="mock",
            confidence_score=0.8
        )
        mock_reranker._cache_result(request, result)
        
        # Should now be cached
        cached = mock_reranker._check_cache(request)
        assert cached is not None
        assert cached.confidence_score == 0.8
        assert cached.cache_hit is True
    
    @pytest.mark.asyncio
    async def test_mock_api_call(self, mock_reranker):
        """Test mock API call"""
        response = await mock_reranker._call_haiku_api("test prompt")
        
        assert "RANKING" in response
        assert "CONFIDENCE" in response
        assert isinstance(response, str)
    
    def test_build_ranking_prompt(self, mock_reranker, sample_results):
        """Test ranking prompt construction"""
        request = ReRankingRequest(
            query="fibonacci calculation",
            results=sample_results,
            context="Looking for mathematical functions"
        )
        
        prompt = mock_reranker._build_ranking_prompt(request)
        
        assert "fibonacci calculation" in prompt
        assert "Looking for mathematical functions" in prompt
        assert "RANKING" in prompt
        assert "CONFIDENCE" in prompt
        
        # Should include all results
        for result in sample_results:
            snippet = result.content[:50]
            assert any(snippet[:30] in prompt for snippet in [result.content[:30]])
    
    def test_parse_ranking_response(self, mock_reranker, sample_results):
        """Test parsing of ranking response"""
        mock_response = """RANKING (most relevant first):
1. Fibonacci function with recursive implementation
2. Calculator class with basic operations
3. Simple hello world script

CONFIDENCE: 0.85"""
        
        indices, confidence = mock_reranker._parse_ranking_response(mock_response, sample_results)
        
        assert len(indices) == len(sample_results)
        assert confidence == 0.85
        assert all(isinstance(i, int) for i in indices)
    
    def test_parse_fallback_response(self, mock_reranker, sample_results):
        """Test parsing of fallback response"""
        fallback_response = "FALLBACK_RANKING: Maintaining original order.\nCONFIDENCE: 0.50"
        
        indices, confidence = mock_reranker._parse_ranking_response(fallback_response, sample_results)
        
        assert indices == list(range(len(sample_results)))
        assert confidence == 0.5
    
    @pytest.mark.asyncio
    async def test_rerank_basic(self, mock_reranker, sample_results):
        """Test basic re-ranking functionality"""
        request = ReRankingRequest(
            query="fibonacci function",
            results=sample_results,
            max_results=2
        )
        
        result = await mock_reranker.rerank(request)
        
        assert isinstance(result, ReRankingResult)
        assert len(result.results) <= 2
        assert result.processing_time > 0
        assert result.model_used == "mock"
        assert 0 <= result.confidence_score <= 1
        assert result.cache_hit is False
    
    @pytest.mark.asyncio
    async def test_rerank_with_context(self, mock_reranker, sample_results):
        """Test re-ranking with context"""
        request = ReRankingRequest(
            query="mathematical calculation",
            results=sample_results,
            context="Looking for algorithms and data structures"
        )
        
        result = await mock_reranker.rerank(request)
        
        assert len(result.results) == len(sample_results)
        
        # Check that metadata was added
        for ranked_result in result.results:
            assert ranked_result.metadata.get('reranked') is True
            assert 'original_rank' in ranked_result.metadata
            assert 'new_rank' in ranked_result.metadata
            assert 'ranking_confidence' in ranked_result.metadata
    
    @pytest.mark.asyncio
    async def test_rerank_caching(self, mock_reranker, sample_results):
        """Test that re-ranking results are cached"""
        request = ReRankingRequest(
            query="test query",
            results=sample_results
        )
        
        # First call
        result1 = await mock_reranker.rerank(request)
        assert result1.cache_hit is False
        
        # Second call should be cached
        result2 = await mock_reranker.rerank(request)
        assert result2.cache_hit is True
        assert result2.confidence_score == result1.confidence_score
    
    @pytest.mark.asyncio
    async def test_rerank_simple_interface(self, mock_reranker):
        """Test simplified re-ranking interface"""
        results = [
            {"content": "fibonacci function", "score": 0.9, "metadata": {"file": "math.py"}},
            {"content": "calculator class", "score": 0.7, "metadata": {"file": "calc.py"}},
            {"content": "hello world", "score": 0.5, "metadata": {"file": "hello.py"}}
        ]
        
        reranked = await mock_reranker.rerank_simple(
            query="mathematical functions",
            results=results,
            max_results=2
        )
        
        assert len(reranked) == 2
        assert all(isinstance(result, dict) for result in reranked)
        
        # Check that original data is preserved and metadata added
        for result in reranked:
            assert "content" in result
            assert "score" in result
            assert "metadata" in result
            assert result["metadata"].get("reranked") is True
    
    @pytest.mark.asyncio
    async def test_rerank_large_result_set(self, mock_reranker):
        """Test re-ranking with many results (should limit to 20)"""
        # Create 25 results
        large_results = []
        for i in range(25):
            large_results.append(SearchResult(
                content=f"function_{i}()",
                score=0.8 - (i * 0.01),
                metadata={"index": i},
                original_rank=i + 1
            ))
        
        request = ReRankingRequest(
            query="function search",
            results=large_results
        )
        
        result = await mock_reranker.rerank(request)
        
        # Should limit processing to 20 results
        assert len(result.results) <= 20
    
    def test_stats_tracking(self, mock_reranker):
        """Test statistics tracking"""
        initial_stats = mock_reranker.get_stats()
        assert initial_stats['requests'] == 0
        assert initial_stats['cache_hits'] == 0
        assert initial_stats['cache_hit_rate'] == 0
        
        # Mock a request to update stats
        mock_reranker.stats['requests'] = 5
        mock_reranker.stats['cache_hits'] = 2
        
        updated_stats = mock_reranker.get_stats()
        assert updated_stats['requests'] == 5
        assert updated_stats['cache_hits'] == 2
        assert updated_stats['cache_hit_rate'] == 40.0
        assert updated_stats['mock_mode'] is True
    
    def test_cache_cleanup(self, mock_reranker):
        """Test cache cleanup functionality"""
        # Add some entries to cache
        mock_reranker.cache = {
            'key1': ('result1', time.time() - 400),  # Expired
            'key2': ('result2', time.time() - 100),  # Not expired
            'key3': ('result3', time.time() - 500)   # Expired
        }
        
        mock_reranker._clean_cache()
        
        # Should remove expired entries
        assert len(mock_reranker.cache) == 1
        assert 'key2' in mock_reranker.cache
    
    def test_clear_cache(self, mock_reranker):
        """Test manual cache clearing"""
        mock_reranker.cache = {'test': ('data', time.time())}
        assert len(mock_reranker.cache) == 1
        
        mock_reranker.clear_cache()
        assert len(mock_reranker.cache) == 0


class TestConvenienceFunctions:
    """Test convenience functions and global instances"""
    
    @pytest.mark.asyncio
    async def test_rerank_search_results_function(self):
        """Test convenience function for re-ranking"""
        results = [
            {"content": "function test", "score": 0.8},
            {"content": "class example", "score": 0.6},
            {"content": "variable def", "score": 0.4}
        ]
        
        reranked = await rerank_search_results(
            query="programming examples",
            results=results,
            max_results=2
        )
        
        assert len(reranked) == 2
        assert all("reranking_confidence" in result for result in reranked)
    
    def test_global_reranker_instance(self):
        """Test global re-ranker instance management"""
        # First call creates instance
        reranker1 = get_global_reranker()
        assert isinstance(reranker1, HaikuReRanker)
        assert reranker1.mock_mode is True
        
        # Second call returns same instance
        reranker2 = get_global_reranker()
        assert reranker1 is reranker2
    
    def test_global_reranker_with_api_key(self):
        """Test global re-ranker with API key"""
        # Clear global instance first
        import infrastructure.haiku_reranker as hr
        hr._global_reranker = None
        
        reranker = get_global_reranker(api_key="test-key")
        assert reranker.api_key == "test-key"
        assert reranker.mock_mode is False


@pytest.mark.benchmark
class TestHaikuReRankerPerformance:
    """Performance tests for HaikuReRanker"""
    
    @pytest.fixture
    def performance_reranker(self):
        """Create re-ranker for performance testing"""
        return HaikuReRanker(api_key=None, cache_ttl=60)
    
    @pytest.fixture
    def large_result_set(self):
        """Create large result set for performance testing"""
        results = []
        for i in range(100):
            results.append(SearchResult(
                content=f"def function_{i}():\n    return {i} * 2",
                score=0.9 - (i * 0.001),
                metadata={"function_id": i, "file": f"file_{i % 10}.py"},
                original_rank=i + 1
            ))
        return results
    
    @pytest.mark.asyncio
    async def test_reranking_speed_target(self, performance_reranker, large_result_set):
        """ROADMAP EXIT CRITERIA: Sub-100ms re-ranking for up to 20 results"""
        # Test with 20 results (should be under 100ms)
        request = ReRankingRequest(
            query="function implementation",
            results=large_result_set[:20]
        )
        
        start_time = time.perf_counter()
        result = await performance_reranker.rerank(request)
        processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # ROADMAP EXIT CRITERIA: <100ms for 20 results
        assert processing_time < 100, f"Re-ranking took {processing_time:.1f}ms, should be <100ms"
        assert len(result.results) == 20
        assert result.processing_time < 0.1  # Internal timing should also be fast
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, performance_reranker, large_result_set):
        """Test that caching provides significant speedup"""
        request = ReRankingRequest(
            query="performance test",
            results=large_result_set[:10]
        )
        
        # First call (not cached)
        start_time = time.perf_counter()
        result1 = await performance_reranker.rerank(request)
        uncached_time = time.perf_counter() - start_time
        
        # Second call (cached)
        start_time = time.perf_counter()
        result2 = await performance_reranker.rerank(request)
        cached_time = time.perf_counter() - start_time
        
        assert result2.cache_hit is True
        assert cached_time < uncached_time  # Cache should be faster
        assert cached_time < 0.01  # Cache lookup should be very fast
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, performance_reranker):
        """Test batch processing doesn't degrade performance significantly"""
        queries = ["query 1", "query 2", "query 3", "query 4", "query 5"]
        
        # Create results for each query
        all_results = []
        for i, query in enumerate(queries):
            results = []
            for j in range(5):  # 5 results per query
                results.append(SearchResult(
                    content=f"{query} result {j}",
                    score=0.8 - (j * 0.1),
                    metadata={"query_id": i, "result_id": j}
                ))
            all_results.append(results)
        
        # Time sequential processing
        start_time = time.perf_counter()
        for query, results in zip(queries, all_results):
            await performance_reranker.rerank_simple(query, 
                [{"content": r.content, "score": r.score} for r in results])
        sequential_time = time.perf_counter() - start_time
        
        # Verify reasonable performance
        assert sequential_time < 1.0  # Should process 5 queries in under 1 second
        
        # Check cache efficiency
        stats = performance_reranker.get_stats()
        assert stats['requests'] == 5


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in various scenarios"""
    reranker = HaikuReRanker(api_key="invalid-key")  # Will cause API errors
    
    results = [SearchResult("test content", 0.5, {})]
    request = ReRankingRequest("test query", results)
    
    # Should handle API errors gracefully
    result = await reranker.rerank(request)
    
    assert isinstance(result, ReRankingResult)
    assert result.model_used == "fallback"
    assert result.confidence_score < 1.0


@pytest.mark.integration
class TestHaikuReRankerIntegration:
    """Integration tests for real-world usage scenarios"""
    
    @pytest.mark.asyncio
    async def test_code_search_reranking(self):
        """Test re-ranking of code search results"""
        code_results = [
            {
                "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "score": 0.85,
                "metadata": {"file": "algorithms.py", "function": "fibonacci", "language": "python"}
            },
            {
                "content": "class Calculator:\n    def __init__(self):\n        pass\n    def add(self, a, b):\n        return a + b",
                "score": 0.75,
                "metadata": {"file": "calculator.py", "class": "Calculator", "language": "python"}
            },
            {
                "content": "print('Hello, World!')",
                "score": 0.60,
                "metadata": {"file": "hello.py", "language": "python"}
            }
        ]
        
        reranked = await rerank_search_results(
            query="recursive algorithm implementation",
            results=code_results,
            context="Looking for algorithmic implementations with recursion",
            max_results=3
        )
        
        assert len(reranked) == 3
        
        # Fibonacci should likely be ranked higher for recursive algorithm query
        # (though in mock mode, ranking might not change dramatically)
        fibonacci_result = next(r for r in reranked if "fibonacci" in r["content"])
        assert fibonacci_result["metadata"]["reranked"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])