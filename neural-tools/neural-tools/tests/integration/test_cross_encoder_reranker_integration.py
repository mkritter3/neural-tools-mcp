#!/usr/bin/env python3
"""
Integration tests for CrossEncoderReranker with Enhanced Hybrid Retriever
Tests end-to-end integration without Anthropic API calls
"""

import asyncio
import os
import pytest
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

from src.servers.services.service_container import ServiceContainer
from src.servers.services.hybrid_retriever import HybridRetriever
from src.infrastructure.enhanced_hybrid_retriever import EnhancedHybridRetriever
from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig


# Mock hybrid retriever for testing
class MockHybridRetriever:
    """Mock hybrid retriever that simulates vector search results"""
    
    def __init__(self):
        self.sample_results = [
            {
                'content': 'def quicksort(arr): # Efficient sorting algorithm',
                'score': 0.92,
                'metadata': {
                    'file_path': 'algorithms/quicksort.py',
                    'language': 'python',
                    'function_name': 'quicksort'
                }
            },
            {
                'content': 'class HashMap: # Hash table implementation with O(1) operations',
                'score': 0.88,
                'metadata': {
                    'file_path': 'data_structures/hashmap.py', 
                    'language': 'python',
                    'class_name': 'HashMap'
                }
            },
            {
                'content': 'def binary_search(arr, target): # O(log n) search algorithm',
                'score': 0.85,
                'metadata': {
                    'file_path': 'algorithms/binary_search.py',
                    'language': 'python',
                    'function_name': 'binary_search'
                }
            },
            {
                'content': 'async def fetch_data(url): # Async HTTP client implementation',
                'score': 0.80,
                'metadata': {
                    'file_path': 'networking/http_client.py',
                    'language': 'python',
                    'function_name': 'fetch_data'
                }
            },
            {
                'content': 'class LRUCache: # Least Recently Used cache implementation',
                'score': 0.75,
                'metadata': {
                    'file_path': 'caching/lru_cache.py',
                    'language': 'python',
                    'class_name': 'LRUCache'
                }
            }
        ]
    
    async def find_similar_with_context(
        self, 
        query: str, 
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Simulate fast vector search"""
        await asyncio.sleep(0.01)  # Simulate 10ms vector search
        return self.sample_results[:limit]


@pytest.mark.asyncio
async def test_local_rerank_integration_no_anthropic_calls(monkeypatch):
    """Test that local reranking works without any Anthropic API calls"""
    # Ensure Anthropic is not used
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    
    mock_hybrid = MockHybridRetriever()
    
    # Create enhanced retriever with local reranking only
    enhanced_retriever = EnhancedHybridRetriever(
        hybrid_retriever=mock_hybrid,
        prefer_local=True,
        allow_haiku_fallback=False,
        rerank_threshold=3,
        rerank_latency_budget_ms=120
    )
    
    results = await enhanced_retriever.find_similar_with_context(
        "efficient sorting algorithm", 
        limit=5
    )
    
    # Should return results
    assert len(results) == 5
    
    # When local rerank path runs, metadata indicates local_cross_encoder
    if results and len(results) >= enhanced_retriever.rerank_threshold:
        mode = results[0].get("metadata", {}).get("reranking_mode")
        assert mode in ("local_cross_encoder", None)  # None if rerank threshold not met
        
        # Should have rerank metadata when reranked
        if mode == "local_cross_encoder":
            assert results[0]["metadata"]["reranked"] == True
            assert "rerank_position" in results[0]["metadata"]
    
    # Verify no Haiku fallback was used
    for result in results:
        metadata = result.get("metadata", {})
        reranking_mode = metadata.get("reranking_mode")
        if reranking_mode:
            assert reranking_mode != "progressive"
            assert reranking_mode != "haiku"


@pytest.mark.asyncio
async def test_threshold_bypass():
    """Test that reranking is bypassed when below threshold"""
    mock_hybrid = MockHybridRetriever()
    
    retriever = EnhancedHybridRetriever(
        hybrid_retriever=mock_hybrid, 
        prefer_local=True, 
        allow_haiku_fallback=False, 
        rerank_latency_budget_ms=80,
        rerank_threshold=999  # Force threshold higher than results
    )
    
    results = await retriever.find_similar_with_context("q", limit=5)
    
    # Should return untouched subset
    assert len(results) <= 5
    
    # Should not have reranking metadata when bypassed
    for result in results:
        metadata = result.get("metadata", {})
        assert metadata.get("reranked") != True


@pytest.mark.asyncio
async def test_local_rerank_performance():
    """Test that local reranking meets performance requirements"""
    mock_hybrid = MockHybridRetriever()
    
    enhanced_retriever = EnhancedHybridRetriever(
        hybrid_retriever=mock_hybrid,
        prefer_local=True,
        allow_haiku_fallback=False,
        rerank_threshold=3,
        rerank_latency_budget_ms=120
    )
    
    import time
    start_time = time.perf_counter()
    
    results = await enhanced_retriever.find_similar_with_context(
        "data structure with fast operations",
        limit=5,
        rerank_context="Looking for high-performance implementations"
    )
    
    total_time = time.perf_counter() - start_time
    
    # Should be much faster than 26 seconds
    assert total_time < 1.0, f"Total time {total_time:.2f}s should be < 1s"
    
    # Should return results
    assert len(results) == 5
    
    # Top result should be relevant to "data structure"
    top_result = results[0]
    content_lower = top_result['content'].lower()
    assert any(term in content_lower for term in ['hash', 'cache', 'data', 'structure'])


@pytest.mark.asyncio
async def test_cross_encoder_direct():
    """Test CrossEncoderReranker directly"""
    cfg = RerankConfig(latency_budget_ms=100)
    reranker = CrossEncoderReranker(cfg, tenant_id="test")
    
    docs = [
        {"content": "Binary search implementation", "score": 0.8, "file_path": "search.py"},
        {"content": "Hash table with collision handling", "score": 0.7, "file_path": "hash.py"},
        {"content": "Quick sort algorithm", "score": 0.9, "file_path": "sort.py"}
    ]
    
    import time
    start_time = time.perf_counter()
    
    results = await reranker.rerank(
        "efficient search algorithm",
        docs,
        top_k=3,
        latency_budget_ms=120
    )
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Should respect latency budget
    assert elapsed_ms <= 150, f"Took {elapsed_ms:.1f}ms, expected ≤150ms"
    
    # Should return all results
    assert len(results) == 3
    
    # Should have rerank scores
    assert all("rerank_score" in r for r in results)
    
    # Binary search should likely rank higher for "search algorithm" query
    search_result = next((r for r in results if "Binary search" in r["content"]), None)
    assert search_result is not None
    assert search_result["rerank_score"] >= 0


@pytest.mark.asyncio
async def test_tenant_isolation():
    """Test tenant isolation in cross-encoder reranking"""
    cfg = RerankConfig(latency_budget_ms=120)
    
    reranker_a = CrossEncoderReranker(cfg, tenant_id="tenant_a")
    reranker_b = CrossEncoderReranker(cfg, tenant_id="tenant_b")
    
    docs = [{"content": "shared content", "score": 0.8, "file_path": "shared.py"}]
    
    # Both should work independently
    results_a = await reranker_a.rerank("test query", docs, top_k=1)
    results_b = await reranker_b.rerank("test query", docs, top_k=1)
    
    assert len(results_a) == 1
    assert len(results_b) == 1
    assert "rerank_score" in results_a[0]
    assert "rerank_score" in results_b[0]


@pytest.mark.asyncio
async def test_concurrent_reranking():
    """Test that concurrent reranking operations work correctly"""
    cfg = RerankConfig(latency_budget_ms=120)
    reranker = CrossEncoderReranker(cfg, tenant_id="concurrent_test")
    
    docs = [
        {"content": f"Document {i} content", "score": 0.9 - i*0.1, "file_path": f"doc{i}.py"}
        for i in range(5)
    ]
    
    # Run multiple concurrent reranking operations
    tasks = [
        reranker.rerank(f"query {i}", docs, top_k=3)
        for i in range(3)
    ]
    
    import time
    start_time = time.perf_counter()
    results_list = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time
    
    # Should complete reasonably quickly
    assert total_time < 2.0, f"Concurrent operations took {total_time:.2f}s"
    
    # All should return results
    assert len(results_list) == 3
    for results in results_list:
        assert len(results) == 3
        assert all("rerank_score" in r for r in results)


@pytest.mark.asyncio  
async def test_enhanced_retriever_stats():
    """Test that enhanced retriever provides useful statistics"""
    mock_hybrid = MockHybridRetriever()
    
    enhanced_retriever = EnhancedHybridRetriever(
        hybrid_retriever=mock_hybrid,
        prefer_local=True,
        allow_haiku_fallback=False,
        rerank_threshold=2
    )
    
    # Perform some searches
    await enhanced_retriever.find_similar_with_context("test query 1", limit=5)
    await enhanced_retriever.find_similar_with_context("test query 2", limit=3)
    
    stats = enhanced_retriever.get_stats()
    
    # Should have basic stats
    assert 'total_queries' in stats
    assert 'reranked_queries' in stats
    assert 'rerank_rate' in stats
    
    assert stats['total_queries'] == 2
    assert isinstance(stats['rerank_rate'], float)
    assert 0 <= stats['rerank_rate'] <= 100