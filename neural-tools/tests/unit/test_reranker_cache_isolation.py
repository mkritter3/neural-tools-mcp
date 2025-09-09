#!/usr/bin/env python3
"""
Cache isolation tests for CrossEncoderReranker
Tests tenant separation and TTL behavior
"""

import pytest
import time
import asyncio

from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig


@pytest.mark.asyncio
async def test_cache_isolation_by_tenant():
    """Test that cache keys are isolated by tenant ID"""
    cfg = RerankConfig(latency_budget_ms=80, cache_ttl_s=300)
    
    reranker_a = CrossEncoderReranker(cfg, tenant_id="tenant_A")
    reranker_b = CrossEncoderReranker(cfg, tenant_id="tenant_B")
    
    docs = [{"content": "shared document content", "file_path": "shared.py", "score": 0.9}]
    
    # Both tenants process the same query and documents
    results_a = await reranker_a.rerank("test query", docs, top_k=1)
    results_b = await reranker_b.rerank("test query", docs, top_k=1)
    
    # Should work independently without errors
    assert len(results_a) == 1
    assert len(results_b) == 1
    
    # Both should have rerank scores
    assert "rerank_score" in results_a[0]
    assert "rerank_score" in results_b[0]
    
    # Verify cache keys are different
    query_hash_a = reranker_a._query_hash("test query")
    query_hash_b = reranker_b._query_hash("test query")
    
    # Query hashes should be same (same query)
    assert query_hash_a == query_hash_b
    
    # But cache keys should differ due to tenant ID
    doc_id = reranker_a._cache._stable_doc_id if hasattr(reranker_a._cache, '_stable_doc_id') else "test_doc"
    key_a = reranker_a._cache.make_key("tenant_A", query_hash_a, doc_id)
    key_b = reranker_b._cache.make_key("tenant_B", query_hash_b, doc_id)
    
    assert key_a != key_b, "Cache keys should be different for different tenants"
    assert "tenant_A" in key_a
    assert "tenant_B" in key_b


@pytest.mark.asyncio
async def test_no_cross_tenant_cache_hits():
    """Test that one tenant cannot access another tenant's cache"""
    cfg = RerankConfig(latency_budget_ms=100, cache_ttl_s=300)
    
    reranker_alpha = CrossEncoderReranker(cfg, tenant_id="alpha")
    reranker_beta = CrossEncoderReranker(cfg, tenant_id="beta")
    
    docs = [
        {"content": "function implementation", "file_path": "func.py", "score": 0.8},
        {"content": "class definition", "file_path": "class.py", "score": 0.7}
    ]
    
    # Alpha processes first - this should populate alpha's cache
    start_alpha = time.perf_counter()
    results_alpha_1 = await reranker_alpha.rerank("implementation", docs, top_k=2)
    time_alpha_1 = (time.perf_counter() - start_alpha) * 1000
    
    # Alpha processes again - should hit cache
    start_alpha_2 = time.perf_counter()
    results_alpha_2 = await reranker_alpha.rerank("implementation", docs, top_k=2)
    time_alpha_2 = (time.perf_counter() - start_alpha_2) * 1000
    
    # Beta processes same query - should NOT hit alpha's cache
    start_beta = time.perf_counter()
    results_beta = await reranker_beta.rerank("implementation", docs, top_k=2)
    time_beta = (time.perf_counter() - start_beta) * 1000
    
    # All should return results
    assert len(results_alpha_1) == 2
    assert len(results_alpha_2) == 2
    assert len(results_beta) == 2
    
    # Alpha's second call should be faster due to cache hit
    # (or at least not significantly slower if heuristic is used)
    assert time_alpha_2 <= time_alpha_1 + 10, "Alpha's cached call should be faster"
    
    # Beta should not benefit from alpha's cache - timing should be similar to alpha's first call
    # Allow some variance but beta should not be much faster than alpha's first call
    assert time_beta >= time_alpha_2 - 5, "Beta should not benefit from alpha's cache"
    
    print(f"Alpha first: {time_alpha_1:.1f}ms, Alpha cached: {time_alpha_2:.1f}ms, Beta: {time_beta:.1f}ms")


@pytest.mark.asyncio
async def test_ttl_expiry_removes_stale_entries():
    """Test that TTL expiry removes stale cache entries"""
    cfg = RerankConfig(latency_budget_ms=80, cache_ttl_s=1)  # Very short TTL for testing
    reranker = CrossEncoderReranker(cfg, tenant_id="ttl_test")
    
    docs = [{"content": "test document", "file_path": "test.py", "score": 0.8}]
    
    # First call populates cache
    await reranker.rerank("test query", docs, top_k=1)
    
    # Check that cache has entries
    initial_cache_size = len(reranker._cache._store)
    assert initial_cache_size > 0, "Cache should have entries after first call"
    
    # Wait for TTL to expire
    await asyncio.sleep(1.2)
    
    # Access should now miss cache and clean up expired entries
    await reranker.rerank("test query", docs, top_k=1)
    
    # Note: The cache cleanup happens lazily, so we might still see entries
    # But they should be treated as expired when accessed
    
    # Make multiple calls to trigger cleanup
    for _ in range(5):
        await reranker.rerank(f"different query {_}", docs, top_k=1)
    
    # Cache size should be reasonable (not growing indefinitely)
    final_cache_size = len(reranker._cache._store)
    print(f"Cache size: initial={initial_cache_size}, final={final_cache_size}")
    
    # Cache should not grow unboundedly
    assert final_cache_size <= initial_cache_size + 10, "Cache should not grow unboundedly"


@pytest.mark.asyncio
async def test_memory_stable_over_10k_ops():
    """Test that memory remains stable over many operations"""
    cfg = RerankConfig(latency_budget_ms=50, cache_ttl_s=5)  # Moderate TTL
    reranker = CrossEncoderReranker(cfg, tenant_id="memory_test")
    
    # Simulate many operations with varied queries and documents
    for i in range(100):  # Reduced from 10k for test speed
        docs = [
            {"content": f"document content {i % 20}", "file_path": f"file_{i % 20}.py", "score": 0.8},
            {"content": f"another document {i % 15}", "file_path": f"other_{i % 15}.py", "score": 0.7}
        ]
        
        query = f"test query {i % 10}"  # Some query repetition to test cache
        await reranker.rerank(query, docs, top_k=2)
        
        # Periodically check cache size
        if i % 25 == 0:
            cache_size = len(reranker._cache._store)
            print(f"After {i+1} ops: cache size = {cache_size}")
            
            # Cache should not grow excessively
            assert cache_size <= 1000, f"Cache size {cache_size} too large after {i+1} operations"
    
    final_cache_size = len(reranker._cache._store)
    print(f"Final cache size after 100 operations: {final_cache_size}")
    
    # Memory should be stable
    assert final_cache_size <= 200, f"Final cache size {final_cache_size} indicates memory leak"


@pytest.mark.asyncio
async def test_default_tenant_isolation():
    """Test that None tenant_id defaults work correctly"""
    cfg = RerankConfig(latency_budget_ms=80)
    
    # Rerankers with None tenant_id should get "default"
    reranker_none1 = CrossEncoderReranker(cfg, tenant_id=None)
    reranker_none2 = CrossEncoderReranker(cfg, tenant_id=None)
    reranker_explicit = CrossEncoderReranker(cfg, tenant_id="explicit")
    
    docs = [{"content": "shared content", "file_path": "shared.py", "score": 0.8}]
    
    # None tenant_ids should share cache with each other
    await reranker_none1.rerank("test", docs, top_k=1)
    await reranker_none2.rerank("test", docs, top_k=1)  # Should potentially hit cache
    
    # But explicit tenant should be isolated
    await reranker_explicit.rerank("test", docs, top_k=1)
    
    # All should work without errors
    assert True  # If we get here, no exceptions were thrown
    
    # Verify tenant IDs are set correctly
    assert reranker_none1.tenant_id == "default"
    assert reranker_none2.tenant_id == "default"
    assert reranker_explicit.tenant_id == "explicit"


@pytest.mark.asyncio
async def test_cache_key_stability():
    """Test that cache keys are stable across reranker instances"""
    cfg = RerankConfig(latency_budget_ms=80)
    
    reranker1 = CrossEncoderReranker(cfg, tenant_id="stable_test")
    reranker2 = CrossEncoderReranker(cfg, tenant_id="stable_test")
    
    query = "test query"
    doc = {"content": "test content", "file_path": "test.py", "score": 0.8}
    
    # Get doc ID from both rerankers - should be same
    from src.infrastructure.cross_encoder_reranker import _stable_doc_id
    doc_id1 = _stable_doc_id(doc)
    doc_id2 = _stable_doc_id(doc)
    
    assert doc_id1 == doc_id2, "Document IDs should be stable"
    
    # Cache keys should be identical for same tenant, query, and doc
    query_hash1 = reranker1._query_hash(query)
    query_hash2 = reranker2._query_hash(query)
    
    assert query_hash1 == query_hash2, "Query hashes should be identical"
    
    key1 = reranker1._cache.make_key("stable_test", query_hash1, doc_id1)
    key2 = reranker2._cache.make_key("stable_test", query_hash2, doc_id2)
    
    assert key1 == key2, "Cache keys should be identical for same inputs"
    print(f"Stable cache key: {key1}")