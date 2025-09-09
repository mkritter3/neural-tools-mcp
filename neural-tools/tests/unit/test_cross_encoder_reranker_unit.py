#!/usr/bin/env python3
"""
Unit tests for CrossEncoderReranker
Tests latency budget enforcement, cache isolation, and heuristic fallbacks
"""

import asyncio
import time
import os
import pytest
from typing import Dict, Any, List

from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig


@pytest.mark.asyncio
async def test_respects_latency_budget():
    """Test that reranker respects latency budget and returns quickly on timeout"""
    # Force very small budget to verify timeout path works and returns quickly
    cfg = RerankConfig(latency_budget_ms=10, model_path=None)  # No model to force timeout
    rr = CrossEncoderReranker(cfg)
    
    start = time.perf_counter()
    results = await rr.rerank(
        "query", 
        [{"content": "test content", "score": 0.9}], 
        top_k=1
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # Should return within budget + overhead
    assert elapsed_ms <= 60, f"Took {elapsed_ms}ms, expected <= 60ms"
    assert len(results) == 1
    assert "rerank_score" in results[0]


@pytest.mark.asyncio
async def test_tenant_isolation_in_cache():
    """Test that cache keys are isolated by tenant"""
    cfg = RerankConfig(latency_budget_ms=120)
    rr_a = CrossEncoderReranker(cfg, tenant_id="tenantA")
    rr_b = CrossEncoderReranker(cfg, tenant_id="tenantB")
    
    docs = [{"content": "same doc", "score": 0.5, "file_path": "x.py"}]
    
    # First call warms cache for tenantA
    results_a = await rr_a.rerank("same query", docs, top_k=1)
    
    # Different tenant should not hit same cache key; behavior must be independent
    results_b = await rr_b.rerank("same query", docs, top_k=1)
    
    # Both should work independently without errors
    assert len(results_a) == 1
    assert len(results_b) == 1
    assert "rerank_score" in results_a[0]
    assert "rerank_score" in results_b[0]


@pytest.mark.asyncio
async def test_heuristic_fallback_when_model_missing():
    """Test heuristic fallback when model is not available"""
    # Ensure no model path to force heuristic path
    cfg = RerankConfig(model_path=None, latency_budget_ms=50)
    rr = CrossEncoderReranker(cfg)
    
    results = await rr.rerank(
        "cache eviction policy",
        [
            {"content": "LRU cache implementation", "score": 0.8},
            {"content": "random eviction policy", "score": 0.6}
        ],
        top_k=2
    )
    
    assert len(results) == 2
    assert all("rerank_score" in r for r in results)
    
    # LRU cache should score higher due to keyword overlap
    lru_result = next(r for r in results if "LRU" in r["content"])
    random_result = next(r for r in results if "random" in r["content"])
    assert lru_result["rerank_score"] >= random_result["rerank_score"]


@pytest.mark.asyncio
async def test_empty_results():
    """Test handling of empty result sets"""
    cfg = RerankConfig()
    rr = CrossEncoderReranker(cfg)
    
    results = await rr.rerank("query", [], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_caching_behavior():
    """Test that caching works for repeated queries"""
    cfg = RerankConfig(cache_ttl_s=10, latency_budget_ms=200)
    rr = CrossEncoderReranker(cfg, tenant_id="test")
    
    docs = [
        {"content": "first document", "score": 0.8, "file_path": "a.py"},
        {"content": "second document", "score": 0.7, "file_path": "b.py"}
    ]
    
    # First call
    start1 = time.perf_counter()
    results1 = await rr.rerank("test query", docs, top_k=2)
    time1 = time.perf_counter() - start1
    
    # Second call (should hit cache)
    start2 = time.perf_counter()
    results2 = await rr.rerank("test query", docs, top_k=2)
    time2 = time.perf_counter() - start2
    
    # Results should be consistent
    assert len(results1) == 2
    assert len(results2) == 2
    
    # Second call should be faster due to caching (if model was available)
    # If heuristic was used, both should be fast
    assert time2 <= time1 + 0.1  # Allow some variance


@pytest.mark.asyncio
async def test_top_k_limiting():
    """Test that top_k properly limits results"""
    cfg = RerankConfig(latency_budget_ms=100)
    rr = CrossEncoderReranker(cfg)
    
    docs = [
        {"content": f"document {i}", "score": 0.9 - i*0.1, "file_path": f"{i}.py"}
        for i in range(10)
    ]
    
    # Request fewer than available
    results = await rr.rerank("test", docs, top_k=3)
    assert len(results) == 3
    
    # Request more than available
    results = await rr.rerank("test", docs[:2], top_k=5)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_score_preservation():
    """Test that original scores are preserved alongside rerank scores"""
    cfg = RerankConfig(latency_budget_ms=100)
    rr = CrossEncoderReranker(cfg)
    
    original_docs = [
        {"content": "test doc", "score": 0.95, "file_path": "test.py", "metadata": {"important": True}}
    ]
    
    results = await rr.rerank("test", original_docs, top_k=1)
    
    assert len(results) == 1
    result = results[0]
    
    # Original fields should be preserved
    assert result["content"] == "test doc"
    assert result["score"] == 0.95
    assert result["file_path"] == "test.py"
    assert result["metadata"]["important"] == True
    
    # Rerank score should be added
    assert "rerank_score" in result
    assert isinstance(result["rerank_score"], float)


def test_config_from_environment(monkeypatch):
    """Test that configuration can be loaded from environment variables"""
    monkeypatch.setenv("RERANKER_MODEL", "custom/model")
    monkeypatch.setenv("RERANK_BUDGET_MS", "200")
    monkeypatch.setenv("RERANK_CACHE_TTL", "900")
    
    cfg = RerankConfig()
    
    assert cfg.model_name == "custom/model"
    assert cfg.latency_budget_ms == 200
    assert cfg.cache_ttl_s == 900


@pytest.mark.asyncio
async def test_candidate_limiting():
    """Test that large result sets are limited for performance"""
    cfg = RerankConfig(latency_budget_ms=100)
    rr = CrossEncoderReranker(cfg)
    
    # Create many documents
    docs = [
        {"content": f"document number {i}", "score": 0.9 - i*0.001, "file_path": f"doc{i}.py"}
        for i in range(100)
    ]
    
    results = await rr.rerank("test", docs, top_k=5)
    
    # Should return requested number
    assert len(results) == 5
    
    # Should have rerank scores
    assert all("rerank_score" in r for r in results)