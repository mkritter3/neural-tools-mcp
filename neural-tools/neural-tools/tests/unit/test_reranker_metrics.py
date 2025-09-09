#!/usr/bin/env python3
"""
Metrics collection tests for CrossEncoderReranker
Tests p50/p95 tracking and skip counters
"""

import pytest
import asyncio
import time
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig


@pytest.mark.asyncio
async def test_basic_metrics_collection():
    """Test that basic metrics are collected correctly"""
    cfg = RerankConfig(latency_budget_ms=100, cache_ttl_s=300)
    reranker = CrossEncoderReranker(cfg, tenant_id="metrics_test")
    
    # Initial metrics should be zero
    initial_stats = reranker.get_stats()
    assert initial_stats['total_requests'] == 0
    assert initial_stats['heuristic_used'] == 0
    assert initial_stats['cache_hits'] == 0
    assert initial_stats['cache_misses'] == 0
    
    # Perform some reranking operations
    docs = [
        {"content": "test document 1", "score": 0.9, "file_path": "test1.py"},
        {"content": "test document 2", "score": 0.8, "file_path": "test2.py"},
    ]
    
    # First request
    await reranker.rerank("test query", docs, top_k=2)
    
    stats_after_1 = reranker.get_stats()
    assert stats_after_1['total_requests'] == 1
    # Since sentence-transformers isn't available, should use heuristic
    assert stats_after_1['heuristic_used'] >= 1
    # Should have cache misses (first time)
    assert stats_after_1['cache_misses'] >= 0
    
    # Second request (same query/docs) - should potentially hit cache
    await reranker.rerank("test query", docs, top_k=2)
    
    stats_after_2 = reranker.get_stats()
    assert stats_after_2['total_requests'] == 2
    assert stats_after_2['heuristic_used'] >= 1  # At least one heuristic call


@pytest.mark.asyncio
async def test_latency_percentiles_tracking():
    """Test that p50 and p95 latency percentiles are tracked"""
    cfg = RerankConfig(latency_budget_ms=100)
    reranker = CrossEncoderReranker(cfg, tenant_id="latency_test")
    
    docs = [{"content": f"doc {i}", "score": 0.8, "file_path": f"doc{i}.py"} for i in range(5)]
    
    # Perform multiple operations to get latency distribution
    for i in range(10):
        await reranker.rerank(f"query {i}", docs, top_k=3)
    
    stats = reranker.get_stats()
    
    # Should have latency metrics
    assert stats['total_requests'] == 10
    assert 'latency_p50_ms' in stats
    assert 'latency_p95_ms' in stats
    assert stats['latency_window_size'] > 0
    
    # p95 should be >= p50 (unless all latencies are identical)
    p50 = stats['latency_p50_ms']
    p95 = stats['latency_p95_ms']
    
    if isinstance(p50, (int, float)) and isinstance(p95, (int, float)):
        assert p95 >= p50, f"p95 ({p95}ms) should be >= p50 ({p50}ms)"
    
    # Latencies should be reasonable (heuristic is very fast)
    if isinstance(p95, (int, float)):
        assert p95 < 50, f"p95 latency {p95}ms seems too high for heuristic mode"


@pytest.mark.asyncio
async def test_cache_hit_rate_calculation():
    """Test accurate cache hit rate calculation"""
    cfg = RerankConfig(latency_budget_ms=100, cache_ttl_s=600)  # Long TTL
    reranker = CrossEncoderReranker(cfg, tenant_id="cache_test")
    
    docs = [{"content": "cached document", "score": 0.9, "file_path": "cached.py"}]
    
    # First call - should be cache miss
    await reranker.rerank("cache test query", docs, top_k=1)
    
    stats_1 = reranker.get_stats()
    total_ops_1 = stats_1['cache_hits'] + stats_1['cache_misses']
    
    if total_ops_1 > 0:  # Only test if cache operations occurred
        hit_rate_1 = stats_1['cache_hit_rate']
        expected_rate_1 = stats_1['cache_hits'] / total_ops_1
        assert abs(hit_rate_1 - expected_rate_1) < 0.01, f"Hit rate {hit_rate_1} != expected {expected_rate_1}"
    
    # Second call - might hit cache depending on implementation
    await reranker.rerank("cache test query", docs, top_k=1)
    
    stats_2 = reranker.get_stats()
    total_ops_2 = stats_2['cache_hits'] + stats_2['cache_misses']
    
    if total_ops_2 > 0:
        hit_rate_2 = stats_2['cache_hit_rate']
        expected_rate_2 = stats_2['cache_hits'] / total_ops_2
        assert abs(hit_rate_2 - expected_rate_2) < 0.01, f"Hit rate {hit_rate_2} != expected {expected_rate_2}"


@pytest.mark.asyncio  
async def test_budget_timeout_simulation():
    """Test budget skip counter (simulated since actual timeout is hard to trigger)"""
    cfg = RerankConfig(latency_budget_ms=1)  # Very tight budget to potentially trigger timeout
    reranker = CrossEncoderReranker(cfg, tenant_id="budget_test")
    
    # Use many documents to increase processing time
    docs = [
        {"content": f"document {i} with longer content to process", "score": 0.9 - i*0.01, "file_path": f"doc{i}.py"}
        for i in range(20)
    ]
    
    # Perform multiple operations
    for i in range(5):
        await reranker.rerank(f"budget test query {i}", docs, top_k=10)
    
    stats = reranker.get_stats()
    
    # Should have processed requests
    assert stats['total_requests'] == 5
    
    # Budget skips + heuristic used + cross encoder used should sum to total requests
    total_method_usage = (
        stats['budget_skipped'] + 
        stats['heuristic_used'] + 
        stats['cross_encoder_used']
    )
    
    # Note: might be higher than total_requests if heuristic fallback is called after budget skip
    assert total_method_usage >= stats['total_requests']
    
    # With very tight budget, should have at least some non-cross-encoder usage
    assert (stats['budget_skipped'] + stats['heuristic_used']) > 0


@pytest.mark.asyncio
async def test_tenant_isolation_in_metrics():
    """Test that different tenants have separate metrics"""
    cfg = RerankConfig(latency_budget_ms=100)
    
    reranker_a = CrossEncoderReranker(cfg, tenant_id="tenant_a")
    reranker_b = CrossEncoderReranker(cfg, tenant_id="tenant_b")
    
    docs = [{"content": "shared doc", "score": 0.8, "file_path": "shared.py"}]
    
    # Tenant A performs 3 operations
    for i in range(3):
        await reranker_a.rerank(f"query {i}", docs, top_k=1)
    
    # Tenant B performs 2 operations  
    for i in range(2):
        await reranker_b.rerank(f"query {i}", docs, top_k=1)
    
    stats_a = reranker_a.get_stats()
    stats_b = reranker_b.get_stats()
    
    # Each tenant should track their own metrics
    assert stats_a['tenant_id'] == 'tenant_a'
    assert stats_b['tenant_id'] == 'tenant_b'
    assert stats_a['total_requests'] == 3
    assert stats_b['total_requests'] == 2
    
    # Both should have latency tracking
    assert stats_a['latency_window_size'] == 3
    assert stats_b['latency_window_size'] == 2


def test_metrics_structure():
    """Test that get_stats returns expected structure"""
    cfg = RerankConfig()
    reranker = CrossEncoderReranker(cfg, tenant_id="structure_test")
    
    stats = reranker.get_stats()
    
    # Required basic fields
    required_fields = [
        'tenant_id', 'model_available', 'total_requests',
        'budget_skipped', 'heuristic_used', 'cross_encoder_used',
        'cache_size', 'cache_hits', 'cache_misses', 'cache_hit_rate',
        'latency_p50_ms', 'latency_p95_ms', 'latency_window_size'
    ]
    
    for field in required_fields:
        assert field in stats, f"Missing required field: {field}"
    
    # Numeric fields should be numeric
    numeric_fields = [
        'total_requests', 'budget_skipped', 'heuristic_used', 'cross_encoder_used',
        'cache_size', 'cache_hits', 'cache_misses', 'cache_hit_rate', 'latency_window_size'
    ]
    
    for field in numeric_fields:
        assert isinstance(stats[field], (int, float)), f"Field {field} should be numeric, got {type(stats[field])}"
    
    # Boolean fields
    assert isinstance(stats['model_available'], bool), "model_available should be boolean"
    
    # String fields  
    assert isinstance(stats['tenant_id'], str), "tenant_id should be string"


if __name__ == "__main__":
    # Run tests directly
    import sys
    
    async def run_all_tests():
        print("🔢 Running Metrics Tests")
        print("=" * 40)
        
        tests = [
            ("Basic metrics collection", test_basic_metrics_collection),
            ("Latency percentiles tracking", test_latency_percentiles_tracking),
            ("Cache hit rate calculation", test_cache_hit_rate_calculation),
            ("Budget timeout simulation", test_budget_timeout_simulation),
            ("Tenant isolation in metrics", test_tenant_isolation_in_metrics),
        ]
        
        passed = 0
        for test_name, test_func in tests:
            try:
                print(f"🧪 {test_name}...")
                await test_func()
                print(f"   ✅ PASSED")
                passed += 1
            except Exception as e:
                print(f"   ❌ FAILED: {e}")
        
        # Run sync test
        try:
            print(f"🧪 Metrics structure...")
            test_metrics_structure()
            print(f"   ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
        
        print(f"\n📊 Results: {passed}/{len(tests) + 1} tests passed")
        
        if passed == len(tests) + 1:
            print("🎉 All metrics tests passed!")
            return 0
        else:
            print("⚠️  Some metrics tests failed")
            return 1
    
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)