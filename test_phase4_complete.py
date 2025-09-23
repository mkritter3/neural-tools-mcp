#!/usr/bin/env python3
"""
Test Complete Phase 4 Performance Optimization Suite - ADR-0075
Test all Phase 4 optimizations: connection pooling, caching, and monitoring
"""

import asyncio
import sys
import time
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from neural_mcp.neural_server_stdio import dependency_analysis_impl, handle_call_tool

async def test_complete_phase4_optimization():
    """Test complete Phase 4 optimization suite"""
    print("🚀 Phase 4: Complete Optimization Suite Test")
    print("="*60)

    # Test 1: Connection pooling performance
    print("🔧 Test 1: Connection Pooling Performance")
    print("-" * 45)

    test_file = "neural-tools/src/servers/services/service_container.py"

    # Multiple queries to test connection reuse
    for i in range(3):
        print(f"  Query {i+1}/3...", end=" ")
        start_time = time.time()

        result = await dependency_analysis_impl(test_file, "imports", 2)
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        print(f"{duration_ms:.1f}ms")

    print()

    # Test 2: Cache performance
    print("🧠 Test 2: Cache Performance (Repeat queries)")
    print("-" * 50)

    # Same queries should hit cache
    for i in range(3):
        print(f"  Cached Query {i+1}/3...", end=" ")
        start_time = time.time()

        result = await dependency_analysis_impl(test_file, "imports", 2)
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        print(f"{duration_ms:.1f}ms")

    print()

    # Test 3: Performance metrics
    print("📊 Test 3: Performance Metrics")
    print("-" * 35)

    metrics_result = await handle_call_tool("performance_metrics", {})
    import json
    metrics_data = json.loads(metrics_result[0].text)

    if metrics_data.get("status") == "success":
        metrics = metrics_data["metrics"]
        optimizations = metrics_data["optimizations"]

        print(f"  Total Queries: {metrics['total_queries']}")
        print(f"  Cache Hits: {metrics['cache_hits']}")
        print(f"  Cache Misses: {metrics['cache_misses']}")
        print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
        print(f"  Avg Query Time: {metrics['avg_query_time_ms']:.1f}ms")
        print(f"  Cached Queries: {metrics['cached_queries']}")
        print(f"  Service Instances: {metrics['service_creation_count']}")
        print()
        print(f"  Optimizations Status:")
        print(f"    Connection Pooling: {optimizations['connection_pooling']}")
        print(f"    Query Caching: {optimizations['query_caching']}")
        print(f"    Cache TTL: {optimizations['cache_ttl_seconds']}s")
        print(f"    Max Cache Size: {optimizations['max_cache_size']}")

    else:
        print("  ❌ Failed to get metrics")

    print()

    # Test 4: Different query types
    print("🎯 Test 4: Mixed Query Performance")
    print("-" * 40)

    test_queries = [
        ("dependents", 2),
        ("calls", 2),
        ("all", 3),
        ("imports", 1),  # Test the previously slow depth-1
    ]

    for analysis_type, depth in test_queries:
        print(f"  {analysis_type} (depth {depth})...", end=" ")
        start_time = time.time()

        result = await dependency_analysis_impl(test_file, analysis_type, depth)
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000

        # Parse result
        data = json.loads(result[0].text)
        if data.get('status') == 'success':
            summary = data.get('summary', {})
            total_items = summary.get('total_imports', 0) + summary.get('total_dependents', 0) + summary.get('total_calls', 0)
            print(f"{duration_ms:.1f}ms ({total_items} items)")
        else:
            print(f"{duration_ms:.1f}ms (error)")

    print()

    # Final metrics check
    print("📈 Final Performance Summary")
    print("-" * 35)

    final_metrics_result = await handle_call_tool("performance_metrics", {})
    final_metrics_data = json.loads(final_metrics_result[0].text)

    if final_metrics_data.get("status") == "success":
        final_metrics = final_metrics_data["metrics"]

        print(f"  Total Test Queries: {final_metrics['total_queries']}")
        print(f"  Final Cache Hit Rate: {final_metrics['cache_hit_rate']:.1f}%")
        print(f"  Final Avg Query Time: {final_metrics['avg_query_time_ms']:.1f}ms")

        # Performance assessment
        if final_metrics['cache_hit_rate'] > 50:
            print("  ✅ Cache Performance: EXCELLENT")
        elif final_metrics['cache_hit_rate'] > 20:
            print("  ⚠️  Cache Performance: GOOD")
        else:
            print("  ❌ Cache Performance: NEEDS IMPROVEMENT")

        if final_metrics['avg_query_time_ms'] < 50:
            print("  ✅ Query Performance: EXCELLENT")
        elif final_metrics['avg_query_time_ms'] < 200:
            print("  ⚠️  Query Performance: GOOD")
        else:
            print("  ❌ Query Performance: NEEDS IMPROVEMENT")

        if final_metrics['service_creation_count'] <= 1:
            print("  ✅ Connection Pooling: EXCELLENT")
        else:
            print("  ⚠️  Connection Pooling: MULTIPLE SERVICES CREATED")

    print()
    print("🎉 Phase 4 Complete Optimization Suite Test COMPLETED!")
    print("    ADR-0075 Phase 4: Connection Pooling + Caching + Monitoring")

if __name__ == "__main__":
    asyncio.run(test_complete_phase4_optimization())