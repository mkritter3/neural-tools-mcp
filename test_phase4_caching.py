#!/usr/bin/env python3
"""
Test Phase 4 Caching Optimization - ADR-0075
Test intelligent query result caching performance improvements
"""

import asyncio
import sys
import time
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from neural_mcp.neural_server_stdio import dependency_analysis_impl

async def test_caching_performance():
    """Test Phase 4 query result caching optimization"""
    print("🧠 Phase 4: Testing Query Result Caching Optimization")
    print("="*65)

    test_queries = [
        ("neural-tools/src/servers/services/service_container.py", "imports", 2),
        ("neural-tools/src/neural_mcp/neural_server_stdio.py", "imports", 2),
        ("neural-tools/src/servers/services/service_container.py", "dependents", 2),
        ("neural-tools/src/servers/services/service_container.py", "all", 2),
    ]

    print(f"📊 Testing {len(test_queries)} different queries with cache behavior")
    print()

    # Round 1: Cold cache - measure initial performance
    print("🔥 Round 1: Cold Cache (First execution)")
    print("-" * 45)
    round1_times = []

    for i, (file_path, analysis_type, depth) in enumerate(test_queries):
        print(f"Query {i+1}: {file_path.split('/')[-1]} -> {analysis_type} (depth {depth})")

        start_time = time.time()
        try:
            result = await dependency_analysis_impl(file_path, analysis_type, depth)
            end_time = time.time()

            duration_ms = (end_time - start_time) * 1000
            round1_times.append(duration_ms)

            # Parse result
            import json
            data = json.loads(result[0].text)
            if data.get('status') == 'success':
                summary = data.get('summary', {})
                total_items = summary.get('total_imports', 0) + summary.get('total_dependents', 0) + summary.get('total_calls', 0)
                print(f"  ⏱️  {duration_ms:.1f}ms ({total_items} items)")
            else:
                print(f"  ⏱️  {duration_ms:.1f}ms (error)")

        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            round1_times.append(duration_ms)
            print(f"  ⏱️  {duration_ms:.1f}ms (exception)")

    print()

    # Round 2: Warm cache - measure cached performance
    print("🚀 Round 2: Warm Cache (Repeat same queries)")
    print("-" * 50)
    round2_times = []

    for i, (file_path, analysis_type, depth) in enumerate(test_queries):
        print(f"Query {i+1}: {file_path.split('/')[-1]} -> {analysis_type} (depth {depth})")

        start_time = time.time()
        try:
            result = await dependency_analysis_impl(file_path, analysis_type, depth)
            end_time = time.time()

            duration_ms = (end_time - start_time) * 1000
            round2_times.append(duration_ms)

            # Parse result
            import json
            data = json.loads(result[0].text)
            if data.get('status') == 'success':
                summary = data.get('summary', {})
                total_items = summary.get('total_imports', 0) + summary.get('total_dependents', 0) + summary.get('total_calls', 0)
                print(f"  ⚡ {duration_ms:.1f}ms ({total_items} items)")
            else:
                print(f"  ⚡ {duration_ms:.1f}ms (error)")

        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            round2_times.append(duration_ms)
            print(f"  ⚡ {duration_ms:.1f}ms (exception)")

    # Cache Performance Analysis
    print()
    print("🧠 Cache Performance Analysis:")
    print("="*40)

    if round1_times and round2_times:
        avg_cold = sum(round1_times) / len(round1_times)
        avg_warm = sum(round2_times) / len(round2_times)

        print(f"Cold Cache Average: {avg_cold:.1f}ms")
        print(f"Warm Cache Average: {avg_warm:.1f}ms")

        if avg_cold > 0:
            cache_improvement = ((avg_cold - avg_warm) / avg_cold) * 100
            print(f"Cache Improvement: {cache_improvement:.1f}% faster")

        print()
        print("Per-Query Cache Improvements:")
        for i, ((file, analysis, depth), cold_time, warm_time) in enumerate(zip(test_queries, round1_times, round2_times)):
            if cold_time > 0:
                improvement = ((cold_time - warm_time) / cold_time) * 100
                print(f"  Query {i+1}: {improvement:.1f}% faster ({cold_time:.1f}ms -> {warm_time:.1f}ms)")

        # Check cache effectiveness
        ultra_fast_queries = sum(1 for t in round2_times if t < 5.0)  # Sub-5ms queries
        if ultra_fast_queries > 0:
            print(f"✅ Cache SUCCESS: {ultra_fast_queries}/{len(round2_times)} queries <5ms (cache hits)")
        else:
            print("⚠️  Cache PARTIAL: No ultra-fast queries detected")

    print()
    print("✅ Query result caching performance test completed!")

if __name__ == "__main__":
    asyncio.run(test_caching_performance())