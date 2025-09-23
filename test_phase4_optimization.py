#!/usr/bin/env python3
"""
Test Phase 4 Performance Optimizations - ADR-0075
Test shared service connection pooling performance improvements
"""

import asyncio
import sys
import time
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from neural_mcp.neural_server_stdio import dependency_analysis_impl

async def test_connection_pooling_performance():
    """Test Phase 4 connection pooling optimization"""
    print("🚀 Phase 4: Testing Connection Pooling Optimization")
    print("="*60)

    test_file = "neural-tools/src/servers/services/service_container.py"
    analysis_type = "imports"
    depth = 2  # Use depth 2 which was fastest in benchmark

    print(f"📊 Testing repeated queries to {test_file}")
    print(f"   Analysis: {analysis_type}, Depth: {depth}")
    print()

    # Test repeated queries to measure connection reuse performance
    iterations = 5
    durations = []

    for i in range(iterations):
        print(f"🔍 Query {i+1}/{iterations}...", end=" ")

        start_time = time.time()
        try:
            result = await dependency_analysis_impl(test_file, analysis_type, depth)
            end_time = time.time()

            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)

            # Parse result
            import json
            data = json.loads(result[0].text)
            if data.get('status') == 'success':
                summary = data.get('summary', {})
                total_imports = summary.get('total_imports', 0)
                print(f"{duration_ms:.1f}ms ({total_imports} imports)")
            else:
                print(f"{duration_ms:.1f}ms (error)")

        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
            print(f"{duration_ms:.1f}ms (exception: {str(e)[:50]})")

    # Performance analysis
    print()
    print("📈 Connection Pooling Performance Analysis:")
    print("="*50)

    if durations:
        first_query = durations[0]
        subsequent_queries = durations[1:] if len(durations) > 1 else []

        print(f"First Query (cold start): {first_query:.1f}ms")

        if subsequent_queries:
            avg_subsequent = sum(subsequent_queries) / len(subsequent_queries)
            min_subsequent = min(subsequent_queries)
            max_subsequent = max(subsequent_queries)

            print(f"Subsequent Queries:")
            print(f"  Average: {avg_subsequent:.1f}ms")
            print(f"  Min: {min_subsequent:.1f}ms")
            print(f"  Max: {max_subsequent:.1f}ms")

            # Calculate improvement
            if first_query > 0:
                improvement = ((first_query - avg_subsequent) / first_query) * 100
                print(f"  Improvement: {improvement:.1f}% faster than cold start")

            # Check if we eliminated initialization overhead
            if avg_subsequent < 100:  # Target: <100ms for warm queries
                print("✅ Connection pooling SUCCESS: Warm queries <100ms")
            else:
                print("⚠️  Connection pooling PARTIAL: Warm queries still >100ms")

        print(f"\nAll Query Times: {[f'{d:.1f}ms' for d in durations]}")

    print()
    print("✅ Connection pooling performance test completed!")

if __name__ == "__main__":
    asyncio.run(test_connection_pooling_performance())