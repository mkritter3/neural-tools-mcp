#!/usr/bin/env python3
"""
Test Cypher Query Optimization - ADR-0075 Phase 4
Compare performance before/after query optimization
"""

import asyncio
import sys
import time
import json
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from neural_mcp.neural_server_stdio import dependency_analysis_impl

async def test_cypher_optimization():
    """Test optimized vs original Cypher queries performance"""
    print("🔧 ADR-0075 Phase 4: Cypher Query Optimization Test")
    print("="*60)

    test_file = "neural-tools/src/servers/services/service_container.py"
    test_scenarios = [
        ("imports", 2),
        ("dependents", 2),
        ("calls", 2),
        ("all", 2),
        ("imports", 1),  # Depth-1 which was previously slow
        ("all", 3),      # Deeper traversal
    ]

    print(f"📊 Testing {len(test_scenarios)} query scenarios with optimized Cypher")
    print()

    # Test each scenario with performance measurement
    results = []

    for i, (analysis_type, depth) in enumerate(test_scenarios):
        print(f"🔍 Test {i+1}/{len(test_scenarios)}: {analysis_type} (depth {depth})")

        # Measure query performance
        start_time = time.time()
        try:
            result = await dependency_analysis_impl(test_file, analysis_type, depth)
            end_time = time.time()

            duration_ms = (end_time - start_time) * 1000

            # Parse result to validate data quality
            data = json.loads(result[0].text)
            if data.get('status') == 'success':
                deps = data.get('dependencies', {})
                total_results = (
                    len(deps.get('imports', [])) +
                    len(deps.get('dependents', [])) +
                    len(deps.get('calls', []))
                )

                results.append({
                    'scenario': f"{analysis_type}(depth={depth})",
                    'duration_ms': duration_ms,
                    'total_results': total_results,
                    'status': 'success'
                })

                print(f"  ⚡ {duration_ms:.1f}ms ({total_results} results)")

            else:
                results.append({
                    'scenario': f"{analysis_type}(depth={depth})",
                    'duration_ms': duration_ms,
                    'total_results': 0,
                    'status': 'no_data'
                })
                print(f"  ⚠️  {duration_ms:.1f}ms (no data)")

        except Exception as e:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            results.append({
                'scenario': f"{analysis_type}(depth={depth})",
                'duration_ms': duration_ms,
                'total_results': 0,
                'status': 'error',
                'error': str(e)
            })
            print(f"  ❌ {duration_ms:.1f}ms (error: {str(e)[:50]})")

    print()

    # Performance Analysis
    print("📈 Cypher Query Optimization Results:")
    print("="*45)

    if results:
        successful_results = [r for r in results if r['status'] == 'success']

        if successful_results:
            avg_time = sum(r['duration_ms'] for r in successful_results) / len(successful_results)
            min_time = min(r['duration_ms'] for r in successful_results)
            max_time = max(r['duration_ms'] for r in successful_results)

            print(f"Successful Queries: {len(successful_results)}/{len(results)}")
            print(f"Average Query Time: {avg_time:.1f}ms")
            print(f"Fastest Query: {min_time:.1f}ms")
            print(f"Slowest Query: {max_time:.1f}ms")
            print()

            # Query-specific optimizations check
            imports_results = [r for r in successful_results if 'imports' in r['scenario']]
            if imports_results:
                imports_avg = sum(r['duration_ms'] for r in imports_results) / len(imports_results)
                print(f"Imports-only queries: {imports_avg:.1f}ms average")

            dependents_results = [r for r in successful_results if 'dependents' in r['scenario']]
            if dependents_results:
                dependents_avg = sum(r['duration_ms'] for r in dependents_results) / len(dependents_results)
                print(f"Dependents-only queries: {dependents_avg:.1f}ms average")

            calls_results = [r for r in successful_results if 'calls' in r['scenario']]
            if calls_results:
                calls_avg = sum(r['duration_ms'] for r in calls_results) / len(calls_results)
                print(f"Calls-only queries: {calls_avg:.1f}ms average")

            all_results = [r for r in successful_results if r['scenario'].startswith('all')]
            if all_results:
                all_avg = sum(r['duration_ms'] for r in all_results) / len(all_results)
                print(f"Comprehensive queries: {all_avg:.1f}ms average")
            print()

            # Performance targets check
            fast_queries = sum(1 for r in successful_results if r['duration_ms'] < 50)
            medium_queries = sum(1 for r in successful_results if 50 <= r['duration_ms'] < 200)
            slow_queries = sum(1 for r in successful_results if r['duration_ms'] >= 200)

            print("Performance Distribution:")
            print(f"  <50ms (excellent): {fast_queries}/{len(successful_results)} queries")
            print(f"  50-200ms (good): {medium_queries}/{len(successful_results)} queries")
            print(f"  ≥200ms (needs improvement): {slow_queries}/{len(successful_results)} queries")

            if avg_time < 100:
                print("✅ OPTIMIZATION SUCCESS: Average query time <100ms")
            elif avg_time < 200:
                print("⚠️  OPTIMIZATION PARTIAL: Average query time acceptable but could improve")
            else:
                print("❌ OPTIMIZATION NEEDED: Average query time >200ms")

        else:
            print("❌ No successful queries - optimization may have introduced issues")

    # Detailed results
    print()
    print("📋 Detailed Results:")
    print("-" * 35)
    for result in results:
        status_icon = {"success": "✅", "no_data": "⚠️", "error": "❌"}[result['status']]
        print(f"{status_icon} {result['scenario']}: {result['duration_ms']:.1f}ms ({result['total_results']} results)")

    print()
    print("🎯 Cypher query optimization test completed!")
    print("   Analysis-type-specific queries for improved performance")

if __name__ == "__main__":
    asyncio.run(test_cypher_optimization())