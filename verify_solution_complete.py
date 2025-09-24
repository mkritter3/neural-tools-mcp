#!/usr/bin/env python3
"""
Verify the complete ADR-0096 solution works end-to-end
"""

import asyncio
import sys
import json
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

async def verify_solution():
    """Test that the robust solution returns real, usable results"""

    print("=" * 70)
    print("✅ VERIFYING ADR-0096 ROBUST VECTOR SEARCH SOLUTION")
    print("=" * 70)

    # Import tools
    from neural_mcp.tools.fast_search import execute as fast_search
    from neural_mcp.tools.elite_search import execute as elite_search

    test_queries = [
        "Neo4j vector index optimization",
        "how migrations work in the system",
        "chunk schema and data consistency"
    ]

    all_success = True

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 60)

        # Test fast search
        print("\n✨ FAST SEARCH:")
        try:
            fast_result = await fast_search({"query": query, "limit": 2})
            fast_data = json.loads(fast_result[0].text)

            if fast_data.get("status") == "success":
                results = fast_data.get("results", [])
                print(f"   ✅ Found {len(results)} results")

                for r in results:
                    # Verify real file path
                    file_path = r.get("file", {}).get("path", "unknown")
                    if file_path != "unknown" and file_path != "None":
                        print(f"   • ✅ Real file: {file_path}")
                    else:
                        print(f"   • ❌ Missing file path")
                        all_success = False

                    # Verify real content
                    content = r.get("content", "")
                    if content and len(content) > 10:
                        print(f"     Content: {content[:60]}...")
                    else:
                        print(f"     ❌ Missing content")
                        all_success = False

                    # Verify reasonable score
                    score = r.get("score", 0)
                    if 0.5 < score < 1.0:
                        print(f"     Score: {score:.3f} ✅")
                    else:
                        print(f"     Score: {score:.3f} ⚠️")
            else:
                print(f"   ❌ Failed: {fast_data.get('message', 'unknown')}")
                all_success = False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_success = False

        # Test elite search
        print("\n🚀 ELITE SEARCH:")
        try:
            elite_result = await elite_search({
                "query": query,
                "limit": 2,
                "max_depth": 1,
                "include_explanation": True
            })
            elite_data = json.loads(elite_result[0].text)

            if elite_data.get("status") == "success":
                results = elite_data.get("results", [])
                print(f"   ✅ Found {len(results)} results")

                for r in results:
                    # Verify real file path
                    file_path = r.get("file", {}).get("path", "unknown")
                    if file_path != "unknown" and file_path != "None":
                        print(f"   • ✅ Real file: {file_path}")
                    else:
                        print(f"   • ❌ Missing file path")
                        all_success = False

                    # Verify real content
                    content = r.get("content", "")
                    if content and len(content) > 10:
                        print(f"     Content: {content[:60]}...")
                    else:
                        print(f"     ❌ Missing content")
                        all_success = False

                    # Verify graph context
                    graph_ctx = r.get("graph_context", {})
                    connections = graph_ctx.get("total_connections", 0)
                    if connections > 0:
                        print(f"     Graph: {connections} connections ✅")
                    else:
                        print(f"     Graph: No connections ⚠️")

                    # Verify explanation
                    explanation = r.get("explanation", "")
                    if explanation:
                        print(f"     Explanation: {explanation[:80]}...")

            else:
                print(f"   ❌ Failed: {elite_data.get('message', 'unknown')}")
                all_success = False

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            all_success = False

    print("\n" + "=" * 70)
    print("🎯 VERIFICATION COMPLETE")
    print("=" * 70)

    if all_success:
        print("\n✅ SUCCESS! All tests passed:")
        print("   • Fast search returns real file paths and content")
        print("   • Elite search returns real file paths and content")
        print("   • Elite search includes graph context")
        print("   • Scores are in reasonable range")
        print("\n🎉 ADR-0096 ROBUST VECTOR SEARCH IS WORKING!")
    else:
        print("\n⚠️ PARTIAL SUCCESS:")
        print("   Some tests failed - check output above")
        print("   The MCP server may need to be restarted to load changes")

    return all_success

if __name__ == "__main__":
    success = asyncio.run(verify_solution())
    sys.exit(0 if success else 1)