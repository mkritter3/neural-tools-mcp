#!/usr/bin/env python3
"""
Test MCP tools directly to find the disconnect
"""

import asyncio
import sys
import json
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

async def test_mcp_tools():
    """Test MCP tools to ensure they return real results"""

    print("=" * 70)
    print("🔍 TESTING MCP TOOLS FOR REAL RESULTS")
    print("=" * 70)

    # Test different queries
    test_queries = [
        "migrations in Neo4j",
        "vector search implementation",
        "how indexing works"
    ]

    # Import the tools
    from neural_mcp.tools.fast_search import execute as fast_search
    from neural_mcp.tools.elite_search import execute as elite_search

    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        print("-" * 60)

        # Test fast_search
        print("\n1️⃣ Fast Search:")
        try:
            fast_result = await fast_search({"query": query, "limit": 2})
            fast_data = json.loads(fast_result[0].text)

            if fast_data.get("status") == "success":
                results = fast_data.get("results", [])
                print(f"   ✅ Found {len(results)} results")
                for r in results:
                    print(f"      • Score {r['score']:.3f}: {r['file']['path']}")
                    print(f"        Content: {r['content'][:80]}...")
            else:
                print(f"   ❌ {fast_data.get('status')}: {fast_data.get('message', 'unknown')}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test elite_search
        print("\n2️⃣ Elite Search:")
        try:
            elite_result = await elite_search({
                "query": query,
                "limit": 2,
                "max_depth": 1,
                "vector_weight": 0.7,
                "include_explanation": False
            })
            elite_data = json.loads(elite_result[0].text)

            if elite_data.get("status") == "success":
                results = elite_data.get("results", [])
                print(f"   ✅ Found {len(results)} results")
                for r in results:
                    print(f"      • Score {r['scores']['final']:.3f}: {r['file']['path']}")
                    print(f"        Content: {r['content'][:80]}...")
                    if r.get('graph_context'):
                        ctx = r['graph_context']
                        print(f"        Graph: {ctx.get('total_connections', 0)} connections")
            else:
                print(f"   ❌ {elite_data.get('status')}: {elite_data.get('message', 'unknown')}")

                # Debug: Check what the underlying service returns
                print("\n   🔍 Debugging underlying service...")
                from servers.services.service_container import ServiceContainer
                from servers.services.project_context_manager import ProjectContextManager

                context = ProjectContextManager()
                await context.set_project("/Users/mkr/local-coding/claude-l9-template")
                container = ServiceContainer(context, "claude-l9-template")
                await container.initialize_all_services()

                # Get embedding
                embedding = await container.nomic.get_embedding(query, task_type="search_query")

                # Call hybrid_search directly
                neo4j_results = await container.neo4j.hybrid_search_with_fanout(
                    query_text=query,
                    query_embedding=embedding,
                    max_depth=1,
                    limit=2,
                    vector_weight=0.7
                )

                print(f"      Neo4j returned {len(neo4j_results)} results directly")
                if neo4j_results:
                    chunk = neo4j_results[0].get('chunk', {})
                    print(f"      First result: {chunk.get('file_path', 'unknown')}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("🎯 MCP TOOLS TEST COMPLETE")
    print("=" * 70)

    print("\n📊 SUMMARY:")
    print("   Testing complete. Check results above for:")
    print("   • Real file paths (not 'unknown' or None)")
    print("   • Actual content (not empty)")
    print("   • Reasonable scores")
    print("   • Graph context for elite search")

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())