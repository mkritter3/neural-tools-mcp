#!/usr/bin/env python3
"""
Test Graph-Aware MCP Search - ADR-0075 Phase 1 Validation
"""

import asyncio
import sys
import json
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from neural_mcp.neural_server_stdio import semantic_code_search_impl

async def test_graph_aware_search():
    """Test the new graph-aware search functionality"""
    print("🧪 Testing Graph-Aware Elite Search")
    print("="*50)

    # Test query about service container functionality
    query = "service container initialization"
    limit = 3

    print(f"🔍 Query: '{query}'")
    print(f"📊 Limit: {limit}")
    print()

    try:
        results = await semantic_code_search_impl(query, limit)

        if results and len(results) > 0:
            response_data = json.loads(results[0].text)

            print("✅ Search Results:")
            print(f"   Status: {response_data.get('status')}")
            print(f"   Architecture: {response_data.get('architecture')}")
            print(f"   Graph Context Enabled: {response_data.get('graph_context_enabled')}")
            print(f"   Total Found: {response_data.get('total_found')}")
            print()

            # Check individual results
            for i, result in enumerate(response_data.get('results', []), 1):
                print(f"📄 Result {i}:")
                print(f"   Score: {result.get('score', 0):.4f}")
                print(f"   File: {result.get('file_path', 'N/A')}")
                print(f"   Search Method: {result.get('search_method')}")

                # Check for graph context
                graph_context = result.get('graph_context', {})
                if graph_context:
                    print("   🕸️  Graph Context:")

                    related_functions = graph_context.get('related_functions', [])
                    if related_functions:
                        print(f"      Functions: {', '.join(related_functions[:3])}")

                    imported_modules = graph_context.get('imported_modules', [])
                    if imported_modules:
                        print(f"      Modules: {', '.join(imported_modules[:3])}")

                    related_files = graph_context.get('related_files', [])
                    if related_files:
                        print(f"      Files: {', '.join(related_files[:3])}")
                else:
                    print("   🔍 No graph context (chunk not in graph)")

                print()

            print("🎯 Graph-aware search test completed successfully!")

        else:
            print("❌ No results returned")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_graph_aware_search())