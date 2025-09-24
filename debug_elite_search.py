#!/usr/bin/env python3
"""
Debug why elite_search returns empty results
"""

import asyncio
import sys
import json
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

async def debug_elite_search():
    """Debug elite search step by step"""

    from servers.services.service_container import ServiceContainer
    from servers.services.project_context_manager import ProjectContextManager

    # Set up for claude-l9-template
    project_path = "/Users/mkr/local-coding/claude-l9-template"
    project_name = "claude-l9-template"

    print("=" * 70)
    print("🔍 DEBUGGING ELITE SEARCH EMPTY RESULTS")
    print("=" * 70)

    # Create context and container
    context = ProjectContextManager()
    await context.set_project(project_path)

    container = ServiceContainer(context, project_name)
    await container.initialize_all_services()

    test_query = "how do migrations work"

    print(f"\n🔍 Query: '{test_query}'")
    print("-" * 60)

    # Generate embedding
    embedding = await container.nomic.get_embedding(test_query, task_type="search_query")
    print(f"✅ Generated embedding: dimension={len(embedding)}")

    # Test hybrid_search_with_fanout directly
    print("\n1️⃣ Testing neo4j.hybrid_search_with_fanout directly...")
    results = await container.neo4j.hybrid_search_with_fanout(
        query_text=test_query,
        query_embedding=embedding,
        max_depth=1,
        limit=5,
        vector_weight=0.7
    )

    print(f"   Results from neo4j service: {len(results)} items")
    if results:
        print(f"   First result type: {type(results[0])}")
        print(f"   First result keys: {list(results[0].keys())}")

        # Show first result
        first = results[0]
        chunk = first.get("chunk", {})
        print(f"\n   Chunk data:")
        print(f"   - chunk_id: {chunk.get('chunk_id', 'MISSING')}")
        print(f"   - file_path: {chunk.get('file_path', 'MISSING')}")
        print(f"   - content preview: {chunk.get('content', 'MISSING')[:100]}...")

    # Now test through elite_search tool
    print("\n2️⃣ Testing through elite_search tool...")

    from neural_mcp.tools.elite_search import execute as elite_execute

    elite_args = {
        "query": test_query,
        "limit": 5,
        "max_depth": 1,
        "vector_weight": 0.7,
        "include_explanation": False
    }

    # Call elite search
    print("   Calling elite_search.execute()...")
    elite_result = await elite_execute(elite_args)

    # Parse result
    elite_data = json.loads(elite_result[0].text)

    print(f"\n   Elite search response:")
    print(f"   - Status: {elite_data.get('status')}")
    print(f"   - Results count: {len(elite_data.get('results', []))}")

    if elite_data.get('status') == 'no_results':
        print(f"   - Message: {elite_data.get('message')}")
        print(f"   - Suggestion: {elite_data.get('suggestion')}")

    # Let's trace through the elite_search code
    print("\n3️⃣ Tracing elite_search execution...")

    # Import the format function directly
    from neural_mcp.tools.elite_search import _format_elite_results

    # Call the formatter directly with our results
    print("   Calling _format_elite_results directly...")
    formatted = _format_elite_results(results, test_query, False)

    print(f"\n   Formatted response:")
    print(f"   - Status: {formatted.get('status')}")
    print(f"   - Results count: {len(formatted.get('results', []))}")

    if formatted.get('status') == 'success' and formatted.get('results'):
        print(f"\n   First formatted result:")
        first_formatted = formatted['results'][0]
        print(json.dumps(first_formatted, indent=2))

    print("\n" + "=" * 70)
    print("🎯 DIAGNOSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(debug_elite_search())