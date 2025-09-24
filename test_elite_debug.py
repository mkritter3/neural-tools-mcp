#!/usr/bin/env python3
"""
Debug elite search to understand the result format
"""

import asyncio
import sys
import json
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

async def test_elite_search():
    """Test elite search with debug output"""

    from servers.services.service_container import ServiceContainer
    from servers.services.project_context_manager import ProjectContextManager

    # Set up for claude-l9-template
    project_path = "/Users/mkr/local-coding/claude-l9-template"
    project_name = "claude-l9-template"

    print("=" * 70)
    print("🔍 DEBUGGING ELITE SEARCH")
    print("=" * 70)

    # Create context and container
    context = ProjectContextManager()
    await context.set_project(project_path)

    container = ServiceContainer(context, project_name)
    await container.initialize_all_services()

    print("\n1️⃣ Generating embedding...")
    print("-" * 60)

    test_query = "how do migrations work in the indexer"
    embedding = await container.nomic.get_embedding(test_query, task_type="search_query")

    if embedding and len(embedding) == 768:
        print(f"✅ Generated embedding: dimension={len(embedding)}")
    else:
        print(f"❌ Failed to generate embedding")
        return

    print("\n2️⃣ Testing hybrid_search_with_fanout...")
    print("-" * 60)

    # Call hybrid_search_with_fanout
    results = await container.neo4j.hybrid_search_with_fanout(
        query_text=test_query,
        query_embedding=embedding,
        max_depth=2,
        limit=5,
        vector_weight=0.7
    )

    print(f"✅ Got {len(results)} results")

    if results:
        print("\n3️⃣ Examining result structure...")
        print("-" * 60)

        # Look at first result structure
        first = results[0]
        print(f"Result type: {type(first)}")
        print(f"Result keys: {list(first.keys())}")

        print("\n📊 First result details:")
        print(json.dumps(first, indent=2, default=str))

        # Check chunk structure
        chunk = first.get("chunk", {})
        print(f"\nChunk keys: {list(chunk.keys())}")

        # Check if file_path is in chunk
        if chunk.get("file_path"):
            print(f"✅ Chunk has file_path: {chunk['file_path']}")
        else:
            print("❌ Chunk missing file_path")

        # Check graph context
        graph_ctx = first.get("graph_context", {})
        print(f"\nGraph context type: {type(graph_ctx)}")
        if isinstance(graph_ctx, list) and graph_ctx:
            print(f"Graph context items: {len(graph_ctx)}")
            print(f"First context item: {graph_ctx[0] if graph_ctx else 'empty'}")
        elif isinstance(graph_ctx, dict):
            print(f"Graph context keys: {list(graph_ctx.keys())}")

    else:
        print("❌ No results returned")

    print("\n" + "=" * 70)
    print("🎯 DEBUG COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_elite_search())