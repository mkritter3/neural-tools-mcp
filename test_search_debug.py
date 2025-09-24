#!/usr/bin/env python3
"""
Debug script to test vector search and find the disconnect
"""

import asyncio
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

async def test_vector_search():
    """Test vector search with debug output"""

    from servers.services.service_container import ServiceContainer
    from servers.services.project_context_manager import ProjectContextManager

    # Set up for claude-l9-template
    project_path = "/Users/mkr/local-coding/claude-l9-template"
    project_name = "claude-l9-template"

    print("=" * 70)
    print("🔍 DEBUGGING VECTOR SEARCH DISCONNECT")
    print("=" * 70)

    # Create context and container
    context = ProjectContextManager()
    await context.set_project(project_path)

    container = ServiceContainer(context, project_name)
    await container.initialize_all_services()

    print("\n1️⃣ Testing direct Neo4j query for chunks...")
    print("-" * 60)

    # Check if we have chunks with embeddings
    check_query = """
    MATCH (c:Chunk)
    WHERE c.project = $project AND c.embedding IS NOT NULL
    RETURN count(c) as total,
           avg(size(c.embedding)) as avg_embedding_size,
           collect(DISTINCT c.file_path)[0..3] as sample_files
    """

    result = await container.neo4j.execute_cypher(check_query, {"project": project_name})

    if result['status'] == 'success' and result['result']:
        data = result['result'][0]
        print(f"✅ Found {data['total']} chunks with embeddings")
        print(f"   Average embedding size: {data['avg_embedding_size']}")
        print(f"   Sample files: {data['sample_files']}")
    else:
        print("❌ No chunks with embeddings found!")
        return

    print("\n2️⃣ Testing embedding generation...")
    print("-" * 60)

    # Generate test embedding
    test_query = "how do migrations work in the indexer"
    embedding = await container.nomic.get_embedding(test_query, task_type="search_query")

    if embedding and len(embedding) == 768:
        print(f"✅ Generated embedding for query: dimension={len(embedding)}")
        print(f"   Sample values: {embedding[:3]}")
    else:
        print(f"❌ Failed to generate embedding: {embedding}")
        return

    print("\n3️⃣ Testing raw vector search...")
    print("-" * 60)

    # Test direct vector search
    vector_search_query = """
    CALL db.index.vector.queryNodes('chunk_embeddings_index', 10, $query_vector)
    YIELD node, score
    WHERE node.project = $project
    RETURN node.chunk_id as chunk_id,
           node.file_path as file_path,
           score,
           substring(node.content, 0, 100) as preview
    ORDER BY score DESC
    LIMIT 5
    """

    try:
        result = await container.neo4j.execute_cypher(
            vector_search_query,
            {"query_vector": embedding, "project": project_name}
        )

        if result['status'] == 'success' and result['result']:
            print(f"✅ Vector search returned {len(result['result'])} results:")
            for r in result['result']:
                print(f"   • Score {r['score']:.3f}: {r['file_path']}")
                print(f"     Preview: {r['preview'][:50]}...")
        else:
            print(f"❌ Vector search failed: {result}")
    except Exception as e:
        print(f"❌ Vector search error: {e}")

    print("\n4️⃣ Testing vector_similarity_search method...")
    print("-" * 60)

    # Test the service method
    results = await container.neo4j.vector_similarity_search(
        query_embedding=embedding,
        node_type="Chunk",
        limit=5,
        min_score=0.5
    )

    if results:
        print(f"✅ vector_similarity_search returned {len(results)} results")
        for r in results[:3]:
            print(f"   • Score {r.get('score', 0):.3f}: {r.get('node', {}).get('file_path', 'unknown')}")
    else:
        print("❌ vector_similarity_search returned no results")

    print("\n5️⃣ Testing hybrid_search_with_fanout...")
    print("-" * 60)

    # Test the full hybrid search
    results = await container.neo4j.hybrid_search_with_fanout(
        query_text=test_query,
        query_embedding=embedding,
        max_depth=1,
        limit=5,
        vector_weight=0.7
    )

    if results:
        print(f"✅ hybrid_search_with_fanout returned {len(results)} results")
        for r in results[:3]:
            print(f"   • {r}")
    else:
        print("❌ hybrid_search_with_fanout returned no results")

    print("\n" + "=" * 70)
    print("🎯 DIAGNOSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_vector_search())