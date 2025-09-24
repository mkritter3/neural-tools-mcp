#!/usr/bin/env python3
"""
Test ADR-0090 Phase 4: Elite hybrid search with graph fan-out
"""
import asyncio
import sys
import os
from pathlib import Path

# Add path for imports
sys.path.insert(0, 'neural-tools/src')
sys.path.insert(0, 'neural-tools/src/servers/services')

async def test_hybrid_search():
    from service_container import ServiceContainer
    from servers.services.project_context_manager import ProjectContextManager

    print("🚀 Testing Elite GraphRAG Hybrid Search with Fan-out")
    print("=" * 50)

    # Initialize services
    project_path = os.getcwd()
    project_name = "claude-l9-template"

    context = ProjectContextManager()
    container = ServiceContainer(context, project_name)

    print("📦 Initializing services...")
    await container.initialize_all_services()

    if not container.neo4j:
        print("❌ Neo4j not available")
        return

    if not container.nomic:
        print("❌ Nomic not available")
        return

    # Test queries
    test_queries = [
        "authentication and user management",
        "vector search and embeddings",
        "Neo4j graph database operations"
    ]

    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 40)

        # Get embedding for the query
        try:
            embedding_result = await container.nomic.get_embeddings([query])
            if embedding_result and len(embedding_result) > 0:
                query_embedding = embedding_result[0]
                print(f"✅ Got embedding (dim={len(query_embedding)})")
            else:
                print("❌ Failed to get embedding")
                continue
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            continue

        # Test 1: Basic hybrid search
        print("\n1. Basic hybrid search:")
        try:
            basic_results = await container.neo4j.hybrid_search(
                query_text=query,
                query_embedding=query_embedding,
                limit=5,
                vector_weight=0.7
            )
            print(f"   Found {len(basic_results)} results")
            for i, result in enumerate(basic_results[:3]):
                score = result.get('combined_score', 0)
                node_type = result.get('node_type', 'unknown')
                print(f"   [{i+1}] Score: {score:.3f}, Type: {node_type}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        # Test 2: Elite hybrid search with fan-out
        print("\n2. Elite hybrid search with graph fan-out:")
        try:
            elite_results = await container.neo4j.hybrid_search_with_fanout(
                query_text=query,
                query_embedding=query_embedding,
                max_depth=2,
                limit=5,
                vector_weight=0.7
            )
            print(f"   Found {len(elite_results)} results with graph context")

            for i, result in enumerate(elite_results[:3]):
                final_score = result.get('final_score', 0)
                vector_score = result.get('vector_score', 0)
                context = result.get('graph_context', {})

                print(f"\n   [{i+1}] Final score: {final_score:.3f}")
                print(f"       Vector score: {vector_score:.3f}")
                print(f"       Graph context:")
                print(f"         - Import relevance: {context.get('import_relevance', 0)}")
                print(f"         - Call depth: {context.get('call_depth', 0)}")
                print(f"         - Variable usage: {context.get('variable_usage', 0)}")
                print(f"         - Class usage: {context.get('class_usage', 0)}")

                # Show some context details
                imports = context.get('imports', [])
                if imports:
                    print(f"         - Related imports: {len(imports)} files")

                call_chain = context.get('call_chain', [])
                if call_chain:
                    print(f"         - Call chain: {len(call_chain)} functions")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 50)
    print("✅ Elite GraphRAG hybrid search test complete!")
    print("\nSummary:")
    print("- Basic hybrid search: Combines vector + text scoring")
    print("- Elite fan-out search: Adds graph context (imports, calls, uses)")
    print("- Graph context boosts relevance of interconnected code")

if __name__ == '__main__':
    asyncio.run(test_hybrid_search())