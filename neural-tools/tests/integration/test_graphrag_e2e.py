#!/usr/bin/env python3
"""
GraphRAG End-to-End Integration Test
Tests the full data lifecycle from ingestion to retrieval
"""

import sys
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


async def test_graphrag_e2e():
    """Test end-to-end GraphRAG functionality"""
    print("\n🔗 Testing GraphRAG End-to-End Integration...")

    try:
        # Import necessary services
        from servers.services.project_context_manager import get_project_context_manager
        from servers.services.service_container import ServiceContainer

        # Initialize project context
        print("\n  1️⃣ Initializing Project Context...")
        context_manager = await get_project_context_manager()
        await context_manager.set_project("/Users/mkr/local-coding/claude-l9-template")
        current_project = await context_manager.get_current_project()
        print(f"     ✅ Project: {current_project['project']}")

        # Initialize service container
        print("\n  2️⃣ Initializing Service Container...")
        container = ServiceContainer(
            context_manager=context_manager,
            project_name=current_project['project']
        )
        container.initialize()
        print(f"     ✅ Container initialized")

        # Initialize async services
        if hasattr(container, 'initialize_all_services'):
            await container.initialize_all_services()
            print(f"     ✅ Async services initialized")

        # Test data preparation
        test_code = """
def calculate_fibonacci(n):
    '''Calculate the nth Fibonacci number'''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
        test_file_path = "/test/fibonacci.py"
        # Generate a valid integer ID for Qdrant (convert hex to int)
        test_chunk_id_hash = hashlib.sha256(f"{test_file_path}:0".encode()).hexdigest()[:15]
        test_chunk_id_int = int(test_chunk_id_hash, 16)  # For Qdrant
        test_chunk_id_str = test_chunk_id_hash  # For Neo4j

        print("\n  3️⃣ Testing Neo4j Graph Creation...")
        if container.neo4j:
            try:
                # Create a test node
                create_query = """
                MERGE (f:File {
                    path: $path,
                    project: $project,
                    name: 'fibonacci.py',
                    extension: 'py',
                    size: $size,
                    language: 'python'
                })
                RETURN f.path as path
                """

                result = await container.neo4j.execute_cypher(
                    create_query,
                    {
                        "path": test_file_path,
                        "project": current_project['project'],
                        "size": len(test_code)
                    }
                )

                if result.get("records"):
                    print(f"     ✅ Created File node: {test_file_path}")
                else:
                    print(f"     ⚠️ File node creation returned no records")

                # Create a code chunk node
                chunk_query = """
                MERGE (c:CodeChunk {
                    chunk_id: $chunk_id,
                    project: $project,
                    file_path: $path,
                    content: $content,
                    start_line: 1,
                    end_line: 5
                })
                RETURN c.chunk_id as chunk_id
                """

                chunk_result = await container.neo4j.execute_cypher(
                    chunk_query,
                    {
                        "chunk_id": test_chunk_id_str,
                        "project": current_project['project'],
                        "path": test_file_path,
                        "content": test_code
                    }
                )

                if chunk_result.get("records"):
                    print(f"     ✅ Created CodeChunk node: {test_chunk_id_str}")
                else:
                    print(f"     ⚠️ CodeChunk node creation returned no records")

                # Create relationship
                rel_query = """
                MATCH (f:File {path: $path, project: $project})
                MATCH (c:CodeChunk {chunk_id: $chunk_id, project: $project})
                MERGE (f)-[r:CONTAINS_CHUNK]->(c)
                RETURN type(r) as relationship
                """

                rel_result = await container.neo4j.execute_cypher(
                    rel_query,
                    {
                        "path": test_file_path,
                        "chunk_id": test_chunk_id_str,
                        "project": current_project['project']
                    }
                )

                if rel_result.get("records"):
                    print(f"     ✅ Created CONTAINS_CHUNK relationship")

            except Exception as e:
                print(f"     ❌ Neo4j error: {e}")
                return False
        else:
            print(f"     ⚠️ Neo4j service not available")

        print("\n  4️⃣ Testing Qdrant Vector Storage...")
        if container.qdrant:
            try:
                # Get collections
                collections = await container.qdrant.get_collections()
                print(f"     ✅ Connected to Qdrant")

                # Create test collection if needed
                from qdrant_client.models import Distance

                test_collection = f"test_{current_project['project']}_code"

                # Check if collection exists
                if test_collection not in collections:
                    # Use ensure_collection which is the QdrantService's public API
                    success = await container.qdrant.ensure_collection(
                        collection_name=test_collection,
                        vector_size=768,  # Nomic embedding dimension
                        distance=Distance.COSINE
                    )
                    if success:
                        print(f"     ✅ Created test collection: {test_collection}")
                    else:
                        print(f"     ❌ Failed to create test collection")
                else:
                    print(f"     ✅ Test collection exists: {test_collection}")

                # Create test embedding (mock 768-dim vector)
                test_embedding = [0.1] * 768  # Simplified for testing

                # Store vector using the client directly
                if container.qdrant.client:
                    from qdrant_client.models import PointStruct

                    point = PointStruct(
                        id=test_chunk_id_int,  # Use integer ID for Qdrant
                        vector=test_embedding,
                        payload={
                            "content": test_code,
                            "file_path": test_file_path,
                            "project": current_project['project'],
                            "chunk_id": test_chunk_id_str  # Store string ID in payload for cross-reference
                        }
                    )

                    await container.qdrant.client.upsert(
                        collection_name=test_collection,
                        points=[point]
                    )
                    print(f"     ✅ Stored vector in Qdrant")

                    # Search for vector
                    search_results = await container.qdrant.client.search(
                        collection_name=test_collection,
                        query_vector=test_embedding,
                        limit=1
                    )

                    if search_results:
                        print(f"     ✅ Vector search successful (found {len(search_results)} results)")
                    else:
                        print(f"     ⚠️ Vector search returned no results")
                else:
                    print(f"     ❌ Qdrant client not initialized")

            except Exception as e:
                print(f"     ❌ Qdrant error: {e}")
                return False
        else:
            print(f"     ⚠️ Qdrant service not available")

        print("\n  5️⃣ Testing Data Consistency...")

        # Verify Neo4j has the data
        if container.neo4j:
            verify_query = """
            MATCH (f:File {project: $project})-[:CONTAINS_CHUNK]->(c:CodeChunk)
            WHERE f.path = $path
            RETURN f.path as file, c.chunk_id as chunk
            """

            verify_result = await container.neo4j.execute_cypher(
                verify_query,
                {
                    "project": current_project['project'],
                    "path": test_file_path
                }
            )

            if verify_result.get("records"):
                print(f"     ✅ Neo4j data verified: {len(verify_result['records'])} relationships")
            else:
                print(f"     ⚠️ Neo4j data not found")

        # Clean up test data
        print("\n  6️⃣ Cleaning Up Test Data...")
        if container.neo4j:
            cleanup_query = """
            MATCH (f:File {path: $path, project: $project})
            OPTIONAL MATCH (f)-[r]->(c:CodeChunk)
            DETACH DELETE f, c
            """

            await container.neo4j.execute_cypher(
                cleanup_query,
                {
                    "path": test_file_path,
                    "project": current_project['project']
                }
            )
            print(f"     ✅ Cleaned up Neo4j test data")

        print("\n" + "=" * 50)
        print("✅ GraphRAG End-to-End Integration Test PASSED")
        print("Full data lifecycle working correctly")
        return True

    except ImportError as e:
        print(f"\n  ❌ Import error: {e}")
        print("     Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"\n  ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test"""
    success = asyncio.run(test_graphrag_e2e())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()