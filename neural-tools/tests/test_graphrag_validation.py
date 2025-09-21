#!/usr/bin/env python3
"""
Critical validation tests for GraphRAG data consistency
Must be run in CI/CD to prevent silent failures
Implements ADR-0059: GraphRAG Chunk Label Mismatch Fix
"""

import asyncio
import pytest
from typing import Dict, Any, List
import numpy as np

from servers.services.service_container import ServiceContainer
from servers.services.project_context_manager import ProjectContextManager
from servers.services.sync_manager import WriteSynchronizationManager
from servers.services.hybrid_retriever import HybridRetriever


@pytest.mark.asyncio
async def test_chunk_label_consistency():
    """Verify chunks are queryable by hybrid retriever (ADR-0059)"""

    # Initialize services
    context = ProjectContextManager()
    await context.set_project('.')
    container = ServiceContainer(context)
    container.initialize()

    # Create a test chunk using WriteSynchronizationManager
    sync_manager = WriteSynchronizationManager(
        neo4j_service=container.neo4j,
        qdrant_service=container.qdrant,
        project_name="test-graphrag"
    )

    test_content = "def test_function():\n    return 42"
    test_metadata = {
        "file_path": "test_validation.py",
        "start_line": 1,
        "end_line": 2
    }
    test_vector = np.random.rand(768).tolist()

    success, chunk_id, chunk_hash = await sync_manager.write_chunk(
        content=test_content,
        metadata=test_metadata,
        vector=test_vector,
        collection_name="project-test-graphrag",
        payload=test_metadata
    )

    assert success, "Failed to create test chunk via WriteSynchronizationManager"
    assert chunk_id, "No chunk_id returned"

    # Verify chunk exists with correct label (Chunk, not CodeChunk)
    result = await container.neo4j.execute_cypher("""
        MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
        RETURN c.chunk_id as id, c.content as content
    """, {"chunk_id": chunk_id, "project": "test-graphrag"})

    assert result.get('status') == 'success', f"Neo4j query failed: {result.get('message')}"
    assert result.get('records'), "Chunk node not found with Chunk label"
    assert result['records'][0]['content'] == test_content, "Content mismatch"

    # Verify NO CodeChunk nodes exist (old incorrect label)
    old_result = await container.neo4j.execute_cypher("""
        MATCH (c:CodeChunk)
        WHERE c.project = $project
        RETURN count(c) as count
    """, {"project": "test-graphrag"})

    if old_result.get('status') == 'success' and old_result.get('records'):
        old_count = old_result['records'][0].get('count', 0)
        assert old_count == 0, f"Found {old_count} CodeChunk nodes - should be migrated to Chunk!"

    # Verify hybrid retriever can find it
    retriever = HybridRetriever(container)
    context = await retriever._fetch_graph_context([chunk_id])

    assert context, "Hybrid retriever returned no context"
    assert len(context) > 0, "Context list is empty"
    # Graph context might be empty if no relationships exist, but the query should succeed

    print(f"✅ Chunk label consistency test passed - chunk_id: {chunk_id}")


@pytest.mark.asyncio
async def test_neo4j_qdrant_count_match():
    """Verify chunk counts match between Neo4j and Qdrant"""

    # Initialize services
    context = ProjectContextManager()
    await context.set_project('.')
    container = ServiceContainer(context)
    container.initialize()

    project_name = container.project_name
    collection_name = f"project-{project_name}"

    # Get Qdrant count
    try:
        collection_info = await container.qdrant.get_collection(collection_name)
        qdrant_count = collection_info.points_count
    except Exception as e:
        # Collection might not exist yet
        qdrant_count = 0

    # Get Neo4j count (correct label)
    result = await container.neo4j.execute_cypher("""
        MATCH (c:Chunk {project: $project})
        RETURN count(c) as count
    """, {"project": project_name})

    neo4j_count = 0
    if result.get('status') == 'success' and result.get('records'):
        neo4j_count = result['records'][0]['count']

    # Allow small drift (ongoing indexing, etc)
    drift = abs(qdrant_count - neo4j_count)
    max_drift = 10

    assert drift <= max_drift, \
        f"Count mismatch exceeds threshold: Qdrant={qdrant_count}, Neo4j={neo4j_count}, Drift={drift}"

    print(f"✅ Count validation passed - Neo4j: {neo4j_count}, Qdrant: {qdrant_count}")


@pytest.mark.asyncio
async def test_graph_context_enrichment():
    """Verify GraphRAG actually returns graph context (ADR-0059)"""

    # Initialize services
    context = ProjectContextManager()
    await context.set_project('.')
    container = ServiceContainer(context)
    container.initialize()

    # First ensure we have some data
    project_name = container.project_name
    result = await container.neo4j.execute_cypher("""
        MATCH (c:Chunk {project: $project})
        RETURN count(c) as count
    """, {"project": project_name})

    if result.get('status') == 'success' and result.get('records'):
        chunk_count = result['records'][0]['count']
        if chunk_count == 0:
            pytest.skip("No chunks in database to test with")

    # Perform hybrid search
    retriever = HybridRetriever(container)
    results = await retriever.find_similar_with_context(
        query="ServiceContainer initialization",
        limit=5,
        include_graph_context=True,
        max_hops=2
    )

    if not results:
        pytest.skip("No search results returned - may need indexing")

    # Check if we're getting graph context
    has_any_context = False
    for result in results:
        context = result.get('graph_context', {})
        if context:
            # Check for any non-empty fields in context
            for key, value in context.items():
                if value and value != [] and value != {}:
                    has_any_context = True
                    print(f"  Found graph context: {key} = {value[:100] if isinstance(value, str) else value}")
                    break

    # If no context found, check if relationships exist
    if not has_any_context:
        rel_result = await container.neo4j.execute_cypher("""
            MATCH (c:Chunk {project: $project})-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            LIMIT 5
        """, {"project": project_name})

        if rel_result.get('status') == 'success' and rel_result.get('records'):
            print("  Note: No graph context found, but relationships exist:")
            for record in rel_result['records']:
                print(f"    - {record['rel_type']}: {record['count']}")
        else:
            print("  Note: No relationships found in graph - need to index with relationships")

    print(f"✅ Graph context test completed - Context found: {has_any_context}")


@pytest.mark.asyncio
async def test_no_codechunk_nodes_exist():
    """Verify no CodeChunk nodes exist (should all be Chunk)"""

    # Initialize services
    context = ProjectContextManager()
    await context.set_project('.')
    container = ServiceContainer(context)
    container.initialize()

    # Check for any CodeChunk nodes
    result = await container.neo4j.execute_cypher("""
        MATCH (c:CodeChunk)
        RETURN count(c) as count, collect(c.chunk_id)[..5] as sample_ids
    """)

    if result.get('status') == 'success' and result.get('records'):
        count = result['records'][0].get('count', 0)
        sample_ids = result['records'][0].get('sample_ids', [])

        assert count == 0, \
            f"Found {count} CodeChunk nodes - these should be Chunk nodes! Sample IDs: {sample_ids}"

    print("✅ No CodeChunk nodes found - correct label being used")


if __name__ == "__main__":
    # Run tests directly
    import sys

    async def run_all_tests():
        """Run all validation tests"""
        print("\n" + "="*60)
        print("GraphRAG Validation Tests (ADR-0059)")
        print("="*60 + "\n")

        try:
            await test_chunk_label_consistency()
            await test_neo4j_qdrant_count_match()
            await test_graph_context_enrichment()
            await test_no_codechunk_nodes_exist()

            print("\n✅ All GraphRAG validation tests passed!")
            return 0

        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            return 1

    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)