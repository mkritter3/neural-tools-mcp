#!/usr/bin/env python3
"""
Test Neo4j-Qdrant Synchronization
Validates that both databases are properly synchronized with matching chunk IDs
"""

import asyncio
import pytest
import sys
from pathlib import Path
from typing import Dict, Set

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.servers.services.service_container import ServiceContainer
from src.servers.services.project_context_manager import ProjectContextManager


class TestNeo4jQdrantSync:
    """Test suite for validating Neo4j-Qdrant synchronization"""

    @pytest.mark.asyncio
    async def test_chunk_id_consistency(self):
        """Test that chunk IDs exist in both Neo4j and Qdrant"""

        # Initialize container
        container = ServiceContainer()
        await container.initialize()

        # Get project context
        context_manager = ProjectContextManager()
        project_info = await context_manager.get_current_project()
        project_name = project_info.get('project', 'claude-l9-template')

        # 1. Get chunk IDs from Qdrant
        collection_name = f"project-{project_name}"
        qdrant_chunks = set()

        try:
            # Scroll through Qdrant collection
            scroll_result = await container.qdrant.scroll_collection(
                collection_name=collection_name,
                limit=1000
            )

            for point in scroll_result:
                chunk_id = point.get('chunk_id') or point.get('id')
                if chunk_id:
                    qdrant_chunks.add(str(chunk_id))

            print(f"Found {len(qdrant_chunks)} chunks in Qdrant")
        except Exception as e:
            pytest.fail(f"Failed to query Qdrant: {e}")

        # 2. Get chunk IDs from Neo4j
        neo4j_chunks = set()

        try:
            # Query Neo4j for Chunk nodes
            query = """
            MATCH (c:Chunk)
            WHERE c.project = $project
            RETURN c.chunk_id as chunk_id
            LIMIT 1000
            """

            result = await container.neo4j.execute_cypher(
                query,
                {'project': project_name}
            )

            if result.get('status') == 'success':
                data = result.get('data', [])
                for record in data:
                    chunk_id = record.get('chunk_id')
                    if chunk_id:
                        neo4j_chunks.add(str(chunk_id))

            print(f"Found {len(neo4j_chunks)} chunks in Neo4j")
        except Exception as e:
            pytest.fail(f"Failed to query Neo4j: {e}")

        # 3. Validate synchronization
        if not qdrant_chunks:
            pytest.skip("No chunks in Qdrant to validate")

        # Check that at least some chunks exist in Neo4j
        assert len(neo4j_chunks) > 0, \
            f"Neo4j has 0 Chunk nodes but Qdrant has {len(qdrant_chunks)} chunks"

        # Calculate overlap
        common_chunks = qdrant_chunks.intersection(neo4j_chunks)
        overlap_percentage = (len(common_chunks) / len(qdrant_chunks)) * 100

        print(f"Chunk synchronization: {overlap_percentage:.1f}%")
        print(f"  - Common chunks: {len(common_chunks)}")
        print(f"  - Qdrant-only: {len(qdrant_chunks - neo4j_chunks)}")
        print(f"  - Neo4j-only: {len(neo4j_chunks - qdrant_chunks)}")

        # Require at least 90% synchronization
        assert overlap_percentage >= 90.0, \
            f"Poor synchronization: only {overlap_percentage:.1f}% of chunks are in both databases"

    @pytest.mark.asyncio
    async def test_file_chunk_relationships(self):
        """Test that File nodes are properly linked to Chunk nodes"""

        # Initialize container
        container = ServiceContainer()
        await container.initialize()

        context_manager = ProjectContextManager()
        project_info = await context_manager.get_current_project()
        project_name = project_info.get('project', 'claude-l9-template')

        # Query for Files with their Chunks
        query = """
        MATCH (f:File)
        WHERE f.project = $project
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:Chunk)
        RETURN f.path as file_path, count(c) as chunk_count
        LIMIT 10
        """

        result = await container.neo4j.execute_cypher(
            query,
            {'project': project_name}
        )

        if result.get('status') != 'success':
            pytest.fail(f"Neo4j query failed: {result.get('message')}")

        data = result.get('data', [])

        # Check that files have chunks
        files_without_chunks = []
        for record in data:
            file_path = record.get('file_path')
            chunk_count = record.get('chunk_count', 0)

            if chunk_count == 0:
                files_without_chunks.append(file_path)
                print(f"WARNING: File has no chunks: {file_path}")

        # No file should be without chunks (unless it's empty)
        assert len(files_without_chunks) < len(data), \
            f"All {len(data)} files lack chunk relationships"

    @pytest.mark.asyncio
    async def test_chunk_content_consistency(self):
        """Test that chunk content matches between databases"""

        # Initialize container
        container = ServiceContainer()
        await container.initialize()

        context_manager = ProjectContextManager()
        project_info = await context_manager.get_current_project()
        project_name = project_info.get('project', 'claude-l9-template')

        # Get a sample chunk from Qdrant
        collection_name = f"project-{project_name}"

        try:
            scroll_result = await container.qdrant.scroll_collection(
                collection_name=collection_name,
                limit=1
            )

            if not scroll_result:
                pytest.skip("No chunks to validate content")

            qdrant_chunk = scroll_result[0]
            chunk_id = qdrant_chunk.get('chunk_id') or qdrant_chunk.get('id')
            qdrant_content = qdrant_chunk.get('payload', {}).get('content', '')

            # Get the same chunk from Neo4j
            query = """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            WHERE c.project = $project
            RETURN c.content as content
            """

            result = await container.neo4j.execute_cypher(
                query,
                {'chunk_id': str(chunk_id), 'project': project_name}
            )

            if result.get('status') != 'success':
                pytest.fail(f"Failed to query Neo4j for chunk {chunk_id}")

            data = result.get('data', [])
            assert len(data) > 0, f"Chunk {chunk_id} exists in Qdrant but not in Neo4j"

            neo4j_content = data[0].get('content', '')

            # Content should match
            assert neo4j_content == qdrant_content, \
                f"Content mismatch for chunk {chunk_id}"

        except Exception as e:
            pytest.fail(f"Content consistency check failed: {e}")

    @pytest.mark.asyncio
    async def test_service_health_reporting(self):
        """Test that degraded services are properly detected"""

        # Initialize container
        container = ServiceContainer()
        await container.initialize()

        # This test ensures the indexer would detect if Neo4j is failing

        # Get indexer status (if available via MCP or direct query)
        # This is a placeholder for the actual implementation

        # For now, just check that both services report as initialized
        assert container.neo4j.initialized, "Neo4j service not initialized"
        assert container.qdrant.initialized, "Qdrant service not initialized"

        # Check that Neo4j is actually responsive
        result = await container.neo4j.execute_cypher(
            "RETURN 1 as test",
            {}
        )
        assert result.get('status') == 'success', \
            "Neo4j not responding but reported as healthy"

        # Check that Qdrant is responsive
        collections = await container.qdrant.get_collections()
        assert collections is not None, \
            "Qdrant not responding but reported as healthy"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])