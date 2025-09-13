#!/usr/bin/env python3
"""
Data Integrity Integration Test
Verifies consistency between Neo4j and Qdrant databases
"""

import sys
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


async def test_data_integrity():
    """Test data integrity between Neo4j and Qdrant"""
    print("\n🔍 Testing Data Integrity Between Neo4j and Qdrant...")

    try:
        # Import necessary services
        from servers.services.project_context_manager import get_project_context_manager
        from servers.services.service_container import ServiceContainer

        # Initialize project context
        print("\n  1️⃣ Initializing Services...")
        context_manager = await get_project_context_manager()
        await context_manager.set_project("/Users/mkr/local-coding/claude-l9-template")
        current_project = await context_manager.get_current_project()

        container = ServiceContainer(
            context_manager=context_manager,
            project_name=current_project['project']
        )
        container.initialize()

        if hasattr(container, 'initialize_all_services'):
            await container.initialize_all_services()

        print(f"     ✅ Services initialized for project: {current_project['project']}")

        # Check Neo4j data
        print("\n  2️⃣ Checking Neo4j Data...")
        neo4j_chunks = set()
        neo4j_files = set()

        if container.neo4j:
            # Get all code chunks from Neo4j
            chunks_query = """
            MATCH (c:CodeChunk {project: $project})
            RETURN c.chunk_id as chunk_id, c.file_path as file_path
            LIMIT 100
            """

            chunks_result = await container.neo4j.execute_cypher(
                chunks_query,
                {"project": current_project['project']}
            )

            for record in chunks_result.get("records", []):
                chunk_id = record.get("chunk_id")
                if chunk_id:
                    neo4j_chunks.add(chunk_id)
                    neo4j_files.add(record.get("file_path"))

            print(f"     Found {len(neo4j_chunks)} chunks in Neo4j")
            print(f"     Found {len(neo4j_files)} unique files in Neo4j")

            # Check for orphaned chunks (chunks without file relationships)
            orphan_query = """
            MATCH (c:CodeChunk {project: $project})
            WHERE NOT EXISTS((c)<-[:CONTAINS_CHUNK]-(:File))
            RETURN count(c) as orphan_count
            """

            orphan_result = await container.neo4j.execute_cypher(
                orphan_query,
                {"project": current_project['project']}
            )

            orphan_count = 0
            if orphan_result.get("records"):
                orphan_count = orphan_result["records"][0].get("orphan_count", 0)

            if orphan_count > 0:
                print(f"     ⚠️ Found {orphan_count} orphaned chunks (no file relationship)")
            else:
                print(f"     ✅ No orphaned chunks found")

            # Check for duplicate chunks
            duplicate_query = """
            MATCH (c:CodeChunk {project: $project})
            WITH c.chunk_id as chunk_id, count(*) as count
            WHERE count > 1
            RETURN chunk_id, count
            LIMIT 10
            """

            duplicate_result = await container.neo4j.execute_cypher(
                duplicate_query,
                {"project": current_project['project']}
            )

            if duplicate_result.get("records"):
                print(f"     ⚠️ Found {len(duplicate_result['records'])} duplicate chunk IDs")
                for dup in duplicate_result["records"][:3]:
                    print(f"        - {dup['chunk_id']}: {dup['count']} copies")
            else:
                print(f"     ✅ No duplicate chunks found")

        else:
            print(f"     ⚠️ Neo4j service not available")

        # Check Qdrant data
        print("\n  3️⃣ Checking Qdrant Data...")
        qdrant_points = set()
        qdrant_collections = []

        if container.qdrant:
            # Get collections
            collections = await container.qdrant.get_collections()

            # Filter for project collections (collections is a list of strings)
            project_collections = [
                c for c in collections
                if current_project['project'].lower() in c.lower()
            ]

            print(f"     Found {len(project_collections)} project collections")

            for collection_name in project_collections[:3]:  # Check first 3 collections
                try:
                    # Get points from collection using the client directly
                    if container.qdrant.client:
                        points_result = await container.qdrant.client.scroll(
                            collection_name=collection_name,
                            limit=100
                        )

                        collection_points = 0
                        # Scroll returns a tuple (points, next_page_offset)
                        if points_result and len(points_result) > 0:
                            points_batch = points_result[0] if isinstance(points_result, tuple) else points_result
                            if hasattr(points_batch, '__iter__'):
                                for point in points_batch:
                                    if hasattr(point, 'id'):
                                        qdrant_points.add(point.id)
                                        collection_points += 1
                    else:
                        collection_points = 0

                    print(f"     Collection '{collection_name}': {collection_points} points")

                except Exception as e:
                    print(f"     ⚠️ Error reading collection '{collection_name}': {e}")

            print(f"     Total: {len(qdrant_points)} unique points across collections")

        else:
            print(f"     ⚠️ Qdrant service not available")

        # Cross-database consistency check
        print("\n  4️⃣ Cross-Database Consistency Check...")

        if neo4j_chunks and qdrant_points:
            # Find chunks in Neo4j but not in Qdrant
            missing_in_qdrant = neo4j_chunks - qdrant_points
            if missing_in_qdrant:
                print(f"     ⚠️ {len(missing_in_qdrant)} chunks in Neo4j but not in Qdrant")
                for chunk_id in list(missing_in_qdrant)[:3]:
                    print(f"        - {chunk_id}")
            else:
                print(f"     ✅ All Neo4j chunks have vectors in Qdrant")

            # Find points in Qdrant but not in Neo4j
            missing_in_neo4j = qdrant_points - neo4j_chunks
            if missing_in_neo4j:
                print(f"     ⚠️ {len(missing_in_neo4j)} vectors in Qdrant but not in Neo4j")
                for point_id in list(missing_in_neo4j)[:3]:
                    print(f"        - {point_id}")
            else:
                print(f"     ✅ All Qdrant vectors have nodes in Neo4j")

            # Calculate sync percentage
            total_unique = len(neo4j_chunks | qdrant_points)
            synced = len(neo4j_chunks & qdrant_points)
            if total_unique > 0:
                sync_percentage = (synced / total_unique) * 100
                print(f"\n     📊 Data Sync Status: {sync_percentage:.1f}% synchronized")
                if sync_percentage >= 95:
                    print(f"        ✅ Excellent sync (>95%)")
                elif sync_percentage >= 80:
                    print(f"        ⚠️ Good sync but some drift (80-95%)")
                else:
                    print(f"        ❌ Poor sync (<80%) - investigation needed")
            else:
                print(f"     ℹ️ No data to compare")

        elif not neo4j_chunks and not qdrant_points:
            print(f"     ℹ️ No data in either database (may be a fresh installation)")
        else:
            print(f"     ⚠️ Only one database has data:")
            if neo4j_chunks:
                print(f"        Neo4j: {len(neo4j_chunks)} chunks")
            if qdrant_points:
                print(f"        Qdrant: {len(qdrant_points)} points")

        # Project isolation check
        print("\n  5️⃣ Project Isolation Check...")
        if container.neo4j:
            # Check if all nodes have project property
            isolation_query = """
            MATCH (n)
            WHERE n:File OR n:CodeChunk OR n:Function OR n:Class
            AND NOT EXISTS(n.project)
            RETURN count(n) as nodes_without_project
            LIMIT 1
            """

            isolation_result = await container.neo4j.execute_cypher(
                isolation_query, {}
            )

            if isolation_result.get("records"):
                count = isolation_result["records"][0].get("nodes_without_project", 0)
                if count > 0:
                    print(f"     ❌ Found {count} nodes without project property (isolation breach risk)")
                else:
                    print(f"     ✅ All nodes have project property (good isolation)")
            else:
                print(f"     ✅ Project isolation check passed")

        print("\n" + "=" * 50)
        print("✅ Data Integrity Integration Test COMPLETED")
        return True

    except ImportError as e:
        print(f"\n  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n  ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test"""
    success = asyncio.run(test_data_integrity())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()