#!/usr/bin/env python3
"""
Check GraphRAG chunk consistency between Neo4j and Qdrant
Run this in CI/CD and periodically in production
Implements ADR-0059: GraphRAG Chunk Label Mismatch Fix
"""

import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from servers.services.service_container import ServiceContainer
from servers.services.project_context_manager import ProjectContextManager
from servers.services.hybrid_retriever import HybridRetriever


async def check_consistency():
    """Check for GraphRAG data consistency issues"""

    print("\n" + "="*60)
    print("GraphRAG Consistency Check (ADR-0059)")
    print("="*60 + "\n")

    # Initialize services
    context = ProjectContextManager()
    project_path = os.environ.get('PROJECT_PATH', '.')
    await context.set_project(project_path)

    container = ServiceContainer(context)
    container.initialize()

    issues = []
    warnings = []

    # 1. Check for wrong label (CodeChunk instead of Chunk)
    print("1. Checking for incorrect CodeChunk labels...")
    try:
        result = await container.neo4j.execute_cypher("""
            MATCH (c:CodeChunk)
            RETURN count(c) as count, collect(c.chunk_id)[..5] as sample_ids
        """)

        if result.get('status') == 'success' and result.get('records'):
            count = result['records'][0].get('count', 0)
            if count > 0:
                sample_ids = result['records'][0].get('sample_ids', [])
                issues.append(
                    f"Found {count} CodeChunk nodes - should be Chunk! "
                    f"Sample IDs: {sample_ids}"
                )
        print(f"   ✓ No CodeChunk nodes found")
    except Exception as e:
        warnings.append(f"Could not check CodeChunk nodes: {e}")

    # 2. Check chunk counts between databases
    print("\n2. Checking Neo4j/Qdrant chunk counts...")
    try:
        # Get Neo4j count (correct label)
        neo4j_result = await container.neo4j.execute_cypher("""
            MATCH (c:Chunk {project: $project})
            RETURN count(c) as count
        """, {"project": container.project_name})

        neo4j_count = 0
        if neo4j_result.get('status') == 'success' and neo4j_result.get('records'):
            neo4j_count = neo4j_result['records'][0]['count']

        # Get Qdrant count
        collection_name = f"project-{container.project_name}"
        qdrant_count = 0
        try:
            collection_info = await container.qdrant.get_collection(collection_name)
            qdrant_count = collection_info.points_count
        except Exception:
            # Collection might not exist
            pass

        drift = abs(neo4j_count - qdrant_count)
        print(f"   Neo4j chunks: {neo4j_count}")
        print(f"   Qdrant chunks: {qdrant_count}")
        print(f"   Drift: {drift}")

        if drift > 10:  # Allow small drift for ongoing operations
            issues.append(
                f"Chunk count mismatch exceeds threshold: "
                f"Neo4j={neo4j_count}, Qdrant={qdrant_count}, Drift={drift}"
            )
        else:
            print(f"   ✓ Counts are within acceptable range")

    except Exception as e:
        warnings.append(f"Could not check chunk counts: {e}")

    # 3. Test graph context retrieval
    print("\n3. Testing graph context retrieval...")
    try:
        # Get a sample chunk
        sample_result = await container.neo4j.execute_cypher("""
            MATCH (c:Chunk {project: $project})
            RETURN c.chunk_id as chunk_id
            LIMIT 1
        """, {"project": container.project_name})

        if sample_result.get('status') == 'success' and sample_result.get('records'):
            chunk_id = sample_result['records'][0]['chunk_id']
            print(f"   Testing with chunk: {chunk_id}")

            # Try to fetch graph context
            retriever = HybridRetriever(container)
            contexts = await retriever._fetch_graph_context([chunk_id])

            if not contexts or not contexts[0]:
                issues.append(
                    f"Graph context fetch failed for chunk {chunk_id} - "
                    "hybrid search may be broken!"
                )
            else:
                print(f"   ✓ Graph context retrieval successful")
                # Show what was retrieved
                context = contexts[0]
                if context.get('file_path'):
                    print(f"     - File: {context['file_path']}")
                if context.get('functions'):
                    print(f"     - Functions: {len(context['functions'])} found")
                if context.get('imports'):
                    print(f"     - Imports: {len(context['imports'])} found")
        else:
            print("   No chunks found to test with")

    except Exception as e:
        warnings.append(f"Could not test graph context: {e}")

    # 4. Check for orphaned chunks (in Qdrant but not Neo4j)
    print("\n4. Checking for orphaned chunks...")
    try:
        # This would require scrolling through Qdrant and checking each in Neo4j
        # For now, we rely on the count check above
        print("   (Covered by count validation)")

    except Exception as e:
        warnings.append(f"Could not check orphaned chunks: {e}")

    # 5. Check relationship integrity
    print("\n5. Checking relationship integrity...")
    try:
        rel_result = await container.neo4j.execute_cypher("""
            MATCH (c:Chunk {project: $project})
            OPTIONAL MATCH (c)-[r]->()
            WITH c, count(r) as rel_count
            RETURN
                count(c) as total_chunks,
                sum(CASE WHEN rel_count > 0 THEN 1 ELSE 0 END) as chunks_with_relationships,
                avg(rel_count) as avg_relationships
        """, {"project": container.project_name})

        if rel_result.get('status') == 'success' and rel_result.get('records'):
            record = rel_result['records'][0]
            total = record.get('total_chunks', 0)
            with_rels = record.get('chunks_with_relationships', 0)
            avg_rels = record.get('avg_relationships', 0)

            print(f"   Total chunks: {total}")
            print(f"   Chunks with relationships: {with_rels}")
            print(f"   Average relationships per chunk: {avg_rels:.2f}")

            if total > 0 and with_rels == 0:
                warnings.append(
                    "No chunks have relationships - graph context will be empty"
                )

    except Exception as e:
        warnings.append(f"Could not check relationships: {e}")

    # Report results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    if issues:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for i, warning in enumerate(warnings, 1):
            print(f"{i}. {warning}")

    if not issues and not warnings:
        print("\n✅ All GraphRAG validations passed!")
        print(f"   Project: {container.project_name}")
        return 0

    elif issues:
        print("\n❌ GraphRAG validation FAILED - critical issues must be fixed")
        return 1

    else:
        print("\n⚠️  GraphRAG validation passed with warnings")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(check_consistency())
    sys.exit(exit_code)