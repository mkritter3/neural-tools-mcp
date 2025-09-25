#!/usr/bin/env python3
"""
ADR-0102: Diagnose search issues related to project field
This script diagnoses why search is not working by checking:
1. What project the system thinks it's using
2. What project values exist in the database
3. Whether there's a mismatch

Usage:
    python diagnose-search-issue.py
"""
import os
import sys
import asyncio
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'neural-tools', 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def diagnose_search_issue():
    """Diagnose why search is not returning results"""
    try:
        # Import required modules
        from servers.services.project_context_manager import get_project_context_manager
        from servers.services.service_container import ServiceContainer

        logger.info("=" * 60)
        logger.info("NEURAL TOOLS SEARCH DIAGNOSTIC")
        logger.info("=" * 60)

        # 1. Check what project the system detects
        context_manager = await get_project_context_manager()
        project_info = await context_manager.get_current_project(force_refresh=True)
        detected_project = project_info.get("project")
        detected_path = project_info.get("path")

        logger.info("\n📍 DETECTED PROJECT CONTEXT:")
        logger.info(f"   Project Name: {detected_project}")
        logger.info(f"   Project Path: {detected_path}")
        logger.info(f"   Detection Source: {project_info.get('source', 'unknown')}")

        if not detected_project:
            logger.error("❌ No project detected! This is why search isn't working.")
            logger.error("   Use set_project_context tool to set a project.")
            return

        # 2. Initialize service container
        logger.info("\n🔄 Connecting to Neo4j...")
        container = ServiceContainer(project_name=detected_project)
        await container.initialize_all_services()

        # 3. Check what projects exist in the database
        projects_query = """
        MATCH (n)
        WHERE n.project IS NOT NULL
        RETURN DISTINCT n.project as project, labels(n)[0] as node_type, count(n) as count
        ORDER BY count DESC
        """

        logger.info("\n📊 DATABASE PROJECT DISTRIBUTION:")
        result = await container.neo4j.execute_cypher(projects_query, {})
        if result.get('status') == 'success' and result.get('result'):
            total_with_project = 0
            projects_found = set()
            for row in result['result']:
                project = row.get('project')
                node_type = row.get('node_type')
                count = row.get('count')
                projects_found.add(project)
                total_with_project += count
                logger.info(f"   {project:30} | {node_type:10} | {count:,} nodes")

            if detected_project not in projects_found:
                logger.warning(f"\n⚠️  MISMATCH: Detected project '{detected_project}' not found in database!")
                logger.warning(f"   Database has: {list(projects_found)}")
        else:
            logger.info("   No nodes with project field found")

        # 4. Check for NULL project nodes
        null_query = """
        MATCH (n)
        WHERE n.project IS NULL OR n.project = ''
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY count DESC
        """

        logger.info("\n🔍 NULL PROJECT ANALYSIS:")
        null_result = await container.neo4j.execute_cypher(null_query, {})
        if null_result.get('status') == 'success' and null_result.get('result'):
            total_null = 0
            for row in null_result['result']:
                node_type = row.get('node_type')
                count = row.get('count')
                total_null += count
                logger.info(f"   {node_type:10} | {count:,} nodes with NULL project")

            if total_null > 0:
                logger.warning(f"\n⚠️  Found {total_null:,} nodes with NULL project field!")
                logger.warning("   These nodes won't be found by search!")
        else:
            logger.info("   ✅ No nodes with NULL project found")

        # 5. Check specific Chunk statistics for detected project
        chunk_query = """
        MATCH (c:Chunk)
        RETURN
            count(CASE WHEN c.project = $project THEN 1 END) as matching_project,
            count(CASE WHEN c.project IS NULL THEN 1 END) as null_project,
            count(CASE WHEN c.project <> $project AND c.project IS NOT NULL THEN 1 END) as other_project,
            count(c) as total_chunks
        """

        logger.info(f"\n📈 CHUNK STATISTICS FOR PROJECT '{detected_project}':")
        chunk_result = await container.neo4j.execute_cypher(
            chunk_query,
            {'project': detected_project}
        )
        if chunk_result.get('status') == 'success' and chunk_result.get('result'):
            stats = chunk_result['result'][0]
            matching = stats.get('matching_project', 0)
            null_proj = stats.get('null_project', 0)
            other = stats.get('other_project', 0)
            total = stats.get('total_chunks', 0)

            logger.info(f"   Total Chunks: {total:,}")
            logger.info(f"   Matching '{detected_project}': {matching:,} ({matching/max(total,1)*100:.1f}%)")
            logger.info(f"   NULL project: {null_proj:,} ({null_proj/max(total,1)*100:.1f}%)")
            logger.info(f"   Other projects: {other:,} ({other/max(total,1)*100:.1f}%)")

            # 6. Diagnosis
            logger.info("\n🔬 DIAGNOSIS:")
            if matching == 0 and null_proj > 0:
                logger.error("❌ PROBLEM IDENTIFIED: All chunks have NULL project!")
                logger.error(f"   Search is looking for project='{detected_project}' but finds nothing.")
                logger.error(f"   SOLUTION: Run ./scripts/fix-null-project-chunks.py {detected_project}")
            elif matching == 0 and other > 0:
                logger.error("❌ PROBLEM IDENTIFIED: Chunks belong to different project!")
                logger.error(f"   Search is looking for project='{detected_project}' but chunks have other projects.")
                logger.error(f"   SOLUTION: Reindex the project or fix project names.")
            elif matching > 0:
                logger.info(f"✅ {matching} chunks found for project '{detected_project}'")
                logger.info("   Search should be working. If not, check:")
                logger.info("   1. Vector embeddings are present")
                logger.info("   2. Search query is using correct project name")

        # 7. Test actual search
        logger.info(f"\n🧪 TESTING SEARCH FOR PROJECT '{detected_project}':")

        # Simple test query
        test_query = """
        MATCH (c:Chunk {project: $project})
        RETURN c.chunk_id as chunk_id, c.file_path as file_path
        LIMIT 3
        """

        test_result = await container.neo4j.execute_cypher(
            test_query,
            {'project': detected_project}
        )

        if test_result.get('status') == 'success' and test_result.get('result'):
            logger.info(f"   ✅ Found {len(test_result['result'])} sample chunks:")
            for chunk in test_result['result']:
                logger.info(f"      - {chunk.get('file_path', 'unknown')}")
        else:
            logger.error("   ❌ No chunks found for this project!")

        # 8. Check if indexer is running
        logger.info("\n🐳 INDEXER STATUS:")
        try:
            from servers.services.indexer_orchestrator import IndexerOrchestrator
            orchestrator = IndexerOrchestrator()
            orchestrator.docker_client = None  # Will be initialized
            await orchestrator.initialize()

            # Check for indexer container
            if orchestrator.docker_client:
                containers = orchestrator.docker_client.containers.list(
                    filters={'label': f'com.l9.project={detected_project}'}
                )
                if containers:
                    for c in containers:
                        logger.info(f"   ✅ Indexer running: {c.name} (status: {c.status})")
                else:
                    logger.warning(f"   ⚠️  No indexer container found for project '{detected_project}'")
        except Exception as e:
            logger.warning(f"   Could not check indexer status: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("DIAGNOSTIC COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Diagnostic error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    asyncio.run(diagnose_search_issue())


if __name__ == "__main__":
    main()