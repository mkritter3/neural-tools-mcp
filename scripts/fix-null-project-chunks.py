#!/usr/bin/env python3
"""
ADR-0102: Fix NULL project fields in existing chunks
One-time migration script to backfill project field for chunks that were indexed
before proper project handling was implemented.

Usage:
    python fix-null-project-chunks.py [project_name]

If project_name is not provided, will attempt to detect from current directory.
"""
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'neural-tools', 'src'))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def fix_null_project_chunks(project_name: str = None):
    """Fix chunks with NULL project field"""
    try:
        # Import required modules
        from servers.services.project_context_manager import get_project_context_manager
        from servers.services.service_container import ServiceContainer

        # Get project context
        context_manager = await get_project_context_manager()

        if project_name:
            # Use provided project name
            logger.info(f"Using provided project name: {project_name}")
            await context_manager.set_project(os.getcwd())
            # Override detected name with provided one
            project_info = await context_manager.get_current_project()
            project_info['project'] = project_name
        else:
            # Auto-detect project
            project_info = await context_manager.get_current_project(force_refresh=True)
            project_name = project_info.get("project")

        if not project_name:
            logger.error("Could not determine project name. Please provide it as an argument.")
            sys.exit(1)

        logger.info(f"🎯 Fixing NULL project chunks for: {project_name}")
        logger.info(f"   Project path: {project_info.get('path')}")

        # Initialize service container
        container = ServiceContainer(project_name=project_name)
        await container.initialize_all_services()

        # First, count chunks with NULL project
        count_query = """
        MATCH (c:Chunk)
        WHERE c.project IS NULL OR c.project = ''
        RETURN count(c) as null_count
        """

        result = await container.neo4j.execute_cypher(count_query, {})
        if result.get('status') != 'success' or not result.get('result'):
            logger.info("No chunks with NULL project found.")
            return

        null_count = result['result'][0].get('null_count', 0)
        logger.info(f"📊 Found {null_count} chunks with NULL project field")

        if null_count == 0:
            logger.info("✅ No chunks need fixing!")
            return

        # Get sample of NULL chunks to verify
        sample_query = """
        MATCH (c:Chunk)
        WHERE c.project IS NULL OR c.project = ''
        RETURN c.chunk_id as chunk_id, c.file_path as file_path
        LIMIT 5
        """

        sample_result = await container.neo4j.execute_cypher(sample_query, {})
        if sample_result.get('status') == 'success' and sample_result.get('result'):
            logger.info("📋 Sample of chunks to be fixed:")
            for chunk in sample_result['result']:
                logger.info(f"   - {chunk.get('chunk_id', 'unknown')}: {chunk.get('file_path', 'unknown')}")

        # Ask for confirmation
        response = input(f"\n⚠️  This will update {null_count} chunks with project='{project_name}'. Continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Operation cancelled.")
            return

        # Update chunks with NULL project
        update_query = """
        MATCH (c:Chunk)
        WHERE c.project IS NULL OR c.project = ''
        SET c.project = $project
        RETURN count(c) as updated_count
        """

        logger.info(f"🔄 Updating chunks with project='{project_name}'...")
        update_result = await container.neo4j.execute_cypher(
            update_query,
            {'project': project_name}
        )

        if update_result.get('status') == 'success':
            updated_count = update_result['result'][0].get('updated_count', 0)
            logger.info(f"✅ Successfully updated {updated_count} chunks!")
        else:
            logger.error(f"❌ Failed to update chunks: {update_result.get('error')}")
            return

        # Also fix File nodes
        file_count_query = """
        MATCH (f:File)
        WHERE f.project IS NULL OR f.project = ''
        RETURN count(f) as null_count
        """

        file_result = await container.neo4j.execute_cypher(file_count_query, {})
        if file_result.get('status') == 'success' and file_result.get('result'):
            file_null_count = file_result['result'][0].get('null_count', 0)
            if file_null_count > 0:
                logger.info(f"📊 Found {file_null_count} files with NULL project field")

                # Update files
                file_update_query = """
                MATCH (f:File)
                WHERE f.project IS NULL OR f.project = ''
                SET f.project = $project
                RETURN count(f) as updated_count
                """

                file_update_result = await container.neo4j.execute_cypher(
                    file_update_query,
                    {'project': project_name}
                )

                if file_update_result.get('status') == 'success':
                    file_updated = file_update_result['result'][0].get('updated_count', 0)
                    logger.info(f"✅ Successfully updated {file_updated} files!")

        # Verify the fix
        verify_query = """
        MATCH (c:Chunk {project: $project})
        RETURN count(c) as project_count
        """

        verify_result = await container.neo4j.execute_cypher(
            verify_query,
            {'project': project_name}
        )

        if verify_result.get('status') == 'success':
            project_count = verify_result['result'][0].get('project_count', 0)
            logger.info(f"\n🎉 Verification: {project_count} chunks now have project='{project_name}'")
            logger.info(f"✨ Search should now work correctly for project '{project_name}'!")

        # Check if any chunks still have NULL project
        remaining_query = """
        MATCH (c:Chunk)
        WHERE c.project IS NULL OR c.project = ''
        RETURN count(c) as remaining_null
        """

        remaining_result = await container.neo4j.execute_cypher(remaining_query, {})
        if remaining_result.get('status') == 'success':
            remaining = remaining_result['result'][0].get('remaining_null', 0)
            if remaining > 0:
                logger.warning(f"⚠️  {remaining} chunks still have NULL project - may belong to other projects")

    except Exception as e:
        logger.error(f"Error fixing NULL project chunks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    # Get project name from command line if provided
    project_name = sys.argv[1] if len(sys.argv) > 1 else None

    if project_name:
        logger.info(f"Using provided project name: {project_name}")
    else:
        logger.info("No project name provided, will attempt auto-detection...")

    # Run the async function
    asyncio.run(fix_null_project_chunks(project_name))


if __name__ == "__main__":
    main()