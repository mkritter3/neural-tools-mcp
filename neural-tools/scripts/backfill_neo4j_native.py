#!/usr/bin/env python3
"""
Simple Neo4j native backfill using CALL {} IN TRANSACTIONS
No APOC required - uses Neo4j 4.4+ native batching
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def backfill_with_native_neo4j():
    """Use Neo4j's native CALL {} IN TRANSACTIONS for batch updates"""
    import sys
    sys.path.insert(0, '../src')
    from servers.services.service_container import ServiceContainer
    from servers.services.project_context_manager import get_project_context_manager

    # Get project context
    manager = await get_project_context_manager()
    await manager.set_project('/Users/mkr/local-coding/claude-l9-template')

    # Initialize Neo4j connection
    container = ServiceContainer(context_manager=manager)
    await container.initialize_all_services()

    # Step 1: Add metadata_version field to all chunks (if missing)
    logger.info("Step 1: Adding metadata_version field...")

    # Simple update without batching (Neo4j handles internally)
    init_query = """
    MATCH (c:Chunk {project: $project})
    WHERE c.metadata_version IS NULL
    SET c.metadata_version = '1.0',
        c.backfill_timestamp = datetime()
    RETURN count(c) as updated
    """

    result = await container.neo4j.execute_cypher(
        init_query,
        {'project': 'claude-l9-template'}
    )

    if result.get('status') == 'success' and result.get('result'):
        total = result['result'][0]['updated']
        logger.info(f"  ✓ Initialized {total} chunks to version 1.0")

    # Step 2: Add new v1.1 fields with defaults
    logger.info("Step 2: Adding v1.1 fields with defaults...")

    # Get current timestamp for file_modified_at default
    now = datetime.utcnow().isoformat()

    # Simple update query without CALL IN TRANSACTIONS
    v11_fields_query = """
    MATCH (c:Chunk {project: $project})
    WHERE c.metadata_version = '1.0'
    SET c.file_modified_at = coalesce(c.file_modified_at, $now)
        SET c.file_created_at = coalesce(c.file_created_at, $now)
        SET c.git_commit_sha = coalesce(c.git_commit_sha, 'unknown')
        SET c.git_author = coalesce(c.git_author, 'unknown')
        SET c.file_type = coalesce(c.file_type,
            CASE
                WHEN c.file_path ENDS WITH '.py' THEN 'py'
                WHEN c.file_path ENDS WITH '.js' THEN 'js'
                WHEN c.file_path ENDS WITH '.ts' THEN 'ts'
                WHEN c.file_path ENDS WITH '.tsx' THEN 'tsx'
                WHEN c.file_path ENDS WITH '.jsx' THEN 'jsx'
                WHEN c.file_path ENDS WITH '.java' THEN 'java'
                WHEN c.file_path ENDS WITH '.go' THEN 'go'
                WHEN c.file_path ENDS WITH '.rs' THEN 'rs'
                WHEN c.file_path ENDS WITH '.cpp' THEN 'cpp'
                WHEN c.file_path ENDS WITH '.c' THEN 'c'
                WHEN c.file_path ENDS WITH '.cs' THEN 'cs'
                WHEN c.file_path ENDS WITH '.rb' THEN 'rb'
                WHEN c.file_path ENDS WITH '.php' THEN 'php'
                WHEN c.file_path ENDS WITH '.swift' THEN 'swift'
                WHEN c.file_path ENDS WITH '.kt' THEN 'kt'
                WHEN c.file_path ENDS WITH '.scala' THEN 'scala'
                WHEN c.file_path ENDS WITH '.r' THEN 'r'
                WHEN c.file_path ENDS WITH '.m' THEN 'm'
                WHEN c.file_path ENDS WITH '.mm' THEN 'mm'
                WHEN c.file_path ENDS WITH '.h' THEN 'h'
                WHEN c.file_path ENDS WITH '.hpp' THEN 'hpp'
                WHEN c.file_path ENDS WITH '.md' THEN 'md'
                WHEN c.file_path ENDS WITH '.json' THEN 'json'
                WHEN c.file_path ENDS WITH '.yaml' THEN 'yaml'
                WHEN c.file_path ENDS WITH '.yml' THEN 'yml'
                WHEN c.file_path ENDS WITH '.xml' THEN 'xml'
                WHEN c.file_path ENDS WITH '.html' THEN 'html'
                WHEN c.file_path ENDS WITH '.css' THEN 'css'
                WHEN c.file_path ENDS WITH '.scss' THEN 'scss'
                WHEN c.file_path ENDS WITH '.sass' THEN 'sass'
                WHEN c.file_path ENDS WITH '.less' THEN 'less'
                WHEN c.file_path ENDS WITH '.vue' THEN 'vue'
                WHEN c.file_path ENDS WITH '.svelte' THEN 'svelte'
                ELSE 'unknown'
            END
        )
        SET c.language = coalesce(c.language,
            CASE
                WHEN c.file_type IN ['py'] THEN 'python'
                WHEN c.file_type IN ['js', 'jsx'] THEN 'javascript'
                WHEN c.file_type IN ['ts', 'tsx'] THEN 'typescript'
                WHEN c.file_type = 'java' THEN 'java'
                WHEN c.file_type = 'go' THEN 'go'
                WHEN c.file_type = 'rs' THEN 'rust'
                WHEN c.file_type IN ['cpp', 'cc', 'cxx'] THEN 'cpp'
                WHEN c.file_type = 'c' THEN 'c'
                WHEN c.file_type = 'cs' THEN 'csharp'
                WHEN c.file_type = 'rb' THEN 'ruby'
                WHEN c.file_type = 'php' THEN 'php'
                WHEN c.file_type = 'swift' THEN 'swift'
                WHEN c.file_type = 'kt' THEN 'kotlin'
                WHEN c.file_type = 'scala' THEN 'scala'
                WHEN c.file_type = 'r' THEN 'r'
                WHEN c.file_type IN ['m', 'mm'] THEN 'objective-c'
                WHEN c.file_type IN ['h', 'hpp'] THEN 'cpp'
                WHEN c.file_type IN ['md'] THEN 'markdown'
                WHEN c.file_type IN ['json', 'yaml', 'yml'] THEN 'config'
                WHEN c.file_type IN ['xml', 'html'] THEN 'markup'
                WHEN c.file_type IN ['css', 'scss', 'sass', 'less'] THEN 'stylesheet'
                WHEN c.file_type IN ['vue', 'svelte'] THEN 'framework'
                ELSE 'unknown'
            END
        )
        SET c.is_canonical = coalesce(c.is_canonical,
            NOT (c.file_path CONTAINS 'test'
                OR c.file_path CONTAINS 'spec'
                OR c.file_path CONTAINS 'example'
                OR c.file_path CONTAINS 'demo'
                OR c.file_path CONTAINS '__pycache__'
                OR c.file_path CONTAINS 'node_modules'
                OR c.file_path CONTAINS '.git'
                OR c.file_path STARTS WITH '.')
        )
        SET c.metadata_version = '1.1'
    RETURN count(c) as updated
    """

    result = await container.neo4j.execute_cypher(
        v11_fields_query,
        {'project': 'claude-l9-template', 'now': now}
    )

    if result.get('status') == 'success' and result.get('result'):
        total = result['result'][0]['updated']
        logger.info(f"  ✓ Upgraded {total} chunks to version 1.1")

    # Step 3: Verify the backfill
    logger.info("Step 3: Verifying backfill...")
    verify_query = """
    MATCH (c:Chunk {project: $project})
    WITH
        count(CASE WHEN c.metadata_version = '1.1' THEN 1 END) as v11_count,
        count(CASE WHEN c.metadata_version = '1.0' THEN 1 END) as v10_count,
        count(CASE WHEN c.metadata_version IS NULL THEN 1 END) as null_count,
        count(c) as total
    RETURN v11_count, v10_count, null_count, total
    """

    result = await container.neo4j.execute_cypher(
        verify_query,
        {'project': 'claude-l9-template'}
    )

    if result.get('status') == 'success' and result.get('result'):
        stats = result['result'][0]
        logger.info(f"  Version 1.1: {stats['v11_count']} chunks")
        logger.info(f"  Version 1.0: {stats['v10_count']} chunks")
        logger.info(f"  Unversioned: {stats['null_count']} chunks")
        logger.info(f"  Total: {stats['total']} chunks")

        if stats['v11_count'] == stats['total']:
            logger.info("✅ All chunks successfully upgraded to v1.1!")
        elif stats['v11_count'] > 0:
            logger.info(f"⚠️ Partial upgrade: {stats['v11_count']}/{stats['total']} chunks at v1.1")
        else:
            logger.info("❌ No chunks upgraded")

    # Cleanup isn't needed - connections are pooled


if __name__ == "__main__":
    asyncio.run(backfill_with_native_neo4j())