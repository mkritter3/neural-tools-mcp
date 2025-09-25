#!/usr/bin/env python3
"""
Simple Schema Backfill Tool
Adds new metadata fields to existing Chunk nodes using native Neo4j

Author: L9 Engineering Team
Date: 2025-09-24
"""

from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Tool configuration for MCP auto-discovery
TOOL_CONFIG = {
    "name": "schema_backfill",
    "description": "Simple backfill tool for adding metadata fields to Chunk nodes. No APOC required.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["status", "backfill", "verify"],
                "default": "status",
                "description": "Operation: status (check current state), backfill (add v1.1 fields), verify (check results)"
            },
            "project": {
                "type": "string",
                "description": "Project to backfill (defaults to current project)"
            }
        },
        "required": []
    }
}


async def execute(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute schema backfill operations

    Simple approach: Just run UPDATE queries to add new fields.
    Neo4j handles the transaction internally.
    """
    from servers.services.service_container import ServiceContainer
    from servers.services.project_context_manager import get_project_context_manager

    try:
        # Get project context
        manager = await get_project_context_manager()
        project = arguments.get("project") or manager.current_project

        if not project:
            return {
                "error": "No project context",
                "message": "Please set project context first using set_project_context"
            }

        # Initialize service container
        container = ServiceContainer(context_manager=manager)
        await container.initialize_all_services()

        operation = arguments.get("operation", "status")
        now = datetime.utcnow().isoformat()

        if operation == "status":
            # Check current state
            query = """
            MATCH (c:Chunk {project: $project})
            WITH
                count(CASE WHEN c.metadata_version IS NOT NULL THEN 1 END) as with_version,
                count(CASE WHEN c.file_type IS NOT NULL THEN 1 END) as with_file_type,
                count(CASE WHEN c.language IS NOT NULL THEN 1 END) as with_language,
                count(CASE WHEN c.is_canonical IS NOT NULL THEN 1 END) as with_canonical,
                count(c) as total
            RETURN with_version, with_file_type, with_language, with_canonical, total
            """

            result = await container.neo4j.execute_cypher(query, {'project': project})

            if result.get('status') == 'success' and result.get('result'):
                stats = result['result'][0]
                return {
                    "operation": "status",
                    "project": project,
                    "total_chunks": stats['total'],
                    "chunks_with_version": stats['with_version'],
                    "chunks_with_file_type": stats['with_file_type'],
                    "chunks_with_language": stats['with_language'],
                    "chunks_with_canonical": stats['with_canonical'],
                    "needs_backfill": stats['total'] - stats['with_version'],
                    "message": f"{stats['with_version']}/{stats['total']} chunks have metadata_version"
                }

        elif operation == "backfill":
            # Step 1: Add metadata_version to all chunks
            logger.info(f"Adding metadata fields to chunks in project {project}")

            init_query = """
            MATCH (c:Chunk {project: $project})
            WHERE c.metadata_version IS NULL
            SET c.metadata_version = '1.0'
            RETURN count(c) as updated
            """

            result = await container.neo4j.execute_cypher(init_query, {'project': project})
            step1_count = 0
            if result.get('status') == 'success' and result.get('result'):
                step1_count = result['result'][0]['updated']
                logger.info(f"Step 1: Initialized {step1_count} chunks to v1.0")

            # Step 2: Add v1.1 fields with defaults
            v11_query = """
            MATCH (c:Chunk {project: $project})
            WHERE c.metadata_version = '1.0'
            SET c.file_modified_at = coalesce(c.file_modified_at, $now),
                c.file_created_at = coalesce(c.file_created_at, $now),
                c.git_commit_sha = coalesce(c.git_commit_sha, 'unknown'),
                c.git_author = coalesce(c.git_author, 'unknown'),
                c.file_type = coalesce(c.file_type,
                    CASE
                        WHEN c.file_path ENDS WITH '.py' THEN 'py'
                        WHEN c.file_path ENDS WITH '.js' THEN 'js'
                        WHEN c.file_path ENDS WITH '.ts' THEN 'ts'
                        WHEN c.file_path ENDS WITH '.tsx' THEN 'tsx'
                        WHEN c.file_path ENDS WITH '.jsx' THEN 'jsx'
                        WHEN c.file_path ENDS WITH '.md' THEN 'md'
                        WHEN c.file_path ENDS WITH '.json' THEN 'json'
                        WHEN c.file_path ENDS WITH '.yaml' THEN 'yaml'
                        WHEN c.file_path ENDS WITH '.yml' THEN 'yml'
                        WHEN c.file_path ENDS WITH '.html' THEN 'html'
                        WHEN c.file_path ENDS WITH '.css' THEN 'css'
                        WHEN c.file_path ENDS WITH '.sh' THEN 'sh'
                        WHEN c.file_path ENDS WITH '.rs' THEN 'rs'
                        WHEN c.file_path ENDS WITH '.go' THEN 'go'
                        WHEN c.file_path ENDS WITH '.java' THEN 'java'
                        WHEN c.file_path ENDS WITH '.cpp' THEN 'cpp'
                        WHEN c.file_path ENDS WITH '.c' THEN 'c'
                        WHEN c.file_path ENDS WITH '.h' THEN 'h'
                        ELSE 'unknown'
                    END
                ),
                c.language = coalesce(c.language,
                    CASE
                        WHEN c.file_type = 'py' THEN 'python'
                        WHEN c.file_type IN ['js', 'jsx'] THEN 'javascript'
                        WHEN c.file_type IN ['ts', 'tsx'] THEN 'typescript'
                        WHEN c.file_type IN ['md'] THEN 'markdown'
                        WHEN c.file_type IN ['json', 'yaml', 'yml'] THEN 'config'
                        WHEN c.file_type IN ['html', 'css'] THEN 'web'
                        WHEN c.file_type = 'sh' THEN 'bash'
                        WHEN c.file_type = 'rs' THEN 'rust'
                        WHEN c.file_type = 'go' THEN 'go'
                        WHEN c.file_type = 'java' THEN 'java'
                        WHEN c.file_type IN ['cpp', 'c', 'h'] THEN 'c/cpp'
                        ELSE 'unknown'
                    END
                ),
                c.is_canonical = coalesce(c.is_canonical,
                    NOT (c.file_path CONTAINS 'test'
                        OR c.file_path CONTAINS 'spec'
                        OR c.file_path CONTAINS '__pycache__'
                        OR c.file_path CONTAINS 'node_modules'
                        OR c.file_path CONTAINS '.git')
                ),
                c.metadata_version = '1.1',
                c.backfill_timestamp = datetime()
            RETURN count(c) as updated
            """

            result = await container.neo4j.execute_cypher(v11_query, {'project': project, 'now': now})
            step2_count = 0
            if result.get('status') == 'success' and result.get('result'):
                step2_count = result['result'][0]['updated']
                logger.info(f"Step 2: Upgraded {step2_count} chunks to v1.1")

            return {
                "operation": "backfill",
                "project": project,
                "chunks_initialized": step1_count,
                "chunks_upgraded": step2_count,
                "total_processed": step1_count + step2_count,
                "success": True,
                "message": f"Successfully added metadata fields to {step1_count + step2_count} chunks"
            }

        elif operation == "verify":
            # Verify the backfill worked
            verify_query = """
            MATCH (c:Chunk {project: $project})
            WITH
                count(CASE WHEN c.metadata_version = '1.1' THEN 1 END) as v11_count,
                count(CASE WHEN c.metadata_version = '1.0' THEN 1 END) as v10_count,
                count(CASE WHEN c.metadata_version IS NULL THEN 1 END) as null_count,
                count(c) as total,
                collect(DISTINCT c.file_type)[..10] as sample_types,
                collect(DISTINCT c.language)[..10] as sample_langs
            RETURN v11_count, v10_count, null_count, total, sample_types, sample_langs
            """

            result = await container.neo4j.execute_cypher(verify_query, {'project': project})

            if result.get('status') == 'success' and result.get('result'):
                stats = result['result'][0]

                success = stats['v11_count'] == stats['total']

                return {
                    "operation": "verify",
                    "project": project,
                    "total_chunks": stats['total'],
                    "v1.1_chunks": stats['v11_count'],
                    "v1.0_chunks": stats['v10_count'],
                    "unversioned_chunks": stats['null_count'],
                    "sample_file_types": stats['sample_types'],
                    "sample_languages": stats['sample_langs'],
                    "success": success,
                    "message": f"{'✅ All' if success else '⚠️ Only ' + str(stats['v11_count']) + '/' + str(stats['total'])} chunks at v1.1"
                }

        else:
            return {
                "error": "Invalid operation",
                "message": f"Unknown operation: {operation}. Valid operations: status, backfill, verify"
            }

    except Exception as e:
        logger.error(f"Schema backfill failed: {e}")
        return {
            "error": "Schema backfill failed",
            "message": str(e)
        }