"""
Fix NULL Project Fields Tool - ADR-0102
One-time migration to backfill project field for chunks indexed before proper project handling
"""
import json
import logging
from typing import List, Dict, Any
from mcp import types
from ..shared.connection_pool import get_shared_neo4j_service

logger = logging.getLogger(__name__)

# Tool metadata for automatic registration
TOOL_CONFIG = {
    "name": "fix_null_project",
    "description": "Fix chunks with NULL project field. Backfills project name for chunks indexed before proper project handling was implemented. Use when search returns no results due to NULL project fields.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "project_name": {
                "type": "string",
                "description": "Optional project name to use. If not provided, will use detected project."
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, only shows what would be fixed without making changes",
                "default": True
            },
            "force": {
                "type": "boolean",
                "description": "Skip confirmation and apply fixes immediately",
                "default": False
            }
        },
        "required": []
    }
}

async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Fix chunks with NULL project field"""
    try:
        # Get project context
        from servers.services.project_context_manager import get_project_context_manager
        context_manager = await get_project_context_manager()
        project_info = await context_manager.get_current_project(force_refresh=True)

        # Use provided project name or detected one
        project_name = arguments.get("project_name") or project_info.get("project")
        dry_run = arguments.get("dry_run", True)
        force = arguments.get("force", False)

        if not project_name:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "Could not determine project name. Use set_project_context first or provide project_name parameter."
                }, indent=2)
            )]

        logger.info(f"🎯 Fixing NULL project chunks for: {project_name}")

        # Get Neo4j service
        neo4j_service = await get_shared_neo4j_service(project_name)

        # Count chunks with NULL project
        count_query = """
        MATCH (c:Chunk)
        WHERE c.project IS NULL OR c.project = ''
        RETURN count(c) as null_count
        """

        result = await neo4j_service.execute_cypher(count_query, {})
        if result.get('status') != 'success' or not result.get('result'):
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "message": "No chunks with NULL project found",
                    "null_chunks": 0,
                    "action": "none_needed"
                }, indent=2)
            )]

        null_count = result['result'][0].get('null_count', 0)

        if null_count == 0:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "message": "No chunks need fixing!",
                    "null_chunks": 0,
                    "action": "none_needed"
                }, indent=2)
            )]

        # Get sample of NULL chunks
        sample_query = """
        MATCH (c:Chunk)
        WHERE c.project IS NULL OR c.project = ''
        RETURN c.chunk_id as chunk_id, c.file_path as file_path
        LIMIT 5
        """

        sample_result = await neo4j_service.execute_cypher(sample_query, {})
        sample_chunks = []
        if sample_result.get('status') == 'success' and sample_result.get('result'):
            sample_chunks = [
                {"chunk_id": c.get('chunk_id', 'unknown'), "file_path": c.get('file_path', 'unknown')}
                for c in sample_result['result']
            ]

        if dry_run:
            # Count files with NULL project too
            file_count_query = """
            MATCH (f:File)
            WHERE f.project IS NULL OR f.project = ''
            RETURN count(f) as null_count
            """

            file_result = await neo4j_service.execute_cypher(file_count_query, {})
            file_null_count = 0
            if file_result.get('status') == 'success' and file_result.get('result'):
                file_null_count = file_result['result'][0].get('null_count', 0)

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "dry_run",
                    "message": f"Found {null_count} chunks and {file_null_count} files with NULL project",
                    "project_to_set": project_name,
                    "null_chunks": null_count,
                    "null_files": file_null_count,
                    "sample_chunks": sample_chunks,
                    "action": "Run with dry_run=false to apply fixes"
                }, indent=2)
            )]

        # Apply the fix
        if not force:
            logger.info(f"⚠️  Updating {null_count} chunks with project='{project_name}'")

        # Update chunks
        update_query = """
        MATCH (c:Chunk)
        WHERE c.project IS NULL OR c.project = ''
        SET c.project = $project
        RETURN count(c) as updated_count
        """

        update_result = await neo4j_service.execute_cypher(
            update_query,
            {'project': project_name}
        )

        updated_chunks = 0
        if update_result.get('status') == 'success':
            updated_chunks = update_result['result'][0].get('updated_count', 0)

        # Also fix File nodes
        file_update_query = """
        MATCH (f:File)
        WHERE f.project IS NULL OR f.project = ''
        SET f.project = $project
        RETURN count(f) as updated_count
        """

        file_update_result = await neo4j_service.execute_cypher(
            file_update_query,
            {'project': project_name}
        )

        updated_files = 0
        if file_update_result.get('status') == 'success':
            updated_files = file_update_result['result'][0].get('updated_count', 0)

        # Verify the fix
        verify_query = """
        MATCH (c:Chunk {project: $project})
        RETURN count(c) as project_count
        """

        verify_result = await neo4j_service.execute_cypher(
            verify_query,
            {'project': project_name}
        )

        total_with_project = 0
        if verify_result.get('status') == 'success':
            total_with_project = verify_result['result'][0].get('project_count', 0)

        # Check remaining NULL
        remaining_query = """
        MATCH (c:Chunk)
        WHERE c.project IS NULL OR c.project = ''
        RETURN count(c) as remaining_null
        """

        remaining_result = await neo4j_service.execute_cypher(remaining_query, {})
        remaining_null = 0
        if remaining_result.get('status') == 'success':
            remaining_null = remaining_result['result'][0].get('remaining_null', 0)

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "message": f"Successfully fixed NULL project fields",
                "project_set_to": project_name,
                "chunks_updated": updated_chunks,
                "files_updated": updated_files,
                "total_chunks_with_project": total_with_project,
                "remaining_null_chunks": remaining_null,
                "search_status": "Search should now work correctly!" if total_with_project > 0 else "No chunks found after fix"
            }, indent=2)
        )]

    except Exception as e:
        logger.error(f"Error fixing NULL project chunks: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Failed to fix NULL project chunks: {str(e)}"
            }, indent=2)
        )]