"""
Diagnose Search Issues Tool - ADR-0102
Diagnoses why search is not returning results by checking project mismatches
"""
import json
import logging
from typing import List, Dict, Any
from mcp import types
from ..shared.connection_pool import get_shared_neo4j_service

logger = logging.getLogger(__name__)

# Tool metadata for automatic registration
TOOL_CONFIG = {
    "name": "diagnose_search",
    "description": "Diagnose why search is not returning results. Checks for project mismatches, NULL project fields, and database statistics. Use when elite_search or fast_search return no results.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "verbose": {
                "type": "boolean",
                "description": "Show detailed diagnostics including sample data",
                "default": True
            }
        },
        "required": []
    }
}

async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Diagnose search issues"""
    try:
        verbose = arguments.get("verbose", True)

        # Get project context
        from servers.services.project_context_manager import get_project_context_manager
        context_manager = await get_project_context_manager()
        project_info = await context_manager.get_current_project(force_refresh=True)
        detected_project = project_info.get("project")
        detected_path = project_info.get("path")

        diagnostics = {
            "detected_project": {
                "name": detected_project,
                "path": detected_path,
                "source": project_info.get('source', 'unknown')
            }
        }

        if not detected_project:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "diagnosis": "No project detected",
                    "solution": "Use set_project_context tool to set a project",
                    "details": diagnostics
                }, indent=2)
            )]

        # Get Neo4j service
        neo4j_service = await get_shared_neo4j_service(detected_project)

        # Check project distribution in database
        projects_query = """
        MATCH (n)
        WHERE n.project IS NOT NULL
        RETURN DISTINCT n.project as project, labels(n)[0] as node_type, count(n) as count
        ORDER BY count DESC
        """

        result = await neo4j_service.execute_cypher(projects_query, {})
        project_distribution = []
        projects_found = set()

        if result.get('status') == 'success' and result.get('result'):
            for row in result['result']:
                project = row.get('project')
                projects_found.add(project)
                project_distribution.append({
                    "project": project,
                    "node_type": row.get('node_type'),
                    "count": row.get('count')
                })

        diagnostics["database_projects"] = list(projects_found)
        diagnostics["project_distribution"] = project_distribution if verbose else f"{len(project_distribution)} entries"

        # Check for NULL project nodes
        null_query = """
        MATCH (n)
        WHERE n.project IS NULL OR n.project = ''
        RETURN labels(n)[0] as node_type, count(n) as count
        ORDER BY count DESC
        """

        null_result = await neo4j_service.execute_cypher(null_query, {})
        null_distribution = []
        total_null = 0

        if null_result.get('status') == 'success' and null_result.get('result'):
            for row in null_result['result']:
                count = row.get('count', 0)
                total_null += count
                null_distribution.append({
                    "node_type": row.get('node_type'),
                    "count": count
                })

        diagnostics["null_project_nodes"] = {
            "total": total_null,
            "by_type": null_distribution if verbose else f"{len(null_distribution)} types"
        }

        # Check chunk statistics for detected project
        chunk_query = """
        MATCH (c:Chunk)
        RETURN
            count(CASE WHEN c.project = $project THEN 1 END) as matching_project,
            count(CASE WHEN c.project IS NULL THEN 1 END) as null_project,
            count(CASE WHEN c.project <> $project AND c.project IS NOT NULL THEN 1 END) as other_project,
            count(c) as total_chunks
        """

        chunk_result = await neo4j_service.execute_cypher(
            chunk_query,
            {'project': detected_project}
        )

        chunk_stats = {
            "total": 0,
            "matching_project": 0,
            "null_project": 0,
            "other_projects": 0
        }

        if chunk_result.get('status') == 'success' and chunk_result.get('result'):
            stats = chunk_result['result'][0]
            chunk_stats = {
                "total": stats.get('total_chunks', 0),
                "matching_project": stats.get('matching_project', 0),
                "null_project": stats.get('null_project', 0),
                "other_projects": stats.get('other_project', 0)
            }

        diagnostics["chunk_statistics"] = chunk_stats

        # Test actual search
        if verbose:
            test_query = """
            MATCH (c:Chunk {project: $project})
            RETURN c.chunk_id as chunk_id, c.file_path as file_path
            LIMIT 3
            """

            test_result = await neo4j_service.execute_cypher(
                test_query,
                {'project': detected_project}
            )

            sample_chunks = []
            if test_result.get('status') == 'success' and test_result.get('result'):
                sample_chunks = [
                    {"chunk_id": c.get('chunk_id'), "file_path": c.get('file_path')}
                    for c in test_result['result']
                ]

            diagnostics["sample_chunks"] = sample_chunks

        # Determine diagnosis
        diagnosis = "unknown"
        solution = "Unable to determine issue"

        if chunk_stats["matching_project"] == 0 and chunk_stats["null_project"] > 0:
            diagnosis = "All chunks have NULL project field"
            solution = f"Run fix_null_project tool to set project='{detected_project}'"
        elif chunk_stats["matching_project"] == 0 and chunk_stats["other_projects"] > 0:
            diagnosis = f"Chunks belong to different projects (not '{detected_project}')"
            solution = "Reindex the project or check project name"
        elif chunk_stats["matching_project"] > 0:
            diagnosis = f"Found {chunk_stats['matching_project']} chunks for project '{detected_project}'"
            solution = "Search should work. Check if embeddings are present and search query is correct"
        elif chunk_stats["total"] == 0:
            diagnosis = "No chunks in database"
            solution = "Index the project first using reindex tool"

        # Check if project mismatch
        if detected_project not in projects_found and len(projects_found) > 0:
            diagnostics["warning"] = f"Detected project '{detected_project}' not found in database"

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "diagnosis": diagnosis,
                "solution": solution,
                "diagnostics": diagnostics,
                "summary": {
                    "detected_project": detected_project,
                    "chunks_for_project": chunk_stats["matching_project"],
                    "null_chunks": chunk_stats["null_project"],
                    "total_chunks": chunk_stats["total"],
                    "search_should_work": chunk_stats["matching_project"] > 0
                }
            }, indent=2)
        )]

    except Exception as e:
        logger.error(f"Diagnostic error: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Diagnostic failed: {str(e)}"
            }, indent=2)
        )]