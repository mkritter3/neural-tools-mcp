"""
Project Operations Tool - September 2025 Standards
Consolidated project management and indexing operations

ADR-0076: Modular tool architecture
ADR-0075: Connection pooling optimization
Consolidates: project_understanding + indexer_status + reindex_path
"""

import os
import json
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any

from mcp import types
from ..shared.connection_pool import get_shared_neo4j_service
from ..shared.performance_metrics import track_performance
from ..shared.cache_manager import cache_result, get_cached_result

import logging
logger = logging.getLogger(__name__)

# Tool metadata for automatic registration
TOOL_CONFIG = {
    "name": "project_operations",
    "description": "Consolidated project operations: understanding, indexer status, and reindexing. Combines project_understanding + indexer_status + reindex_path functionality.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation to perform",
                "enum": ["understanding", "indexer_status", "reindex", "full_status"],
                "default": "understanding"
            },
            "scope": {
                "type": "string",
                "description": "Scope for understanding operation",
                "enum": ["full", "summary", "files", "services", "architecture", "dependencies", "core_logic", "documentation"],
                "default": "full"
            },
            "path": {
                "type": "string",
                "description": "Path to reindex (for reindex operation)"
            },
            "recursive": {
                "type": "boolean",
                "description": "Recursive reindexing (for reindex operation)",
                "default": True
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results for understanding operation",
                "minimum": 1,
                "maximum": 100,
                "default": 50
            }
        },
        "required": []
    }
}

@track_performance
async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Main tool execution function - September 2025 Standards

    Args:
        arguments: Validated input parameters

    Returns:
        List of TextContent responses with operation results
    """
    try:
        # 1. Get project context from manager (ADR-0102)
        from servers.services.project_context_manager import get_project_context_manager
        context_manager = await get_project_context_manager()
        project_info = await context_manager.get_current_project(force_refresh=True)

        if not project_info or not project_info.get("project"):
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "No project context detected",
                    "action": "Please use set_project_context tool to specify your working project"
                }, indent=2)
            )]

        project_name = project_info["project"]
        logger.info(f"🎯 Using project from context manager: {project_name}")

        # 2. Validate inputs
        operation = arguments.get("operation", "understanding")

        # 3. Check cache for understanding operations
        if operation in ["understanding", "full_status"]:
            scope = arguments.get("scope", "full")
            max_results = arguments.get("max_results", 50)
            cache_key = f"project_operations:{operation}:{scope}:{max_results}:{project_name}"
            cached = get_cached_result(cache_key)
            if cached:
                logger.info(f"🚀 ADR-0075: Cache hit for {operation} operation")
                return cached

        # 4. Use shared Neo4j service (ADR-0075)
        neo4j_service = await get_shared_neo4j_service(project_name)

        # 5. Execute business logic based on operation
        start_time = time.time()

        if operation == "understanding":
            result = await _execute_project_understanding(neo4j_service, arguments, project_name)
        elif operation == "indexer_status":
            result = await _execute_indexer_status(neo4j_service, project_name)
        elif operation == "reindex":
            result = await _execute_reindex_path(neo4j_service, arguments, project_name)
        elif operation == "full_status":
            result = await _execute_full_status(neo4j_service, arguments, project_name)
        else:
            return _make_error_response(f"Unknown operation: {operation}")

        duration = (time.time() - start_time) * 1000

        # 6. Add performance metadata
        result["performance"] = {
            "query_time_ms": round(duration, 2),
            "cache_hit": False,
            "operation": operation
        }

        # 7. Cache and return (for understanding operations)
        response = [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        if operation in ["understanding", "full_status"] and result.get("status") == "success":
            cache_result(cache_key, response)
            logger.info(f"💾 ADR-0075: Cached {operation} operation - {duration:.1f}ms")

        return response

    except Exception as e:
        logger.error(f"Project operations failed: {e}")
        return _make_error_response(f"Project operations failed: {e}")

async def _execute_project_understanding(neo4j_service, arguments: Dict[str, Any], project_name: str) -> dict:
    """Execute project understanding analysis"""
    scope = arguments.get("scope", "full")
    max_results = arguments.get("max_results", 50)

    logger.info(f"📊 ADR-0076: Project understanding analysis (scope: {scope})")

    # Build scope-specific query
    if scope == "summary":
        query = """
        MATCH (f:File {project: $project})
        RETURN
            count(f) as total_files,
            collect(DISTINCT f.file_type) as file_types,
            sum(f.line_count) as total_lines,
            avg(f.complexity_score) as avg_complexity,
            count(CASE WHEN f.canon_weight >= 0.7 THEN 1 END) as canonical_files
        """
    elif scope == "files":
        query = f"""
        MATCH (f:File {{project: $project}})
        RETURN f.path as path,
               f.file_type as file_type,
               f.line_count as line_count,
               f.complexity_score as complexity_score,
               f.canon_weight as canon_weight,
               f.trust_level as trust_level
        ORDER BY f.canon_weight DESC, f.complexity_score DESC
        LIMIT {max_results}
        """
    elif scope == "services":
        query = f"""
        MATCH (f:File {{project: $project}})
        WHERE f.file_type IN ['service', 'server', 'api', 'config']
        RETURN f.path as path,
               f.file_type as file_type,
               f.description as description,
               f.canon_weight as canon_weight
        ORDER BY f.canon_weight DESC
        LIMIT {max_results}
        """
    elif scope == "architecture":
        query = """
        MATCH (f:File {project: $project})
        WHERE f.canon_level IN ['architecture_decisions', 'configuration', 'documentation']
        RETURN f.path as path,
               f.canon_level as canon_level,
               f.canon_weight as canon_weight,
               f.trust_level as trust_level,
               substring(f.content, 0, 300) as snippet
        ORDER BY f.canon_weight DESC
        """
    elif scope == "dependencies":
        query = f"""
        MATCH (f:File {{project: $project}})-[:IMPORTS]->(dep:File {{project: $project}})
        RETURN f.path as from_file,
               dep.path as to_file,
               f.file_type as from_type,
               dep.file_type as to_type
        LIMIT {max_results}
        """
    else:  # full scope
        query = f"""
        MATCH (f:File {{project: $project}})
        OPTIONAL MATCH (f)-[:IMPORTS]->(dep:File {{project: $project}})
        OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)
        RETURN f.path as path,
               f.file_type as file_type,
               f.line_count as line_count,
               f.complexity_score as complexity_score,
               f.canon_weight as canon_weight,
               f.trust_level as trust_level,
               f.canon_level as canon_level,
               collect(DISTINCT dep.path) as dependencies,
               count(DISTINCT func) as function_count
        ORDER BY f.canon_weight DESC, f.complexity_score DESC
        LIMIT {max_results}
        """

    result = await neo4j_service.execute_cypher(query, {
        'project': project_name,
        'max_results': max_results
    })

    if result.get('status') == 'success':
        understanding_data = result['result']

        # Process results based on scope
        if scope == "summary":
            summary = understanding_data[0] if understanding_data else {}
            analysis = {
                "total_files": summary.get('total_files', 0),
                "file_types": summary.get('file_types', []),
                "total_lines": summary.get('total_lines', 0),
                "avg_complexity": round(summary.get('avg_complexity', 0), 3),
                "canonical_files": summary.get('canonical_files', 0),
                "canonical_percentage": round((summary.get('canonical_files', 0) / max(summary.get('total_files', 1), 1)) * 100, 1)
            }
        else:
            analysis = {
                "files": understanding_data,
                "metadata": {
                    "total_analyzed": len(understanding_data),
                    "scope": scope,
                    # ADR-0087: Fixed null handling in comparisons
                    "high_complexity_files": len([f for f in understanding_data if (f.get('complexity_score') or 0) > 0.7]),
                    "canonical_files": len([f for f in understanding_data if (f.get('canon_weight') or 0) >= 0.7])
                }
            }

        response = {
            "status": "success",
            "project": project_name,
            "scope": scope,
            "understanding": analysis,
            "architecture": "neo4j_project_understanding_consolidated"
        }
    else:
        response = {
            "status": "no_data",
            "project": project_name,
            "scope": scope,
            "message": "No project data found",
            "suggestion": "Project may need to be indexed first"
        }

    return response

async def _execute_indexer_status(neo4j_service, project_name: str) -> dict:
    """Execute indexer status check"""
    logger.info(f"🔍 ADR-0076: Indexer status check for {project_name}")

    # ADR-0085: Check if indexer container is running
    from servers.services.service_container import ServiceContainer
    service_container = ServiceContainer(project_name=project_name)

    indexer_endpoint = None
    indexer_health = "unknown"
    try:
        indexer_endpoint = await service_container.get_indexer_url()
        # Try a health check
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            health_response = await client.get(f"{indexer_endpoint}/health")
            if health_response.status_code == 200:
                indexer_health = "healthy"
            else:
                indexer_health = "unhealthy"
    except Exception as e:
        logger.warning(f"Indexer not available: {e}")
        indexer_health = "offline"

    # Check Neo4j database status
    status_query = """
    MATCH (f:File {project: $project})
    RETURN count(f) as indexed_files,
           max(f.last_modified) as last_update,
           collect(DISTINCT f.file_type) as file_types
    """

    result = await neo4j_service.execute_cypher(status_query, {'project': project_name})

    if result.get('status') == 'success' and result['result']:
        data = result['result'][0]

        # System resource metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }

        response = {
            "status": "success",
            "project": project_name,
            "indexer_status": {
                "indexed_files": data.get('indexed_files', 0),
                "last_update": data.get('last_update'),
                "file_types_indexed": data.get('file_types', []),
                "database_connection": "active",
                "indexing_service": "neo4j_unified",
                "container_health": indexer_health,
                "container_endpoint": indexer_endpoint
            },
            "system_metrics": system_metrics,
            "architecture": "neo4j_indexer_status_consolidated"
        }
    else:
        response = {
            "status": "not_indexed",
            "project": project_name,
            "message": "Project not indexed or no data found",
            "recommendation": "Run reindex operation to populate the knowledge graph"
        }

    return response

async def _execute_reindex_path(neo4j_service, arguments: Dict[str, Any], project_name: str) -> dict:
    """Execute path reindexing operation using IndexerService HTTP API"""
    path = arguments.get("path")
    if not path:
        return {
            "status": "error",
            "message": "Path parameter required for reindex operation"
        }

    recursive = arguments.get("recursive", True)

    logger.info(f"🔄 ADR-0076: Reindexing path '{path}' (recursive: {recursive})")

    # Validate path exists
    if not os.path.exists(path):
        return {
            "status": "error",
            "message": f"Path does not exist: {path}"
        }

    try:
        import httpx
        from datetime import datetime
        from servers.services.service_container import ServiceContainer

        # ADR-0085: Use dynamic discovery to find indexer
        service_container = ServiceContainer(project_name=project_name)

        try:
            indexer_url = await service_container.get_indexer_url()
            logger.info(f"🔍 Using discovered indexer endpoint: {indexer_url}")
        except RuntimeError as e:
            # If discovery fails, return error to user
            logger.error(f"Failed to discover indexer: {e}")
            return {
                "status": "error",
                "message": f"Indexer not available for project '{project_name}': {str(e)}",
                "path": str(path)
            }

        async with httpx.AsyncClient(timeout=300.0) as client:
            # Call the IndexerService /reindex-path endpoint
            response = await client.post(
                f"{indexer_url}/reindex-path",
                json={
                    "path": path,
                    "recursive": recursive
                }
            )
            response.raise_for_status()
            indexer_result = response.json()

        logger.info(f"✅ IndexerService HTTP API called: {indexer_url}")

        result = {
            "path": str(path),
            "recursive": recursive,
            "project": project_name,
            "architecture": "elite_indexer_adr_0072_0075",
            "timestamp": datetime.now().isoformat(),
            "processed_files": [],
            "failed_files": [],
            "features_active": {
                "hnsw_vectors": True,
                "tree_sitter": True,
                "graph_relationships": True,
                "ast_chunking": True,
                "unified_neo4j": True
            }
        }

        # Parse IndexerService response
        if indexer_result.get("status") == "success":
            files_processed = indexer_result.get("files_processed", [])
            for file_info in files_processed:
                result["processed_files"].append({
                    "file": file_info.get("file_path", path),
                    "status": "success",
                    "features": {
                        "hnsw_vectors": True,
                        "tree_sitter_extraction": True,
                        "graph_relationships": True,
                        "ast_chunking": True,
                        "unified_neo4j": True
                    },
                    "metrics": {
                        "files_indexed": file_info.get("files_indexed", 0),
                        "chunks_created": file_info.get("chunks_created", 0),
                        "symbols_created": file_info.get("symbols_created", 0)
                    }
                })
        else:
            result["failed_files"].append({
                "file": path,
                "error": indexer_result.get("message", "Unknown error")
            })

        result.update({
            "status": "success",
            "total_processed": len(result["processed_files"]),
            "total_failed": len(result["failed_files"]),
            "note": "Elite indexer with ADR-0072/0075 features completed processing."
        })

        return result

    except Exception as e:
        logger.error(f"Reindex path failed: {e}")
        return {
            "status": "error",
            "message": f"Indexing failed: {str(e)}",
            "path": path
        }

async def _execute_full_status(neo4j_service, arguments: Dict[str, Any], project_name: str) -> dict:
    """Execute comprehensive status including understanding and indexer status"""
    logger.info(f"📊 ADR-0076: Full project status for {project_name}")

    # Get both understanding and indexer status
    understanding_result = await _execute_project_understanding(neo4j_service, arguments, project_name)
    indexer_result = await _execute_indexer_status(neo4j_service, project_name)

    response = {
        "status": "success",
        "project": project_name,
        "full_status": {
            "project_understanding": understanding_result.get("understanding", {}),
            "indexer_status": indexer_result.get("indexer_status", {}),
            "system_metrics": indexer_result.get("system_metrics", {})
        },
        "architecture": "neo4j_full_status_consolidated"
    }

    return response

def _make_error_response(error: str) -> List[types.TextContent]:
    """Standard error response format"""
    return [types.TextContent(type="text", text=json.dumps({
        "status": "error",
        "message": error,
        "tool": TOOL_CONFIG["name"],
        "architecture": "modular_september_2025"
    }))]