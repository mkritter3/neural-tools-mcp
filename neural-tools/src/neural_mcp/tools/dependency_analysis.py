"""
Dependency Analysis Tool - September 2025 Standards
Advanced multi-hop dependency analysis for code files

ADR-0075: Connection pooling optimization
ADR-0076: Modular tool architecture
"""

import os
import json
import time
from typing import List, Dict, Any

from mcp import types
from ..shared.connection_pool import get_shared_neo4j_service
from ..shared.performance_metrics import track_performance
from ..shared.cache_manager import cache_result, get_cached_result

import logging
logger = logging.getLogger(__name__)

# Tool metadata for automatic registration
TOOL_CONFIG = {
    "name": "dependency_analysis",
    "description": "ADR-0075: Advanced multi-hop dependency analysis for code files",
    "inputSchema": {
        "type": "object",
        "properties": {
            "target_file": {
                "type": "string",
                "description": "File path to analyze dependencies for"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis ('imports', 'dependents', 'calls', 'all')",
                "enum": ["imports", "dependents", "calls", "all"],
                "default": "all"
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum traversal depth (1-5)",
                "minimum": 1,
                "maximum": 5,
                "default": 3
            }
        },
        "required": ["target_file"]
    }
}

@track_performance
async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Main tool execution function - September 2025 Standards

    Args:
        arguments: Validated input parameters

    Returns:
        List of TextContent responses with dependency analysis
    """
    try:
        # 1. Validate inputs
        target_file = arguments.get("target_file", "").strip()
        if not target_file:
            return _make_error_response("Missing required parameter: target_file")

        analysis_type = arguments.get("analysis_type", "all")
        max_depth = arguments.get("max_depth", 3)

        # Validate parameters
        if max_depth < 1 or max_depth > 5:
            max_depth = 3

        valid_types = ['imports', 'dependents', 'calls', 'all']
        if analysis_type not in valid_types:
            analysis_type = 'all'

        # 2. Check cache
        project_name = arguments.get("project", "claude-l9-template")
        cache_key = f"dependency_analysis:{target_file}:{analysis_type}:{max_depth}:{project_name}"
        cached = get_cached_result(cache_key)
        if cached:
            logger.info(f"🚀 ADR-0075: Cache hit for {target_file} ({analysis_type}, depth {max_depth})")
            return cached

        # 3. Use shared Neo4j service (ADR-0075)
        neo4j_service = await get_shared_neo4j_service(project_name)

        # 4. Execute business logic
        start_time = time.time()
        result = await _execute_dependency_analysis(
            neo4j_service, target_file, analysis_type, max_depth, project_name
        )
        duration = (time.time() - start_time) * 1000

        # 5. Add performance metadata
        result["performance"] = {
            "query_time_ms": round(duration, 2),
            "cache_hit": False,
            "analysis_depth": max_depth
        }

        # 6. Cache and return
        response = [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        if result.get("status") == "success":
            cache_result(cache_key, response)
            logger.info(f"💾 ADR-0075: Cached result for {target_file} ({analysis_type}, depth {max_depth}) - {duration:.1f}ms")

        return response

    except Exception as e:
        logger.error(f"Dependency analysis failed: {e}")
        return _make_error_response(f"Dependency analysis failed: {e}")

async def _execute_dependency_analysis(neo4j_service, target_file: str, analysis_type: str,
                                     max_depth: int, project_name: str) -> dict:
    """Execute the dependency analysis with optimized queries"""

    logger.info(f"🕸️ ADR-0075: Multi-hop dependency analysis for {target_file}")

    # Build optimized analysis-type-specific queries for performance
    analysis_query = await _build_optimized_dependency_query(analysis_type, max_depth)

    result = await neo4j_service.execute_cypher(analysis_query, {
        'target_file': target_file,
        'project': project_name,
        'analysis_type': analysis_type,
        'max_depth': max_depth
    })

    if result.get('status') == 'success' and result['result']:
        analysis_data = result['result'][0]

        response = {
            "status": "success",
            "target_file": target_file,
            "analysis_type": analysis_type,
            "max_depth": max_depth,
            "dependencies": {
                "imports": analysis_data.get('import_dependencies', []),
                "dependents": analysis_data.get('reverse_dependencies', []),
                "calls": analysis_data.get('call_dependencies', [])
            },
            "summary": {
                "total_imports": len(analysis_data.get('import_dependencies', [])),
                "total_dependents": len(analysis_data.get('reverse_dependencies', [])),
                "total_calls": len(analysis_data.get('call_dependencies', [])),
                "max_import_depth": max([dep.get('path_length', 0) or 0 for dep in analysis_data.get('import_dependencies', [])] + [0]),
                "max_dependent_depth": max([dep.get('path_length', 0) or 0 for dep in analysis_data.get('reverse_dependencies', [])] + [0]),
                "max_call_depth": max([dep.get('call_depth', 0) or 0 for dep in analysis_data.get('call_dependencies', [])] + [0])
            },
            "architecture": "neo4j_multi_hop_analysis_modular"
        }
    else:
        response = {
            "status": "no_data",
            "target_file": target_file,
            "message": "No dependency data found for the specified file",
            "suggestion": "File may not be indexed or may not have dependencies"
        }

    logger.info(f"🎯 Dependency analysis completed for {target_file}")
    return response

async def _build_optimized_dependency_query(analysis_type: str, max_depth: int) -> str:
    """Build optimized Cypher query based on analysis type"""

    if analysis_type == "imports":
        return f"""
        MATCH (target:File {{path: $target_file, project: $project}})
        OPTIONAL MATCH path = (target)-[:IMPORTS*1..{max_depth}]->(imported:File {{project: $project}})
        WITH target,
             [rel in relationships(path) | {{
                 from: startNode(rel).path,
                 to: endNode(rel).path,
                 type: 'import',
                 path_length: length(path)
             }}] as import_dependencies
        RETURN {{
            import_dependencies: import_dependencies,
            reverse_dependencies: [],
            call_dependencies: []
        }}
        """

    elif analysis_type == "dependents":
        return f"""
        MATCH (target:File {{path: $target_file, project: $project}})
        OPTIONAL MATCH path = (dependent:File {{project: $project}})-[:IMPORTS*1..{max_depth}]->(target)
        WITH target,
             [rel in relationships(path) | {{
                 from: startNode(rel).path,
                 to: endNode(rel).path,
                 type: 'dependent',
                 path_length: length(path)
             }}] as reverse_dependencies
        RETURN {{
            import_dependencies: [],
            reverse_dependencies: reverse_dependencies,
            call_dependencies: []
        }}
        """

    elif analysis_type == "calls":
        return f"""
        MATCH (target:File {{path: $target_file, project: $project}})
        OPTIONAL MATCH (target)-[:CONTAINS]->(func:Function)-[:CALLS*1..{max_depth}]->(called:Function)<-[:CONTAINS]-(called_file:File {{project: $project}})
        WITH target,
             [{{type: 'call', from_function: func.name, to_function: called.name, call_depth: 1}}] as call_dependencies
        RETURN {{
            import_dependencies: [],
            reverse_dependencies: [],
            call_dependencies: call_dependencies
        }}
        """

    else:  # "all"
        return f"""
        MATCH (target:File {{path: $target_file, project: $project}})

        OPTIONAL MATCH import_path = (target)-[:IMPORTS*1..{max_depth}]->(imported:File {{project: $project}})
        WITH target,
             [rel in relationships(import_path) | {{
                 from: startNode(rel).path,
                 to: endNode(rel).path,
                 type: 'import',
                 path_length: length(import_path)
             }}] as import_deps

        OPTIONAL MATCH dependent_path = (dependent:File {{project: $project}})-[:IMPORTS*1..{max_depth}]->(target)
        WITH target, import_deps,
             [rel in relationships(dependent_path) | {{
                 from: startNode(rel).path,
                 to: endNode(rel).path,
                 type: 'dependent',
                 path_length: length(dependent_path)
             }}] as reverse_deps

        OPTIONAL MATCH (target)-[:CONTAINS]->(func:Function)-[:CALLS*1..{max_depth}]->(called:Function)<-[:CONTAINS]-(called_file:File {{project: $project}})
        WITH target, import_deps, reverse_deps,
             [{{type: 'call', from_function: func.name, to_function: called.name, call_depth: 1}}] as call_deps

        RETURN {{
            import_dependencies: import_deps,
            reverse_dependencies: reverse_deps,
            call_dependencies: call_deps
        }}
        """

def _make_error_response(error: str) -> List[types.TextContent]:
    """Standard error response format"""
    return [types.TextContent(type="text", text=json.dumps({
        "status": "error",
        "message": error,
        "tool": TOOL_CONFIG["name"],
        "architecture": "modular_september_2025"
    }))]