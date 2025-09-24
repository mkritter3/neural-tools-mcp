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
    "description": "ADR-0091: Enhanced dependency analysis with USES/INSTANTIATES relationships",
    "inputSchema": {
        "type": "object",
        "properties": {
            "target_file": {
                "type": "string",
                "description": "File path to analyze dependencies for"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis",
                "enum": ["imports", "dependents", "calls", "uses", "instantiates", "all"],
                "default": "all"
            },
            "relationship_types": {
                "type": "array",
                "description": "Specific relationships to analyze (overrides analysis_type)",
                "items": {
                    "enum": ["IMPORTS", "CALLS", "USES", "INSTANTIATES", "INHERITS", "DEFINED_IN", "HAS_CHUNK"]
                },
                "default": None
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum traversal depth (1-5)",
                "minimum": 1,
                "maximum": 5,
                "default": 3
            },
            "include_metrics": {
                "type": "boolean",
                "description": "Include usage metrics and patterns",
                "default": True
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

        # 4. Get additional parameters for ADR-0091 enhancements
        relationship_types = arguments.get("relationship_types")
        include_metrics = arguments.get("include_metrics", True)

        # 4. Execute business logic
        start_time = time.time()
        result = await _execute_dependency_analysis(
            neo4j_service, target_file, analysis_type, max_depth, project_name,
            relationship_types, include_metrics
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
                                     max_depth: int, project_name: str,
                                     relationship_types: List[str] = None,
                                     include_metrics: bool = True) -> dict:
    """Execute the dependency analysis with optimized queries"""

    logger.info(f"🕸️ ADR-0075: Multi-hop dependency analysis for {target_file}")

    # ADR-0087: Normalize path to relative format for Neo4j compatibility
    if target_file.startswith('/'):
        # Convert absolute to relative by removing common project paths
        base_paths = [
            '/Users/mkr/local-coding/claude-l9-template/',
            '/home/user/projects/',
            '/app/',
            '/workspace/'
        ]
        for base_path in base_paths:
            if target_file.startswith(base_path):
                target_file = target_file[len(base_path):]
                logger.info(f"Normalized path from absolute to relative: {target_file}")
                break

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

        # ADR-0091: Include new USES and INSTANTIATES relationships
        response = {
            "status": "success",
            "target_file": target_file,
            "analysis_type": analysis_type,
            "max_depth": max_depth,
            "dependencies": {
                "imports": analysis_data.get('import_dependencies', []),
                "dependents": analysis_data.get('reverse_dependencies', []),
                "calls": analysis_data.get('call_dependencies', []),
                "uses": analysis_data.get('uses_dependencies', []),
                "instantiates": analysis_data.get('instantiates_dependencies', [])
            },
            "summary": {
                "total_imports": len(analysis_data.get('import_dependencies', [])),
                "total_dependents": len(analysis_data.get('reverse_dependencies', [])),
                "total_calls": len(analysis_data.get('call_dependencies', [])),
                "total_uses": len(analysis_data.get('uses_dependencies', [])),
                "total_instantiates": len(analysis_data.get('instantiates_dependencies', [])),
                "max_import_depth": max([dep.get('path_length', 0) or 0 for dep in analysis_data.get('import_dependencies', [])] + [0]),
                "max_dependent_depth": max([dep.get('path_length', 0) or 0 for dep in analysis_data.get('reverse_dependencies', [])] + [0]),
                "max_call_depth": max([dep.get('call_depth', 0) or 0 for dep in analysis_data.get('call_dependencies', [])] + [0])
            },
            "architecture": "neo4j_multi_hop_analysis_modular"
        }

        # Add metrics if requested and available
        if include_metrics and analysis_type in ["uses", "instantiates", "all"]:
            response["metrics"] = {
                "variable_patterns": analysis_data.get('variable_patterns', []),
                "instantiation_chains": analysis_data.get('instantiation_chains', []),
                "usage_frequency": analysis_data.get('usage_frequency', {})
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
        MATCH (target:File {{project: $project}})
        WHERE target.path = $target_file OR target.path ENDS WITH $target_file
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
        MATCH (target:File {{project: $project}})
        WHERE target.path = $target_file OR target.path ENDS WITH $target_file
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

    elif analysis_type == "uses":
        # ADR-0091: New USES relationship for variable/attribute usage
        return f"""
        MATCH (target:File {{project: $project}})
        WHERE target.path = $target_file OR target.path ENDS WITH $target_file
        OPTIONAL MATCH (target)-[:HAS_FUNCTION]->(func:Function)-[:USES*1..{max_depth}]->(var:Variable)
        WITH target,
             [{{type: 'uses', from_function: func.name, to_variable: var.name, usage_depth: 1}}] as uses_dependencies
        RETURN {{
            import_dependencies: [],
            reverse_dependencies: [],
            call_dependencies: [],
            uses_dependencies: uses_dependencies,
            instantiates_dependencies: []
        }}
        """

    elif analysis_type == "instantiates":
        # ADR-0091: New INSTANTIATES relationship for class instantiation
        return f"""
        MATCH (target:File {{project: $project}})
        WHERE target.path = $target_file OR target.path ENDS WITH $target_file
        OPTIONAL MATCH (target)-[:HAS_FUNCTION]->(func:Function)-[:INSTANTIATES*1..{max_depth}]->(cls:Class)
        WITH target,
             [{{type: 'instantiates', from_function: func.name, to_class: cls.name, instantiation_depth: 1}}] as instantiates_dependencies
        RETURN {{
            import_dependencies: [],
            reverse_dependencies: [],
            call_dependencies: [],
            uses_dependencies: [],
            instantiates_dependencies: instantiates_dependencies
        }}
        """

    elif analysis_type == "calls":
        return f"""
        MATCH (target:File {{project: $project}})
        WHERE target.path = $target_file OR target.path ENDS WITH $target_file
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
        MATCH (target:File {{project: $project}})
        WHERE target.path = $target_file OR target.path ENDS WITH $target_file

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

        // ADR-0091: Include USES relationships
        OPTIONAL MATCH (target)-[:HAS_FUNCTION]->(func2:Function)-[:USES*1..{max_depth}]->(var:Variable)
        WITH target, import_deps, reverse_deps, call_deps,
             [{{type: 'uses', from_function: func2.name, to_variable: var.name, usage_depth: 1}}] as uses_deps

        // ADR-0091: Include INSTANTIATES relationships
        OPTIONAL MATCH (target)-[:HAS_FUNCTION]->(func3:Function)-[:INSTANTIATES*1..{max_depth}]->(cls:Class)
        WITH target, import_deps, reverse_deps, call_deps, uses_deps,
             [{{type: 'instantiates', from_function: func3.name, to_class: cls.name, instantiation_depth: 1}}] as instantiates_deps

        RETURN {{
            import_dependencies: import_deps,
            reverse_dependencies: reverse_deps,
            call_dependencies: call_deps,
            uses_dependencies: uses_deps,
            instantiates_dependencies: instantiates_deps
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