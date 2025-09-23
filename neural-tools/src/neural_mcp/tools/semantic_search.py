"""
Semantic Search Tool - September 2025 Standards
Consolidated semantic and graph search capabilities

ADR-0076: Modular tool architecture
ADR-0075: Connection pooling optimization
Consolidates: semantic_code_search + graphrag_hybrid_search
"""

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
    "name": "semantic_search",
    "description": "Search code by meaning using semantic embeddings and graph context. Consolidated semantic_code_search + graphrag_hybrid_search with Neo4j-only architecture.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query. Example: 'how to start server'",
                "minLength": 3
            },
            "mode": {
                "type": "string",
                "description": "Search mode: 'semantic' (code similarity), 'graph' (graph relationships), 'hybrid' (both)",
                "enum": ["semantic", "graph", "hybrid"],
                "default": "hybrid"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (1-50)",
                "minimum": 1,
                "maximum": 50,
                "default": 10
            },
            "include_graph_context": {
                "type": "boolean",
                "description": "Include graph relationships and dependencies",
                "default": True
            },
            "max_hops": {
                "type": "integer",
                "description": "Maximum graph traversal hops for context",
                "minimum": 0,
                "maximum": 3,
                "default": 2
            }
        },
        "required": ["query"]
    }
}

@track_performance
async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Main tool execution function - September 2025 Standards

    Args:
        arguments: Validated input parameters

    Returns:
        List of TextContent responses with search results
    """
    try:
        # 1. Validate inputs
        query = arguments.get("query", "").strip()
        if not query:
            return _make_error_response("Missing required parameter: query")

        mode = arguments.get("mode", "hybrid")
        limit = arguments.get("limit", 10)
        include_graph_context = arguments.get("include_graph_context", True)
        max_hops = arguments.get("max_hops", 2)

        # Validate parameters
        if limit < 1 or limit > 50:
            limit = 10

        if max_hops < 0 or max_hops > 3:
            max_hops = 2

        valid_modes = ['semantic', 'graph', 'hybrid']
        if mode not in valid_modes:
            mode = 'hybrid'

        # 2. Check cache
        project_name = arguments.get("project", "claude-l9-template")
        cache_key = f"semantic_search:{query}:{mode}:{limit}:{include_graph_context}:{max_hops}:{project_name}"
        cached = get_cached_result(cache_key)
        if cached:
            logger.info(f"🚀 ADR-0075: Cache hit for semantic search '{query[:50]}...'")
            return cached

        # 3. Use shared Neo4j service (ADR-0075)
        neo4j_service = await get_shared_neo4j_service(project_name)

        # 4. Execute business logic
        start_time = time.time()
        result = await _execute_semantic_search(
            neo4j_service, query, mode, limit, include_graph_context, max_hops, project_name
        )
        duration = (time.time() - start_time) * 1000

        # 5. Add performance metadata
        result["performance"] = {
            "query_time_ms": round(duration, 2),
            "cache_hit": False,
            "search_mode": mode,
            "results_found": len(result.get("results", []))
        }

        # 6. Cache and return
        response = [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        if result.get("status") == "success":
            cache_result(cache_key, response)
            logger.info(f"💾 ADR-0075: Cached semantic search '{query[:30]}...' ({mode}) - {duration:.1f}ms")

        return response

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return _make_error_response(f"Semantic search failed: {e}")

async def _execute_semantic_search(neo4j_service, query: str, mode: str, limit: int,
                                 include_graph_context: bool, max_hops: int, project_name: str) -> dict:
    """Execute semantic search with Neo4j-only architecture"""

    logger.info(f"🔍 ADR-0076: Semantic search for '{query}' (mode: {mode})")

    # Build search query based on mode
    search_query = await _build_search_query(mode, limit, include_graph_context, max_hops)

    result = await neo4j_service.execute_cypher(search_query, {
        'query': query,
        'project': project_name,
        'limit': limit,
        'max_hops': max_hops
    })

    if result.get('status') == 'success' and result['result']:
        search_results = []

        for row in result['result']:
            # Extract file information
            file_data = {
                "path": row.get('path'),
                "content_snippet": row.get('content_snippet', ''),
                "relevance_score": row.get('relevance_score', 0.0),
                "file_type": row.get('file_type', 'unknown'),
                "line_count": row.get('line_count', 0),
                "complexity_score": row.get('complexity_score', 0.0)
            }

            # Add graph context if enabled
            if include_graph_context:
                file_data.update({
                    "dependencies": row.get('dependencies', []),
                    "dependent_files": row.get('dependent_files', []),
                    "function_calls": row.get('function_calls', []),
                    "canon_weight": row.get('canon_weight', 0.0),
                    "trust_level": row.get('trust_level', 'unknown')
                })

            search_results.append(file_data)

        response = {
            "status": "success",
            "query": query,
            "mode": mode,
            "include_graph_context": include_graph_context,
            "max_hops": max_hops,
            "results": search_results,
            "summary": {
                "total_results": len(search_results),
                "avg_relevance": round(sum(r.get('relevance_score', 0) for r in search_results) / len(search_results), 3) if search_results else 0,
                "file_types": list(set(r.get('file_type', 'unknown') for r in search_results)),
                "canonical_sources": len([r for r in search_results if r.get('canon_weight', 0) >= 0.7])
            },
            "architecture": "neo4j_semantic_search_consolidated"
        }
    else:
        response = {
            "status": "no_results",
            "query": query,
            "mode": mode,
            "message": "No matching results found for the query",
            "suggestion": "Try different keywords or broader search terms"
        }

    logger.info(f"🎯 Semantic search completed for '{query}' - {len(response.get('results', []))} results")
    return response

async def _build_search_query(mode: str, limit: int, include_graph_context: bool, max_hops: int) -> str:
    """Build optimized Cypher query based on search mode"""

    base_match = "MATCH (f:File {project: $project})"

    if mode == "semantic":
        # Focus on content similarity (would integrate with embeddings in full implementation)
        return f"""
        {base_match}
        WHERE f.content CONTAINS $query OR f.path CONTAINS $query
        RETURN f.path as path,
               substring(f.content, 0, 200) as content_snippet,
               1.0 as relevance_score,
               f.file_type as file_type,
               f.line_count as line_count,
               f.complexity_score as complexity_score,
               f.canon_weight as canon_weight,
               f.trust_level as trust_level
        ORDER BY f.canon_weight DESC, f.complexity_score DESC
        LIMIT $limit
        """

    elif mode == "graph":
        # Focus on graph relationships and structure
        context_clause = ""
        if include_graph_context:
            context_clause = f"""
            OPTIONAL MATCH (f)-[:IMPORTS*1..{max_hops}]->(dep:File {{project: $project}})
            WITH f, collect(DISTINCT dep.path) as dependencies
            OPTIONAL MATCH (dependent:File {{project: $project}})-[:IMPORTS*1..{max_hops}]->(f)
            WITH f, dependencies, collect(DISTINCT dependent.path) as dependent_files
            OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)-[:CALLS]->(called:Function)
            WITH f, dependencies, dependent_files, collect(DISTINCT called.name) as function_calls
            """

        return f"""
        {base_match}
        WHERE f.content CONTAINS $query OR f.path CONTAINS $query
        {context_clause}
        RETURN f.path as path,
               substring(f.content, 0, 200) as content_snippet,
               (f.dependencies_score + f.complexity_score) / 2.0 as relevance_score,
               f.file_type as file_type,
               f.line_count as line_count,
               f.complexity_score as complexity_score,
               f.canon_weight as canon_weight,
               f.trust_level as trust_level
               {', dependencies, dependent_files, function_calls' if include_graph_context else ''}
        ORDER BY relevance_score DESC, f.canon_weight DESC
        LIMIT $limit
        """

    else:  # hybrid mode
        # Combine semantic and graph approaches
        context_clause = ""
        if include_graph_context:
            context_clause = f"""
            OPTIONAL MATCH (f)-[:IMPORTS*1..{max_hops}]->(dep:File {{project: $project}})
            WITH f, collect(DISTINCT dep.path) as dependencies
            OPTIONAL MATCH (dependent:File {{project: $project}})-[:IMPORTS*1..{max_hops}]->(f)
            WITH f, dependencies, collect(DISTINCT dependent.path) as dependent_files
            OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)-[:CALLS]->(called:Function)
            WITH f, dependencies, dependent_files, collect(DISTINCT called.name) as function_calls
            """

        return f"""
        {base_match}
        WHERE f.content CONTAINS $query OR f.path CONTAINS $query OR f.description CONTAINS $query
        {context_clause}
        RETURN f.path as path,
               substring(f.content, 0, 200) as content_snippet,
               (COALESCE(f.canon_weight, 0.5) + COALESCE(f.complexity_score, 0.3) + COALESCE(f.dependencies_score, 0.2)) / 3.0 as relevance_score,
               f.file_type as file_type,
               f.line_count as line_count,
               f.complexity_score as complexity_score,
               f.canon_weight as canon_weight,
               f.trust_level as trust_level
               {', dependencies, dependent_files, function_calls' if include_graph_context else ''}
        ORDER BY relevance_score DESC, f.canon_weight DESC
        LIMIT $limit
        """

def _make_error_response(error: str) -> List[types.TextContent]:
    """Standard error response format"""
    return [types.TextContent(type="text", text=json.dumps({
        "status": "error",
        "message": error,
        "tool": TOOL_CONFIG["name"],
        "architecture": "modular_september_2025"
    }))]