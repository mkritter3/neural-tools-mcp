"""
Semantic Search Tool - September 2025 Standards
Consolidated semantic and graph search capabilities with vector embeddings

ADR-0079: Vector search integration with graceful text search fallback
ADR-0076: Modular tool architecture
ADR-0075: Connection pooling optimization
Consolidates: semantic_code_search + graphrag_hybrid_search
"""

import hashlib
import json
import time
from typing import List, Dict, Any, Optional

from mcp import types
from ..shared.connection_pool import (
    get_shared_neo4j_service,
    get_shared_embedding_service,
)
from ..shared.performance_metrics import track_performance
from ..shared.cache_manager import cache_result, get_cached_result

import logging

logger = logging.getLogger(__name__)

# Tool metadata for automatic registration
TOOL_CONFIG = {
    "name": "semantic_search",
    "description": "Search code by meaning using vector embeddings + graph context. ADR-0079: Neo4j HNSW vector search with graceful text search fallback.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query. Example: 'how to start server'",
                "minLength": 3,
            },
            "mode": {
                "type": "string",
                "description": "Search mode: 'semantic' (code similarity), 'graph' (graph relationships), 'hybrid' (both)",
                "enum": ["semantic", "graph", "hybrid"],
                "default": "hybrid",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (1-50)",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
            },
            "include_graph_context": {
                "type": "boolean",
                "description": "Include graph relationships and dependencies",
                "default": True,
            },
            "max_hops": {
                "type": "integer",
                "description": "Maximum graph traversal hops for context",
                "minimum": 0,
                "maximum": 3,
                "default": 2,
            },
            "vector_weight": {
                "type": "number",
                "description": "Weight for vector similarity in hybrid search (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.7,
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum similarity score for vector search results (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.7,
            },
        },
        "required": ["query"],
    },
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
        vector_weight = arguments.get("vector_weight", 0.7)
        min_similarity = arguments.get("min_similarity", 0.7)

        # Validate parameters
        if limit < 1 or limit > 50:
            limit = 10

        if max_hops < 0 or max_hops > 3:
            max_hops = 2

        if vector_weight < 0.0 or vector_weight > 1.0:
            vector_weight = 0.7

        if min_similarity < 0.0 or min_similarity > 1.0:
            min_similarity = 0.7

        valid_modes = ["semantic", "graph", "hybrid"]
        if mode not in valid_modes:
            mode = "hybrid"

        # 2. Check cache (ADR-0079: Use SHA256 for consistent cache keys)
        project_name = arguments.get("project", "claude-l9-template")
        cache_params = f"{query}:{mode}:{limit}:{include_graph_context}:{max_hops}:{vector_weight}:{min_similarity}:{project_name}"
        cache_hash = hashlib.sha256(cache_params.encode()).hexdigest()
        cache_key = f"semantic_search:{cache_hash}"
        cached = get_cached_result(cache_key)
        if cached:
            logger.info(f"🚀 ADR-0075: Cache hit for semantic search '{query[:50]}...'")
            return cached

        # 3. Use shared Neo4j service (ADR-0075)
        neo4j_service = await get_shared_neo4j_service(project_name)

        # 4. Execute business logic (ADR-0079: Vector search with fallback)
        start_time = time.time()
        result = await _execute_semantic_search_with_vector_fallback(
            neo4j_service,
            query,
            mode,
            limit,
            include_graph_context,
            max_hops,
            vector_weight,
            min_similarity,
            project_name,
        )
        duration = (time.time() - start_time) * 1000

        # 5. Add performance metadata
        result["performance"] = {
            "query_time_ms": round(duration, 2),
            "cache_hit": False,
            "search_mode": mode,
            "results_found": len(result.get("results", [])),
        }

        # 6. Cache and return
        response = [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        if result.get("status") == "success":
            cache_result(cache_key, response)
            logger.info(
                f"💾 ADR-0075: Cached semantic search '{query[:30]}...' ({mode}) - {duration:.1f}ms"
            )

        return response

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return _make_error_response(f"Semantic search failed: {e}")


async def _execute_semantic_search_with_vector_fallback(
    neo4j_service,
    query: str,
    mode: str,
    limit: int,
    include_graph_context: bool,
    max_hops: int,
    vector_weight: float,
    min_similarity: float,
    project_name: str,
) -> dict:
    """ADR-0079: Execute vector search with graceful fallback to text search"""

    logger.info(f"🔍 ADR-0079: Vector search for '{query}' (mode: {mode})")

    # Try vector search first (new capability)
    try:
        query_embedding = await _generate_query_embedding(query, project_name)
        if query_embedding and len(query_embedding) == 768:
            logger.info(f"🚀 ADR-0079: Using vector search for '{query}'")
            return await _execute_vector_search(
                neo4j_service,
                query,
                query_embedding,
                mode,
                limit,
                include_graph_context,
                max_hops,
                vector_weight,
                min_similarity,
                project_name,
            )
    except (ConnectionError, TimeoutError, ValueError) as e:
        logger.warning(
            f"Vector search failed ({type(e).__name__}), falling back to text search: {e}"
        )
    except Exception as e:
        logger.warning(
            f"Unexpected vector search error, falling back to text search: {e}"
        )

    # Fallback to existing text search (preserve existing functionality)
    logger.info(f"📝 ADR-0079: Using text search fallback for '{query}'")
    return await _execute_text_search(
        neo4j_service, query, mode, limit, include_graph_context, max_hops, project_name
    )


async def _generate_query_embedding(
    query: str, project_name: str
) -> Optional[List[float]]:
    """Generate embedding for search query using existing Nomic service with caching"""
    try:
        # Check cache first (ADR-0079: SHA256 for consistent cache keys)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cache_key = f"l9:prod:neural_tools:embedding:{query_hash}:{project_name}"

        cached_embedding = get_cached_result(cache_key)
        if cached_embedding:
            logger.info(f"🚀 ADR-0079: Cache hit for embedding '{query[:30]}...'")
            return cached_embedding

        # Generate embedding using existing service (ADR-0075 connection pooling)
        embedding_service = await get_shared_embedding_service(project_name)
        embedding = await embedding_service.get_embedding(query)

        if embedding and isinstance(embedding, list) and len(embedding) == 768:
            # Cache for 1 hour
            cache_result(cache_key, embedding, ttl=3600)
            logger.info(f"💾 ADR-0079: Cached embedding for '{query[:30]}...'")
            return embedding
        else:
            logger.warning(
                f"Invalid embedding received: type={type(embedding)}, len={len(embedding) if embedding else 0}"
            )
            return None

    except Exception as e:
        logger.error(f"Failed to generate embedding for query '{query}': {e}")
        return None


async def _execute_vector_search(
    neo4j_service,
    query: str,
    query_embedding: List[float],
    mode: str,
    limit: int,
    include_graph_context: bool,
    max_hops: int,
    vector_weight: float,
    min_similarity: float,
    project_name: str,
) -> dict:
    """Execute vector search using existing Neo4j service methods"""

    if mode == "semantic":
        # Pure vector similarity search (ADR-0072 HNSW indexes)
        vector_results = await neo4j_service.vector_similarity_search(
            query_embedding, node_type="Chunk", limit=limit, min_score=min_similarity
        )
        return _format_vector_results(vector_results, query, mode, "vector_similarity")

    elif mode == "hybrid":
        # Combined vector + text + graph (September 2025 pattern)
        hybrid_results = await neo4j_service.hybrid_search(
            query, query_embedding, limit=limit, vector_weight=vector_weight
        )
        return _format_vector_results(hybrid_results, query, mode, "hybrid_vector_text")

    else:  # graph mode - preserve existing text + graph functionality
        return await _execute_text_search(
            neo4j_service,
            query,
            mode,
            limit,
            include_graph_context,
            max_hops,
            project_name,
        )


def _format_vector_results(
    vector_results: List[Dict], query: str, mode: str, search_type: str
) -> dict:
    """Format vector search results to match expected response format"""
    if not vector_results:
        return {
            "status": "no_results",
            "query": query,
            "mode": mode,
            "message": "No matching results found for the query",
            "suggestion": "Try different keywords or broader search terms",
            "search_type": search_type,
        }

    search_results = []
    for result in vector_results:
        node = result.get("node", {})
        file_data = {
            "path": node.get("file_path") or node.get("path"),
            "content_snippet": node.get("content", "")[:200],
            "relevance_score": result.get("similarity_score", 0.0),
            "file_type": node.get("file_type", "unknown"),
            "line_count": node.get("line_count", 0),
            "complexity_score": node.get("complexity_score", 0.0),
            "canon_weight": node.get("canon_weight", 0.0),
            "trust_level": node.get("trust_level", "unknown"),
            "search_type": search_type,
            "vector_score": result.get("similarity_score", 0.0),
        }
        search_results.append(file_data)

    return {
        "status": "success",
        "query": query,
        "mode": mode,
        "results": search_results,
        "summary": {
            "total_results": len(search_results),
            "avg_relevance": round(
                sum(r.get("relevance_score", 0) for r in search_results)
                / len(search_results),
                3,
            ),
            "search_type": search_type,
            "vector_enabled": True,
        },
        "architecture": "neo4j_vector_search_adr_0079",
    }


async def _execute_text_search(
    neo4j_service,
    query: str,
    mode: str,
    limit: int,
    include_graph_context: bool,
    max_hops: int,
    project_name: str,
) -> dict:
    """Execute text search (preserved existing functionality)"""

    # Build search query based on mode (preserve existing logic)
    search_query = await _build_search_query(
        mode, limit, include_graph_context, max_hops
    )

    result = await neo4j_service.execute_cypher(
        search_query,
        {"query": query, "project": project_name, "limit": limit, "max_hops": max_hops},
    )

    if result.get("status") == "success" and result["result"]:
        search_results = []

        for row in result["result"]:
            # ADR-0079: Fix TypeError with safe numeric handling
            try:
                relevance_score = (
                    float(row.get("relevance_score", 0.0))
                    if row.get("relevance_score") is not None
                    else 0.0
                )
                complexity_score = (
                    float(row.get("complexity_score", 0.0))
                    if row.get("complexity_score") is not None
                    else 0.0
                )
                canon_weight = (
                    float(row.get("canon_weight", 0.0))
                    if row.get("canon_weight") is not None
                    else 0.0
                )
                line_count = (
                    int(row.get("line_count", 0))
                    if row.get("line_count") is not None
                    else 0
                )
            except (TypeError, ValueError) as e:
                logger.warning(f"ADR-0079: Data type conversion error in row data: {e}")
                relevance_score = complexity_score = canon_weight = 0.0
                line_count = 0

            # Extract file information
            file_data = {
                "path": row.get("path"),
                "content_snippet": row.get("content_snippet", ""),
                "relevance_score": relevance_score,
                "file_type": row.get("file_type", "unknown"),
                "line_count": line_count,
                "complexity_score": complexity_score,
                "search_type": "text_search",
            }

            # Add graph context if enabled
            if include_graph_context:
                file_data.update(
                    {
                        "dependencies": row.get("dependencies", []),
                        "dependent_files": row.get("dependent_files", []),
                        "function_calls": row.get("function_calls", []),
                        "canon_weight": canon_weight,
                        "trust_level": row.get("trust_level", "unknown"),
                    }
                )

            search_results.append(file_data)

        # ADR-0079: Safe average calculation to prevent TypeError
        try:
            avg_relevance = (
                round(
                    sum(r.get("relevance_score", 0) for r in search_results)
                    / len(search_results),
                    3,
                )
                if search_results
                else 0
            )
            canonical_sources = len(
                [r for r in search_results if r.get("canon_weight", 0) >= 0.7]
            )
        except (TypeError, ZeroDivisionError) as e:
            logger.warning(f"ADR-0079: Error calculating summary metrics: {e}")
            avg_relevance = 0.0
            canonical_sources = 0

        response = {
            "status": "success",
            "query": query,
            "mode": mode,
            "include_graph_context": include_graph_context,
            "max_hops": max_hops,
            "results": search_results,
            "summary": {
                "total_results": len(search_results),
                "avg_relevance": avg_relevance,
                "file_types": list(
                    set(r.get("file_type", "unknown") for r in search_results)
                ),
                "canonical_sources": canonical_sources,
                "search_type": "text_search",
            },
            "architecture": "neo4j_text_search_adr_0079_fallback",
        }
    else:
        response = {
            "status": "no_results",
            "query": query,
            "mode": mode,
            "message": "No matching results found for the query",
            "suggestion": "Try different keywords or broader search terms",
            "search_type": "text_search",
        }

    logger.info(
        f"🎯 Text search completed for '{query}' - {len(response.get('results', []))} results"
    )
    return response


async def _build_search_query(
    mode: str, limit: int, include_graph_context: bool, max_hops: int
) -> str:
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
    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "status": "error",
                    "message": error,
                    "tool": TOOL_CONFIG["name"],
                    "architecture": "modular_september_2025",
                }
            ),
        )
    ]
