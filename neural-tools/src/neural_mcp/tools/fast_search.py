"""
Fast Search Tool - ADR-0091 Lightweight Search Implementation
Optimized for speed with basic vector similarity and text matching

September 2025 Standards:
- No graph traversal for maximum speed
- Simple vector similarity search
- Longer cache TTL (stable results)
- Clear separation from elite_search
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
    "name": "fast_search",
    "description": "Fast, lightweight code search using vector similarity. No graph traversal for speed.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
                "minLength": 3,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-30)",
                "minimum": 1,
                "maximum": 30,
                "default": 10,
            },
            "min_similarity": {
                "type": "number",
                "description": "Minimum similarity score for results (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.6,  # Lower threshold for broader results
            },
            "search_type": {
                "type": "string",
                "description": "Type of search: vector (similarity) or text (keyword)",
                "enum": ["vector", "text", "both"],
                "default": "vector",
            },
        },
        "required": ["query"],
    },
}


@track_performance
async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Execute fast, lightweight search optimized for speed

    ADR-0091: Simplified search without graph traversal
    Perfect for: autocomplete, quick lookups, IDE integration
    """
    try:
        # 1. Validate inputs
        query = arguments.get("query", "").strip()
        if not query:
            return _make_error_response("Missing required parameter: query")

        limit = min(max(arguments.get("limit", 10), 1), 30)
        min_similarity = min(max(arguments.get("min_similarity", 0.6), 0.0), 1.0)
        search_type = arguments.get("search_type", "vector")

        # 2. Check cache (longer TTL for fast search - 1 hour)
        cache_key = _generate_cache_key(query, limit, min_similarity, search_type)
        cached_response = get_cached_result(cache_key)
        if cached_response:
            logger.info(f"🚀 Fast search cache hit for '{query[:30]}...'")
            result = cached_response[0].text if cached_response else "{}"
            parsed = json.loads(result)
            parsed["performance"]["cache_hit"] = True
            return [types.TextContent(type="text", text=json.dumps(parsed, indent=2))]

        # 3. Get services with proper project detection (ADR-0091)
        # First try from arguments, then detect from context manager
        project_name = arguments.get("project_name")

        if not project_name:
            # Use the same approach as indexer - get from ProjectContextManager
            try:
                from servers.services.project_context_manager import get_project_context_manager
                context_manager = await get_project_context_manager()
                project_info = await context_manager.get_current_project()
                project_name = project_info["project"]
                logger.info(f"🎯 Detected project: {project_name}")
            except Exception as e:
                logger.warning(f"Failed to detect project: {e}, using default")
                project_name = "claude-l9-template"

        neo4j_service = await get_shared_neo4j_service(project_name)

        start_time = time.time()
        results = []

        # 4. Execute appropriate search type
        if search_type in ["vector", "both"]:
            # Generate embedding for vector search
            embedding_service = await get_shared_embedding_service(project_name)
            query_embedding = await _generate_query_embedding(
                embedding_service, query, project_name
            )

            if query_embedding:
                # Pure vector similarity search (no graph)
                vector_results = await neo4j_service.vector_similarity_search(
                    query_embedding,
                    node_type="Chunk",
                    limit=limit,
                    min_score=min_similarity
                )
                results.extend(vector_results)

        if search_type in ["text", "both"]:
            # Simple text search (no graph)
            text_results = await neo4j_service.semantic_search(query, limit=limit)

            # Merge with vector results if both
            if search_type == "both":
                existing_ids = {r.get("node", {}).get("chunk_id") for r in results}
                for text_result in text_results:
                    chunk_id = text_result.get("n", {}).get("chunk_id")
                    if chunk_id not in existing_ids:
                        results.append(text_result)
            else:
                results = text_results

        duration = (time.time() - start_time) * 1000

        # 5. Format simple results (no graph context)
        formatted_results = _format_fast_results(results[:limit], query, search_type)

        # 6. Add performance metadata
        formatted_results["performance"] = {
            "query_time_ms": round(duration, 2),
            "cache_hit": False,
            "search_type": search_type,
            "results_found": len(results),
            "mode": "fast_search"
        }

        # 7. Cache with longer TTL (1 hour for stable fast results)
        response = [types.TextContent(
            type="text",
            text=json.dumps(formatted_results, indent=2)
        )]
        cache_result(cache_key, response, ttl=3600)

        logger.info(
            f"⚡ Fast search '{query[:30]}...' - {len(results)} results in {duration:.1f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Fast search failed: {e}")
        return _make_error_response(f"Fast search failed: {str(e)}")


async def _generate_query_embedding(
    embedding_service, query: str, project_name: str
) -> Optional[List[float]]:
    """Generate embedding for vector search"""
    try:
        # Check embedding cache first
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cache_key = f"l9:prod:neural_tools:embedding:{query_hash}:{project_name}"

        cached_embedding = get_cached_result(cache_key)
        if cached_embedding:
            logger.info(f"🚀 Embedding cache hit for '{query[:30]}...'")
            return cached_embedding

        # Generate new embedding
        embedding = await embedding_service.get_embedding(
            query, task_type="search_query"
        )

        if embedding and len(embedding) == 768:
            # Cache for 1 hour
            cache_result(cache_key, embedding, ttl=3600)
            return embedding
        else:
            logger.warning(f"Invalid embedding: len={len(embedding) if embedding else 0}")
            return None

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def _format_fast_results(
    results: List[Dict], query: str, search_type: str
) -> Dict:
    """Format results for fast search (simple, no graph context)"""

    if not results:
        return {
            "status": "no_results",
            "query": query,
            "message": "No matching results found",
            "suggestion": "Try different keywords or use elite_search for deeper analysis"
        }

    formatted_results = []

    for i, result in enumerate(results):
        # Handle different result formats
        if "node" in result:
            # Vector search result
            node = result["node"]
            score = result.get("similarity_score", 0)
        else:
            # Text search result
            node = result.get("n", {})
            score = 1.0  # Text search doesn't have scores

        formatted_result = {
            "rank": i + 1,
            "content": node.get("content", "")[:300],  # First 300 chars
            "file": {
                "path": node.get("file_path", "unknown"),
                "language": node.get("language", "unknown")
            },
            "score": round(score, 3),
            "chunk_id": node.get("chunk_id", "unknown")
        }

        # Add line numbers if available
        if "start_line" in node:
            formatted_result["location"] = {
                "start_line": node.get("start_line"),
                "end_line": node.get("end_line")
            }

        formatted_results.append(formatted_result)

    return {
        "status": "success",
        "query": query,
        "results": formatted_results,
        "total_results": len(results),
        "search_type": search_type,
        "mode": "fast_search",
        "note": "For deeper analysis with graph context, use elite_search"
    }


def _generate_cache_key(
    query: str, limit: int, min_similarity: float, search_type: str
) -> str:
    """Generate deterministic cache key for fast search"""

    params = {
        "query": query,
        "limit": limit,
        "min_similarity": min_similarity,
        "search_type": search_type
    }

    content = json.dumps(params, sort_keys=True)
    hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"l9:prod:neural_tools:fast_search:{hash_value}"


def _make_error_response(message: str) -> List[types.TextContent]:
    """Create standardized error response"""

    error_result = {
        "status": "error",
        "message": message,
        "mode": "fast_search"
    }

    return [types.TextContent(
        type="text",
        text=json.dumps(error_result, indent=2)
    )]