"""
Elite Search Tool - ADR-0091 Implementation
Advanced GraphRAG search with DRIFT-inspired graph fan-out and rich context

September 2025 Standards:
- Neo4j 2025.08.0 with HNSW optimization (M=24, ef=150, int8 quantization)
- Nomic Embed v2 with task prefixes
- Graph fan-out with USES, INSTANTIATES relationships
- Community context (when available)
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
    "name": "elite_search",
    "description": "Elite GraphRAG search with graph fan-out for maximum context. Uses DRIFT-inspired traversal with USES/INSTANTIATES relationships.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
                "minLength": 3,
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum graph traversal depth (1-3 hops)",
                "minimum": 1,
                "maximum": 3,
                "default": 2,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-20)",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
            },
            "vector_weight": {
                "type": "number",
                "description": "Weight for vector similarity vs graph context (0.0-1.0)",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.7,
            },
            "include_explanation": {
                "type": "boolean",
                "description": "Include explanation of why results are relevant",
                "default": True,
            },
        },
        "required": ["query"],
    },
}


@track_performance
async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Execute elite GraphRAG search with graph fan-out

    ADR-0091: Always uses hybrid_search_with_fanout for maximum context
    """
    try:
        # 1. Validate inputs
        query = arguments.get("query", "").strip()
        if not query:
            return _make_error_response("Missing required parameter: query")

        max_depth = min(max(arguments.get("max_depth", 2), 1), 3)
        limit = min(max(arguments.get("limit", 10), 1), 20)
        vector_weight = min(max(arguments.get("vector_weight", 0.7), 0.0), 1.0)
        include_explanation = arguments.get("include_explanation", True)

        # 2. Check cache (elite search has shorter TTL due to context richness)
        cache_key = _generate_cache_key(query, max_depth, limit, vector_weight)
        cached_response = get_cached_result(cache_key)
        if cached_response:
            logger.info(f"🚀 Elite search cache hit for '{query[:30]}...'")
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
        embedding_service = await get_shared_embedding_service(project_name)

        # 4. Generate embedding with code-specific task prefix
        start_time = time.time()
        query_embedding = await _generate_query_embedding(
            embedding_service, query, project_name
        )

        if not query_embedding:
            return _make_error_response(
                "Failed to generate embedding. Elite search requires embeddings."
            )

        # 5. Execute elite search with graph fan-out
        results = await neo4j_service.hybrid_search_with_fanout(
            query_text=query,
            query_embedding=query_embedding,
            max_depth=max_depth,
            limit=limit,
            vector_weight=vector_weight
        )

        duration = (time.time() - start_time) * 1000

        # 6. Format results with rich context
        formatted_results = _format_elite_results(
            results, query, include_explanation
        )

        # 7. Add performance metadata
        formatted_results["performance"] = {
            "query_time_ms": round(duration, 2),
            "cache_hit": False,
            "graph_depth": max_depth,
            "vector_weight": vector_weight,
            "results_found": len(results),
            "mode": "elite_graphrag"
        }

        # 8. Cache with shorter TTL (15 minutes for elite search)
        response = [types.TextContent(
            type="text",
            text=json.dumps(formatted_results, indent=2)
        )]
        cache_result(cache_key, response, ttl=900)

        logger.info(
            f"✅ Elite search '{query[:30]}...' - {len(results)} results in {duration:.1f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Elite search failed: {e}")
        return _make_error_response(f"Elite search failed: {str(e)}")


async def _generate_query_embedding(
    embedding_service, query: str, project_name: str
) -> Optional[List[float]]:
    """Generate embedding with code search optimization"""
    try:
        # ADR-0084: Use 'search_query' task type for optimized embeddings
        embedding = await embedding_service.get_embedding(
            query, task_type="search_query"
        )

        if embedding and len(embedding) == 768:
            return embedding
        else:
            logger.warning(f"Invalid embedding: len={len(embedding) if embedding else 0}")
            return None

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def _detect_language(file_path: str) -> str:
    """Detect language from file extension"""
    if not file_path or file_path == "unknown":
        return "unknown"

    ext = file_path.split('.')[-1].lower() if '.' in file_path else ""
    language_map = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "jsx": "javascript",
        "tsx": "typescript",
        "md": "markdown",
        "yaml": "yaml",
        "yml": "yaml",
        "json": "json",
        "toml": "toml",
        "rs": "rust",
        "go": "go",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "h": "c",
        "hpp": "cpp",
        "cs": "csharp",
        "rb": "ruby",
        "php": "php",
        "swift": "swift",
        "kt": "kotlin",
        "scala": "scala",
        "r": "r",
        "sh": "bash",
        "sql": "sql",
        "html": "html",
        "css": "css",
        "scss": "scss",
        "vue": "vue",
        "svelte": "svelte"
    }

    return language_map.get(ext, "unknown")


def _format_elite_results(
    results: List[Dict], query: str, include_explanation: bool
) -> Dict:
    """Format elite search results with rich graph context"""

    if not results:
        return {
            "status": "no_results",
            "query": query,
            "message": "No matching results found",
            "suggestion": "Try broader search terms or reduce graph depth"
        }

    formatted_results = []

    for i, result in enumerate(results):
        chunk = result.get("chunk", {})

        # ADR-0096: Handle both dict and potential list formats
        if not isinstance(chunk, dict):
            logger.warning(f"Unexpected chunk type: {type(chunk)}")
            chunk = {}

        # Extract file_path safely
        file_path = chunk.get("file_path") if chunk else None
        if not file_path and chunk.get("chunk_id"):
            # Try to extract from chunk_id
            file_path = chunk["chunk_id"].split(":")[0]
        if not file_path:
            file_path = "unknown"

        graph_context = result.get("graph_context", [])

        formatted_result = {
            "rank": i + 1,
            "content": chunk.get("content", "")[:500],  # First 500 chars
            "file": {
                "path": file_path,
                "language": _detect_language(file_path)
            },
            "scores": {
                "final": round(result.get("final_score", 0), 3),
                "vector": round(result.get("vector_score", 0), 3)
            },
            "graph_context": {
                "total_connections": len(graph_context) if isinstance(graph_context, list) else 0,
                "related_chunks": len([n for n in graph_context if isinstance(n, dict) and n.get("type") == "Chunk"]) if isinstance(graph_context, list) else 0,
                "related_files": len(set([n.get("file_path") for n in graph_context if isinstance(n, dict) and n.get("file_path")])) if isinstance(graph_context, list) else 0
            }
        }

        # Add explanation if requested
        if include_explanation:
            formatted_result["explanation"] = _generate_explanation(
                result, query
            )

        # Add related entities if present (graph_context is now a list)
        if isinstance(graph_context, list) and graph_context:
            # Extract unique file paths from related nodes
            related_files = list(set([
                n.get("file_path", "")
                for n in graph_context
                if isinstance(n, dict) and n.get("file_path") and n.get("file_path") != file_path
            ]))[:5]

            if related_files:
                formatted_result["related_files"] = related_files

        formatted_results.append(formatted_result)

    return {
        "status": "success",
        "query": query,
        "results": formatted_results,
        "total_results": len(results),
        "search_type": "elite_graphrag",
        "features_used": [
            "vector_similarity",
            "graph_traversal",
            "relationship_analysis",
            "context_enrichment"
        ]
    }


def _generate_explanation(result: Dict, query: str) -> str:
    """Generate explanation for why result is relevant"""

    explanations = []

    vector_score = result.get("vector_score", 0)
    if vector_score > 0.8:
        explanations.append(f"High semantic similarity ({vector_score:.2f})")
    elif vector_score > 0.6:
        explanations.append(f"Good semantic match ({vector_score:.2f})")

    context = result.get("graph_context", [])

    # Handle both list and dict formats for graph_context
    if isinstance(context, list):
        # For list format, count connections
        if len(context) > 5:
            explanations.append(f"Connected to {len(context)} graph nodes")
        elif len(context) > 0:
            explanations.append(f"Has {len(context)} related connections")
    elif isinstance(context, dict):
        # Legacy dict format handling
        if context.get("import_relevance", 0) > 3:
            explanations.append(
                f"Connected to {context['import_relevance']} related imports"
            )

        if context.get("call_depth", 0) > 2:
            explanations.append(
                f"Part of {context['call_depth']}-level call chain"
            )

        if context.get("variable_usage", 0) > 5:
            explanations.append(
                f"Uses {context['variable_usage']} relevant variables"
            )

        if context.get("class_usage", 0) > 1:
            explanations.append(
                f"Instantiates {context['class_usage']} related classes"
            )

    if not explanations:
        explanations.append("Contextually relevant based on graph relationships")

    return " | ".join(explanations)


def _generate_cache_key(
    query: str, max_depth: int, limit: int, vector_weight: float
) -> str:
    """Generate deterministic cache key for elite search"""

    params = {
        "query": query,
        "max_depth": max_depth,
        "limit": limit,
        "vector_weight": vector_weight
    }

    content = json.dumps(params, sort_keys=True)
    hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"l9:prod:neural_tools:elite_search:{hash_value}"


def _make_error_response(message: str) -> List[types.TextContent]:
    """Create standardized error response"""

    error_result = {
        "status": "error",
        "message": message,
        "search_type": "elite_graphrag"
    }

    return [types.TextContent(
        type="text",
        text=json.dumps(error_result, indent=2)
    )]