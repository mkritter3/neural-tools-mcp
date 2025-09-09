#!/usr/bin/env python3
"""
L9 Enhanced MCP Server - Proper STDIO Transport Implementation
Features: Neo4j GraphRAG + Nomic Embed v2-MoE + Tree-sitter + Qdrant hybrid search

This implementation follows the MCP specification for STDIO transport:
- Server runs as a long-lived subprocess
- Reads JSON-RPC messages from stdin (newline-delimited)
- Writes JSON-RPC responses to stdout
- Maintains session state throughout its lifetime
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Fix PYTHONPATH conflict - ensure we import real MCP SDK, not our local mcp module
original_sys_path = sys.path.copy()
sys.path = [p for p in sys.path if not p.endswith('/app/src')]

# Official MCP SDK for 2025-06-18 protocol  
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Restore sys.path for other imports
sys.path = original_sys_path

# Configure logging to stderr (NEVER to stdout for STDIO transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Critical: Log to stderr, not stdout
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PROJECT_NAME = os.environ.get('PROJECT_NAME', 'default')

# Multi-project service instances - cached per project
class MultiProjectServiceState:
    """Holds persistent service state with per-project isolation"""
    def __init__(self):
        self.project_containers = {}  # project_name -> ServiceContainer
        self.project_retrievers = {}  # project_name -> HybridRetriever
        self.global_initialized = False
        
    def detect_project_from_path(self, file_path: str) -> str:
        """Extract project name from workspace path"""
        if not file_path:
            return DEFAULT_PROJECT_NAME
            
        # Handle /workspace/project-name/... format
        if file_path.startswith('/workspace/'):
            parts = file_path.strip('/').split('/')
            if len(parts) >= 2 and parts[1] != '':
                return parts[1]  # project name
        
        # Handle query context with file references
        if 'project-' in file_path:
            for part in file_path.split('/'):
                if part.startswith('project-'):
                    return part
                    
        return DEFAULT_PROJECT_NAME
    
    async def get_project_container(self, project_name: str):
        """Get or create ServiceContainer for specific project"""
        if project_name not in self.project_containers:
            logger.info(f"🏗️ Initializing services for project: {project_name}")
            
            # Import services
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from servers.services.service_container import ServiceContainer
            
            container = ServiceContainer(project_name)
            await container.initialize_all_services()
            self.project_containers[project_name] = container
        return self.project_containers[project_name]
    
    async def get_project_retriever(self, project_name: str):
        if project_name not in self.project_retrievers:
            container = await self.get_project_container(project_name)
            if container.neo4j and container.qdrant:
                from servers.services.hybrid_retriever import HybridRetriever
                self.project_retrievers[project_name] = HybridRetriever(container)
            else:
                self.project_retrievers[project_name] = None
        return self.project_retrievers[project_name]


state = MultiProjectServiceState()
server = Server("l9-neural-enhanced")


async def initialize_services():
    # Trigger initialization for default project
    await state.get_project_container(DEFAULT_PROJECT_NAME)
    state.global_initialized = True
    return True


async def get_project_context(arguments: Dict[str, Any]):
    # Derive project name (simple heuristic)
    project_name = arguments.get('project') or DEFAULT_PROJECT_NAME
    container = await state.get_project_container(project_name)
    retriever = await state.get_project_retriever(project_name)
    return project_name, container, retriever


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    if not state.global_initialized:
        await initialize_services()
    return [
        types.Tool(
            name="neural_system_status",
            description="Get comprehensive neural system status and health",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="semantic_code_search",
            description="Search code by meaning using semantic embeddings",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="graphrag_hybrid_search",
            description="Hybrid search with graph context",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 5},
                    "include_graph_context": {"type": "boolean", "default": True},
                    "max_hops": {"type": "integer", "default": 2}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="project_understanding",
            description="Get condensed project understanding",
            inputSchema={"type": "object", "properties": {"scope": {"type": "string", "default": "full"}}, "additionalProperties": False}
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    if not state.global_initialized:
        await initialize_services()

    try:
        if name == "neural_system_status":
            return await neural_system_status_impl()
        elif name == "semantic_code_search":
            return await semantic_code_search_impl(arguments.get("query", ""), arguments.get("limit", 10))
        elif name == "graphrag_hybrid_search":
            return await graphrag_hybrid_search_impl(
                arguments.get("query", ""),
                arguments.get("limit", 5),
                arguments.get("include_graph_context", True),
                arguments.get("max_hops", 2)
            )
        elif name == "project_understanding":
            return await project_understanding_impl(arguments.get("scope", "full"))
        else:
            return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def semantic_code_search_impl(query: str, limit: int) -> List[types.TextContent]:
    try:
        project_name, container, _ = await get_project_context({})
        embeddings = await container.nomic.get_embeddings([query])
        query_vector = embeddings[0]
        # Use Qdrant client wrapper
        collection_name = f"project_{project_name}_code"
        try:
            search_results = await container.qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return await _fallback_neo4j_search(query, limit)

        formatted = []
        for hit in search_results:
            payload = hit.payload if hasattr(hit, 'payload') else hit
            content = payload.get("content", "")
            formatted.append({
                "score": getattr(hit, 'score', payload.get("score", 1.0)),
                "file_path": payload.get("file_path", ""),
                "snippet": (content[:200] + "...") if len(content) > 200 else content
            })

        return [types.TextContent(type="text", text=json.dumps({
            "status": "success",
            "query": query,
            "results": formatted,
            "total_found": len(formatted)
        }, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def _fallback_neo4j_search(query: str, limit: int) -> List[types.TextContent]:
    try:
        project_name, container, _ = await get_project_context({})
        cypher = """
        MATCH (c:CodeChunk)
        WHERE c.content IS NOT NULL AND (c.content CONTAINS $query OR c.file_path CONTAINS $query)
        RETURN c.file_path as file_path, c.content as content LIMIT $limit
        """
        result = await container.neo4j.execute_cypher(cypher, {"query": query, "limit": limit})
        formatted = []
        data = result.get('data') if isinstance(result, dict) else result
        if data:
            for r in data:
                content = r.get("content", "")
                formatted.append({
                    "file_path": r.get("file_path", ""),
                    "snippet": (content[:200] + "...") if len(content) > 200 else content,
                    "fallback": True
                })
        return [types.TextContent(type="text", text=json.dumps({
            "status": "success",
            "message": "Results from Neo4j fallback search (Qdrant unavailable)",
            "query": query,
            "results": formatted,
            "total_found": len(formatted),
            "fallback_mode": True
        }, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def graphrag_hybrid_search_impl(query: str, limit: int, include_graph_context: bool, max_hops: int) -> List[types.TextContent]:
    try:
        project_name, _, retriever = await get_project_context({})
        if not retriever:
            return [types.TextContent(type="text", text=json.dumps({"error": "GraphRAG not available"}))]

        results = await retriever.find_similar_with_context(query, limit, include_graph_context, max_hops)
        formatted_results = []
        for res in results:
            entry = {
                "score": res.get("score", 0),
                "file": res.get("file_path", ""),
                "lines": f"{res.get('start_line', 0)}-{res.get('end_line', 0)}",
                "content": (res.get("content", "")[:200] + "...")
            }
            if include_graph_context and res.get("graph_context"):
                ctx = res["graph_context"]
                entry["graph_context"] = {
                    "imports": ctx.get("imports", []),
                    "imported_by": ctx.get("imported_by", []),
                    "related_chunks": len(ctx.get("related_chunks", []))
                }
            formatted_results.append(entry)
        return [types.TextContent(type="text", text=json.dumps({
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def project_understanding_impl(scope: str = "full") -> List[types.TextContent]:
    try:
        understanding = {
            "project": DEFAULT_PROJECT_NAME,
            "timestamp": datetime.now().isoformat(),
            "scope": scope,
            "protocol": "2025-06-18",
            "transport": "stdio"
        }
        return [types.TextContent(type="text", text=json.dumps({"status": "success", "understanding": understanding}, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def run():
    logger.info("🚀 Starting L9 Neural MCP Server (STDIO Transport)")
    await initialize_services()
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="l9-neural-enhanced",
                server_version="2.0.0-stdio",
                capabilities=server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
            ),
        )

