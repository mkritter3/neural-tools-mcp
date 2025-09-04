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
            from services.service_container import ServiceContainer
            from services.hybrid_retriever import HybridRetriever
            
            # Create project-specific container
            container = ServiceContainer(project_name)
            result = await container.initialize_all_services()
            
            # Cache container and retriever
            self.project_containers[project_name] = container
            
            if container.neo4j and container.qdrant:
                self.project_retrievers[project_name] = HybridRetriever(container)
                logger.info(f"✅ GraphRAG initialized for project: {project_name}")
            else:
                logger.warning(f"⚠️ GraphRAG disabled for {project_name} - services unavailable")
                
        return self.project_containers[project_name]
    
    async def get_project_retriever(self, project_name: str):
        """Get HybridRetriever for specific project"""
        await self.get_project_container(project_name)  # Ensure initialized
        return self.project_retrievers.get(project_name)
        
service_state = MultiProjectServiceState()

# Initialize MCP Server
server = Server("l9-neural-enhanced")

async def initialize_services():
    """Initialize default project services for backward compatibility"""
    if service_state.global_initialized:
        return True
        
    try:
        logger.info("🚀 Initializing L9 Multi-Project Neural Services")
        
        # Initialize default project
        await service_state.get_project_container(DEFAULT_PROJECT_NAME)
        service_state.global_initialized = True
        
        logger.info("✅ Multi-project services initialized")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

async def get_project_context(arguments: dict) -> tuple:
    """Extract project context from tool arguments"""
    # Try to detect project from various argument fields
    project_name = DEFAULT_PROJECT_NAME
    
    # Check common argument fields for file paths
    for field in ['file_path', 'query', 'cypher_query', 'path']:
        if field in arguments and arguments[field]:
            detected = service_state.detect_project_from_path(str(arguments[field]))
            if detected != DEFAULT_PROJECT_NAME:
                project_name = detected
                break
    
    # Get project-specific container and retriever
    container = await service_state.get_project_container(project_name)
    retriever = await service_state.get_project_retriever(project_name)
    
    return project_name, container, retriever

# MCP Protocol Handlers

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available neural tools"""
    # Ensure services are initialized
    if not service_state.initialized:
        await initialize_services()
    
    return [
        types.Tool(
            name="neural_system_status",
            description="Get comprehensive neural system status and health",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="neo4j_graph_query",
            description="Execute Cypher query against the Neo4j graph database",
            inputSchema={
                "type": "object",
                "properties": {
                    "cypher_query": {
                        "type": "string",
                        "description": "Cypher query to execute"
                    },
                    "parameters": {
                        "type": "string",
                        "description": "JSON string of parameters for the query",
                        "default": "{}"
                    }
                },
                "required": ["cypher_query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="semantic_code_search",
            description="Search code by meaning using semantic embeddings",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 10
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="project_understanding",
            description="Get condensed project understanding without reading all files",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "description": "Analysis scope: full, architecture, dependencies, core_logic",
                        "default": "full"
                    }
                },
                "additionalProperties": False
            }
        ),
        # GraphRAG Tools
        types.Tool(
            name="graphrag_hybrid_search",
            description="GraphRAG: Find semantically similar code with graph context (Neo4j + Qdrant)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query or code snippet"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    },
                    "include_graph_context": {
                        "type": "boolean",
                        "description": "Include graph relationships from Neo4j",
                        "default": True
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum graph traversal depth",
                        "default": 2
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="graphrag_impact_analysis",
            description="GraphRAG: Analyze the impact of changing a file using graph relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file being changed"
                    },
                    "change_type": {
                        "type": "string",
                        "description": "Type of change: modify, delete, or refactor",
                        "default": "modify"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="graphrag_find_dependencies",
            description="GraphRAG: Find file dependencies through import relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File to analyze"
                    },
                    "direction": {
                        "type": "string",
                        "description": "Direction: imports, imported_by, or both",
                        "default": "both"
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="graphrag_find_related",
            description="GraphRAG: Find related code by graph traversal",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Starting file path"
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relationship types to follow",
                        "default": ["IMPORTS", "CALLS", "PART_OF"]
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum traversal depth",
                        "default": 3
                    }
                },
                "required": ["file_path"],
                "additionalProperties": False
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls"""
    # Ensure services are initialized
    if not service_state.initialized:
        await initialize_services()
    
    try:
        if name == "neural_system_status":
            return await neural_system_status_impl()
        elif name == "neo4j_graph_query":
            return await neo4j_graph_query_impl(
                arguments.get("cypher_query", ""),
                arguments.get("parameters", "{}")
            )
        elif name == "semantic_code_search":
            return await semantic_code_search_impl(
                arguments.get("query", ""),
                arguments.get("limit", 10)
            )
        elif name == "project_understanding":
            return await project_understanding_impl(
                arguments.get("scope", "full")
            )
        # GraphRAG tools
        elif name == "graphrag_hybrid_search":
            return await graphrag_hybrid_search_impl(
                arguments.get("query", ""),
                arguments.get("limit", 5),
                arguments.get("include_graph_context", True),
                arguments.get("max_hops", 2)
            )
        elif name == "graphrag_impact_analysis":
            return await graphrag_impact_analysis_impl(
                arguments.get("file_path", ""),
                arguments.get("change_type", "modify")
            )
        elif name == "graphrag_find_dependencies":
            return await graphrag_find_dependencies_impl(
                arguments.get("file_path", ""),
                arguments.get("direction", "both")
            )
        elif name == "graphrag_find_related":
            return await graphrag_find_related_impl(
                arguments.get("file_path", ""),
                arguments.get("relationship_types", ["IMPORTS", "CALLS", "PART_OF"]),
                arguments.get("max_depth", 3)
            )
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

# Tool implementations

async def neural_system_status_impl() -> List[types.TextContent]:
    """Get comprehensive neural system status"""
    try:
        # Get status for all initialized projects
        project_status = {}
        overall_healthy = 0
        overall_total = 0
        
        if service_state.project_containers:
            for project_name, container in service_state.project_containers.items():
                services_status = {
                    "qdrant": container.qdrant is not None,
                    "neo4j": container.neo4j is not None,
                    "nomic": container.nomic is not None,
                    "graphrag": project_name in service_state.project_retrievers
                }
                
                project_status[project_name] = {
                    "status": "healthy" if all(services_status.values()) else "degraded", 
                    "services": services_status,
                    "healthy_services": sum(services_status.values()),
                    "total_services": len(services_status)
                }
                
                overall_healthy += sum(services_status.values())
                overall_total += len(services_status)
        else:
            # No projects initialized yet
            project_status["none"] = {"status": "not_initialized"}
        
        result = {
            "status": "healthy" if overall_healthy == overall_total and overall_total > 0 else "degraded",
            "mode": "multi-project",
            "default_project": DEFAULT_PROJECT_NAME,
            "version": "l9-neural-enhanced-multi-project",
            "protocol": "2025-06-18",
            "transport": "stdio", 
            "timestamp": datetime.utcnow().isoformat(),
            "projects": project_status,
            "total_projects": len(service_state.project_containers),
            "healthy_services": overall_healthy,
            "total_services": overall_total
        }
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"neural_system_status error: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "error": str(e)
        }))]

async def neo4j_graph_query_impl(cypher_query: str, parameters: str = "{}") -> List[types.TextContent]:
    """Execute Cypher query against Neo4j"""
    try:
        if not service_state.neo4j_client:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": "Neo4j client not initialized"
            }))]
        
        # Parse parameters
        try:
            query_params = json.loads(parameters) if parameters else {}
        except json.JSONDecodeError as e:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": f"Invalid JSON parameters: {str(e)}"
            }))]
        
        result = await service_state.neo4j_client.execute_cypher(cypher_query, query_params)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"neo4j_graph_query error: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e)
        }))]

async def semantic_code_search_impl(query: str, limit: int = 10) -> List[types.TextContent]:
    """Search code semantically using embeddings"""
    try:
        # Get project context from query
        arguments = {"query": query}
        project_name, container, retriever = await get_project_context(arguments)
        
        if not container.nomic or not container.qdrant:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": f"Required services not initialized for project '{project_name}'"
            }))]
        
        logger.info(f"🔍 Semantic search in project '{project_name}': {query}")
        
        # Generate embedding for query
        embeddings = await container.nomic.get_embeddings([query])
        query_vector = embeddings[0]
        
        collection_name = f"project_{project_name}_code"
        
        # Search vectors
        search_results = await container.qdrant.search_vectors(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.7
        )
        
        # Format results
        formatted_results = []
        for hit in search_results:
            content = hit.get("content", "")
            snippet = content[:200] + "..." if len(content) > 200 else content
            
            formatted_results.append({
                "score": hit.get("score", 1.0),
                "file_path": hit.get("file_path", "unknown"),
                "type": hit.get("type", "code"),
                "snippet": snippet
            })
        
        result = {
            "status": "success",
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results)
        }
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"semantic_code_search error: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e)
        }))]

async def project_understanding_impl(scope: str = "full") -> List[types.TextContent]:
    """Generate condensed project understanding"""
    try:
        understanding = {
            "project": PROJECT_NAME,
            "timestamp": datetime.now().isoformat(),
            "scope": scope,
            "protocol": "2025-06-18",
            "transport": "stdio"
        }
        
        # Get high-level stats from Qdrant if available
        if service_state.qdrant_client:
            try:
                collections = service_state.qdrant_client.client.get_collections().collections
                collection_prefix = f"project_{PROJECT_NAME}_"
                project_collections = [c.name for c in collections if c.name.startswith(collection_prefix)]
                
                understanding["indexed_categories"] = [
                    c.replace(collection_prefix, "") for c in project_collections
                ]
            except Exception as e:
                logger.warning(f"Qdrant analysis failed: {e}")
        
        result = {
            "status": "success",
            "understanding": understanding
        }
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"Project understanding failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e)
        }))]

async def run():
    """Run the MCP server with proper STDIO transport"""
    logger.info("🚀 Starting L9 Neural MCP Server (STDIO Transport)")
    
    # Initialize services at startup
    await initialize_services()
    
    # Run the server with STDIO transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="l9-neural-enhanced",
                server_version="2.0.0-stdio",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                ),
            ),
        )

# GraphRAG Tool Implementations

async def graphrag_hybrid_search_impl(query: str, limit: int, include_graph_context: bool, max_hops: int) -> List[types.TextContent]:
    """GraphRAG: Hybrid search combining semantic similarity with graph relationships"""
    try:
        # Get project context from query
        arguments = {"query": query}
        project_name, container, retriever = await get_project_context(arguments)
        
        if not retriever:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"GraphRAG not available for project '{project_name}' - Neo4j or Qdrant not initialized"})
            )]
        
        logger.info(f"🔍 GraphRAG hybrid search in project '{project_name}': {query}")
        results = await retriever.find_similar_with_context(
            query=query,
            limit=limit,
            include_graph_context=include_graph_context,
            max_hops=max_hops
        )
        
        # Format results for output
        formatted_results = []
        for result in results:
            formatted = {
                "score": result.get("score", 0),
                "file": result.get("file_path", ""),
                "lines": f"{result.get('start_line', 0)}-{result.get('end_line', 0)}",
                "content": result.get("content", "")[:200] + "..."
            }
            
            if include_graph_context and result.get("graph_context"):
                ctx = result["graph_context"]
                formatted["graph_context"] = {
                    "imports": ctx.get("imports", []),
                    "imported_by": ctx.get("imported_by", []),
                    "related_chunks": len(ctx.get("related_chunks", []))
                }
            
            formatted_results.append(formatted)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "query": query,
                "results_count": len(formatted_results),
                "results": formatted_results
            }, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"GraphRAG hybrid search error: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

async def graphrag_impact_analysis_impl(file_path: str, change_type: str) -> List[types.TextContent]:
    """GraphRAG: Analyze impact of file changes using graph relationships"""
    try:
        # Get project context from file path
        arguments = {"file_path": file_path}
        project_name, container, retriever = await get_project_context(arguments)
        
        if not retriever:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"GraphRAG not available for project '{project_name}'"})
            )]
        
        logger.info(f"📊 GraphRAG impact analysis in project '{project_name}': {file_path}")
        impact = await retriever.analyze_impact(
            file_path=file_path,
            change_type=change_type
        )
        
        return [types.TextContent(
            type="text",
            text=json.dumps(impact, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"GraphRAG impact analysis error: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

async def graphrag_find_dependencies_impl(file_path: str, direction: str) -> List[types.TextContent]:
    """GraphRAG: Find file dependencies through graph relationships"""
    try:
        # Get project context from file path
        arguments = {"file_path": file_path}
        project_name, container, retriever = await get_project_context(arguments)
        
        if not retriever:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"GraphRAG not available for project '{project_name}'"})
            )]
        
        logger.info(f"🔗 GraphRAG find dependencies in project '{project_name}': {file_path} ({direction})")
        deps = await retriever.find_dependencies(
            file_path=file_path,
            direction=direction
        )
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "file": file_path,
                "direction": direction,
                "dependencies": deps
            }, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"GraphRAG dependencies error: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

async def graphrag_find_related_impl(file_path: str, relationship_types: List[str], max_depth: int) -> List[types.TextContent]:
    """GraphRAG: Find related code through graph traversal"""
    try:
        # Get project context from file path
        arguments = {"file_path": file_path}
        project_name, container, retriever = await get_project_context(arguments)
        
        if not retriever:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"GraphRAG not available for project '{project_name}'"})
            )]
        
        logger.info(f"🕸️ GraphRAG find related in project '{project_name}': {file_path}")
        related = await retriever.find_related_by_graph(
            file_path=file_path,
            relationship_types=relationship_types,
            max_depth=max_depth
        )
        
        # Format for output
        result = {
            "file": file_path,
            "relationship_types": relationship_types,
            "max_depth": max_depth,
            "nodes_found": len(related.get("nodes", [])),
            "relationships_found": len(related.get("relationships", [])),
            "nodes": [str(node) for node in related.get("nodes", [])[:10]],  # Limit output
            "note": "Full graph data available through Neo4j query"
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"GraphRAG find related error: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)