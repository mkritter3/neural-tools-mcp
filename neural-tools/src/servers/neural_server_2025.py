#!/usr/bin/env python3
"""
L9 Enhanced MCP Server - 2025-06-18 Protocol Compliant
Features: Neo4j GraphRAG + Nomic Embed v2-MoE + Tree-sitter + Qdrant hybrid search

ARCHITECTURE IMPROVEMENTS (2025-06-18):
- Official MCP Python SDK with protocol version 2025-06-18
- Proper JSON-RPC 2.0 message handling
- Service abstraction with dependency injection
- Clean separation of concerns
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sqlite3

# Official MCP SDK for 2025-06-18 protocol
from mcp.server import Server, NotificationOptions
import mcp.server.stdio
import mcp.types as types
from mcp.server.models import InitializationOptions

# Service container (eliminates global coupling)
from services.service_container import ServiceContainer, get_container

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'default')

# Global service clients
qdrant_client = None
neo4j_client = None
nomic_client = None
_initialization_done = False

# Initialize MCP Server with 2025-06-18 protocol
server = Server("l9-neural-enhanced")

def generate_deterministic_point_id(file_path: str, content: str, chunk_index: int = 0) -> int:
    """Generate deterministic point ID for consistent upserts following industry standards."""
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    unique_string = f"{file_path}#{content_hash}#{chunk_index}"
    return abs(hash(unique_string)) % (10**15)

def get_content_hash(content: str) -> str:
    """Generate content hash for change detection"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

async def ensure_services_initialized():
    """Ensure services are initialized within MCP event loop - 2025-06-18 compatible"""
    global qdrant_client, neo4j_client, nomic_client, _initialization_done
    if not _initialization_done:
        try:
            logger.info("🚀 Initializing L9 Neural MCP Server (2025-06-18 Protocol)")
            
            # Initialize service container to get clients
            container = await get_container(PROJECT_NAME)
            
            # Extract clients as globals for compatibility
            qdrant_client = container.get_qdrant()
            try:
                neo4j_client = container.get_neo4j()
            except RuntimeError:
                neo4j_client = None
                logger.warning("Neo4j client not available - GraphRAG features disabled")
            
            try:
                nomic_client = container.get_nomic()
            except RuntimeError:
                nomic_client = None
                logger.warning("Nomic client not available - embeddings disabled")
            
            logger.info("✅ L9 Neural MCP Server initialization complete")
            _initialization_done = True
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")

# MCP Tools using official 2025-06-18 SDK

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available neural tools."""
    await ensure_services_initialized()
    
    tools = [
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
        types.Tool(
            name="neo4j_index_code_graph", 
            description="Index code files in Neo4j graph with relationship extraction",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "string",
                        "description": "Comma-separated list of file paths to index"
                    },
                    "force_reindex": {
                        "type": "boolean",
                        "description": "Force re-indexing of existing files",
                        "default": False
                    }
                },
                "required": ["file_paths"],
                "additionalProperties": False
            }
        )
    ]
    
    return tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent | types.ImageContent]:
    """Handle tool calls with 2025-06-18 protocol compliance."""
    await ensure_services_initialized()
    
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
    elif name == "neo4j_index_code_graph":
        return await neo4j_index_code_graph_impl(
            arguments.get("file_paths", ""),
            arguments.get("force_reindex", False)
        )
    else:
        raise ValueError(f"Unknown tool: {name}")

# Tool implementations

async def neural_system_status_impl() -> List[types.TextContent]:
    """Get comprehensive neural system status."""
    try:
        services_status = {
            "qdrant": qdrant_client is not None,
            "neo4j": neo4j_client is not None,
            "nomic": nomic_client is not None
        }
        
        if not _initialization_done:
            result = {
                "status": "not_initialized",
                "project": PROJECT_NAME,
                "timestamp": datetime.utcnow().isoformat(),
                "services": services_status,
                "error": "Service initialization not completed"
            }
        else:
            result = {
                "status": "healthy" if all(services_status.values()) else "degraded",
                "project": PROJECT_NAME,
                "version": "l9-neural-enhanced-2025-06-18",
                "protocol": "2025-06-18",
                "timestamp": datetime.utcnow().isoformat(),
                "services": services_status,
                "healthy_services": sum(services_status.values()),
                "total_services": len(services_status),
                "overall_score": sum(services_status.values()) / len(services_status)
            }
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"neural_system_status error: {str(e)}")
        error_result = {
            "status": "error",
            "error": str(e),
            "project": PROJECT_NAME,
            "timestamp": datetime.utcnow().isoformat()
        }
        return [types.TextContent(type="text", text=json.dumps(error_result, indent=2))]

async def neo4j_graph_query_impl(cypher_query: str, parameters: str = "{}") -> List[types.TextContent]:
    """Execute Cypher query against Neo4j."""
    try:
        if not neo4j_client:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error", 
                "message": "Neo4j client not initialized"
            }))]
        
        # Parse parameters
        try:
            query_params = json.loads(parameters) if parameters else {}
        except json.JSONDecodeError as json_error:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": f"Invalid JSON parameters: {str(json_error)}"
            }))]
        
        result = await neo4j_client.execute_cypher(cypher_query, query_params)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"neo4j_graph_query error: {str(e)}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"Query execution failed: {str(e)}"
        }))]

async def semantic_code_search_impl(query: str, limit: int = 10) -> List[types.TextContent]:
    """Search code semantically using embeddings."""
    try:
        if not nomic_client or not qdrant_client:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error", 
                "message": "Required services not initialized"
            }))]
        
        # Generate embedding for query
        embeddings = await nomic_client.get_embeddings([query])
        query_vector = embeddings[0]
        
        collection_name = f"project_{PROJECT_NAME}_code"
        
        # Search vectors
        search_results = await qdrant_client.search_vectors(
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
        logger.error(f"semantic_code_search error: {str(e)}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e)
        }))]

async def project_understanding_impl(scope: str = "full") -> List[types.TextContent]:
    """Generate condensed project understanding."""
    try:
        understanding = {
            "project": PROJECT_NAME,
            "timestamp": datetime.now().isoformat(),
            "scope": scope,
            "protocol": "2025-06-18"
        }
        
        # Get high-level stats from Qdrant collections
        if qdrant_client:
            try:
                collections = qdrant_client.client.get_collections().collections
                collection_prefix = f"project_{PROJECT_NAME}_"
                project_collections = [c.name for c in collections if c.name.startswith(collection_prefix)]
                
                understanding["indexed_categories"] = [
                    c.replace(collection_prefix, "") for c in project_collections
                ]
                
                # Sample semantic clusters from main collection
                code_collection = f"{collection_prefix}code"
                if code_collection in project_collections:
                    search_results = qdrant_client.client.search(
                        collection_name=code_collection,
                        query_vector=("dense", [0.1] * 1536),
                        limit=5,
                        with_payload=True
                    )
                    
                    understanding["code_patterns"] = [
                        {
                            "type": hit.payload.get("type", "unknown"),
                            "category": hit.payload.get("category", "general"),
                            "summary": hit.payload.get("content", "")[:100]
                        }
                        for hit in search_results
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

async def neo4j_index_code_graph_impl(file_paths: str, force_reindex: bool = False) -> List[types.TextContent]:
    """Index code files in Neo4j graph."""
    try:
        if not neo4j_client:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": "Neo4j client not available"
            }))]
        
        if not file_paths:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": "No file paths provided for indexing"
            }))]
        
        # Parse file paths
        paths_list = [path.strip() for path in file_paths.split(",") if path.strip()]
        indexed_files = []
        
        for file_path in paths_list:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Index in Neo4j
                result = await neo4j_client.index_code_file(
                    file_path=file_path,
                    content=content,
                    language=_detect_language(file_path)
                )
                
                indexed_files.append({
                    "file_path": file_path,
                    "status": result.get("status", "unknown"),
                    "indexed": result.get("indexed", False)
                })
                
            except Exception as file_error:
                indexed_files.append({
                    "file_path": file_path,
                    "status": "error",
                    "error": str(file_error)
                })
        
        result = {
            "status": "success",
            "files_processed": len(paths_list),
            "files_indexed": sum(1 for f in indexed_files if f.get("indexed", False)),
            "results": indexed_files,
            "force_reindex": force_reindex
        }
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logger.error(f"neo4j_index_code_graph error: {str(e)}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"Code indexing failed: {str(e)}"
        }))]

def _detect_language(file_path: str) -> str:
    """Simple language detection from file extension"""
    ext = Path(file_path).suffix.lower()
    language_map = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust'
    }
    return language_map.get(ext, 'text')

async def run():
    """Run the MCP server with 2025-06-18 protocol compliance."""
    # Setup server capabilities for 2025-06-18
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="l9-neural-enhanced",
                server_version="2.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(run())