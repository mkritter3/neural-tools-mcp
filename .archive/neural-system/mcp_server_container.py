#!/usr/bin/env python3
"""
L9 Neural Memory MCP Server - Container Version
Runs inside Docker container with direct access to Qdrant and neural systems
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Container paths
sys.path.append('/app/neural-system')

from config_manager import get_config
from project_isolation import ProjectIsolation
from neural_embeddings import CodeSpecificEmbedder
from memory_system import MemorySystem
from shared_model_client import SharedModelClient

# MCP imports
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
import mcp.types as types

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class L9NeuralMCPServer:
    """
    L9-grade MCP server running inside Docker container
    Direct access to Qdrant, neural systems, and shared models
    """
    
    def __init__(self):
        self.server = Server("l9-neural-memory")
        self.embedder = None
        self.memory_system = None
        self.model_client = None
        self.project_name = os.environ.get('PROJECT_NAME', 'default')
        
        # Container-specific Qdrant connection
        self.qdrant_host = 'qdrant'  # Docker network name
        self.qdrant_rest_port = 6333
        self.qdrant_grpc_port = 6334
        
        # Register all MCP tools
        self._register_tools()
        
    async def initialize(self):
        """Initialize the MCP server with container resources"""
        try:
            # Initialize embedder
            self.embedder = CodeSpecificEmbedder()
            
            # Initialize shared model client (if model server is available)
            try:
                self.model_client = SharedModelClient(
                    model_server_url="http://model-server:8090"
                )
                logger.info("✅ Connected to shared model server")
            except:
                logger.info("⚠️  Model server not available, using local embedder")
            
            # Initialize memory system with Qdrant
            self.memory_system = MemorySystem(
                project_name=self.project_name,
                qdrant_host=self.qdrant_host,
                qdrant_port=self.qdrant_grpc_port,
                prefer_grpc=True
            )
            await self.memory_system.initialize()
            
            logger.info(f"🚀 L9 Neural MCP Server initialized")
            logger.info(f"📍 Project: {self.project_name}")
            logger.info(f"🔌 Qdrant: {self.qdrant_host}:{self.qdrant_grpc_port}")
            logger.info(f"📂 Data dir: /app/data")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    def _register_tools(self):
        """Register all L9 MCP tools"""
        
        # Memory tools
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="memory_store",
                    description="Store information in neural memory with semantic search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "category": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "metadata": {"type": "object"}
                        },
                        "required": ["content", "category"]
                    }
                ),
                Tool(
                    name="memory_search",
                    description="Search neural memory using semantic similarity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "category": {"type": "string"},
                            "limit": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="memory_update",
                    description="Update existing memory entry",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "metadata": {"type": "object"}
                        },
                        "required": ["id"]
                    }
                ),
                Tool(
                    name="memory_delete",
                    description="Delete memory entry by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"}
                        },
                        "required": ["id"]
                    }
                ),
                Tool(
                    name="code_index",
                    description="Index code files for semantic search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "language": {"type": "string"},
                            "recursive": {"type": "boolean", "default": True}
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="code_search",
                    description="Search indexed code using natural language",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "language": {"type": "string"},
                            "limit": {"type": "integer", "default": 10}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="collection_create",
                    description="Create a new Qdrant collection with custom schema",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "vector_size": {"type": "integer", "default": 768},
                            "distance": {"type": "string", "default": "Cosine"}
                        },
                        "required": ["name"]
                    }
                ),
                Tool(
                    name="collection_list",
                    description="List all collections in current project",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="stats",
                    description="Get memory system statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
            try:
                if name == "memory_store":
                    result = await self.memory_system.store(
                        content=arguments["content"],
                        category=arguments["category"],
                        tags=arguments.get("tags", []),
                        metadata=arguments.get("metadata", {})
                    )
                    return [types.TextContent(type="text", text=json.dumps(result))]
                
                elif name == "memory_search":
                    results = await self.memory_system.search(
                        query=arguments["query"],
                        category=arguments.get("category"),
                        limit=arguments.get("limit", 10)
                    )
                    return [types.TextContent(type="text", text=json.dumps(results))]
                
                elif name == "memory_update":
                    result = await self.memory_system.update(
                        id=arguments["id"],
                        content=arguments.get("content"),
                        metadata=arguments.get("metadata")
                    )
                    return [types.TextContent(type="text", text=json.dumps(result))]
                
                elif name == "memory_delete":
                    result = await self.memory_system.delete(arguments["id"])
                    return [types.TextContent(type="text", text=json.dumps(result))]
                
                elif name == "code_index":
                    result = await self.memory_system.index_code(
                        path=arguments["path"],
                        language=arguments.get("language"),
                        recursive=arguments.get("recursive", True)
                    )
                    return [types.TextContent(type="text", text=json.dumps(result))]
                
                elif name == "code_search":
                    results = await self.memory_system.search_code(
                        query=arguments["query"],
                        language=arguments.get("language"),
                        limit=arguments.get("limit", 10)
                    )
                    return [types.TextContent(type="text", text=json.dumps(results))]
                
                elif name == "collection_create":
                    result = await self.memory_system.create_collection(
                        name=arguments["name"],
                        vector_size=arguments.get("vector_size", 768),
                        distance=arguments.get("distance", "Cosine")
                    )
                    return [types.TextContent(type="text", text=json.dumps(result))]
                
                elif name == "collection_list":
                    collections = await self.memory_system.list_collections()
                    return [types.TextContent(type="text", text=json.dumps(collections))]
                
                elif name == "stats":
                    stats = await self.memory_system.get_stats()
                    return [types.TextContent(type="text", text=json.dumps(stats))]
                
                else:
                    return [types.TextContent(
                        type="text", 
                        text=f"Unknown tool: {name}"
                    )]
                    
            except Exception as e:
                logger.error(f"Tool {name} failed: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]
    
    async def run(self):
        """Run the MCP server"""
        async with self.server.run_async() as running_server:
            # Initialize when server starts
            await running_server.initialize(InitializationOptions())
            await self.initialize()
            
            # Keep running
            logger.info("🎯 L9 MCP Server running...")
            await asyncio.Event().wait()

if __name__ == "__main__":
    server = L9NeuralMCPServer()
    asyncio.run(server.run())