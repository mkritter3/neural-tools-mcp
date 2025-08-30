#!/usr/bin/env python3
"""
Neural Flow MCP Server
Provides neural embeddings, memory management, and project indexing via MCP protocol
"""

import asyncio
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add neural-system to path
sys.path.insert(0, str(Path(__file__).parent))

from neural_embeddings import get_neural_system
from feature_flags import get_feature_manager
from neural_dynamic_memory_system import NeuralDynamicMemorySystem
from project_neural_indexer import ProjectNeuralIndexer

# MCP Protocol imports - L9 2025 JSON-RPC 2.0 compliance  
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# L9 optimization: MCP is required for full compliance
MCP_AVAILABLE = True

logger = logging.getLogger(__name__)

class NeuralFlowMCPServer:
    """MCP Server for Neural Flow intelligence system"""
    
    def __init__(self):
        self.server = Server("neural-flow")
        self.neural_system = None
        self.memory_system = None
        self.project_indexer = None
        self.feature_manager = None
        
    async def initialize(self):
        """Initialize all neural systems"""
        try:
            logger.info("🔮 Initializing Neural Flow systems...")
            
            # Initialize core systems
            self.neural_system = get_neural_system()
            self.feature_manager = get_feature_manager()
            self.memory_system = NeuralDynamicMemorySystem()
            
            # Initialize project indexer - L9 always provides code search capability
            project_root = Path.cwd()
            try:
                self.project_indexer = ProjectNeuralIndexer(project_root)
                logger.info(f"🔍 Project indexer initialized for: {project_root}")
            except Exception as e:
                logger.warning(f"⚠️ Project indexer initialization failed: {e}")
                self.project_indexer = None
                
            logger.info("✅ Neural Flow systems initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neural Flow systems: {e}")
            raise
    
    def register_handlers(self):
        """Register MCP request handlers"""
        
        # Resources
        @self.server.list_resources()
        async def list_resources() -> list[types.Resource]:
            """List available neural resources"""
            resources = []
            
            # Memory resources
            resources.append(types.Resource(
                uri="neural://memory/conversations",
                name="Conversation Memory",
                description="Dynamic conversation memory with tiered storage",
                mimeType="application/json"
            ))
            
            # Project resources
            if self.project_indexer:
                resources.append(types.Resource(
                    uri="neural://project/index",
                    name="Project Code Index", 
                    description="AST-analyzed project code with semantic embeddings",
                    mimeType="application/json"
                ))
                
            # Feature flags
            resources.append(types.Resource(
                uri="neural://config/features",
                name="Feature Configuration",
                description="Current feature flags and A/B test settings",
                mimeType="application/json"
            ))
            
            return resources
        
        @self.server.read_resource()
        async def get_resource(uri: str) -> types.ReadResourceResult:
            """Get specific neural resource"""
            
            if uri == "neural://memory/conversations":
                # Return conversation memory stats
                stats = self.memory_system.get_memory_stats()
                content = types.TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2)
                )
                
            elif uri == "neural://project/index" and self.project_indexer:
                # Return project index status
                status = {
                    "indexed_files": len(self.project_indexer.file_metadata),
                    "total_embeddings": sum(len(chunks) for chunks in self.project_indexer.file_chunks.values()),
                    "last_updated": str(self.project_indexer.last_update) if hasattr(self.project_indexer, 'last_update') else None
                }
                content = types.TextContent(
                    type="text", 
                    text=json.dumps(status, indent=2)
                )
                
            elif uri == "neural://config/features":
                # Return feature flag configuration
                config = self.feature_manager.get_stats()
                content = types.TextContent(
                    type="text",
                    text=json.dumps(config, indent=2)
                )
                
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
            
            return types.ReadResourceResult(contents=[content])
        
        # Tools
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            """List available neural tools"""
            tools = []
            
            # Memory search tool
            tools.append(types.Tool(
                name="neural_memory_search",
                description="Search conversation memory using semantic embeddings",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "n_results": {"type": "integer", "default": 5, "description": "Number of results"},
                        "priority_filter": {"type": "integer", "description": "Minimum priority level"}
                    },
                    "required": ["query"]
                }
            ))
            
            # Project search tool
            if self.project_indexer:
                tools.append(types.Tool(
                    name="neural_project_search",
                    description="Search project code using AST-aware semantic embeddings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Code search query"},
                            "file_types": {"type": "array", "items": {"type": "string"}, "description": "File extensions to search"},
                            "n_results": {"type": "integer", "default": 10, "description": "Number of results"}
                        },
                        "required": ["query"]
                    }
                ))
            
            # Memory indexing tool
            tools.append(types.Tool(
                name="neural_memory_index",
                description="Index new content into conversation memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to index"},
                        "priority": {"type": "integer", "default": 5, "description": "Memory priority (1-10)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"}
                    },
                    "required": ["content"]
                }
            ))
            
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            """Execute neural tool"""
            
            if name == "neural_memory_search":
                # Perform memory search
                results = self.memory_system.search_memory(
                    query=arguments["query"],
                    n_results=arguments.get("n_results", 5),
                    priority_filter=arguments.get("priority_filter")
                )
                
                content = types.TextContent(
                    type="text",
                    text=json.dumps([{
                        "content": r["content"],
                        "priority": r.get("priority", 0),
                        "timestamp": str(r.get("timestamp", "")),
                        "tags": r.get("tags", [])
                    } for r in results], indent=2)
                )
                
            elif name == "neural_project_search" and self.project_indexer:
                # Perform project code search
                results = self.project_indexer.search_code(
                    query=arguments["query"],
                    file_types=arguments.get("file_types"),
                    n_results=arguments.get("n_results", 10)
                )
                
                content = types.TextContent(
                    type="text",
                    text=json.dumps([{
                        "file_path": r["file_path"],
                        "chunk_content": r["content"],
                        "line_range": r.get("line_range"),
                        "ast_type": r.get("ast_type"),
                        "similarity_score": r.get("similarity", 0)
                    } for r in results], indent=2)
                )
                
            elif name == "neural_memory_index":
                # Index new content
                success = self.memory_system.add_memory(
                    content=arguments["content"],
                    priority=arguments.get("priority", 5),
                    tags=arguments.get("tags", [])
                )
                
                content = types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": success,
                        "message": "Content indexed successfully" if success else "Failed to index content"
                    })
                )
                
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            return [content]

async def main():
    """Main MCP server entry point - L9 2025 optimized"""
    
    # Configure logging for L9 environment
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting L9 Neural Flow MCP Server...")
    logger.info("🔮 L9 Mode: Single Qodo-Embed architecture")
    logger.info("📡 MCP Protocol: JSON-RPC 2.0 compliant")
    
    # Create and initialize L9-optimized server
    mcp_server = NeuralFlowMCPServer() 
    await mcp_server.initialize()
    mcp_server.register_handlers()
    
    logger.info("✅ L9 Neural Flow MCP Server ready")
    
    # Start stdio server with modern MCP SDK pattern
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await mcp_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="neural-flow-l9",
                server_version="1.0.0",
                capabilities=mcp_server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())