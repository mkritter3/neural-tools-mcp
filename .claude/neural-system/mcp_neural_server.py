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
from neural_dynamic_memory_system import DynamicMemorySystem
from project_neural_indexer import ProjectNeuralIndexer

# MCP Protocol imports (these would need to be added to requirements)
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Resource, Tool, TextContent, Prompt,
        GetResourceRequest, GetResourceResult,
        ListResourcesRequest, ListResourcesResult,
        CallToolRequest, CallToolResult,
        ListToolsRequest, ListToolsResult,
        GetPromptRequest, GetPromptResult,
        ListPromptsRequest, ListPromptsResult,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP SDK not available. Install with: pip install mcp", file=sys.stderr)

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
            self.memory_system = DynamicMemorySystem()
            
            # Initialize project indexer if in project context
            project_root = Path.cwd()
            if (project_root / "src").exists() or (project_root / ".git").exists():
                self.project_indexer = ProjectNeuralIndexer(project_root)
                
            logger.info("✅ Neural Flow systems initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neural Flow systems: {e}")
            raise
    
    def register_handlers(self):
        """Register MCP request handlers"""
        
        # Resources
        @self.server.list_resources()
        async def list_resources() -> list[Resource]:
            """List available neural resources"""
            resources = []
            
            # Memory resources
            resources.append(Resource(
                uri="neural://memory/conversations",
                name="Conversation Memory",
                description="Dynamic conversation memory with tiered storage",
                mimeType="application/json"
            ))
            
            # Project resources
            if self.project_indexer:
                resources.append(Resource(
                    uri="neural://project/index",
                    name="Project Code Index", 
                    description="AST-analyzed project code with semantic embeddings",
                    mimeType="application/json"
                ))
                
            # Feature flags
            resources.append(Resource(
                uri="neural://config/features",
                name="Feature Configuration",
                description="Current feature flags and A/B test settings",
                mimeType="application/json"
            ))
            
            return resources
        
        @self.server.get_resource()
        async def get_resource(request: GetResourceRequest) -> GetResourceResult:
            """Get specific neural resource"""
            uri = request.uri
            
            if uri == "neural://memory/conversations":
                # Return conversation memory stats
                stats = self.memory_system.get_memory_stats()
                content = TextContent(
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
                content = TextContent(
                    type="text", 
                    text=json.dumps(status, indent=2)
                )
                
            elif uri == "neural://config/features":
                # Return feature flag configuration
                config = self.feature_manager.get_stats()
                content = TextContent(
                    type="text",
                    text=json.dumps(config, indent=2)
                )
                
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
            
            return GetResourceResult(contents=[content])
        
        # Tools
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available neural tools"""
            tools = []
            
            # Memory search tool
            tools.append(Tool(
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
                tools.append(Tool(
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
            tools.append(Tool(
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
        async def call_tool(request: CallToolRequest) -> CallToolResult:
            """Execute neural tool"""
            tool_name = request.name
            args = request.arguments or {}
            
            if tool_name == "neural_memory_search":
                # Perform memory search
                results = self.memory_system.search_memory(
                    query=args["query"],
                    n_results=args.get("n_results", 5),
                    priority_filter=args.get("priority_filter")
                )
                
                content = TextContent(
                    type="text",
                    text=json.dumps([{
                        "content": r["content"],
                        "priority": r.get("priority", 0),
                        "timestamp": str(r.get("timestamp", "")),
                        "tags": r.get("tags", [])
                    } for r in results], indent=2)
                )
                
            elif tool_name == "neural_project_search" and self.project_indexer:
                # Perform project code search
                results = self.project_indexer.search_code(
                    query=args["query"],
                    file_types=args.get("file_types"),
                    n_results=args.get("n_results", 10)
                )
                
                content = TextContent(
                    type="text",
                    text=json.dumps([{
                        "file_path": r["file_path"],
                        "chunk_content": r["content"],
                        "line_range": r.get("line_range"),
                        "ast_type": r.get("ast_type"),
                        "similarity_score": r.get("similarity", 0)
                    } for r in results], indent=2)
                )
                
            elif tool_name == "neural_memory_index":
                # Index new content
                success = self.memory_system.add_memory(
                    content=args["content"],
                    priority=args.get("priority", 5),
                    tags=args.get("tags", [])
                )
                
                content = TextContent(
                    type="text",
                    text=json.dumps({
                        "success": success,
                        "message": "Content indexed successfully" if success else "Failed to index content"
                    })
                )
                
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            return CallToolResult(contents=[content])

async def main():
    """Main MCP server entry point"""
    if not MCP_AVAILABLE:
        print("❌ MCP SDK required. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)
        
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and initialize server
    mcp_server = NeuralFlowMCPServer() 
    await mcp_server.initialize()
    mcp_server.register_handlers()
    
    # Start stdio server
    async with stdio_server(mcp_server.server) as streams:
        await mcp_server.server.run(streams[0], streams[1], InitializationOptions())

if __name__ == "__main__":
    if not MCP_AVAILABLE:
        print("⚠️  Running in compatibility mode without MCP protocol", file=sys.stderr)
        # Basic neural system test
        import asyncio
        async def test_systems():
            server = NeuralFlowMCPServer()
            await server.initialize()
            print("✅ Neural systems initialized successfully")
        asyncio.run(test_systems())
    else:
        asyncio.run(main())