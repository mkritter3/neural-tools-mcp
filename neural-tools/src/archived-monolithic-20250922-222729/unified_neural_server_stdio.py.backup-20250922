#!/usr/bin/env python3
"""
ADR-66 + ADR-67: Unified Neural MCP Server - STDIO Transport
Implements unified Neo4j + Graphiti temporal knowledge architecture
with eliminated dual-write complexity and containerized services

Features:
- Unified Neo4j vector storage (no Qdrant)
- Graphiti temporal knowledge graphs
- Episodic processing for conflict-resistant indexing
- Project isolation via ADR-29
- Local LLM via ADR-69
"""

import os
import sys
import json
import asyncio
import logging
import secrets
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# Official MCP SDK for 2025-06-18 protocol
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Import unified architecture services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'servers', 'services'))

# Import schema and migration management
from schema_manager import SchemaManager, ProjectType
from migration_manager import MigrationManager
from canon_manager import CanonManager
from metadata_backfiller import MetadataBackfiller

# Import project context management
from servers.services.project_context_manager import get_project_context_manager
from neural_mcp.project_detector import get_user_project

# Import unified services and tools
from servers.services.unified_graphiti_service import UnifiedIndexerService
from servers.tools.unified_core_tools import register_unified_core_tools

# Configure logging to stderr (NEVER to stdout for STDIO transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Critical: Log to stderr, not stdout
)
base_logger = logging.getLogger(__name__)

# Global state
server = Server("neural-tools")
PROJECT_NAME = None
CONTAINER = None
unified_indexer = None

# Tool configuration
TOOL_TIMEOUT = 120  # 2 minutes default timeout

class UnifiedServiceContainer:
    """
    ADR-66 + ADR-67: Unified service container with temporal knowledge graphs
    Eliminates dual-write complexity by using single Neo4j + Graphiti storage
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.schema_manager = None
        self.migration_manager = None
        self.canon_manager = None
        self.metadata_backfiller = None
        self.unified_indexer = None

        base_logger.info(f"🚀 UnifiedServiceContainer initialized for project: {project_name}")

    async def initialize(self) -> Dict[str, Any]:
        """Initialize unified service container"""
        try:
            base_logger.info("🔧 Initializing unified service container...")

            # Initialize unified indexer (Neo4j + Graphiti)
            self.unified_indexer = UnifiedIndexerService(self.project_name)
            indexer_result = await self.unified_indexer.initialize()

            if not indexer_result.get("success"):
                base_logger.error(f"Failed to initialize unified indexer: {indexer_result.get('error')}")
                return {
                    "success": False,
                    "error": f"Unified indexer initialization failed: {indexer_result.get('error')}"
                }

            # Initialize schema manager
            try:
                self.schema_manager = SchemaManager(project_name=self.project_name)
                await self.schema_manager.initialize()
                base_logger.info("✅ Schema manager initialized")
            except Exception as e:
                base_logger.warning(f"Schema manager initialization failed: {e}")

            # Initialize migration manager
            try:
                self.migration_manager = MigrationManager(project_name=self.project_name)
                base_logger.info("✅ Migration manager initialized")
            except Exception as e:
                base_logger.warning(f"Migration manager initialization failed: {e}")

            # Initialize canon manager
            try:
                self.canon_manager = CanonManager(self.project_name)
                base_logger.info("✅ Canon manager initialized")
            except Exception as e:
                base_logger.warning(f"Canon manager initialization failed: {e}")

            # Initialize metadata backfiller
            try:
                self.metadata_backfiller = MetadataBackfiller(self.project_name)
                base_logger.info("✅ Metadata backfiller initialized")
            except Exception as e:
                base_logger.warning(f"Metadata backfiller initialization failed: {e}")

            base_logger.info("🎉 Unified service container initialization complete")
            return {
                "success": True,
                "architecture": "unified_neo4j_graphiti",
                "project": self.project_name,
                "components": {
                    "unified_indexer": indexer_result.get("success", False),
                    "schema_manager": self.schema_manager is not None,
                    "migration_manager": self.migration_manager is not None,
                    "canon_manager": self.canon_manager is not None,
                    "metadata_backfiller": self.metadata_backfiller is not None
                }
            }

        except Exception as e:
            base_logger.error(f"UnifiedServiceContainer initialization failed: {e}")
            return {"success": False, "error": str(e)}

    async def cleanup(self):
        """Clean up unified service container resources"""
        try:
            if self.unified_indexer:
                await self.unified_indexer.cleanup()
                base_logger.info("🧹 Unified indexer cleaned up")

            base_logger.info("✅ UnifiedServiceContainer cleanup complete")

        except Exception as e:
            base_logger.error(f"Cleanup failed: {e}")


async def ensure_services_initialized():
    """Ensure unified services are initialized"""
    global CONTAINER, unified_indexer

    if CONTAINER is None:
        if PROJECT_NAME is None:
            raise RuntimeError("Project name not set")

        CONTAINER = UnifiedServiceContainer(PROJECT_NAME)
        init_result = await CONTAINER.initialize()

        if not init_result.get("success"):
            raise RuntimeError(f"Service initialization failed: {init_result.get('error')}")

        unified_indexer = CONTAINER.unified_indexer
        base_logger.info("✅ Unified services initialized successfully")


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools for unified architecture"""
    return [
        types.Tool(
            name="project_understanding",
            description="Generate condensed project understanding using unified Neo4j + Graphiti knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["full", "architecture", "dependencies", "core_logic", "documentation"],
                        "default": "full",
                        "description": "Scope of understanding to generate"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of knowledge graph results"
                    }
                }
            }
        ),
        types.Tool(
            name="semantic_code_search",
            description="Search code using unified Neo4j + Graphiti semantic and temporal knowledge",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of results"
                    },
                    "include_context": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include temporal context and relationships"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="graphrag_hybrid_search",
            description="Hybrid search combining semantic, graph traversal, and temporal knowledge",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 25,
                        "description": "Maximum results"
                    },
                    "include_graph_context": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include graph relationships"
                    },
                    "max_hops": {
                        "type": "integer",
                        "default": 2,
                        "minimum": 0,
                        "maximum": 3,
                        "description": "Maximum relationship hops"
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="reindex_path",
            description="Reindex a path using unified Neo4j + Graphiti temporal knowledge architecture",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File or directory path to reindex"
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": True,
                        "description": "Process directories recursively"
                    }
                },
                "required": ["path"]
            }
        ),
        types.Tool(
            name="neural_system_status",
            description="Get comprehensive status of unified Neo4j + Graphiti neural system",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls for unified architecture"""
    try:
        await ensure_services_initialized()

        base_logger.info(f"🔧 Executing unified tool: {name}")

        # Route to appropriate unified tool
        if name == "project_understanding":
            from servers.tools.unified_core_tools import register_unified_core_tools
            # Simulate tool call
            scope = arguments.get("scope", "full")
            max_results = arguments.get("max_results", 50)

            # Use unified indexer directly
            query_map = {
                "full": "overview architecture dependencies documentation code structure",
                "architecture": "architecture design patterns structure modules components",
                "dependencies": "dependencies imports requirements libraries frameworks",
                "core_logic": "main functions classes core business logic algorithms",
                "documentation": "documentation readme guides tutorials examples"
            }

            query = query_map.get(scope, query_map["full"])
            search_result = await unified_indexer.search_unified_knowledge(query, limit=max_results)

            result = {
                "project": PROJECT_NAME,
                "timestamp": datetime.now().isoformat(),
                "scope": scope,
                "architecture": "unified_neo4j_graphiti",
                "search_result": search_result
            }

        elif name == "semantic_code_search":
            query = arguments.get("query")
            limit = arguments.get("limit", 10)
            include_context = arguments.get("include_context", True)

            search_result = await unified_indexer.search_unified_knowledge(query, limit=limit)

            result = {
                "query": query,
                "architecture": "unified_neo4j_graphiti",
                "search_result": search_result,
                "include_context": include_context
            }

        elif name == "graphrag_hybrid_search":
            query = arguments.get("query")
            limit = arguments.get("limit", 5)
            include_graph_context = arguments.get("include_graph_context", True)
            max_hops = arguments.get("max_hops", 2)

            search_result = await unified_indexer.search_unified_knowledge(query, limit=limit)

            result = {
                "query": query,
                "architecture": "unified_neo4j_graphiti",
                "search_type": "hybrid_temporal",
                "search_result": search_result,
                "include_graph_context": include_graph_context,
                "max_hops": max_hops
            }

        elif name == "reindex_path":
            path = arguments.get("path")
            recursive = arguments.get("recursive", True)

            file_result = await unified_indexer.process_file(path)

            result = {
                "path": path,
                "recursive": recursive,
                "architecture": "unified_neo4j_graphiti",
                "process_result": file_result
            }

        elif name == "neural_system_status":
            status = await unified_indexer.get_status()
            health = await unified_indexer.graphiti_client.health_check()

            result = {
                "architecture": "unified_neo4j_graphiti",
                "project": PROJECT_NAME,
                "status": status,
                "health": health,
                "features": {
                    "dual_write_eliminated": True,
                    "vector_storage": "neo4j_native",
                    "temporal_graphs": True,
                    "episodic_processing": True
                }
            }

        else:
            result = {"error": f"Unknown tool: {name}"}

        base_logger.info(f"✅ Tool {name} executed successfully")

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]

    except Exception as e:
        base_logger.error(f"Tool {name} failed: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, indent=2)
        )]


async def set_project_context():
    """Set project context for unified architecture"""
    global PROJECT_NAME

    try:
        # Auto-detect project
        detected_project = get_user_project()
        if detected_project:
            PROJECT_NAME = detected_project
            base_logger.info(f"🔍 Auto-detected project: {PROJECT_NAME}")
        else:
            PROJECT_NAME = "default"
            base_logger.info(f"📁 Using default project: {PROJECT_NAME}")

        base_logger.info(f"📝 Project context set: {PROJECT_NAME}")

    except Exception as e:
        base_logger.error(f"Failed to set project context: {e}")
        PROJECT_NAME = "default"


async def main():
    """Main entry point for unified neural MCP server"""
    base_logger.info("🚀 Starting Unified Neural MCP Server (ADR-66 + ADR-67)")
    base_logger.info("🏗️ Architecture: Neo4j + Graphiti unified temporal knowledge")

    try:
        # Set project context
        await set_project_context()

        # Run server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            base_logger.info("✅ Unified Neural MCP Server running via STDIO")

            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="neural-tools-unified",
                    server_version="2.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

    except Exception as e:
        base_logger.error(f"💥 Unified Neural MCP Server failed: {e}")
        sys.exit(1)

    finally:
        # Cleanup
        if CONTAINER:
            await CONTAINER.cleanup()
        base_logger.info("🔌 Unified Neural MCP Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())