#!/usr/bin/env python3
"""
MCP Server - September 2025 Modular Architecture
Orchestrates tool execution with automatic discovery and registration

ADR-0076: Modular tool architecture with auto-discovery
"""

import os
import sys
import json
import asyncio
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any

# Official MCP SDK for 2025-06-18 protocol
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Shared utilities
from .shared.connection_pool import cleanup_shared_services, get_connection_stats
from .shared.performance_metrics import get_performance_metrics, reset_metrics
from .shared.cache_manager import get_cache_stats, cleanup_expired_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global server instance
server = Server("neural-tools-modular")

def discover_tools() -> List[types.Tool]:
    """Auto-discover and register all tools from tools/ directory"""
    tools_dir = Path(__file__).parent / "tools"
    discovered_tools = []

    logger.info(f"🔍 Discovering tools in {tools_dir}")

    for tool_file in tools_dir.glob("*.py"):
        if tool_file.stem.startswith("_"):
            continue

        try:
            module_name = f"neural_mcp.tools.{tool_file.stem}"
            module = importlib.import_module(module_name)

            if hasattr(module, 'TOOL_CONFIG') and hasattr(module, 'execute'):
                tool = types.Tool(
                    name=module.TOOL_CONFIG["name"],
                    description=module.TOOL_CONFIG["description"],
                    inputSchema=module.TOOL_CONFIG["inputSchema"]
                )
                discovered_tools.append(tool)
                logger.info(f"✅ Registered tool: {module.TOOL_CONFIG['name']}")
            else:
                logger.warning(f"⚠️  Tool {tool_file.stem} missing TOOL_CONFIG or execute function")

        except Exception as e:
            logger.error(f"❌ Failed to load tool {tool_file.stem}: {e}")

    logger.info(f"🎯 Discovered {len(discovered_tools)} tools")
    return discovered_tools

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available tools via auto-discovery"""
    return discover_tools()

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Route tool calls to appropriate modules"""
    logger.info(f"🔧 Executing tool: {name}")

    try:
        module_name = f"neural_mcp.tools.{name}"
        module = importlib.import_module(module_name)

        if not hasattr(module, 'execute'):
            return _make_error_response(f"Tool {name} missing execute function")

        # Execute tool with error handling
        result = await module.execute(arguments)
        return result

    except ImportError:
        logger.error(f"Tool not found: {name}")
        return _make_error_response(f"Tool not found: {name}")
    except Exception as e:
        logger.error(f"Tool execution failed for {name}: {e}")
        return _make_error_response(f"Tool execution failed: {e}")

@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """List available resources"""
    return [
        types.Resource(
            uri="neural://system/status",
            name="System Status",
            description="Neural system status and health metrics",
            mimeType="application/json"
        ),
        types.Resource(
            uri="neural://system/performance",
            name="Performance Metrics",
            description="Tool performance and caching statistics",
            mimeType="application/json"
        ),
        types.Resource(
            uri="neural://system/connections",
            name="Connection Stats",
            description="Connection pool statistics",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read system resources"""
    if uri == "neural://system/status":
        return json.dumps({
            "status": "healthy",
            "architecture": "modular_september_2025",
            "tools_available": len(discover_tools()),
            "shared_services": "active"
        }, indent=2)

    elif uri == "neural://system/performance":
        return json.dumps(get_performance_metrics(), indent=2)

    elif uri == "neural://system/connections":
        return json.dumps(get_connection_stats(), indent=2)

    else:
        raise ValueError(f"Unknown resource: {uri}")

def _make_error_response(error: str) -> List[types.TextContent]:
    """Standard error response format"""
    return [types.TextContent(type="text", text=json.dumps({
        "status": "error",
        "message": error,
        "architecture": "modular_september_2025"
    }))]

async def cleanup_resources():
    """Cleanup all resources on shutdown"""
    logger.info("🧹 Cleaning up resources...")
    await cleanup_shared_services()
    cleanup_expired_cache()
    logger.info("✅ Cleanup complete")

async def main():
    """Main server entry point"""
    logger.info("🚀 Starting Neural Tools MCP Server (September 2025 Modular Architecture)")

    # Proactive container initialization (ADR-0100)
    try:
        logger.info("🎭 Initializing Container Orchestrator...")
        from servers.services.container_orchestrator import get_container_orchestrator
        orchestrator = await get_container_orchestrator()
        logger.info("✅ Container Orchestrator ready - all services pre-warmed")
    except Exception as e:
        logger.warning(f"⚠️ Container Orchestrator initialization failed: {e}")
        logger.warning("⚠️ Falling back to on-demand container startup")

    # Initialize server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("📡 STDIO transport initialized")

        try:
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="neural-tools-modular",
                    server_version="2025.09.22",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            await cleanup_resources()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Server shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)