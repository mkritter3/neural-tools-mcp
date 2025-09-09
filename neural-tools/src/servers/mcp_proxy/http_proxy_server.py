#!/usr/bin/env python3
"""
MCP HTTP Proxy Server - Lightweight Implementation
Thin proxy layer that converts MCP tool calls to HTTP API calls to FastAPI backend

This replaces the complex database-accessing MCP server with a simple HTTP client
that follows the 2025-06-18 MCP protocol specification.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import httpx

# Official MCP SDK for 2025-06-18 protocol
from mcp.server import Server, NotificationOptions  
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Import our HTTP proxy components
from .config import settings, get_config_summary
from .factory import ToolProxyFactory
from .errors import MCPProxyError, create_error_response

# Configure logging to stderr (NEVER to stdout for STDIO transport)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Critical: Log to stderr, not stdout for MCP STDIO
)
logger = logging.getLogger(__name__)


class HTTPProxyMCPServer:
    """
    Lightweight MCP server that proxies tool calls to FastAPI HTTP backend
    
    This server:
    1. Initializes HTTP client with API key authentication
    2. Dynamically loads tool definitions from JSON configuration  
    3. Registers HTTP proxy functions for each tool
    4. Handles MCP protocol compliance and error responses
    """
    
    def __init__(self):
        # Initialize MCP server
        self.server = Server("neural-search-http-proxy")
        self.http_client: Optional[httpx.AsyncClient] = None
        self.factory: Optional[ToolProxyFactory] = None
        self.initialized = False
        
        logger.info(f"🚀 Initializing MCP HTTP Proxy Server")
        logger.info(f"📋 Configuration: {get_config_summary()}")

    async def setup(self):
        """Initialize HTTP client and register tools"""
        try:
            await self._initialize_http_client()
            await self._register_tools()
            self.initialized = True
            logger.info("✅ MCP HTTP Proxy Server setup complete")
            
        except Exception as e:
            logger.error(f"❌ Server setup failed: {e}")
            raise

    async def _initialize_http_client(self):
        """Initialize HTTP client with timeouts (authentication disabled)"""
        headers = {
            "User-Agent": "MCP-HTTP-Proxy/1.0.0"
        }
        
        # Add API key header only if configured
        if settings.FASTAPI_API_KEY:
            headers["X-API-Key"] = settings.FASTAPI_API_KEY
        
        self.http_client = httpx.AsyncClient(
            base_url=settings.FASTAPI_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(settings.HTTP_TIMEOUT),
            follow_redirects=True
        )
        
        # Test connection to FastAPI backend
        try:
            response = await self.http_client.get("/health")
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Connected to FastAPI backend: {health_data.get('status', 'unknown')}")
            else:
                logger.warning(f"⚠️ FastAPI backend health check returned {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to FastAPI backend at {settings.FASTAPI_BASE_URL}: {e}")
            # Don't raise - allow server to start even if backend is temporarily unavailable

    async def _register_tools(self):
        """Load tool definitions and register proxy functions"""
        try:
            # Create factory and load tools from JSON
            self.factory = ToolProxyFactory.from_json_file(
                self.server,
                self.http_client,
                settings.MCP_TOOLS_CONFIG
            )
            
            # Register all tools
            tool_count = self.factory.register_tools()
            
            if tool_count == 0:
                logger.warning("⚠️ No tools were registered - check tool definitions")
            else:
                registered_tools = self.factory.get_registered_tools()
                logger.info(f"✅ Registered {tool_count} tools: {', '.join(registered_tools)}")
                
        except FileNotFoundError:
            logger.error(f"❌ Tool configuration file not found: {settings.MCP_TOOLS_CONFIG}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to register tools: {e}")
            raise

    async def cleanup(self):
        """Clean up resources on shutdown"""
        logger.info("🛑 Shutting down MCP HTTP Proxy Server")
        
        if self.http_client:
            try:
                await self.http_client.aclose()
                logger.info("✅ HTTP client closed")
            except Exception as e:
                logger.error(f"❌ Error closing HTTP client: {e}")

    def run(self):
        """Run the MCP server with STDIO transport"""
        try:
            logger.info("🚀 Starting MCP HTTP Proxy Server with STDIO transport")
            
            # Run setup and server in the event loop
            async def run_server():
                try:
                    await self.setup()
                    
                    # Register server lifecycle hooks
                    @self.server.list_tools()
                    async def handle_list_tools() -> List[types.Tool]:
                        """List all available tools"""
                        if not self.factory:
                            return []
                        
                        tools = []
                        for tool_def in self.factory.tool_definitions:
                            tools.append(types.Tool(
                                name=tool_def["name"],
                                description=tool_def.get("description", ""),
                                inputSchema=tool_def.get("parameters", {})
                            ))
                        
                        return tools

                    @self.server.call_tool()
                    async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
                        """Handle tool execution requests"""
                        if not self.factory or not self.initialized:
                            error_response = create_error_response(
                                MCPProxyError(4004, "Server not properly initialized")
                            )
                            return [types.TextContent(
                                type="text",
                                text=str(error_response)
                            )]
                        
                        # Find the tool definition
                        tool_def = None
                        for definition in self.factory.tool_definitions:
                            if definition["name"] == name:
                                tool_def = definition
                                break
                        
                        if not tool_def:
                            error_response = create_error_response(
                                MCPProxyError(2004, f"Tool '{name}' not found")
                            )
                            return [types.TextContent(
                                type="text",
                                text=str(error_response)
                            )]
                        
                        # Create and execute proxy function
                        try:
                            proxy_func = self.factory._create_proxy_function(tool_def)
                            result = await proxy_func(**arguments)
                            return result
                        except Exception as e:
                            logger.error(f"Tool execution failed for {name}: {e}", exc_info=True)
                            error_response = create_error_response(e)
                            return [types.TextContent(
                                type="text",
                                text=str(error_response)
                            )]
                    
                    # Run the MCP server
                    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                        await self.server.run(
                            read_stream,
                            write_stream,
                            InitializationOptions(
                                server_name="neural-search-http-proxy",
                                server_version="1.0.0",
                                capabilities=self.server.get_capabilities(
                                    notification_options=NotificationOptions(),
                                    experimental_capabilities={}
                                )
                            )
                        )
                        
                except Exception as e:
                    logger.error(f"❌ Server error: {e}", exc_info=True)
                    raise
                finally:
                    await self.cleanup()
            
            # Run the async server
            asyncio.run(run_server())
            
        except KeyboardInterrupt:
            logger.info("👋 Server stopped by user")
        except Exception as e:
            logger.error(f"💥 Fatal server error: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Main entry point for the MCP HTTP proxy server"""
    try:
        # Validate configuration
        if not settings.FASTAPI_BASE_URL:
            logger.error("❌ FASTAPI_BASE_URL not configured")
            sys.exit(1)
        
        if not settings.FASTAPI_API_KEY:
            logger.warning("⚠️ FASTAPI_API_KEY not configured - API calls may fail")
        
        # Create and run server
        server = HTTPProxyMCPServer()
        server.run()
        
    except Exception as e:
        logger.error(f"💥 Failed to start server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()