#!/usr/bin/env python3
"""
DEPRECATED: Legacy MCP STDIO wrapper

Use the unified stdio server instead:
  neural-tools/run_mcp_server.py  (which launches src/servers/neural_server_stdio.py)

This file is kept for backward compatibility but should not be used in new setups.
"""

import asyncio
import sys
import json
import os
from pathlib import Path

# Add the server directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

async def main():
    """Main entry point for MCP stdio communication"""
    try:
        # Import the neural server wrapper
        from neural_mcp_wrapper import _neural_server as server
        
        # Check if server is loaded
        if server is None:
            print(json.dumps({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": "Neural server not loaded"}
            }), flush=True)
            return
        
        # Basic MCP protocol handling
        async def handle_request(request):
            try:
                method = request.get('method')
                params = request.get('params', {})
                request_id = request.get('id')
                
                if method == 'initialize':
                    # Return MCP initialization response
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {
                                    "listChanged": False
                                }
                            },
                            "serverInfo": {
                                "name": "neural-tools-mcp-server",
                                "version": "1.0.0"
                            }
                        }
                    }
                    return response
                    
                elif method == 'tools/list':
                    # List all available MCP tools
                    tools = [
                        {"name": "memory_store_enhanced", "description": "Store content with enhanced metadata"},
                        {"name": "memory_search_enhanced", "description": "Search stored content with filters"},
                        {"name": "graph_query", "description": "Execute Kuzu graph queries"},
                        {"name": "schema_customization", "description": "Customize collection schemas"},
                        {"name": "atomic_dependency_tracer", "description": "Trace code dependencies"},
                        {"name": "project_understanding", "description": "Generate project understanding"},
                        {"name": "semantic_code_search", "description": "Search code semantically"},
                        {"name": "vibe_preservation", "description": "Preserve coding patterns"},
                        {"name": "project_auto_index", "description": "Auto-index project files"},
                        {"name": "neural_system_status", "description": "Get system status"},
                        {"name": "neo4j_graph_query", "description": "Execute Neo4j Cypher queries"},
                        {"name": "neo4j_semantic_graph_search", "description": "Search Neo4j graph semantically"},
                        {"name": "neo4j_code_dependencies", "description": "Get code dependencies from Neo4j"},
                        {"name": "neo4j_migration_status", "description": "Check Neo4j migration status"},
                        {"name": "neo4j_index_code_graph", "description": "Index code into Neo4j graph"}
                    ]
                    
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": tools}
                    }
                    return response
                    
                elif method == 'tools/call':
                    # Call a specific tool
                    tool_name = params.get('name')
                    tool_args = params.get('arguments', {})
                    
                    if hasattr(server, tool_name):
                        tool_func = getattr(server, tool_name)
                        try:
                            if asyncio.iscoroutinefunction(tool_func):
                                result = await tool_func(**tool_args)
                            else:
                                result = tool_func(**tool_args)
                            
                            response = {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "result": result
                            }
                            return response
                        except Exception as e:
                            response = {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32000,
                                    "message": f"Tool execution failed: {str(e)}"
                                }
                            }
                            return response
                    else:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": f"Tool '{tool_name}' not found"
                            }
                        }
                        return response
                else:
                    # Unknown method
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method '{method}' not found"
                        }
                    }
                    return response
                    
            except Exception as e:
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get('id'),
                    "error": {
                        "code": -32000,
                        "message": f"Request processing failed: {str(e)}"
                    }
                }
                return response
        
        # Read from stdin and process requests
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    request = json.loads(line)
                    response = await handle_request(request)
                    print(json.dumps(response), flush=True)
                except json.JSONDecodeError as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {str(e)}"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                    
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32000,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                print(json.dumps(error_response), flush=True)
                
    except Exception as e:
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": f"Server initialization failed: {str(e)}"
            }
        }
        print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    asyncio.run(main())
