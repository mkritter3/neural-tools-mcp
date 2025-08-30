#!/usr/bin/env python3
"""
Robust MCP Server for Neural Flow Tools with Smart Pre-loading
Combines reliability with performance optimization
"""

import asyncio
import json
import sys
import os
import time
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize server
server = Server("neural-flow")

# Global instances - thread-safe singleton pattern
_lock = threading.Lock()
_memory_system = None
_indexer = None
_initialization_status = "pending"  # pending, loading, ready, failed
_initialization_error = None
_initialization_start = None

# Background executor for pre-loading
_executor = ThreadPoolExecutor(max_workers=1)

def initialize_neural_systems():
    """Initialize neural systems in background thread"""
    global _memory_system, _indexer, _initialization_status, _initialization_error, _initialization_start
    
    _initialization_start = time.time()
    _initialization_status = "loading"
    
    try:
        print("🚀 Background: Starting neural system initialization...", file=sys.stderr)
        
        # Import and initialize memory system
        from neural_dynamic_memory_system import NeuralDynamicMemorySystem
        _memory_system = NeuralDynamicMemorySystem()
        print("  ✅ Background: Memory system ready", file=sys.stderr)
        
        # Import and initialize indexer
        from project_neural_indexer import ProjectNeuralIndexer
        _indexer = ProjectNeuralIndexer()
        print("  ✅ Background: Project indexer ready", file=sys.stderr)
        
        # Warm up with minimal operations
        try:
            _memory_system.store_memory(
                conversation_id="warmup",
                text="System initialization test",
                metadata={"type": "warmup", "timestamp": time.time()}
            )
            print("  ✅ Background: Models warmed up", file=sys.stderr)
        except:
            pass  # Warm-up is optional
        
        load_time = time.time() - _initialization_start
        print(f"✅ Background: Neural systems ready in {load_time:.2f}s", file=sys.stderr)
        _initialization_status = "ready"
        
    except Exception as e:
        print(f"❌ Background: Initialization failed: {e}", file=sys.stderr)
        _initialization_error = str(e)
        _initialization_status = "failed"

# Start background initialization immediately
_init_future = _executor.submit(initialize_neural_systems)

def get_neural_systems():
    """Get neural systems, waiting if still loading"""
    global _memory_system, _indexer
    
    if _initialization_status == "ready":
        return _memory_system, _indexer
    elif _initialization_status == "loading":
        # Wait up to 5 seconds for initialization
        max_wait = 5.0
        start = time.time()
        while _initialization_status == "loading" and (time.time() - start) < max_wait:
            time.sleep(0.1)
        
        if _initialization_status == "ready":
            return _memory_system, _indexer
    
    return None, None

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="memory_query",
            description="Query neural dynamic memory with semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search query"},
                    "namespace": {"type": "string", "description": "Optional namespace"},
                    "limit": {"type": "integer", "default": 10},
                    "use_dynamic_scoring": {"type": "boolean", "default": True}
                },
                "required": ["pattern"]
            }
        ),
        Tool(
            name="memory_store",
            description="Store content in neural memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Unique key"},
                    "value": {"type": "string", "description": "Content to store"},
                    "namespace": {"type": "string", "default": "default"},
                    "metadata": {"type": "object", "description": "Optional metadata"}
                },
                "required": ["key", "value"]
            }
        ),
        Tool(
            name="memory_stats",
            description="Get neural memory system statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Optional namespace filter"}
                }
            }
        ),
        Tool(
            name="index_project_files",
            description="Index project files with neural embeddings",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root": {"type": "string", "description": "Project directory"},
                    "max_files": {"type": "integer", "description": "Max files to index"},
                    "use_neural": {"type": "boolean", "default": True}
                }
            }
        ),
        Tool(
            name="search_project_files",
            description="Search project files using neural embeddings",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10},
                    "similarity_threshold": {"type": "number", "default": 0.0}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="familiarize_with_project",
            description="Comprehensive project analysis with neural understanding",
            inputSchema={
                "type": "object",
                "properties": {
                    "depth": {"type": "string", "enum": ["basic", "detailed", "comprehensive"], "default": "comprehensive"},
                    "project_root": {"type": "string", "description": "Project directory"}
                }
            }
        ),
        Tool(
            name="system_status",
            description="Check neural system status and performance",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls with pre-loaded or fallback systems"""
    
    start_time = time.time()
    
    try:
        # Special status tool
        if name == "system_status":
            elapsed = time.time() - _initialization_start if _initialization_start else 0
            result = {
                "success": True,
                "status": _initialization_status,
                "memory_system": "Ready" if _memory_system else "Not loaded",
                "indexer": "Ready" if _indexer else "Not loaded",
                "initialization_time": f"{elapsed:.2f}s" if _initialization_start else "Not started",
                "error": _initialization_error,
                "expected_latency": "<50ms" if _initialization_status == "ready" else "~774ms (cold)"
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        # Try to use pre-loaded systems first
        memory_system, indexer = get_neural_systems()
        
        if memory_system and indexer:
            # Use pre-loaded systems directly for best performance
            if name == "memory_query":
                results = memory_system.retrieve_relevant_memories(
                    query=arguments["pattern"],
                    conversation_id=arguments.get("namespace"),
                    limit=arguments.get("limit", 10)
                )
                # Convert results to JSON-serializable format
                formatted_results = []
                for chunk in results:
                    formatted_results.append({
                        'id': chunk.id,
                        'conversation': chunk.conversation_id,
                        'summary': chunk.summary,
                        'neural_score': round(chunk.neural_score, 3),
                        'dynamic_score': round(chunk.dynamic_score, 3),
                        'combined_score': round(chunk.combined_score, 3),
                        'timestamp': chunk.timestamp,
                        'storage_tier': chunk.storage_tier,
                        'token_count': chunk.token_count or 0,
                        'metadata': chunk.metadata
                    })
                result = {
                    'success': True,
                    'results': formatted_results,
                    'total_found': len(formatted_results),
                    'query_time_ms': round((time.time() - start_time) * 1000, 1)
                }
                
            elif name == "memory_store":
                memory_id = memory_system.store_memory(
                    conversation_id=arguments.get("namespace", "default"),
                    text=arguments["value"],
                    metadata={
                        "key": arguments["key"],
                        **(arguments.get("metadata") or {})
                    }
                )
                result = {
                    'success': True,
                    'memory_id': memory_id,
                    'key': arguments["key"],
                    'store_time_ms': round((time.time() - start_time) * 1000, 1)
                }
                
            elif name == "memory_stats":
                stats = memory_system.get_system_stats()
                result = {
                    'success': True,
                    'stats': stats,
                    'namespace': arguments.get("namespace")
                }
                
            elif name == "index_project_files":
                stats = indexer.index_project(
                    max_files=arguments.get("max_files")
                )
                result = {
                    'success': True,
                    'indexed_files': stats.get('indexed_files', 0),
                    'total_chunks': stats.get('total_chunks', 0),
                    'index_time_ms': round((time.time() - start_time) * 1000, 1)
                }
                
            elif name == "search_project_files":
                chunks = indexer.search_project(
                    query=arguments["query"],
                    limit=arguments.get("limit", 10)
                )
                results = []
                for chunk in chunks:
                    if chunk.neural_score >= arguments.get("similarity_threshold", 0.0):
                        results.append({
                            'file_path': chunk.file_path,
                            'score': round(chunk.neural_score, 3),
                            'chunk_index': chunk.chunk_index,
                            'lines': f"{chunk.start_line}-{chunk.end_line}",
                            'content': chunk.content
                        })
                result = {
                    'success': True,
                    'results': results,
                    'total_found': len(results),
                    'search_time_ms': round((time.time() - start_time) * 1000, 1)
                }
                
            else:
                # Fall back to neural_flow_tools for complex operations
                from neural_flow_tools import familiarize_with_project
                
                if name == "familiarize_with_project":
                    result = familiarize_with_project(
                        depth=arguments.get("depth", "comprehensive"),
                        project_root=arguments.get("project_root")
                    )
                else:
                    result = {"success": False, "error": f"Unknown tool: {name}"}
        
        else:
            # Fallback to importing and using neural_flow_tools functions
            print(f"⚠️  Using fallback mode (systems not pre-loaded)", file=sys.stderr)
            
            from neural_flow_tools import (
                memory_query, memory_store, memory_stats,
                index_project_files, search_project_files,
                familiarize_with_project
            )
            
            if name == "memory_query":
                result = memory_query(
                    pattern=arguments["pattern"],
                    namespace=arguments.get("namespace"),
                    limit=arguments.get("limit", 10),
                    use_dynamic_scoring=arguments.get("use_dynamic_scoring", True)
                )
            elif name == "memory_store":
                result = memory_store(
                    key=arguments["key"],
                    value=arguments["value"],
                    namespace=arguments.get("namespace", "default"),
                    metadata=arguments.get("metadata")
                )
            elif name == "memory_stats":
                result = memory_stats(
                    namespace=arguments.get("namespace")
                )
            elif name == "index_project_files":
                result = index_project_files(
                    project_root=arguments.get("project_root"),
                    max_files=arguments.get("max_files"),
                    use_neural=arguments.get("use_neural", True)
                )
            elif name == "search_project_files":
                result = search_project_files(
                    query=arguments["query"],
                    limit=arguments.get("limit", 10),
                    similarity_threshold=arguments.get("similarity_threshold", 0.0)
                )
            elif name == "familiarize_with_project":
                result = familiarize_with_project(
                    depth=arguments.get("depth", "comprehensive"),
                    project_root=arguments.get("project_root")
                )
            else:
                result = {"success": False, "error": f"Unknown tool: {name}"}
        
        # Add performance metrics
        if isinstance(result, dict) and 'query_time_ms' not in result and 'store_time_ms' not in result:
            result['response_time_ms'] = round((time.time() - start_time) * 1000, 1)
        
        # Safe JSON serialization
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 1)
        }
        return [TextContent(
            type="text",
            text=json.dumps(error_result, indent=2)
        )]

async def main():
    """Run the MCP server"""
    try:
        # Server starts immediately, pre-loading happens in background
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
    finally:
        # Cleanup
        _executor.shutdown(wait=False)
        
        if _memory_system:
            try:
                _memory_system.close()
            except:
                pass
        if _indexer:
            try:
                _indexer.close()
            except:
                pass
        
        print("🔚 Server shutdown complete", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())