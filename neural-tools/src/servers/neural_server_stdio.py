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

# Official MCP SDK for 2025-06-18 protocol
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Configure logging to stderr (NEVER to stdout for STDIO transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Critical: Log to stderr, not stdout
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'default')

# Service instances - persistent throughout server lifetime
class ServiceState:
    """Holds persistent service state"""
    def __init__(self):
        self.qdrant_client = None
        self.neo4j_client = None
        self.nomic_client = None
        self.container = None  # Store full container for HybridRetriever
        self.hybrid_retriever = None  # GraphRAG retriever
        self.initialized = False
        
service_state = ServiceState()

# Initialize MCP Server
server = Server("l9-neural-enhanced")

async def initialize_services():
    """Initialize services once at server startup"""
    if service_state.initialized:
        return True
        
    try:
        logger.info("🚀 Initializing L9 Neural Services")
        
        # Import and initialize services
        sys.path.insert(0, str(Path(__file__).parent.parent))  # Add src to path first
        from services.service_container import ServiceContainer
        from services.hybrid_retriever import HybridRetriever
        
        container = ServiceContainer(PROJECT_NAME)
        result = await container.initialize_all_services()
        
        # Store service clients
        service_state.container = container
        service_state.qdrant_client = container.qdrant
        service_state.neo4j_client = container.neo4j
        service_state.nomic_client = container.nomic
        
        # Initialize GraphRAG HybridRetriever
        if container.neo4j and container.qdrant:
            service_state.hybrid_retriever = HybridRetriever(container)
            logger.info("✅ GraphRAG HybridRetriever initialized")
        else:
            logger.warning("⚠️ GraphRAG disabled - Neo4j or Qdrant unavailable")
        
        service_state.initialized = True
        
        logger.info(f"✅ Services initialized: {result.get('overall_health', 'unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

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
        ),
        types.Tool(
            name="project_indexer",
            description="Index project files into Neo4j graph and Qdrant vectors for GraphRAG search",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to index (relative to /app/project or absolute). Defaults to entire project.",
                        "default": "/app/project"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Index recursively through subdirectories",
                        "default": True
                    },
                    "clear_existing": {
                        "type": "boolean",
                        "description": "Clear existing index data before indexing",
                        "default": False
                    },
                    "file_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File patterns to index (e.g., .py, .js, .md). Defaults to common code files.",
                        "default": None
                    },
                    "force_reindex": {
                        "type": "boolean",
                        "description": "Force re-indexing even if files haven't changed",
                        "default": False
                    }
                },
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
        elif name == "project_indexer":
            return await project_indexer_impl(
                arguments.get("path", "/app/project"),
                arguments.get("recursive", True),
                arguments.get("clear_existing", False),
                arguments.get("file_patterns", None),
                arguments.get("force_reindex", False)
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
        services_status = {
            "qdrant": service_state.qdrant_client is not None,
            "neo4j": service_state.neo4j_client is not None,
            "nomic": service_state.nomic_client is not None,
            "graphrag": service_state.hybrid_retriever is not None
        }
        
        result = {
            "status": "healthy" if all(services_status.values()) else "degraded",
            "project": PROJECT_NAME,
            "version": "l9-neural-enhanced-graphrag",
            "protocol": "2025-06-18",
            "transport": "stdio",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status,
            "healthy_services": sum(services_status.values()),
            "total_services": len(services_status)
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
        if not service_state.nomic_client or not service_state.qdrant_client:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": "Required services not initialized"
            }))]
        
        # Generate embedding for query
        embeddings = await service_state.nomic_client.get_embeddings([query])
        query_vector = embeddings[0]
        
        collection_name = f"project_{PROJECT_NAME}_code"
        
        # Search vectors
        search_results = await service_state.qdrant_client.search_vectors(
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
        if not service_state.hybrid_retriever:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "GraphRAG not available - Neo4j or Qdrant not initialized"})
            )]
        
        results = await service_state.hybrid_retriever.find_similar_with_context(
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
        if not service_state.hybrid_retriever:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "GraphRAG not available"})
            )]
        
        impact = await service_state.hybrid_retriever.analyze_impact(
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
        if not service_state.hybrid_retriever:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "GraphRAG not available"})
            )]
        
        deps = await service_state.hybrid_retriever.find_dependencies(
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
        if not service_state.hybrid_retriever:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "GraphRAG not available"})
            )]
        
        related = await service_state.hybrid_retriever.find_related_by_graph(
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

async def project_indexer_impl(path: str, recursive: bool, clear_existing: bool, 
                               file_patterns: Optional[List[str]], force_reindex: bool) -> List[types.TextContent]:
    """
    Systematically index project files into Neo4j and Qdrant for GraphRAG
    Uses the IncrementalIndexer to handle all file types with intelligent chunking
    """
    try:
        from pathlib import Path
        import sys
        import importlib.util
        
        # Ensure the correct path is in sys.path for nested imports
        neural_src = '/app/project/neural-tools/src'
        if neural_src not in sys.path:
            sys.path.insert(0, neural_src)
        
        # Also add the servers directory for relative imports
        servers_dir = '/app/project/neural-tools/src/servers'
        if servers_dir not in sys.path:
            sys.path.insert(0, servers_dir)
        
        # Now use importlib to load the module
        indexer_path = "/app/project/neural-tools/src/servers/services/indexer_service.py"
        spec = importlib.util.spec_from_file_location("indexer_service", indexer_path)
        if spec and spec.loader:
            indexer_module = importlib.util.module_from_spec(spec)
            # Add the module to sys.modules before exec to allow nested imports
            sys.modules["indexer_service"] = indexer_module
            spec.loader.exec_module(indexer_module)
            IncrementalIndexer = indexer_module.IncrementalIndexer
        else:
            raise ImportError(f"Could not load indexer from {indexer_path}")
        
        # Validate path
        index_path = Path(path)
        if not index_path.exists():
            # Try relative to /app/project if not absolute
            if not index_path.is_absolute():
                index_path = Path("/app/project") / path
                if not index_path.exists():
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"Path not found: {path}",
                            "tried_paths": [str(Path(path)), str(index_path)]
                        })
                    )]
        
        logger.info(f"Starting indexing of {index_path}")
        
        # Clear existing data if requested
        if clear_existing and service_state.neo4j_client:
            logger.info("Clearing existing index data...")
            await service_state.neo4j_client.execute_cypher(
                "MATCH (n) WHERE n:Document OR n:Chunk OR n:CodeChunk OR n:Class OR n:Method OR n:Function DETACH DELETE n"
            )
            # Note: Clearing Qdrant would require recreating collections
        
        # Create indexer instance
        indexer = IncrementalIndexer(str(index_path), PROJECT_NAME)
        
        # Override file patterns if specified
        if file_patterns:
            indexer.watch_patterns = set(file_patterns)
        
        # Initialize services using existing container
        indexer.container = service_state.container
        indexer.services_initialized = True
        
        # Set force reindex flag
        if force_reindex:
            indexer.file_hashes.clear()  # Clear hash tracking to force reindex
        
        # Perform initial indexing (not watching for changes)
        stats = {
            "files_indexed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "errors": 0,
            "file_patterns": list(indexer.watch_patterns),
            "path": str(index_path)
        }
        
        # Walk the directory and index files
        if recursive:
            file_list = []
            for pattern in indexer.watch_patterns:
                file_list.extend(index_path.rglob(f"*{pattern}"))
        else:
            file_list = []
            for pattern in indexer.watch_patterns:
                file_list.extend(index_path.glob(f"*{pattern}"))
        
        # Filter out ignored patterns
        filtered_files = []
        for file_path in file_list:
            should_ignore = False
            for ignore_pattern in indexer.ignore_patterns:
                if ignore_pattern in str(file_path):
                    should_ignore = True
                    break
            if not should_ignore:
                filtered_files.append(file_path)
        
        logger.info(f"Found {len(filtered_files)} files to index")
        
        # Index each file
        for file_path in filtered_files:
            try:
                # Check file size
                if file_path.stat().st_size > indexer.max_file_size:
                    logger.warning(f"Skipping large file: {file_path}")
                    stats["files_skipped"] += 1
                    continue
                
                # Index the file
                await indexer.index_file(str(file_path))
                stats["files_indexed"] += 1
                
            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                stats["errors"] += 1
        
        # Get final statistics from indexer
        stats["chunks_created"] = indexer.metrics.get("chunks_created", 0)
        stats["neo4j_available"] = not indexer.degraded_mode.get("neo4j", True)
        stats["qdrant_available"] = not indexer.degraded_mode.get("qdrant", True)
        stats["nomic_available"] = not indexer.degraded_mode.get("nomic", True)
        
        # Query for totals
        if service_state.neo4j_client:
            try:
                result = await service_state.neo4j_client.execute_cypher(
                    "MATCH (d:Document) RETURN count(d) as doc_count"
                )
                if result:
                    stats["total_documents_in_graph"] = result[0].get("doc_count", 0)
                
                result = await service_state.neo4j_client.execute_cypher(
                    "MATCH (c:Chunk) RETURN count(c) as chunk_count"
                )
                if result:
                    stats["total_chunks_in_graph"] = result[0].get("chunk_count", 0)
                    
                result = await service_state.neo4j_client.execute_cypher(
                    "MATCH (n) WHERE n:Class OR n:Method OR n:Function RETURN count(n) as entity_count"
                )
                if result:
                    stats["total_entities_in_graph"] = result[0].get("entity_count", 0)
                    
            except Exception as e:
                logger.error(f"Error getting graph statistics: {e}")
        
        logger.info(f"Indexing complete: {stats}")
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "message": f"Indexed {stats['files_indexed']} files successfully",
                "statistics": stats
            }, indent=2)
        )]
        
    except Exception as e:
        logger.error(f"Project indexer error: {e}")
        import traceback
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        )]

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)