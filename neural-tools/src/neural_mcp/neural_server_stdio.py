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
import secrets
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Official MCP SDK for 2025-06-18 protocol (no namespace collision with neural_mcp)
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
DEFAULT_PROJECT_NAME = os.environ.get('PROJECT_NAME', 'default')
LIMITS = {
    "semantic_limit_max": 50,
    "graphrag_limit_max": 25,
    "max_hops_max": 3,
}


def get_instance_id() -> str:
    """
    Get or generate a unique instance ID for this MCP server instance.
    This enables isolation between different Claude instances.
    
    Returns a consistent ID for this MCP server process that persists
    throughout its lifetime but differs between Claude instances.
    """
    # Priority 1: Environment variable (if Claude provides it in future)
    instance_id = os.getenv('INSTANCE_ID')
    if instance_id:
        logger.info(f"🔐 Using provided instance ID: {instance_id}")
        return instance_id
    
    # Priority 2: Process-based ID (fallback)
    pid = os.getpid()
    ppid = os.getppid()
    
    # Priority 3: Hash of stdin/stdout file descriptors (unique per connection)
    try:
        stdin_stat = os.fstat(sys.stdin.fileno())
        stdout_stat = os.fstat(sys.stdout.fileno())
        unique_string = f"{pid}:{ppid}:{stdin_stat.st_ino}:{stdin_stat.st_dev}:{stdout_stat.st_ino}"
    except:
        # Fallback if file descriptors not available
        unique_string = f"{pid}:{ppid}:{datetime.now().isoformat()}"
    
    # Generate short, consistent hash
    instance_hash = hashlib.md5(unique_string.encode()).hexdigest()[:8]
    logger.info(f"🔐 Generated instance ID: {instance_hash} (pid:{pid}, ppid:{ppid})")
    return instance_hash


def _make_validation_error(
    tool: str,
    message: str,
    *,
    missing: Optional[List[str]] = None,
    invalid: Optional[List[Dict[str, str]]] = None,
    example: Optional[Dict[str, Any]] = None,
    received: Optional[Dict[str, Any]] = None,
    normalized: Optional[Dict[str, Any]] = None,
    hint: Optional[str] = None,
) -> str:
    payload: Dict[str, Any] = {
        "status": "error",
        "code": "validation_error",
        "tool": tool,
        "message": message,
        "missing": missing or [],
        "invalid": invalid or [],
        "example": example or {},
        "received": received or {},
    }
    if normalized is not None:
        payload["normalized_args"] = normalized
        payload["next_call"] = normalized
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, indent=2)

# Multi-project service instances - cached per project AND instance
class MultiProjectServiceState:
    """Holds persistent service state with instance-level and per-project isolation"""
    def __init__(self):
        # Instance-level isolation (Phase 1 of ADR-19)
        self.instance_id = get_instance_id()
        self.instance_containers = {}  # instance_id -> { project_containers, project_retrievers, ... }
        self.global_initialized = False
        
        # Initialize container for this instance
        self._init_instance_container()
        
    def _init_instance_container(self):
        """Initialize container for this instance"""
        if self.instance_id not in self.instance_containers:
            self.instance_containers[self.instance_id] = {
                'project_containers': {},
                'project_retrievers': {},
                'session_started': datetime.now(),
                'last_activity': datetime.now()
            }
            logger.info(f"🔐 Initialized instance container: {self.instance_id}")
    
    def _get_instance_data(self):
        """Get data container for current instance"""
        # Update last activity
        if self.instance_id in self.instance_containers:
            self.instance_containers[self.instance_id]['last_activity'] = datetime.now()
        return self.instance_containers[self.instance_id]
        
    def detect_project_from_path(self, file_path: str) -> str:
        """Extract project name from workspace path"""
        if not file_path:
            return DEFAULT_PROJECT_NAME
            
        # Handle /workspace/project-name/... format
        if file_path.startswith('/workspace/'):
            parts = file_path.strip('/').split('/')
            if len(parts) >= 2 and parts[1] != '':
                return parts[1]  # project name
        
        # Handle query context with file references
        if 'project-' in file_path:
            for part in file_path.split('/'):
                if part.startswith('project-'):
                    return part
                    
        return DEFAULT_PROJECT_NAME
    
    async def get_project_container(self, project_name: str):
        """Get or create ServiceContainer for specific project with lazy initialization and instance isolation"""
        instance_data = self._get_instance_data()
        project_containers = instance_data['project_containers']
        
        if project_name not in project_containers:
            logger.info(f"🏗️ [Instance {self.instance_id}] Creating service container for project: {project_name}")
            
            # Import services
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from servers.services.service_container import ServiceContainer
            
            container = ServiceContainer(project_name)
            # Don't initialize services yet - will happen on first use
            project_containers[project_name] = container
        
        # Lazy initialization - only initialize when actually needed
        container = project_containers[project_name]
        if not container.initialized:
            logger.info(f"⚡ [Instance {self.instance_id}] Lazy-initializing services for project: {project_name} (first tool use)")
            await container.initialize_all_services()
            
            # Initialize L9 connection pools and session management
            await container.initialize_connection_pools()
            if container.session_manager:
                await container.session_manager.initialize()
            
            # Initialize Phase 3 security and monitoring if available
            if hasattr(container, 'initialize_security_services'):
                await container.initialize_security_services()
        
        return container
    
    async def get_project_retriever(self, project_name: str):
        """Get or create HybridRetriever for specific project with instance isolation"""
        instance_data = self._get_instance_data()
        project_retrievers = instance_data['project_retrievers']
        
        if project_name not in project_retrievers:
            container = await self.get_project_container(project_name)
            if container.neo4j and container.qdrant:
                from servers.services.hybrid_retriever import HybridRetriever
                project_retrievers[project_name] = HybridRetriever(container)
                logger.info(f"🔍 [Instance {self.instance_id}] Created HybridRetriever for project: {project_name}")
            else:
                project_retrievers[project_name] = None
        return project_retrievers[project_name]


state = MultiProjectServiceState()
server = Server("l9-neural-enhanced")


# Removed initialize_services() - services now initialized lazily on first use
# This prevents blocking the MCP handshake with 30+ second service initialization


async def get_project_context(arguments: Dict[str, Any]):
    # Derive project name (simple heuristic)
    project_name = arguments.get('project') or DEFAULT_PROJECT_NAME
    container = await state.get_project_container(project_name)
    retriever = await state.get_project_retriever(project_name)
    return project_name, container, retriever


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    # Tools are statically defined - no initialization needed for listing
    return [
        types.Tool(
            name="neural_system_status",
            description="Get comprehensive neural system status and health",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="semantic_code_search",
            description=(
                "Search code by meaning using semantic embeddings.\n"
                "Usage: {\"query\": \"natural language\", \"limit\": 10}\n"
                f"limit: 1..{LIMITS['semantic_limit_max']} (default 10)."
            ),
            inputSchema={
                "type": "object",
                "title": "Semantic Code Search",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 3,
                        "description": "Plain text query. Example: 'how to start server'"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": LIMITS["semantic_limit_max"],
                        "default": 10,
                        "description": f"Results to return (1..{LIMITS['semantic_limit_max']})."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="graphrag_hybrid_search",
            description=(
                "Hybrid search with graph context.\n"
                "Usage: {\"query\": \"...\", \"limit\": 5, \"include_graph_context\": true, \"max_hops\": 2}"
            ),
            inputSchema={
                "type": "object",
                "title": "GraphRAG Hybrid Search",
                "properties": {
                    "query": {"type": "string", "minLength": 3, "description": "Search text."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": LIMITS["graphrag_limit_max"], "default": 5},
                    "include_graph_context": {"type": "boolean", "default": True},
                    "max_hops": {"type": "integer", "minimum": 0, "maximum": LIMITS["max_hops_max"], "default": 2}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="project_understanding",
            description=(
                "Get condensed project understanding.\n"
                "Usage: {\"scope\": \"full|summary|files|services\"}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["full", "summary", "files", "services"], "default": "full"}
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="indexer_status",
            description="Get neural indexer sidecar status and metrics",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="reindex_path",
            description=(
                "Enqueue a file or folder for reindexing. Path is relative to workspace unless absolute.\n"
                "Usage: {\"path\": \"src/\", \"recursive\": true}"
            ),
            inputSchema={
                "type": "object",
                "title": "Reindex Path",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "description": "File or directory to reindex."},
                    "recursive": {"type": "boolean", "default": True}
                },
                "required": ["path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="neural_tools_help",
            description=(
                "Show usage examples and constraints for all neural tools."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    # Services will be initialized lazily on first actual tool use
    instance_id = state.instance_id
    
    # Log tool call with instance ID for debugging
    logger.info(f"🔧 [Instance {instance_id}] Tool call: {name}")

    # L9 2025: Session-aware tool execution
    try:
        # Generate or extract session ID (simplified for now)
        session_id = arguments.get('session_id') or secrets.token_urlsafe(16)
        
        # Get project container with connection pooling
        project_name, container, retriever = await get_project_context(arguments)
        
        # Get or create session
        if container.session_manager:
            session = await container.session_manager.get_or_create_session(session_id)
            
            # Check rate limits
            if not await session.check_rate_limit():
                return [types.TextContent(type="text", text=json.dumps({
                    "status": "error",
                    "code": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Current limit: {session.resource_limits['queries_per_minute']} queries/minute.",
                    "session_id": session_id[:8] + "..."
                }, indent=2))]
                
            # Clean up expired sessions periodically
            await container.session_manager.cleanup_expired_sessions()
        else:
            session = None
        
        # Execute tools with session context
        if name == "neural_system_status":
            return await neural_system_status_impl()
        elif name == "semantic_code_search":
            # Validate and normalize
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            if not isinstance(query, str) or len(query.strip()) < 3:
                return [types.TextContent(type="text", text=_make_validation_error(
                    "semantic_code_search",
                    "Missing or too-short 'query' (string, minLength 3).",
                    missing=["query"] if not query else [],
                    example={"query": "how to start server", "limit": 10},
                    received=arguments
                ))]
            # Coerce and clamp limit
            try:
                limit = int(limit)
            except Exception:
                limit = 10
            if limit < 1:
                limit = 1
            if limit > LIMITS["semantic_limit_max"]:
                limit = LIMITS["semantic_limit_max"]
            return await semantic_code_search_impl(query.strip(), limit)
        elif name == "graphrag_hybrid_search":
            query = arguments.get("query", "")
            if not isinstance(query, str) or len(query.strip()) < 3:
                return [types.TextContent(type="text", text=_make_validation_error(
                    "graphrag_hybrid_search",
                    "Missing or too-short 'query' (string, minLength 3).",
                    missing=["query"] if not query else [],
                    example={"query": "find graph relationships", "limit": 5, "include_graph_context": True, "max_hops": 2},
                    received=arguments
                ))]
            limit = arguments.get("limit", 5)
            try:
                limit = int(limit)
            except Exception:
                limit = 5
            limit = max(1, min(limit, LIMITS["graphrag_limit_max"]))
            include_graph_context = bool(arguments.get("include_graph_context", True))
            max_hops = arguments.get("max_hops", 2)
            try:
                max_hops = int(max_hops)
            except Exception:
                max_hops = 2
            max_hops = max(0, min(max_hops, LIMITS["max_hops_max"]))
            return await graphrag_hybrid_search_impl(query.strip(), limit, include_graph_context, max_hops)
        elif name == "project_understanding":
            scope = arguments.get("scope", "full")
            allowed = ["full", "summary", "files", "services"]
            if scope not in allowed:
                return [types.TextContent(type="text", text=_make_validation_error(
                    "project_understanding",
                    "Invalid 'scope'. Must be one of: full, summary, files, services.",
                    invalid=[{"field": "scope", "reason": f"allowed: {allowed}"}],
                    example={"scope": "summary"},
                    received=arguments,
                    normalized={"scope": "full"}
                ))]
            return await project_understanding_impl(scope)
        elif name == "indexer_status":
            return await indexer_status_impl()
        elif name == "reindex_path":
            p = arguments.get("path", "")
            if not isinstance(p, str) or len(p.strip()) == 0:
                return [types.TextContent(type="text", text=_make_validation_error(
                    "reindex_path",
                    "Missing required field: 'path' (string, minLength 1).",
                    missing=["path"],
                    example={"path": "src/", "recursive": True},
                    received=arguments
                ))]
            return await reindex_path_impl(p)
        elif name == "neural_tools_help":
            return await neural_tools_help_impl()
        else:
            return [types.TextContent(type="text", text=json.dumps({
                "error": f"Unknown tool: {name}",
                "_debug": {"instance_id": instance_id}
            }))]
    except Exception as e:
        logger.error(f"[Instance {instance_id}] Tool call failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e),
            "_debug": {"instance_id": instance_id}
        }))]


async def semantic_code_search_impl(query: str, limit: int) -> List[types.TextContent]:
    try:
        project_name, container, _ = await get_project_context({})
        embeddings = await container.nomic.get_embeddings([query])
        query_vector = embeddings[0]
        # Use Qdrant client wrapper
        collection_name = f"project_{project_name}_code"
        try:
            search_results = await container.qdrant.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return await _fallback_neo4j_search(query, limit)

        formatted = []
        for hit in search_results:
            if isinstance(hit, dict):
                payload = hit.get("payload", {})
                score = hit.get("score", 0.0)
            else:
                # Fallback for client object shape
                payload = getattr(hit, 'payload', {}) or {}
                score = float(getattr(hit, 'score', 0.0))
            content = payload.get("content", "")
            formatted.append({
                "score": float(score),
                "file_path": payload.get("file_path", ""),
                "snippet": (content[:200] + "...") if len(content) > 200 else content
            })

        return [types.TextContent(type="text", text=json.dumps({
            "status": "success",
            "query": query,
            "results": formatted,
            "total_found": len(formatted)
        }, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def _fallback_neo4j_search(query: str, limit: int) -> List[types.TextContent]:
    try:
        project_name, container, _ = await get_project_context({})
        cypher = """
        MATCH (c:CodeChunk)
        WHERE c.content IS NOT NULL AND (c.content CONTAINS $query OR c.file_path CONTAINS $query)
        RETURN c.file_path as file_path, c.content as content LIMIT $limit
        """
        result = await container.neo4j.execute_cypher(cypher, {"query": query, "limit": limit})
        formatted = []
        data = result.get('data') if isinstance(result, dict) else result
        if data:
            for r in data:
                content = r.get("content", "")
                formatted.append({
                    "file_path": r.get("file_path", ""),
                    "snippet": (content[:200] + "...") if len(content) > 200 else content,
                    "fallback": True
                })
        return [types.TextContent(type="text", text=json.dumps({
            "status": "success",
            "message": "Results from Neo4j fallback search (Qdrant unavailable)",
            "query": query,
            "results": formatted,
            "total_found": len(formatted),
            "fallback_mode": True
        }, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def graphrag_hybrid_search_impl(query: str, limit: int, include_graph_context: bool, max_hops: int) -> List[types.TextContent]:
    try:
        project_name, _, retriever = await get_project_context({})
        if not retriever:
            return [types.TextContent(type="text", text=json.dumps({"error": "GraphRAG not available"}))]

        results = await retriever.find_similar_with_context(query, limit, include_graph_context, max_hops)
        formatted_results = []
        for res in results:
            entry = {
                "score": res.get("score", 0),
                "file": res.get("file_path", ""),
                "lines": f"{res.get('start_line', 0)}-{res.get('end_line', 0)}",
                "content": (res.get("content", "")[:200] + "...")
            }
            if include_graph_context and res.get("graph_context"):
                ctx = res["graph_context"]
                entry["graph_context"] = {
                    "imports": ctx.get("imports", []),
                    "imported_by": ctx.get("imported_by", []),
                    "related_chunks": len(ctx.get("related_chunks", []))
                }
            formatted_results.append(entry)
        return [types.TextContent(type="text", text=json.dumps({
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def neural_system_status_impl() -> List[types.TextContent]:
    """Get comprehensive neural system status and health - REAL service connections only"""
    try:
        project_name, container, retriever = await get_project_context({})
        
        # Check REAL service connections
        status = {
            "project": project_name,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "neo4j": {
                    "connected": container.neo4j is not None,
                    "type": "REAL GraphDatabase connection" if container.neo4j else "Not connected",
                    "status": "healthy" if container.neo4j else "unavailable"
                },
                "qdrant": {
                    "connected": container.qdrant is not None,
                    "type": "REAL Qdrant vector database" if container.qdrant else "Not connected", 
                    "status": "healthy" if container.qdrant else "unavailable"
                },
                "nomic": {
                    "connected": container.nomic is not None,
                    "type": "REAL Nomic embedding service" if container.nomic else "Not connected",
                    "status": "healthy" if container.nomic else "unavailable"
                }
            },
            "hybrid_retriever": {
                "available": retriever is not None,
                "status": "active" if retriever else "inactive"
            },
            "indexing_status": "REAL services - NO MOCKS!"
        }
        
        # Test actual connections
        if container.neo4j:
            try:
                # Neo4j service wrapper - use async query
                if hasattr(container.neo4j, 'client') and container.neo4j.client:
                    result = await container.neo4j.client.execute_query(
                        "MATCH (n) RETURN count(n) as node_count"
                    )
                    if result["status"] == "success" and result["result"]:
                        node_count = result["result"][0]["node_count"]
                        status["services"]["neo4j"]["node_count"] = node_count
                    else:
                        status["services"]["neo4j"]["node_count"] = 0
                else:
                    status["services"]["neo4j"]["node_count"] = "N/A - client not initialized"
            except Exception as e:
                status["services"]["neo4j"]["error"] = str(e)
                
        if container.qdrant:
            try:
                # get_collections returns a list of strings directly
                collections = await container.qdrant.get_collections()
                status["services"]["qdrant"]["collections"] = collections
                status["services"]["qdrant"]["collection_count"] = len(collections)
            except Exception as e:
                status["services"]["qdrant"]["error"] = str(e)
        
        return [types.TextContent(type="text", text=json.dumps({"status": "success", "system_status": status}, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def project_understanding_impl(scope: str = "full") -> List[types.TextContent]:
    try:
        understanding = {
            "project": DEFAULT_PROJECT_NAME,
            "timestamp": datetime.now().isoformat(),
            "scope": scope,
            "protocol": "2025-06-18",
            "transport": "stdio"
        }
        return [types.TextContent(type="text", text=json.dumps({"status": "success", "understanding": understanding}, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def neural_tools_help_impl() -> List[types.TextContent]:
    """Return usage guidance, examples, and constraints for all tools."""
    try:
        help_payload = {
            "tools": {
                "semantic_code_search": {
                    "usage": {"query": "how to start server", "limit": 10},
                    "constraints": {"query_minLength": 3, "limit": [1, LIMITS["semantic_limit_max"]]},
                    "returns": {"status": "success", "results": "[]", "total_found": 0}
                },
                "graphrag_hybrid_search": {
                    "usage": {"query": "find relationships", "limit": 5, "include_graph_context": True, "max_hops": 2},
                    "constraints": {"query_minLength": 3, "limit": [1, LIMITS["graphrag_limit_max"]], "max_hops": [0, LIMITS["max_hops_max"]]},
                    "returns": {"results_count": 0, "results": "[]"}
                },
                "project_understanding": {
                    "usage": {"scope": "summary"},
                    "allowed_scopes": ["full", "summary", "files", "services"],
                    "returns": {"status": "success", "understanding": {"project": "..."}}
                },
                "indexer_status": {
                    "usage": {},
                    "returns": {"status": "success", "indexer_status": "healthy|unhealthy|disconnected|error"}
                },
                "reindex_path": {
                    "usage": {"path": "src/", "recursive": True},
                    "constraints": {"path_minLength": 1},
                    "returns": {"status": "enqueued"}
                }
            }
        }
        return [types.TextContent(type="text", text=json.dumps(help_payload, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

async def indexer_status_impl() -> List[types.TextContent]:
    """Get neural indexer sidecar status and metrics via HTTP API"""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try to connect to indexer sidecar on localhost:48080
            response = await client.get("http://localhost:48080/status")
            
            if response.status_code == 200:
                status_data = response.json()
                
                # Format the response nicely
                formatted_status = {
                    "indexer_status": "healthy",
                    "timestamp": status_data.get("timestamp", "unknown"),
                    "metrics": {
                        "queue_depth": status_data.get("queue_depth", 0),
                        "files_processed": status_data.get("files_processed", 0),
                        "degraded_mode": status_data.get("degraded_mode", False)
                    },
                    "sidecar_connection": "connected"
                }
                
                return [types.TextContent(
                    type="text", 
                    text=json.dumps(formatted_status, indent=2)
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "indexer_status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "sidecar_connection": "failed"
                    }, indent=2)
                )]
                
    except httpx.ConnectError:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "indexer_status": "disconnected",
                "error": "Could not connect to indexer sidecar on localhost:48080",
                "sidecar_connection": "failed",
                "note": "Indexer sidecar may not be running in container mode"
            }, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "indexer_status": "error",
                "error": str(e),
                "sidecar_connection": "unknown"
            }, indent=2)
        )]


async def reindex_path_impl(path: str) -> List[types.TextContent]:
    """Trigger reindexing of a specific path via indexer sidecar HTTP API"""
    import httpx
    
    if not path:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "Path is required for reindexing"}, indent=2)
        )]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Send reindex request to sidecar
            response = await client.post(
                "http://localhost:48080/reindex-path",
                params={"path": path}
            )
            
            if response.status_code == 200:
                result = response.json()
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": f"Reindex request queued for path: {path}",
                        "details": result
                    }, indent=2)
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "path": path
                    }, indent=2)
                )]
                
    except httpx.ConnectError:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "failed",
                "error": "Could not connect to indexer sidecar on localhost:48080",
                "path": path,
                "note": "Indexer sidecar may not be running in container mode"
            }, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "error": str(e),
                "path": path
            }, indent=2)
        )]


def cleanup_resources():
    """Clean up resources on server shutdown (called when stdin closes)"""
    logger.info("🔄 Cleaning up resources (stdin closed)")
    
    # Remove PID file if it exists
    pid = os.getpid()
    pid_file = Path(f"/tmp/mcp_pids/mcp_{pid}.pid")
    if pid_file.exists():
        try:
            pid_file.unlink()
            logger.info(f"✅ Removed PID file: {pid_file}")
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")
    
    # Close any active connections
    if 'state' in globals() and state:
        for project_state in state.project_states.values():
            if project_state.container:
                try:
                    # Close Neo4j connections
                    if hasattr(project_state.container, 'neo4j_driver'):
                        project_state.container.neo4j_driver.close()
                    # Close Qdrant connections
                    if hasattr(project_state.container, 'qdrant_client'):
                        project_state.container.qdrant_client.close()
                except Exception as e:
                    logger.error(f"Error closing connections: {e}")
    
    logger.info("👋 MCP server shutdown complete")


async def run():
    """
    Main entry point for MCP server following 2025-06-18 specification.
    
    Per spec:
    - Server reads from stdin, writes to stdout
    - Logs go to stderr only
    - Client initiates shutdown by closing stdin
    - No signal handling in server (client handles termination)
    """
    logger.info(f"🚀 Starting L9 Neural MCP Server (STDIO Transport)")
    logger.info(f"🔐 Instance ID: {state.instance_id}")
    
    # Create PID file for tracking (helps with orphan cleanup)
    pid = os.getpid()
    pid_dir = Path("/tmp/mcp_pids")
    pid_dir.mkdir(exist_ok=True)
    pid_file = pid_dir / f"mcp_{pid}.pid"
    pid_file.write_text(str(pid))
    logger.info(f"📝 PID file created: {pid_file}")
    
    try:
        # CRITICAL FIX: Don't initialize services here - wait until after handshake
        # Services will be initialized lazily on first tool use
        logger.info("⏳ Delaying service initialization until after MCP handshake")
        
        # MCP STDIO server handles the transport layer
        # It will detect when stdin closes (client disconnect)
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="l9-neural-enhanced",
                    server_version="2.0.0-stdio",
                    capabilities=server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
                ),
            )
    except (EOFError, BrokenPipeError):
        # Normal shutdown - stdin closed by client
        logger.info("📥 Client disconnected (stdin closed)")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise
    finally:
        cleanup_resources()
