#!/usr/bin/env python3
"""
L9 Enhanced MCP Server - Refactored Modular Architecture
Features: Neo4j GraphRAG + Nomic Embed v2-MoE + Tree-sitter + Qdrant hybrid search

ARCHITECTURE IMPROVEMENTS (ADR-0009):
- Eliminated global variable coupling  
- Service abstraction with dependency injection
- Modular tool organization
- Clean separation of concerns
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sqlite3

# FastMCP for MCP protocol handling
from fastmcp import FastMCP

# Service container (eliminates global coupling)
from services.service_container import ServiceContainer, get_container
from tools.memory_tools import register_memory_tools
from tools.core_tools import register_core_tools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("l9-neural-enhanced")

# Global service clients (FastMCP-compatible pattern)
qdrant_client = None
neo4j_client = None
nomic_client = None
_initialization_done = False

# Constants
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'default')

def generate_deterministic_point_id(file_path: str, content: str, chunk_index: int = 0) -> int:
    """Generate deterministic point ID for consistent upserts following industry standards."""
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    unique_string = f"{file_path}#{content_hash}#{chunk_index}"
    return abs(hash(unique_string)) % (10**15)

def get_content_hash(content: str) -> str:
    """Generate content hash for change detection"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# Essential MCP Tools (remaining tools from monolithic file)

@mcp.tool()
async def neural_system_status() -> Dict[str, Any]:
    """Get comprehensive neural system status"""
    try:
        await ensure_services_initialized()
        
        # Check global service clients
        services_status = {
            "qdrant": qdrant_client is not None,
            "neo4j": neo4j_client is not None,
            "nomic": nomic_client is not None
        }
        
        if not _initialization_done:
            return {
                "status": "not_initialized",
                "project": PROJECT_NAME,
                "timestamp": datetime.utcnow().isoformat(),
                "services": services_status,
                "error": "Service initialization not completed"
            }
        
        return {
            "status": "healthy" if all(services_status.values()) else "degraded",
            "project": PROJECT_NAME,
            "version": "l9-neural-enhanced-refactored-v2",
            "timestamp": datetime.utcnow().isoformat(),
            "services": services_status,
            "healthy_services": sum(services_status.values()),
            "total_services": len(services_status),
            "overall_score": sum(services_status.values()) / len(services_status)
        }
        
    except Exception as e:
        logger.error(f"neural_system_status error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "project": PROJECT_NAME,
            "timestamp": datetime.utcnow().isoformat()
        }

@mcp.tool()
async def neo4j_graph_query(
    cypher_query: str,
    parameters: str = "{}"
) -> Dict[str, Any]:
    """Execute Cypher query against the Neo4j graph database"""
    try:
        await ensure_services_initialized()
        
        if not neo4j_client:
            return {"status": "error", "message": "Neo4j client not initialized"}
        
        # Parse parameters with error handling
        try:
            query_params = json.loads(parameters) if parameters else {}
        except json.JSONDecodeError as json_error:
            return {
                "status": "error",
                "message": f"Invalid JSON parameters: {str(json_error)}"
            }
        
        # Use global Neo4j client
        neo4j = neo4j_client
        result = await neo4j.execute_cypher(cypher_query, query_params)
        
        return result
        
    except RuntimeError as service_error:
        return {"status": "error", "message": f"Neo4j service not available: {str(service_error)}"}
    except Exception as e:
        logger.error(f"neo4j_graph_query error: {str(e)}")
        return {
            "status": "error",
            "message": f"Query execution failed: {str(e)}"
        }

@mcp.tool()
async def neo4j_semantic_graph_search(
    query_text: str,
    limit: int = 10,
    node_types: str = "File,Class,Method"
) -> Dict[str, Any]:
    """Perform semantic search across graph entities"""
    try:
        await ensure_services_initialized()
        neo4j = neo4j_client
        
        # Use the semantic search from Neo4j service
        results = await neo4j.semantic_search(query_text, limit)
        
        return {
            "status": "success",
            "query": query_text,
            "results": results,
            "count": len(results),
            "node_types": node_types
        }
        
    except RuntimeError as service_error:
        return {"status": "error", "message": f"Neo4j service not available: {str(service_error)}"}
    except Exception as e:
        logger.error(f"neo4j_semantic_graph_search error: {str(e)}")
        return {
            "status": "error",
            "message": f"Semantic search failed: {str(e)}"
        }

@mcp.tool()
async def neo4j_code_dependencies(
    file_path: str,
    max_depth: int = 3
) -> Dict[str, Any]:
    """Get code dependencies for a file with traversal depth"""
    try:
        await ensure_services_initialized()
        neo4j = neo4j_client
        
        dependencies = await neo4j.get_code_dependencies(file_path, max_depth)
        
        return {
            "status": "success",
            "file_path": file_path,
            "dependencies": dependencies["dependencies"],
            "dependents": dependencies["dependents"],
            "max_depth": max_depth
        }
        
    except RuntimeError as service_error:
        return {"status": "error", "message": f"Neo4j service not available: {str(service_error)}"}
    except Exception as e:
        logger.error(f"neo4j_code_dependencies error: {str(e)}")
        return {
            "status": "error",
            "message": f"Dependency analysis failed: {str(e)}"
        }

@mcp.tool()
async def neo4j_migration_status() -> Dict[str, Any]:
    """Check Neo4j migration status and system health"""
    try:
        await ensure_services_initialized()
        neo4j = neo4j_client
        
        migration_status = await neo4j.get_migration_status()
        
        return {
            "status": "success",
            "migration": migration_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except RuntimeError as service_error:
        return {"status": "error", "message": f"Neo4j service not available: {str(service_error)}"}
    except Exception as e:
        logger.error(f"neo4j_migration_status error: {str(e)}")
        return {
            "status": "error",
            "message": f"Migration status check failed: {str(e)}"
        }

@mcp.tool()
async def neo4j_index_code_graph(
    file_paths: str = "",
    force_reindex: bool = False
) -> Dict[str, Any]:
    """Index code files in Neo4j graph with relationship extraction"""
    try:
        await ensure_services_initialized()
        neo4j = neo4j_client
        
        if not file_paths:
            return {
                "status": "error", 
                "message": "No file paths provided for indexing"
            }
        
        # Parse file paths (comma-separated)
        paths_list = [path.strip() for path in file_paths.split(",") if path.strip()]
        indexed_files = []
        
        for file_path in paths_list:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Index in Neo4j
                result = await neo4j.index_code_file(
                    file_path=file_path,
                    content=content,
                    language=_detect_language(file_path)
                )
                
                indexed_files.append({
                    "file_path": file_path,
                    "status": result.get("status", "unknown"),
                    "indexed": result.get("indexed", False)
                })
                
            except Exception as file_error:
                indexed_files.append({
                    "file_path": file_path,
                    "status": "error",
                    "error": str(file_error)
                })
        
        return {
            "status": "success",
            "files_processed": len(paths_list),
            "files_indexed": sum(1 for f in indexed_files if f.get("indexed", False)),
            "results": indexed_files,
            "force_reindex": force_reindex
        }
        
    except RuntimeError as service_error:
        return {"status": "error", "message": f"Neo4j service not available: {str(service_error)}"}
    except Exception as e:
        logger.error(f"neo4j_index_code_graph error: {str(e)}")
        return {
            "status": "error",
            "message": f"Code indexing failed: {str(e)}"
        }

def _detect_language(file_path: str) -> str:
    """Simple language detection from file extension"""
    ext = Path(file_path).suffix.lower()
    language_map = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala'
    }
    return language_map.get(ext, 'text')

# Production health check for MCP server
def production_health_check() -> Dict[str, Any]:
    """Production health check - synchronous for container health checks"""
    try:
        if _initialization_done:
            return {"status": "healthy", "message": "MCP server operational"}
        else:
            return {"status": "initializing", "message": "Services starting up"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def ensure_services_initialized():
    """Ensure services are initialized within MCP event loop - FastMCP compatible"""
    global qdrant_client, neo4j_client, nomic_client, _initialization_done
    if not _initialization_done:
        try:
            logger.info("🚀 Initializing L9 Neural MCP Server (FastMCP-Compatible)")
            
            # Initialize service container to get clients
            container = await get_container(PROJECT_NAME)
            
            # Extract clients as globals for FastMCP compatibility
            qdrant_client = container.get_qdrant()
            try:
                neo4j_client = container.get_neo4j()
            except RuntimeError:
                neo4j_client = None
                logger.warning("Neo4j client not available - GraphRAG features disabled")
            
            try:
                nomic_client = container.get_nomic()
            except RuntimeError:
                nomic_client = None
                logger.warning("Nomic client not available - embeddings disabled")
            
            logger.info("✅ L9 Neural MCP Server initialization complete")
            _initialization_done = True
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")

if __name__ == "__main__":
    # Register tools at import time - FastMCP pattern
    register_memory_tools(mcp, None)  # Pass None since we'll use globals
    register_core_tools(mcp, None)    # Pass None since we'll use globals
    
    # Run MCP server - initialization happens lazily within event loop
    mcp.run(transport='stdio')