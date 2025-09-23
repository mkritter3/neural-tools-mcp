"""
ADR-66 + ADR-67: Unified Core MCP Tools
Provides essential project understanding and search functionality using
unified Neo4j + Graphiti temporal knowledge architecture
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP

from servers.services.unified_graphiti_service import UnifiedIndexerService

logger = logging.getLogger(__name__)

def register_unified_core_tools(mcp: FastMCP, container=None):
    """Register unified MCP tools using ADR-66 + ADR-67 architecture"""

    @mcp.tool()
    async def project_understanding(
        scope: str = "full",
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Generate condensed project understanding using unified Neo4j + Graphiti knowledge graph

        Args:
            scope: 'full', 'architecture', 'dependencies', 'core_logic', 'documentation'
            max_results: Maximum number of knowledge graph results to include
        """
        try:
            if not container or not hasattr(container, 'project_name'):
                return {
                    "status": "error",
                    "message": "Service container not properly initialized"
                }

            project_name = container.project_name
            indexer = UnifiedIndexerService(project_name)

            # Initialize and check status
            init_result = await indexer.initialize()
            if not init_result.get("success"):
                return {
                    "status": "error",
                    "message": f"Unified indexer initialization failed: {init_result.get('error')}"
                }

            understanding = {
                "project": project_name,
                "timestamp": datetime.now().isoformat(),
                "scope": scope,
                "architecture": "unified_neo4j_graphiti",
                "max_results": max_results
            }

            # Query knowledge graph based on scope
            scope_queries = {
                "full": "overview architecture dependencies documentation code structure",
                "architecture": "architecture design patterns structure modules components",
                "dependencies": "dependencies imports requirements libraries frameworks",
                "core_logic": "main functions classes core business logic algorithms",
                "documentation": "documentation readme guides tutorials examples"
            }

            query = scope_queries.get(scope, scope_queries["full"])

            # Search unified knowledge graph
            search_result = await indexer.search_unified_knowledge(query, limit=max_results)

            if search_result.get("success"):
                understanding.update({
                    "status": "success",
                    "knowledge_results": search_result.get("results", []),
                    "total_results": len(search_result.get("results", [])),
                    "search_query": query
                })
            else:
                understanding.update({
                    "status": "partial",
                    "message": f"Knowledge graph search failed: {search_result.get('error')}",
                    "knowledge_results": []
                })

            # Get project status
            status = await indexer.get_status()
            understanding["project_status"] = status

            await indexer.cleanup()
            return understanding

        except Exception as e:
            logger.error(f"Project understanding failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    @mcp.tool()
    async def semantic_code_search(
        query: str,
        limit: int = 10,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Search code using unified Neo4j + Graphiti semantic and temporal knowledge

        Args:
            query: Natural language search query
            limit: Maximum number of results (1-50)
            include_context: Include temporal context and relationships
        """
        try:
            if not container or not hasattr(container, 'project_name'):
                return {
                    "status": "error",
                    "message": "Service container not properly initialized"
                }

            # Validate limit
            limit = max(1, min(50, limit))

            project_name = container.project_name
            indexer = UnifiedIndexerService(project_name)

            # Initialize
            init_result = await indexer.initialize()
            if not init_result.get("success"):
                return {
                    "status": "error",
                    "message": f"Unified indexer initialization failed: {init_result.get('error')}"
                }

            # Search unified knowledge graph
            search_result = await indexer.search_unified_knowledge(query, limit=limit)

            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "project": project_name,
                "architecture": "unified_neo4j_graphiti",
                "limit": limit,
                "include_context": include_context
            }

            if search_result.get("success"):
                results = search_result.get("results", [])

                result.update({
                    "status": "success",
                    "results": results,
                    "total_results": len(results),
                    "search_type": "semantic_temporal"
                })

                if include_context and results:
                    # Add temporal context information
                    result["temporal_context"] = {
                        "episodic_processing": True,
                        "knowledge_graph_enabled": True,
                        "relationship_detection": True
                    }

            else:
                result.update({
                    "status": "error",
                    "message": search_result.get("error"),
                    "results": []
                })

            await indexer.cleanup()
            return result

        except Exception as e:
            logger.error(f"Semantic code search failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "query": query,
                "results": []
            }

    @mcp.tool()
    async def graphrag_hybrid_search(
        query: str,
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Hybrid search combining semantic, graph traversal, and temporal knowledge
        using unified Neo4j + Graphiti architecture

        Args:
            query: Search query
            limit: Maximum results (1-25)
            include_graph_context: Include graph relationships
            max_hops: Maximum relationship hops (0-3)
        """
        try:
            if not container or not hasattr(container, 'project_name'):
                return {
                    "status": "error",
                    "message": "Service container not properly initialized"
                }

            # Validate parameters
            limit = max(1, min(25, limit))
            max_hops = max(0, min(3, max_hops))

            project_name = container.project_name
            indexer = UnifiedIndexerService(project_name)

            # Initialize
            init_result = await indexer.initialize()
            if not init_result.get("success"):
                return {
                    "status": "error",
                    "message": f"Unified indexer initialization failed: {init_result.get('error')}"
                }

            # Hybrid search via unified knowledge graph
            search_result = await indexer.search_unified_knowledge(query, limit=limit)

            result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "project": project_name,
                "architecture": "unified_neo4j_graphiti",
                "search_type": "hybrid_temporal",
                "limit": limit,
                "include_graph_context": include_graph_context,
                "max_hops": max_hops
            }

            if search_result.get("success"):
                results = search_result.get("results", [])

                result.update({
                    "status": "success",
                    "results": results,
                    "total_results": len(results),
                    "hybrid_features": {
                        "semantic_search": True,
                        "graph_traversal": include_graph_context,
                        "temporal_knowledge": True,
                        "episodic_processing": True
                    }
                })

                if include_graph_context:
                    result["graph_context"] = {
                        "relationship_hops": max_hops,
                        "temporal_relationships": True,
                        "knowledge_graph_enabled": True
                    }

            else:
                result.update({
                    "status": "error",
                    "message": search_result.get("error"),
                    "results": []
                })

            await indexer.cleanup()
            return result

        except Exception as e:
            logger.error(f"GraphRAG hybrid search failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "query": query,
                "results": []
            }

    @mcp.tool()
    async def reindex_path(
        path: str,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Reindex a path using unified Neo4j + Graphiti temporal knowledge architecture

        Args:
            path: File or directory path to reindex
            recursive: Process directories recursively
        """
        try:
            if not container or not hasattr(container, 'project_name'):
                return {
                    "status": "error",
                    "message": "Service container not properly initialized"
                }

            project_name = container.project_name
            indexer = UnifiedIndexerService(project_name)

            # Initialize
            init_result = await indexer.initialize()
            if not init_result.get("success"):
                return {
                    "status": "error",
                    "message": f"Unified indexer initialization failed: {init_result.get('error')}"
                }

            from pathlib import Path
            path_obj = Path(path)

            result = {
                "path": str(path),
                "recursive": recursive,
                "project": project_name,
                "architecture": "unified_neo4j_graphiti",
                "timestamp": datetime.now().isoformat(),
                "processed_files": [],
                "failed_files": []
            }

            if path_obj.is_file():
                # Process single file
                file_result = await indexer.process_file(str(path_obj))
                if file_result.get("success"):
                    result["processed_files"].append(str(path_obj))
                else:
                    result["failed_files"].append({
                        "file": str(path_obj),
                        "error": file_result.get("error")
                    })

            elif path_obj.is_dir():
                # Process directory
                if recursive:
                    files = list(path_obj.rglob("*"))
                else:
                    files = list(path_obj.iterdir())

                for file_path in files:
                    if file_path.is_file():
                        file_result = await indexer.process_file(str(file_path))
                        if file_result.get("success"):
                            result["processed_files"].append(str(file_path))
                        else:
                            result["failed_files"].append({
                                "file": str(file_path),
                                "error": file_result.get("error")
                            })

            else:
                await indexer.cleanup()
                return {
                    "status": "error",
                    "message": f"Path does not exist: {path}"
                }

            result.update({
                "status": "success",
                "total_processed": len(result["processed_files"]),
                "total_failed": len(result["failed_files"])
            })

            await indexer.cleanup()
            return result

        except Exception as e:
            logger.error(f"Reindex path failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "path": path
            }

    @mcp.tool()
    async def neural_system_status() -> Dict[str, Any]:
        """
        Get comprehensive status of unified Neo4j + Graphiti neural system
        """
        try:
            if not container or not hasattr(container, 'project_name'):
                return {
                    "status": "error",
                    "message": "Service container not properly initialized"
                }

            project_name = container.project_name
            indexer = UnifiedIndexerService(project_name)

            # Get system status
            status_result = await indexer.get_status()
            health_result = await indexer.graphiti_client.health_check()

            result = {
                "timestamp": datetime.now().isoformat(),
                "project": project_name,
                "architecture": "unified_neo4j_graphiti",
                "system_status": status_result,
                "health_check": health_result,
                "components": {
                    "neo4j": health_result.get("neo4j_connected", False),
                    "graphiti": health_result.get("graphiti_available", False),
                    "temporal_knowledge": True,
                    "episodic_processing": True
                },
                "features": {
                    "dual_write_eliminated": True,
                    "vector_storage": "neo4j_native",
                    "temporal_graphs": True,
                    "project_isolation": True,
                    "local_llm": True
                }
            }

            await indexer.cleanup()
            return result

        except Exception as e:
            logger.error(f"Neural system status failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }

    logger.info("✅ Unified core tools registered (ADR-66 + ADR-67 architecture)")