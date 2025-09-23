#!/usr/bin/env python3
"""
IndexerServiceAdapter - Interface Adapter for Elite IncrementalIndexer

ADR-0077: Adapter wrapper for existing IncrementalIndexer
Provides exact interface needed by modular MCP tools while preserving
all ADR-0072 (HNSW vectors) and ADR-0075 (graph context) optimizations

This adapter enables the modular MCP tools to use the sophisticated
IncrementalIndexer service with its elite features:
- Neo4j HNSW vector indexes (O(log n) performance)
- Tree-sitter code structure extraction
- Complete graph relationships (File→Chunk, File→Symbol, Method→Calls)
- AST-aware semantic chunking
- Unified Neo4j atomic transactions

Author: L9 Engineering Team
Date: September 22, 2025
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class IndexerServiceAdapter:
    """
    ADR-0077: Adapter wrapper for existing IncrementalIndexer
    Provides exact interface needed by modular MCP tools while preserving
    all ADR-0072 (HNSW vectors) and ADR-0075 (graph context) optimizations
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.incremental_indexer = None  # Will hold IncrementalIndexer instance
        self.container = None  # ServiceContainer with Neo4j + Nomic

    async def initialize(self) -> Dict[str, Any]:
        """Initialize IncrementalIndexer with all ADR-0072/0075 optimizations"""
        try:
            # Import required classes - use absolute imports to match ServiceContainer
            from servers.services.indexer_service import IncrementalIndexer
            from servers.services.service_container import ServiceContainer
            from servers.services.project_context_manager import get_project_context_manager

            # Get or create ProjectContextManager singleton (ADR-0044)
            context_manager = await get_project_context_manager()

            # Set project context or detect it
            if self.project_name and self.project_name != "default":
                # Try to switch to specific project if we have a name
                try:
                    await context_manager.switch_project(self.project_name)
                except ValueError:
                    # Project not found, try to detect from current directory
                    logger.info(f"Project '{self.project_name}' not registered, detecting from current directory")
                    await context_manager.detect_project()
            else:
                # Auto-detect project from current context
                await context_manager.detect_project()

            # Create ServiceContainer with ProjectContextManager (proper architecture)
            self.container = ServiceContainer(context_manager=context_manager)

            # Get the detected project info
            project_info = await context_manager.get_current_project()
            self.project_name = project_info["project"]
            project_path = project_info["path"]

            # Create IncrementalIndexer with proper context manager - this gets us:
            # - Neo4j HNSW vector indexes (ADR-0072)
            # - Tree-sitter code extraction (ADR-0075)
            # - Complete graph relationships (ADR-0075)
            # - AST-aware semantic chunking
            # - Unified Neo4j storage
            self.incremental_indexer = IncrementalIndexer(
                project_path=project_path,
                project_name=self.project_name,
                container=self.container
            )

            # Initialize services - this creates HNSW indexes
            logger.info(f"🔧 Initializing IncrementalIndexer services for {self.project_name}...")
            try:
                success = await self.incremental_indexer.initialize_services()
                logger.info(f"🔧 IncrementalIndexer initialization result: {success}")
            except Exception as init_error:
                logger.error(f"❌ IncrementalIndexer initialization failed: {init_error}")
                raise RuntimeError(f"IncrementalIndexer initialization failed: {init_error}")

            if success:
                logger.info(f"✅ Elite indexer ready (ADR-0072/0075): {self.project_name}")
                return {
                    "success": True,
                    "architecture": "incremental_indexer_adr_0072_0075",
                    "project": self.project_name,
                    "project_path": project_path,
                    "detection_method": project_info.get("method", "unknown"),
                    "detection_confidence": project_info.get("confidence", 0.0),
                    "features": {
                        "hnsw_vectors": "enabled",
                        "tree_sitter": "enabled",
                        "graph_relationships": "enabled",
                        "ast_chunking": "enabled",
                        "unified_neo4j": "enabled",
                        "project_context_manager": "enabled"
                    }
                }
            else:
                raise RuntimeError("IncrementalIndexer returned False for initialization")

        except Exception as e:
            logger.error(f"Failed to initialize elite indexer: {e}")
            return {"success": False, "error": str(e)}

    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process file using IncrementalIndexer with all elite features:
        - HNSW vector indexes (ADR-0072)
        - Tree-sitter code extraction (ADR-0075)
        - Complete graph relationships (ADR-0075)
        - AST-aware semantic chunking
        - Unified Neo4j storage
        """
        try:
            if not self.incremental_indexer:
                raise RuntimeError("Indexer not initialized")

            # Use IncrementalIndexer's elite processing
            # This includes:
            # - Tree-sitter symbol extraction
            # - AST-aware semantic chunking
            # - HNSW vector embedding generation
            # - Complete graph relationship creation
            # - Unified Neo4j atomic transaction storage
            await self.incremental_indexer.index_file(file_path, action="process")

            # Get metrics from the indexer
            metrics = self.incremental_indexer.get_metrics()

            logger.info(f"✅ Elite indexing complete: {file_path}")
            return {
                "success": True,
                "architecture": "incremental_indexer_adr_0072_0075",
                "file_path": file_path,
                "project": self.project_name,
                "features_used": {
                    "hnsw_vectors": True,
                    "tree_sitter_extraction": True,
                    "graph_relationships": True,
                    "ast_chunking": True,
                    "unified_neo4j": True
                },
                "metrics": {
                    "files_indexed": metrics.get('files_indexed', 0),
                    "chunks_created": metrics.get('chunks_created', 0),
                    "symbols_created": metrics.get('symbols_created', 0)
                }
            }

        except Exception as e:
            logger.error(f"Elite indexing failed for {file_path}: {e}")
            return {"success": False, "error": str(e), "file_path": file_path}

    async def cleanup(self) -> Dict[str, Any]:
        """Cleanup elite indexer resources"""
        try:
            if self.incremental_indexer:
                # Use IncrementalIndexer's graceful shutdown
                await self.incremental_indexer.shutdown()

            if self.container:
                # Clean up service container
                if hasattr(self.container, 'cleanup'):
                    await self.container.cleanup()

            logger.info(f"✅ Elite indexer cleanup completed: {self.project_name}")
            return {"success": True, "action": "cleanup_completed"}
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return {"success": False, "error": str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from underlying IncrementalIndexer"""
        if self.incremental_indexer:
            base_metrics = self.incremental_indexer.get_metrics()
            # Enhance with adapter-specific info
            base_metrics.update({
                "adapter_type": "indexer_service_adapter",
                "adr_compliance": ["ADR-0072", "ADR-0075"],
                "elite_features": {
                    "hnsw_vectors": True,
                    "tree_sitter": True,
                    "graph_relationships": True,
                    "ast_chunking": True,
                    "unified_neo4j": True
                }
            })
            return base_metrics
        else:
            return {"error": "Indexer not initialized"}

    @property
    def is_initialized(self) -> bool:
        """Check if the adapter and underlying indexer are ready"""
        return (self.incremental_indexer is not None and
                self.container is not None and
                self.incremental_indexer.services_initialized)