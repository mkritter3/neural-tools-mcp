#!/usr/bin/env python3
"""
ADR-66 + ADR-67: Unified Graphiti Service Client
Interfaces with containerized Graphiti service for temporal knowledge graphs
with unified Neo4j vector storage
"""

import os
import logging
import asyncio
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class UnifiedGraphitiClient:
    """
    Client for containerized Graphiti service implementing ADR-66 + ADR-67
    Provides temporal knowledge graph capabilities with unified Neo4j storage
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.graphiti_host = os.getenv("GRAPHITI_HOST", "localhost")
        self.graphiti_port = os.getenv("GRAPHITI_PORT", "48080")
        self.base_url = f"http://{self.graphiti_host}:{self.graphiti_port}"

        # HTTP client with timeout configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )

        logger.info(f"🔗 UnifiedGraphitiClient initialized for project: {project_name}")
        logger.info(f"📡 Graphiti service: {self.base_url}")

    async def health_check(self) -> Dict[str, Any]:
        """Check Graphiti service health"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Graphiti health check failed: {e}")
            return {"status": "error", "message": str(e)}

    async def add_code_episode(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add code file as episodic knowledge to temporal graph
        ADR-67: Episodic processing for incremental, conflict-resistant indexing
        """
        try:
            episode_data = {
                "content": content,
                "episode_type": "text",
                "group_id": self.project_name,
                "metadata": {
                    "file_path": file_path,
                    "file_type": "code",
                    "timestamp": datetime.now().isoformat(),
                    "project": self.project_name,
                    **(metadata or {})
                }
            }

            response = await self.client.post(
                f"{self.base_url}/projects/{self.project_name}/episodes",
                json=episode_data
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ Code episode added for {file_path}: {result.get('episode_id')}")
            return result

        except Exception as e:
            logger.error(f"Failed to add code episode for {file_path}: {e}")
            return {"success": False, "error": str(e)}

    async def add_documentation_episode(
        self,
        file_path: str,
        content: str,
        doc_type: str = "markdown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add documentation as episodic knowledge with relationship detection
        ADR-69: LLM enhanced relationship detection between docs and code
        """
        try:
            episode_data = {
                "content": content,
                "episode_type": "text",
                "group_id": self.project_name,
                "metadata": {
                    "file_path": file_path,
                    "file_type": "documentation",
                    "doc_type": doc_type,
                    "timestamp": datetime.now().isoformat(),
                    "project": self.project_name,
                    **(metadata or {})
                }
            }

            response = await self.client.post(
                f"{self.base_url}/projects/{self.project_name}/episodes",
                json=episode_data
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"✅ Documentation episode added for {file_path}: {result.get('episode_id')}")
            return result

        except Exception as e:
            logger.error(f"Failed to add documentation episode for {file_path}: {e}")
            return {"success": False, "error": str(e)}

    async def search_knowledge_graph(
        self,
        query: str,
        limit: int = 10,
        group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search temporal knowledge graph with hybrid capabilities
        ADR-66: Unified Neo4j storage with vector + graph + full-text search
        """
        try:
            search_data = {
                "query": query,
                "limit": limit,
                "group_id": group_id or self.project_name
            }

            response = await self.client.post(
                f"{self.base_url}/projects/{self.project_name}/search",
                json=search_data
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"🔍 Knowledge graph search completed: {len(result.get('results', []))} results")
            return result

        except Exception as e:
            logger.error(f"Knowledge graph search failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    async def get_project_status(self) -> Dict[str, Any]:
        """Get project's knowledge graph status and statistics"""
        try:
            response = await self.client.get(f"{self.base_url}/projects")
            response.raise_for_status()

            projects_data = response.json()
            is_active = self.project_name in projects_data.get("projects", [])

            return {
                "project_name": self.project_name,
                "is_active": is_active,
                "total_projects": projects_data.get("total", 0)
            }

        except Exception as e:
            logger.error(f"Failed to get project status: {e}")
            return {"project_name": self.project_name, "is_active": False, "error": str(e)}

    async def cleanup_project(self) -> Dict[str, Any]:
        """Clean up project's temporal knowledge graph"""
        try:
            response = await self.client.delete(f"{self.base_url}/projects/{self.project_name}")
            response.raise_for_status()

            result = response.json()
            logger.info(f"🧹 Project {self.project_name} knowledge graph cleaned up")
            return result

        except Exception as e:
            logger.error(f"Failed to cleanup project {self.project_name}: {e}")
            return {"success": False, "error": str(e)}

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
        logger.info(f"🔌 UnifiedGraphitiClient closed for project: {self.project_name}")


class UnifiedIndexerService:
    """
    ADR-66 + ADR-67: Enhanced indexer using unified Neo4j + Graphiti architecture
    Eliminates dual-write complexity while adding temporal knowledge capabilities
    """

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.graphiti_client = UnifiedGraphitiClient(project_name)
        logger.info(f"🚀 UnifiedIndexerService initialized for project: {project_name}")

    async def initialize(self) -> Dict[str, Any]:
        """Initialize unified indexer service"""
        try:
            # Check Graphiti service health
            health = await self.graphiti_client.health_check()

            if health.get("status") == "healthy":
                logger.info(f"✅ Unified indexer ready for project: {self.project_name}")
                return {
                    "success": True,
                    "architecture": "unified_neo4j_graphiti",
                    "project": self.project_name,
                    "graphiti_health": health
                }
            else:
                logger.warning(f"⚠️ Graphiti service not healthy: {health}")
                return {
                    "success": False,
                    "error": "Graphiti service unhealthy",
                    "health": health
                }

        except Exception as e:
            logger.error(f"Failed to initialize unified indexer: {e}")
            return {"success": False, "error": str(e)}

    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process file using unified Neo4j + Graphiti temporal knowledge architecture
        ADR-66: No dual-write complexity, ADR-67: Episodic processing
        """
        try:
            path_obj = Path(file_path)

            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Determine file type and processing strategy
            if path_obj.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.rs', '.go']:
                # Code file - add as code episode
                result = await self.graphiti_client.add_code_episode(
                    file_path=str(file_path),
                    content=content,
                    metadata={
                        "language": path_obj.suffix[1:],
                        "size": len(content),
                        "lines": len(content.split('\n'))
                    }
                )
            elif path_obj.suffix in ['.md', '.rst', '.txt', '.adoc']:
                # Documentation file - add as documentation episode
                result = await self.graphiti_client.add_documentation_episode(
                    file_path=str(file_path),
                    content=content,
                    doc_type=path_obj.suffix[1:],
                    metadata={
                        "size": len(content),
                        "lines": len(content.split('\n'))
                    }
                )
            else:
                logger.info(f"⏭️ Skipping unsupported file type: {file_path}")
                return {"success": True, "action": "skipped", "reason": "unsupported_type"}

            if result.get("success"):
                logger.info(f"✅ File processed via unified architecture: {file_path}")
                return {
                    "success": True,
                    "architecture": "unified_neo4j_graphiti",
                    "episode_id": result.get("episode_id"),
                    "file_path": file_path
                }
            else:
                logger.error(f"❌ Failed to process file: {file_path} - {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error"),
                    "file_path": file_path
                }

        except Exception as e:
            logger.error(f"Exception processing file {file_path}: {e}")
            return {"success": False, "error": str(e), "file_path": file_path}

    async def search_unified_knowledge(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search unified Neo4j + Graphiti knowledge graph
        ADR-66: Single storage system, ADR-67: Temporal knowledge
        """
        return await self.graphiti_client.search_knowledge_graph(query, limit)

    async def get_status(self) -> Dict[str, Any]:
        """Get unified indexer service status"""
        return await self.graphiti_client.get_project_status()

    async def cleanup(self) -> Dict[str, Any]:
        """Clean up unified indexer resources"""
        cleanup_result = await self.graphiti_client.cleanup_project()
        await self.graphiti_client.close()
        return cleanup_result