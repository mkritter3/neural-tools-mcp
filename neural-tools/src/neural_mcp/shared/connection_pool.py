"""
ADR-0075 Connection Pooling - September 2025 Standards
Shared Neo4j service instances with connection pooling
"""

import asyncio
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Global shared services with connection pooling
_shared_neo4j_services: Dict[str, any] = {}
_shared_embedding_services: Dict[str, any] = {}
_service_lock = asyncio.Lock()

async def get_shared_neo4j_service(project_name: str):
    """
    ADR-0075 Phase 4: Get or create shared Neo4j service instance
    Eliminates 1.5s initialization overhead per query
    """
    async with _service_lock:
        if project_name not in _shared_neo4j_services:
            from servers.services.neo4j_service import Neo4jService
            neo4j = Neo4jService(project_name)

            neo4j_result = await neo4j.initialize()
            if not neo4j_result.get("success"):
                raise Exception(f"Neo4j initialization failed: {neo4j_result}")

            _shared_neo4j_services[project_name] = neo4j
            logger.info(f"🔄 ADR-0075: Created shared Neo4j service for {project_name}")

        return _shared_neo4j_services[project_name]

async def get_shared_embedding_service(project_name: str):
    """
    ADR-0079: Get or create shared Embedding service instance
    Provides vector embedding generation for semantic search
    """
    async with _service_lock:
        if project_name not in _shared_embedding_services:
            from servers.services.nomic_service import NomicService
            embedding_service = NomicService()

            embedding_result = await embedding_service.initialize()
            if not embedding_result.get("success"):
                raise Exception(f"Embedding service initialization failed: {embedding_result}")

            _shared_embedding_services[project_name] = embedding_service
            logger.info(f"🚀 ADR-0079: Created shared Embedding service for {project_name}")

        return _shared_embedding_services[project_name]

async def cleanup_shared_services():
    """Cleanup all shared services"""
    async with _service_lock:
        # Cleanup Neo4j services
        for project_name, service in _shared_neo4j_services.items():
            try:
                if hasattr(service, 'close'):
                    await service.close()
                logger.info(f"🧹 Cleaned up Neo4j service for {project_name}")
            except Exception as e:
                logger.warning(f"Error cleaning up Neo4j service for {project_name}: {e}")
        _shared_neo4j_services.clear()

        # Cleanup Embedding services
        for project_name, service in _shared_embedding_services.items():
            try:
                if hasattr(service, 'close'):
                    await service.close()
                logger.info(f"🧹 Cleaned up Embedding service for {project_name}")
            except Exception as e:
                logger.warning(f"Error cleaning up Embedding service for {project_name}: {e}")
        _shared_embedding_services.clear()

def get_connection_stats() -> dict:
    """Get connection pool statistics"""
    return {
        'active_connections': len(_shared_neo4j_services) + len(_shared_embedding_services),
        'neo4j_connections': len(_shared_neo4j_services),
        'embedding_connections': len(_shared_embedding_services),
        'projects': list(set(list(_shared_neo4j_services.keys()) + list(_shared_embedding_services.keys())))
    }