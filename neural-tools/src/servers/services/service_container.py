#!/usr/bin/env python3
"""
Service Container - Dependency injection for neural MCP services
Eliminates global variable coupling and enables proper testing
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .nomic_local_service import NomicService
from .qdrant_service import QdrantService
from .neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Service health status"""
    healthy: bool
    message: str
    details: Dict[str, Any]

class ServiceContainer:
    """Dependency injection container for all neural services"""
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self._initialized = False
        
        # Service instances
        self.nomic: Optional[NomicService] = None
        self.qdrant: Optional[QdrantService] = None
        self.neo4j: Optional[Neo4jService] = None
        
        # Service health tracking
        self._service_health: Dict[str, ServiceHealth] = {}
        
    async def initialize_all_services(self) -> Dict[str, Any]:
        """Initialize all services with proper error handling and health tracking"""
        if self._initialized:
            return {"status": "already_initialized", "services": self._service_health}
            
        initialization_results = {}
        
        try:
            logger.info("🚀 Initializing neural services...")
            
            # Initialize Nomic service
            logger.info("📡 Initializing Nomic embedding service...")
            self.nomic = NomicService()
            nomic_result = await self.nomic.initialize()
            initialization_results["nomic"] = nomic_result
            self._service_health["nomic"] = ServiceHealth(
                healthy=nomic_result.get("success", False),
                message=nomic_result.get("message", "Unknown status"),
                details=nomic_result
            )
            
            # Initialize Qdrant service
            logger.info("🗃️ Initializing Qdrant vector database...")
            self.qdrant = QdrantService(self.project_name)
            qdrant_result = await self.qdrant.initialize()
            initialization_results["qdrant"] = qdrant_result
            self._service_health["qdrant"] = ServiceHealth(
                healthy=qdrant_result.get("success", False),
                message=qdrant_result.get("message", "Unknown status"),
                details=qdrant_result
            )
            
            # Initialize Neo4j service
            logger.info("🔗 Initializing Neo4j GraphRAG...")
            self.neo4j = Neo4jService(self.project_name)
            neo4j_result = await self.neo4j.initialize()
            initialization_results["neo4j"] = neo4j_result
            self._service_health["neo4j"] = ServiceHealth(
                healthy=neo4j_result.get("success", False),
                message=neo4j_result.get("message", "Unknown status"),
                details=neo4j_result
            )
            
            # Check overall health
            healthy_services = sum(1 for health in self._service_health.values() if health.healthy)
            total_services = len(self._service_health)
            
            # Require at least 2/3 services to be healthy for operational status
            self._initialized = healthy_services >= 2
            
            logger.info(f"✅ Service initialization complete: {healthy_services}/{total_services} services healthy")
            
            return {
                "status": "success" if self._initialized else "partial",
                "healthy_services": healthy_services,
                "total_services": total_services,
                "service_details": initialization_results,
                "overall_health": f"{healthy_services}/{total_services} services operational"
            }
            
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Service initialization failed: {str(e)}",
                "service_details": initialization_results
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all services"""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "healthy_services": 0,
                "total_services": 3,
                "services": {}
            }
        
        health_results = {}
        
        # Check each service
        if self.nomic:
            health_results["nomic"] = await self.nomic.health_check()
        else:
            health_results["nomic"] = {"healthy": False, "message": "Service not initialized"}
            
        if self.qdrant:
            health_results["qdrant"] = await self.qdrant.health_check()
        else:
            health_results["qdrant"] = {"healthy": False, "message": "Service not initialized"}
            
        if self.neo4j:
            health_results["neo4j"] = await self.neo4j.health_check()
        else:
            health_results["neo4j"] = {"healthy": False, "message": "Service not initialized"}
        
        # Calculate overall health
        healthy_count = sum(1 for health in health_results.values() if health.get("healthy", False))
        total_count = len(health_results)
        
        return {
            "status": "healthy" if healthy_count >= 2 else "degraded",
            "healthy_services": healthy_count,
            "total_services": total_count,
            "services": health_results,
            "overall_score": f"{healthy_count}/{total_count}"
        }
    
    def get_nomic(self) -> NomicService:
        """Get Nomic service with validation"""
        if not self._initialized or not self.nomic:
            raise RuntimeError("Nomic service not initialized")
        return self.nomic
    
    def get_qdrant(self) -> QdrantService:
        """Get Qdrant service with validation"""
        if not self._initialized or not self.qdrant:
            raise RuntimeError("Qdrant service not initialized")
        return self.qdrant
    
    def get_neo4j(self) -> Neo4jService:
        """Get Neo4j service with validation"""
        if not self._initialized or not self.neo4j:
            raise RuntimeError("Neo4j service not initialized")
        return self.neo4j
    
    @property
    def initialized(self) -> bool:
        """Check if container is initialized"""
        return self._initialized
    
    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status for specific service"""
        return self._service_health.get(service_name)

# Global container instance (will be replaced by proper DI in neural_server.py)
_container: Optional[ServiceContainer] = None

async def get_container(project_name: str = "default") -> ServiceContainer:
    """Get or create service container (singleton pattern)"""
    global _container
    
    if _container is None:
        _container = ServiceContainer(project_name)
        await _container.initialize_all_services()
    
    return _container

async def ensure_services_initialized(project_name: str = "default"):
    """Ensure services are initialized - backward compatibility helper"""
    await get_container(project_name)