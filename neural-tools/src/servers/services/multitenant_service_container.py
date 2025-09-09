#!/usr/bin/env python3
"""
Multitenant Service Container for Neural Tools
Extends ServiceContainer with tenant isolation and resource management
Following roadmap Phase 1.3 specifications
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from service_container import ServiceContainer, ServiceHealth
from infrastructure.multitenancy import TenantManager, TenantConfig, MultitenantQueryFilter, TenantResourceMonitor

logger = logging.getLogger(__name__)

class MultitenantServiceContainer(ServiceContainer):
    """
    Enhanced ServiceContainer with tenant isolation and resource management
    Provides secure multi-tenant operations across all neural services
    """
    
    def __init__(self, tenant_id: str, project_name: str = "default"):
        """
        Initialize multitenant service container
        
        Args:
            tenant_id: Tenant identifier for isolation
            project_name: Project name within tenant
        """
        super().__init__(project_name)
        self.tenant_id = tenant_id
        self.original_project_name = project_name
        
        # Tenant management components
        self.tenant_manager: Optional[TenantManager] = None
        self.query_filter: Optional[MultitenantQueryFilter] = None
        self.resource_monitor: Optional[TenantResourceMonitor] = None
        
        # Override project name with tenant namespace
        self._setup_tenant_namespace()
        
    def _setup_tenant_namespace(self):
        """Setup tenant-namespaced project name"""
        # Create tenant-isolated project identifier
        self.project_name = f"tenant_{self.tenant_id}_project_{self.original_project_name}"
        logger.info(f"Initialized multitenant container: {self.tenant_id} -> {self.project_name}")
    
    async def initialize_tenant_services(self, tenant_manager: TenantManager) -> Dict[str, Any]:
        """
        Initialize services with tenant management
        
        Args:
            tenant_manager: Configured tenant manager instance
            
        Returns:
            Initialization results with tenant validation
        """
        # Store tenant management components
        self.tenant_manager = tenant_manager
        self.query_filter = MultitenantQueryFilter(tenant_manager)
        self.resource_monitor = TenantResourceMonitor(tenant_manager)
        
        # Validate tenant access to project
        if not await self.query_filter.validate_access(self.tenant_id, self.original_project_name):
            return {
                "status": "error",
                "message": f"Tenant {self.tenant_id} does not have access to project {self.original_project_name}",
                "tenant_id": self.tenant_id,
                "project_name": self.original_project_name
            }
        
        # Initialize base services
        base_result = await self.initialize_all_services()
        
        # Add tenant-specific information
        base_result.update({
            "tenant_id": self.tenant_id,
            "original_project_name": self.original_project_name,
            "namespaced_project": self.project_name,
            "tenant_isolation_enabled": True
        })
        
        logger.info(f"Tenant services initialized for {self.tenant_id}: {base_result['status']}")
        return base_result
    
    async def create_tenant_collections(self) -> Dict[str, Any]:
        """
        Create tenant-isolated collections with proper naming
        """
        if not self.tenant_manager:
            return {"status": "error", "message": "Tenant manager not initialized"}
        
        try:
            results = {}
            
            # Create Qdrant collections with tenant isolation
            if self.qdrant and self._service_health.get("qdrant", {}).healthy:
                collection_types = ["code", "docs", "general"]
                
                for collection_type in collection_types:
                    collection_name = self.tenant_manager.get_collection_name(
                        self.tenant_id, self.original_project_name, collection_type
                    )
                    
                    # Create collection with tenant metadata
                    collection_result = await self.qdrant.ensure_collection(
                        collection_name,
                        metadata={
                            "tenant_id": self.tenant_id,
                            "project_name": self.original_project_name,
                            "collection_type": collection_type
                        }
                    )
                    results[f"qdrant_{collection_type}"] = collection_result
            
            # Setup Neo4j with tenant-specific labels
            if self.neo4j and self._service_health.get("neo4j", {}).healthy:
                labels = self.tenant_manager.get_neo4j_labels(
                    self.tenant_id, self.original_project_name
                )
                
                # Create indexes with tenant labels
                neo4j_result = await self.neo4j.setup_tenant_schema(labels)
                results["neo4j_schema"] = neo4j_result
            
            return {
                "status": "success",
                "tenant_id": self.tenant_id,
                "project_name": self.original_project_name,
                "collections_created": results
            }
            
        except Exception as e:
            logger.error(f"Failed to create tenant collections: {e}")
            return {
                "status": "error",
                "message": str(e),
                "tenant_id": self.tenant_id
            }
    
    async def execute_tenant_query(
        self, 
        query: str, 
        limit: int = 10,
        collection_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Execute query with tenant isolation and resource limits
        
        Args:
            query: Search query
            limit: Result limit
            collection_types: Types of collections to search
            
        Returns:
            Filtered query results
        """
        if not self.query_filter or not self.resource_monitor:
            return {"status": "error", "message": "Tenant services not initialized"}
        
        # Check rate limits
        if not await self.resource_monitor.track_request(self.tenant_id):
            return {
                "status": "rate_limited",
                "message": f"Rate limit exceeded for tenant {self.tenant_id}",
                "tenant_id": self.tenant_id
            }
        
        try:
            # Get accessible collections
            if collection_types is None:
                collection_types = ["code", "docs", "general"]
            
            accessible_collections = await self.query_filter.get_accessible_collections(
                self.tenant_id, collection_types
            )
            
            if not accessible_collections:
                return {
                    "status": "no_access",
                    "message": f"Tenant {self.tenant_id} has no accessible collections",
                    "tenant_id": self.tenant_id
                }
            
            # Execute queries with tenant filtering
            results = []
            
            # Qdrant search with tenant filter
            if self.qdrant:
                qdrant_filter = self.query_filter.create_qdrant_filter(
                    self.tenant_id, self.original_project_name
                )
                
                for collection_name in accessible_collections:
                    try:
                        search_results = await self.qdrant.search_with_filter(
                            collection_name=collection_name,
                            query=query,
                            limit=limit,
                            filter_conditions=qdrant_filter
                        )
                        
                        # Add tenant metadata to results
                        for result in search_results:
                            result["tenant_id"] = self.tenant_id
                            result["collection"] = collection_name
                        
                        results.extend(search_results)
                        
                    except Exception as e:
                        logger.warning(f"Search failed for collection {collection_name}: {e}")
            
            # Neo4j search with tenant labels
            if self.neo4j:
                labels = self.tenant_manager.get_neo4j_labels(
                    self.tenant_id, self.original_project_name
                )
                
                neo4j_filter = self.query_filter.create_neo4j_filter(
                    self.tenant_id, self.original_project_name
                )
                
                try:
                    graph_results = await self.neo4j.search_with_tenant_filter(
                        query=query,
                        labels=labels,
                        filter_clause=neo4j_filter,
                        limit=limit
                    )
                    
                    # Add tenant metadata to graph results
                    for result in graph_results:
                        result["tenant_id"] = self.tenant_id
                        result["source"] = "neo4j"
                    
                    results.extend(graph_results)
                    
                except Exception as e:
                    logger.warning(f"Neo4j search failed: {e}")
            
            return {
                "status": "success",
                "tenant_id": self.tenant_id,
                "project_name": self.original_project_name,
                "query": query,
                "total_results": len(results),
                "results": results[:limit],  # Enforce limit
                "collections_searched": accessible_collections
            }
            
        except Exception as e:
            logger.error(f"Tenant query failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "tenant_id": self.tenant_id
            }
    
    async def index_file_with_tenant_isolation(
        self, 
        file_path: str, 
        content: str,
        file_type: str = "code"
    ) -> Dict[str, Any]:
        """
        Index file with tenant isolation and resource tracking
        
        Args:
            file_path: Path to file
            content: File content
            file_type: Type of file (code, docs, general)
            
        Returns:
            Indexing result with tenant information
        """
        if not self.resource_monitor:
            return {"status": "error", "message": "Resource monitor not initialized"}
        
        # Track storage usage
        content_size = len(content.encode('utf-8'))
        if not await self.resource_monitor.track_storage_usage(
            self.tenant_id, self.original_project_name, content_size
        ):
            return {
                "status": "storage_exceeded",
                "message": f"Storage limit exceeded for tenant {self.tenant_id}",
                "tenant_id": self.tenant_id,
                "file_size_bytes": content_size
            }
        
        try:
            # Get tenant-specific collection name
            collection_name = self.tenant_manager.get_collection_name(
                self.tenant_id, self.original_project_name, file_type
            )
            
            # Add tenant metadata to content
            tenant_metadata = {
                "tenant_id": self.tenant_id,
                "project_name": self.original_project_name,
                "file_path": file_path,
                "file_type": file_type,
                "indexed_at": datetime.now().isoformat()
            }
            
            # Index in Qdrant with tenant isolation
            qdrant_result = None
            if self.qdrant:
                qdrant_result = await self.qdrant.index_content(
                    collection_name=collection_name,
                    content=content,
                    metadata=tenant_metadata
                )
            
            # Index in Neo4j with tenant labels
            neo4j_result = None
            if self.neo4j:
                labels = self.tenant_manager.get_neo4j_labels(
                    self.tenant_id, self.original_project_name
                )
                
                neo4j_result = await self.neo4j.extract_and_store_with_labels(
                    file_path=file_path,
                    content=content,
                    labels=labels,
                    metadata=tenant_metadata
                )
            
            return {
                "status": "success",
                "tenant_id": self.tenant_id,
                "project_name": self.original_project_name,
                "file_path": file_path,
                "file_size_bytes": content_size,
                "collection_name": collection_name,
                "qdrant_result": qdrant_result,
                "neo4j_result": neo4j_result
            }
            
        except Exception as e:
            logger.error(f"Tenant indexing failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "tenant_id": self.tenant_id,
                "file_path": file_path
            }
    
    async def get_tenant_health(self) -> Dict[str, Any]:
        """Get comprehensive tenant health including resource usage"""
        base_health = await self.health_check()
        
        # Add tenant-specific health information
        tenant_health = {
            **base_health,
            "tenant_id": self.tenant_id,
            "project_name": self.original_project_name,
            "namespaced_project": self.project_name
        }
        
        # Add resource usage if available
        if self.resource_monitor:
            try:
                usage = await self.resource_monitor.get_tenant_usage(self.tenant_id)
                tenant_health["resource_usage"] = usage
            except Exception as e:
                tenant_health["resource_usage_error"] = str(e)
        
        # Add tenant manager status
        if self.tenant_manager:
            try:
                tenant_config = await self.tenant_manager.get_tenant(self.tenant_id)
                if tenant_config:
                    tenant_health["tenant_config"] = {
                        "name": tenant_config.name,
                        "max_projects": tenant_config.max_projects,
                        "max_storage_mb": tenant_config.max_storage_mb,
                        "rate_limit": tenant_config.rate_limit_requests_per_minute
                    }
            except Exception as e:
                tenant_health["tenant_config_error"] = str(e)
        
        return tenant_health
    
    async def cleanup_tenant_data(self) -> Dict[str, Any]:
        """
        Clean up all data for current tenant project
        WARNING: This is destructive
        """
        if not self.tenant_manager:
            return {"status": "error", "message": "Tenant manager not initialized"}
        
        try:
            results = {}
            
            # Clean up Qdrant collections
            if self.qdrant:
                collection_types = ["code", "docs", "general"]
                for collection_type in collection_types:
                    collection_name = self.tenant_manager.get_collection_name(
                        self.tenant_id, self.original_project_name, collection_type
                    )
                    
                    cleanup_result = await self.qdrant.delete_collection(collection_name)
                    results[f"qdrant_{collection_type}"] = cleanup_result
            
            # Clean up Neo4j data
            if self.neo4j:
                labels = self.tenant_manager.get_neo4j_labels(
                    self.tenant_id, self.original_project_name
                )
                
                cleanup_result = await self.neo4j.delete_tenant_data(labels)
                results["neo4j"] = cleanup_result
            
            logger.warning(f"CLEANED UP tenant data: {self.tenant_id}/{self.original_project_name}")
            
            return {
                "status": "success",
                "message": "Tenant data cleaned up",
                "tenant_id": self.tenant_id,
                "project_name": self.original_project_name,
                "cleanup_results": results
            }
            
        except Exception as e:
            logger.error(f"Tenant cleanup failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "tenant_id": self.tenant_id
            }

# Factory function for creating multitenant containers
async def create_multitenant_container(
    tenant_id: str, 
    project_name: str,
    tenant_manager: TenantManager
) -> MultitenantServiceContainer:
    """
    Factory function to create and initialize multitenant service container
    
    Args:
        tenant_id: Tenant identifier
        project_name: Project name
        tenant_manager: Configured tenant manager
        
    Returns:
        Initialized multitenant service container
    """
    container = MultitenantServiceContainer(tenant_id, project_name)
    
    # Initialize with tenant management
    init_result = await container.initialize_tenant_services(tenant_manager)
    
    if init_result["status"] != "success":
        logger.error(f"Failed to initialize multitenant container: {init_result}")
        raise ValueError(f"Multitenant initialization failed: {init_result['message']}")
    
    # Create tenant collections
    collection_result = await container.create_tenant_collections()
    if collection_result["status"] != "success":
        logger.warning(f"Failed to create some tenant collections: {collection_result}")
    
    return container

# Example usage
async def example_multitenant_usage():
    """Demonstrate multitenant service container"""
    from infrastructure.multitenancy import create_example_tenants
    
    print("🏢 Testing multitenant service container...")
    
    # Create tenant manager
    tenant_manager = await create_example_tenants()
    
    # Create multitenant containers for different tenants
    acme_container = await create_multitenant_container("acme_corp", "backend_service", tenant_manager)
    startup_container = await create_multitenant_container("startup_xyz", "mvp_prototype", tenant_manager)
    
    # Test isolated queries
    acme_results = await acme_container.execute_tenant_query("authentication function")
    print(f"   🔍 ACME search: {acme_results['total_results']} results")
    
    startup_results = await startup_container.execute_tenant_query("authentication function")
    print(f"   🔍 Startup search: {startup_results['total_results']} results")
    
    # Test health checks
    acme_health = await acme_container.get_tenant_health()
    print(f"   💚 ACME health: {acme_health['status']}")
    
    startup_health = await startup_container.get_tenant_health()
    print(f"   💚 Startup health: {startup_health['status']}")
    
    print("✅ Multitenant container test completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_multitenant_usage())