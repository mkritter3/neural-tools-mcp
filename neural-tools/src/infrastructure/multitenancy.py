#!/usr/bin/env python3
"""
Enhanced Multitenancy Support for Neural Tools
Provides tenant isolation, resource management, and cross-tenant security
Following roadmap Phase 1.3 specifications
"""

import asyncio
import hashlib
import logging
import re
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits for a tenant"""
    max_collections: int = 10
    max_documents: int = 100000
    max_storage_mb: int = 1000
    max_requests_per_minute: int = 100
    
    def is_valid(self) -> bool:
        """Check if resource limits are valid"""
        return (self.max_collections >= 0 and
                self.max_documents >= 0 and
                self.max_storage_mb >= 0 and
                self.max_requests_per_minute >= 0)

@dataclass
class TenantConfig:
    """Enhanced configuration for a tenant"""
    tenant_id: str
    name: str
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    allowed_file_types: Set[str] = field(default_factory=lambda: {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java'})
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "resource_limits": {
                "max_collections": self.resource_limits.max_collections,
                "max_documents": self.resource_limits.max_documents,
                "max_storage_mb": self.resource_limits.max_storage_mb,
                "max_requests_per_minute": self.resource_limits.max_requests_per_minute
            },
            "allowed_file_types": list(self.allowed_file_types),
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TenantConfig':
        """Create from dictionary"""
        resource_limits = ResourceLimits(**data.get("resource_limits", {}))
        return cls(
            tenant_id=data["tenant_id"],
            name=data["name"],
            resource_limits=resource_limits,
            allowed_file_types=set(data.get("allowed_file_types", [])),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )

class TenantManager:
    """
    Manages multiple tenants with isolation and resource limits
    Provides secure tenant operations and data segregation
    """
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self.tenant_projects: Dict[str, Set[str]] = {}  # tenant_id -> project_names
        self.project_tenant_map: Dict[str, str] = {}   # project_name -> tenant_id
        self._lock = asyncio.Lock()
        
        # Security patterns
        self.valid_tenant_pattern = re.compile(r'^[a-zA-Z0-9_-]{1,50}$')
        self.valid_project_pattern = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')
        
        logger.info("Initialized tenant manager")
    
    def _validate_tenant_id(self, tenant_id: str) -> bool:
        """Validate tenant ID format"""
        return bool(self.valid_tenant_pattern.match(tenant_id))
    
    def _validate_project_name(self, project_name: str) -> bool:
        """Validate project name format"""
        return bool(self.valid_project_pattern.match(project_name))
    
    def _generate_tenant_namespace(self, tenant_id: str, resource_type: str) -> str:
        """Generate namespaced resource identifier"""
        return f"tenant_{tenant_id}_{resource_type}"
    
    async def create_tenant(self, config: TenantConfig) -> bool:
        """
        Create a new tenant with configuration
        
        Args:
            config: Tenant configuration
            
        Returns:
            True if created successfully
        """
        if not self._validate_tenant_id(config.tenant_id):
            logger.error(f"Invalid tenant ID format: {config.tenant_id}")
            return False
        
        async with self._lock:
            if config.tenant_id in self.tenants:
                logger.warning(f"Tenant {config.tenant_id} already exists")
                return False
            
            self.tenants[config.tenant_id] = config
            self.tenant_projects[config.tenant_id] = set()
            
            logger.info(f"Created tenant: {config.tenant_id} ({config.name})")
            return True
    
    async def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration"""
        async with self._lock:
            return self.tenants.get(tenant_id)
    
    async def list_tenants(self) -> List[TenantConfig]:
        """List all tenants"""
        async with self._lock:
            return list(self.tenants.values())
    
    async def delete_tenant(self, tenant_id: str) -> bool:
        """
        Delete tenant and all associated data
        WARNING: This is destructive and cannot be undone
        """
        async with self._lock:
            if tenant_id not in self.tenants:
                return False
            
            # Remove all projects for this tenant
            projects = self.tenant_projects.get(tenant_id, set())
            for project_name in projects:
                self.project_tenant_map.pop(project_name, None)
            
            # Remove tenant
            del self.tenants[tenant_id]
            del self.tenant_projects[tenant_id]
            
            logger.warning(f"DELETED tenant: {tenant_id} and {len(projects)} projects")
            return True
    
    async def add_project_to_tenant(self, tenant_id: str, project_name: str) -> bool:
        """
        Add project to tenant with validation
        
        Args:
            tenant_id: Tenant identifier
            project_name: Project name to add
            
        Returns:
            True if added successfully
        """
        if not self._validate_project_name(project_name):
            logger.error(f"Invalid project name format: {project_name}")
            return False
        
        async with self._lock:
            # Check tenant exists
            tenant = self.tenants.get(tenant_id)
            if not tenant:
                logger.error(f"Tenant {tenant_id} does not exist")
                return False
            
            # Check project limits
            current_projects = len(self.tenant_projects.get(tenant_id, set()))
            if current_projects >= tenant.max_projects:
                logger.error(f"Tenant {tenant_id} has reached project limit ({tenant.max_projects})")
                return False
            
            # Check if project already belongs to another tenant
            if project_name in self.project_tenant_map:
                existing_tenant = self.project_tenant_map[project_name]
                if existing_tenant != tenant_id:
                    logger.error(f"Project {project_name} already belongs to tenant {existing_tenant}")
                    return False
            
            # Add project
            if tenant_id not in self.tenant_projects:
                self.tenant_projects[tenant_id] = set()
            
            self.tenant_projects[tenant_id].add(project_name)
            self.project_tenant_map[project_name] = tenant_id
            
            logger.info(f"Added project {project_name} to tenant {tenant_id}")
            return True
    
    async def remove_project_from_tenant(self, tenant_id: str, project_name: str) -> bool:
        """Remove project from tenant"""
        async with self._lock:
            if tenant_id not in self.tenant_projects:
                return False
            
            if project_name in self.tenant_projects[tenant_id]:
                self.tenant_projects[tenant_id].discard(project_name)
                self.project_tenant_map.pop(project_name, None)
                
                logger.info(f"Removed project {project_name} from tenant {tenant_id}")
                return True
            
            return False
    
    async def get_tenant_for_project(self, project_name: str) -> Optional[str]:
        """Get tenant ID for a project"""
        async with self._lock:
            return self.project_tenant_map.get(project_name)
    
    async def get_projects_for_tenant(self, tenant_id: str) -> Set[str]:
        """Get all projects for a tenant"""
        async with self._lock:
            return self.tenant_projects.get(tenant_id, set()).copy()
    
    def get_collection_name(self, tenant_id: str, project_name: str, collection_type: str) -> str:
        """
        Generate tenant-isolated collection name
        
        Args:
            tenant_id: Tenant identifier
            project_name: Project name
            collection_type: Type of collection (code, docs, general)
            
        Returns:
            Namespaced collection name
        """
        # Ensure tenant owns project
        if self.project_tenant_map.get(project_name) != tenant_id:
            raise ValueError(f"Project {project_name} does not belong to tenant {tenant_id}")
        
        return f"tenant_{tenant_id}_project_{project_name}_{collection_type}"
    
    def get_neo4j_labels(self, tenant_id: str, project_name: str) -> Dict[str, str]:
        """
        Generate tenant-isolated Neo4j labels
        
        Returns:
            Dictionary of label mappings
        """
        if self.project_tenant_map.get(project_name) != tenant_id:
            raise ValueError(f"Project {project_name} does not belong to tenant {tenant_id}")
        
        return {
            'File': f'File_T{tenant_id}_P{project_name}',
            'CodeChunk': f'CodeChunk_T{tenant_id}_P{project_name}',
            'Module': f'Module_T{tenant_id}_P{project_name}',
            'Function': f'Function_T{tenant_id}_P{project_name}',
            'Class': f'Class_T{tenant_id}_P{project_name}'
        }

class MultitenantQueryFilter:
    """
    Provides tenant-aware query filtering and access control
    Ensures queries only access data from authorized tenants
    """
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
    
    async def validate_access(self, tenant_id: str, project_name: str) -> bool:
        """Validate tenant has access to project"""
        project_tenant = await self.tenant_manager.get_tenant_for_project(project_name)
        return project_tenant == tenant_id
    
    async def get_accessible_collections(self, tenant_id: str, collection_types: List[str]) -> List[str]:
        """
        Get list of collections accessible to tenant
        
        Args:
            tenant_id: Tenant identifier
            collection_types: Types of collections needed
            
        Returns:
            List of collection names tenant can access
        """
        collections = []
        projects = await self.tenant_manager.get_projects_for_tenant(tenant_id)
        
        for project_name in projects:
            for collection_type in collection_types:
                collection_name = self.tenant_manager.get_collection_name(
                    tenant_id, project_name, collection_type
                )
                collections.append(collection_name)
        
        return collections
    
    def create_qdrant_filter(self, tenant_id: str, project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create Qdrant filter for tenant isolation
        
        Args:
            tenant_id: Tenant identifier
            project_name: Optional project filter
            
        Returns:
            Qdrant filter conditions
        """
        conditions = [
            {"key": "tenant_id", "match": {"value": tenant_id}}
        ]
        
        if project_name:
            conditions.append(
                {"key": "project_name", "match": {"value": project_name}}
            )
        
        return {
            "must": conditions
        }
    
    def create_neo4j_filter(self, tenant_id: str, project_name: Optional[str] = None) -> str:
        """
        Create Neo4j WHERE clause for tenant isolation
        
        Args:
            tenant_id: Tenant identifier
            project_name: Optional project filter
            
        Returns:
            WHERE clause string
        """
        conditions = [f"n.tenant_id = '{tenant_id}'"]
        
        if project_name:
            conditions.append(f"n.project_name = '{project_name}'")
        
        return f"WHERE {' AND '.join(conditions)}"

class TenantResourceMonitor:
    """
    Monitors and enforces tenant resource limits
    Tracks usage and prevents abuse
    """
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        self.usage_tracking: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def track_storage_usage(self, tenant_id: str, project_name: str, size_bytes: int):
        """Track storage usage for tenant"""
        async with self._lock:
            if tenant_id not in self.usage_tracking:
                self.usage_tracking[tenant_id] = {
                    'storage_bytes': 0,
                    'requests_count': 0,
                    'last_request': datetime.now()
                }
            
            self.usage_tracking[tenant_id]['storage_bytes'] += size_bytes
            
            # Check limits
            tenant = await self.tenant_manager.get_tenant(tenant_id)
            if tenant:
                max_bytes = tenant.max_storage_mb * 1024 * 1024
                current_bytes = self.usage_tracking[tenant_id]['storage_bytes']
                
                if current_bytes > max_bytes:
                    logger.warning(f"Tenant {tenant_id} exceeded storage limit: {current_bytes / 1024 / 1024:.1f}MB")
                    return False
            
            return True
    
    async def track_request(self, tenant_id: str) -> bool:
        """Track API request and check rate limits"""
        async with self._lock:
            current_time = datetime.now()
            
            if tenant_id not in self.usage_tracking:
                self.usage_tracking[tenant_id] = {
                    'storage_bytes': 0,
                    'requests_count': 0,
                    'last_request': current_time
                }
            
            # Reset counter if more than a minute has passed
            last_request = self.usage_tracking[tenant_id]['last_request']
            if (current_time - last_request).total_seconds() > 60:
                self.usage_tracking[tenant_id]['requests_count'] = 0
            
            self.usage_tracking[tenant_id]['requests_count'] += 1
            self.usage_tracking[tenant_id]['last_request'] = current_time
            
            # Check rate limit
            tenant = await self.tenant_manager.get_tenant(tenant_id)
            if tenant:
                if self.usage_tracking[tenant_id]['requests_count'] > tenant.rate_limit_requests_per_minute:
                    logger.warning(f"Tenant {tenant_id} exceeded rate limit")
                    return False
            
            return True
    
    async def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get current usage statistics for tenant"""
        async with self._lock:
            usage = self.usage_tracking.get(tenant_id, {})
            tenant = await self.tenant_manager.get_tenant(tenant_id)
            
            if not tenant:
                return {}
            
            storage_mb = usage.get('storage_bytes', 0) / 1024 / 1024
            storage_percent = (storage_mb / tenant.max_storage_mb) * 100
            
            return {
                'tenant_id': tenant_id,
                'storage_used_mb': storage_mb,
                'storage_limit_mb': tenant.max_storage_mb,
                'storage_usage_percent': storage_percent,
                'requests_this_minute': usage.get('requests_count', 0),
                'rate_limit': tenant.rate_limit_requests_per_minute,
                'projects_count': len(await self.tenant_manager.get_projects_for_tenant(tenant_id)),
                'projects_limit': tenant.max_projects
            }

# Example usage and integration
async def create_example_tenants():
    """Create example tenant configuration"""
    tenant_manager = TenantManager()
    
    # Create tenants
    await tenant_manager.create_tenant(TenantConfig(
        tenant_id="acme_corp",
        name="ACME Corporation",
        max_projects=20,
        max_storage_mb=5000,
        rate_limit_requests_per_minute=200
    ))
    
    await tenant_manager.create_tenant(TenantConfig(
        tenant_id="startup_xyz",
        name="Startup XYZ",
        max_projects=5,
        max_storage_mb=1000,
        rate_limit_requests_per_minute=50
    ))
    
    # Add projects
    await tenant_manager.add_project_to_tenant("acme_corp", "backend_service")
    await tenant_manager.add_project_to_tenant("acme_corp", "frontend_app")
    await tenant_manager.add_project_to_tenant("startup_xyz", "mvp_prototype")
    
    return tenant_manager

async def demo_multitenancy():
    """Demonstrate multitenancy features"""
    print("🏢 Testing multitenancy system...")
    
    # Create tenant manager and example tenants
    tenant_manager = await create_example_tenants()
    query_filter = MultitenantQueryFilter(tenant_manager)
    resource_monitor = TenantResourceMonitor(tenant_manager)
    
    # Test access validation
    valid = await query_filter.validate_access("acme_corp", "backend_service")
    print(f"   ✅ ACME can access backend_service: {valid}")
    
    invalid = await query_filter.validate_access("startup_xyz", "backend_service")
    print(f"   ❌ Startup XYZ cannot access backend_service: {not invalid}")
    
    # Test collection name generation
    collection_name = tenant_manager.get_collection_name("acme_corp", "backend_service", "code")
    print(f"   📦 Collection name: {collection_name}")
    
    # Test Neo4j labels
    labels = tenant_manager.get_neo4j_labels("acme_corp", "backend_service")
    print(f"   🏷️  Neo4j labels: {labels}")
    
    # Test resource tracking
    await resource_monitor.track_storage_usage("acme_corp", "backend_service", 1024 * 1024)  # 1MB
    await resource_monitor.track_request("acme_corp")
    
    usage = await resource_monitor.get_tenant_usage("acme_corp")
    print(f"   📊 ACME usage: {usage['storage_used_mb']:.1f}MB, {usage['requests_this_minute']} requests")
    
    # Test Qdrant filter
    qdrant_filter = query_filter.create_qdrant_filter("acme_corp", "backend_service")
    print(f"   🔍 Qdrant filter: {qdrant_filter}")
    
    # Test Neo4j filter
    neo4j_filter = query_filter.create_neo4j_filter("acme_corp", "backend_service")
    print(f"   🔍 Neo4j filter: {neo4j_filter}")
    
    print("✅ Multitenancy test completed")

@dataclass 
class TenantResult:
    """Result of a tenant operation"""
    success: bool
    tenant_id: Optional[str] = None
    error: Optional[str] = None
    results: Optional[List[Any]] = None

class MultitenantServiceContainer:
    """
    Service container that manages tenant-isolated services
    Provides resource limits, rate limiting, and data isolation
    """
    
    def __init__(self):
        self.tenants: Dict[str, TenantConfig] = {}
        self.services: Dict[str, Dict[str, Any]] = {}  # tenant_id -> service_type -> service
        self.rate_limits: Dict[str, List[float]] = {}  # tenant_id -> request_timestamps
        self.storage_usage: Dict[str, float] = {}  # tenant_id -> storage_mb
        self.collection_counts: Dict[str, int] = {}  # tenant_id -> collection_count
        self.document_counts: Dict[str, int] = {}  # tenant_id -> document_count
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the service container"""
        logger.info("Initialized multitenant service container")
    
    async def shutdown(self):
        """Shutdown and cleanup resources"""
        logger.info("Shutdown multitenant service container")
    
    async def register_tenant(self, config: TenantConfig) -> TenantResult:
        """Register a new tenant"""
        if not config.tenant_id or not config.tenant_id.strip():
            return TenantResult(success=False, error="Invalid tenant_id")
        
        if not config.resource_limits.is_valid():
            return TenantResult(success=False, error="Invalid resource limits")
        
        async with self._lock:
            if config.tenant_id in self.tenants:
                return TenantResult(success=False, error="Tenant already exists")
            
            self.tenants[config.tenant_id] = config
            self.services[config.tenant_id] = {}
            self.rate_limits[config.tenant_id] = []
            self.storage_usage[config.tenant_id] = 0.0
            self.collection_counts[config.tenant_id] = 0
            self.document_counts[config.tenant_id] = 0
            
            logger.info(f"Registered tenant: {config.tenant_id}")
            return TenantResult(success=True, tenant_id=config.tenant_id)
    
    async def deregister_tenant(self, tenant_id: str) -> TenantResult:
        """Deregister a tenant and cleanup resources"""
        async with self._lock:
            if tenant_id not in self.tenants:
                return TenantResult(success=False, error="Tenant not found")
            
            # Cleanup tenant data
            del self.tenants[tenant_id]
            self.services.pop(tenant_id, None)
            self.rate_limits.pop(tenant_id, None)
            self.storage_usage.pop(tenant_id, None)
            self.collection_counts.pop(tenant_id, None)
            self.document_counts.pop(tenant_id, None)
            
            logger.info(f"Deregistered tenant: {tenant_id}")
            return TenantResult(success=True, tenant_id=tenant_id)
    
    async def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration"""
        return self.tenants.get(tenant_id)
    
    def _get_namespace(self, tenant_id: str) -> str:
        """Get namespace for tenant resources"""
        return tenant_id
    
    async def get_service(self, tenant_id: str, service_type: str) -> Optional[Any]:
        """Get a service instance for a tenant"""
        if tenant_id not in self.tenants:
            return None
        
        # Create mock service if not exists
        if tenant_id not in self.services:
            self.services[tenant_id] = {}
        
        if service_type not in self.services[tenant_id]:
            # Create a simple mock service for testing
            self.services[tenant_id][service_type] = MockService(tenant_id, service_type)
        
        return self.services[tenant_id][service_type]
    
    async def create_collection(self, tenant_id: str, collection_name: str) -> TenantResult:
        """Create a collection for a tenant"""
        if tenant_id not in self.tenants:
            return TenantResult(success=False, error="Tenant not found")
        
        config = self.tenants[tenant_id]
        current_count = self.collection_counts.get(tenant_id, 0)
        
        if current_count >= config.resource_limits.max_collections:
            return TenantResult(success=False, error="Collection resource limit exceeded")
        
        self.collection_counts[tenant_id] = current_count + 1
        logger.debug(f"Created collection {collection_name} for tenant {tenant_id}")
        return TenantResult(success=True, tenant_id=tenant_id)
    
    async def add_document(self, tenant_id: str, collection_name: str, document: Dict[str, Any]) -> TenantResult:
        """Add a document to a tenant's collection"""
        if tenant_id not in self.tenants:
            return TenantResult(success=False, error="Tenant not found")
        
        config = self.tenants[tenant_id]
        current_count = await self._get_document_count(tenant_id)
        
        if current_count >= config.resource_limits.max_documents:
            return TenantResult(success=False, error="Document limit exceeded")
        
        self.document_counts[tenant_id] = current_count + 1
        logger.debug(f"Added document to {collection_name} for tenant {tenant_id}")
        return TenantResult(success=True, tenant_id=tenant_id)
    
    async def _get_document_count(self, tenant_id: str) -> int:
        """Get document count for tenant"""
        return self.document_counts.get(tenant_id, 0)
    
    async def check_rate_limit(self, tenant_id: str) -> TenantResult:
        """Check if request is within rate limit"""
        if tenant_id not in self.tenants:
            return TenantResult(success=False, error="Tenant not found")
        
        config = self.tenants[tenant_id]
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        if tenant_id not in self.rate_limits:
            self.rate_limits[tenant_id] = []
        
        self.rate_limits[tenant_id] = [
            t for t in self.rate_limits[tenant_id] 
            if current_time - t < 60  # Last minute
        ]
        
        # Check if within limit
        if len(self.rate_limits[tenant_id]) >= config.resource_limits.max_requests_per_minute:
            return TenantResult(success=False, error="Rate limit exceeded")
        
        # Add current request
        self.rate_limits[tenant_id].append(current_time)
        return TenantResult(success=True, tenant_id=tenant_id)
    
    async def check_storage_limit(self, tenant_id: str, additional_mb: float) -> TenantResult:
        """Check if storage would exceed limit"""
        if tenant_id not in self.tenants:
            return TenantResult(success=False, error="Tenant not found")
        
        config = self.tenants[tenant_id]
        current_usage = await self._get_storage_usage_mb(tenant_id)
        
        if current_usage + additional_mb > config.resource_limits.max_storage_mb:
            return TenantResult(success=False, error="Storage limit exceeded")
        
        return TenantResult(success=True, tenant_id=tenant_id)
    
    async def _get_storage_usage_mb(self, tenant_id: str) -> float:
        """Get storage usage for tenant"""
        return self.storage_usage.get(tenant_id, 0)
    
    async def search(self, tenant_id: str, collection_name: str, query: str, limit: int = 10) -> TenantResult:
        """Search within tenant's data"""
        if tenant_id not in self.tenants:
            return TenantResult(success=False, error="Tenant not found")
        
        # Mock search results that respect tenant isolation
        results = [
            {"id": f"{tenant_id}_result_{i}", "content": f"Mock result {i} for {tenant_id}"}
            for i in range(min(limit, 3))
        ]
        
        return TenantResult(success=True, tenant_id=tenant_id, results=results)
    
    async def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get metrics for a tenant"""
        if tenant_id not in self.tenants:
            return {}
        
        return {
            "tenant_id": tenant_id,
            "collections": self.collection_counts.get(tenant_id, 0),
            "documents": self.document_counts.get(tenant_id, 0),
            "storage_mb": self.storage_usage.get(tenant_id, 0),
            "requests_last_minute": len(self.rate_limits.get(tenant_id, []))
        }

class MockService:
    """Mock service for testing"""
    def __init__(self, tenant_id: str, service_type: str):
        self.tenant_id = tenant_id
        self.service_type = service_type

# Update TenantConfig to use ResourceLimits
@dataclass
class TenantConfig:
    """Enhanced configuration for a tenant"""
    tenant_id: str
    name: str
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    allowed_file_types: Set[str] = field(default_factory=lambda: {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java'})
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "resource_limits": {
                "max_collections": self.resource_limits.max_collections,
                "max_documents": self.resource_limits.max_documents,
                "max_storage_mb": self.resource_limits.max_storage_mb,
                "max_requests_per_minute": self.resource_limits.max_requests_per_minute
            },
            "allowed_file_types": list(self.allowed_file_types),
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TenantConfig':
        """Create from dictionary"""
        resource_limits = ResourceLimits(**data.get("resource_limits", {}))
        return cls(
            tenant_id=data["tenant_id"],
            name=data["name"],
            resource_limits=resource_limits,
            allowed_file_types=set(data.get("allowed_file_types", [])),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_multitenancy())