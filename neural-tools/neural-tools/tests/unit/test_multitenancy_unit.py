#!/usr/bin/env python3
"""
Phase 1.3 Unit Tests - Multitenancy Support
Tests tenant isolation, resource limits, and access control
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.multitenancy import MultitenantServiceContainer, TenantConfig, ResourceLimits

class TestMultitenantServiceContainer:
    """Test suite for multitenancy functionality"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic tenant configuration for testing"""
        return TenantConfig(
            tenant_id="test_tenant",
            name="Test Tenant",
            resource_limits=ResourceLimits(
                max_collections=5,
                max_documents=10000,
                max_storage_mb=100,
                max_requests_per_minute=60
            )
        )
    
    @pytest.fixture  
    def premium_config(self):
        """Premium tenant configuration for testing"""
        return TenantConfig(
            tenant_id="premium_tenant", 
            name="Premium Tenant",
            resource_limits=ResourceLimits(
                max_collections=50,
                max_documents=1000000,
                max_storage_mb=10000,
                max_requests_per_minute=6000
            )
        )
    
    @pytest.fixture
    def container(self):
        """Create service container for testing"""
        return MultitenantServiceContainer()
    
    @pytest.mark.asyncio
    async def test_tenant_registration(self, container, basic_config):
        """Test tenant registration and configuration"""
        result = await container.register_tenant(basic_config)
        assert result.success
        assert result.tenant_id == "test_tenant"
        
        # Verify tenant is registered
        tenant = await container.get_tenant_config("test_tenant")
        assert tenant is not None
        assert tenant.tenant_id == "test_tenant"
        assert tenant.name == "Test Tenant"
        assert tenant.resource_limits.max_collections == 5
    
    @pytest.mark.asyncio
    async def test_tenant_isolation(self, container, basic_config, premium_config):
        """Test that tenants are properly isolated"""
        await container.register_tenant(basic_config)
        await container.register_tenant(premium_config)
        
        # Get services for different tenants
        basic_service = await container.get_service("test_tenant", "indexer")
        premium_service = await container.get_service("premium_tenant", "indexer")
        
        assert basic_service is not None
        assert premium_service is not None
        assert basic_service != premium_service  # Different instances
        
        # Verify namespace isolation
        basic_namespace = container._get_namespace("test_tenant")
        premium_namespace = container._get_namespace("premium_tenant")
        assert basic_namespace != premium_namespace
        assert basic_namespace == "test_tenant"
        assert premium_namespace == "premium_tenant"
    
    @pytest.mark.asyncio
    async def test_resource_limits_enforcement(self, container, basic_config):
        """Test that resource limits are properly enforced"""
        await container.register_tenant(basic_config)
        
        # Test collection limit
        for i in range(5):
            result = await container.create_collection("test_tenant", f"collection_{i}")
            assert result.success
        
        # Should fail on 6th collection (exceeds limit of 5)
        result = await container.create_collection("test_tenant", "collection_6")
        assert not result.success
        assert "resource limit" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_document_limit_enforcement(self, container, basic_config):
        """Test document count limits"""
        await container.register_tenant(basic_config)
        await container.create_collection("test_tenant", "test_collection")
        
        # Mock document count check
        with patch.object(container, '_get_document_count', return_value=9999):
            result = await container.add_document("test_tenant", "test_collection", {"content": "test"})
            assert result.success
        
        # Should fail when at limit
        with patch.object(container, '_get_document_count', return_value=10000):
            result = await container.add_document("test_tenant", "test_collection", {"content": "test"}) 
            assert not result.success
            assert "document limit" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, container, basic_config):
        """Test request rate limiting"""
        await container.register_tenant(basic_config)
        
        # Simulate rapid requests
        success_count = 0
        rate_limited_count = 0
        
        for i in range(70):  # More than the 60 per minute limit
            result = await container.check_rate_limit("test_tenant")
            if result.success:
                success_count += 1
            else:
                rate_limited_count += 1
        
        # Should have some rate limiting
        assert success_count <= 60  # Within rate limit
        assert rate_limited_count >= 10  # Some requests rate limited
    
    @pytest.mark.asyncio
    async def test_tenant_not_found(self, container):
        """Test behavior with non-existent tenant"""
        result = await container.get_service("nonexistent_tenant", "indexer")
        assert result is None
        
        config = await container.get_tenant_config("nonexistent_tenant")
        assert config is None
    
    @pytest.mark.asyncio
    async def test_storage_limit_enforcement(self, container, basic_config):
        """Test storage space limits"""
        await container.register_tenant(basic_config)
        
        # Mock storage usage check
        with patch.object(container, '_get_storage_usage_mb', return_value=95):
            result = await container.check_storage_limit("test_tenant", 4)  # 95 + 4 = 99MB (under limit)
            assert result.success
        
        with patch.object(container, '_get_storage_usage_mb', return_value=95):
            result = await container.check_storage_limit("test_tenant", 10)  # 95 + 10 = 105MB (over 100MB limit)
            assert not result.success
            assert "storage limit" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_tenant_operations(self, container, basic_config):
        """Test concurrent operations don't interfere"""
        await container.register_tenant(basic_config)
        
        # Create multiple concurrent requests
        tasks = []
        for i in range(20):
            task = asyncio.create_task(
                container.create_collection("test_tenant", f"concurrent_collection_{i}")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should succeed (within limit), others should fail
        success_count = sum(1 for r in results if isinstance(r, object) and hasattr(r, 'success') and r.success)
        assert success_count <= 5  # Within the collection limit
        assert success_count > 0  # Some should succeed
    
    @pytest.mark.asyncio 
    async def test_tenant_deregistration(self, container, basic_config):
        """Test tenant deregistration and cleanup"""
        await container.register_tenant(basic_config)
        await container.create_collection("test_tenant", "test_collection")
        
        # Verify tenant exists
        config = await container.get_tenant_config("test_tenant")
        assert config is not None
        
        # Deregister tenant
        result = await container.deregister_tenant("test_tenant")
        assert result.success
        
        # Verify tenant is removed
        config = await container.get_tenant_config("test_tenant")
        assert config is None
        
        # Services should no longer be available
        service = await container.get_service("test_tenant", "indexer")
        assert service is None
    
    @pytest.mark.asyncio
    async def test_tenant_metrics_isolation(self, container, basic_config, premium_config):
        """Test that tenant metrics are isolated"""
        await container.register_tenant(basic_config)
        await container.register_tenant(premium_config)
        
        # Perform operations for each tenant
        await container.create_collection("test_tenant", "basic_collection")
        await container.create_collection("premium_tenant", "premium_collection")
        
        # Get metrics for each tenant
        basic_metrics = await container.get_tenant_metrics("test_tenant")
        premium_metrics = await container.get_tenant_metrics("premium_tenant")
        
        assert basic_metrics != premium_metrics
        assert basic_metrics["tenant_id"] == "test_tenant"
        assert premium_metrics["tenant_id"] == "premium_tenant"
        
        # Each should show their own collection count
        assert basic_metrics["collections"] == 1
        assert premium_metrics["collections"] == 1
    
    @pytest.mark.asyncio
    async def test_tenant_configuration_validation(self, container):
        """Test validation of tenant configuration"""
        # Invalid tenant ID (empty)
        invalid_config = TenantConfig(
            tenant_id="",
            name="Invalid Tenant",
            resource_limits=ResourceLimits()
        )
        
        result = await container.register_tenant(invalid_config)
        assert not result.success
        assert "tenant_id" in result.error.lower()
        
        # Invalid resource limits (negative values)
        invalid_config2 = TenantConfig(
            tenant_id="invalid_tenant",
            name="Invalid Tenant",
            resource_limits=ResourceLimits(
                max_collections=-1,
                max_documents=-100
            )
        )
        
        result = await container.register_tenant(invalid_config2)
        assert not result.success
        assert "resource limits" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_service_type_isolation(self, container, basic_config):
        """Test that different service types are isolated"""
        await container.register_tenant(basic_config)
        
        # Get different service types
        indexer_service = await container.get_service("test_tenant", "indexer")
        retriever_service = await container.get_service("test_tenant", "retriever")
        cache_service = await container.get_service("test_tenant", "cache")
        
        # All should be different instances
        assert indexer_service != retriever_service
        assert indexer_service != cache_service
        assert retriever_service != cache_service
        
        # But all should be configured for the same tenant
        assert indexer_service is not None
        assert retriever_service is not None
        assert cache_service is not None

class TestTenantConfig:
    """Test tenant configuration validation and serialization"""
    
    def test_resource_limits_validation(self):
        """Test resource limits validation"""
        # Valid limits
        valid_limits = ResourceLimits(
            max_collections=10,
            max_documents=1000,
            max_storage_mb=100,
            max_requests_per_minute=60
        )
        assert valid_limits.is_valid()
        
        # Invalid limits (negative values)
        invalid_limits = ResourceLimits(
            max_collections=-1,
            max_documents=1000,
            max_storage_mb=100,
            max_requests_per_minute=60
        )
        assert not invalid_limits.is_valid()
    
    def test_tenant_config_serialization(self):
        """Test tenant configuration serialization/deserialization"""
        config = TenantConfig(
            tenant_id="test_tenant",
            name="Test Tenant",
            resource_limits=ResourceLimits(
                max_collections=5,
                max_documents=10000
            )
        )
        
        # Serialize
        serialized = config.to_dict()
        assert serialized["tenant_id"] == "test_tenant"
        assert serialized["name"] == "Test Tenant"
        assert "resource_limits" in serialized
        
        # Deserialize
        deserialized = TenantConfig.from_dict(serialized)
        assert deserialized.tenant_id == config.tenant_id
        assert deserialized.name == config.name
        assert deserialized.resource_limits.max_collections == config.resource_limits.max_collections

if __name__ == "__main__":
    pytest.main([__file__, "-v"])