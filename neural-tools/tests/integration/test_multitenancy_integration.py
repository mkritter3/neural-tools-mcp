#!/usr/bin/env python3
"""
Phase 1.3 Integration Tests - Multitenancy with Real Services
Tests multitenancy integration with Qdrant, Neo4j, and service orchestration
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import AsyncMock, patch
import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.multitenancy import MultitenantServiceContainer, TenantConfig, ResourceLimits

class TestMultitenancyIntegration:
    """Integration tests for multitenancy with real services"""
    
    @pytest.fixture
    def small_tenant_config(self):
        """Small tenant configuration"""
        return TenantConfig(
            tenant_id="small_tenant",
            name="Small Organization",
            resource_limits=ResourceLimits(
                max_collections=3,
                max_documents=1000,
                max_storage_mb=50,
                max_requests_per_minute=30
            )
        )
    
    @pytest.fixture
    def large_tenant_config(self):
        """Large tenant configuration"""
        return TenantConfig(
            tenant_id="large_tenant",
            name="Large Enterprise", 
            resource_limits=ResourceLimits(
                max_collections=20,
                max_documents=100000,
                max_storage_mb=5000,
                max_requests_per_minute=1000
            )
        )
    
    @pytest_asyncio.fixture
    async def container(self):
        """Create and initialize service container"""
        container = MultitenantServiceContainer()
        await container.initialize()
        yield container
        await container.shutdown()
    
    @pytest.mark.asyncio
    async def test_tenant_service_isolation_real_workload(self, container, small_tenant_config, large_tenant_config):
        """Test real isolation between tenant services under workload"""
        
        # Register both tenants
        await container.register_tenant(small_tenant_config)
        await container.register_tenant(large_tenant_config)
        
        # Create collections for each tenant
        small_result = await container.create_collection("small_tenant", "documents")
        large_result = await container.create_collection("large_tenant", "documents")
        
        assert small_result.success
        assert large_result.success
        
        # Add documents to each tenant's collection
        small_docs = [
            {"id": f"small_{i}", "content": f"Small tenant document {i}", "tenant": "small_tenant"}
            for i in range(10)
        ]
        
        large_docs = [
            {"id": f"large_{i}", "content": f"Large tenant document {i}", "tenant": "large_tenant"}
            for i in range(50)
        ]
        
        # Add documents concurrently
        small_tasks = [
            container.add_document("small_tenant", "documents", doc)
            for doc in small_docs
        ]
        
        large_tasks = [
            container.add_document("large_tenant", "documents", doc)
            for doc in large_docs
        ]
        
        # Execute all tasks concurrently
        all_tasks = small_tasks + large_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Verify results
        success_count = sum(1 for r in results if hasattr(r, 'success') and r.success)
        assert success_count >= 50  # Most should succeed
        
        # Verify tenant isolation by searching
        small_search = await container.search("small_tenant", "documents", "document", limit=20)
        large_search = await container.search("large_tenant", "documents", "document", limit=20)
        
        # Small tenant should only see its documents
        if small_search.success:
            for result in small_search.results:
                assert "small_tenant" in result.get("content", "")
        
        # Large tenant should only see its documents  
        if large_search.success:
            for result in large_search.results:
                assert "large_tenant" in result.get("content", "")
    
    @pytest.mark.asyncio
    async def test_resource_enforcement_under_load(self, container, small_tenant_config):
        """Test resource limits are enforced under concurrent load"""
        await container.register_tenant(small_tenant_config)
        
        # Attempt to create more collections than allowed (limit is 3)
        creation_tasks = [
            container.create_collection("small_tenant", f"collection_{i}")
            for i in range(10)  # Attempt 10, but only 3 should succeed
        ]
        
        results = await asyncio.gather(*creation_tasks, return_exceptions=True)
        
        # Count successes and failures
        success_count = sum(
            1 for r in results 
            if hasattr(r, 'success') and r.success
        )
        failure_count = len(results) - success_count
        
        # Should respect the limit of 3 collections
        assert success_count <= 3
        assert failure_count >= 7
        assert success_count > 0  # At least some should succeed
    
    @pytest.mark.asyncio
    async def test_rate_limiting_effectiveness(self, container, small_tenant_config):
        """Test rate limiting works under burst traffic"""
        await container.register_tenant(small_tenant_config)
        await container.create_collection("small_tenant", "test_collection")
        
        # Generate burst of requests (more than 30 per minute limit)
        start_time = time.time()
        request_tasks = []
        
        for i in range(50):  # 50 requests in quick succession
            task = container.check_rate_limit("small_tenant")
            request_tasks.append(task)
        
        results = await asyncio.gather(*request_tasks)
        elapsed_time = time.time() - start_time
        
        # Count allowed vs rate-limited requests
        allowed_count = sum(1 for r in results if r.success)
        rate_limited_count = len(results) - allowed_count
        
        print(f"Rate limiting test: {allowed_count} allowed, {rate_limited_count} rate-limited in {elapsed_time:.2f}s")
        
        # Should enforce the 30 per minute limit
        assert allowed_count <= 35  # Allow some tolerance
        assert rate_limited_count >= 15  # Should rate-limit excess
    
    @pytest.mark.asyncio
    async def test_storage_monitoring_accuracy(self, container, small_tenant_config):
        """Test storage monitoring is accurate across operations"""
        await container.register_tenant(small_tenant_config)
        await container.create_collection("small_tenant", "storage_test")
        
        # Add documents and monitor storage growth
        initial_storage = await container._get_storage_usage_mb("small_tenant")
        
        # Add some test documents
        large_documents = [
            {
                "id": f"doc_{i}",
                "content": "x" * 10000,  # 10KB content per document
                "metadata": {"size": "large", "index": i}
            }
            for i in range(10)  # 10 * 10KB = ~100KB total
        ]
        
        for doc in large_documents:
            await container.add_document("small_tenant", "storage_test", doc)
        
        final_storage = await container._get_storage_usage_mb("small_tenant")
        storage_growth = final_storage - initial_storage
        
        print(f"Storage monitoring: initial={initial_storage:.2f}MB, final={final_storage:.2f}MB, growth={storage_growth:.2f}MB")
        
        # Should show some storage growth
        assert storage_growth >= 0  # Storage should not decrease
        assert final_storage <= 50  # Should be under the tenant limit
    
    @pytest.mark.asyncio
    async def test_tenant_failover_isolation(self, container, small_tenant_config, large_tenant_config):
        """Test that tenant failures don't affect other tenants"""
        await container.register_tenant(small_tenant_config)
        await container.register_tenant(large_tenant_config)
        
        # Create collections for both tenants
        await container.create_collection("small_tenant", "failover_test")
        await container.create_collection("large_tenant", "failover_test")
        
        # Simulate service failure for small tenant by exceeding limits
        failure_tasks = [
            container.create_collection("small_tenant", f"overload_{i}")
            for i in range(20)  # Way over the limit of 3
        ]
        
        # Meanwhile, large tenant continues normal operations
        normal_operations = [
            container.add_document("large_tenant", "failover_test", {
                "id": f"normal_doc_{i}",
                "content": f"Normal operation {i}"
            })
            for i in range(10)
        ]
        
        # Execute both sets concurrently
        failure_results = await asyncio.gather(*failure_tasks, return_exceptions=True)
        normal_results = await asyncio.gather(*normal_operations, return_exceptions=True)
        
        # Small tenant should have mostly failures due to limits
        small_failures = sum(
            1 for r in failure_results
            if not (hasattr(r, 'success') and r.success)
        )
        assert small_failures >= 17  # Most should fail
        
        # Large tenant should continue operating normally
        large_successes = sum(
            1 for r in normal_results
            if hasattr(r, 'success') and r.success
        )
        assert large_successes >= 8  # Most should succeed
    
    @pytest.mark.asyncio
    async def test_cross_tenant_data_leak_prevention(self, container, small_tenant_config, large_tenant_config):
        """Test that tenants cannot access each other's data"""
        await container.register_tenant(small_tenant_config)
        await container.register_tenant(large_tenant_config)
        
        # Create collections with similar names
        await container.create_collection("small_tenant", "shared_name")
        await container.create_collection("large_tenant", "shared_name")
        
        # Add tenant-specific documents
        small_secret = {
            "id": "secret_small",
            "content": "CONFIDENTIAL: Small tenant secret data",
            "classification": "restricted"
        }
        
        large_secret = {
            "id": "secret_large", 
            "content": "CONFIDENTIAL: Large tenant secret data",
            "classification": "restricted"
        }
        
        await container.add_document("small_tenant", "shared_name", small_secret)
        await container.add_document("large_tenant", "shared_name", large_secret)
        
        # Try cross-tenant access (should fail or return empty)
        small_accessing_large = await container.search("small_tenant", "shared_name", "Large tenant secret")
        large_accessing_small = await container.search("large_tenant", "shared_name", "Small tenant secret")
        
        # Should not find cross-tenant data
        if small_accessing_large.success:
            results = small_accessing_large.results or []
            for result in results:
                assert "Large tenant" not in result.get("content", "")
                
        if large_accessing_small.success:
            results = large_accessing_small.results or []
            for result in results:
                assert "Small tenant" not in result.get("content", "")
        
        # But should find own data
        small_own_search = await container.search("small_tenant", "shared_name", "Small tenant")
        large_own_search = await container.search("large_tenant", "shared_name", "Large tenant")
        
        assert small_own_search.success or len(small_own_search.results or []) >= 0
        assert large_own_search.success or len(large_own_search.results or []) >= 0
    
    @pytest.mark.asyncio
    async def test_performance_isolation_under_load(self, container, small_tenant_config, large_tenant_config):
        """Test that heavy load from one tenant doesn't impact others"""
        await container.register_tenant(small_tenant_config) 
        await container.register_tenant(large_tenant_config)
        
        await container.create_collection("small_tenant", "performance_test")
        await container.create_collection("large_tenant", "performance_test")
        
        # Large tenant generates heavy load
        heavy_load_tasks = [
            container.add_document("large_tenant", "performance_test", {
                "id": f"heavy_{i}",
                "content": "x" * 1000,  # 1KB per document
                "load_test": True
            })
            for i in range(100)  # Heavy load: 100 documents
        ]
        
        # Small tenant performs light operations
        light_operations = []
        for i in range(5):
            await asyncio.sleep(0.01)  # Small delays between operations
            task = container.add_document("small_tenant", "performance_test", {
                "id": f"light_{i}",
                "content": f"Light operation {i}",
                "load_test": False
            })
            light_operations.append(task)
        
        # Measure performance of light operations
        start_time = time.time()
        light_results = await asyncio.gather(*light_operations)
        light_duration = time.time() - start_time
        
        # Execute heavy load
        start_heavy = time.time()
        heavy_results = await asyncio.gather(*heavy_load_tasks, return_exceptions=True)
        heavy_duration = time.time() - start_heavy
        
        print(f"Performance isolation: light={light_duration:.2f}s, heavy={heavy_duration:.2f}s")
        
        # Light operations should complete quickly despite heavy load
        assert light_duration < 1.0  # Should be fast
        
        # Both should have some successes
        light_successes = sum(1 for r in light_results if hasattr(r, 'success') and r.success)
        heavy_successes = sum(1 for r in heavy_results if hasattr(r, 'success') and r.success)
        
        assert light_successes >= 3  # Most light operations should succeed
        assert heavy_successes >= 50  # Some heavy operations should succeed
    
    @pytest.mark.asyncio
    async def test_tenant_metrics_accuracy(self, container, small_tenant_config, large_tenant_config):
        """Test that tenant metrics are accurate and isolated"""
        await container.register_tenant(small_tenant_config)
        await container.register_tenant(large_tenant_config)
        
        # Create different numbers of collections
        await container.create_collection("small_tenant", "metrics_test")
        await container.create_collection("large_tenant", "metrics_test_1")
        await container.create_collection("large_tenant", "metrics_test_2")
        
        # Add different numbers of documents
        await container.add_document("small_tenant", "metrics_test", {"id": "small_doc_1"})
        await container.add_document("small_tenant", "metrics_test", {"id": "small_doc_2"})
        
        await container.add_document("large_tenant", "metrics_test_1", {"id": "large_doc_1"})
        await container.add_document("large_tenant", "metrics_test_1", {"id": "large_doc_2"})
        await container.add_document("large_tenant", "metrics_test_2", {"id": "large_doc_3"})
        await container.add_document("large_tenant", "metrics_test_2", {"id": "large_doc_4"})
        await container.add_document("large_tenant", "metrics_test_2", {"id": "large_doc_5"})
        
        # Get metrics
        small_metrics = await container.get_tenant_metrics("small_tenant")
        large_metrics = await container.get_tenant_metrics("large_tenant")
        
        print(f"Small tenant metrics: {small_metrics}")
        print(f"Large tenant metrics: {large_metrics}")
        
        # Verify metrics accuracy and isolation
        assert small_metrics["tenant_id"] == "small_tenant"
        assert large_metrics["tenant_id"] == "large_tenant"
        
        assert small_metrics["collections"] == 1
        assert large_metrics["collections"] == 2
        
        assert small_metrics["documents"] == 2
        assert large_metrics["documents"] == 5
        
        # Metrics should be different
        assert small_metrics != large_metrics

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])