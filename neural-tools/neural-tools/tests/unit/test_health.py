#!/usr/bin/env python3
"""
Health Check Infrastructure Tests

Tests comprehensive health monitoring:
- Liveness and readiness probes
- Dependency health checkers (Qdrant, Neo4j, Redis, API keys)
- Prometheus metrics collection
- FastAPI endpoint integration
- Error handling and timeouts
- Circuit breaker integration
"""

import pytest
import asyncio
import time
import os
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

# Import components to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.health import (
    HealthChecker,
    HealthConfig,
    HealthStatus,
    HealthCheckResult,
    DependencyHealthChecker,
    QdrantHealthChecker,
    Neo4jHealthChecker,
    RedisHealthChecker,
    APIKeyHealthChecker,
    get_health_checker,
    setup_health_endpoints,
    FASTAPI_AVAILABLE,
    PROMETHEUS_AVAILABLE
)


class TestHealthConfig:
    """Test health configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = HealthConfig()
        
        assert config.qdrant_timeout == 2.0
        assert config.neo4j_timeout == 2.0
        assert config.redis_timeout == 1.0
        assert config.api_key_check_timeout == 0.5
        
        assert config.dependency_check_interval == 30
        assert config.metrics_collection_interval == 10
        
        assert config.enable_circuit_breaker_checks is True
        assert config.required_dependencies == ["qdrant", "neo4j"]
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = HealthConfig(
            qdrant_timeout=3.0,
            neo4j_timeout=1.5,
            required_dependencies=["qdrant", "neo4j", "redis"],
            enable_circuit_breaker_checks=False
        )
        
        assert config.qdrant_timeout == 3.0
        assert config.neo4j_timeout == 1.5
        assert config.required_dependencies == ["qdrant", "neo4j", "redis"]
        assert config.enable_circuit_breaker_checks is False


class TestHealthCheckResult:
    """Test health check result handling"""
    
    def test_health_check_result_creation(self):
        """Test creating health check results"""
        from datetime import datetime
        
        result = HealthCheckResult(
            name="test_dependency",
            status=HealthStatus.HEALTHY,
            message="All good",
            latency_ms=45.5,
            timestamp=datetime.utcnow(),
            metadata={"version": "1.0"}
        )
        
        assert result.name == "test_dependency"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.latency_ms == 45.5
        assert result.metadata["version"] == "1.0"
    
    def test_health_check_result_to_dict(self):
        """Test converting result to dictionary"""
        from datetime import datetime
        
        timestamp = datetime.utcnow()
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.UNHEALTHY,
            message="Failed",
            latency_ms=123.45,
            timestamp=timestamp
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["name"] == "test"
        assert result_dict["status"] == "unhealthy"
        assert result_dict["message"] == "Failed"
        assert result_dict["latency_ms"] == 123.45
        assert result_dict["timestamp"] == timestamp.isoformat()
        assert result_dict["metadata"] == {}


class TestDependencyHealthChecker:
    """Test base dependency health checker"""
    
    class MockHealthChecker(DependencyHealthChecker):
        def __init__(self, name: str, should_pass: bool = True, should_timeout: bool = False):
            super().__init__(name, timeout=0.1)
            self.should_pass = should_pass
            self.should_timeout = should_timeout
        
        async def _perform_health_check(self) -> bool:
            if self.should_timeout:
                await asyncio.sleep(0.2)  # Longer than timeout
            return self.should_pass
    
    @pytest.mark.asyncio
    async def test_successful_health_check(self):
        """Test successful health check"""
        checker = self.MockHealthChecker("test_service", should_pass=True)
        result = await checker.check_health()
        
        assert result.name == "test_service"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "OK"
        assert result.latency_ms < 200  # Should be fast
        assert checker.last_result is not None
    
    @pytest.mark.asyncio
    async def test_failed_health_check(self):
        """Test failed health check"""
        checker = self.MockHealthChecker("test_service", should_pass=False)
        result = await checker.check_health()
        
        assert result.name == "test_service"
        assert result.status == HealthStatus.UNHEALTHY
        assert result.message == "Health check failed"
        assert checker.last_result is not None
    
    @pytest.mark.asyncio
    async def test_timeout_health_check(self):
        """Test health check timeout"""
        checker = self.MockHealthChecker("test_service", should_timeout=True)
        result = await checker.check_health()
        
        assert result.name == "test_service"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Timeout" in result.message
        assert result.latency_ms >= 100  # Should reflect timeout
    
    @pytest.mark.asyncio
    async def test_exception_health_check(self):
        """Test health check with exception"""
        class ExceptionChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                raise ValueError("Test error")
        
        checker = ExceptionChecker("error_service", timeout=1.0)
        result = await checker.check_health()
        
        assert result.name == "error_service"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Error: Test error" in result.message


class TestSpecificHealthCheckers:
    """Test specific dependency health checkers"""
    
    @pytest.mark.asyncio
    async def test_qdrant_health_checker_success(self):
        """Test successful Qdrant health check"""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = {"collections": []}
        
        checker = QdrantHealthChecker(mock_client, timeout=1.0)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        mock_client.get_collections.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_qdrant_health_checker_no_client(self):
        """Test Qdrant health check without client"""
        checker = QdrantHealthChecker(None, timeout=1.0)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_neo4j_health_checker_success(self):
        """Test successful Neo4j health check"""
        mock_client = AsyncMock()
        mock_client.execute_cypher.return_value = [{"health": 1}]
        
        checker = Neo4jHealthChecker(mock_client, timeout=1.0)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        mock_client.execute_cypher.assert_called_once_with("RETURN 1 as health")
    
    @pytest.mark.asyncio
    async def test_neo4j_health_checker_no_client(self):
        """Test Neo4j health check without client"""
        checker = Neo4jHealthChecker(None, timeout=1.0)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_redis_health_checker_success(self):
        """Test successful Redis health check"""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        
        checker = RedisHealthChecker(mock_client, timeout=1.0)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_health_checker_no_client(self):
        """Test Redis health check without client"""
        checker = RedisHealthChecker(None, timeout=1.0)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_api_key_health_checker_with_key(self):
        """Test API key health check with key present"""
        with patch.dict(os.environ, {"TEST_API_KEY": "sk-test-key-123"}):
            checker = APIKeyHealthChecker("TEST_API_KEY", timeout=0.5)
            result = await checker.check_health()
            
            assert result.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_api_key_health_checker_without_key(self):
        """Test API key health check without key"""
        with patch.dict(os.environ, {}, clear=True):
            checker = APIKeyHealthChecker("MISSING_API_KEY", timeout=0.5)
            result = await checker.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_api_key_health_checker_empty_key(self):
        """Test API key health check with empty key"""
        with patch.dict(os.environ, {"EMPTY_API_KEY": "   "}):
            checker = APIKeyHealthChecker("EMPTY_API_KEY", timeout=0.5)
            result = await checker.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY


class TestHealthChecker:
    """Test main health checker functionality"""
    
    def setup_method(self):
        """Setup health checker for testing"""
        self.config = HealthConfig(
            required_dependencies=["qdrant", "neo4j"],
            qdrant_timeout=1.0,
            neo4j_timeout=1.0
        )
        self.health_checker = HealthChecker(self.config)
    
    def test_health_checker_initialization(self):
        """Test health checker initialization"""
        assert self.health_checker.config == self.config
        assert isinstance(self.health_checker.dependency_checkers, dict)
        assert len(self.health_checker.dependency_checkers) == 0
    
    def test_add_dependency_checker(self):
        """Test adding dependency checkers"""
        class TestChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                return True
        
        checker = TestChecker("test_dep", timeout=1.0)
        self.health_checker.add_dependency_checker(checker)
        
        assert "test_dep" in self.health_checker.dependency_checkers
        assert self.health_checker.dependency_checkers["test_dep"] == checker
    
    def test_add_specific_checkers(self):
        """Test adding specific dependency checkers"""
        mock_qdrant = Mock()
        mock_neo4j = Mock()
        mock_redis = Mock()
        
        self.health_checker.add_qdrant_checker(mock_qdrant)
        self.health_checker.add_neo4j_checker(mock_neo4j)
        self.health_checker.add_redis_checker(mock_redis)
        self.health_checker.add_api_key_checker("TEST_API_KEY")
        
        assert "qdrant" in self.health_checker.dependency_checkers
        assert "neo4j" in self.health_checker.dependency_checkers
        assert "redis" in self.health_checker.dependency_checkers
        assert "api_key_test_api_key" in self.health_checker.dependency_checkers
    
    @pytest.mark.asyncio
    async def test_check_dependency_success(self):
        """Test checking specific dependency"""
        class SuccessChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                return True
        
        checker = SuccessChecker("success_dep", timeout=1.0)
        self.health_checker.add_dependency_checker(checker)
        
        result = await self.health_checker.check_dependency("success_dep")
        
        assert result is not None
        assert result.name == "success_dep"
        assert result.status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_check_nonexistent_dependency(self):
        """Test checking non-existent dependency"""
        result = await self.health_checker.check_dependency("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_check_all_dependencies(self):
        """Test checking all dependencies"""
        class MockChecker(DependencyHealthChecker):
            def __init__(self, name: str, healthy: bool):
                super().__init__(name, timeout=0.1)
                self.healthy = healthy
            
            async def _perform_health_check(self) -> bool:
                return self.healthy
        
        # Add multiple checkers
        self.health_checker.add_dependency_checker(MockChecker("dep1", True))
        self.health_checker.add_dependency_checker(MockChecker("dep2", False))
        self.health_checker.add_dependency_checker(MockChecker("dep3", True))
        
        results = await self.health_checker.check_all_dependencies()
        
        assert len(results) == 3
        assert results["dep1"].status == HealthStatus.HEALTHY
        assert results["dep2"].status == HealthStatus.UNHEALTHY
        assert results["dep3"].status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_liveness_probe(self):
        """Test liveness probe"""
        result = await self.health_checker.liveness_probe()
        
        assert result["status"] == "alive"
        assert "timestamp" in result
        assert "version" in result
    
    @pytest.mark.asyncio
    async def test_readiness_probe_ready(self):
        """Test readiness probe when service is ready"""
        class HealthyChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                return True
        
        # Add required dependencies as healthy
        self.health_checker.add_dependency_checker(HealthyChecker("qdrant", timeout=0.1))
        self.health_checker.add_dependency_checker(HealthyChecker("neo4j", timeout=0.1))
        
        result = await self.health_checker.readiness_probe()
        
        assert result["ready"] is True
        assert len(result["failed_required"]) == 0
        assert "checks" in result
        assert "latency_ms" in result
    
    @pytest.mark.asyncio
    async def test_readiness_probe_not_ready(self):
        """Test readiness probe when service is not ready"""
        class UnhealthyChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                return False
        
        # Add required dependencies as unhealthy
        self.health_checker.add_dependency_checker(UnhealthyChecker("qdrant", timeout=0.1))
        self.health_checker.add_dependency_checker(UnhealthyChecker("neo4j", timeout=0.1))
        
        result = await self.health_checker.readiness_probe()
        
        assert result["ready"] is False
        assert len(result["failed_required"]) > 0
        assert "qdrant" in result["failed_required"]
        assert "neo4j" in result["failed_required"]
    
    @pytest.mark.asyncio
    async def test_startup_probe(self):
        """Test startup probe"""
        result = await self.health_checker.startup_probe()
        
        assert "probe_type" in result
        assert result["probe_type"] == "startup"
        assert "ready" in result
    
    def test_get_metrics(self):
        """Test metrics generation"""
        metrics = self.health_checker.get_metrics()
        
        assert isinstance(metrics, str)
        if PROMETHEUS_AVAILABLE:
            assert len(metrics) > 0
        else:
            assert "not available" in metrics
    
    def test_get_health_stats(self):
        """Test health statistics"""
        stats = self.health_checker.get_health_stats()
        
        assert "config" in stats
        assert "dependencies" in stats
        assert "last_overall_check" in stats
        
        assert stats["config"]["required_dependencies"] == ["qdrant", "neo4j"]
        assert "prometheus_available" in stats["config"]
        assert "fastapi_available" in stats["config"]


class TestHealthIntegration:
    """Test integration between health components"""
    
    def test_global_health_checker_singleton(self):
        """Test global health checker singleton behavior"""
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        
        assert checker1 is checker2
        assert isinstance(checker1, HealthChecker)
    
    def test_global_health_checker_with_config(self):
        """Test global health checker with custom config"""
        config = HealthConfig(required_dependencies=["test"])
        checker = get_health_checker(config)
        
        assert isinstance(checker, HealthChecker)
        # Note: Global singleton will use first config, not subsequent ones


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not available")
class TestFastAPIIntegration:
    """Test FastAPI health endpoint integration"""
    
    def test_setup_health_endpoints_no_container(self):
        """Test setting up health endpoints without container"""
        from fastapi import FastAPI
        
        app = FastAPI()
        initial_routes = len(app.routes)
        
        setup_health_endpoints(app, container=None)
        
        # Should add health endpoints
        assert len(app.routes) > initial_routes
    
    def test_setup_health_endpoints_with_container(self):
        """Test setting up health endpoints with container"""
        from fastapi import FastAPI
        
        app = FastAPI()
        
        # Mock container with dependencies
        mock_container = Mock()
        mock_container.qdrant = Mock()
        mock_container.neo4j = Mock()
        mock_container.cache = Mock()
        
        setup_health_endpoints(app, container=mock_container)
        
        # Should have configured health checker with dependencies
        health_checker = get_health_checker()
        assert "qdrant" in health_checker.dependency_checkers
        assert "neo4j" in health_checker.dependency_checkers
        assert "redis" in health_checker.dependency_checkers


class TestHealthPerformance:
    """Test health check performance"""
    
    def setup_method(self):
        """Setup performance test environment"""
        self.health_checker = HealthChecker()
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance is acceptable"""
        class FastChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                await asyncio.sleep(0.001)  # 1ms simulated work
                return True
        
        # Add multiple fast checkers
        for i in range(5):
            checker = FastChecker(f"fast_dep_{i}", timeout=1.0)
            self.health_checker.add_dependency_checker(checker)
        
        # Measure performance
        start_time = time.perf_counter()
        results = await self.health_checker.check_all_dependencies()
        elapsed = time.perf_counter() - start_time
        
        assert len(results) == 5
        assert elapsed < 0.1  # Should complete in <100ms
        assert all(r.status == HealthStatus.HEALTHY for r in results.values())
    
    @pytest.mark.asyncio
    async def test_readiness_probe_performance(self):
        """Test readiness probe performance"""
        class FastChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                await asyncio.sleep(0.002)  # 2ms simulated work
                return True
        
        # Configure required dependencies
        self.health_checker.config.required_dependencies = ["dep1", "dep2"]
        self.health_checker.add_dependency_checker(FastChecker("dep1", timeout=1.0))
        self.health_checker.add_dependency_checker(FastChecker("dep2", timeout=1.0))
        
        # Measure readiness probe performance
        start_time = time.perf_counter()
        result = await self.health_checker.readiness_probe()
        elapsed = time.perf_counter() - start_time
        
        assert result["ready"] is True
        assert elapsed < 0.05  # Should complete in <50ms
        assert result["latency_ms"] < 50
    
    def test_metrics_generation_performance(self):
        """Test metrics generation performance"""
        # Generate metrics multiple times
        iterations = 100
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            metrics = self.health_checker.get_metrics()
            assert isinstance(metrics, str)
        elapsed = time.perf_counter() - start_time
        
        ops_per_second = iterations / elapsed
        
        # Should handle high throughput metrics generation (>1000 ops/sec)
        assert ops_per_second > 1000, f"Metrics generation too slow: {ops_per_second:.1f} ops/sec"


class TestHealthErrorHandling:
    """Test health check error handling"""
    
    def setup_method(self):
        """Setup error handling test environment"""
        self.health_checker = HealthChecker()
    
    @pytest.mark.asyncio
    async def test_dependency_checker_exception_handling(self):
        """Test handling of exceptions in dependency checkers"""
        class ExceptionChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                raise RuntimeError("Simulated failure")
        
        checker = ExceptionChecker("error_dep", timeout=1.0)
        self.health_checker.add_dependency_checker(checker)
        
        result = await self.health_checker.check_dependency("error_dep")
        
        assert result is not None
        assert result.status == HealthStatus.UNHEALTHY
        assert "Error: Simulated failure" in result.message
    
    @pytest.mark.asyncio
    async def test_multiple_dependency_failures(self):
        """Test handling multiple dependency failures"""
        class FailingChecker(DependencyHealthChecker):
            def __init__(self, name: str):
                super().__init__(name, timeout=0.1)
            
            async def _perform_health_check(self) -> bool:
                raise ConnectionError(f"{self.name} connection failed")
        
        # Add multiple failing checkers
        for i in range(3):
            checker = FailingChecker(f"failing_dep_{i}")
            self.health_checker.add_dependency_checker(checker)
        
        results = await self.health_checker.check_all_dependencies()
        
        assert len(results) == 3
        assert all(r.status == HealthStatus.UNHEALTHY for r in results.values())
        assert all("connection failed" in r.message for r in results.values())
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test proper timeout handling"""
        class SlowChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                await asyncio.sleep(0.2)  # 200ms - longer than timeout
                return True
        
        checker = SlowChecker("slow_dep", timeout=0.05)  # 50ms timeout
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "Timeout" in result.message
        assert result.latency_ms >= 50  # Should reflect timeout duration


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_health"
    ])