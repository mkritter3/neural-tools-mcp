#!/usr/bin/env python3
"""
Real Integration Tests for Health Infrastructure

Tests against ACTUAL running services:
- Neo4j database connectivity and queries
- Qdrant vector database operations  
- Redis caching (if available)
- API endpoints with real HTTP requests
- Prometheus metrics collection
- End-to-end health monitoring

These tests require Docker Compose services to be running.
Run: docker-compose up -d neo4j qdrant before testing.
"""

import pytest
import asyncio
import aiohttp
import time
import os
from typing import Dict, Any
from datetime import datetime, timezone

# Real service clients (not mocks!)
try:
    import neo4j
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, CreateCollection
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from infrastructure.health import (
    HealthChecker, 
    HealthConfig,
    Neo4jHealthChecker,
    QdrantHealthChecker,
    APIKeyHealthChecker,
    HealthStatus
)

# Real service configurations (not mocked!)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")  
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag-password")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Test configuration
INTEGRATION_TIMEOUT = 30.0  # Generous timeout for real services
TEST_COLLECTION_NAME = "health_test_collection"


@pytest.mark.integration
class TestRealNeo4jHealthChecker:
    """Test health checker against actual Neo4j database"""
    
    @pytest.fixture(scope="class")
    def neo4j_driver(self):
        """Connect to real Neo4j instance"""
        if not NEO4J_AVAILABLE:
            pytest.skip("Neo4j driver not available")
            
        driver = None
        try:
            driver = GraphDatabase.driver(
                NEO4J_URI, 
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                max_connection_lifetime=200
            )
            # Test connection
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                assert result.single()["test"] == 1
                
            yield driver
        except Exception as e:
            pytest.skip(f"Cannot connect to Neo4j: {e}")
        finally:
            if driver:
                driver.close()
    
    @pytest.mark.asyncio
    async def test_neo4j_health_check_real_connection(self, neo4j_driver):
        """Test health check against real Neo4j database"""
        checker = Neo4jHealthChecker(neo4j_driver, timeout=INTEGRATION_TIMEOUT)
        
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "neo4j"
        assert "response_time_ms" in result.metadata
        assert result.metadata["response_time_ms"] < INTEGRATION_TIMEOUT * 1000
        assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.asyncio 
    async def test_neo4j_health_check_with_query(self, neo4j_driver):
        """Test health check with actual Neo4j query execution"""
        checker = Neo4jHealthChecker(neo4j_driver, timeout=INTEGRATION_TIMEOUT)
        
        # Execute health check multiple times to test consistency
        results = []
        for _ in range(3):
            result = await checker.check_health()
            results.append(result)
            await asyncio.sleep(0.1)
        
        # All checks should pass
        for result in results:
            assert result.status == HealthStatus.HEALTHY
            assert result.metadata["response_time_ms"] > 0
        
        # Response times should be reasonable (< 5 seconds)
        avg_response_time = sum(r.metadata["response_time_ms"] for r in results) / len(results)
        assert avg_response_time < 5000, f"Average response time too high: {avg_response_time}ms"


@pytest.mark.integration
class TestRealQdrantHealthChecker:
    """Test health checker against actual Qdrant vector database"""
    
    @pytest.fixture(scope="class")
    def qdrant_client(self):
        """Connect to real Qdrant instance"""
        if not QDRANT_AVAILABLE:
            pytest.skip("Qdrant client not available")
            
        client = None
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=INTEGRATION_TIMEOUT)
            
            # Test connection by listing collections
            collections = client.get_collections()
            assert collections is not None
            
            yield client
        except Exception as e:
            pytest.skip(f"Cannot connect to Qdrant: {e}")
        finally:
            if client:
                try:
                    client.close()
                except:
                    pass  # Ignore cleanup errors
    
    @pytest.mark.asyncio
    async def test_qdrant_health_check_real_connection(self, qdrant_client):
        """Test health check against real Qdrant database"""
        # Convert sync client to async-compatible checker
        checker = QdrantHealthChecker(qdrant_client, timeout=INTEGRATION_TIMEOUT)
        
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "qdrant"
        assert result.latency_ms > 0  # Should have measured latency
        assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_qdrant_collection_operations(self, qdrant_client):
        """Test health check with real Qdrant collection operations"""
        checker = QdrantHealthChecker(qdrant_client, timeout=INTEGRATION_TIMEOUT)
        
        # Clean up test collection if it exists
        try:
            qdrant_client.delete_collection(TEST_COLLECTION_NAME)
        except:
            pass  # Collection might not exist
        
        # Create test collection
        qdrant_client.create_collection(
            collection_name=TEST_COLLECTION_NAME,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )
        
        try:
            # Test health check with collection present
            result = await checker.check_health()
            assert result.status == HealthStatus.HEALTHY
            
            # Verify collection exists
            collections = qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            assert TEST_COLLECTION_NAME in collection_names
            
        finally:
            # Cleanup test collection
            try:
                qdrant_client.delete_collection(TEST_COLLECTION_NAME)
            except:
                pass  # Ignore cleanup errors


@pytest.mark.integration
class TestRealRedisHealthChecker:
    """Test health checker against actual Redis instance (if available)"""
    
    @pytest.fixture
    def redis_client(self, event_loop):
        """Connect to real Redis instance if available"""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis client not available")
            
        async def _get_redis_client():
            client = None
            try:
                # Use async Redis client
                client = redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", decode_responses=True)
                
                # Test connection
                await client.ping()
                return client
                
            except Exception as e:
                if client:
                    await client.aclose()
                pytest.skip(f"Cannot connect to Redis: {e}")
        
        client = event_loop.run_until_complete(_get_redis_client())
        
        yield client
        
        # Cleanup
        async def _cleanup():
            await client.aclose()
        
        event_loop.run_until_complete(_cleanup())
    
    @pytest.mark.asyncio
    async def test_redis_health_check_real_connection(self, redis_client):
        """Test health check against real Redis instance"""
        from infrastructure.health import RedisHealthChecker
        
        checker = RedisHealthChecker(redis_client, timeout=INTEGRATION_TIMEOUT)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "redis"
        assert result.latency_ms > 0  # Should have measured latency
        assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_redis_operations(self, redis_client):
        """Test health check with real Redis operations"""
        from infrastructure.health import RedisHealthChecker
        
        checker = RedisHealthChecker(redis_client, timeout=INTEGRATION_TIMEOUT)
        
        # Set and get a test value
        test_key = "health_test_key"
        test_value = "health_test_value"
        
        await redis_client.set(test_key, test_value, ex=60)  # 60 second expiry
        retrieved_value = await redis_client.get(test_key)
        assert retrieved_value == test_value
        
        # Test health check
        result = await checker.check_health()
        assert result.status == HealthStatus.HEALTHY
        
        # Cleanup
        await redis_client.delete(test_key)


@pytest.mark.integration
class TestRealApiHealthChecker:
    """Test API health checks against real HTTP endpoints"""
    
    @pytest.mark.asyncio
    async def test_api_key_health_checker_real_validation(self):
        """Test API key validation with real configuration"""
        # Test with environment variable API key - set a test value if not present
        if "TEST_API_KEY" not in os.environ:
            os.environ["TEST_API_KEY"] = "test-key-123"
        
        checker = APIKeyHealthChecker("TEST_API_KEY", timeout=INTEGRATION_TIMEOUT)
        result = await checker.check_health()
        
        # Should be healthy if key exists (even if dummy)
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "api_key_test_api_key"
        assert isinstance(result.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_http_endpoint_health_check(self):
        """Test health check against real HTTP endpoints"""
        # Test against localhost services if available
        test_endpoints = [
            "http://localhost:6333/health",  # Qdrant health endpoint
            "http://localhost:7474/",        # Neo4j browser (if accessible)
        ]
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=INTEGRATION_TIMEOUT)) as session:
            for endpoint in test_endpoints:
                try:
                    async with session.get(endpoint) as response:
                        if response.status == 200:
                            print(f"✅ {endpoint} is healthy (status: {response.status})")
                        else:
                            print(f"⚠️  {endpoint} returned status: {response.status}")
                except Exception as e:
                    print(f"❌ {endpoint} failed: {e}")


@pytest.mark.integration
class TestRealHealthCheckerIntegration:
    """Test complete health checker integration with real services"""
    
    @pytest.fixture
    def real_health_checker(self, event_loop):
        """Create health checker with real service connections"""
        def _create_health_checker():
            config = HealthConfig(
                required_dependencies=["neo4j", "qdrant"],
                neo4j_timeout=INTEGRATION_TIMEOUT,
                qdrant_timeout=INTEGRATION_TIMEOUT,
                api_key_check_timeout=INTEGRATION_TIMEOUT
            )
            
            checker = HealthChecker(config)
            
            # Add real Neo4j checker if available
            if NEO4J_AVAILABLE:
                try:
                    driver = GraphDatabase.driver(
                        NEO4J_URI, 
                        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
                    )
                    neo4j_checker = Neo4jHealthChecker(driver, timeout=INTEGRATION_TIMEOUT)
                    checker.add_dependency_checker(neo4j_checker)
                except Exception as e:
                    print(f"Could not add Neo4j checker: {e}")
            
            # Add real Qdrant checker if available  
            if QDRANT_AVAILABLE:
                try:
                    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                    qdrant_checker = QdrantHealthChecker(client, timeout=INTEGRATION_TIMEOUT)
                    checker.add_dependency_checker(qdrant_checker)
                except Exception as e:
                    print(f"Could not add Qdrant checker: {e}")
            
            return checker

        # Create health checker synchronously
        checker = _create_health_checker()
        
        yield checker
        
        # Cleanup - handle async close properly
        async def _cleanup():
            try:
                if hasattr(checker, 'close'):
                    await checker.close()
            except Exception:
                pass
        
        # Run cleanup with event loop
        if hasattr(checker, 'close'):
            event_loop.run_until_complete(_cleanup())
    
    @pytest.mark.asyncio
    async def test_real_readiness_probe(self, real_health_checker):
        """Test readiness probe with real service dependencies"""
        result = await real_health_checker.readiness_probe()
        
        # Log results for debugging
        print(f"Readiness probe result: {result}")
        
        # Should contain readiness status
        assert "ready" in result
        assert isinstance(result["ready"], bool)
        assert "checks" in result
        assert "timestamp" in result
        
        # If services are running, should be ready
        if result["ready"]:
            assert len(result["checks"]) > 0
            for dep_name, dep_status in result["checks"].items():
                assert "status" in dep_status
                if dep_status["status"] == "healthy":
                    assert "metadata" in dep_status
                    assert "response_time_ms" in dep_status["metadata"]
    
    @pytest.mark.asyncio
    async def test_real_liveness_probe(self, real_health_checker):
        """Test liveness probe (should always pass)"""
        result = await real_health_checker.liveness_probe()
        
        assert result["status"] == "alive"
        assert "timestamp" in result
        assert "version" in result
    
    @pytest.mark.asyncio
    async def test_real_startup_probe(self, real_health_checker):
        """Test startup probe with real initialization checks"""
        result = await real_health_checker.startup_probe()
        
        assert "ready" in result
        assert "probe_type" in result
        assert result["probe_type"] == "startup"
        assert "timestamp" in result


@pytest.mark.integration
class TestRealPrometheusMetrics:
    """Test Prometheus metrics collection from real health checks"""
    
    @pytest.mark.asyncio
    async def test_real_metrics_collection(self):
        """Test that real health checks generate Prometheus metrics"""
        config = HealthConfig(
            required_dependencies=["qdrant"], 
            qdrant_timeout=INTEGRATION_TIMEOUT
        )
        checker = HealthChecker(config)
        
        # Add real Qdrant checker if available
        if QDRANT_AVAILABLE:
            try:
                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                qdrant_checker = QdrantHealthChecker(client, timeout=INTEGRATION_TIMEOUT)
                checker.add_dependency_checker(qdrant_checker)
                
                # Perform several health checks
                for i in range(3):
                    await checker.check_dependency("qdrant")
                    await asyncio.sleep(0.5)
                
                # Get metrics
                metrics = await checker.get_metrics()
                assert len(metrics) > 0
                
                # Should contain Prometheus format
                metrics_text = "\n".join(metrics)
                assert "health_checks_total" in metrics_text
                assert "health_check_duration_seconds" in metrics_text
                
            except Exception as e:
                pytest.skip(f"Could not test Qdrant metrics: {e}")
        else:
            pytest.skip("Qdrant not available for metrics testing")


@pytest.mark.integration
class TestRealEndToEndScenarios:
    """End-to-end integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_service_failure_detection(self):
        """Test health monitoring during simulated service failures"""
        config = HealthConfig(
            required_dependencies=["fake_service"],
            neo4j_timeout=1.0,  # Short timeout for faster failure detection
        )
        checker = HealthChecker(config)
        
        # Add a checker for non-existent service
        from infrastructure.health import DependencyHealthChecker
        
        class FailingServiceChecker(DependencyHealthChecker):
            async def _perform_health_check(self):
                await asyncio.sleep(0.1)
                # Raise an exception with our custom message so it gets formatted as "Error: <message>"
                raise Exception("Service intentionally unavailable")
        
        failing_checker = FailingServiceChecker("fake_service", timeout=1.0)
        checker.add_dependency_checker(failing_checker)
        
        # Check that readiness probe correctly detects failure
        result = await checker.readiness_probe()
        
        assert result["ready"] is False
        assert "fake_service" in result["checks"]
        assert result["checks"]["fake_service"]["status"] == "unhealthy"
        assert "Service intentionally unavailable" in result["checks"]["fake_service"]["message"]
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test health monitoring performance with concurrent requests"""
        config = HealthConfig(
            required_dependencies=["qdrant"] if QDRANT_AVAILABLE else [],
            qdrant_timeout=INTEGRATION_TIMEOUT
        )
        checker = HealthChecker(config)
        
        if QDRANT_AVAILABLE:
            try:
                client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                qdrant_checker = QdrantHealthChecker(client, timeout=INTEGRATION_TIMEOUT)
                checker.add_dependency_checker(qdrant_checker)
            except Exception as e:
                pytest.skip(f"Could not connect to Qdrant for load testing: {e}")
        else:
            pytest.skip("Qdrant not available for load testing")
        
        # Perform concurrent health checks
        start_time = time.time()
        tasks = []
        
        for i in range(10):  # 10 concurrent requests
            task = checker.readiness_probe()
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Verify all requests completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 10, f"Expected 10 successful results, got {len(successful_results)}"
        
        # Verify reasonable performance (all 10 requests in < 30 seconds)
        total_time = end_time - start_time
        assert total_time < 30.0, f"Load test took too long: {total_time}s"
        
        # Average response time should be reasonable
        avg_time = total_time / 10
        print(f"Average response time under load: {avg_time:.2f}s")
        assert avg_time < 10.0, f"Average response time too high: {avg_time}s"


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_health_integration_real.py -v -m integration
    print("Real Integration Tests for Health Infrastructure")
    print("=" * 60)
    print("Requirements:")
    print("1. Docker Compose services running: docker-compose up -d neo4j qdrant")  
    print("2. Services accessible at default ports (Neo4j: 7687, Qdrant: 6333)")
    print("3. Environment variables set if non-default (NEO4J_URI, QDRANT_HOST, etc.)")
    print("\nRun tests with: pytest tests/integration/test_health_integration_real.py -v -m integration")