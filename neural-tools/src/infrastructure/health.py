#!/usr/bin/env python3
"""
Health Checks & Readiness Probes Infrastructure (Phase 2.5)

Provides production-grade health monitoring:
- Kubernetes liveness probes
- Kubernetes readiness probes
- Dependency health checks (Qdrant, Neo4j, Redis, API keys)
- Prometheus metrics endpoints
- Component status monitoring
- Circuit breaker integration

Key components:
- HealthChecker for centralized health monitoring
- Individual dependency checkers with timeouts
- Metrics collection for observability
- FastAPI endpoint integration
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

try:
    from fastapi import FastAPI, status, Response
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    status = None
    Response = None
    JSONResponse = None
    FASTAPI_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = None
    Histogram = None
    Gauge = None
    generate_latest = lambda: "# Prometheus client not available\n"
    CONTENT_TYPE_LATEST = "text/plain"
    PROMETHEUS_AVAILABLE = False

from infrastructure.structured_logging import get_logger
from infrastructure.telemetry import get_telemetry

logger = get_logger("health")
telemetry = get_telemetry()


class HealthStatus(Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy" 
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


@dataclass
class HealthConfig:
    """Configuration for health monitoring"""
    
    # Timeouts for dependency checks
    qdrant_timeout: float = 2.0
    neo4j_timeout: float = 2.0
    redis_timeout: float = 1.0
    api_key_check_timeout: float = 0.5
    
    # Health check intervals
    dependency_check_interval: int = 30  # seconds
    metrics_collection_interval: int = 10  # seconds
    
    # Circuit breaker integration
    enable_circuit_breaker_checks: bool = True
    
    # Readiness criteria
    required_dependencies: List[str] = None  # If None, uses default
    
    def __post_init__(self):
        if self.required_dependencies is None:
            self.required_dependencies = ["qdrant", "neo4j"]


class DependencyHealthChecker:
    """Base class for dependency health checkers"""
    
    def __init__(self, name: str, timeout: float = 2.0):
        self.name = name
        self.timeout = timeout
        self.last_check_time: Optional[datetime] = None
        self.last_result: Optional[HealthCheckResult] = None
    
    async def check_health(self) -> HealthCheckResult:
        """Check the health of this dependency"""
        start_time = time.perf_counter()
        
        try:
            is_healthy = await asyncio.wait_for(
                self._perform_health_check(),
                timeout=self.timeout
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
            message = "OK" if is_healthy else "Health check failed"
            
            result = HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                metadata={"response_time_ms": latency_ms}
            )
            
        except asyncio.TimeoutError:
            latency_ms = self.timeout * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Timeout after {self.timeout}s",
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                metadata={"response_time_ms": latency_ms, "error_type": "timeout"}
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                metadata={"response_time_ms": latency_ms, "error_type": "exception", "error": str(e)}
            )
        
        self.last_check_time = datetime.utcnow()
        self.last_result = result
        
        return result
    
    async def _perform_health_check(self) -> bool:
        """Override this method to implement specific health check logic"""
        raise NotImplementedError


class QdrantHealthChecker(DependencyHealthChecker):
    """Health checker for Qdrant vector database"""
    
    def __init__(self, qdrant_client=None, timeout: float = 2.0):
        super().__init__("qdrant", timeout)
        self.client = qdrant_client
    
    async def _perform_health_check(self) -> bool:
        """Check Qdrant health by listing collections"""
        if not self.client:
            return False
        
        try:
            # Try to get collections - lightweight operation (sync call)
            collections = self.client.get_collections()
            return collections is not None
        except Exception:
            return False


class Neo4jHealthChecker(DependencyHealthChecker):
    """Health checker for Neo4j graph database"""
    
    def __init__(self, neo4j_client=None, timeout: float = 2.0):
        super().__init__("neo4j", timeout)
        self.client = neo4j_client
    
    async def _perform_health_check(self) -> bool:
        """Check Neo4j health by running simple query"""
        if not self.client:
            return False
        
        try:
            # Simple query to check connectivity using sync driver
            with self.client.session() as session:
                result = session.run("RETURN 1 as health")
                record = result.single()
                return record and record["health"] == 1
        except Exception:
            return False


class RedisHealthChecker(DependencyHealthChecker):
    """Health checker for Redis cache"""
    
    def __init__(self, redis_client=None, timeout: float = 1.0):
        super().__init__("redis", timeout)
        self.client = redis_client
    
    async def _perform_health_check(self) -> bool:
        """Check Redis health with ping"""
        if not self.client:
            logger.warning("Redis health check: no client provided")
            return False
        
        try:
            # Redis ping
            logger.debug(f"Redis health check: pinging client {type(self.client)}")
            result = await self.client.ping()
            logger.debug(f"Redis health check: ping result = {result}")
            return result == True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False


class APIKeyHealthChecker(DependencyHealthChecker):
    """Health checker for API key availability"""
    
    def __init__(self, api_key_env_var: str = "ANTHROPIC_API_KEY", timeout: float = 0.5):
        super().__init__(f"api_key_{api_key_env_var.lower()}", timeout)
        self.api_key_env_var = api_key_env_var
    
    async def _perform_health_check(self) -> bool:
        """Check if API key is configured"""
        api_key = os.getenv(self.api_key_env_var)
        return bool(api_key and len(api_key.strip()) > 0)


class HealthChecker:
    """Centralized health monitoring system"""
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.dependency_checkers: Dict[str, DependencyHealthChecker] = {}
        self.last_overall_check: Optional[datetime] = None
        self.cached_readiness_result: Optional[Dict[str, Any]] = None
        
        # Prometheus metrics (if available and not in test mode)
        test_mode = os.getenv("PYTEST_CURRENT_TEST") is not None
        if PROMETHEUS_AVAILABLE and not test_mode:
            try:
                self.health_check_counter = Counter(
                    'health_checks_total',
                    'Total health checks performed',
                    ['dependency', 'status']
                )
                self.health_check_latency = Histogram(
                    'health_check_duration_seconds',
                    'Health check latency',
                    ['dependency']
                )
                self.dependency_status_gauge = Gauge(
                    'dependency_healthy',
                    'Dependency health status (1=healthy, 0=unhealthy)',
                    ['dependency']
                )
                self.readiness_gauge = Gauge(
                    'service_ready',
                    'Overall service readiness (1=ready, 0=not ready)'
                )
            except ValueError as e:
                # Handle duplicate registration gracefully
                logger.warning("Prometheus metrics already registered", error=str(e))
                self.health_check_counter = None
                self.health_check_latency = None
                self.dependency_status_gauge = None
                self.readiness_gauge = None
        else:
            self.health_check_counter = None
            self.health_check_latency = None
            self.dependency_status_gauge = None
            self.readiness_gauge = None
        
        logger.info("Health checker initialized", 
                   required_dependencies=self.config.required_dependencies,
                   prometheus_available=PROMETHEUS_AVAILABLE)
    
    def add_dependency_checker(self, checker: DependencyHealthChecker):
        """Add a dependency health checker"""
        self.dependency_checkers[checker.name] = checker
        logger.info("Added dependency checker", dependency=checker.name, timeout=checker.timeout)
    
    def add_qdrant_checker(self, qdrant_client):
        """Add Qdrant health checker"""
        checker = QdrantHealthChecker(qdrant_client, self.config.qdrant_timeout)
        self.add_dependency_checker(checker)
    
    def add_neo4j_checker(self, neo4j_client):
        """Add Neo4j health checker"""
        checker = Neo4jHealthChecker(neo4j_client, self.config.neo4j_timeout)
        self.add_dependency_checker(checker)
    
    def add_redis_checker(self, redis_client):
        """Add Redis health checker"""
        checker = RedisHealthChecker(redis_client, self.config.redis_timeout)
        self.add_dependency_checker(checker)
    
    def add_api_key_checker(self, env_var: str = "ANTHROPIC_API_KEY"):
        """Add API key health checker"""
        checker = APIKeyHealthChecker(env_var, self.config.api_key_check_timeout)
        self.add_dependency_checker(checker)
    
    async def check_dependency(self, dependency_name: str) -> Optional[HealthCheckResult]:
        """Check health of specific dependency"""
        if dependency_name not in self.dependency_checkers:
            logger.warning("Unknown dependency requested", dependency=dependency_name)
            return None
        
        checker = self.dependency_checkers[dependency_name]
        
        with telemetry.trace_operation(f"health_check_{dependency_name}", {
            "dependency": dependency_name,
            "timeout": checker.timeout
        }) as span:
            
            result = await checker.check_health()
            
            # Record metrics
            if PROMETHEUS_AVAILABLE and self.health_check_counter:
                self.health_check_counter.labels(
                    dependency=dependency_name,
                    status=result.status.value
                ).inc()
                
                if self.health_check_latency:
                    self.health_check_latency.labels(
                        dependency=dependency_name
                    ).observe(result.latency_ms / 1000)
                
                if self.dependency_status_gauge:
                    self.dependency_status_gauge.labels(
                        dependency=dependency_name
                    ).set(1 if result.status == HealthStatus.HEALTHY else 0)
            
            # Update telemetry
            if span:
                span.set_attribute("health_status", result.status.value)
                span.set_attribute("latency_ms", result.latency_ms)
            
            # Log result
            if result.status == HealthStatus.HEALTHY:
                logger.debug("Health check passed", 
                           dependency=dependency_name, 
                           latency_ms=result.latency_ms)
            else:
                logger.warning("Health check failed",
                             dependency=dependency_name,
                             status=result.status.value,
                             message=result.message,
                             latency_ms=result.latency_ms)
            
            return result
    
    async def check_all_dependencies(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered dependencies"""
        results = {}
        
        # Run all health checks concurrently
        tasks = {
            name: self.check_dependency(name)
            for name in self.dependency_checkers.keys()
        }
        
        if tasks:
            completed_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for name, result in zip(tasks.keys(), completed_results):
                if isinstance(result, Exception):
                    results[name] = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check exception: {result}",
                        latency_ms=0,
                        timestamp=datetime.utcnow()
                    )
                elif result is not None:
                    results[name] = result
        
        self.last_overall_check = datetime.utcnow()
        return results
    
    async def liveness_probe(self) -> Dict[str, Any]:
        """
        Kubernetes liveness probe - basic service health
        
        Returns:
            Simple status indicating the service is alive
        """
        return {
            "status": "alive",
            "timestamp": datetime.utcnow().isoformat(),
            "version": os.getenv("SERVICE_VERSION", "unknown")
        }
    
    async def readiness_probe(self) -> Dict[str, Any]:
        """
        Kubernetes readiness probe - service readiness to handle requests
        
        Returns:
            Detailed readiness status including dependency checks
        """
        start_time = time.perf_counter()
        
        # Check all dependencies
        dependency_results = await self.check_all_dependencies()
        
        # Determine overall readiness
        is_ready = True
        failed_required = []
        
        for dep_name in self.config.required_dependencies:
            if dep_name in dependency_results:
                result = dependency_results[dep_name]
                if result.status != HealthStatus.HEALTHY:
                    is_ready = False
                    failed_required.append(dep_name)
            else:
                # Required dependency not configured
                is_ready = False
                failed_required.append(f"{dep_name} (not configured)")
        
        # Calculate overall latency
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Build response
        checks_dict = {name: result.to_dict() for name, result in dependency_results.items()}
        
        readiness_response = {
            "ready": is_ready,
            "checks": checks_dict,
            "required_dependencies": self.config.required_dependencies,
            "failed_required": failed_required,
            "timestamp": datetime.utcnow().isoformat(),
            "version": os.getenv("SERVICE_VERSION", "unknown"),
            "latency_ms": round(total_latency_ms, 2)
        }
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE and self.readiness_gauge:
            self.readiness_gauge.set(1 if is_ready else 0)
        
        # Cache result briefly
        self.cached_readiness_result = readiness_response
        
        # Log readiness status
        if is_ready:
            logger.info("Readiness probe passed", latency_ms=total_latency_ms)
        else:
            logger.warning("Readiness probe failed",
                         failed_dependencies=failed_required,
                         latency_ms=total_latency_ms)
        
        return readiness_response
    
    async def startup_probe(self) -> Dict[str, Any]:
        """
        Kubernetes startup probe - service has started successfully
        
        Returns:
            Startup status with initialization checks
        """
        # Similar to readiness but with more lenient timeouts
        result = await self.readiness_probe()
        result["probe_type"] = "startup"
        return result
    
    def get_metrics(self) -> str:
        """
        Get Prometheus metrics
        
        Returns:
            Prometheus-formatted metrics string
        """
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        try:
            return generate_latest()
        except Exception as e:
            logger.error("Failed to generate metrics", error=str(e))
            return f"# Error generating metrics: {e}\n"
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health monitoring statistics"""
        return {
            "config": {
                "required_dependencies": self.config.required_dependencies,
                "prometheus_available": PROMETHEUS_AVAILABLE,
                "fastapi_available": FASTAPI_AVAILABLE
            },
            "dependencies": {
                name: {
                    "last_check": checker.last_check_time.isoformat() if checker.last_check_time else None,
                    "last_status": checker.last_result.status.value if checker.last_result else None,
                    "timeout": checker.timeout
                }
                for name, checker in self.dependency_checkers.items()
            },
            "last_overall_check": self.last_overall_check.isoformat() if self.last_overall_check else None
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker(config: Optional[HealthConfig] = None) -> HealthChecker:
    """Get or create global health checker"""
    global _health_checker
    
    if _health_checker is None:
        _health_checker = HealthChecker(config)
    
    return _health_checker


def setup_health_endpoints(app, container=None):
    """
    Setup health check endpoints on FastAPI app
    
    Args:
        app: FastAPI application instance
        container: Service container with dependencies
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping health endpoint setup")
        return
    
    health_checker = get_health_checker()
    
    # Configure dependency checkers from container
    if container:
        if hasattr(container, 'qdrant') and container.qdrant:
            health_checker.add_qdrant_checker(container.qdrant)
        
        if hasattr(container, 'neo4j') and container.neo4j:
            health_checker.add_neo4j_checker(container.neo4j)
        
        # Add Redis health checkers for both cache and queue instances
        if hasattr(container, 'get_redis_cache_client'):
            try:
                import asyncio
                cache_client = asyncio.create_task(container.get_redis_cache_client())
                health_checker.add_dependency_checker(
                    RedisHealthChecker(cache_client, health_checker.config.redis_timeout)
                )
                # Rename to distinguish from queue Redis
                health_checker.dependency_checkers['redis-cache'] = health_checker.dependency_checkers.pop('redis')
            except Exception as e:
                logger.warning("Failed to add Redis cache health checker", error=str(e))
        
        if hasattr(container, 'get_redis_queue_client'):
            try:
                import asyncio
                queue_client = asyncio.create_task(container.get_redis_queue_client())
                queue_checker = RedisHealthChecker(queue_client, health_checker.config.redis_timeout)
                queue_checker.name = 'redis-queue'  # Rename for clarity
                health_checker.add_dependency_checker(queue_checker)
            except Exception as e:
                logger.warning("Failed to add Redis queue health checker", error=str(e))
    
    # Always add API key checker
    health_checker.add_api_key_checker("ANTHROPIC_API_KEY")
    
    @app.get("/health/live", status_code=status.HTTP_200_OK)
    async def liveness_probe():
        """Kubernetes liveness probe"""
        return await health_checker.liveness_probe()
    
    @app.get("/health/ready")
    async def readiness_probe():
        """Kubernetes readiness probe"""
        result = await health_checker.readiness_probe()
        
        # Return 503 if not ready
        if not result.get("ready", False):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=result
            )
        
        return result
    
    @app.get("/health/startup")  
    async def startup_probe():
        """Kubernetes startup probe"""
        result = await health_checker.startup_probe()
        
        # Return 503 if not ready
        if not result.get("ready", False):
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=result
            )
        
        return result
    
    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint"""
        metrics_content = health_checker.get_metrics()
        return Response(content=metrics_content, media_type=CONTENT_TYPE_LATEST)
    
    @app.get("/health/stats")
    async def health_stats():
        """Health monitoring statistics"""
        return health_checker.get_health_stats()
    
    logger.info("Health check endpoints configured",
               endpoints=["/health/live", "/health/ready", "/health/startup", "/metrics", "/health/stats"])


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        print("🏥 Testing Health Check Infrastructure")
        
        # Setup health checker
        config = HealthConfig(
            required_dependencies=["qdrant", "neo4j"],
            qdrant_timeout=1.0,
            neo4j_timeout=1.0
        )
        health_checker = HealthChecker(config)
        print("✓ Health checker initialized")
        
        # Add mock checkers for testing
        class MockHealthyChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                await asyncio.sleep(0.01)  # Simulate work
                return True
        
        class MockUnhealthyChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                await asyncio.sleep(0.01)
                return False
        
        class MockTimeoutChecker(DependencyHealthChecker):
            async def _perform_health_check(self) -> bool:
                await asyncio.sleep(2.0)  # Will timeout
                return True
        
        # Add test checkers
        health_checker.add_dependency_checker(MockHealthyChecker("qdrant", 1.0))
        health_checker.add_dependency_checker(MockUnhealthyChecker("neo4j", 1.0))
        health_checker.add_dependency_checker(MockTimeoutChecker("redis", 0.5))
        health_checker.add_api_key_checker("ANTHROPIC_API_KEY")
        
        print("\n🔍 Testing individual health checks...")
        
        # Test individual checks
        for dep_name in ["qdrant", "neo4j", "redis", "api_key_anthropic_api_key"]:
            result = await health_checker.check_dependency(dep_name)
            if result:
                status_icon = "✅" if result.status == HealthStatus.HEALTHY else "❌"
                print(f"{status_icon} {dep_name}: {result.status.value} ({result.latency_ms:.1f}ms)")
        
        print("\n🧪 Testing liveness probe...")
        liveness = await health_checker.liveness_probe()
        print(f"✓ Liveness: {liveness['status']}")
        
        print("\n🧪 Testing readiness probe...")
        readiness = await health_checker.readiness_probe()
        ready_icon = "✅" if readiness['ready'] else "❌"
        print(f"{ready_icon} Readiness: {'ready' if readiness['ready'] else 'not ready'}")
        print(f"  Failed required: {readiness['failed_required']}")
        print(f"  Latency: {readiness['latency_ms']:.1f}ms")
        
        print("\n📊 Testing metrics...")
        metrics = health_checker.get_metrics()
        metric_lines = len([line for line in metrics.split('\n') if line and not line.startswith('#')])
        print(f"✓ Generated {metric_lines} metric lines")
        
        print("\n📈 Health stats:")
        stats = health_checker.get_health_stats()
        print(f"  Dependencies configured: {len(stats['dependencies'])}")
        print(f"  Required dependencies: {stats['config']['required_dependencies']}")
        print(f"  Prometheus available: {stats['config']['prometheus_available']}")
        
        print("\n🏥 Health check infrastructure tests complete!")
    
    # Run the test
    asyncio.run(main())