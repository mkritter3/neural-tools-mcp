"""
L9 2025 Health Check and Monitoring Service for MCP Server
Production-grade health monitoring with Prometheus metrics
"""

import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import redis.asyncio as redis
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ServiceHealth:
    name: str
    status: HealthStatus
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class PrometheusMetrics:
    """Prometheus-compatible metrics collection"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.metrics = {}
        self.last_export = time.time()
        
    async def increment_counter(self, name: str, labels: Dict[str, str] = None, value: float = 1):
        """Increment a counter metric"""
        key = self._build_metric_key(name, labels)
        self.metrics[key] = self.metrics.get(key, 0) + value
        
        # Store in Redis for persistence
        if self.redis_client:
            try:
                await self.redis_client.incrbyfloat(f"metric:counter:{key}", value)
                await self.redis_client.expire(f"metric:counter:{key}", 86400)  # 24h TTL
            except Exception as e:
                logger.error(f"Failed to store counter metric: {e}")
    
    async def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric"""
        key = self._build_metric_key(name, labels)
        self.metrics[key] = value
        
        if self.redis_client:
            try:
                await self.redis_client.setex(f"metric:gauge:{key}", 3600, str(value))
            except Exception as e:
                logger.error(f"Failed to store gauge metric: {e}")
    
    async def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value"""
        key = self._build_metric_key(name, labels)
        
        # Simple histogram implementation - store values for percentile calculation
        hist_key = f"metric:histogram:{key}"
        
        if self.redis_client:
            try:
                # Add value to sorted set for percentile calculation
                timestamp = time.time()
                await self.redis_client.zadd(hist_key, {str(value): timestamp})
                
                # Keep only last hour of data
                cutoff = timestamp - 3600
                await self.redis_client.zremrangebyscore(hist_key, 0, cutoff)
                
            except Exception as e:
                logger.error(f"Failed to store histogram metric: {e}")
    
    def _build_metric_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Build metric key with labels"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    async def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        lines.append("# L9 2025 Neural MCP Server Metrics")
        lines.append(f"# Generated at {datetime.utcnow().isoformat()}")
        lines.append("")
        
        # Group metrics by type
        counters = {}
        gauges = {}
        
        for key, value in self.metrics.items():
            if "counter" in key:
                counters[key] = value
            else:
                gauges[key] = value
        
        # Export counters
        if counters:
            lines.append("# TYPE neural_mcp_requests_total counter")
            for key, value in counters.items():
                lines.append(f"neural_mcp_requests_total{{{key}}} {value}")
        
        # Export gauges
        if gauges:
            lines.append("# TYPE neural_mcp_connections_active gauge")
            for key, value in gauges.items():
                lines.append(f"neural_mcp_connections_active{{{key}}} {value}")
        
        return "\n".join(lines)

class HealthMonitor:
    """Comprehensive health monitoring for MCP server"""
    
    def __init__(self, service_container):
        self.container = service_container
        self.redis_client: Optional[redis.Redis] = None
        self.metrics = PrometheusMetrics()
        self.health_checks = {}
        self._monitoring = False
        self._monitor_task = None
        
        # Health check configuration
        self.check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))  # seconds
        self.timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '5'))  # seconds
        
    async def initialize(self):
        """Initialize health monitor"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=46379,
                password='cache-secret-key',
                decode_responses=True,
                db=3  # Use db 3 for health data
            )
            await self.redis_client.ping()
            
            # Initialize metrics with Redis
            self.metrics = PrometheusMetrics(self.redis_client)
            
            logger.info("✅ HealthMonitor initialized with Redis metrics storage")
        except Exception as e:
            logger.warning(f"Redis not available for health monitoring, using in-memory: {e}")
            self.redis_client = None
    
    async def start_monitoring(self):
        """Start periodic health monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"🔍 Health monitoring started (interval: {self.check_interval}s)")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                await self._run_health_checks()
                await self._update_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _run_health_checks(self):
        """Run all configured health checks"""
        checks = [
            self._check_redis_cache(),
            self._check_redis_queue(),
            self._check_neo4j(),
            self._check_qdrant(),
            self._check_nomic_service(),
            self._check_connection_pools(),
            self._check_system_resources()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Health check {i} failed: {result}")
    
    async def _check_redis_cache(self) -> ServiceHealth:
        """Check Redis cache health"""
        start_time = time.time()
        
        try:
            cache_client = await self.container.get_redis_cache_client()
            await cache_client.ping()
            
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                name="redis_cache",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            health = ServiceHealth(
                name="redis_cache",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        self.health_checks["redis_cache"] = health
        await self.metrics.record_histogram("health_check_duration_ms", health.response_time_ms, {"service": "redis_cache"})
        await self.metrics.set_gauge("service_health", 1 if health.status == HealthStatus.HEALTHY else 0, {"service": "redis_cache"})
        
        return health
    
    async def _check_redis_queue(self) -> ServiceHealth:
        """Check Redis queue health"""
        start_time = time.time()
        
        try:
            queue_client = await self.container.get_redis_queue_client()
            await queue_client.ping()
            
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                name="redis_queue",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            health = ServiceHealth(
                name="redis_queue",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        self.health_checks["redis_queue"] = health
        await self.metrics.record_histogram("health_check_duration_ms", health.response_time_ms, {"service": "redis_queue"})
        await self.metrics.set_gauge("service_health", 1 if health.status == HealthStatus.HEALTHY else 0, {"service": "redis_queue"})
        
        return health
    
    async def _check_neo4j(self) -> ServiceHealth:
        """Check Neo4j health"""
        start_time = time.time()
        
        try:
            if not self.container.neo4j:
                raise Exception("Neo4j service not initialized")
            
            # Simple health check query
            client = self.container.get_neo4j_client()
            with client.session() as session:
                result = session.run("RETURN 1 as health_check")
                health_value = result.single()["health_check"]
                
            if health_value != 1:
                raise Exception("Neo4j health check failed")
            
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                name="neo4j",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            health = ServiceHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        self.health_checks["neo4j"] = health
        await self.metrics.record_histogram("health_check_duration_ms", health.response_time_ms, {"service": "neo4j"})
        await self.metrics.set_gauge("service_health", 1 if health.status == HealthStatus.HEALTHY else 0, {"service": "neo4j"})
        
        return health
    
    async def _check_qdrant(self) -> ServiceHealth:
        """Check Qdrant health"""
        start_time = time.time()
        
        try:
            if not self.container.qdrant:
                raise Exception("Qdrant service not initialized")
            
            # Use existing Qdrant client
            from qdrant_client import QdrantClient
            QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
            QDRANT_PORT = int(os.getenv("QDRANT_PORT", "46333"))
            
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
            collections = client.get_collections()
            
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                name="qdrant",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                details={"collections_count": len(collections.collections)}
            )
            
        except Exception as e:
            health = ServiceHealth(
                name="qdrant",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        self.health_checks["qdrant"] = health
        await self.metrics.record_histogram("health_check_duration_ms", health.response_time_ms, {"service": "qdrant"})
        await self.metrics.set_gauge("service_health", 1 if health.status == HealthStatus.HEALTHY else 0, {"service": "qdrant"})
        
        return health
    
    async def _check_nomic_service(self) -> ServiceHealth:
        """Check Nomic embedding service health"""
        start_time = time.time()
        
        try:
            if not self.container.nomic:
                raise Exception("Nomic service not initialized")
            
            # Simple ping to Nomic service
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:48000/health", timeout=3) as response:
                    if response.status != 200:
                        raise Exception(f"Nomic service returned status {response.status}")
            
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                name="nomic",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.utcnow()
            )
            
        except Exception as e:
            health = ServiceHealth(
                name="nomic",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        self.health_checks["nomic"] = health
        await self.metrics.record_histogram("health_check_duration_ms", health.response_time_ms, {"service": "nomic"})
        await self.metrics.set_gauge("service_health", 1 if health.status == HealthStatus.HEALTHY else 0, {"service": "nomic"})
        
        return health
    
    async def _check_connection_pools(self) -> ServiceHealth:
        """Check connection pool health"""
        start_time = time.time()
        
        try:
            if not hasattr(self.container, 'connection_pools'):
                raise Exception("Connection pools not initialized")
            
            pool_stats = {}
            total_utilization = 0
            pool_count = 0
            
            for service, pool_data in self.container.connection_pools.items():
                utilization = (pool_data['active'] / pool_data['max_size']) * 100
                pool_stats[service] = {
                    'active': pool_data['active'],
                    'max_size': pool_data['max_size'],
                    'utilization': utilization
                }
                total_utilization += utilization
                pool_count += 1
            
            avg_utilization = total_utilization / pool_count if pool_count > 0 else 0
            
            # Determine health based on average utilization
            if avg_utilization > 90:
                status = HealthStatus.UNHEALTHY
            elif avg_utilization > 70:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                name="connection_pools",
                status=status,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                details={
                    'average_utilization': round(avg_utilization, 2),
                    'pools': pool_stats
                }
            )
            
        except Exception as e:
            health = ServiceHealth(
                name="connection_pools",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        self.health_checks["connection_pools"] = health
        await self.metrics.set_gauge("connection_pool_utilization", health.details.get('average_utilization', 0) if health.details else 0)
        
        return health
    
    async def _check_system_resources(self) -> ServiceHealth:
        """Check system resource health"""
        start_time = time.time()
        
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine health based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.UNHEALTHY
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            response_time = (time.time() - start_time) * 1000
            
            health = ServiceHealth(
                name="system_resources",
                status=status,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent
                }
            )
            
            # Update system metrics
            await self.metrics.set_gauge("system_cpu_percent", cpu_percent)
            await self.metrics.set_gauge("system_memory_percent", memory.percent)
            await self.metrics.set_gauge("system_disk_percent", disk.percent)
            
        except ImportError:
            health = ServiceHealth(
                name="system_resources",
                status=HealthStatus.DEGRADED,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow(),
                error_message="psutil not available for system monitoring"
            )
        except Exception as e:
            health = ServiceHealth(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
        
        self.health_checks["system_resources"] = health
        return health
    
    async def _update_metrics(self):
        """Update general metrics"""
        try:
            # Connection pool metrics
            if hasattr(self.container, 'connection_pools'):
                for service, pool_data in self.container.connection_pools.items():
                    await self.metrics.set_gauge(
                        "connection_pool_active", 
                        pool_data['active'], 
                        {"service": service}
                    )
                    await self.metrics.set_gauge(
                        "connection_pool_max", 
                        pool_data['max_size'], 
                        {"service": service}
                    )
            
            # Session metrics
            if hasattr(self.container, 'session_manager') and self.container.session_manager:
                session_stats = self.container.session_manager.get_session_stats()
                await self.metrics.set_gauge("active_sessions", session_stats['active_sessions'])
                await self.metrics.set_gauge("total_queries", session_stats['total_queries'])
                
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.health_checks:
            await self._run_health_checks()
        
        healthy_count = sum(1 for check in self.health_checks.values() 
                          if check.status == HealthStatus.HEALTHY)
        total_count = len(self.health_checks)
        
        # Determine overall status
        if healthy_count == total_count:
            overall_status = HealthStatus.HEALTHY
        elif healthy_count >= total_count * 0.7:  # 70% healthy
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "healthy": healthy_count,
                "total": total_count,
                "health_percentage": round((healthy_count / total_count) * 100, 1)
            },
            "services": {
                name: {
                    "status": check.status.value,
                    "response_time_ms": check.response_time_ms,
                    "last_check": check.last_check.isoformat(),
                    "error": check.error_message,
                    "details": check.details
                }
                for name, check in self.health_checks.items()
            }
        }
    
    async def get_metrics(self) -> str:
        """Get Prometheus-formatted metrics"""
        return await self.metrics.export_prometheus_format()