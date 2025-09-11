"""
Pipeline Monitoring and Observability
L9 2025 Architecture - ADR-0024 Implementation

Provides comprehensive monitoring for the async preprocessing pipeline,
including metrics, health checks, and alerting.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ServiceHealth:
    """Health status for a single service"""
    name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_count: int = 0
    success_rate: float = 1.0
    details: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    timestamp: datetime
    queue_depths: Dict[str, int]
    throughput: Dict[str, float]  # items per minute
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rates: Dict[str, float]
    worker_utilization: Dict[str, float]


class PipelineMonitor:
    """
    Monitor async pipeline health and performance
    Provides Prometheus-compatible metrics
    """
    
    def __init__(self, container, pipeline, indexer):
        self.container = container
        self.pipeline = pipeline
        self.indexer = indexer
        
        # Health check configuration
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5  # seconds
        
        # Alert thresholds
        self.alert_thresholds = {
            'queue_depth_warning': 100,
            'queue_depth_critical': 500,
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.20,  # 20%
            'latency_p95_warning': 2000,  # 2 seconds
            'latency_p95_critical': 5000,  # 5 seconds
            'worker_utilization_warning': 0.80,  # 80%
            'worker_utilization_critical': 0.95,  # 95%
        }
        
        # Metrics storage (circular buffer for efficiency)
        self.metrics_buffer_size = 1000
        self.metrics_buffer: List[PipelineMetrics] = []
        self.service_health: Dict[str, ServiceHealth] = {}
        
        # Latency tracking
        self.latency_samples: Dict[str, List[float]] = {
            'tagging': [],
            'embedding': [],
            'storage': []
        }
        self.max_latency_samples = 1000
        
        # Alert state
        self.active_alerts: Dict[str, Dict] = {}
        
        # Monitoring task
        self.monitoring_task = None
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Pipeline monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Pipeline monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                self._store_metrics(metrics)
                
                # Check service health
                await self._check_service_health()
                
                # Evaluate alerts
                self._evaluate_alerts(metrics)
                
                # Log summary
                self._log_summary(metrics)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_metrics(self) -> PipelineMetrics:
        """Collect current pipeline metrics"""
        # Get queue depths from pipeline
        queue_status = await self.pipeline.get_queue_status()
        queue_depths = queue_status['queue_depths']
        
        # Calculate throughput (items per minute)
        throughput = {
            'tagging': self._calculate_throughput('files_tagged'),
            'embedding': self._calculate_throughput('files_embedded'),
            'storage': self._calculate_throughput('files_stored')
        }
        
        # Calculate latency percentiles
        latencies = self._calculate_latency_percentiles()
        
        # Calculate error rates
        error_rates = self._calculate_error_rates()
        
        # Calculate worker utilization
        worker_utilization = await self._calculate_worker_utilization()
        
        return PipelineMetrics(
            timestamp=datetime.now(),
            queue_depths=queue_depths,
            throughput=throughput,
            latency_p50_ms=latencies['p50'],
            latency_p95_ms=latencies['p95'],
            latency_p99_ms=latencies['p99'],
            error_rates=error_rates,
            worker_utilization=worker_utilization
        )
    
    def _calculate_throughput(self, metric_name: str) -> float:
        """Calculate throughput for a specific metric"""
        current_value = self.pipeline.metrics.get(metric_name, 0)
        
        # Find value from 1 minute ago
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        old_value = current_value  # Default to current if no history
        
        for metric in reversed(self.metrics_buffer):
            if metric.timestamp <= one_minute_ago:
                # Extract old value from metrics
                # This is simplified - in production you'd store raw counters
                break
        
        # Calculate rate
        return max(0, current_value - old_value)
    
    def _calculate_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles across all stages"""
        all_latencies = []
        for stage_latencies in self.latency_samples.values():
            all_latencies.extend(stage_latencies)
        
        if not all_latencies:
            return {'p50': 0, 'p95': 0, 'p99': 0}
        
        all_latencies.sort()
        n = len(all_latencies)
        
        return {
            'p50': all_latencies[int(n * 0.50)] if n > 0 else 0,
            'p95': all_latencies[int(n * 0.95)] if n > 0 else 0,
            'p99': all_latencies[int(n * 0.99)] if n > 0 else 0
        }
    
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates for each stage"""
        metrics = self.pipeline.metrics
        
        error_rates = {}
        
        # Tagging error rate
        total_tagged = metrics.get('files_tagged', 0) + metrics.get('tagging_errors', 0)
        if total_tagged > 0:
            error_rates['tagging'] = metrics.get('tagging_errors', 0) / total_tagged
        else:
            error_rates['tagging'] = 0
        
        # Embedding error rate
        total_embedded = metrics.get('files_embedded', 0) + metrics.get('embedding_errors', 0)
        if total_embedded > 0:
            error_rates['embedding'] = metrics.get('embedding_errors', 0) / total_embedded
        else:
            error_rates['embedding'] = 0
        
        # Storage error rate
        total_stored = metrics.get('files_stored', 0) + metrics.get('storage_errors', 0)
        if total_stored > 0:
            error_rates['storage'] = metrics.get('storage_errors', 0) / total_stored
        else:
            error_rates['storage'] = 0
        
        return error_rates
    
    async def _calculate_worker_utilization(self) -> Dict[str, float]:
        """Calculate worker utilization"""
        # This is simplified - in production you'd track actual worker activity
        queue_status = await self.pipeline.get_queue_status()
        
        utilization = {}
        
        # Estimate based on queue depths
        if queue_status['queue_depths']['raw'] > 10:
            utilization['taggers'] = min(1.0, queue_status['queue_depths']['raw'] / 50)
        else:
            utilization['taggers'] = 0.2
        
        if queue_status['queue_depths']['tagged'] > 10:
            utilization['embedders'] = min(1.0, queue_status['queue_depths']['tagged'] / 50)
        else:
            utilization['embedders'] = 0.2
        
        return utilization
    
    async def _check_service_health(self):
        """Check health of all services"""
        services = {
            'neo4j': self._check_neo4j_health,
            'qdrant': self._check_qdrant_health,
            'redis': self._check_redis_health,
            'gemma': self._check_gemma_health,
            'nomic': self._check_nomic_health
        }
        
        for service_name, check_func in services.items():
            start_time = time.time()
            try:
                is_healthy = await asyncio.wait_for(
                    check_func(),
                    timeout=self.health_check_timeout
                )
                
                response_time = (time.time() - start_time) * 1000
                
                # Update or create health record
                if service_name not in self.service_health:
                    self.service_health[service_name] = ServiceHealth(
                        name=service_name,
                        status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                        last_check=datetime.now(),
                        response_time_ms=response_time
                    )
                else:
                    health = self.service_health[service_name]
                    health.status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                    health.last_check = datetime.now()
                    health.response_time_ms = response_time
                    
                    # Update success rate
                    if is_healthy:
                        health.success_rate = min(1.0, health.success_rate * 0.9 + 0.1)
                    else:
                        health.success_rate = max(0.0, health.success_rate * 0.9)
                        health.error_count += 1
                
            except asyncio.TimeoutError:
                logger.warning(f"Health check timeout for {service_name}")
                if service_name in self.service_health:
                    self.service_health[service_name].status = HealthStatus.UNHEALTHY
                    self.service_health[service_name].error_count += 1
            except Exception as e:
                logger.error(f"Health check error for {service_name}: {e}")
                if service_name in self.service_health:
                    self.service_health[service_name].status = HealthStatus.CRITICAL
                    self.service_health[service_name].error_count += 1
    
    async def _check_neo4j_health(self) -> bool:
        """Check Neo4j health"""
        try:
            async with self.container.neo4j_service.driver.session() as session:
                result = await session.run("RETURN 1 as health")
                await result.single()
                return True
        except Exception:
            return False
    
    async def _check_qdrant_health(self) -> bool:
        """Check Qdrant health"""
        try:
            collections = await self.container.qdrant_service.get_collections()
            return collections is not None
        except Exception:
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis health"""
        try:
            if self.pipeline.redis_queue:
                await self.pipeline.redis_queue.ping()
                return True
            return False
        except Exception:
            return False
    
    async def _check_gemma_health(self) -> bool:
        """Check Gemma LLM health"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:48001/api/tags",
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _check_nomic_health(self) -> bool:
        """Check Nomic embedding service health"""
        try:
            # Use the container's nomic service health check
            return await self.container.nomic_service.health_check()
        except Exception:
            return False
    
    def _evaluate_alerts(self, metrics: PipelineMetrics):
        """Evaluate alert conditions"""
        alerts = []
        
        # Queue depth alerts
        for queue_name, depth in metrics.queue_depths.items():
            if depth > self.alert_thresholds['queue_depth_critical']:
                alerts.append({
                    'severity': 'critical',
                    'type': 'queue_depth',
                    'queue': queue_name,
                    'value': depth,
                    'threshold': self.alert_thresholds['queue_depth_critical']
                })
            elif depth > self.alert_thresholds['queue_depth_warning']:
                alerts.append({
                    'severity': 'warning',
                    'type': 'queue_depth',
                    'queue': queue_name,
                    'value': depth,
                    'threshold': self.alert_thresholds['queue_depth_warning']
                })
        
        # Error rate alerts
        for stage, rate in metrics.error_rates.items():
            if rate > self.alert_thresholds['error_rate_critical']:
                alerts.append({
                    'severity': 'critical',
                    'type': 'error_rate',
                    'stage': stage,
                    'value': rate,
                    'threshold': self.alert_thresholds['error_rate_critical']
                })
            elif rate > self.alert_thresholds['error_rate_warning']:
                alerts.append({
                    'severity': 'warning',
                    'type': 'error_rate',
                    'stage': stage,
                    'value': rate,
                    'threshold': self.alert_thresholds['error_rate_warning']
                })
        
        # Latency alerts
        if metrics.latency_p95_ms > self.alert_thresholds['latency_p95_critical']:
            alerts.append({
                'severity': 'critical',
                'type': 'latency',
                'percentile': 'p95',
                'value': metrics.latency_p95_ms,
                'threshold': self.alert_thresholds['latency_p95_critical']
            })
        elif metrics.latency_p95_ms > self.alert_thresholds['latency_p95_warning']:
            alerts.append({
                'severity': 'warning',
                'type': 'latency',
                'percentile': 'p95',
                'value': metrics.latency_p95_ms,
                'threshold': self.alert_thresholds['latency_p95_warning']
            })
        
        # Process alerts
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert.get('queue', alert.get('stage', 'global'))}"
            
            if alert_key not in self.active_alerts:
                # New alert
                self.active_alerts[alert_key] = {
                    **alert,
                    'started': datetime.now(),
                    'count': 1
                }
                self._log_alert(alert, is_new=True)
            else:
                # Existing alert
                self.active_alerts[alert_key]['count'] += 1
        
        # Clear resolved alerts
        resolved = []
        for alert_key in list(self.active_alerts.keys()):
            if not any(
                f"{a['type']}_{a.get('queue', a.get('stage', 'global'))}" == alert_key
                for a in alerts
            ):
                resolved.append(alert_key)
        
        for alert_key in resolved:
            logger.info(f"Alert resolved: {alert_key}")
            del self.active_alerts[alert_key]
    
    def _log_alert(self, alert: Dict, is_new: bool = False):
        """Log an alert"""
        prefix = "NEW ALERT" if is_new else "ONGOING ALERT"
        severity = alert['severity'].upper()
        
        message = (
            f"{prefix} [{severity}] - "
            f"Type: {alert['type']}, "
            f"Value: {alert['value']:.2f}, "
            f"Threshold: {alert['threshold']:.2f}"
        )
        
        if alert['severity'] == 'critical':
            logger.critical(message)
        elif alert['severity'] == 'warning':
            logger.warning(message)
        else:
            logger.info(message)
    
    def _log_summary(self, metrics: PipelineMetrics):
        """Log a summary of current metrics"""
        logger.info(
            f"Pipeline Status - "
            f"Queues[raw:{metrics.queue_depths['raw']}, "
            f"tagged:{metrics.queue_depths['tagged']}, "
            f"embed:{metrics.queue_depths['embed']}] "
            f"Throughput[tag:{metrics.throughput['tagging']:.1f}/min, "
            f"embed:{metrics.throughput['embedding']:.1f}/min] "
            f"Latency[p95:{metrics.latency_p95_ms:.0f}ms] "
            f"Errors[tag:{metrics.error_rates['tagging']:.1%}, "
            f"embed:{metrics.error_rates['embedding']:.1%}]"
        )
    
    def _store_metrics(self, metrics: PipelineMetrics):
        """Store metrics in circular buffer"""
        self.metrics_buffer.append(metrics)
        if len(self.metrics_buffer) > self.metrics_buffer_size:
            self.metrics_buffer.pop(0)
    
    def record_latency(self, stage: str, latency_ms: float):
        """Record a latency sample"""
        if stage in self.latency_samples:
            self.latency_samples[stage].append(latency_ms)
            if len(self.latency_samples[stage]) > self.max_latency_samples:
                self.latency_samples[stage].pop(0)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        current_metrics = self.metrics_buffer[-1] if self.metrics_buffer else None
        
        return {
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'service_health': {
                name: asdict(health) 
                for name, health in self.service_health.items()
            },
            'active_alerts': self.active_alerts,
            'historical_metrics': [
                asdict(m) for m in self.metrics_buffer[-100:]  # Last 100 samples
            ]
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        lines = []
        
        # Add metric definitions
        lines.append("# HELP pipeline_queue_depth Current depth of pipeline queues")
        lines.append("# TYPE pipeline_queue_depth gauge")
        
        if self.metrics_buffer:
            latest = self.metrics_buffer[-1]
            
            # Queue depths
            for queue, depth in latest.queue_depths.items():
                lines.append(f'pipeline_queue_depth{{queue="{queue}"}} {depth}')
            
            # Throughput
            lines.append("# HELP pipeline_throughput Items processed per minute")
            lines.append("# TYPE pipeline_throughput gauge")
            for stage, rate in latest.throughput.items():
                lines.append(f'pipeline_throughput{{stage="{stage}"}} {rate:.2f}')
            
            # Latencies
            lines.append("# HELP pipeline_latency_ms Processing latency in milliseconds")
            lines.append("# TYPE pipeline_latency_ms summary")
            lines.append(f'pipeline_latency_ms{{quantile="0.5"}} {latest.latency_p50_ms:.2f}')
            lines.append(f'pipeline_latency_ms{{quantile="0.95"}} {latest.latency_p95_ms:.2f}')
            lines.append(f'pipeline_latency_ms{{quantile="0.99"}} {latest.latency_p99_ms:.2f}')
            
            # Error rates
            lines.append("# HELP pipeline_error_rate Error rate by stage")
            lines.append("# TYPE pipeline_error_rate gauge")
            for stage, rate in latest.error_rates.items():
                lines.append(f'pipeline_error_rate{{stage="{stage}"}} {rate:.4f}')
        
        # Service health
        lines.append("# HELP service_health Service health status (1=healthy, 0=unhealthy)")
        lines.append("# TYPE service_health gauge")
        for name, health in self.service_health.items():
            value = 1 if health.status == HealthStatus.HEALTHY else 0
            lines.append(f'service_health{{service="{name}"}} {value}')
        
        return "\n".join(lines)