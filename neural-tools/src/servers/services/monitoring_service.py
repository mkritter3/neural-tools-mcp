#!/usr/bin/env python3
"""
ADR-0084 Phase 3: Centralized Monitoring Service
Aggregates health, metrics, and circuit breaker status from all services
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ServiceMetrics:
    """Metrics for a single service"""
    service_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    request_timestamps: List[float] = field(default_factory=list)
    error_timestamps: List[float] = field(default_factory=list)
    last_error: Optional[str] = None
    last_success: Optional[float] = None

    def record_request(self, success: bool, latency_ms: float, error: Optional[str] = None):
        """Record a request outcome"""
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        current_time = time.time()

        if success:
            self.successful_requests += 1
            self.last_success = current_time
        else:
            self.failed_requests += 1
            self.error_timestamps.append(current_time)
            if error:
                self.last_error = error

        self.request_timestamps.append(current_time)

        # Keep only last hour of timestamps
        cutoff = current_time - 3600
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff]
        self.error_timestamps = [t for t in self.error_timestamps if t > cutoff]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        success_rate = (
            self.successful_requests / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        avg_latency = (
            self.total_latency_ms / self.total_requests
            if self.total_requests > 0 else 0
        )

        # Calculate requests per minute for last 5 minutes
        current_time = time.time()
        recent_requests = [
            t for t in self.request_timestamps
            if t > current_time - 300  # Last 5 minutes
        ]
        rpm = len(recent_requests) / 5.0 if recent_requests else 0

        # Calculate error rate for last 5 minutes
        recent_errors = [
            t for t in self.error_timestamps
            if t > current_time - 300
        ]
        error_rate = len(recent_errors) / len(recent_requests) * 100 if recent_requests else 0

        return {
            "service": self.service_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{success_rate:.1f}%",
            "avg_latency_ms": f"{avg_latency:.1f}",
            "requests_per_minute": f"{rpm:.1f}",
            "error_rate_5min": f"{error_rate:.1f}%",
            "last_error": self.last_error,
            "last_success": (
                datetime.fromtimestamp(self.last_success).isoformat()
                if self.last_success else None
            )
        }


class MonitoringService:
    """
    Centralized monitoring for all neural services

    ADR-0084 Phase 3: Provides unified health checks, metrics aggregation,
    and alerting capabilities for the neural system
    """

    def __init__(self):
        self.metrics: Dict[str, ServiceMetrics] = {}
        self.services = {}  # Service references
        self.alerts: List[Dict[str, Any]] = []
        self.initialized = False

    def register_service(self, name: str, service: Any):
        """Register a service for monitoring"""
        self.services[name] = service
        if name not in self.metrics:
            self.metrics[name] = ServiceMetrics(service_name=name)
        logger.info(f"📊 Registered service '{name}' for monitoring")

    def record_request(
        self,
        service_name: str,
        success: bool,
        latency_ms: float,
        error: Optional[str] = None
    ):
        """Record a service request outcome"""
        if service_name not in self.metrics:
            self.metrics[service_name] = ServiceMetrics(service_name=service_name)

        self.metrics[service_name].record_request(success, latency_ms, error)

        # Check for alerts
        if not success:
            self._check_alerts(service_name, error)

    def _check_alerts(self, service_name: str, error: Optional[str]):
        """Check if we should generate alerts"""
        metrics = self.metrics[service_name]

        # Alert if error rate > 50% in last 5 minutes
        current_time = time.time()
        recent_requests = [
            t for t in metrics.request_timestamps
            if t > current_time - 300
        ]
        recent_errors = [
            t for t in metrics.error_timestamps
            if t > current_time - 300
        ]

        if len(recent_requests) >= 10:  # Need minimum requests
            error_rate = len(recent_errors) / len(recent_requests) * 100
            if error_rate > 50:
                self.alerts.append({
                    "timestamp": datetime.now().isoformat(),
                    "service": service_name,
                    "type": "HIGH_ERROR_RATE",
                    "message": f"Error rate {error_rate:.1f}% exceeds 50% threshold",
                    "error": error
                })
                logger.error(f"🚨 ALERT: {service_name} error rate {error_rate:.1f}% > 50%")

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "services": {},
            "circuit_breakers": {},
            "metrics": {},
            "alerts": self.alerts[-10:]  # Last 10 alerts
        }

        # Check each service
        for name, service in self.services.items():
            try:
                # Get health status
                if hasattr(service, 'health_check'):
                    service_health = await service.health_check()
                    health["services"][name] = service_health
                    if not service_health.get("healthy", False):
                        health["overall_healthy"] = False

                # Get circuit breaker status
                if hasattr(service, 'get_circuit_breaker_status'):
                    cb_status = service.get_circuit_breaker_status()
                    health["circuit_breakers"][name] = cb_status

                    # Check if circuit is open
                    if cb_status.get("state") == "OPEN":
                        health["overall_healthy"] = False

                # Add metrics
                if name in self.metrics:
                    health["metrics"][name] = self.metrics[name].get_metrics()

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health["services"][name] = {
                    "healthy": False,
                    "error": str(e)
                }
                health["overall_healthy"] = False

        # Add system-wide metrics
        total_requests = sum(m.total_requests for m in self.metrics.values())
        total_success = sum(m.successful_requests for m in self.metrics.values())
        overall_success_rate = (
            total_success / total_requests * 100
            if total_requests > 0 else 0
        )

        health["system_metrics"] = {
            "total_requests": total_requests,
            "overall_success_rate": f"{overall_success_rate:.1f}%",
            "active_alerts": len(self.alerts),
            "monitored_services": len(self.services)
        }

        return health

    async def start_monitoring_loop(self, interval: int = 60):
        """Start background monitoring loop"""
        logger.info(f"🔍 Starting monitoring loop (interval: {interval}s)")

        while True:
            try:
                # Perform periodic health checks
                health = await self.get_system_health()

                # Log summary
                if health["overall_healthy"]:
                    logger.info(f"✅ System healthy - {health['system_metrics']['overall_success_rate']} success rate")
                else:
                    logger.warning(f"⚠️ System degraded - check health status")

                # Clear old alerts (keep last hour)
                cutoff = datetime.now() - timedelta(hours=1)
                self.alerts = [
                    a for a in self.alerts
                    if datetime.fromisoformat(a["timestamp"]) > cutoff
                ]

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

            await asyncio.sleep(interval)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all service metrics"""
        return {
            service_name: metrics.get_metrics()
            for service_name, metrics in self.metrics.items()
        }

    def clear_metrics(self):
        """Clear all metrics (useful for testing)"""
        for metrics in self.metrics.values():
            metrics.total_requests = 0
            metrics.successful_requests = 0
            metrics.failed_requests = 0
            metrics.total_latency_ms = 0
            metrics.request_timestamps = []
            metrics.error_timestamps = []
            metrics.last_error = None
            metrics.last_success = None
        self.alerts = []
        logger.info("📊 Metrics cleared")


# Global monitoring instance
monitoring_service = MonitoringService()