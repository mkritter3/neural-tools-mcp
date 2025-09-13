#!/usr/bin/env python3
"""
OpenTelemetry instrumentation for production observability

Features:
- Distributed tracing for GraphRAG operations
- Metrics collection for performance monitoring  
- Auto-instrumentation for FastAPI, HTTP clients, Redis
- OTLP export for DataDog, New Relic, etc.
"""

import os
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Any

# OpenTelemetry core
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

# Exporters
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

# Constants
logger = logging.getLogger(__name__)

# Auto-instrumentation - conditionally import to avoid version conflicts
try:
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor  
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    AUTO_INSTRUMENTATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Auto-instrumentation not available: {e}")
    RequestsInstrumentor = None
    URLLib3Instrumentor = None
    RedisInstrumentor = None
    AUTO_INSTRUMENTATION_AVAILABLE = False

# Service information
SERVICE_NAME = "neural-tools-graphrag"
SERVICE_VERSION = "1.0.0"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


class TelemetryManager:
    """Centralized telemetry management for Neural Tools"""
    
    def __init__(self):
        self.tracer = None
        self.meter = None
        self._initialized = False
        
        # Metrics
        self.query_counter = None
        self.query_duration = None
        self.retrieval_counter = None
        self.rerank_duration = None
        self.cache_hit_counter = None
        self.error_counter = None
    
    def initialize(self, 
                   otlp_endpoint: Optional[str] = None,
                   enable_console_export: bool = False) -> None:
        """Initialize OpenTelemetry with proper configuration"""
        
        if self._initialized:
            logger.warning("Telemetry already initialized, skipping")
            return
            
        try:
            # Create resource with service info
            resource = Resource.create({
                "service.name": SERVICE_NAME,
                "service.version": SERVICE_VERSION,
                "deployment.environment": ENVIRONMENT,
                "service.namespace": "neural-tools"
            })
            
            # Setup tracing
            self._setup_tracing(resource, otlp_endpoint, enable_console_export)
            
            # Setup metrics  
            self._setup_metrics(resource, otlp_endpoint, enable_console_export)
            
            # Auto-instrument common libraries
            self._setup_auto_instrumentation()
            
            self._initialized = True
            logger.info(f"OpenTelemetry initialized for {SERVICE_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            # Don't fail the application if telemetry setup fails
            self._setup_noop_telemetry()
    
    def _setup_tracing(self, 
                      resource: Resource, 
                      otlp_endpoint: Optional[str],
                      enable_console: bool) -> None:
        """Configure distributed tracing"""
        
        trace_provider = TracerProvider(resource=resource)
        
        # OTLP exporter for production
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=ENVIRONMENT == "development"
            )
            trace_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            
        # Console exporter for development
        if enable_console or ENVIRONMENT == "development":
            from opentelemetry.exporter.console import ConsoleSpanExporter
            console_exporter = ConsoleSpanExporter()
            trace_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
        
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(__name__)
    
    def _setup_metrics(self, 
                      resource: Resource,
                      otlp_endpoint: Optional[str],
                      enable_console: bool) -> None:
        """Configure metrics collection"""
        
        readers = []
        
        # OTLP metrics exporter
        if otlp_endpoint:
            otlp_metrics_exporter = OTLPMetricExporter(
                endpoint=otlp_endpoint.replace("4317", "4318"),  # Metrics port
                insecure=ENVIRONMENT == "development"
            )
            readers.append(
                PeriodicExportingMetricReader(otlp_metrics_exporter, export_interval_millis=10000)
            )
        
        # Console metrics for development
        if enable_console or ENVIRONMENT == "development":
            from opentelemetry.exporter.console import ConsoleMetricExporter
            console_metrics = ConsoleMetricExporter()
            readers.append(
                PeriodicExportingMetricReader(console_metrics, export_interval_millis=30000)
            )
        
        if readers:
            meter_provider = MeterProvider(resource=resource, metric_readers=readers)
        else:
            meter_provider = MeterProvider(resource=resource)
            
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)
        
        # Create application metrics
        self._create_metrics()
    
    def _create_metrics(self) -> None:
        """Create application-specific metrics"""
        
        if not self.meter:
            return
            
        # Query metrics
        self.query_counter = self.meter.create_counter(
            name="graphrag_queries_total",
            description="Total number of GraphRAG queries processed",
            unit="1"
        )
        
        self.query_duration = self.meter.create_histogram(
            name="graphrag_query_duration_seconds",
            description="Duration of GraphRAG queries",
            unit="s"
        )
        
        # Retrieval metrics
        self.retrieval_counter = self.meter.create_counter(
            name="retrieval_operations_total", 
            description="Total retrieval operations by source",
            unit="1"
        )
        
        self.rerank_duration = self.meter.create_histogram(
            name="rerank_duration_seconds",
            description="Duration of reranking operations",
            unit="s"
        )
        
        # Cache metrics
        self.cache_hit_counter = self.meter.create_counter(
            name="cache_operations_total",
            description="Cache hit/miss operations",
            unit="1"
        )
        
        # Error tracking
        self.error_counter = self.meter.create_counter(
            name="errors_total",
            description="Application errors by type",
            unit="1"
        )
    
    def _setup_auto_instrumentation(self) -> None:
        """Enable auto-instrumentation for common libraries"""
        
        if not AUTO_INSTRUMENTATION_AVAILABLE:
            logger.info("Auto-instrumentation not available, skipping")
            return
            
        try:
            # HTTP client instrumentation (for Anthropic API calls)
            if RequestsInstrumentor:
                RequestsInstrumentor().instrument()
            if URLLib3Instrumentor:
                URLLib3Instrumentor().instrument()
            
            # Redis instrumentation (if using Redis caching)
            if RedisInstrumentor:
                RedisInstrumentor().instrument()
            
            logger.info("Auto-instrumentation enabled for HTTP clients and Redis")
            
        except Exception as e:
            logger.warning(f"Auto-instrumentation setup failed: {e}")
    
    def _setup_noop_telemetry(self) -> None:
        """Setup no-op telemetry if initialization fails"""
        self.tracer = trace.NoOpTracer()
        self._initialized = True
        logger.warning("Using no-op telemetry due to initialization failure")
    
    @contextmanager
    def trace_operation(self, 
                       operation_name: str, 
                       attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations"""
        
        if not self.tracer:
            yield None
            return
            
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    if value is not None:
                        span.set_attribute(key, str(value))
            yield span
    
    def record_query(self, query: str, project: str, duration: float, success: bool) -> None:
        """Record a GraphRAG query metric"""
        
        if not self.query_counter:
            return
            
        attributes = {
            "project": project,
            "success": str(success)
        }
        
        self.query_counter.add(1, attributes)
        if self.query_duration:
            self.query_duration.record(duration, attributes)
    
    def record_retrieval(self, source: str, count: int, duration: float) -> None:
        """Record retrieval operation metrics"""
        
        if not self.retrieval_counter:
            return
            
        attributes = {"source": source}
        self.retrieval_counter.add(count, attributes)
    
    def record_rerank(self, mode: str, input_count: int, duration: float) -> None:
        """Record reranking operation metrics"""
        
        if not self.rerank_duration:
            return
            
        attributes = {
            "mode": mode,
            "input_count": str(input_count)
        }
        self.rerank_duration.record(duration, attributes)
    
    def record_cache_operation(self, operation: str, hit: bool) -> None:
        """Record cache hit/miss metrics"""
        
        if not self.cache_hit_counter:
            return
            
        attributes = {
            "operation": operation,
            "result": "hit" if hit else "miss"
        }
        self.cache_hit_counter.add(1, attributes)
    
    def record_error(self, error_type: str, component: str) -> None:
        """Record application errors"""
        
        if not self.error_counter:
            return
            
        attributes = {
            "error_type": error_type,
            "component": component
        }
        self.error_counter.add(1, attributes)


# Global telemetry instance
telemetry = TelemetryManager()


def setup_telemetry(otlp_endpoint: Optional[str] = None) -> TelemetryManager:
    """Setup telemetry for the application"""
    
    # Get OTLP endpoint from environment if not provided
    if not otlp_endpoint:
        otlp_endpoint = os.getenv("OTLP_ENDPOINT")
    
    # Enable console export in development
    enable_console = ENVIRONMENT in ["development", "test"]
    
    telemetry.initialize(
        otlp_endpoint=otlp_endpoint,
        enable_console_export=enable_console
    )
    
    return telemetry


def get_telemetry() -> TelemetryManager:
    """Get the global telemetry instance"""
    return telemetry