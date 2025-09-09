#!/usr/bin/env python3
"""
Structured Logging Infrastructure (Phase 2.3)

Provides production-grade structured logging with:
- JSON output for log aggregation (ELK, Datadog, etc.)
- Structured context and metadata
- Call site information (function, line, file)
- ISO timestamp formatting
- Exception stack traces
- Environment-based log levels
- Performance-optimized processors

Key features:
- Replace all print() statements with structured logs
- Consistent log schema across components
- Integration with telemetry and monitoring
- Production-ready log formatting
"""

import os
import sys
import logging
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# Conditionally import structlog with graceful fallback
try:
    import structlog
    from structlog.processors import CallsiteParameterAdder, JSONRenderer
    from structlog import CallsiteParameter
    STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None
    CallsiteParameterAdder = None
    JSONRenderer = None
    CallsiteParameter = None
    STRUCTLOG_AVAILABLE = False
    print("WARNING: structlog not available, falling back to standard logging")

from infrastructure.telemetry import get_telemetry

# Service information
SERVICE_NAME = os.getenv("SERVICE_NAME", "neural-tools")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "2.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


class StructuredLogger:
    """
    Structured logger with production-grade features
    
    Provides consistent structured logging interface whether
    structlog is available or not.
    """
    
    def __init__(self, name: str = SERVICE_NAME):
        self.name = name
        self.telemetry = get_telemetry()
        
        if STRUCTLOG_AVAILABLE:
            self._setup_structlog()
            self.logger = structlog.get_logger(name)
        else:
            self._setup_standard_logging()
            self.logger = logging.getLogger(name)
    
    def _setup_structlog(self):
        """Configure structlog for production"""
        if not STRUCTLOG_AVAILABLE:
            return
            
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        # Add callsite info if available
        try:
            processors.insert(3, CallsiteParameterAdder(
                parameters=[
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                    CallsiteParameter.PATHNAME,
                ]
            ))
        except (AttributeError, TypeError):
            # Older version of structlog might not have CallsiteParameterAdder
            pass
        
        # JSON output for production, pretty for development
        if ENVIRONMENT == "development":
            processors.append(structlog.dev.ConsoleRenderer())
        else:
            processors.append(JSONRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard logging
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
            stream=sys.stdout
        )
    
    def _setup_standard_logging(self):
        """Setup standard logging when structlog is not available"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        root_logger.addHandler(handler)
    
    def _add_context(self, event: str, **kwargs) -> Dict[str, Any]:
        """Add standard context to log entries"""
        context = {
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "environment": ENVIRONMENT,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        # Add performance metrics if telemetry is available
        try:
            context["telemetry_initialized"] = self.telemetry._initialized
        except AttributeError:
            pass
        
        return context
    
    def debug(self, event: str, **kwargs):
        """Log debug message with structured context"""
        context = self._add_context(event, **kwargs)
        
        if STRUCTLOG_AVAILABLE:
            self.logger.debug(event, **context)
        else:
            self.logger.debug(f"{event}: {context}")
    
    def info(self, event: str, **kwargs):
        """Log info message with structured context"""
        context = self._add_context(event, **kwargs)
        
        if STRUCTLOG_AVAILABLE:
            self.logger.info(event, **context)
        else:
            self.logger.info(f"{event}: {context}")
    
    def warning(self, event: str, **kwargs):
        """Log warning message with structured context"""
        context = self._add_context(event, **kwargs)
        
        if STRUCTLOG_AVAILABLE:
            self.logger.warning(event, **context)
        else:
            self.logger.warning(f"{event}: {context}")
    
    def error(self, event: str, **kwargs):
        """Log error message with structured context and telemetry"""
        context = self._add_context(event, **kwargs)
        
        # Record error in telemetry if available
        error_type = kwargs.get('error_type', 'unknown')
        component = kwargs.get('component', self.name)
        try:
            self.telemetry.record_error(error_type, component)
        except Exception:
            pass  # Don't fail logging if telemetry fails
        
        if STRUCTLOG_AVAILABLE:
            self.logger.error(event, **context)
        else:
            self.logger.error(f"{event}: {context}")
    
    def critical(self, event: str, **kwargs):
        """Log critical message with structured context"""
        context = self._add_context(event, **kwargs)
        
        # Always record critical errors in telemetry
        try:
            self.telemetry.record_error("critical", self.name)
        except Exception:
            pass
        
        if STRUCTLOG_AVAILABLE:
            self.logger.critical(event, **context)
        else:
            self.logger.critical(f"{event}: {context}")
    
    def exception(self, event: str, **kwargs):
        """Log exception with full stack trace and context"""
        context = self._add_context(event, **kwargs)
        
        try:
            self.telemetry.record_error("exception", self.name)
        except Exception:
            pass
        
        if STRUCTLOG_AVAILABLE:
            self.logger.exception(event, **context)
        else:
            self.logger.exception(f"{event}: {context}")
    
    # Convenience methods for common log events
    def query_start(self, query: str, **kwargs):
        """Log query start event"""
        self.info("query_started",
                 query=query[:100] + "..." if len(query) > 100 else query,
                 query_length=len(query),
                 **kwargs)
    
    def query_complete(self, query: str, duration: float, 
                      result_count: int, **kwargs):
        """Log query completion event"""
        self.info("query_completed",
                 query=query[:100] + "..." if len(query) > 100 else query,
                 duration_ms=round(duration * 1000, 2),
                 result_count=result_count,
                 **kwargs)
    
    def search_result(self, score: float, file_path: str, 
                     query: str, **kwargs):
        """Log search result event"""
        self.info("search_result",
                 score=round(score, 3),
                 file_path=file_path,
                 query=query[:50] + "..." if len(query) > 50 else query,
                 **kwargs)
    
    def rerank_start(self, query: str, doc_count: int, 
                    mode: str = "unknown", **kwargs):
        """Log reranking start event"""
        self.info("rerank_started",
                 query=query[:100] + "..." if len(query) > 100 else query,
                 doc_count=doc_count,
                 mode=mode,
                 **kwargs)
    
    def rerank_complete(self, query: str, duration: float, 
                       confidence: float, mode: str, **kwargs):
        """Log reranking completion event"""
        self.info("rerank_completed",
                 query=query[:100] + "..." if len(query) > 100 else query,
                 duration_ms=round(duration * 1000, 2),
                 confidence=round(confidence, 3),
                 mode=mode,
                 **kwargs)
    
    def rerank_failed(self, query: str, error: str, doc_count: int, 
                     mode: str = "unknown", **kwargs):
        """Log reranking failure event"""
        self.error("rerank_failed",
                  query=query[:100] + "..." if len(query) > 100 else query,
                  error=str(error),
                  error_type=type(error).__name__ if hasattr(error, '__class__') else "unknown",
                  doc_count=doc_count,
                  mode=mode,
                  component="reranker",
                  **kwargs)
    
    def api_request(self, method: str, endpoint: str, 
                   status_code: Optional[int] = None,
                   duration: Optional[float] = None, **kwargs):
        """Log API request event"""
        log_data = {
            "method": method,
            "endpoint": endpoint,
            **kwargs
        }
        
        if status_code is not None:
            log_data["status_code"] = status_code
        
        if duration is not None:
            log_data["duration_ms"] = round(duration * 1000, 2)
        
        if status_code and status_code >= 400:
            self.error("api_request_failed", **log_data)
        else:
            self.info("api_request", **log_data)
    
    def cache_operation(self, operation: str, key: str, 
                       hit: bool, **kwargs):
        """Log cache operation event"""
        self.debug("cache_operation",
                  operation=operation,
                  cache_key=key,
                  cache_hit=hit,
                  **kwargs)
    
    def performance_metric(self, metric_name: str, value: Union[int, float],
                          unit: str = "ms", **kwargs):
        """Log performance metric"""
        self.info("performance_metric",
                 metric=metric_name,
                 value=value,
                 unit=unit,
                 **kwargs)


# Global structured logger instance
_global_logger: Optional[StructuredLogger] = None


def get_logger(name: str = SERVICE_NAME) -> StructuredLogger:
    """Get or create structured logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = StructuredLogger(name)
    
    return _global_logger


def setup_structured_logging(service_name: str = SERVICE_NAME) -> StructuredLogger:
    """
    Setup structured logging for the entire application
    
    Returns:
        Configured StructuredLogger instance
    """
    global _global_logger
    _global_logger = StructuredLogger(service_name)
    return _global_logger


# Convenience function for migration from print statements
def log_info(message: str, **kwargs):
    """Convenience function to replace print statements"""
    logger = get_logger()
    logger.info("application_log", message=message, **kwargs)


def log_error(message: str, error: Optional[Exception] = None, **kwargs):
    """Convenience function for error logging"""
    logger = get_logger()
    context = {"message": message, **kwargs}
    
    if error:
        context["error"] = str(error)
        context["error_type"] = type(error).__name__
    
    logger.error("application_error", **context)


def log_debug(message: str, **kwargs):
    """Convenience function for debug logging"""
    logger = get_logger()
    logger.debug("application_debug", message=message, **kwargs)


def log_performance(operation: str, duration: float, **kwargs):
    """Convenience function for performance logging"""
    logger = get_logger()
    logger.performance_metric(
        f"{operation}_duration",
        duration * 1000,  # Convert to ms
        unit="ms",
        operation=operation,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Structured Logging Infrastructure")
    
    # Setup logging
    logger = setup_structured_logging("test-service")
    print("✓ Structured logging setup complete")
    
    # Test different log levels
    logger.debug("test_debug", component="test", value=42)
    logger.info("test_info", component="test", status="running")
    logger.warning("test_warning", component="test", reason="high_memory_usage")
    
    # Test convenience methods
    logger.query_start("test query for search", project="test-project")
    
    import time
    time.sleep(0.01)  # Simulate work
    
    logger.query_complete("test query for search", 0.01, 5, project="test-project")
    
    logger.search_result(0.85, "/path/to/file.py", "test query")
    
    logger.rerank_start("test rerank query", 10, mode="haiku")
    logger.rerank_complete("test rerank query", 0.05, 0.9, "haiku")
    
    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger.rerank_failed("test rerank query", e, 10, mode="haiku")
    
    # Test performance logging
    logger.performance_metric("test_latency", 45.3, "ms", operation="search")
    
    # Test convenience functions
    log_info("Migration test: replacing print statement", component="migration")
    log_error("Migration test: error handling", ValueError("test error"))
    log_performance("test_operation", 0.123)
    
    print("✅ Structured logging tests complete!")
    print("📊 All log events emitted with structured context")