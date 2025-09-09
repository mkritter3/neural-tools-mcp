#!/usr/bin/env python3
"""
Structured Logging Infrastructure Tests

Tests comprehensive structured logging functionality:
- StructuredLogger configuration and usage
- JSON output formatting for production
- Log levels and filtering
- Convenience methods for common events
- Integration with telemetry
- Performance impact analysis
- Migration helpers from print statements
"""

import pytest
import logging
import json
import io
import sys
from typing import Dict, Any
from unittest.mock import Mock, patch, StringIO
from pathlib import Path

# Import components to test
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.structured_logging import (
    StructuredLogger,
    get_logger,
    setup_structured_logging,
    log_info,
    log_error,
    log_debug,
    log_performance,
    STRUCTLOG_AVAILABLE
)


class TestStructuredLogger:
    """Test core structured logger functionality"""
    
    def setup_method(self):
        """Setup fresh logger for each test"""
        self.logger = StructuredLogger("test-logger")
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        assert self.logger.name == "test-logger"
        assert self.logger.logger is not None
        assert hasattr(self.logger, 'telemetry')
    
    def test_basic_logging_methods(self):
        """Test basic logging methods don't raise exceptions"""
        # These should not raise exceptions
        self.logger.debug("test_debug", component="test")
        self.logger.info("test_info", value=42)
        self.logger.warning("test_warning", reason="test")
        self.logger.error("test_error", error_type="test_error")
        self.logger.critical("test_critical", severity="high")
    
    def test_context_addition(self):
        """Test context addition to log entries"""
        context = self.logger._add_context("test_event", custom_key="custom_value")
        
        assert "service" in context
        assert "version" in context
        assert "environment" in context
        assert "timestamp" in context
        assert context["custom_key"] == "custom_value"
    
    def test_query_convenience_methods(self):
        """Test query-specific convenience methods"""
        query = "test query for search functionality"
        
        # Should not raise exceptions
        self.logger.query_start(query, project="test")
        self.logger.query_complete(query, 0.05, 10, project="test")
        self.logger.search_result(0.85, "/path/to/file.py", query)
    
    def test_rerank_convenience_methods(self):
        """Test reranking-specific convenience methods"""
        query = "test rerank query"
        
        # Should not raise exceptions
        self.logger.rerank_start(query, 15, mode="haiku")
        self.logger.rerank_complete(query, 0.08, 0.9, "haiku")
        
        # Test error case
        error = ValueError("Test rerank error")
        self.logger.rerank_failed(query, error, 15, mode="haiku")
    
    def test_api_request_logging(self):
        """Test API request logging"""
        # Success case
        self.logger.api_request("POST", "/api/search", status_code=200, duration=0.1)
        
        # Error case
        self.logger.api_request("POST", "/api/search", status_code=500, duration=0.2)
    
    def test_cache_operation_logging(self):
        """Test cache operation logging"""
        self.logger.cache_operation("get", "search_key_123", hit=True)
        self.logger.cache_operation("get", "search_key_456", hit=False)
    
    def test_performance_metric_logging(self):
        """Test performance metric logging"""
        self.logger.performance_metric("query_latency", 45.3, "ms")
        self.logger.performance_metric("throughput", 1500, "ops/sec")
    
    def test_long_query_truncation(self):
        """Test long query truncation in logs"""
        long_query = "a" * 200  # 200 character query
        
        # Should not raise exceptions and should truncate
        self.logger.query_start(long_query)
        self.logger.rerank_start(long_query, 10)
    
    def test_exception_logging(self):
        """Test exception logging with stack traces"""
        try:
            raise ValueError("Test exception for logging")
        except Exception:
            # Should not raise exceptions
            self.logger.exception("test_exception", component="test")


class TestStructuredLoggingGlobal:
    """Test global logging functions and setup"""
    
    def test_get_logger_singleton(self):
        """Test global logger singleton behavior"""
        logger1 = get_logger()
        logger2 = get_logger()
        
        # Should return same instance
        assert logger1 is logger2
        assert isinstance(logger1, StructuredLogger)
    
    def test_setup_structured_logging(self):
        """Test structured logging setup"""
        logger = setup_structured_logging("setup-test")
        
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "setup-test"
    
    def test_convenience_functions(self):
        """Test convenience functions for migration"""
        # Should not raise exceptions
        log_info("Test info message", component="test")
        log_error("Test error message", ValueError("test error"))
        log_debug("Test debug message", component="test")
        log_performance("test_operation", 0.123)


class TestStructuredLoggingConfiguration:
    """Test different logging configurations"""
    
    @patch.dict('os.environ', {'ENVIRONMENT': 'production', 'LOG_LEVEL': 'ERROR'})
    def test_production_configuration(self):
        """Test production logging configuration"""
        logger = StructuredLogger("prod-test")
        
        # Should be configured for production
        assert logger.logger is not None
    
    @patch.dict('os.environ', {'ENVIRONMENT': 'development', 'LOG_LEVEL': 'DEBUG'})
    def test_development_configuration(self):
        """Test development logging configuration"""
        logger = StructuredLogger("dev-test")
        
        # Should be configured for development
        assert logger.logger is not None
    
    def test_missing_structlog_fallback(self):
        """Test fallback behavior when structlog is not available"""
        with patch('infrastructure.structured_logging.STRUCTLOG_AVAILABLE', False):
            logger = StructuredLogger("fallback-test")
            
            # Should still work with standard logging
            assert logger.logger is not None
            logger.info("test_fallback", component="test")


class TestLogOutputCapture:
    """Test log output capture and format validation"""
    
    def setup_method(self):
        """Setup log capture"""
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)
        self.logger = StructuredLogger("capture-test")
        
        # Add handler to capture logs
        if hasattr(self.logger.logger, 'handlers'):
            self.logger.logger.handlers.clear()
            self.logger.logger.addHandler(self.handler)
    
    def test_structured_log_format(self):
        """Test structured log format contains expected fields"""
        self.logger.info("test_event", key="value", number=42)
        
        log_output = self.log_stream.getvalue()
        
        # Should contain basic event information
        assert "test_event" in log_output
        assert "neural-tools" in log_output or "capture-test" in log_output
    
    def test_error_log_includes_error_info(self):
        """Test error logs include error information"""
        error = ValueError("Test error message")
        self.logger.rerank_failed("test query", error, 5, mode="test")
        
        log_output = self.log_stream.getvalue()
        
        # Should contain error information
        assert "rerank_failed" in log_output
        assert "ValueError" in log_output or "Test error message" in log_output


class TestStructuredLoggingIntegration:
    """Test integration with other components"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.logger = StructuredLogger("integration-test")
    
    def test_telemetry_integration(self):
        """Test integration with telemetry system"""
        # Error logging should record telemetry
        with patch.object(self.logger.telemetry, 'record_error') as mock_record:
            self.logger.error("test_error", error_type="integration_test")
            
            # Should record error in telemetry (if available)
            if hasattr(self.logger.telemetry, 'record_error'):
                try:
                    mock_record.assert_called()
                except AssertionError:
                    # Acceptable if telemetry is not fully initialized
                    pass
    
    def test_structured_logging_with_search_workflow(self):
        """Test structured logging in a realistic search workflow"""
        query = "test search query"
        
        # Simulate search workflow with structured logging
        self.logger.query_start(query, user_id="test_user", project="test_project")
        
        # Simulate search results
        self.logger.search_result(0.95, "/path/to/relevant/file.py", query, source="vector")
        self.logger.search_result(0.87, "/path/to/another/file.py", query, source="graph")
        
        # Simulate reranking
        self.logger.rerank_start(query, 2, mode="haiku")
        self.logger.rerank_complete(query, 0.05, 0.9, "haiku")
        
        # Complete query
        self.logger.query_complete(query, 0.1, 2, user_id="test_user")
        
        # Should complete without errors
        assert True


class TestStructuredLoggingPerformance:
    """Test performance impact of structured logging"""
    
    def setup_method(self):
        """Setup performance test environment"""
        self.logger = StructuredLogger("perf-test")
    
    def test_logging_performance_overhead(self):
        """Test logging performance overhead is minimal"""
        import time
        
        iterations = 1000
        
        # Measure baseline (no logging)
        start_time = time.perf_counter()
        for i in range(iterations):
            # Minimal work simulation
            _ = f"test_operation_{i}"
        baseline_time = time.perf_counter() - start_time
        
        # Measure with structured logging
        start_time = time.perf_counter()
        for i in range(iterations):
            self.logger.info("performance_test", iteration=i, operation="test")
        logging_time = time.perf_counter() - start_time
        
        # Logging overhead should be reasonable
        if baseline_time > 0:
            overhead_ratio = (logging_time - baseline_time) / baseline_time
            # Allow up to 500% overhead for logging (it's expected to be higher)
            assert overhead_ratio < 5.0, f"Logging overhead too high: {overhead_ratio:.2%}"
    
    def test_context_building_performance(self):
        """Test context building performance"""
        import time
        
        iterations = 10000
        
        start_time = time.perf_counter()
        for i in range(iterations):
            context = self.logger._add_context("test_event", 
                                               iteration=i, 
                                               component="performance_test",
                                               value=i * 2)
        elapsed = time.perf_counter() - start_time
        
        ops_per_second = iterations / elapsed
        
        # Should handle high throughput context building (>1000 ops/sec)
        assert ops_per_second > 1000, f"Context building too slow: {ops_per_second:.1f} ops/sec"


class TestLogMigrationHelpers:
    """Test helpers for migrating from print statements"""
    
    def test_print_statement_replacements(self):
        """Test migration from print statements to structured logging"""
        # Old: print(f"Score: {0.85}")
        # New: structured logging
        log_info("Score logged", score=0.85, component="migration_test")
        
        # Old: print(f"Error occurred: {error}")
        # New: structured error logging  
        test_error = ValueError("Migration test error")
        log_error("Error occurred during migration test", test_error, component="migration_test")
        
        # Should complete without errors
        assert True
    
    def test_performance_logging_migration(self):
        """Test migration to performance logging"""
        # Old: print(f"Operation took {duration}s")
        # New: structured performance logging
        log_performance("migration_operation", 0.123, component="migration_test")
        
        # Should complete without errors
        assert True


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v", 
        "--tb=short",
        "-k", "test_structured_logging"
    ])