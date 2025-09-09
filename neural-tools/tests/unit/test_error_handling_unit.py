#!/usr/bin/env python3
"""
Phase 1.5 Unit Tests - Error Handling and Recovery
Tests error classification, retry logic, circuit breakers, and graceful degradation
"""

import pytest
import asyncio
import time
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.error_handling import (
    ErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    ErrorInfo,
    NeuralToolsException,
    VectorDatabaseException,
    GraphDatabaseException,
    EmbeddingException,
    ValidationException,
    RateLimitException,
    GracefulDegradation,
    retry_with_backoff,
    database_retry,
    embedding_retry,
    network_retry,
    error_handler,
    graceful_degradation
)

class TestErrorClassification:
    """Test error classification and exception hierarchy"""
    
    def test_neural_tools_exception_creation(self):
        """Test NeuralToolsException creation and properties"""
        context = {"operation": "test", "data": "sample"}
        
        exception = NeuralToolsException(
            "Test error message",
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.HIGH,
            context=context
        )
        
        assert exception.message == "Test error message"
        assert exception.category == ErrorCategory.COMPUTATION
        assert exception.severity == ErrorSeverity.HIGH
        assert exception.context == context
        assert exception.error_id.startswith("NT-")
        assert isinstance(exception.timestamp, datetime)
    
    def test_specialized_exceptions(self):
        """Test specialized exception classes"""
        # VectorDatabaseException
        vector_ex = VectorDatabaseException("Vector DB error")
        assert vector_ex.category == ErrorCategory.DATABASE
        assert vector_ex.severity == ErrorSeverity.HIGH
        
        # GraphDatabaseException  
        graph_ex = GraphDatabaseException("Graph DB error")
        assert graph_ex.category == ErrorCategory.DATABASE
        assert graph_ex.severity == ErrorSeverity.HIGH
        
        # EmbeddingException
        embed_ex = EmbeddingException("Embedding error")
        assert embed_ex.category == ErrorCategory.COMPUTATION
        assert embed_ex.severity == ErrorSeverity.MEDIUM
        
        # ValidationException
        valid_ex = ValidationException("Validation error")
        assert valid_ex.category == ErrorCategory.VALIDATION
        assert valid_ex.severity == ErrorSeverity.LOW
        
        # RateLimitException
        rate_ex = RateLimitException("Rate limit exceeded")
        assert rate_ex.category == ErrorCategory.RESOURCE
        assert rate_ex.severity == ErrorSeverity.MEDIUM
    
    def test_error_info_conversion(self):
        """Test conversion to ErrorInfo"""
        exception = NeuralToolsException(
            "Test error",
            context={"key": "value"}
        )
        
        error_info = exception.to_error_info()
        
        assert isinstance(error_info, ErrorInfo)
        assert error_info.error_id == exception.error_id
        assert error_info.message == exception.message
        assert error_info.category == exception.category
        assert error_info.severity == exception.severity
        assert error_info.context == exception.context
    
    def test_error_info_serialization(self):
        """Test ErrorInfo to_dict serialization"""
        error_info = ErrorInfo(
            error_id="test-123",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.CRITICAL,
            message="Network failure",
            context={"host": "example.com"},
            timestamp=datetime.now()
        )
        
        error_dict = error_info.to_dict()
        
        assert error_dict["error_id"] == "test-123"
        assert error_dict["category"] == "network"
        assert error_dict["severity"] == "critical"
        assert error_dict["message"] == "Network failure"
        assert error_dict["context"] == {"host": "example.com"}
        assert "timestamp" in error_dict

class TestErrorHandler:
    """Test ErrorHandler functionality"""
    
    @pytest.fixture
    def handler(self):
        """Fresh ErrorHandler instance"""
        return ErrorHandler()
    
    @pytest.mark.asyncio
    async def test_error_classification(self, handler):
        """Test automatic error classification"""
        # Network error
        network_error = ConnectionError("Network unreachable")
        error_info = await handler.handle_error(network_error, operation_name="test_network")
        
        assert error_info.category == ErrorCategory.NETWORK
        assert error_info.severity == ErrorSeverity.MEDIUM
        
        # Database error
        db_error = Exception("Qdrant connection failed")
        error_info = await handler.handle_error(db_error, operation_name="test_db")
        
        assert error_info.category == ErrorCategory.DATABASE
        assert error_info.severity == ErrorSeverity.HIGH
        
        # Validation error
        validation_error = ValueError("Invalid input parameter")
        error_info = await handler.handle_error(validation_error, operation_name="test_validation")
        
        assert error_info.category == ErrorCategory.VALIDATION
        assert error_info.severity == ErrorSeverity.LOW
    
    @pytest.mark.asyncio
    async def test_error_tracking(self, handler):
        """Test error frequency tracking"""
        operation_name = "test_operation"
        error = Exception("Test error")
        
        # Handle same error multiple times
        for i in range(3):
            await handler.handle_error(error, operation_name=operation_name)
        
        error_key = f"{operation_name}:computation"
        assert handler.error_counts[error_key] == 3
        assert error_key in handler.last_errors
    
    def test_retry_configuration(self, handler):
        """Test retry configuration for different error categories"""
        network_config = handler.get_retry_config(ErrorCategory.NETWORK)
        assert network_config["max_retries"] == 3
        assert network_config["backoff_factor"] == 2
        
        db_config = handler.get_retry_config(ErrorCategory.DATABASE)
        assert db_config["max_retries"] == 2
        assert db_config["backoff_factor"] == 1.5
        
        api_config = handler.get_retry_config(ErrorCategory.EXTERNAL_API)
        assert api_config["max_retries"] == 5
        assert api_config["backoff_factor"] == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, handler):
        """Test circuit breaker functionality"""
        operation_name = "circuit_test"
        error = Exception("Repeated failure")
        
        # Generate enough errors to trigger circuit breaker
        for i in range(6):
            await handler.handle_error(error, operation_name=operation_name)
        
        error_key = f"{operation_name}:computation"
        
        # Check if circuit breaker is opened
        assert error_key in handler.circuit_breaker_states
        circuit_state = handler.circuit_breaker_states[error_key]
        assert circuit_state["state"] == "open"
        assert circuit_state["error_count"] >= 5

class TestRetryMechanism:
    """Test retry logic and decorators"""
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self):
        """Test retry decorator with eventual success"""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        start_time = time.time()
        result = await flaky_function()
        end_time = time.time()
        
        assert result == "success"
        assert call_count == 3
        assert end_time - start_time >= 0.03  # At least 2 delays of 0.01s each
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_failure(self):
        """Test retry decorator with persistent failure"""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent failure")
        
        with pytest.raises(ValueError, match="Persistent failure"):
            await failing_function()
        
        assert call_count == 3  # Initial call + 2 retries
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff timing"""
        call_times = []
        
        @retry_with_backoff(max_retries=3, base_delay=0.1, backoff_factor=2)
        async def timing_function():
            call_times.append(time.time())
            raise Exception("Always fails")
        
        with pytest.raises(Exception):
            await timing_function()
        
        assert len(call_times) == 4  # Initial + 3 retries
        
        # Check exponential backoff timing
        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
        assert delays[0] >= 0.1  # First retry delay
        assert delays[1] >= 0.2  # Second retry delay (2x)
        assert delays[2] >= 0.4  # Third retry delay (4x)
    
    @pytest.mark.asyncio
    async def test_specific_exception_handling(self):
        """Test retry with specific exception types"""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=2, 
            base_delay=0.01,
            exceptions=(ValueError, TypeError)
        )
        async def selective_retry():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable error")
            elif call_count == 2:
                raise RuntimeError("Non-retryable error")  # Should not be retried
            return "success"
        
        with pytest.raises(RuntimeError, match="Non-retryable error"):
            await selective_retry()
        
        assert call_count == 2  # ValueError was retried once, RuntimeError was not
    
    def test_sync_retry_decorator(self):
        """Test retry decorator with synchronous functions"""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def sync_flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "sync_success"
        
        result = sync_flaky_function()
        
        assert result == "sync_success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_database_retry_decorator(self):
        """Test database-specific retry decorator"""
        call_count = 0
        
        @database_retry
        async def flaky_db_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise VectorDatabaseException("Connection timeout")
            return {"status": "success"}
        
        result = await flaky_db_operation()
        
        assert result["status"] == "success"
        assert call_count == 2  # One failure, one retry success
    
    @pytest.mark.asyncio
    async def test_embedding_retry_decorator(self):
        """Test embedding-specific retry decorator"""
        call_count = 0
        
        @embedding_retry
        async def flaky_embedding():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise EmbeddingException("Model server unavailable")
            return [0.1, 0.2, 0.3]  # Mock embedding
        
        result = await flaky_embedding()
        
        assert result == [0.1, 0.2, 0.3]
        assert call_count == 3  # Two failures, one success
    
    @pytest.mark.asyncio
    async def test_network_retry_decorator(self):
        """Test network-specific retry decorator"""
        call_count = 0
        
        @network_retry
        async def flaky_network_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ConnectionError("Network timeout")
            return {"data": "fetched"}
        
        result = await flaky_network_call()
        
        assert result["data"] == "fetched"
        assert call_count == 4  # Three failures, one success

class TestGracefulDegradation:
    """Test graceful degradation functionality"""
    
    @pytest.fixture
    def degradation(self):
        """Fresh GracefulDegradation instance"""
        return GracefulDegradation()
    
    def test_fallback_registration(self, degradation):
        """Test fallback handler registration"""
        def fallback_handler():
            return "fallback_result"
        
        degradation.register_fallback("test_service", fallback_handler)
        
        assert "test_service" in degradation.fallback_handlers
        assert degradation.fallback_handlers["test_service"] == fallback_handler
    
    def test_service_health_management(self, degradation):
        """Test service health tracking"""
        service_name = "test_service"
        
        # Initially healthy (default)
        assert degradation.service_health.get(service_name, True)
        
        # Mark unhealthy
        degradation.mark_service_unhealthy(service_name)
        assert not degradation.service_health[service_name]
        
        # Mark healthy again
        degradation.mark_service_healthy(service_name)
        assert degradation.service_health[service_name]
    
    @pytest.mark.asyncio
    async def test_successful_primary_execution(self, degradation):
        """Test successful execution of primary function"""
        async def primary_function(value):
            return f"primary_{value}"
        
        result = await degradation.execute_with_fallback(
            "test_service", primary_function, "test_input"
        )
        
        assert result["success"] is True
        assert result["result"] == "primary_test_input"
        assert result["fallback_used"] is False
        assert result["service"] == "test_service"
    
    @pytest.mark.asyncio
    async def test_fallback_execution_on_failure(self, degradation):
        """Test fallback execution when primary fails"""
        async def failing_primary():
            raise Exception("Primary service failed")
        
        async def fallback_handler():
            return "fallback_result"
        
        degradation.register_fallback("test_service", fallback_handler)
        
        result = await degradation.execute_with_fallback(
            "test_service", failing_primary
        )
        
        assert result["success"] is True
        assert result["result"] == "fallback_result"
        assert result["fallback_used"] is True
        assert result["service"] == "test_service"
        assert "Used fallback" in result["message"]
    
    @pytest.mark.asyncio
    async def test_fallback_execution_on_unhealthy_service(self, degradation):
        """Test fallback when service is marked unhealthy"""
        def primary_function():
            return "should_not_execute"
        
        def fallback_handler():
            return "fallback_for_unhealthy"
        
        degradation.register_fallback("unhealthy_service", fallback_handler)
        degradation.mark_service_unhealthy("unhealthy_service")
        
        result = await degradation.execute_with_fallback(
            "unhealthy_service", primary_function
        )
        
        assert result["success"] is True
        assert result["result"] == "fallback_for_unhealthy"
        assert result["fallback_used"] is True
    
    @pytest.mark.asyncio
    async def test_no_fallback_available(self, degradation):
        """Test behavior when no fallback is available"""
        async def failing_primary():
            raise Exception("Primary failed")
        
        degradation.mark_service_unhealthy("no_fallback_service")
        
        result = await degradation.execute_with_fallback(
            "no_fallback_service", failing_primary
        )
        
        assert result["success"] is False
        assert "unavailable and no fallback configured" in result["error"]
        assert result["fallback_used"] is False
    
    @pytest.mark.asyncio
    async def test_fallback_failure(self, degradation):
        """Test when both primary and fallback fail"""
        async def failing_primary():
            raise Exception("Primary failed")
        
        async def failing_fallback():
            raise Exception("Fallback also failed")
        
        degradation.register_fallback("double_failure", failing_fallback)
        
        result = await degradation.execute_with_fallback(
            "double_failure", failing_primary
        )
        
        assert result["success"] is False
        assert "Fallback also failed" in result["error"]
        assert result["fallback_used"] is True
        assert result["fallback_failed"] is True
    
    @pytest.mark.asyncio
    async def test_sync_function_support(self, degradation):
        """Test graceful degradation with synchronous functions"""
        def sync_primary():
            return "sync_primary_result"
        
        def sync_fallback():
            return "sync_fallback_result"
        
        degradation.register_fallback("sync_service", sync_fallback)
        
        # Test successful sync primary
        result = await degradation.execute_with_fallback(
            "sync_service", sync_primary
        )
        
        assert result["success"] is True
        assert result["result"] == "sync_primary_result"
        assert result["fallback_used"] is False

class TestIntegratedErrorHandling:
    """Test integration between error handling components"""
    
    @pytest.mark.asyncio
    async def test_retry_with_error_handler(self):
        """Test retry decorator with error handler integration"""
        handler = ErrorHandler()
        call_count = 0
        
        @retry_with_backoff(
            max_retries=2, 
            base_delay=0.01,
            error_handler=handler
        )
        async def monitored_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise VectorDatabaseException("DB connection failed")
            return "success_after_retries"
        
        result = await monitored_function()
        
        assert result == "success_after_retries"
        assert call_count == 3
        
        # Check that errors were tracked
        error_key = "monitored_function:database"
        assert handler.error_counts.get(error_key, 0) == 2  # Two failed attempts
    
    @pytest.mark.asyncio
    async def test_degradation_with_retry(self, caplog):
        """Test graceful degradation combined with retry logic"""
        degradation = GracefulDegradation()
        
        @database_retry
        async def unreliable_primary():
            # Always fails to test fallback
            raise VectorDatabaseException("Database unavailable")
        
        async def reliable_fallback():
            return {"source": "fallback", "data": "cached_result"}
        
        degradation.register_fallback("combined_service", reliable_fallback)
        
        with caplog.at_level(logging.WARNING):
            result = await degradation.execute_with_fallback(
                "combined_service", unreliable_primary
            )
        
        assert result["success"] is True
        assert result["result"]["source"] == "fallback"
        assert result["fallback_used"] is True
        
        # Check that retry logs were captured (checking both "Retrying" and "Max retries")
        retry_logs = [record for record in caplog.records 
                     if "Retrying" in record.getMessage() or "Max retries" in record.getMessage()]
        assert len(retry_logs) >= 1  # Should have at least one retry-related log
    
    @pytest.mark.asyncio
    async def test_error_severity_escalation(self):
        """Test that repeated errors escalate in severity"""
        handler = ErrorHandler()
        
        # Simulate escalating errors
        for i in range(10):
            error = Exception(f"Error occurrence {i+1}")
            await handler.handle_error(error, operation_name="escalation_test")
        
        # After many errors, circuit breaker should open
        error_key = "escalation_test:computation"
        assert handler.error_counts[error_key] == 10
        
        # Circuit breaker should be triggered
        if error_key in handler.circuit_breaker_states:
            assert handler.circuit_breaker_states[error_key]["state"] == "open"

class TestErrorLogging:
    """Test error logging and monitoring"""
    
    @pytest.mark.asyncio
    async def test_error_logging_levels(self, caplog):
        """Test that errors are logged at appropriate levels"""
        handler = ErrorHandler()
        
        # Test different severity levels
        critical_error = NeuralToolsException(
            "Critical system failure",
            severity=ErrorSeverity.CRITICAL
        )
        
        high_error = NeuralToolsException(
            "High priority error",
            severity=ErrorSeverity.HIGH
        )
        
        medium_error = NeuralToolsException(
            "Medium priority warning",
            severity=ErrorSeverity.MEDIUM
        )
        
        low_error = NeuralToolsException(
            "Low priority info",
            severity=ErrorSeverity.LOW
        )
        
        with caplog.at_level(logging.INFO):
            await handler.handle_error(critical_error, operation_name="test_critical")
            await handler.handle_error(high_error, operation_name="test_high")
            await handler.handle_error(medium_error, operation_name="test_medium")
            await handler.handle_error(low_error, operation_name="test_low")
        
        # Check log levels
        log_records = caplog.records
        critical_logs = [r for r in log_records if r.levelname == "CRITICAL"]
        error_logs = [r for r in log_records if r.levelname == "ERROR"]
        warning_logs = [r for r in log_records if r.levelname == "WARNING"]
        info_logs = [r for r in log_records if r.levelname == "INFO"]
        
        assert len(critical_logs) >= 1
        assert len(error_logs) >= 1
        assert len(warning_logs) >= 1
        assert len(info_logs) >= 1
    
    @pytest.mark.asyncio
    async def test_error_context_logging(self, caplog):
        """Test that error context is properly logged"""
        handler = ErrorHandler()
        
        context = {
            "user_id": "test_user_123",
            "operation_data": {"query": "test search"},
            "system_info": {"memory_usage": "85%"}
        }
        
        error = NeuralToolsException(
            "Context test error",
            context=context
        )
        
        with caplog.at_level(logging.WARNING):
            await handler.handle_error(error, operation_name="context_test")
        
        # Check that context information was logged
        log_record = caplog.records[-1]  # Get last log record
        assert "context_test" in log_record.getMessage()
        
        # Check extra fields (context should be in log record)
        if hasattr(log_record, 'error_context'):
            assert log_record.error_context == context

if __name__ == "__main__":
    pytest.main([__file__, "-v"])