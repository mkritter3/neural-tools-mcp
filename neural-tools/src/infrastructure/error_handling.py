#!/usr/bin/env python3
"""
Enhanced Error Handling for Neural Tools
Provides comprehensive error handling, retry logic, and graceful degradation
Following roadmap Phase 1.5 specifications
"""

import asyncio
import logging
import functools
import traceback
from typing import Dict, Any, Optional, Callable, Type, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Non-critical, can continue
    MEDIUM = "medium"     # Important but not blocking
    HIGH = "high"         # Critical, affects functionality
    CRITICAL = "critical" # System-breaking, immediate attention

class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    DATABASE = "database"
    COMPUTATION = "computation"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    EXTERNAL_API = "external_api"

@dataclass
class ErrorInfo:
    """Structured error information"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    traceback: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'error_id': self.error_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }

class NeuralToolsException(Exception):
    """Base exception for neural tools"""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Dict[str, Any] = None,
        original_exception: Exception = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_exception = original_exception
        self.error_id = self._generate_error_id()
        self.timestamp = datetime.now()
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        import uuid
        return f"NT-{uuid.uuid4().hex[:8]}"
    
    def to_error_info(self) -> ErrorInfo:
        """Convert to ErrorInfo object"""
        return ErrorInfo(
            error_id=self.error_id,
            category=self.category,
            severity=self.severity,
            message=self.message,
            context=self.context,
            timestamp=self.timestamp,
            traceback=traceback.format_exc() if self.original_exception else None
        )

class VectorDatabaseException(NeuralToolsException):
    """Vector database related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class GraphDatabaseException(NeuralToolsException):
    """Graph database related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class EmbeddingException(NeuralToolsException):
    """Embedding generation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class ValidationException(NeuralToolsException):
    """Input validation errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )

class RateLimitException(NeuralToolsException):
    """Rate limiting errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class ErrorHandler:
    """Centralized error handling and recovery"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, datetime] = {}
        self.circuit_breaker_states: Dict[str, Dict[str, Any]] = {}
        self.retry_configs: Dict[ErrorCategory, Dict[str, Any]] = {
            ErrorCategory.NETWORK: {
                'max_retries': 3,
                'backoff_factor': 2,
                'base_delay': 1
            },
            ErrorCategory.DATABASE: {
                'max_retries': 2,
                'backoff_factor': 1.5,
                'base_delay': 0.5
            },
            ErrorCategory.EXTERNAL_API: {
                'max_retries': 5,
                'backoff_factor': 3,
                'base_delay': 2
            },
            ErrorCategory.COMPUTATION: {
                'max_retries': 1,
                'backoff_factor': 1,
                'base_delay': 0.1
            }
        }
    
    def get_retry_config(self, category: ErrorCategory) -> Dict[str, Any]:
        """Get retry configuration for error category"""
        return self.retry_configs.get(category, {
            'max_retries': 1,
            'backoff_factor': 1,
            'base_delay': 0.5
        })
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Dict[str, Any] = None,
        operation_name: str = "unknown"
    ) -> ErrorInfo:
        """Handle error with classification and logging"""
        
        # Convert to NeuralToolsException if needed
        if isinstance(error, NeuralToolsException):
            neural_error = error
        else:
            neural_error = self._classify_error(error, context)
        
        error_info = neural_error.to_error_info()
        
        # Track error frequency
        error_key = f"{operation_name}:{error_info.category.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = datetime.now()
        
        # Log error with appropriate level
        self._log_error(error_info, operation_name)
        
        # Check if circuit breaker should open
        await self._check_circuit_breaker(error_key, error_info)
        
        return error_info
    
    def _classify_error(self, error: Exception, context: Dict[str, Any] = None) -> NeuralToolsException:
        """Classify generic exception into NeuralToolsException"""
        error_msg = str(error)
        context = context or {}
        
        # Database errors (check first, more specific)
        if any(term in error_msg.lower() for term in ['database', 'sql', 'query', 'qdrant', 'neo4j']):
            return VectorDatabaseException(error_msg, context=context, original_exception=error)
        
        # Network-related errors
        if any(term in error_msg.lower() for term in ['connection', 'timeout', 'network', 'unreachable']):
            return NeuralToolsException(
                error_msg,
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                context=context,
                original_exception=error
            )
        
        
        # Validation errors
        if any(term in error_msg.lower() for term in ['invalid', 'validation', 'missing', 'required']):
            return ValidationException(error_msg, context=context, original_exception=error)
        
        # Default classification
        return NeuralToolsException(
            error_msg,
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_exception=error
        )
    
    def _log_error(self, error_info: ErrorInfo, operation_name: str):
        """Log error with appropriate level"""
        # Create safe log data that won't conflict with logging internals
        safe_log_data = {
            'error_id': error_info.error_id,
            'error_category': error_info.category.value,
            'error_severity': error_info.severity.value,
            'operation': operation_name,
            'error_context': error_info.context
        }
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR in {operation_name}: {error_info.message}", extra=safe_log_data)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(f"ERROR in {operation_name}: {error_info.message}", extra=safe_log_data)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"WARNING in {operation_name}: {error_info.message}", extra=safe_log_data)
        else:
            logger.info(f"INFO in {operation_name}: {error_info.message}", extra=safe_log_data)
    
    async def _check_circuit_breaker(self, error_key: str, error_info: ErrorInfo):
        """Check if circuit breaker should open"""
        error_count = self.error_counts.get(error_key, 0)
        last_error = self.last_errors.get(error_key)
        
        # Circuit breaker thresholds
        if error_count >= 5 and last_error:
            time_window = datetime.now() - last_error
            if time_window <= timedelta(minutes=5):
                # Open circuit breaker
                self.circuit_breaker_states[error_key] = {
                    'state': 'open',
                    'opened_at': datetime.now(),
                    'error_count': error_count
                }
                logger.warning(f"Circuit breaker OPENED for {error_key} after {error_count} errors")

def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2,
    base_delay: float = 1,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    error_handler: Optional[ErrorHandler] = None
):
    """Decorator for retry logic with exponential backoff"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if error_handler:
                        error_info = await error_handler.handle_error(
                            e, 
                            context={'attempt': attempt, 'max_retries': max_retries},
                            operation_name=func.__name__
                        )
                    
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.info(f"Retrying {func.__name__} in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.info(f"Retrying {func.__name__} in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class GracefulDegradation:
    """Handles graceful degradation when services fail"""
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
        self.service_health: Dict[str, bool] = {}
    
    def register_fallback(self, service_name: str, handler: Callable):
        """Register fallback handler for service"""
        self.fallback_handlers[service_name] = handler
        logger.info(f"Registered fallback handler for {service_name}")
    
    def mark_service_unhealthy(self, service_name: str):
        """Mark service as unhealthy"""
        self.service_health[service_name] = False
        logger.warning(f"Service {service_name} marked as unhealthy")
    
    def mark_service_healthy(self, service_name: str):
        """Mark service as healthy"""
        self.service_health[service_name] = True
        logger.info(f"Service {service_name} marked as healthy")
    
    async def execute_with_fallback(
        self,
        service_name: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute function with fallback if service is unhealthy"""
        
        # Check if service is healthy
        if self.service_health.get(service_name, True):
            try:
                if asyncio.iscoroutinefunction(primary_func):
                    result = await primary_func(*args, **kwargs)
                else:
                    result = primary_func(*args, **kwargs)
                
                return {
                    'success': True,
                    'result': result,
                    'fallback_used': False,
                    'service': service_name
                }
                
            except Exception as e:
                logger.error(f"Primary service {service_name} failed: {e}")
                self.mark_service_unhealthy(service_name)
                # Fall through to fallback logic
        
        # Use fallback if available
        fallback_handler = self.fallback_handlers.get(service_name)
        if fallback_handler:
            try:
                logger.info(f"Using fallback handler for {service_name}")
                
                if asyncio.iscoroutinefunction(fallback_handler):
                    fallback_result = await fallback_handler(*args, **kwargs)
                else:
                    fallback_result = fallback_handler(*args, **kwargs)
                
                return {
                    'success': True,
                    'result': fallback_result,
                    'fallback_used': True,
                    'service': service_name,
                    'message': f'Used fallback for {service_name}'
                }
                
            except Exception as e:
                logger.error(f"Fallback handler for {service_name} also failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'fallback_used': True,
                    'fallback_failed': True,
                    'service': service_name
                }
        
        # No fallback available
        return {
            'success': False,
            'error': f'Service {service_name} unavailable and no fallback configured',
            'fallback_used': False,
            'service': service_name
        }

# Global instances
error_handler = ErrorHandler()
graceful_degradation = GracefulDegradation()

# Convenience decorators with pre-configured settings
def database_retry(func):
    """Decorator for database operations with retry logic"""
    return retry_with_backoff(
        max_retries=2,
        backoff_factor=1.5,
        base_delay=0.5,
        exceptions=(VectorDatabaseException, GraphDatabaseException, Exception),
        error_handler=error_handler
    )(func)

def embedding_retry(func):
    """Decorator for embedding operations with retry logic"""
    return retry_with_backoff(
        max_retries=3,
        backoff_factor=2,
        base_delay=1,
        exceptions=(EmbeddingException, Exception),
        error_handler=error_handler
    )(func)

def network_retry(func):
    """Decorator for network operations with retry logic"""
    return retry_with_backoff(
        max_retries=5,
        backoff_factor=3,
        base_delay=2,
        exceptions=(ConnectionError, TimeoutError, Exception),
        error_handler=error_handler
    )(func)

# Example usage and integration
async def demo_error_handling():
    """Demonstrate error handling capabilities"""
    print("🛡️ Testing error handling system...")
    
    # Test retry decorator
    @database_retry
    async def flaky_database_operation():
        import random
        if random.random() < 0.7:  # 70% failure rate
            raise VectorDatabaseException("Database connection failed")
        return {"status": "success", "data": "mock_data"}
    
    # Test graceful degradation
    async def primary_search(query: str):
        raise Exception("Search service down")
    
    async def fallback_search(query: str):
        return {"results": [], "message": "Fallback search - limited results"}
    
    graceful_degradation.register_fallback("search", fallback_search)
    
    try:
        # Test retry logic
        print("   🔄 Testing retry logic...")
        result = await flaky_database_operation()
        print(f"   ✅ Database operation succeeded: {result}")
    except Exception as e:
        print(f"   ❌ Database operation failed after retries: {e}")
    
    # Test graceful degradation
    print("   🔄 Testing graceful degradation...")
    degradation_result = await graceful_degradation.execute_with_fallback(
        "search", primary_search, "test query"
    )
    print(f"   📊 Degradation result: {degradation_result}")
    
    # Test error classification
    print("   🔄 Testing error classification...")
    try:
        raise ConnectionError("Network unreachable")
    except Exception as e:
        error_info = await error_handler.handle_error(e, operation_name="test_operation")
        print(f"   📋 Error classified as: {error_info.category.value} / {error_info.severity.value}")
    
    print("✅ Error handling test completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_error_handling())