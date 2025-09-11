"""
L9 2025 Comprehensive Error Handling and Logging for MCP Server
Production-grade error handling with structured logging and alerting
"""

import os
import sys
import traceback
import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import redis.asyncio as redis

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATABASE = "database"
    NETWORK = "network"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"
    PERFORMANCE = "performance"
    SECURITY = "security"

class StructuredLogger:
    """Structured logging with context and correlation IDs"""
    
    def __init__(self, name: str, redis_client: Optional[redis.Redis] = None):
        self.logger = logging.getLogger(name)
        self.redis_client = redis_client
        self.service_name = "neural-mcp-server"
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
        # Configure structured logging
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure structured JSON logging"""
        handler = logging.StreamHandler(sys.stdout)
        
        # L9 2025: Structured JSON format for production
        if self.environment == 'production':
            formatter = StructuredJsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_with_context(
        self, 
        level: str, 
        message: str, 
        context: Dict[str, Any] = None,
        correlation_id: str = None,
        session_id: str = None,
        **kwargs
    ):
        """Log with structured context"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': self.service_name,
            'environment': self.environment,
            'level': level.upper(),
            'message': message,
            'correlation_id': correlation_id or str(uuid.uuid4()),
            'session_id': session_id,
            **kwargs
        }
        
        if context:
            log_data['context'] = context
        
        # Log to standard logger
        getattr(self.logger, level.lower())(json.dumps(log_data))
        
        # Store in Redis for centralized logging (if available)
        if self.redis_client:
            try:
                asyncio.create_task(self._store_log(log_data))
            except Exception:
                pass  # Don't fail on logging storage issues
    
    async def _store_log(self, log_data: Dict[str, Any]):
        """Store log entry in Redis for centralized collection"""
        try:
            log_key = f"logs:{log_data['level']}:{int(time.time())}"
            await self.redis_client.lpush('application_logs', json.dumps(log_data))
            await self.redis_client.ltrim('application_logs', 0, 9999)  # Keep last 10k logs
        except Exception as e:
            # Fallback to console if Redis fails
            print(f"Redis logging failed: {e}")

class StructuredJsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

class ErrorHandler:
    """Comprehensive error handling for MCP operations"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.logger = StructuredLogger(__name__, redis_client)
        self.redis_client = redis_client
        self.error_counts = {}
        self.alert_thresholds = {
            ErrorSeverity.LOW: 100,      # Alert after 100 low errors/hour
            ErrorSeverity.MEDIUM: 50,    # Alert after 50 medium errors/hour  
            ErrorSeverity.HIGH: 10,      # Alert after 10 high errors/hour
            ErrorSeverity.CRITICAL: 1    # Alert immediately on critical errors
        }
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        session_id: str = None,
        correlation_id: str = None,
        user_message: str = None
    ) -> Dict[str, Any]:
        """Handle error with comprehensive logging and alerting"""
        
        error_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Extract error details
        error_details = {
            'error_id': error_id,
            'timestamp': timestamp.isoformat(),
            'type': type(error).__name__,
            'message': str(error),
            'severity': severity.value,
            'category': category.value,
            'session_id': session_id,
            'correlation_id': correlation_id,
            'traceback': traceback.format_exc() if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None,
            'context': context or {}
        }
        
        # Log error with structured context
        self.logger.log_with_context(
            level='error' if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else 'warning',
            message=f"{category.value} error: {error}",
            context=error_details,
            correlation_id=correlation_id,
            session_id=session_id,
            error_id=error_id,
            severity=severity.value,
            category=category.value
        )
        
        # Store error for analysis
        await self._store_error(error_details)
        
        # Check alert thresholds
        await self._check_alert_thresholds(severity, category, error_details)
        
        # Return user-friendly error response
        return {
            'error': {
                'code': self._get_error_code(category, error),
                'message': user_message or self._get_user_message(category, error),
                'error_id': error_id,
                'timestamp': timestamp.isoformat(),
                'retry_after': self._get_retry_delay(category, error)
            }
        }
    
    async def _store_error(self, error_details: Dict[str, Any]):
        """Store error details for analysis and metrics"""
        if self.redis_client:
            try:
                # Store individual error
                error_key = f"error:{error_details['error_id']}"
                await self.redis_client.setex(
                    error_key, 
                    86400,  # 24 hour retention
                    json.dumps(error_details)
                )
                
                # Add to error stream for real-time monitoring
                await self.redis_client.xadd(
                    'error_stream',
                    error_details,
                    maxlen=10000  # Keep last 10k errors
                )
                
                # Update error counts
                hour_key = f"error_count:{datetime.utcnow().strftime('%Y%m%d%H')}"
                severity_key = f"{hour_key}:{error_details['severity']}"
                await self.redis_client.incr(severity_key)
                await self.redis_client.expire(severity_key, 3600)  # 1 hour TTL
                
            except Exception as e:
                self.logger.log_with_context('error', f"Failed to store error: {e}")
    
    async def _check_alert_thresholds(
        self, 
        severity: ErrorSeverity, 
        category: ErrorCategory, 
        error_details: Dict[str, Any]
    ):
        """Check if error count exceeds alert thresholds"""
        if not self.redis_client:
            return
        
        try:
            hour_key = f"error_count:{datetime.utcnow().strftime('%Y%m%d%H')}"
            severity_key = f"{hour_key}:{severity.value}"
            
            current_count = await self.redis_client.get(severity_key)
            current_count = int(current_count) if current_count else 0
            
            threshold = self.alert_thresholds.get(severity, 0)
            
            if current_count >= threshold:
                await self._send_alert(severity, category, current_count, threshold, error_details)
                
        except Exception as e:
            self.logger.log_with_context('error', f"Alert threshold check failed: {e}")
    
    async def _send_alert(
        self, 
        severity: ErrorSeverity, 
        category: ErrorCategory, 
        count: int, 
        threshold: int,
        error_details: Dict[str, Any]
    ):
        """Send alert for high error rates or critical errors"""
        alert_data = {
            'alert_id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'error_threshold_exceeded' if count > threshold else 'critical_error',
            'severity': severity.value,
            'category': category.value,
            'count': count,
            'threshold': threshold,
            'hour': datetime.utcnow().strftime('%Y-%m-%d %H:00'),
            'sample_error': error_details,
            'service': 'neural-mcp-server'
        }
        
        # Store alert
        if self.redis_client:
            try:
                await self.redis_client.lpush('alerts', json.dumps(alert_data))
                await self.redis_client.ltrim('alerts', 0, 999)  # Keep last 1000 alerts
                
                # Also add to high-priority alert stream for immediate notification
                if severity == ErrorSeverity.CRITICAL:
                    await self.redis_client.xadd('critical_alerts', alert_data, maxlen=1000)
                    
            except Exception as e:
                print(f"Alert storage failed: {e}")
        
        # Log alert
        self.logger.log_with_context(
            level='critical' if severity == ErrorSeverity.CRITICAL else 'error',
            message=f"Alert: {severity.value} {category.value} errors exceeded threshold",
            context=alert_data,
            alert_id=alert_data['alert_id']
        )
    
    def _get_error_code(self, category: ErrorCategory, error: Exception) -> str:
        """Get standardized error code"""
        base_codes = {
            ErrorCategory.AUTHENTICATION: 'AUTH_',
            ErrorCategory.AUTHORIZATION: 'AUTHZ_',
            ErrorCategory.VALIDATION: 'VALID_',
            ErrorCategory.DATABASE: 'DB_',
            ErrorCategory.NETWORK: 'NET_',
            ErrorCategory.EXTERNAL_SERVICE: 'EXT_',
            ErrorCategory.INTERNAL: 'INT_',
            ErrorCategory.PERFORMANCE: 'PERF_',
            ErrorCategory.SECURITY: 'SEC_'
        }
        
        error_type = type(error).__name__
        return f"{base_codes.get(category, 'ERR_')}{error_type.upper()}"
    
    def _get_user_message(self, category: ErrorCategory, error: Exception) -> str:
        """Get user-friendly error message"""
        friendly_messages = {
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to perform this action.",
            ErrorCategory.VALIDATION: "Invalid request. Please check your input.",
            ErrorCategory.DATABASE: "Database temporarily unavailable. Please try again later.",
            ErrorCategory.NETWORK: "Network connectivity issue. Please try again later.",
            ErrorCategory.EXTERNAL_SERVICE: "External service temporarily unavailable.",
            ErrorCategory.PERFORMANCE: "Request timed out. Please try again.",
            ErrorCategory.SECURITY: "Security check failed.",
            ErrorCategory.INTERNAL: "Internal server error. Please try again later."
        }
        
        return friendly_messages.get(category, "An unexpected error occurred.")
    
    def _get_retry_delay(self, category: ErrorCategory, error: Exception) -> Optional[int]:
        """Get suggested retry delay in seconds"""
        retry_delays = {
            ErrorCategory.DATABASE: 5,
            ErrorCategory.NETWORK: 3,
            ErrorCategory.EXTERNAL_SERVICE: 10,
            ErrorCategory.PERFORMANCE: 30,
            ErrorCategory.INTERNAL: 15
        }
        
        return retry_delays.get(category)
    
    async def get_error_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        if not self.redis_client:
            return {'error': 'Redis not available for stats'}
        
        try:
            stats = {
                'timeframe_hours': hours,
                'total_errors': 0,
                'by_severity': {},
                'by_category': {},
                'error_rate': 0,
                'recent_alerts': []
            }
            
            # Get error counts by severity
            current_time = datetime.utcnow()
            for i in range(hours):
                hour = (current_time - timedelta(hours=i)).strftime('%Y%m%d%H')
                for severity in ErrorSeverity:
                    key = f"error_count:{hour}:{severity.value}"
                    count = await self.redis_client.get(key)
                    if count:
                        stats['by_severity'][severity.value] = stats['by_severity'].get(severity.value, 0) + int(count)
                        stats['total_errors'] += int(count)
            
            # Get recent alerts
            alerts = await self.redis_client.lrange('alerts', 0, 9)
            stats['recent_alerts'] = [json.loads(alert) for alert in alerts]
            
            # Calculate error rate (errors per hour)
            stats['error_rate'] = round(stats['total_errors'] / hours, 2)
            
            return stats
            
        except Exception as e:
            return {'error': f'Failed to get error stats: {e}'}


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None

def get_error_handler(redis_client: Optional[redis.Redis] = None) -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(redis_client)
    return _error_handler

async def handle_mcp_error(
    error: Exception,
    context: Dict[str, Any] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.INTERNAL,
    session_id: str = None,
    correlation_id: str = None,
    user_message: str = None
) -> Dict[str, Any]:
    """Convenience function for handling MCP errors"""
    handler = get_error_handler()
    return await handler.handle_error(
        error=error,
        context=context,
        severity=severity,
        category=category,
        session_id=session_id,
        correlation_id=correlation_id,
        user_message=user_message
    )