"""
Circuit Breaker for Service Connections (ADR 0018 Phase 4)
Prevents connection storms and provides graceful degradation
"""

import time
import asyncio
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Raised when circuit is open and requests are blocked"""
    pass


class ServiceConnectionBreaker:
    """
    Circuit breaker for service connections with exponential backoff
    
    Prevents repeated connection attempts to failing services,
    reducing load and allowing time for recovery.
    """
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        half_open_requests: int = 1
    ):
        """
        Initialize circuit breaker
        
        Args:
            service_name: Name of the service (for logging)
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            half_open_requests: Number of test requests in half-open state
        """
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_attempts = 0
        
        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.circuit_opens = 0
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_requests:
                # Service recovered, close circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.half_open_attempts = 0
                logger.info(f"Circuit breaker for {self.service_name} CLOSED - service recovered")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                # Trip the circuit
                self.state = CircuitState.OPEN
                self.circuit_opens += 1
                logger.warning(
                    f"Circuit breaker for {self.service_name} OPEN - "
                    f"{self.failure_count} failures exceeded threshold"
                )
        elif self.state == CircuitState.HALF_OPEN:
            # Service still failing, reopen circuit
            self.state = CircuitState.OPEN
            self.failure_count = 0
            self.success_count = 0
            logger.warning(f"Circuit breaker for {self.service_name} reopened - test failed")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Args:
            func: Async function to call
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result if successful
            
        Raises:
            CircuitOpenError: If circuit is open
            Exception: If function fails
        """
        self.total_calls += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                # Try half-open state
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                logger.info(f"Circuit breaker for {self.service_name} HALF-OPEN - testing recovery")
            else:
                # Circuit still open, fail fast
                retry_in = self.recovery_timeout - (time.time() - self.last_failure_time)
                raise CircuitOpenError(
                    f"Circuit breaker for {self.service_name} is OPEN. "
                    f"Retry in {retry_in:.0f} seconds"
                )
        
        # Attempt the call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
            
        except Exception:
            self._on_failure()
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state and metrics"""
        return {
            "service": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "circuit_opens": self.circuit_opens,
            "success_rate": (
                self.total_successes / self.total_calls 
                if self.total_calls > 0 else 0
            ),
            "last_failure": (
                datetime.fromtimestamp(self.last_failure_time).isoformat()
                if self.last_failure_time else None
            )
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_attempts = 0
        logger.info(f"Circuit breaker for {self.service_name} manually reset")


class ServiceCircuitBreakerManager:
    """Manages circuit breakers for multiple services"""
    
    def __init__(self):
        self.breakers: Dict[str, ServiceConnectionBreaker] = {}
    
    def get_breaker(
        self,
        service_name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 60
    ) -> ServiceConnectionBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.breakers:
            self.breakers[service_name] = ServiceConnectionBreaker(
                service_name,
                failure_threshold,
                recovery_timeout
            )
        return self.breakers[service_name]
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state of all circuit breakers"""
        return {
            name: breaker.get_state()
            for name, breaker in self.breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()