"""
L9 Circuit Breaker - Production-grade resilience for MCP integrations
Implements Netflix-style circuit breaker pattern for graceful degradation
"""

import time
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
import threading
import logging


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing fast
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 3      # Failures before opening
    recovery_timeout: int = 30      # Seconds before trying half-open
    success_threshold: int = 2      # Successes needed to close
    timeout_seconds: float = 2.0    # Call timeout


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class MCPTimeoutError(Exception):
    """Raised when MCP call times out"""
    pass


class CircuitBreaker:
    """Production-grade circuit breaker for MCP calls"""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger("l9_circuit_breaker")
        
        # Statistics
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'timeouts': 0,
            'circuit_opened_count': 0,
            'avg_response_time': 0.0
        }
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)"""
        with self.lock:
            return self._should_open_circuit()
    
    def _should_open_circuit(self) -> bool:
        """Internal logic to determine if circuit should be open"""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                return False
            return True
        return False
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            self.stats['total_calls'] += 1
            
            # Check if we should fail fast
            if self._should_open_circuit():
                self.logger.warning("Circuit breaker is OPEN, failing fast")
                raise CircuitBreakerError("Circuit breaker is open")
        
        # Execute the call
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            self._record_success(time.time() - start_time)
            return result
            
        except Exception as e:
            self._record_failure(time.time() - start_time, e)
            raise
    
    def _record_success(self, response_time: float):
        """Record successful call"""
        with self.lock:
            self.stats['successful_calls'] += 1
            self._update_avg_response_time(response_time)
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.logger.info("Circuit breaker CLOSED after recovery")
            else:
                self.failure_count = 0  # Reset failure count on success
    
    def _record_failure(self, response_time: float, error: Exception):
        """Record failed call"""
        with self.lock:
            self.stats['failed_calls'] += 1
            self._update_avg_response_time(response_time)
            
            if isinstance(error, (TimeoutError, MCPTimeoutError)):
                self.stats['timeouts'] += 1
            
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Check if we should open the circuit
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                
                self.state = CircuitState.OPEN
                self.stats['circuit_opened_count'] += 1
                self.logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self.state = CircuitState.OPEN
                self.logger.warning("Circuit breaker back to OPEN from HALF_OPEN")
    
    def _update_avg_response_time(self, response_time: float):
        """Update running average response time"""
        total_calls = self.stats['total_calls']
        current_avg = self.stats['avg_response_time']
        self.stats['avg_response_time'] = (
            (current_avg * (total_calls - 1) + response_time) / total_calls
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self.lock:
            return {
                **self.stats,
                'current_state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'success_rate': (
                    self.stats['successful_calls'] / self.stats['total_calls'] 
                    if self.stats['total_calls'] > 0 else 0
                )
            }
    
    def reset(self):
        """Reset circuit breaker (for testing)"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0
            self.stats = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'timeouts': 0,
                'circuit_opened_count': 0,
                'avg_response_time': 0.0
            }
            self.logger.info("Circuit breaker reset")


class TimeoutWrapper:
    """Wrapper for adding timeouts to function calls"""
    
    @staticmethod
    def with_timeout(func: Callable, timeout_seconds: float, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        import signal
        
        class TimeoutHandler:
            def __init__(self, seconds):
                self.seconds = seconds
                
            def __enter__(self):
                signal.signal(signal.SIGALRM, self._timeout_handler)
                signal.alarm(int(self.seconds))
                
            def __exit__(self, type, value, traceback):
                signal.alarm(0)
                
            def _timeout_handler(self, signum, frame):
                raise MCPTimeoutError(f"Function timed out after {self.seconds} seconds")
        
        try:
            with TimeoutHandler(timeout_seconds):
                return func(*args, **kwargs)
        except MCPTimeoutError:
            raise
        except Exception as e:
            # Convert other timeout-like errors
            if "timeout" in str(e).lower():
                raise MCPTimeoutError(f"Timeout during execution: {e}")
            raise