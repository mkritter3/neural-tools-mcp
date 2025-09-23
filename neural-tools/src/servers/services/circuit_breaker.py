#!/usr/bin/env python3
"""
ADR-0084 Phase 3: Circuit Breaker Pattern for Service Resilience
Prevents cascade failures by opening circuit after repeated failures
"""

import time
import asyncio
import logging
from typing import Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failures exceeded threshold, rejecting calls
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker implementation for service resilience

    ADR-0084 Phase 3: Prevents cascade failures by:
    - Opening after failure_threshold failures within timeout period
    - Rejecting calls when OPEN to prevent overload
    - Testing recovery in HALF_OPEN state
    - Auto-recovering to CLOSED when service is healthy
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        timeout: int = 60,
        recovery_timeout: int = 30
    ):
        """
        Initialize circuit breaker

        Args:
            name: Circuit breaker name for logging
            failure_threshold: Number of failures to open circuit
            timeout: Time window for counting failures (seconds)
            recovery_timeout: Time to wait before attempting recovery (seconds)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout

        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.circuit_opened_time: Optional[float] = None

        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            RuntimeError: When circuit is OPEN
            Exception: Original exception from function
        """
        self.total_calls += 1

        # Check if circuit is OPEN
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.circuit_opened_time and \
               time.time() - self.circuit_opened_time > self.recovery_timeout:
                # Transition to HALF_OPEN for testing
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
                logger.info(f"🔄 Circuit breaker '{self.name}' transitioning to HALF_OPEN")
            else:
                # Still in timeout period, reject call
                self.rejected_calls += 1
                time_left = self.recovery_timeout - (time.time() - self.circuit_opened_time)
                raise RuntimeError(
                    f"Circuit breaker '{self.name}' is OPEN - "
                    f"service unavailable (recovery in {time_left:.0f}s)"
                )

        # Execute the function
        try:
            result = await func(*args, **kwargs)

            # Success - update state
            self.successful_calls += 1

            if self.state == CircuitState.HALF_OPEN:
                # Service recovered, close circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.last_failure_time = None
                logger.info(f"✅ Circuit breaker '{self.name}' CLOSED - service recovered")
            elif self.state == CircuitState.CLOSED:
                # Reset failure count if outside timeout window
                if self.last_failure_time and \
                   time.time() - self.last_failure_time > self.timeout:
                    self.failure_count = 0
                    self.last_failure_time = None

            return result

        except Exception as e:
            # Failure - update state
            self.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    self.circuit_opened_time = time.time()
                    logger.error(
                        f"🚨 Circuit breaker '{self.name}' OPENED after "
                        f"{self.failure_count} failures - {str(e)}"
                    )

            # Re-raise the original exception
            raise

    def get_state(self) -> dict:
        """Get current circuit breaker state and metrics"""
        success_rate = (
            self.successful_calls / self.total_calls * 100
            if self.total_calls > 0 else 0
        )

        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "success_rate": f"{success_rate:.1f}%",
            "last_failure_time": (
                datetime.fromtimestamp(self.last_failure_time).isoformat()
                if self.last_failure_time else None
            ),
            "circuit_opened_time": (
                datetime.fromtimestamp(self.circuit_opened_time).isoformat()
                if self.circuit_opened_time else None
            )
        }

    def reset(self):
        """Reset circuit breaker to initial state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_opened_time = None
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        logger.info(f"🔄 Circuit breaker '{self.name}' reset")


class CircuitBreakerManager:
    """Manage multiple circuit breakers for different services"""

    def __init__(self):
        self.breakers = {}

    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: int = 60,
        recovery_timeout: int = 30
    ) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                timeout=timeout,
                recovery_timeout=recovery_timeout
            )
        return self.breakers[name]

    def get_all_states(self) -> dict:
        """Get states of all circuit breakers"""
        return {
            name: breaker.get_state()
            for name, breaker in self.breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()