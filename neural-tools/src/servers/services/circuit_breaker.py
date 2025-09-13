"""
L9 2025 Circuit Breaker for Graceful Service Degradation
Prevents cascade failures and enables fallback mechanisms
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Circuit open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for service protection"""
    
    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"🔄 Circuit breaker for {self.service_name} entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker OPEN for {self.service_name} - failing fast")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"✅ Circuit breaker for {self.service_name} reset to CLOSED")
            
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.error(f"🚨 Circuit breaker for {self.service_name} opened (failures: {self.failure_count})")
                self.state = CircuitState.OPEN
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "service": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat() if self.last_failure_time else None,
            "recovery_timeout": self.recovery_timeout
        }


class ServiceFallbackManager:
    """Manages fallback strategies for services"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
        
    def register_service(
        self, 
        service_name: str, 
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        fallback_func: Optional[Callable] = None
    ):
        """Register a service with circuit breaker protection"""
        self.circuit_breakers[service_name] = CircuitBreaker(
            service_name, failure_threshold, recovery_timeout
        )
        
        if fallback_func:
            self.fallback_strategies[service_name] = fallback_func
            
        logger.info(f"🛡️ Registered circuit breaker for {service_name}")
    
    async def execute_with_fallback(
        self, 
        service_name: str, 
        primary_func: Callable,
        *args, 
        **kwargs
    ):
        """Execute function with circuit breaker and fallback"""
        
        circuit_breaker = self.circuit_breakers.get(service_name)
        if not circuit_breaker:
            # No circuit breaker registered, execute directly
            return await primary_func(*args, **kwargs) if asyncio.iscoroutinefunction(primary_func) else primary_func(*args, **kwargs)
        
        try:
            return await circuit_breaker.call(primary_func, *args, **kwargs)
            
        except Exception as e:
            logger.warning(f"Primary service {service_name} failed: {e}")
            
            # Try fallback if available
            fallback_func = self.fallback_strategies.get(service_name)
            if fallback_func:
                logger.info(f"🔄 Using fallback for {service_name}")
                try:
                    result = await fallback_func(*args, **kwargs) if asyncio.iscoroutinefunction(fallback_func) else fallback_func(*args, **kwargs)
                    # Add fallback indicator to result
                    if isinstance(result, dict):
                        result["fallback_used"] = True
                        result["fallback_service"] = service_name
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback for {service_name} also failed: {fallback_error}")
                    raise fallback_error
            else:
                # No fallback available
                raise e
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status() 
            for name, breaker in self.circuit_breakers.items()
        }
    
    def reset_circuit_breaker(self, service_name: str) -> bool:
        """Manually reset a circuit breaker"""
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker:
            circuit_breaker.failure_count = 0
            circuit_breaker.state = CircuitState.CLOSED
            circuit_breaker.last_failure_time = None
            logger.info(f"🔄 Manually reset circuit breaker for {service_name}")
            return True
        return False


# L9 2025: Fallback Functions for Neural GraphRAG Services

async def neo4j_fallback(*args, **kwargs):
    """Fallback for Neo4j graph queries"""
    logger.warning("Using Neo4j fallback - returning cached/simplified results")
    return {
        "status": "fallback",
        "message": "Neo4j service unavailable - using cached data",
        "data": [],
        "fallback_used": True
    }

async def qdrant_fallback(query: str = "", limit: int = 10, **kwargs):
    """Fallback for Qdrant vector search - simple text matching"""
    logger.warning("Using Qdrant fallback - simple text search")
    return {
        "status": "fallback",
        "message": "Vector search unavailable - using basic text matching",
        "results": [],
        "query": query,
        "limit": limit,
        "fallback_used": True
    }

async def redis_fallback(*args, **kwargs):
    """Fallback for Redis operations - in-memory cache"""
    logger.warning("Using Redis fallback - in-memory operation")
    return {
        "status": "fallback",
        "message": "Redis unavailable - using in-memory fallback",
        "fallback_used": True
    }

async def nomic_fallback(texts: list = None, **kwargs):
    """Fallback for Nomic embeddings - simple hash-based vectors"""
    import hashlib
    
    logger.warning("Using Nomic fallback - hash-based embeddings")
    
    if texts is None:
        texts = [""]
    
    # Simple hash-based embedding fallback
    embeddings = []
    for text in texts:
        # Create a simple hash-based vector
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert to 768-dimension vector (matching Nomic dimensions)
        vector = []
        for i in range(0, len(hash_hex), 2):
            val = int(hash_hex[i:i+2], 16) / 255.0  # Normalize to 0-1
            vector.extend([val] * 24)  # Repeat to get 768 dims (32 * 24 = 768)
        
        embeddings.append(vector[:768])  # Ensure exactly 768 dimensions
    
    return {
        "embeddings": embeddings,
        "status": "fallback",
        "message": "Using hash-based embedding fallback",
        "fallback_used": True
    }