#!/usr/bin/env python3
"""
Resilience Infrastructure (Phase 2.2)

Provides production-grade resilience patterns:
- Rate limiting for API endpoints
- Circuit breakers for external dependencies  
- Retry logic with exponential backoff
- Timeout handling
- Graceful degradation

Key components:
- Rate limiting with slowapi
- Circuit breakers with pybreaker
- Retry decorators with tenacity
- Timeout wrappers with asyncio
"""

import asyncio
import logging
import functools
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    logging.warning("slowapi not available, rate limiting disabled")
    Limiter = None
    get_remote_address = None
    _rate_limit_exceeded_handler = None
    RATE_LIMITING_AVAILABLE = False

try:
    from pybreaker import CircuitBreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    logging.warning("pybreaker not available, circuit breakers disabled")
    CircuitBreaker = None
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    RETRY_AVAILABLE = True
except ImportError:
    logging.warning("tenacity not available, retry logic disabled")
    retry = lambda *args, **kwargs: lambda f: f
    stop_after_attempt = lambda x: None
    wait_exponential = lambda **kwargs: None
    retry_if_exception_type = lambda x: None
    RETRY_AVAILABLE = False

from infrastructure.telemetry import get_telemetry

logger = logging.getLogger(__name__)
telemetry = get_telemetry()


@dataclass
class ResilienceConfig:
    """Configuration for resilience patterns"""
    
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 100
    burst_limit: int = 20
    
    # Circuit breakers
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    
    # Retry logic
    retry_enabled: bool = True
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    
    # Timeouts
    default_timeout: float = 5.0
    api_timeout: float = 2.0
    db_timeout: float = 1.0


class ResilienceManager:
    """Centralized resilience pattern management"""
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.circuit_breakers: Dict[str, Any] = {}
        self.rate_limiters: Dict[str, Any] = {}
        self._setup_circuit_breakers()
        self._setup_rate_limiters()
    
    def _setup_circuit_breakers(self):
        """Initialize circuit breakers for external services"""
        if not CIRCUIT_BREAKER_AVAILABLE or not self.config.circuit_breaker_enabled:
            logger.info("Circuit breakers disabled or unavailable")
            return
            
        # Anthropic API circuit breaker
        self.circuit_breakers['anthropic'] = CircuitBreaker(
            fail_max=self.config.failure_threshold,
            reset_timeout=self.config.recovery_timeout,
            exclude=[asyncio.TimeoutError],  # Don't trip on timeouts
            name="anthropic_api"
        )
        
        # Neo4j circuit breaker
        self.circuit_breakers['neo4j'] = CircuitBreaker(
            fail_max=10,  # Higher threshold for DB
            reset_timeout=30,
            name="neo4j_db"
        )
        
        # Qdrant circuit breaker
        self.circuit_breakers['qdrant'] = CircuitBreaker(
            fail_max=10,  # Higher threshold for vector DB
            reset_timeout=30,
            name="qdrant_db"
        )
        
        logger.info(f"Initialized {len(self.circuit_breakers)} circuit breakers")
    
    def _setup_rate_limiters(self):
        """Initialize rate limiters"""
        if not RATE_LIMITING_AVAILABLE or not self.config.rate_limit_enabled:
            logger.info("Rate limiting disabled or unavailable")
            return
            
        # Global rate limiter
        self.rate_limiters['global'] = Limiter(
            key_func=get_remote_address if get_remote_address else lambda: "global",
            default_limits=[f"{self.config.requests_per_minute}/minute"]
        )
        
        # API-specific rate limiter
        self.rate_limiters['api'] = Limiter(
            key_func=get_remote_address if get_remote_address else lambda: "api",
            default_limits=[f"{self.config.burst_limit}/minute"]
        )
        
        logger.info(f"Initialized {len(self.rate_limiters)} rate limiters")
    
    def get_circuit_breaker(self, service_name: str) -> Optional[Any]:
        """Get circuit breaker for service"""
        return self.circuit_breakers.get(service_name)
    
    def get_rate_limiter(self, limiter_name: str = 'global') -> Optional[Any]:
        """Get rate limiter"""
        return self.rate_limiters.get(limiter_name)
    
    def with_circuit_breaker(self, service_name: str):
        """Decorator to add circuit breaker protection"""
        def decorator(func):
            if not CIRCUIT_BREAKER_AVAILABLE or service_name not in self.circuit_breakers:
                return func  # Pass through if not available
                
            breaker = self.circuit_breakers[service_name]
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    try:
                        with telemetry.trace_operation(f"circuit_breaker_{service_name}", {
                            "service": service_name,
                            "breaker_state": str(breaker.current_state)
                        }):
                            result = await breaker(func)(*args, **kwargs)
                            telemetry.record_cache_operation(f"circuit_breaker_{service_name}", True)
                            return result
                    except Exception as e:
                        telemetry.record_error(f"circuit_breaker_trip_{service_name}", "resilience")
                        telemetry.record_cache_operation(f"circuit_breaker_{service_name}", False)
                        logger.error(f"Circuit breaker {service_name} tripped: {e}")
                        raise
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    try:
                        with telemetry.trace_operation(f"circuit_breaker_{service_name}", {
                            "service": service_name,
                            "breaker_state": str(breaker.current_state)
                        }):
                            result = breaker(func)(*args, **kwargs)
                            telemetry.record_cache_operation(f"circuit_breaker_{service_name}", True)
                            return result
                    except Exception as e:
                        telemetry.record_error(f"circuit_breaker_trip_{service_name}", "resilience")
                        telemetry.record_cache_operation(f"circuit_breaker_{service_name}", False)
                        logger.error(f"Circuit breaker {service_name} tripped: {e}")
                        raise
                return sync_wrapper
        return decorator
    
    def with_retry(self, max_attempts: Optional[int] = None, 
                   base_delay: Optional[float] = None,
                   max_delay: Optional[float] = None):
        """Decorator to add retry logic with exponential backoff"""
        if not RETRY_AVAILABLE or not self.config.retry_enabled:
            return lambda f: f  # Pass through if not available
            
        attempts = max_attempts or self.config.max_attempts
        base = base_delay or self.config.base_delay  
        max_d = max_delay or self.config.max_delay
        
        return retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=base, min=base, max=max_d),
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying after {retry_state.outcome.exception()}, "
                f"attempt {retry_state.attempt_number}/{attempts}"
            )
        )
    
    def with_timeout(self, timeout: Optional[float] = None):
        """Decorator to add timeout protection"""
        timeout_seconds = timeout or self.config.default_timeout
        
        def decorator(func):
            if not asyncio.iscoroutinefunction(func):
                return func  # Only works with async functions
                
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs), 
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError as e:
                    telemetry.record_error("timeout", "resilience")
                    logger.warning(f"Function {func.__name__} timed out after {timeout_seconds}s")
                    raise TimeoutError(f"Operation timed out after {timeout_seconds}s") from e
            
            return wrapper
        return decorator


# Global resilience manager instance
_resilience_manager = None

def get_resilience_manager(config: Optional[ResilienceConfig] = None) -> ResilienceManager:
    """Get or create global resilience manager"""
    global _resilience_manager
    
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager(config)
    
    return _resilience_manager


def resilient_api_call(service_name: str, 
                      timeout: Optional[float] = None,
                      max_attempts: Optional[int] = None):
    """
    Composite decorator for resilient API calls
    
    Combines circuit breaker + retry + timeout protection
    """
    def decorator(func):
        rm = get_resilience_manager()
        
        # Apply decorators in order: timeout -> retry -> circuit_breaker
        protected_func = func
        protected_func = rm.with_timeout(timeout)(protected_func)
        protected_func = rm.with_retry(max_attempts=max_attempts)(protected_func)  
        protected_func = rm.with_circuit_breaker(service_name)(protected_func)
        
        return protected_func
    
    return decorator


class ResilientHaikuService:
    """Haiku service with comprehensive resilience patterns"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 config: Optional[ResilienceConfig] = None):
        self.api_key = api_key
        self.config = config or ResilienceConfig()
        self.resilience = get_resilience_manager(config)
    
    @resilient_api_call("anthropic", timeout=2.0, max_attempts=3)
    async def rerank_with_resilience(self, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Rerank with full resilience patterns:
        - Circuit breaker protection
        - Exponential backoff retry
        - Timeout handling  
        - Telemetry recording
        """
        with telemetry.trace_operation("resilient_haiku_rerank", {
            "query_length": len(query),
            "doc_count": len(docs),
            "service": "anthropic"
        }) as span:
            
            if not self.api_key:
                # Mock response for testing
                await asyncio.sleep(0.01)  # Simulate latency
                return {
                    "reranked_docs": docs[:5],  # Top 5
                    "confidence": 0.85,
                    "model": "mock-haiku", 
                    "processing_time": 0.01
                }
            
            try:
                # Real Anthropic API call would go here
                import httpx
                
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                }
                
                payload = {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 500,
                    "messages": [{
                        "role": "user",
                        "content": f"Rerank these documents for query: {query}\n\nDocuments:\n" + 
                                  "\n".join(f"{i+1}. {doc.get('content', str(doc))}" for i, doc in enumerate(docs))
                    }]
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    content = data["content"][0]["text"]
                    
                    # Parse ranking from response
                    reranked_docs = self._parse_ranking_response(content, docs)
                    
                    if span:
                        span.set_attribute("success", True)
                        span.set_attribute("response_length", len(content))
                    
                    return {
                        "reranked_docs": reranked_docs,
                        "confidence": 0.8,
                        "model": "claude-3-haiku",
                        "raw_response": content
                    }
                    
            except Exception as e:
                telemetry.record_error("haiku_rerank_failed", "resilient_haiku_service")
                logger.error(f"Resilient Haiku rerank failed: {e}", extra={
                    "query": query,
                    "doc_count": len(docs),
                    "error_type": type(e).__name__
                })
                
                if span:
                    span.set_attribute("success", False) 
                    span.set_attribute("error", str(e))
                
                raise
    
    def _parse_ranking_response(self, response: str, original_docs: List[Dict]) -> List[Dict]:
        """Parse Haiku ranking response and reorder documents"""
        # Simple parsing - look for numbered lists
        lines = response.split('\n')
        reranked = []
        
        for line in lines:
            if line.strip() and line.strip()[0].isdigit():
                try:
                    # Extract original index
                    idx = int(line.strip().split('.')[0]) - 1
                    if 0 <= idx < len(original_docs):
                        reranked.append(original_docs[idx])
                except (ValueError, IndexError):
                    continue
        
        # Fill in any missing docs
        added_indices = set()
        for doc in reranked:
            for i, orig in enumerate(original_docs):
                if orig is doc:
                    added_indices.add(i)
                    break
        
        for i, doc in enumerate(original_docs):
            if i not in added_indices and len(reranked) < len(original_docs):
                reranked.append(doc)
        
        return reranked[:len(original_docs)]  # Ensure same length
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resilience statistics"""
        stats = {
            "circuit_breakers": {},
            "config": {
                "rate_limit_enabled": self.config.rate_limit_enabled,
                "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
                "retry_enabled": self.config.retry_enabled,
                "max_attempts": self.config.max_attempts,
                "default_timeout": self.config.default_timeout
            }
        }
        
        # Add circuit breaker states
        for name, breaker in self.resilience.circuit_breakers.items():
            if hasattr(breaker, 'current_state'):
                stats["circuit_breakers"][name] = {
                    "state": str(breaker.current_state),
                    "fail_counter": getattr(breaker, 'fail_counter', 0),
                    "last_failure": getattr(breaker, 'last_failure', None)
                }
        
        return stats


# Convenience functions for common resilience patterns
async def resilient_db_query(func: Callable[..., Awaitable], 
                           service: str = "database",
                           *args, **kwargs):
    """Execute database query with resilience patterns"""
    rm = get_resilience_manager()
    
    @rm.with_timeout(rm.config.db_timeout)
    @rm.with_retry(max_attempts=2)  # Fewer retries for DB
    @rm.with_circuit_breaker(service)
    async def protected_query():
        return await func(*args, **kwargs)
    
    return await protected_query()


async def resilient_api_request(func: Callable[..., Awaitable],
                              service: str = "api", 
                              *args, **kwargs):
    """Execute API request with full resilience"""
    rm = get_resilience_manager()
    
    @rm.with_timeout(rm.config.api_timeout)
    @rm.with_retry()
    @rm.with_circuit_breaker(service)
    async def protected_request():
        return await func(*args, **kwargs)
    
    return await protected_request()


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create resilient Haiku service
        service = ResilientHaikuService()
        
        # Test reranking with resilience
        docs = [
            {"content": "Python async programming", "score": 0.8},
            {"content": "FastAPI web development", "score": 0.7},
            {"content": "Database optimization", "score": 0.6}
        ]
        
        try:
            result = await service.rerank_with_resilience(
                "web development frameworks", 
                docs
            )
            print("✓ Resilient reranking successful:")
            print(f"  Reranked {len(result['reranked_docs'])} documents")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Model: {result['model']}")
            
            # Show stats
            stats = service.get_stats()
            print("\n📊 Resilience Stats:")
            print(f"  Circuit breakers: {len(stats['circuit_breakers'])}")
            print(f"  Config: {stats['config']}")
            
        except Exception as e:
            print(f"❌ Resilient reranking failed: {e}")
    
    asyncio.run(main())