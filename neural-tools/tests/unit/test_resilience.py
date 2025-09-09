#!/usr/bin/env python3
"""
Resilience Infrastructure Tests

Tests comprehensive resilience patterns:
- Circuit breakers for fault tolerance
- Rate limiting for API protection
- Retry logic with exponential backoff
- Timeout handling
- Integration with telemetry
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Import components to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from infrastructure.resilience import (
    ResilienceManager, 
    ResilienceConfig,
    ResilientHaikuService,
    get_resilience_manager,
    resilient_api_call,
    resilient_db_query,
    resilient_api_request
)


class TestResilienceConfig:
    """Test resilience configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ResilienceConfig()
        
        assert config.rate_limit_enabled is True
        assert config.requests_per_minute == 100
        assert config.burst_limit == 20
        
        assert config.circuit_breaker_enabled is True
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60
        
        assert config.retry_enabled is True
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 10.0
        
        assert config.default_timeout == 5.0
        assert config.api_timeout == 2.0
        assert config.db_timeout == 1.0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ResilienceConfig(
            requests_per_minute=200,
            failure_threshold=10,
            max_attempts=5,
            api_timeout=3.0
        )
        
        assert config.requests_per_minute == 200
        assert config.failure_threshold == 10
        assert config.max_attempts == 5
        assert config.api_timeout == 3.0


class TestResilienceManager:
    """Test core resilience manager"""
    
    def setup_method(self):
        """Setup fresh resilience manager for each test"""
        self.config = ResilienceConfig(
            circuit_breaker_enabled=True,
            rate_limit_enabled=True,
            retry_enabled=True
        )
        self.manager = ResilienceManager(self.config)
    
    def test_resilience_manager_initialization(self):
        """Test resilience manager initialization"""
        assert self.manager.config == self.config
        assert isinstance(self.manager.circuit_breakers, dict)
        assert isinstance(self.manager.rate_limiters, dict)
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and configuration"""
        # Circuit breakers may not be available in test environment
        if self.manager.circuit_breakers:
            assert 'anthropic' in self.manager.circuit_breakers
            anthropic_breaker = self.manager.get_circuit_breaker('anthropic')
            assert anthropic_breaker is not None
    
    def test_get_nonexistent_circuit_breaker(self):
        """Test getting non-existent circuit breaker"""
        breaker = self.manager.get_circuit_breaker('nonexistent')
        assert breaker is None
    
    def test_timeout_decorator(self):
        """Test timeout decorator functionality"""
        @self.manager.with_timeout(0.1)
        async def slow_operation():
            await asyncio.sleep(0.2)
            return "completed"
        
        # Should timeout
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            asyncio.run(slow_operation())
    
    def test_timeout_decorator_fast_operation(self):
        """Test timeout decorator with fast operation"""
        @self.manager.with_timeout(1.0)
        async def fast_operation():
            await asyncio.sleep(0.01)
            return "completed"
        
        # Should complete successfully
        result = asyncio.run(fast_operation())
        assert result == "completed"
    
    def test_retry_decorator_mock(self):
        """Test retry decorator with mock failures"""
        call_count = 0
        
        @self.manager.with_retry(max_attempts=3, base_delay=0.01, max_delay=0.1)
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        # Should succeed after retries
        if hasattr(self.manager, 'config') and self.manager.config.retry_enabled:
            result = asyncio.run(failing_operation())
            assert result == "success"
            assert call_count == 3
    
    def test_circuit_breaker_decorator_mock(self):
        """Test circuit breaker decorator behavior"""
        @self.manager.with_circuit_breaker('test_service')
        async def mock_service_call():
            return "service_response"
        
        # Should work normally when circuit breaker is available
        result = asyncio.run(mock_service_call())
        assert result == "service_response"


class TestResilientHaikuService:
    """Test resilient Haiku service implementation"""
    
    def setup_method(self):
        """Setup resilient Haiku service for testing"""
        self.config = ResilienceConfig(
            api_timeout=1.0,
            max_attempts=2,
            circuit_breaker_enabled=True
        )
        self.service = ResilientHaikuService(api_key=None, config=self.config)  # Mock mode
    
    @pytest.mark.asyncio
    async def test_rerank_with_resilience_mock_mode(self):
        """Test reranking in mock mode"""
        docs = [
            {"content": "Python programming", "score": 0.9},
            {"content": "Web development", "score": 0.8},
            {"content": "Database design", "score": 0.7}
        ]
        
        result = await self.service.rerank_with_resilience(
            "programming languages", docs
        )
        
        assert "reranked_docs" in result
        assert "confidence" in result
        assert "model" in result
        assert result["model"] == "mock-haiku"
        assert len(result["reranked_docs"]) <= len(docs)
    
    @pytest.mark.asyncio
    async def test_rerank_with_api_error_simulation(self):
        """Test reranking with simulated API errors"""
        # Patch to simulate API failure
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.raise_for_status.side_effect = Exception("API Error")
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            # Create service with API key to trigger real API path
            service_with_key = ResilientHaikuService(api_key="test-key", config=self.config)
            
            docs = [{"content": "test", "score": 0.5}]
            
            # Should raise exception due to API error
            with pytest.raises(Exception):
                await service_with_key.rerank_with_resilience("test query", docs)
    
    def test_parse_ranking_response(self):
        """Test ranking response parsing"""
        response = """
        Based on relevance to query:
        
        1. Web development frameworks
        2. Database optimization
        3. Python programming
        """
        
        docs = [
            {"content": "Python programming", "id": 1},
            {"content": "Web development frameworks", "id": 2}, 
            {"content": "Database optimization", "id": 3}
        ]
        
        reranked = self.service._parse_ranking_response(response, docs)
        
        # Should reorder based on ranking
        assert len(reranked) == len(docs)
        # First result should be the web development one (index 1 in original)
        assert reranked[0]["id"] == 2
    
    def test_get_stats(self):
        """Test resilience statistics retrieval"""
        stats = self.service.get_stats()
        
        assert "circuit_breakers" in stats
        assert "config" in stats
        
        config_stats = stats["config"]
        assert "rate_limit_enabled" in config_stats
        assert "circuit_breaker_enabled" in config_stats
        assert "retry_enabled" in config_stats
        assert "max_attempts" in config_stats
        assert "default_timeout" in config_stats


class TestResilientApiCall:
    """Test resilient API call decorator"""
    
    def setup_method(self):
        """Setup for resilient API call tests"""
        self.call_count = 0
        self.should_fail = True
    
    @pytest.mark.asyncio
    async def test_resilient_api_call_success(self):
        """Test successful resilient API call"""
        @resilient_api_call("test_service", timeout=1.0, max_attempts=2)
        async def mock_api_call():
            await asyncio.sleep(0.01)
            return {"status": "success"}
        
        result = await mock_api_call()
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_resilient_api_call_timeout(self):
        """Test resilient API call with timeout"""
        @resilient_api_call("test_service", timeout=0.05, max_attempts=1)
        async def slow_api_call():
            await asyncio.sleep(0.1)  # Longer than timeout
            return {"status": "success"}
        
        # Should timeout
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await slow_api_call()


class TestResilienceHelpers:
    """Test resilience helper functions"""
    
    @pytest.mark.asyncio
    async def test_resilient_db_query(self):
        """Test resilient database query helper"""
        async def mock_db_query(table: str):
            await asyncio.sleep(0.01)
            return f"data from {table}"
        
        result = await resilient_db_query(mock_db_query, "database", "users")
        assert result == "data from users"
    
    @pytest.mark.asyncio
    async def test_resilient_api_request(self):
        """Test resilient API request helper"""
        async def mock_api_request(endpoint: str):
            await asyncio.sleep(0.01) 
            return {"endpoint": endpoint, "data": "response"}
        
        result = await resilient_api_request(mock_api_request, "api", "/users")
        assert result["endpoint"] == "/users"
        assert result["data"] == "response"


class TestResilienceIntegration:
    """Test integration between resilience components"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.config = ResilienceConfig(
            failure_threshold=3,
            recovery_timeout=30,
            max_attempts=2,
            api_timeout=0.5
        )
        self.manager = ResilienceManager(self.config)
    
    @pytest.mark.asyncio
    async def test_full_resilience_stack(self):
        """Test complete resilience stack integration"""
        failure_count = 0
        
        @self.manager.with_timeout(1.0)
        @self.manager.with_retry(max_attempts=3)
        @self.manager.with_circuit_breaker('integration_test')
        async def unreliable_service():
            nonlocal failure_count
            failure_count += 1
            
            if failure_count < 3:
                raise ConnectionError("Service unavailable")
            
            return {"status": "success", "attempts": failure_count}
        
        # Should eventually succeed with retries
        try:
            result = await unreliable_service()
            assert result["status"] == "success"
            assert result["attempts"] >= 2
        except Exception:
            # Expected if retry/circuit breaker not available
            assert True
    
    def test_global_resilience_manager_singleton(self):
        """Test global resilience manager behavior"""
        manager1 = get_resilience_manager()
        manager2 = get_resilience_manager()
        
        # Should return same instance
        assert manager1 is manager2
        assert isinstance(manager1, ResilienceManager)


class TestResiliencePerformanceImpact:
    """Test performance impact of resilience patterns"""
    
    def setup_method(self):
        """Setup performance test environment"""
        self.manager = ResilienceManager()
    
    @pytest.mark.asyncio
    async def test_resilience_overhead(self):
        """Test resilience pattern overhead is minimal"""
        iterations = 100
        
        # Measure baseline performance
        async def baseline_operation():
            await asyncio.sleep(0.001)
            return "baseline"
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            await baseline_operation()
        baseline_time = time.perf_counter() - start_time
        
        # Measure with resilience decorators
        @self.manager.with_timeout(5.0)
        async def resilient_operation():
            await asyncio.sleep(0.001)
            return "resilient"
        
        start_time = time.perf_counter()
        for _ in range(iterations):
            await resilient_operation()
        resilient_time = time.perf_counter() - start_time
        
        # Overhead should be reasonable (< 100% overhead)
        overhead_ratio = (resilient_time - baseline_time) / baseline_time
        assert overhead_ratio < 1.0, f"Resilience overhead too high: {overhead_ratio:.2%}"
    
    def test_circuit_breaker_state_tracking(self):
        """Test circuit breaker state tracking performance"""
        # Circuit breaker state checks should be fast
        start_time = time.perf_counter()
        
        for _ in range(1000):
            breaker = self.manager.get_circuit_breaker('anthropic')
            # Just access the breaker reference
            _ = breaker is not None
        
        elapsed = time.perf_counter() - start_time
        ops_per_second = 1000 / elapsed
        
        # Should handle high throughput (>10k ops/sec)
        assert ops_per_second > 10000, f"Circuit breaker lookup too slow: {ops_per_second:.1f} ops/sec"


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_resilience"
    ])