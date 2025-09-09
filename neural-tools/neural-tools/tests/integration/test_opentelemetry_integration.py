#!/usr/bin/env python3
"""
OpenTelemetry Integration Tests

Tests comprehensive tracing and metrics collection across all neural components:
- TelemetryManager initialization and configuration
- EnhancedHybridRetriever tracing spans and metrics
- HaikuReRanker telemetry integration
- Cross-component trace correlation
- Metrics collection validation
- OTLP export verification
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import json
import tempfile

# Import components to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Mock the auto-instrumentation imports that may be problematic
sys.modules['opentelemetry.instrumentation.requests'] = Mock()
sys.modules['opentelemetry.instrumentation.urllib3'] = Mock()
sys.modules['opentelemetry.instrumentation.redis'] = Mock()

from infrastructure.telemetry import TelemetryManager, get_telemetry, setup_telemetry
from infrastructure.haiku_reranker import HaikuReRanker, SearchResult, ReRankingRequest


class TestTelemetryManager:
    """Test core telemetry infrastructure"""
    
    def setup_method(self):
        """Setup fresh telemetry for each test"""
        self.telemetry = TelemetryManager()
    
    def test_telemetry_initialization_default(self):
        """Test default telemetry initialization"""
        self.telemetry.initialize()
        
        assert self.telemetry._initialized is True
        assert self.telemetry.tracer is not None
        assert self.telemetry.meter is not None
        assert self.telemetry.query_counter is not None
        assert self.telemetry.query_duration is not None
        
    def test_telemetry_initialization_with_otlp(self):
        """Test telemetry initialization with OTLP endpoint"""
        # Mock OTLP endpoint
        self.telemetry.initialize(
            otlp_endpoint="http://localhost:4317",
            enable_console_export=True
        )
        
        assert self.telemetry._initialized is True
        assert self.telemetry.tracer is not None
        assert self.telemetry.meter is not None
    
    def test_telemetry_trace_operation(self):
        """Test trace operation context manager"""
        self.telemetry.initialize()
        
        with self.telemetry.trace_operation("test_operation", {"key": "value"}) as span:
            assert span is not None
            # Span should be active during context
            time.sleep(0.001)
    
    def test_telemetry_metrics_recording(self):
        """Test metrics recording functionality"""
        self.telemetry.initialize()
        
        # Test query metrics
        self.telemetry.record_query("test query", "test_project", 0.1, True)
        self.telemetry.record_query("failed query", "test_project", 0.2, False)
        
        # Test retrieval metrics
        self.telemetry.record_retrieval("vector", 10, 0.05)
        self.telemetry.record_retrieval("graph", 5, 0.03)
        
        # Test reranking metrics
        self.telemetry.record_rerank("haiku", 15, 0.08)
        self.telemetry.record_rerank("local", 20, 0.02)
        
        # Test cache metrics
        self.telemetry.record_cache_operation("vector_search", True)
        self.telemetry.record_cache_operation("vector_search", False)
        
        # Test error metrics
        self.telemetry.record_error("timeout", "api_client")
        self.telemetry.record_error("parse_error", "response_parser")
        
        # Should not raise exceptions
        assert True
    
    def test_telemetry_initialization_failure_fallback(self):
        """Test graceful fallback when telemetry initialization fails"""
        with patch('infrastructure.telemetry.TracerProvider') as mock_provider:
            mock_provider.side_effect = Exception("Telemetry init failed")
            
            self.telemetry.initialize()
            
            # Should fall back to no-op telemetry
            assert self.telemetry._initialized is True
            assert self.telemetry.tracer is not None  # No-op tracer
    
    def test_global_telemetry_singleton(self):
        """Test global telemetry instance behavior"""
        tel1 = get_telemetry()
        tel2 = get_telemetry()
        
        assert tel1 is tel2  # Same instance
        assert isinstance(tel1, TelemetryManager)


class TestHaikuRerankerTelemetry:
    """Test telemetry integration in HaikuReRanker"""
    
    def setup_method(self):
        """Setup fresh components for each test"""
        self.telemetry = TelemetryManager()
        self.telemetry.initialize()
        self.reranker = HaikuReRanker(api_key=None)  # Mock mode
    
    @pytest.mark.asyncio
    async def test_rerank_operation_tracing(self):
        """Test comprehensive tracing of rerank operation"""
        # Create test request
        results = [
            SearchResult(
                content=f"Test content {i}",
                score=0.8 - (i * 0.1),
                metadata={"file": f"test{i}.py"},
                source="vector",
                original_rank=i
            ) for i in range(5)
        ]
        
        request = ReRankingRequest(
            query="test query",
            results=results,
            context="test context",
            max_results=3
        )
        
        # Mock telemetry tracing
        with patch.object(self.telemetry, 'trace_operation') as mock_trace:
            mock_trace.return_value.__enter__ = Mock(return_value=Mock())
            mock_trace.return_value.__exit__ = Mock(return_value=None)
            
            result = await self.reranker.rerank(request)
            
            # Verify main operation was traced
            assert mock_trace.called
            main_call = mock_trace.call_args_list[0]
            assert main_call[0][0] == "haiku_rerank"
            
            # Verify attributes were set
            attributes = main_call[0][1]
            assert attributes["query_length"] == len(request.query)
            assert attributes["result_count"] == len(results)
            assert attributes["max_results"] == 3
            assert attributes["has_context"] is True
            assert attributes["mock_mode"] is True
    
    @pytest.mark.asyncio 
    async def test_cache_operation_telemetry(self):
        """Test cache hit/miss telemetry recording"""
        results = [
            SearchResult(
                content="Cached test content",
                score=0.9,
                metadata={"file": "cached.py"},
                source="vector"
            )
        ]
        
        request = ReRankingRequest(
            query="cached query",
            results=results
        )
        
        with patch.object(self.telemetry, 'record_cache_operation') as mock_record:
            # First call should be cache miss
            result1 = await self.reranker.rerank(request)
            
            # Second call should be cache hit
            result2 = await self.reranker.rerank(request)
            
            # Verify cache operations were recorded
            assert mock_record.called
            calls = mock_record.call_args_list
            
            # Should have miss then hit
            assert any("haiku_rerank" in str(call) for call in calls)
    
    @pytest.mark.asyncio
    async def test_error_handling_telemetry(self):
        """Test error telemetry recording"""
        with patch.object(self.reranker, '_call_haiku_api') as mock_api:
            mock_api.side_effect = Exception("API failure")
            
            with patch.object(self.telemetry, 'record_error') as mock_error:
                request = ReRankingRequest(
                    query="error query",
                    results=[SearchResult("test", 0.5, {}, "vector")]
                )
                
                result = await self.reranker.rerank(request)
                
                # Should record the error
                mock_error.assert_called_with("reranking_failed", "haiku_reranker")
                
                # Should return fallback result
                assert result.model_used == "fallback"
                assert result.confidence_score == 0.3
    
    @pytest.mark.asyncio
    async def test_rerank_simple_tracing(self):
        """Test tracing for simplified rerank interface"""
        results = [
            {"content": "Simple test 1", "score": 0.8},
            {"content": "Simple test 2", "score": 0.7}
        ]
        
        with patch.object(self.telemetry, 'trace_operation') as mock_trace:
            mock_trace.return_value.__enter__ = Mock(return_value=Mock())
            mock_trace.return_value.__exit__ = Mock(return_value=None)
            
            reranked = await self.reranker.rerank_simple(
                "simple query", results, max_results=2
            )
            
            # Verify both simple and main operations were traced
            assert mock_trace.called
            calls = [call[0][0] for call in mock_trace.call_args_list]
            assert "haiku_rerank_simple" in calls
            assert "haiku_rerank" in calls


@pytest.mark.asyncio
class TestCrossComponentTelemetry:
    """Test telemetry correlation across multiple components"""
    
    def setup_method(self):
        """Setup integrated components"""
        self.telemetry = TelemetryManager()
        self.telemetry.initialize()
    
    async def test_end_to_end_tracing_simulation(self):
        """Simulate end-to-end request tracing across components"""
        query = "test cross-component query"
        
        # Simulate hybrid retrieval pipeline with telemetry
        with self.telemetry.trace_operation("e2e_retrieval_pipeline", {
            "query": query,
            "components": ["vector", "graph", "rerank"]
        }) as main_span:
            
            # Step 1: Initial retrieval (simulated)
            with self.telemetry.trace_operation("initial_retrieval", {
                "source": "vector",
                "limit": 20
            }):
                await asyncio.sleep(0.001)  # Simulate work
                self.telemetry.record_retrieval("vector", 15, 0.05)
            
            # Step 2: Graph context (simulated) 
            with self.telemetry.trace_operation("graph_context", {
                "max_hops": 2
            }):
                await asyncio.sleep(0.001)  # Simulate work
                self.telemetry.record_retrieval("graph", 8, 0.03)
            
            # Step 3: Reranking
            reranker = HaikuReRanker(api_key=None)  # Mock mode
            results = [
                SearchResult(f"Result {i}", 0.8 - i*0.1, {}, "vector") 
                for i in range(10)
            ]
            request = ReRankingRequest(query, results, max_results=5)
            
            rerank_result = await reranker.rerank(request)
            
            # Record overall metrics
            total_time = 0.09  # Simulated
            self.telemetry.record_query(query, "test_project", total_time, True)
            
            if main_span:
                main_span.set_attribute("success", True)
                main_span.set_attribute("total_results", len(rerank_result.results))
                main_span.set_attribute("total_time_ms", total_time * 1000)


class TestTelemetryExportIntegration:
    """Test telemetry export and configuration"""
    
    def test_console_exporter_integration(self):
        """Test console exporter configuration"""
        telemetry = TelemetryManager()
        telemetry.initialize(enable_console_export=True)
        
        assert telemetry._initialized is True
        
        # Test basic operation with console export
        with telemetry.trace_operation("console_test", {"test": True}):
            telemetry.record_query("console query", "test", 0.1, True)
    
    def test_otlp_exporter_configuration(self):
        """Test OTLP exporter configuration"""
        telemetry = TelemetryManager()
        
        # Test with mock OTLP endpoint
        telemetry.initialize(
            otlp_endpoint="http://localhost:4317",
            enable_console_export=False
        )
        
        assert telemetry._initialized is True
        
        # Test operation with OTLP export
        with telemetry.trace_operation("otlp_test"):
            telemetry.record_error("test_error", "test_component")
    
    def test_environment_based_setup(self):
        """Test environment-based telemetry setup"""
        import os
        
        # Test with environment variables
        with patch.dict(os.environ, {
            'OTLP_ENDPOINT': 'http://test-collector:4317',
            'ENVIRONMENT': 'production'
        }):
            telemetry = setup_telemetry()
            
            assert telemetry._initialized is True
            assert isinstance(telemetry, TelemetryManager)


class TestTelemetryMetricsValidation:
    """Test telemetry metrics collection and validation"""
    
    def setup_method(self):
        """Setup telemetry with metrics collection"""
        self.telemetry = TelemetryManager()
        self.telemetry.initialize()
    
    def test_query_metrics_collection(self):
        """Test query duration and counter metrics"""
        # Record various query metrics
        queries = [
            ("fast query", 0.05, True),
            ("medium query", 0.15, True),
            ("slow query", 0.35, True),
            ("failed query", 0.10, False),
        ]
        
        for query, duration, success in queries:
            self.telemetry.record_query(query, "test_project", duration, success)
        
        # Metrics should be recorded without errors
        assert True
    
    def test_retrieval_metrics_collection(self):
        """Test retrieval operation metrics"""
        retrievals = [
            ("vector", 10, 0.05),
            ("graph", 5, 0.03),
            ("hybrid", 15, 0.08),
            ("vector", 8, 0.04),
        ]
        
        for source, count, duration in retrievals:
            self.telemetry.record_retrieval(source, count, duration)
        
        # Should handle multiple source types
        assert True
    
    def test_cache_metrics_patterns(self):
        """Test cache operation patterns"""
        # Simulate realistic cache patterns
        cache_ops = [
            ("vector_search", True),   # Hit
            ("graph_search", False),   # Miss
            ("vector_search", True),   # Hit (same query)
            ("rerank_cache", False),   # Miss
            ("rerank_cache", True),    # Hit
        ]
        
        for operation, hit in cache_ops:
            self.telemetry.record_cache_operation(operation, hit)
        
        # Should track various operation types
        assert True
    
    def test_error_metrics_collection(self):
        """Test error tracking and categorization"""
        errors = [
            ("timeout", "api_client"),
            ("parse_error", "response_handler"), 
            ("rate_limit", "api_client"),
            ("validation_error", "input_processor"),
            ("timeout", "api_client"),  # Duplicate type
        ]
        
        for error_type, component in errors:
            self.telemetry.record_error(error_type, component)
        
        # Should handle error categorization
        assert True


class TestTelemetryPerformanceImpact:
    """Test telemetry performance overhead"""
    
    def setup_method(self):
        """Setup telemetry for performance testing"""
        self.telemetry = TelemetryManager()
        self.telemetry.initialize()
    
    def test_trace_operation_overhead(self):
        """Test tracing overhead is minimal"""
        iterations = 1000
        
        # Measure without tracing
        start_time = time.perf_counter()
        for _ in range(iterations):
            time.sleep(0.0001)  # Minimal work simulation
        baseline_time = time.perf_counter() - start_time
        
        # Measure with tracing
        start_time = time.perf_counter()
        for i in range(iterations):
            with self.telemetry.trace_operation("perf_test", {"iteration": i}):
                time.sleep(0.0001)  # Same work
        traced_time = time.perf_counter() - start_time
        
        # Tracing overhead should be reasonable (< 50% overhead)
        overhead_ratio = (traced_time - baseline_time) / baseline_time
        assert overhead_ratio < 0.5, f"Tracing overhead too high: {overhead_ratio:.2%}"
    
    def test_metrics_recording_performance(self):
        """Test metrics recording performance"""
        iterations = 10000
        
        start_time = time.perf_counter()
        for i in range(iterations):
            self.telemetry.record_query(f"query_{i}", "test", 0.1, True)
            self.telemetry.record_retrieval("vector", 10, 0.05)
            self.telemetry.record_cache_operation("test", i % 2 == 0)
        
        total_time = time.perf_counter() - start_time
        ops_per_second = (iterations * 3) / total_time
        
        # Should handle high throughput (>1000 ops/sec)
        assert ops_per_second > 1000, f"Metrics too slow: {ops_per_second:.1f} ops/sec"


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_telemetry"
    ])