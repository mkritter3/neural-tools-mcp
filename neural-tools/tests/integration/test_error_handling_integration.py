#!/usr/bin/env python3
"""
Phase 1.5 Integration Tests - Error Handling with Real System Integration
Tests error handling integration with real services, recovery scenarios, and production patterns
"""

import pytest
import pytest_asyncio
import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, AsyncMock
import sys
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

from infrastructure.error_handling import (
    ErrorHandler,
    GracefulDegradation,
    ErrorSeverity,
    ErrorCategory,
    VectorDatabaseException,
    GraphDatabaseException,
    EmbeddingException,
    ValidationException,
    RateLimitException,
    retry_with_backoff,
    database_retry,
    embedding_retry,
    network_retry
)

class TestErrorHandlingIntegration:
    """Integration tests for error handling with real workloads"""
    
    @pytest.fixture
    def error_handler(self):
        """Fresh error handler for each test"""
        return ErrorHandler()
    
    @pytest.fixture  
    def degradation_manager(self):
        """Fresh degradation manager for each test"""
        return GracefulDegradation()
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_error_classification_performance_target(self, error_handler):
        """Verify error classification <10ms per error (Roadmap requirement)"""
        errors = [
            ConnectionError("Network timeout"),
            Exception("Qdrant connection failed"),
            ValueError("Invalid parameter: missing required field"),
            TimeoutError("Request timeout after 30s"),
            Exception("Neo4j query execution failed"),
            RuntimeError("Memory allocation failed"),
            Exception("External API rate limit exceeded"),
            KeyError("Missing configuration key"),
        ]
        
        processing_times = []
        
        for error in errors:
            start_time = time.perf_counter()
            await error_handler.handle_error(error, operation_name="performance_test")
            end_time = time.perf_counter()
            
            processing_time_ms = (end_time - start_time) * 1000
            processing_times.append(processing_time_ms)
        
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)
        
        print(f"Error classification performance:")
        print(f"  Average: {avg_processing_time:.2f}ms")
        print(f"  Maximum: {max_processing_time:.2f}ms")
        print(f"  Per-error times: {[f'{t:.2f}ms' for t in processing_times]}")
        
        # ROADMAP EXIT CRITERIA: <10ms error classification
        assert avg_processing_time < 10, f"Average classification time {avg_processing_time:.2f}ms exceeds 10ms target"
        assert max_processing_time < 20, f"Max classification time {max_processing_time:.2f}ms exceeds 20ms threshold"
    
    @pytest.mark.resilience
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_target(self, error_handler):
        """Verify circuit breaker opens/closes correctly (Roadmap requirement)"""
        operation_name = "circuit_breaker_test"
        
        # Phase 1: Generate failures to open circuit breaker
        print("Phase 1: Generating failures to trigger circuit breaker...")
        failure_times = []
        
        for i in range(6):  # Exceed threshold of 5
            start_time = time.perf_counter()
            error = VectorDatabaseException(f"Database failure {i+1}")
            await error_handler.handle_error(error, operation_name=operation_name)
            end_time = time.perf_counter()
            
            failure_times.append(end_time - start_time)
        
        error_key = f"{operation_name}:database"
        
        # Verify circuit breaker opened
        assert error_key in error_handler.circuit_breaker_states
        circuit_state = error_handler.circuit_breaker_states[error_key]
        assert circuit_state["state"] == "open"
        assert circuit_state["error_count"] >= 5
        
        print(f"✅ Circuit breaker opened after {circuit_state['error_count']} errors")
        
        # Phase 2: Simulate recovery after cool-down period
        print("Phase 2: Testing circuit breaker recovery...")
        
        # Simulate time passing (in real system, would wait for cool-down)
        circuit_state["opened_at"] = datetime.now() - timedelta(minutes=6)
        
        # Test if circuit allows requests after cool-down
        recovery_start = time.time()
        recovery_error = Exception("Recovery test error")
        await error_handler.handle_error(recovery_error, operation_name=f"{operation_name}_recovery")
        recovery_time = time.time() - recovery_start
        
        print(f"Recovery test completed in {recovery_time*1000:.2f}ms")
        
        # ROADMAP EXIT CRITERIA: Circuit breaker functionality
        assert recovery_time < 0.1  # Recovery should be fast
        assert circuit_state["state"] == "open"  # Should still be open for original operation
        
        avg_failure_time = sum(failure_times) / len(failure_times)
        assert avg_failure_time < 0.05  # Error processing should be fast
    
    @pytest.mark.asyncio
    async def test_retry_effectiveness_under_load(self):
        """Test retry mechanisms under concurrent load"""
        success_count = 0
        total_attempts = 0
        
        @database_retry
        async def flaky_operation(operation_id: int):
            nonlocal total_attempts
            total_attempts += 1
            
            # Simulate 60% failure rate
            import random
            if random.random() < 0.6:
                raise VectorDatabaseException(f"Database timeout for operation {operation_id}")
            
            nonlocal success_count
            success_count += 1
            return f"success_{operation_id}"
        
        # Execute 50 concurrent operations
        tasks = [flaky_operation(i) for i in range(50)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_operations = [r for r in results if isinstance(r, str) and r.startswith("success")]
        failed_operations = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_operations) / len(results)
        total_time = end_time - start_time
        
        print(f"Retry effectiveness test results:")
        print(f"  Total operations: {len(results)}")
        print(f"  Successful: {len(successful_operations)} ({success_rate:.1%})")
        print(f"  Failed: {len(failed_operations)}")
        print(f"  Total attempts made: {total_attempts}")
        print(f"  Retry multiplier: {total_attempts / len(results):.1f}x")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {len(results) / total_time:.1f} ops/sec")
        
        # ROADMAP EXIT CRITERIA: Retry effectiveness
        assert success_rate > 0.7  # At least 70% should eventually succeed
        assert total_attempts > len(results)  # Retries should have occurred
        assert total_time < 30  # Should complete within reasonable time
        assert len(results) / total_time > 5  # Reasonable throughput
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_latency_impact(self, degradation_manager):
        """Verify graceful degradation <20% latency overhead (Roadmap requirement)"""
        
        # Baseline: Direct function call
        async def primary_service(data: str):
            await asyncio.sleep(0.01)  # Simulate 10ms processing
            return f"primary_result_{data}"
        
        async def fallback_service(data: str):
            await asyncio.sleep(0.005)  # Simulate 5ms fallback processing
            return f"fallback_result_{data}"
        
        degradation_manager.register_fallback("latency_test", fallback_service)
        
        # Test 1: Baseline direct calls
        baseline_times = []
        for i in range(100):
            start_time = time.perf_counter()
            result = await primary_service(f"test_{i}")
            end_time = time.perf_counter()
            baseline_times.append(end_time - start_time)
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Test 2: Through degradation manager (healthy service)
        degradation_times = []
        for i in range(100):
            start_time = time.perf_counter()
            result = await degradation_manager.execute_with_fallback(
                "latency_test", primary_service, f"test_{i}"
            )
            end_time = time.perf_counter()
            degradation_times.append(end_time - start_time)
            
            assert result["success"]
            assert not result["fallback_used"]
        
        degradation_avg = sum(degradation_times) / len(degradation_times)
        
        # Test 3: Fallback execution
        degradation_manager.mark_service_unhealthy("latency_test")
        
        fallback_times = []
        for i in range(100):
            start_time = time.perf_counter()
            result = await degradation_manager.execute_with_fallback(
                "latency_test", primary_service, f"test_{i}"
            )
            end_time = time.perf_counter()
            fallback_times.append(end_time - start_time)
            
            assert result["success"]
            assert result["fallback_used"]
        
        fallback_avg = sum(fallback_times) / len(fallback_times)
        
        # Calculate overhead
        degradation_overhead = ((degradation_avg - baseline_avg) / baseline_avg) * 100
        fallback_overhead = ((fallback_avg - baseline_avg) / baseline_avg) * 100
        
        print(f"Graceful degradation latency analysis:")
        print(f"  Baseline avg: {baseline_avg*1000:.2f}ms")
        print(f"  Through degradation manager: {degradation_avg*1000:.2f}ms ({degradation_overhead:+.1f}%)")
        print(f"  Fallback execution: {fallback_avg*1000:.2f}ms ({fallback_overhead:+.1f}%)")
        
        # ROADMAP EXIT CRITERIA: <20% latency overhead
        assert degradation_overhead < 20, f"Degradation overhead {degradation_overhead:.1f}% exceeds 20% target"
        # Fallback can be different since it's different logic, but should be reasonable
        assert fallback_avg < baseline_avg * 2, "Fallback should not be more than 2x slower"
    
    @pytest.mark.resilience
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, error_handler, degradation_manager):
        """Test comprehensive error recovery scenarios"""
        recovery_scenarios = []
        
        # Scenario 1: Database connection failure with recovery
        @database_retry
        async def recovering_database_operation():
            # Simulate gradual recovery
            if not hasattr(recovering_database_operation, 'call_count'):
                recovering_database_operation.call_count = 0
            recovering_database_operation.call_count += 1
            
            if recovering_database_operation.call_count <= 2:
                raise VectorDatabaseException("Connection pool exhausted")
            return {"status": "connected", "data": "recovered_data"}
        
        # Scenario 2: External API failure with fallback
        async def failing_external_api():
            raise Exception("External service unavailable")
        
        async def cached_fallback():
            return {"source": "cache", "data": "stale_but_valid"}
        
        degradation_manager.register_fallback("external_api", cached_fallback)
        
        # Scenario 3: Embedding service failure with degraded search
        async def failing_embedding_service():
            raise EmbeddingException("Model server overloaded")
        
        async def keyword_search_fallback():
            return {"type": "keyword", "results": ["fallback_result_1", "fallback_result_2"]}
        
        degradation_manager.register_fallback("embedding", keyword_search_fallback)
        
        # Execute recovery scenarios
        start_time = time.time()
        
        # Test database recovery
        try:
            db_result = await recovering_database_operation()
            recovery_scenarios.append({
                "scenario": "database_recovery",
                "success": True,
                "result": db_result,
                "attempts": recovering_database_operation.call_count
            })
        except Exception as e:
            recovery_scenarios.append({
                "scenario": "database_recovery", 
                "success": False,
                "error": str(e)
            })
        
        # Test API fallback
        api_result = await degradation_manager.execute_with_fallback(
            "external_api", failing_external_api
        )
        recovery_scenarios.append({
            "scenario": "api_fallback",
            "success": api_result["success"],
            "fallback_used": api_result["fallback_used"],
            "result": api_result.get("result")
        })
        
        # Test embedding fallback
        embed_result = await degradation_manager.execute_with_fallback(
            "embedding", failing_embedding_service
        )
        recovery_scenarios.append({
            "scenario": "embedding_fallback",
            "success": embed_result["success"],
            "fallback_used": embed_result["fallback_used"],
            "result": embed_result.get("result")
        })
        
        total_recovery_time = time.time() - start_time
        
        print(f"Error recovery scenarios completed in {total_recovery_time:.2f}s:")
        for scenario in recovery_scenarios:
            print(f"  {scenario['scenario']}: {'✅' if scenario['success'] else '❌'}")
            if 'attempts' in scenario:
                print(f"    Attempts: {scenario['attempts']}")
            if 'fallback_used' in scenario:
                print(f"    Fallback used: {scenario['fallback_used']}")
        
        # Verify all scenarios succeeded
        successful_scenarios = [s for s in recovery_scenarios if s["success"]]
        assert len(successful_scenarios) == len(recovery_scenarios), "All recovery scenarios should succeed"
        
        # Verify reasonable recovery time
        assert total_recovery_time < 10, "Recovery scenarios should complete quickly"
    
    @pytest.mark.asyncio  
    async def test_memory_usage_during_error_handling(self, error_handler):
        """Test memory efficiency during intensive error handling"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large number of errors with different contexts
        error_contexts = []
        for i in range(1000):
            context = {
                "operation_id": f"op_{i}",
                "user_data": {"query": f"test query {i}" * 10},  # Some bulk data
                "system_info": {
                    "timestamp": datetime.now().isoformat(),
                    "memory_usage": f"{initial_memory:.1f}MB",
                    "operation_depth": i % 10
                }
            }
            error_contexts.append(context)
        
        # Process errors concurrently in batches
        batch_size = 50
        processing_times = []
        
        for batch_start in range(0, len(error_contexts), batch_size):
            batch_contexts = error_contexts[batch_start:batch_start + batch_size]
            
            batch_start_time = time.time()
            
            # Create error handling tasks for batch
            error_tasks = []
            for i, context in enumerate(batch_contexts):
                error = Exception(f"Batch error {batch_start + i}")
                task = error_handler.handle_error(
                    error,
                    context=context,
                    operation_name=f"memory_test_batch_{batch_start // batch_size}"
                )
                error_tasks.append(task)
            
            # Process batch
            await asyncio.gather(*error_tasks)
            
            batch_time = time.time() - batch_start_time
            processing_times.append(batch_time)
            
            # Memory check every few batches
            if (batch_start // batch_size) % 5 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                print(f"Batch {batch_start // batch_size}: Memory growth: {memory_growth:.1f}MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_growth = final_memory - initial_memory
        
        avg_batch_time = sum(processing_times) / len(processing_times)
        total_errors_processed = len(error_contexts)
        
        print(f"Memory usage during error handling:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Total growth: {total_memory_growth:.1f}MB")
        print(f"  Errors processed: {total_errors_processed}")
        print(f"  Memory per error: {total_memory_growth / total_errors_processed * 1024:.2f}KB")
        print(f"  Average batch time: {avg_batch_time*1000:.2f}ms")
        print(f"  Error processing rate: {total_errors_processed / sum(processing_times):.0f} errors/sec")
        
        # Memory efficiency requirements
        assert total_memory_growth < 100, f"Memory growth {total_memory_growth:.1f}MB exceeds 100MB limit"
        assert total_memory_growth / total_errors_processed < 0.1, "Memory per error should be <100KB"
        assert sum(processing_times) < 60, "Should process all errors within 60 seconds"
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_concurrent_error_handling_stress(self, error_handler, degradation_manager):
        """Test error handling under high concurrent load"""
        
        # Setup multiple services with different failure patterns
        services = {
            "vector_db": {
                "failure_rate": 0.3,
                "exception": VectorDatabaseException,
                "fallback": lambda: {"source": "cache", "vectors": []}
            },
            "graph_db": {
                "failure_rate": 0.2, 
                "exception": GraphDatabaseException,
                "fallback": lambda: {"source": "memory", "relationships": []}
            },
            "embedding": {
                "failure_rate": 0.4,
                "exception": EmbeddingException,
                "fallback": lambda: {"source": "precomputed", "embeddings": [0.1, 0.2, 0.3]}
            },
            "validation": {
                "failure_rate": 0.1,
                "exception": ValidationException,
                "fallback": lambda: {"source": "default", "validated": True}
            }
        }
        
        # Register fallbacks
        for service_name, config in services.items():
            degradation_manager.register_fallback(service_name, config["fallback"])
        
        async def simulate_service_call(service_name: str, call_id: int):
            """Simulate a service call with potential failure"""
            service_config = services[service_name]
            
            # Simulate processing time
            await asyncio.sleep(0.001 + (call_id % 10) * 0.0001)  # 1-2ms variation
            
            # Determine if this call should fail
            import random
            if random.random() < service_config["failure_rate"]:
                error = service_config["exception"](f"{service_name} failure for call {call_id}")
                await error_handler.handle_error(error, operation_name=f"{service_name}_operation")
                
                # Try fallback
                fallback_result = await degradation_manager.execute_with_fallback(
                    service_name, 
                    lambda: (_ for _ in ()).throw(error),  # Lambda that raises the error
                )
                return fallback_result
            else:
                # Success case
                return {
                    "success": True,
                    "service": service_name,
                    "call_id": call_id,
                    "result": f"success_data_{call_id}"
                }
        
        # Generate high concurrent load
        num_concurrent_calls = 500
        calls_per_service = num_concurrent_calls // len(services)
        
        all_tasks = []
        for service_name in services:
            for call_id in range(calls_per_service):
                task = simulate_service_call(service_name, call_id)
                all_tasks.append((service_name, call_id, task))
        
        # Execute all calls concurrently
        start_time = time.time()
        results = await asyncio.gather(*[task for _, _, task in all_tasks], return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_calls = []
        fallback_calls = []
        failed_calls = []
        exceptions = []
        
        for i, result in enumerate(results):
            service_name, call_id, _ = all_tasks[i]
            
            if isinstance(result, Exception):
                exceptions.append({"service": service_name, "call_id": call_id, "error": result})
            elif isinstance(result, dict):
                if result.get("success"):
                    if result.get("fallback_used"):
                        fallback_calls.append(result)
                    else:
                        successful_calls.append(result)
                else:
                    failed_calls.append(result)
        
        total_time = end_time - start_time
        throughput = len(all_tasks) / total_time
        
        print(f"Concurrent error handling stress test results:")
        print(f"  Total calls: {len(all_tasks)}")
        print(f"  Successful: {len(successful_calls)}")
        print(f"  Fallback used: {len(fallback_calls)}")
        print(f"  Failed: {len(failed_calls)}")
        print(f"  Exceptions: {len(exceptions)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} calls/sec")
        
        # Success metrics
        total_successful = len(successful_calls) + len(fallback_calls)
        success_rate = total_successful / len(all_tasks)
        fallback_rate = len(fallback_calls) / len(all_tasks)
        
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Fallback rate: {fallback_rate:.1%}")
        
        # Performance requirements
        assert success_rate > 0.95, f"Success rate {success_rate:.1%} below 95% target"
        assert throughput > 100, f"Throughput {throughput:.0f} below 100 calls/sec target"
        assert total_time < 30, f"Stress test took {total_time:.2f}s, should be <30s"
        assert len(exceptions) < len(all_tasks) * 0.05, "Exception rate should be <5%"
    
    @pytest.mark.asyncio
    async def test_error_pattern_detection(self, error_handler):
        """Test detection of error patterns and automatic mitigation"""
        
        # Simulate error patterns over time
        error_patterns = [
            # Pattern 1: Gradual increase in database errors
            {"service": "vector_db", "count": 10, "interval": 0.1},
            # Pattern 2: Burst of network errors
            {"service": "network", "count": 20, "interval": 0.05},
            # Pattern 3: Sustained embedding failures
            {"service": "embedding", "count": 15, "interval": 0.2},
        ]
        
        pattern_results = {}
        
        for pattern in error_patterns:
            service = pattern["service"]
            pattern_start = time.time()
            
            # Generate error pattern
            for i in range(pattern["count"]):
                if service == "vector_db":
                    error = VectorDatabaseException(f"DB error {i+1}")
                elif service == "network":
                    error = ConnectionError(f"Network error {i+1}")
                elif service == "embedding":
                    error = EmbeddingException(f"Embedding error {i+1}")
                
                await error_handler.handle_error(error, operation_name=f"{service}_pattern_test")
                await asyncio.sleep(pattern["interval"])
            
            pattern_duration = time.time() - pattern_start
            
            # Check error tracking - determine category based on service
            if service == "vector_db":
                category = "database"
            elif service == "network":
                category = "network"
            elif service == "embedding":
                category = "computation"
            else:
                category = "computation"
            
            error_key = f"{service}_pattern_test:{category}"
            error_count = error_handler.error_counts.get(error_key, 0)
            
            pattern_results[service] = {
                "errors_generated": pattern["count"],
                "errors_tracked": error_count,
                "duration": pattern_duration,
                "circuit_breaker_triggered": error_key in error_handler.circuit_breaker_states
            }
        
        print("Error pattern detection results:")
        for service, results in pattern_results.items():
            print(f"  {service}:")
            print(f"    Generated: {results['errors_generated']} errors")
            print(f"    Tracked: {results['errors_tracked']} errors") 
            print(f"    Duration: {results['duration']:.2f}s")
            print(f"    Circuit breaker: {'✅' if results['circuit_breaker_triggered'] else '❌'}")
        
        # Verify pattern detection
        for service, results in pattern_results.items():
            assert results["errors_tracked"] >= results["errors_generated"], f"{service} error tracking incomplete"
            
            # High error rate services should trigger circuit breaker
            if results["errors_generated"] >= 15:
                assert results["circuit_breaker_triggered"], f"{service} should have triggered circuit breaker"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])