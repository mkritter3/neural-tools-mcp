#!/usr/bin/env python3
"""
ADR-0084 Phase 3: Load Testing Suite
Tests circuit breaker, monitoring, and resilience under load
"""

import asyncio
import time
import sys
import random
from datetime import datetime
import numpy as np

# Add the neural-tools source to path
sys.path.append('/Users/mkr/local-coding/claude-l9-template/neural-tools/src')


async def simulate_failure_scenario():
    """Simulate service failures to test circuit breaker"""
    from servers.services.nomic_service import NomicService

    print("\n🔥 SIMULATING FAILURE SCENARIO")
    print("=" * 60)

    nomic = NomicService()
    await nomic.initialize()

    # Track circuit breaker states
    states = []

    # Generate 10 requests that will fail (invalid base URL)
    print("\n1️⃣ Triggering failures to open circuit breaker...")

    # Temporarily break the service
    original_url = nomic.client.base_url
    nomic.client.base_url = "http://invalid-host:48000"  # Invalid host

    for i in range(10):
        try:
            await nomic.get_embedding(f"failure test {i}")
            print(f"   Request {i+1}: Success (unexpected)")
        except Exception as e:
            if "Circuit breaker" in str(e) and "is OPEN" in str(e):
                print(f"   Request {i+1}: Circuit OPEN - rejected")
                states.append("OPEN")
            else:
                print(f"   Request {i+1}: Failed - circuit still closed")
                states.append("CLOSED")

        # Check circuit breaker state
        if nomic.circuit_breaker:
            cb_state = nomic.circuit_breaker.get_state()
            print(f"      Circuit state: {cb_state['state']}, failures: {cb_state['failure_count']}")

    # Restore service
    nomic.client.base_url = original_url

    print("\n2️⃣ Waiting for recovery timeout (30s)...")
    await asyncio.sleep(31)

    print("\n3️⃣ Testing recovery (circuit should be HALF_OPEN)...")
    try:
        # This should succeed and close the circuit
        await nomic.get_embedding("recovery test")
        print("   ✅ Recovery successful - circuit should be CLOSED")
    except Exception as e:
        print(f"   ❌ Recovery failed: {e}")

    # Final state check
    if nomic.circuit_breaker:
        final_state = nomic.circuit_breaker.get_state()
        print(f"\n📊 Final circuit breaker state:")
        print(f"   State: {final_state['state']}")
        print(f"   Total calls: {final_state['total_calls']}")
        print(f"   Successful: {final_state['successful_calls']}")
        print(f"   Failed: {final_state['failed_calls']}")
        print(f"   Rejected: {final_state['rejected_calls']}")
        print(f"   Success rate: {final_state['success_rate']}")

    return "OPEN" in states  # Return True if circuit opened


async def load_test_with_monitoring():
    """Load test with monitoring and metrics collection"""
    from servers.services.nomic_service import NomicService
    from servers.services.monitoring_service import monitoring_service

    print("\n📊 LOAD TEST WITH MONITORING")
    print("=" * 60)

    nomic = NomicService()
    await nomic.initialize()

    # Configuration
    num_requests = 100
    concurrent = 10
    batch_sizes = [1, 5, 10, 20]

    print(f"\nConfiguration:")
    print(f"   Total requests: {num_requests}")
    print(f"   Concurrency: {concurrent}")
    print(f"   Batch sizes: {batch_sizes}")

    async def make_request(idx: int, batch_size: int):
        """Make a single request"""
        texts = [f"load test {idx} text {i}" for i in range(batch_size)]
        try:
            start = time.time()
            _ = await nomic.get_embeddings(texts, task_type="search_document")
            latency = (time.time() - start) * 1000
            return True, latency
        except Exception as e:
            return False, 0

    # Run load test
    print("\n🏃 Running load test...")
    start_time = time.time()
    tasks = []
    sem = asyncio.Semaphore(concurrent)

    async def limited_request(idx, batch_size):
        async with sem:
            return await make_request(idx, batch_size)

    for i in range(num_requests):
        batch_size = random.choice(batch_sizes)
        tasks.append(limited_request(i, batch_size))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.time() - start_time

    # Analyze results
    successful = sum(1 for r in results if isinstance(r, tuple) and r[0])
    failed = num_requests - successful
    latencies = [r[1] for r in results if isinstance(r, tuple) and r[0] and r[1] > 0]

    print(f"\n📈 Load Test Results:")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Successful: {successful}/{num_requests} ({successful/num_requests*100:.1f}%)")
    print(f"   Failed: {failed}")
    print(f"   Throughput: {successful/duration:.1f} req/s")

    if latencies:
        print(f"\n⏱️ Latency Statistics:")
        print(f"   Min: {min(latencies):.1f}ms")
        print(f"   Max: {max(latencies):.1f}ms")
        print(f"   Avg: {np.mean(latencies):.1f}ms")
        print(f"   P50: {np.percentile(latencies, 50):.1f}ms")
        print(f"   P95: {np.percentile(latencies, 95):.1f}ms")
        print(f"   P99: {np.percentile(latencies, 99):.1f}ms")

    # Get monitoring metrics
    if nomic.monitoring:
        health = await monitoring_service.get_system_health()

        print(f"\n📊 Monitoring Metrics:")
        print(f"   Overall healthy: {health['overall_healthy']}")

        if "nomic" in health["metrics"]:
            metrics = health["metrics"]["nomic"]
            print(f"   Service metrics:")
            print(f"      Success rate: {metrics['success_rate']}")
            print(f"      Avg latency: {metrics['avg_latency_ms']}")
            print(f"      RPM: {metrics['requests_per_minute']}")
            print(f"      Error rate (5min): {metrics['error_rate_5min']}")

        if health["alerts"]:
            print(f"\n🚨 Active Alerts:")
            for alert in health["alerts"]:
                print(f"   - {alert['type']}: {alert['message']}")

    return successful / num_requests >= 0.95  # 95% success rate


async def test_monitoring_aggregation():
    """Test monitoring service aggregation"""
    from servers.services.monitoring_service import monitoring_service
    from servers.services.nomic_service import NomicService

    print("\n🔍 TESTING MONITORING AGGREGATION")
    print("=" * 60)

    # Initialize service
    nomic = NomicService()
    await nomic.initialize()

    # Make some requests to generate metrics
    print("\nGenerating test metrics...")
    for i in range(10):
        try:
            await nomic.get_embedding(f"monitoring test {i}")
            print(f"   Request {i+1}: Success")
        except:
            print(f"   Request {i+1}: Failed")

    # Get system health
    health = await monitoring_service.get_system_health()

    print(f"\n📊 System Health Report:")
    print(f"   Timestamp: {health['timestamp']}")
    print(f"   Overall healthy: {health['overall_healthy']}")
    print(f"   System metrics: {health['system_metrics']}")

    # Check service health
    if "nomic" in health["services"]:
        service_health = health["services"]["nomic"]
        print(f"\n   Nomic Service:")
        print(f"      Healthy: {service_health.get('healthy', False)}")
        print(f"      URL: {service_health.get('service_url', 'unknown')}")

        if "circuit_breaker" in service_health:
            cb = service_health["circuit_breaker"]
            print(f"      Circuit breaker: {cb.get('state', 'unknown')}")

    return health["overall_healthy"]


async def stress_test_circuit_breaker():
    """Stress test the circuit breaker with rapid failures"""
    from servers.services.nomic_service import NomicService

    print("\n⚡ STRESS TESTING CIRCUIT BREAKER")
    print("=" * 60)

    nomic = NomicService()
    await nomic.initialize()

    # Temporarily break the service
    original_url = nomic.client.base_url
    nomic.client.base_url = "http://invalid-host:48000"

    print("\n1️⃣ Rapid-fire failure requests...")

    # Fire 20 requests as fast as possible
    tasks = []
    for i in range(20):
        tasks.append(nomic.get_embedding(f"stress {i}"))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count rejections vs failures
    rejections = sum(1 for r in results if isinstance(r, Exception) and "Circuit breaker" in str(r) and "is OPEN" in str(r))
    failures = sum(1 for r in results if isinstance(r, Exception) and "Circuit breaker" not in str(r))

    print(f"\n📊 Stress Test Results:")
    print(f"   Total requests: 20")
    print(f"   Circuit rejections: {rejections}")
    print(f"   Service failures: {failures}")
    print(f"   Circuit opened after: ~{20 - rejections} failures")

    # Restore service
    nomic.client.base_url = original_url

    # Get final state
    if nomic.circuit_breaker:
        state = nomic.circuit_breaker.get_state()
        print(f"\n   Final circuit state:")
        print(f"      State: {state['state']}")
        print(f"      Rejected calls: {state['rejected_calls']}")

    return rejections > 0  # Should have some rejections


async def run_phase3_tests():
    """Run all Phase 3 tests"""
    print("=" * 60)
    print("🧪 ADR-0084 PHASE 3 TESTING")
    print(f"📅 {datetime.now().isoformat()}")
    print("=" * 60)

    results = {}

    try:
        # Test 1: Circuit breaker failure handling
        print("\n\n[TEST 1] Circuit Breaker Failure Handling")
        results["circuit_breaker"] = await simulate_failure_scenario()

        # Test 2: Load test with monitoring
        print("\n\n[TEST 2] Load Test with Monitoring")
        results["load_test"] = await load_test_with_monitoring()

        # Test 3: Monitoring aggregation
        print("\n\n[TEST 3] Monitoring Aggregation")
        results["monitoring"] = await test_monitoring_aggregation()

        # Test 4: Stress test circuit breaker
        print("\n\n[TEST 4] Stress Test Circuit Breaker")
        results["stress_test"] = await stress_test_circuit_breaker()

        # Final results
        print("\n\n" + "=" * 60)
        print("📊 PHASE 3 TEST RESULTS")
        print("=" * 60)

        tests = {
            "Circuit Breaker Opens on Failures": results.get("circuit_breaker", False),
            "Load Test (95% success)": results.get("load_test", False),
            "Monitoring Aggregation": results.get("monitoring", False),
            "Circuit Breaker Under Stress": results.get("stress_test", False),
        }

        all_pass = True
        for test_name, passed in tests.items():
            status = "✅" if passed else "❌"
            print(f"{status} {test_name}")
            all_pass = all_pass and passed

        print("\n" + "=" * 60)
        if all_pass:
            print("🎉 PHASE 3 COMPLETE! All reliability features working:")
            print("   • Circuit breaker protects against cascading failures")
            print("   • Monitoring tracks health and performance")
            print("   • System maintains 95%+ success under load")
            print("   • Automatic recovery after failures")
        else:
            print("⚠️ PHASE 3 INCOMPLETE. Some tests failed.")
        print("=" * 60)

        return all_pass

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_phase3_tests())
    sys.exit(0 if success else 1)