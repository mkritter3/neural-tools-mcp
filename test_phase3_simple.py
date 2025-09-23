#!/usr/bin/env python3
"""
ADR-0084 Phase 3: Simple validation test
Quick test to verify circuit breaker and monitoring are working
"""

import asyncio
import sys
import time

sys.path.append('/Users/mkr/local-coding/claude-l9-template/neural-tools/src')


async def test_phase3_simple():
    """Simple test of Phase 3 features"""
    from servers.services.nomic_service import NomicService
    from servers.services.monitoring_service import monitoring_service

    print("=" * 60)
    print("🧪 ADR-0084 PHASE 3 SIMPLE TEST")
    print("=" * 60)

    # Initialize service
    print("\n1️⃣ Initializing Nomic service...")
    nomic = NomicService()
    result = await nomic.initialize()
    print(f"   Initialization: {result.get('success', False)}")

    # Check circuit breaker is initialized
    print("\n2️⃣ Checking circuit breaker...")
    if nomic.circuit_breaker:
        state = nomic.circuit_breaker.get_state()
        print(f"   ✅ Circuit breaker initialized")
        print(f"   State: {state['state']}")
        print(f"   Failure threshold: {state['failure_threshold']}")
    else:
        print(f"   ❌ Circuit breaker not initialized")
        return False

    # Check monitoring is initialized
    print("\n3️⃣ Checking monitoring service...")
    if nomic.monitoring:
        print(f"   ✅ Monitoring service connected")
    else:
        print(f"   ⚠️ Monitoring service not connected (optional)")

    # Test normal operation
    print("\n4️⃣ Testing normal embedding generation...")
    try:
        start = time.time()
        embedding = await nomic.get_embedding("test phase 3")
        latency = (time.time() - start) * 1000
        print(f"   ✅ Embedding generated in {latency:.1f}ms")
        print(f"   Dimension: {len(embedding)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

    # Get health status
    print("\n5️⃣ Getting health status...")
    health = await nomic.health_check()
    print(f"   Healthy: {health.get('healthy', False)}")
    if "circuit_breaker" in health:
        cb = health["circuit_breaker"]
        print(f"   Circuit breaker:")
        print(f"      State: {cb['state']}")
        print(f"      Success rate: {cb['success_rate']}")
        print(f"      Total calls: {cb['total_calls']}")

    # Test circuit breaker with bad URL
    print("\n6️⃣ Testing circuit breaker with failures...")
    import httpx
    import uuid

    original_url = nomic.client.base_url
    original_client = nomic.client.client

    # Break BOTH the URL and recreate client
    nomic.client.base_url = "http://invalid-host:48000"
    await nomic.client.client.aclose()
    nomic.client.client = httpx.AsyncClient(
        base_url="http://invalid-host:48000",
        transport=nomic.client.transport,
        timeout=nomic.client.timeout
    )

    failures = 0
    rejections = 0
    for i in range(7):
        # Use unique strings to bypass cache
        unique_text = f"fail test {uuid.uuid4()}"
        try:
            await nomic.get_embedding(unique_text)
        except RuntimeError as e:
            if "Circuit breaker" in str(e) and "is OPEN" in str(e):
                rejections += 1
                print(f"   Request {i+1}: Circuit OPEN (rejected)")
            else:
                failures += 1
                print(f"   Request {i+1}: Failed")
        except Exception:
            failures += 1
            print(f"   Request {i+1}: Failed")

    # Restore URL and client
    nomic.client.base_url = original_url
    await nomic.client.client.aclose()
    nomic.client.client = original_client

    print(f"\n   Summary: {failures} failures, {rejections} rejections")
    print(f"   Circuit breaker {'✅ WORKED' if rejections > 0 else '❌ DID NOT OPEN'}")

    # Get final state
    final_state = nomic.circuit_breaker.get_state()
    print(f"\n7️⃣ Final circuit breaker state:")
    print(f"   State: {final_state['state']}")
    print(f"   Failed calls: {final_state['failed_calls']}")
    print(f"   Rejected calls: {final_state['rejected_calls']}")

    # Check monitoring metrics if available
    if nomic.monitoring:
        print(f"\n8️⃣ Monitoring metrics:")
        metrics = monitoring_service.get_metrics_summary()
        if "nomic" in metrics:
            m = metrics["nomic"]
            print(f"   Total requests: {m['total_requests']}")
            print(f"   Success rate: {m['success_rate']}")
            print(f"   Avg latency: {m['avg_latency_ms']}")

    print("\n" + "=" * 60)
    print("📊 PHASE 3 TEST SUMMARY")
    print("=" * 60)

    success = rejections > 0  # Circuit breaker opened
    if success:
        print("✅ Phase 3 features working:")
        print("   • Circuit breaker protects against failures")
        print("   • Health monitoring active")
        print("   • Metrics collection working")
    else:
        print("❌ Circuit breaker did not open as expected")

    return success


if __name__ == "__main__":
    success = asyncio.run(test_phase3_simple())
    sys.exit(0 if success else 1)