#!/usr/bin/env python3
"""
Basic Service Connectivity Smoke Test
Verifies all required services are accessible
"""

import sys
import asyncio
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def check_service_port(host: str, port: int, service_name: str) -> bool:
    """Check if a service is accessible on a port"""
    try:
        # Use nc (netcat) to check port
        result = subprocess.run(
            ["nc", "-z", "-v", "-w", "1", host, str(port)],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            print(f"  ✅ {service_name} is accessible on {host}:{port}")
            return True
        else:
            print(f"  ❌ {service_name} is not accessible on {host}:{port}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ❌ {service_name} connection timed out on {host}:{port}")
        return False
    except Exception as e:
        print(f"  ❌ {service_name} check failed: {e}")
        return False


def check_http_endpoint(url: str, service_name: str) -> bool:
    """Check if an HTTP endpoint is accessible"""
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                print(f"  ✅ {service_name} HTTP endpoint is healthy: {url}")
                return True
            else:
                print(f"  ⚠️ {service_name} returned status {response.status}")
                return True  # Non-200 but accessible
    except urllib.error.URLError as e:
        print(f"  ❌ {service_name} HTTP endpoint not accessible: {e}")
        return False
    except Exception as e:
        print(f"  ❌ {service_name} HTTP check failed: {e}")
        return False


async def test_basic_connectivity():
    """Test basic connectivity to all required services"""
    print("\n🔌 Testing Basic Service Connectivity...")

    all_services_ok = True

    # Check Neo4j
    print("\n  Neo4j Graph Database:")
    if not check_service_port("localhost", 47687, "Neo4j Bolt"):
        all_services_ok = False
    if not check_http_endpoint("http://localhost:47474", "Neo4j Browser"):
        pass  # Browser is optional

    # Check Qdrant
    print("\n  Qdrant Vector Database:")
    if not check_service_port("localhost", 46333, "Qdrant API"):
        all_services_ok = False
    if not check_http_endpoint("http://localhost:46333/collections", "Qdrant Collections"):
        all_services_ok = False

    # Check Redis Cache
    print("\n  Redis Cache:")
    if not check_service_port("localhost", 46379, "Redis Cache"):
        all_services_ok = False

    # Check Redis Queue
    print("\n  Redis Queue:")
    if not check_service_port("localhost", 46380, "Redis Queue"):
        all_services_ok = False

    # Check Nomic Embeddings
    print("\n  Nomic Embeddings Service:")
    if not check_service_port("localhost", 48000, "Nomic API"):
        print("    ⚠️ Nomic is optional, continuing...")
    else:
        check_http_endpoint("http://localhost:48000/health", "Nomic Health")

    # Test Redis connectivity with actual commands (with auth)
    print("\n  Testing Redis Functionality:")
    try:
        # Test Redis Cache (with password)
        cache_result = subprocess.run(
            ["redis-cli", "-p", "46379", "-a", "cache-secret-key", "ping"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if "PONG" in cache_result.stdout:
            print("    ✅ Redis Cache responds to PING")
        else:
            print("    ⚠️ Redis Cache auth required (expected for production)")

        # Test Redis Queue (with password)
        queue_result = subprocess.run(
            ["redis-cli", "-p", "46380", "-a", "queue-secret-key", "ping"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if "PONG" in queue_result.stdout:
            print("    ✅ Redis Queue responds to PING")
        else:
            print("    ⚠️ Redis Queue auth required (expected for production)")

    except FileNotFoundError:
        print("    ⚠️ redis-cli not found, skipping Redis command tests")
    except Exception as e:
        print(f"    ❌ Redis functionality test failed: {e}")

    # Summary
    print("\n" + "=" * 50)
    if all_services_ok:
        print("✅ Basic Connectivity Test PASSED")
        print("All required services are accessible")
        return True
    else:
        print("❌ Basic Connectivity Test FAILED")
        print("Some required services are not accessible")
        print("\nTo fix:")
        print("1. Ensure Docker is running")
        print("2. Run: docker-compose up -d")
        print("3. Wait for containers to be healthy")
        return False


def main():
    """Run the test"""
    success = asyncio.run(test_basic_connectivity())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()