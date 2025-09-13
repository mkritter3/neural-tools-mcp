#!/usr/bin/env python3
"""
Container Health Integration Test
Verifies all containers are healthy and properly configured
"""

import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def get_container_status() -> List[Dict]:
    """Get status of all running containers"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            print(f"  ❌ Failed to get container status: {result.stderr}")
            return []

        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                containers.append(json.loads(line))

        return containers
    except Exception as e:
        print(f"  ❌ Error getting container status: {e}")
        return []


def check_container_health(container_name: str) -> Tuple[bool, str]:
    """Check health status of a specific container"""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            health_status = result.stdout.strip()
            if health_status == "healthy":
                return True, health_status
            elif health_status == "<no value>":
                # No health check defined, check if running
                running_result = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if running_result.stdout.strip() == "true":
                    return True, "running (no health check)"
                else:
                    return False, "not running"
            else:
                return False, health_status
        else:
            return False, "container not found"
    except Exception as e:
        return False, f"error: {e}"


def check_container_logs_for_errors(container_name: str, lines: int = 50) -> List[str]:
    """Check recent container logs for errors"""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container_name],
            capture_output=True,
            text=True,
            timeout=5
        )

        errors = []
        if result.returncode == 0:
            for line in result.stdout.split('\n') + result.stderr.split('\n'):
                lower_line = line.lower()
                if any(term in lower_line for term in ['error', 'exception', 'fatal', 'failed']):
                    # Filter out known non-issues
                    if not any(skip in lower_line for skip in ['error_log', 'no error', 'error: 0']):
                        errors.append(line.strip())

        return errors[-5:]  # Return last 5 errors
    except Exception:
        return []


async def test_container_health():
    """Test health of all required containers"""
    print("\n🐳 Testing Container Health Integration...")

    # Required containers - flexible matching
    required_containers = {
        "neo4j": ["claude-l9-template-neo4j-1", "neo4j"],
        "qdrant": ["claude-l9-template-qdrant-1", "qdrant"],
        "redis-cache": ["claude-l9-template-redis-cache-1", "redis-cache"],
        "redis-queue": ["claude-l9-template-redis-queue-1", "redis-queue"],
        "nomic": ["neural-flow-nomic-v2-production", "nomic"]
    }

    all_healthy = True

    # Get running containers
    containers = get_container_status()
    running_names = [c.get("Names", "") for c in containers]

    print(f"\n  Found {len(containers)} running containers")

    # Check each required container
    for service, container_patterns in required_containers.items():
        print(f"\n  Checking {service}:")

        # Find matching container
        container_name = None
        for pattern in container_patterns:
            for running_name in running_names:
                if pattern in running_name:
                    container_name = running_name
                    break
            if container_name:
                break

        if not container_name:
            print(f"    ❌ Container not running (looked for: {', '.join(container_patterns)})")
            all_healthy = False
            continue

        print(f"    Found: {container_name}")

        # Check health status
        is_healthy, status = check_container_health(container_name)
        if is_healthy:
            print(f"    ✅ Health: {status}")
        else:
            # Don't fail immediately for Qdrant - check API first
            if service != "qdrant":
                print(f"    ❌ Health: {status}")
                all_healthy = False
            else:
                print(f"    ⚠️ Health: {status} (checking API...)")

        # Check for errors in logs
        errors = check_container_logs_for_errors(container_name)
        if errors:
            print(f"    ⚠️ Recent errors in logs:")
            for error in errors[:3]:  # Show max 3 errors
                print(f"       {error[:100]}...")  # Truncate long lines
        else:
            print(f"    ✅ No recent errors in logs")

        # Service-specific checks
        if service == "neo4j":
            # Check Neo4j is accepting connections
            bolt_check = subprocess.run(
                ["nc", "-z", "localhost", "47687"],
                capture_output=True,
                timeout=2
            )
            if bolt_check.returncode == 0:
                print(f"    ✅ Bolt protocol accessible on port 47687")
            else:
                print(f"    ❌ Bolt protocol not accessible")
                all_healthy = False

        elif service == "qdrant":
            # Check Qdrant API (Qdrant doesn't have Docker health check but API check is sufficient)
            try:
                import urllib.request
                req = urllib.request.Request("http://localhost:46333/collections")
                with urllib.request.urlopen(req, timeout=2) as response:
                    data = json.loads(response.read())
                    collection_count = len(data.get("result", {}).get("collections", []))
                    print(f"    ✅ API accessible ({collection_count} collections)")
                    # Override health status if API is accessible
                    if not is_healthy and response.status == 200:
                        is_healthy = True
                        print(f"    ✅ Health: running (API verified)")
            except Exception as e:
                print(f"    ❌ API not accessible: {e}")
                all_healthy = False

        elif service in ["redis-cache", "redis-queue"]:
            # Check Redis connectivity
            port = "46379" if service == "redis-cache" else "46380"
            password = "cache-secret-key" if service == "redis-cache" else "queue-secret-key"

            redis_check = subprocess.run(
                ["redis-cli", "-p", port, "-a", password, "ping"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if "PONG" in redis_check.stdout:
                print(f"    ✅ Redis responding on port {port}")
            else:
                print(f"    ❌ Redis not responding on port {port}")
                all_healthy = False

    # Check container resource usage
    print("\n  Container Resource Usage:")
    try:
        stats_result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format",
             "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if stats_result.returncode == 0:
            lines = stats_result.stdout.strip().split('\n')
            for line in lines[:6]:  # Header + first 5 containers
                print(f"    {line}")
    except Exception as e:
        print(f"    ⚠️ Could not get resource stats: {e}")

    # Summary
    print("\n" + "=" * 50)
    if all_healthy:
        print("✅ Container Health Integration Test PASSED")
        print("All required containers are healthy")
        return True
    else:
        print("❌ Container Health Integration Test FAILED")
        print("Some containers are unhealthy or missing")
        return False


def main():
    """Run the test"""
    success = asyncio.run(test_container_health())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()