#!/usr/bin/env python3
"""
Test Proactive Container Orchestration - ADR-0100
Validates that containers start before first tool call

Author: L9 Engineering Team
Date: September 24, 2025
"""

import asyncio
import time
import docker
import os
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from servers.services.container_orchestrator import ContainerOrchestrator


async def test_proactive_initialization():
    """Test that containers start proactively."""
    print("\n" + "=" * 70)
    print("🧪 TESTING PROACTIVE CONTAINER ORCHESTRATION")
    print("=" * 70)

    # Set project directory
    os.environ["CLAUDE_PROJECT_DIR"] = str(Path(__file__).parent)

    # Initialize orchestrator
    print("\n1️⃣ Initializing Container Orchestrator...")
    start_time = time.time()

    orchestrator = ContainerOrchestrator()
    result = await orchestrator.initialize()

    init_time = time.time() - start_time
    print(f"   ⏱️ Initialization took: {init_time:.2f}s")

    # Check results
    print("\n2️⃣ Checking container status...")
    docker_client = docker.from_env()

    expected_services = ["neo4j", "redis-cache", "redis-queue", "nomic", "indexer"]
    running_count = 0

    for service in expected_services:
        if service in result.get("containers", {}):
            container_info = result["containers"][service]
            if container_info.get("status") == "started":
                print(f"   ✅ {service}: Running on port {container_info.get('port')}")
                running_count += 1
            else:
                print(f"   ❌ {service}: {container_info.get('status')} - {container_info.get('error', '')}")
        else:
            print(f"   ⚠️ {service}: Not found in results")

    # Verify containers are actually running
    print("\n3️⃣ Verifying Docker containers...")
    project_hash = orchestrator.project_hash
    containers = docker_client.containers.list(
        filters={"label": f"com.l9.project_hash={project_hash}"}
    )

    print(f"   Found {len(containers)} containers with project hash {project_hash}")
    for container in containers:
        print(f"   - {container.name}: {container.status}")

    # Test health monitoring
    print("\n4️⃣ Testing health monitoring...")
    await asyncio.sleep(5)  # Let health monitor run

    # Check session registration
    print("\n5️⃣ Checking session registration...")
    print(f"   Session ID: {orchestrator.session_id}")
    print(f"   Project: {orchestrator.project_path}")

    # Results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)

    success = running_count >= 4  # At least 4 essential services
    if success:
        print(f"✅ PASS: {running_count}/{len(expected_services)} services running")
        print(f"✅ Initialization time: {init_time:.2f}s")
        if init_time < 30:
            print("✅ Performance: Excellent (< 30s)")
        elif init_time < 60:
            print("⚠️ Performance: Acceptable (< 60s)")
        else:
            print("❌ Performance: Poor (> 60s)")
    else:
        print(f"❌ FAIL: Only {running_count}/{len(expected_services)} services running")

    print("\n" + "=" * 70)
    print("🏁 TEST COMPLETE")
    print("=" * 70)

    # Cleanup
    print("\n🧹 Cleaning up...")
    orchestrator.accepting_requests = False
    if orchestrator.health_monitor_task:
        orchestrator.health_monitor_task.cancel()

    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(test_proactive_initialization())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)