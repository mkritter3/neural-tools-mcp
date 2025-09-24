#!/usr/bin/env python3
"""
Test Session Management - Verify Redis-based session counting
Tests that containers stop when last session ends

Author: L9 Engineering Team
Date: September 24, 2025
"""

import asyncio
import time
import docker
import os
import sys
import redis
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from servers.services.container_orchestrator import ContainerOrchestrator


async def test_session_management():
    """Test Redis-based session counting and cleanup."""
    print("\n" + "=" * 70)
    print("🧪 TESTING SESSION MANAGEMENT")
    print("=" * 70)

    # Set project directory
    os.environ["CLAUDE_PROJECT_DIR"] = str(Path(__file__).parent)
    project_hash = "fe813e2a1c17"  # First 12 chars of sha256 hash

    # Test Redis connection first
    print("\n1️⃣ Testing Redis connection...")
    try:
        r = redis.Redis(host='localhost', port=46379, password='cache-password', decode_responses=True)
        r.ping()
        print("   ✅ Redis accessible")
        # Clean up any old sessions
        r.delete(f"project:{project_hash}:sessions")
    except Exception as e:
        print(f"   ❌ Redis error: {e}")
        print("   Make sure Redis is running with password 'cache-password'")
        return False

    # Initialize first session
    print("\n2️⃣ Starting first MCP session...")
    session1 = ContainerOrchestrator()
    await session1.initialize()
    print(f"   Session 1 ID: {session1.session_id}")

    # Check session count
    count = r.scard(f"project:{project_hash}:sessions")
    print(f"   Active sessions: {count}")
    assert count == 1, "Should have 1 session"

    # Start second session (simulating second Claude window)
    print("\n3️⃣ Starting second MCP session...")
    session2 = ContainerOrchestrator()
    await session2.initialize()
    print(f"   Session 2 ID: {session2.session_id}")

    # Check session count
    count = r.scard(f"project:{project_hash}:sessions")
    print(f"   Active sessions: {count}")
    assert count == 2, "Should have 2 sessions"

    # Check containers are running
    print("\n4️⃣ Verifying containers are running...")
    docker_client = docker.from_env()
    containers = docker_client.containers.list(
        filters={"label": f"com.l9.project_hash={project_hash}"}
    )
    print(f"   Found {len(containers)} project containers")

    # Unregister first session
    print("\n5️⃣ Closing first session...")
    await session1._unregister_session()
    count = r.scard(f"project:{project_hash}:sessions")
    print(f"   Remaining sessions: {count}")
    assert count == 1, "Should have 1 session remaining"

    # Containers should still be running
    await asyncio.sleep(2)
    containers = docker_client.containers.list(
        filters={"label": f"com.l9.project_hash={project_hash}"}
    )
    print(f"   Containers running: {len(containers)} (should still be running)")

    # Unregister second session
    print("\n6️⃣ Closing last session...")
    await session2._unregister_session()
    count = r.scard(f"project:{project_hash}:sessions")
    print(f"   Remaining sessions: {count}")
    assert count == 0, "Should have 0 sessions"

    # Wait for containers to stop
    print("\n7️⃣ Waiting for containers to stop...")
    await asyncio.sleep(15)  # Give containers time to stop

    # Check containers are stopped
    containers = docker_client.containers.list(
        filters={"label": f"com.l9.project_hash={project_hash}"}
    )
    print(f"   Containers still running: {len(containers)}")

    # Results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)

    if len(containers) == 0:
        print("✅ PASS: Containers stopped after last session ended")
        success = True
    else:
        print("❌ FAIL: Containers still running after all sessions ended")
        for c in containers:
            print(f"   - {c.name}: {c.status}")
        success = False

    # Cleanup
    session1.accepting_requests = False
    session2.accepting_requests = False
    if session1.health_monitor_task:
        session1.health_monitor_task.cancel()
    if session2.health_monitor_task:
        session2.health_monitor_task.cancel()

    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(test_session_management())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)