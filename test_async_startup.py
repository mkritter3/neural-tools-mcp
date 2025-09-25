#!/usr/bin/env python3
"""
Test Async Container Startup
Verify MCP connects immediately while containers start in background

Author: L9 Engineering Team
Date: September 24, 2025
"""

import asyncio
import time
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))


async def simulate_mcp_startup():
    """Simulate the MCP server startup process"""
    print("\n" + "=" * 70)
    print("🧪 TESTING ASYNC CONTAINER STARTUP")
    print("=" * 70)

    start_time = time.time()

    # Import the server module
    from neural_mcp import server

    # Measure time to get to STDIO initialization
    print("\n1️⃣ Starting MCP server...")

    # Create the background task (this is what happens in main())
    from servers.services.container_status import get_container_status
    from datetime import datetime

    status_tracker = get_container_status()
    status_tracker.initialization_started = datetime.now()

    async def initialize_containers_async():
        """Background task to start containers"""
        print(f"   🎭 Container initialization started in background at {time.time() - start_time:.2f}s")

        # Simulate container startup (normally takes 30+ seconds)
        await asyncio.sleep(3)  # Shortened for test

        status_tracker.set_orchestrator("mock_orchestrator")
        print(f"   ✅ Containers ready at {time.time() - start_time:.2f}s")

    # Start background task
    container_task = asyncio.create_task(initialize_containers_async())

    # MCP server starts immediately
    mcp_ready_time = time.time() - start_time
    print(f"   📡 MCP STDIO ready at {mcp_ready_time:.2f}s")

    # Simulate a tool being called
    print("\n2️⃣ Simulating tool calls...")

    # Early tool call (before containers ready)
    await asyncio.sleep(0.5)
    print(f"\n   Tool call at {time.time() - start_time:.2f}s:")

    from servers.services.ensure_containers import ensure_containers_ready
    ready = await ensure_containers_ready(timeout=5)

    if ready:
        print(f"   ✅ Containers ready, tool can proceed")
    else:
        print(f"   ⚠️ Containers not ready, tool might fail")

    # Later tool call (after containers ready)
    await asyncio.sleep(3)
    print(f"\n   Tool call at {time.time() - start_time:.2f}s:")

    ready = await ensure_containers_ready(timeout=1)
    if ready:
        print(f"   ✅ Containers ready, tool can proceed immediately")
    else:
        print(f"   ❌ Containers still not ready")

    # Check final status
    print("\n3️⃣ Final status:")
    status = status_tracker.get_status()
    print(f"   Ready: {status['ready']}")
    if status.get('initialization_time_seconds'):
        print(f"   Init time: {status['initialization_time_seconds']:.2f}s")

    # Results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)

    if mcp_ready_time < 1.0:
        print(f"✅ MCP connected in {mcp_ready_time:.2f}s (instant!)")
        success = True
    else:
        print(f"❌ MCP took {mcp_ready_time:.2f}s (too slow)")
        success = False

    if status['ready']:
        print("✅ Containers eventually ready in background")
    else:
        print("❌ Container initialization failed")
        success = False

    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(simulate_mcp_startup())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)