#!/usr/bin/env python3
"""
Test Indexer Per-Project Containers
Verify that each project gets its own indexer container

Author: L9 Engineering Team
Date: September 24, 2025
"""

import asyncio
import docker
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from servers.services.indexer_orchestrator import IndexerOrchestrator


async def test_indexer_per_project():
    """Test that each project gets its own indexer container."""
    print("\n" + "=" * 70)
    print("🧪 TESTING INDEXER PER-PROJECT CONTAINERS")
    print("=" * 70)

    docker_client = docker.from_env()

    # Create two temporary projects
    temp_dir = tempfile.gettempdir()
    project1_path = Path(temp_dir) / "test_project_alpha"
    project2_path = Path(temp_dir) / "test_project_beta"

    # Clean up any existing test directories
    for path in [project1_path, project2_path]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        (path / "test.py").write_text("print('test')")

    print(f"\n1️⃣ Created test projects:")
    print(f"   Project Alpha: {project1_path}")
    print(f"   Project Beta: {project2_path}")

    # Initialize orchestrator
    orchestrator = IndexerOrchestrator()
    await orchestrator.initialize()

    # Start indexer for project 1
    print("\n2️⃣ Starting indexer for Project Alpha...")
    container1_id = await orchestrator.ensure_indexer("test_project_alpha", str(project1_path))
    print(f"   Container ID: {container1_id[:12]}")

    # Start indexer for project 2
    print("\n3️⃣ Starting indexer for Project Beta...")
    container2_id = await orchestrator.ensure_indexer("test_project_beta", str(project2_path))
    print(f"   Container ID: {container2_id[:12]}")

    # Verify they are different containers
    print("\n4️⃣ Verifying separate containers...")
    if container1_id != container2_id:
        print(f"   ✅ Different containers created")
    else:
        print(f"   ❌ Same container reused (should be different)")

    # Check Docker containers
    containers = docker_client.containers.list(filters={"label": "com.l9.managed=true"})
    indexer_containers = [c for c in containers if "indexer" in c.name]

    print(f"\n5️⃣ Active indexer containers: {len(indexer_containers)}")
    for container in indexer_containers:
        project = container.labels.get("com.l9.project", "unknown")
        mounts = container.attrs.get('Mounts', [])
        mount_path = mounts[0].get('Source', 'none') if mounts else 'none'
        print(f"   - {container.name}: project={project}, mount={mount_path}")

    # Call ensure_indexer again for project 1 - should reuse
    print("\n6️⃣ Re-requesting indexer for Project Alpha...")
    container1_id_2 = await orchestrator.ensure_indexer("test_project_alpha", str(project1_path))
    if container1_id == container1_id_2:
        print(f"   ✅ Container reused (same ID)")
    else:
        print(f"   ❌ New container created (should reuse)")

    # Results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)

    success = True
    if container1_id != container2_id:
        print("✅ Each project has its own indexer container")
    else:
        print("❌ Projects sharing same container")
        success = False

    if len(indexer_containers) >= 2:
        print(f"✅ Found {len(indexer_containers)} indexer containers")
    else:
        print(f"❌ Only {len(indexer_containers)} indexer container(s) found")
        success = False

    # Cleanup
    print("\n7️⃣ Cleaning up test containers...")
    for container in indexer_containers:
        if "test_project" in container.name:
            try:
                container.stop(timeout=5)
                container.remove()
                print(f"   Removed {container.name}")
            except Exception as e:
                print(f"   Failed to remove {container.name}: {e}")

    # Remove test directories
    for path in [project1_path, project2_path]:
        if path.exists():
            shutil.rmtree(path)

    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(test_indexer_per_project())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)