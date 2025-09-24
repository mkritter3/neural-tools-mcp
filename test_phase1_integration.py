#!/usr/bin/env python3
"""
ADR-0098 Phase 1 Integration Test
Test enhanced labels with actual IndexerOrchestrator
"""

import asyncio
import docker
import sys
import os
import time
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from servers.services.indexer_orchestrator import IndexerOrchestrator


async def test_indexer_with_enhanced_labels():
    """Test that IndexerOrchestrator creates containers with enhanced labels"""
    print("=" * 60)
    print("ADR-0098 Phase 1: IndexerOrchestrator Integration Test")
    print("=" * 60)

    # Create and initialize orchestrator
    orchestrator = IndexerOrchestrator()
    await orchestrator.initialize()
    client = docker.from_env()

    # Use current directory as test project
    test_project = f"phase1-test-{int(time.time())}"
    project_path = os.path.abspath(".")

    try:
        print(f"\n📦 Creating indexer for project: {test_project}")
        print(f"   Path: {project_path}")

        # Ensure indexer (this should create container with enhanced labels)
        container_id = await orchestrator.ensure_indexer(test_project, project_path)
        print(f"✅ Created container: {container_id[:12]}")

        # Get container and check labels
        container = client.containers.get(container_id)
        labels = container.labels

        print("\n🏷️ Container Labels:")
        for key, value in sorted(labels.items()):
            if key.startswith('com.l9.'):
                print(f"   {key}: {value}")

        # Verify Phase 1 labels are present
        required_phase1_labels = [
            'com.l9.project_hash',
            'com.l9.port',
            'com.l9.project_path'
        ]

        missing = []
        for label in required_phase1_labels:
            if label not in labels:
                missing.append(label)

        if missing:
            print(f"\n❌ Missing Phase 1 labels: {missing}")
            return False

        print("\n✅ All Phase 1 enhanced labels present!")

        # Verify label accuracy
        import hashlib
        expected_hash = hashlib.sha256(project_path.encode()).hexdigest()[:12]
        actual_hash = labels.get('com.l9.project_hash', '')

        if actual_hash != expected_hash:
            print(f"❌ Hash mismatch: expected {expected_hash}, got {actual_hash}")
            return False
        print(f"✅ Project hash correct: {actual_hash}")

        # Verify port label
        port_label = labels.get('com.l9.port', '')
        if not port_label:
            print("❌ Port label is empty")
            return False

        try:
            port_num = int(port_label)
            if 48100 <= port_num <= 48199:
                print(f"✅ Port in correct range: {port_num}")
            else:
                print(f"⚠️ Port outside expected range: {port_num}")
        except ValueError:
            print(f"❌ Invalid port value: {port_label}")
            return False

        # Verify path label
        path_label = labels.get('com.l9.project_path', '')
        if path_label != project_path:
            print(f"❌ Path mismatch: expected {project_path}, got {path_label}")
            return False
        print(f"✅ Project path correct: {path_label}")

        print("\n" + "=" * 60)
        print("✅ Phase 1 Implementation Verified!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            await orchestrator.stop_indexer(test_project)
        except:
            pass

        # Also try direct cleanup
        try:
            containers = client.containers.list(
                all=True,
                filters={'label': f'com.l9.project={test_project}'}
            )
            for container in containers:
                print(f"   Removing container: {container.name}")
                container.stop()
                container.remove()
        except:
            pass


async def main():
    success = await test_indexer_with_enhanced_labels()

    if success:
        print("\n📋 Next Steps:")
        print("1. Monitor in production for 1 week")
        print("2. Verify 100% of new containers have enhanced labels")
        print("3. Check divergence rate remains < 5%")
        print("4. Then proceed to Phase 2")
    else:
        print("\n❌ Phase 1 implementation needs fixes")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)