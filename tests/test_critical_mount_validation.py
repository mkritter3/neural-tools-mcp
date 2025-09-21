#!/usr/bin/env python3
"""
ADR-63 Mount Validation Regression Test
CRITICAL: This test MUST be in CI/CD to prevent mount validation regression
"""

import asyncio
import tempfile
import docker
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'neural-tools', 'src'))
from servers.services.indexer_orchestrator import IndexerOrchestrator


async def test_mount_validation_regression():
    """
    Test that container mount paths are validated before reuse.
    This test would have caught the ADR-60 regression where containers
    were reused with wrong mount paths.
    """
    docker_client = docker.from_env()
    orchestrator = IndexerOrchestrator()
    orchestrator.docker_client = docker_client

    # Create two different paths
    path1 = tempfile.mkdtemp(prefix='mount-test-1-')
    path2 = tempfile.mkdtemp(prefix='mount-test-2-')

    print(f"Testing mount validation with paths:")
    print(f"  Path 1: {path1}")
    print(f"  Path 2: {path2}")

    try:
        # Create container with first path
        container1 = await orchestrator.ensure_indexer('mount-test', path1)

        # Verify mount
        c1_obj = docker_client.containers.get(container1)
        mounts1 = c1_obj.attrs['Mounts']
        mount1 = next((m['Source'] for m in mounts1 if m['Destination'] == '/workspace'), None)
        assert mount1 == path1, f"Container 1 mount wrong: {mount1} != {path1}"
        print(f"  ✅ Container 1 mount correct: {mount1}")

        # Request same project with DIFFERENT path
        container2 = await orchestrator.ensure_indexer('mount-test', path2)

        # CRITICAL ASSERTION: Must be different container
        assert container1 != container2, "REGRESSION: Container reused with wrong mount!"
        print(f"  ✅ Container recreated (not reused)")

        # Verify second container has correct mount
        c2_obj = docker_client.containers.get(container2)
        mounts2 = c2_obj.attrs['Mounts']
        mount2 = next((m['Source'] for m in mounts2 if m['Destination'] == '/workspace'), None)
        assert mount2 == path2, f"Container 2 mount wrong: {mount2} != {path2}"
        print(f"  ✅ Container 2 mount correct: {mount2}")

        # Verify first container was removed
        try:
            docker_client.containers.get(container1)
            assert False, "Old container should have been removed!"
        except docker.errors.NotFound:
            print(f"  ✅ Old container properly removed")

        print("\n✅ Mount validation test PASSED - Regression prevented!")
        return True

    except AssertionError as e:
        print(f"\n❌ Mount validation test FAILED: {e}")
        print("ADR-63 regression detected!")
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(path1, ignore_errors=True)
        shutil.rmtree(path2, ignore_errors=True)

        # Remove test containers - check both name and label
        for container in docker_client.containers.list(all=True):
            # Check by name or project label
            is_test_container = (
                'mount-test' in container.name or
                container.labels.get('com.l9.project') == 'mount-test'
            )

            if is_test_container:
                try:
                    print(f"  Cleaning up: {container.name}")
                    container.remove(force=True)
                except Exception as e:
                    print(f"  Warning: Could not remove {container.name}: {e}")


if __name__ == "__main__":
    result = asyncio.run(test_mount_validation_regression())
    sys.exit(0 if result else 1)
