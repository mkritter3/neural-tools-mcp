#!/usr/bin/env python3
"""
ADR-64: Integration tests for indexer mount validation
These tests require real Docker and are skipped when prerequisites are missing
"""

import os
import sys
import asyncio
import tempfile
import shutil
import pytest
import docker
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'neural-tools', 'src'))


def docker_available():
    """Check if Docker is available"""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except:
        return False


def indexer_image_available():
    """Check if the indexer image is available"""
    try:
        client = docker.from_env()
        client.images.get('l9-neural-indexer:production')
        return True
    except:
        return False


# Skip all tests in this file if Docker or image not available
pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not docker_available(), reason="Docker not available"),
    pytest.mark.skipif(not indexer_image_available(), reason="l9-neural-indexer:production image not found")
]


from servers.services.indexer_orchestrator import IndexerOrchestrator


class TestIndexerIntegration:
    """
    Integration tests that use real Docker containers
    These verify the complete flow with actual containers
    """

    @classmethod
    def setup_class(cls):
        """Initialize test environment"""
        if docker_available():
            cls.docker_client = docker.from_env()
            cls.test_dirs = []
        else:
            pytest.skip("Docker not available")

    @pytest.mark.asyncio
    async def test_real_mount_validation(self):
        """Test mount validation with real containers"""
        orchestrator = IndexerOrchestrator()
        orchestrator.docker_client = self.docker_client

        # Initialize orchestrator
        await orchestrator.initialize()

        # Create two different paths
        path1 = tempfile.mkdtemp(prefix='integration-mount-test-1-')
        path2 = tempfile.mkdtemp(prefix='integration-mount-test-2-')
        self.test_dirs.extend([path1, path2])

        try:
            # Create container with first path
            container1 = await orchestrator.ensure_indexer('integration-mount-test', path1)

            # Verify mount
            c1_obj = self.docker_client.containers.get(container1)
            mounts1 = c1_obj.attrs['Mounts']
            mount1 = next((m['Source'] for m in mounts1 if m['Destination'] == '/workspace'), None)
            assert mount1 == path1, f"Container 1 mount wrong: {mount1} != {path1}"

            # Request same project with DIFFERENT path
            container2 = await orchestrator.ensure_indexer('integration-mount-test', path2)

            # CRITICAL: Must be different container
            assert container1 != container2, "Container reused with wrong mount!"

            # Verify second container has correct mount
            c2_obj = self.docker_client.containers.get(container2)
            mounts2 = c2_obj.attrs['Mounts']
            mount2 = next((m['Source'] for m in mounts2 if m['Destination'] == '/workspace'), None)
            assert mount2 == path2, f"Container 2 mount wrong: {mount2} != {path2}"

            # Verify first container was removed
            try:
                self.docker_client.containers.get(container1)
                assert False, "Old container should have been removed!"
            except docker.errors.NotFound:
                pass  # Expected

        finally:
            # Cleanup containers
            for container in self.docker_client.containers.list(all=True):
                if 'integration-mount-test' in container.name or \
                   container.labels.get('com.l9.project') == 'integration-mount-test':
                    try:
                        container.remove(force=True)
                    except:
                        pass

    @pytest.mark.asyncio
    async def test_redis_unavailable_fallback(self):
        """Test that system works when Redis is unavailable"""
        orchestrator = IndexerOrchestrator()
        orchestrator.docker_client = self.docker_client

        # Set invalid Redis credentials to force fallback
        os.environ['REDIS_CACHE_HOST'] = 'invalid-host-that-does-not-exist'
        os.environ['REDIS_CACHE_PASSWORD'] = 'wrong-password'

        try:
            # Initialize should succeed despite Redis failure
            await orchestrator.initialize()

            # Redis should be None (fallback mode)
            assert orchestrator.redis_client is None

            # Should still be able to create containers
            test_path = tempfile.mkdtemp(prefix='integration-redis-fallback-')
            self.test_dirs.append(test_path)

            container_id = await orchestrator.ensure_indexer('redis-fallback-test', test_path)
            assert container_id is not None

            # Verify container was created
            container = self.docker_client.containers.get(container_id)
            assert container.status in ['running', 'created']

        finally:
            # Restore environment
            os.environ.pop('REDIS_CACHE_HOST', None)
            os.environ.pop('REDIS_CACHE_PASSWORD', None)

            # Cleanup
            for container in self.docker_client.containers.list(all=True):
                if 'redis-fallback-test' in container.name or \
                   container.labels.get('com.l9.project') == 'redis-fallback-test':
                    try:
                        container.remove(force=True)
                    except:
                        pass

    @pytest.mark.asyncio
    async def test_concurrent_real_containers(self):
        """Test concurrent container creation with real Docker"""
        orchestrator = IndexerOrchestrator()
        orchestrator.docker_client = self.docker_client
        await orchestrator.initialize()

        paths = []
        for i in range(3):
            path = tempfile.mkdtemp(prefix=f'integration-concurrent-{i}-')
            paths.append(path)
            self.test_dirs.append(path)

        try:
            # Concurrent requests
            tasks = [
                orchestrator.ensure_indexer(f'concurrent-test-{i}', path)
                for i, path in enumerate(paths)
            ]

            containers = await asyncio.gather(*tasks)

            # All should be different containers
            assert len(set(containers)) == 3

            # Each should have correct mount
            for container_id, expected_path in zip(containers, paths):
                container = self.docker_client.containers.get(container_id)
                mounts = container.attrs['Mounts']
                mount_source = next((m['Source'] for m in mounts if m['Destination'] == '/workspace'), None)
                assert mount_source == expected_path

        finally:
            # Cleanup
            for container in self.docker_client.containers.list(all=True):
                if 'concurrent-test' in container.name or \
                   (container.labels.get('com.l9.project', '').startswith('concurrent-test')):
                    try:
                        container.remove(force=True)
                    except:
                        pass

    @classmethod
    def teardown_class(cls):
        """Clean up test resources"""
        if hasattr(cls, 'test_dirs'):
            for test_dir in cls.test_dirs:
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir, ignore_errors=True)

        # Clean up any remaining test containers
        if docker_available():
            client = docker.from_env()
            for container in client.containers.list(all=True):
                if any(test_name in container.name for test_name in ['integration-', 'mount-test', 'concurrent-test', 'redis-fallback']):
                    try:
                        container.remove(force=True)
                    except:
                        pass


if __name__ == "__main__":
    if not docker_available():
        print("⚠️  Docker not available - skipping integration tests")
        print("   These tests require Docker daemon to be running")
        sys.exit(0)

    if not indexer_image_available():
        print("⚠️  l9-neural-indexer:production image not found")
        print("   Build the image with: docker build -f docker/Dockerfile.indexer -t l9-neural-indexer:production .")
        sys.exit(0)

    print("\n" + "="*60)
    print("INDEXER MOUNT VALIDATION INTEGRATION TESTS")
    print("Testing with real Docker containers")
    print("="*60)

    # Run with pytest
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )

    if result.returncode == 0:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed")

    sys.exit(result.returncode)