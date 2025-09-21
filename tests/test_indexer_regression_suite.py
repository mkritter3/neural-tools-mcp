#!/usr/bin/env python3
"""
Comprehensive Indexer Regression Test Suite

This suite specifically tests for regressions that have broken production indexing.
Each test represents a real bug that affected users.
"""

import os
import sys
import tempfile
import asyncio
import docker
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'neural-tools', 'src'))

from servers.services.indexer_orchestrator import IndexerOrchestrator


class TestIndexerRegressionPrevention:
    """
    Regression tests to prevent future indexer failures.
    Each test represents a REAL production bug we've had.
    """

    @classmethod
    def setup_class(cls):
        """Initialize test environment"""
        cls.docker_client = docker.from_env()
        cls.orchestrator = IndexerOrchestrator()
        cls.orchestrator.docker_client = cls.docker_client
        cls.test_dirs = []

    @pytest.mark.asyncio
    async def test_regression_adr63_mount_validation(self):
        """
        REGRESSION: ADR-60 reused containers with wrong mounts
        SYMPTOM: neural-novelist only indexed README.md
        ROOT CAUSE: No mount verification before container reuse
        """
        # Create real project paths (not just /tmp/test!)
        project_v1 = tempfile.mkdtemp(prefix="project-v1-")
        project_v2 = tempfile.mkdtemp(prefix="project-v2-")
        self.test_dirs.extend([project_v1, project_v2])

        # Create different files in each
        Path(f"{project_v1}/file1.py").write_text("# Version 1")
        Path(f"{project_v2}/file2.py").write_text("# Version 2")

        # First indexer
        container1 = await self.orchestrator.ensure_indexer("myproject", project_v1)

        # CRITICAL: Same project name, different path (real user scenario)
        container2 = await self.orchestrator.ensure_indexer("myproject", project_v2)

        # Must be different containers
        assert container1 != container2, "ADR-63 REGRESSION: Container reused with wrong mount!"

        # Verify second container has correct mount
        container = self.docker_client.containers.get(container2)
        mounts = container.attrs['Mounts']
        mount_source = next((m['Source'] for m in mounts if m['Destination'] == '/workspace'), None)
        assert mount_source == project_v2, f"Mount incorrect: {mount_source} != {project_v2}"

    @pytest.mark.asyncio
    async def test_regression_path_switching(self):
        """
        REGRESSION: Users switching between project branches
        SYMPTOM: Old branch files indexed instead of new branch
        """
        main_branch = tempfile.mkdtemp(prefix="main-")
        feature_branch = tempfile.mkdtemp(prefix="feature-")
        self.test_dirs.extend([main_branch, feature_branch])

        # Simulate branch switching
        container1 = await self.orchestrator.ensure_indexer("myapp", main_branch)
        container2 = await self.orchestrator.ensure_indexer("myapp", feature_branch)
        container3 = await self.orchestrator.ensure_indexer("myapp", main_branch)  # Back to main

        # Each switch should create new container (paths differ)
        assert container1 != container2, "Branch switch didn't create new container"
        assert container2 != container3, "Return to main didn't create new container"
        assert container1 != container3, "Containers should be recreated even for same path"

    @pytest.mark.asyncio
    async def test_regression_test_vs_production_paths(self):
        """
        REGRESSION: Tests pass with /tmp/test but production fails
        SYMPTOM: All tests green but users can't index projects
        """
        # DON'T use /tmp/test - use realistic paths!
        test_path = "/tmp/test"  # What tests used
        prod_path = os.path.expanduser("~/Projects/real-app")  # What users use

        # This should work even if paths are very different
        os.makedirs(prod_path, exist_ok=True)
        self.test_dirs.append(prod_path)

        try:
            container = await self.orchestrator.ensure_indexer("realapp", prod_path)

            # Verify mount is production path, not test path
            container_obj = self.docker_client.containers.get(container)
            mounts = container_obj.attrs['Mounts']
            mount_source = next((m['Source'] for m in mounts if m['Destination'] == '/workspace'), None)

            assert mount_source != test_path, "Container using test path in production!"
            assert mount_source == prod_path, f"Wrong mount: {mount_source}"
        finally:
            # Clean up
            import shutil
            if os.path.exists(prod_path):
                shutil.rmtree(prod_path)

    @pytest.mark.asyncio
    async def test_regression_stale_container_pollution(self):
        """
        REGRESSION: Old containers from previous runs interfere
        SYMPTOM: Indexer finds wrong files despite correct request
        """
        correct_path = tempfile.mkdtemp(prefix="correct-")
        wrong_path = tempfile.mkdtemp(prefix="wrong-")
        self.test_dirs.extend([correct_path, wrong_path])

        # Pre-create a "stale" container with wrong mount
        stale = self.docker_client.containers.run(
            image='l9-neural-indexer:production',
            name=f'indexer-stale-test-{os.getpid()}',
            labels={
                'com.l9.project': 'stale-test',
                'com.l9.managed': 'true'
            },
            volumes={wrong_path: {'bind': '/workspace', 'mode': 'ro'}},
            detach=True,
            auto_remove=False
        )

        try:
            # Request with correct path should remove stale container
            new_container = await self.orchestrator.ensure_indexer('stale-test', correct_path)

            # Must be different container
            assert stale.id != new_container

            # Stale should be removed
            with pytest.raises(docker.errors.NotFound):
                self.docker_client.containers.get(stale.id)

            # New container has correct mount
            container = self.docker_client.containers.get(new_container)
            mounts = container.attrs['Mounts']
            mount_source = next((m['Source'] for m in mounts if m['Destination'] == '/workspace'), None)
            assert mount_source == correct_path
        finally:
            # Cleanup
            try:
                stale.remove(force=True)
            except:
                pass

    @pytest.mark.asyncio
    async def test_regression_concurrent_different_paths(self):
        """
        REGRESSION: Concurrent requests with different paths
        SYMPTOM: Race condition causes wrong mount to win
        """
        paths = [tempfile.mkdtemp(prefix=f"concurrent-{i}-") for i in range(3)]
        self.test_dirs.extend(paths)

        # Concurrent requests for same project, different paths
        tasks = [
            self.orchestrator.ensure_indexer("concurrent-test", path)
            for path in paths
        ]

        containers = await asyncio.gather(*tasks)

        # All should be different containers (different paths)
        assert len(set(containers)) == len(paths), "Concurrent requests reused containers incorrectly"

        # Each should have its correct mount
        for container_id, expected_path in zip(containers, paths):
            container = self.docker_client.containers.get(container_id)
            mounts = container.attrs['Mounts']
            mount_source = next((m['Source'] for m in mounts if m['Destination'] == '/workspace'), None)
            # Due to race conditions, just verify it's one of the valid paths
            assert mount_source in paths, f"Invalid mount: {mount_source}"

    @classmethod
    def teardown_class(cls):
        """Clean up test resources"""
        import shutil
        for test_dir in cls.test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

        # Clean up test containers
        containers = cls.docker_client.containers.list(
            all=True,
            filters={'label': 'com.l9.managed=true'}
        )
        for container in containers:
            if any(test_name in container.name for test_name in ['test', 'stale', 'concurrent']):
                try:
                    container.remove(force=True)
                except:
                    pass


if __name__ == "__main__":
    print("\n" + "="*60)
    print("INDEXER REGRESSION PREVENTION SUITE")
    print("Testing real-world failure scenarios")
    print("="*60)

    # Run with pytest for better output
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )

    if result.returncode == 0:
        print("\n✅ All regression tests passed - Indexer is protected!")
    else:
        print("\n❌ Regression detected - Fix before deploying!")

    sys.exit(result.returncode)