#!/usr/bin/env python3
"""
ADR-63: Mount Validation Priority Tests

These 3 tests would have caught the critical regression where containers
are reused with wrong mount paths, causing indexers to fail silently.
"""

import os
import tempfile
import shutil
import asyncio
import docker
import pytest
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'neural-tools', 'src'))

from servers.services.indexer_orchestrator import IndexerOrchestrator


class TestADR63MountValidation:
    """Critical tests that must pass to prevent mount validation regression"""

    @classmethod
    def setup_class(cls):
        """Initialize test environment"""
        cls.docker_client = docker.from_env()
        cls.orchestrator = IndexerOrchestrator()
        cls.orchestrator.docker_client = cls.docker_client  # Set docker client explicitly
        cls.test_dirs = []

    @classmethod
    def teardown_class(cls):
        """Clean up test resources"""
        # Remove test directories
        for test_dir in cls.test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

        # Clean up test containers - use both test label and project patterns
        print("\nCleaning up test containers...")

        # Method 1: Look for test label (new containers)
        try:
            test_containers = cls.docker_client.containers.list(
                all=True,
                filters={'label': 'com.l9.test=true'}
            )
            for container in test_containers:
                try:
                    print(f"  Removing (by label): {container.name}")
                    container.remove(force=True)
                except Exception as e:
                    print(f"  Warning: Could not remove {container.name}: {e}")
        except Exception as e:
            print(f"  Warning: Error finding containers by label: {e}")

        # Method 2: Look for test patterns (legacy containers)
        test_project_patterns = [
            'test-mount-change',
            'test-stale-mount',
            'test-env-change',
            'mount-test',
            'test-',
            'adr63-',
            'adr60-'
        ]

        all_containers = cls.docker_client.containers.list(all=True)
        for container in all_containers:
            # Check if it's a test container by project label or name
            labels = container.labels
            project = labels.get('com.l9.project', '')
            name = container.name

            # Remove if it matches any test pattern
            is_test = any(pattern in project or pattern in name for pattern in test_project_patterns)

            if is_test and 'com.l9.test' not in labels:  # Don't double-remove
                try:
                    print(f"  Removing (by pattern): {container.name}")
                    container.remove(force=True)
                except Exception as e:
                    print(f"  Warning: Could not remove {container.name}: {e}")

    def create_test_dir(self, name):
        """Create a test directory with sample files"""
        test_dir = tempfile.mkdtemp(prefix=f"adr63-{name}-")
        self.test_dirs.append(test_dir)

        # Create sample files to index
        Path(f"{test_dir}/app.py").write_text("print('test file')")
        Path(f"{test_dir}/README.md").write_text(f"# Test Project {name}")

        return test_dir

    @pytest.mark.asyncio
    async def test_priority1_mount_changes_force_recreation(self):
        """
        Priority 1: Container recreation on path change
        This test would have caught the ADR-60 regression
        """
        project_name = "test-mount-change"

        # Create two different test directories
        path_a = self.create_test_dir("path-a")
        path_b = self.create_test_dir("path-b")

        print(f"\n[TEST] Priority 1: Mount change forces recreation")
        print(f"  Path A: {path_a}")
        print(f"  Path B: {path_b}")

        # Create container for path A
        container1_id = await self.orchestrator.ensure_indexer(project_name, path_a)
        print(f"  Container 1: {container1_id[:12]}")

        # Verify mount for container 1
        container1 = self.docker_client.containers.get(container1_id)
        mounts1 = container1.attrs.get('Mounts', [])
        mount1_source = next((m['Source'] for m in mounts1 if m['Destination'] == '/workspace'), None)
        assert mount1_source == path_a, f"Container 1 mount wrong: {mount1_source} != {path_a}"

        # Request same project with different path
        container2_id = await self.orchestrator.ensure_indexer(project_name, path_b)
        print(f"  Container 2: {container2_id[:12]}")

        # CRITICAL ASSERTION: Must be different containers
        assert container1_id != container2_id, "Container was reused with wrong mount!"

        # Verify mount for container 2
        container2 = self.docker_client.containers.get(container2_id)
        mounts2 = container2.attrs.get('Mounts', [])
        mount2_source = next((m['Source'] for m in mounts2 if m['Destination'] == '/workspace'), None)
        assert mount2_source == path_b, f"Container 2 mount wrong: {mount2_source} != {path_b}"

        # Verify old container was removed
        with pytest.raises(docker.errors.NotFound):
            self.docker_client.containers.get(container1_id)

        print("  ✅ PASS: Container recreated with correct mount")

    @pytest.mark.asyncio
    async def test_priority2_stale_container_wrong_mount(self):
        """
        Priority 2: Replace containers with wrong mounts
        Tests handling of pre-existing stale containers
        """
        project_name = "test-stale-mount"

        # Create test directories
        wrong_path = self.create_test_dir("wrong")
        correct_path = self.create_test_dir("correct")

        print(f"\n[TEST] Priority 2: Stale container with wrong mount")
        print(f"  Wrong path: {wrong_path}")
        print(f"  Correct path: {correct_path}")

        # Pre-create container with wrong mount (simulating stale container)
        stale_container = self.docker_client.containers.run(
            image='l9-neural-indexer:production',
            name=f'indexer-{project_name}-stale-test',
            labels={
                'com.l9.project': project_name,
                'com.l9.managed': 'true',
                'com.l9.test': 'true'
            },
            volumes={wrong_path: {'bind': '/workspace', 'mode': 'ro'}},
            environment={
                'PROJECT_NAME': project_name,
                'NEO4J_URI': 'bolt://localhost:47687',
                'QDRANT_URL': 'http://localhost:46333'
            },
            detach=True,
            remove=False
        )
        print(f"  Stale container: {stale_container.id[:12]} (wrong mount)")

        # Now request with correct path - should detect and replace
        new_container_id = await self.orchestrator.ensure_indexer(project_name, correct_path)
        print(f"  New container: {new_container_id[:12]}")

        # Verify it's a different container
        assert stale_container.id != new_container_id, "Stale container with wrong mount was reused!"

        # Verify new container has correct mount
        new_container = self.docker_client.containers.get(new_container_id)
        mounts = new_container.attrs.get('Mounts', [])
        mount_source = next((m['Source'] for m in mounts if m['Destination'] == '/workspace'), None)
        assert mount_source == correct_path, f"New container mount wrong: {mount_source} != {correct_path}"

        # Verify stale container was removed
        with pytest.raises(docker.errors.NotFound):
            self.docker_client.containers.get(stale_container.id)

        print("  ✅ PASS: Stale container replaced with correct mount")

    @pytest.mark.asyncio
    async def test_priority3_env_var_change_forces_recreation(self):
        """
        Priority 3: Environment variable changes trigger recreation
        Tests configuration change detection
        """
        project_name = "test-env-change"
        test_path = self.create_test_dir("env-test")

        print(f"\n[TEST] Priority 3: Environment variable change forces recreation")
        print(f"  Test path: {test_path}")

        # Set initial environment
        original_password = os.environ.get('NEO4J_PASSWORD', 'graphrag-password')
        os.environ['NEO4J_PASSWORD'] = 'password-v1'

        # Create container with first password
        container1_id = await self.orchestrator.ensure_indexer(project_name, test_path)
        print(f"  Container 1 (password-v1): {container1_id[:12]}")

        # Verify environment in container 1
        container1 = self.docker_client.containers.get(container1_id)
        env1 = container1.attrs['Config']['Env']
        assert any('NEO4J_PASSWORD=password-v1' in e for e in env1), "Container 1 missing password-v1"

        # Change environment variable
        os.environ['NEO4J_PASSWORD'] = 'password-v2'

        # Request again - should detect config change
        container2_id = await self.orchestrator.ensure_indexer(project_name, test_path)
        print(f"  Container 2 (password-v2): {container2_id[:12]}")

        # Should be different container (config change)
        assert container1_id != container2_id, "Container reused despite env var change!"

        # Verify environment in container 2
        container2 = self.docker_client.containers.get(container2_id)
        env2 = container2.attrs['Config']['Env']
        assert any('NEO4J_PASSWORD=password-v2' in e for e in env2), "Container 2 missing password-v2"

        # Restore original environment
        if original_password:
            os.environ['NEO4J_PASSWORD'] = original_password
        else:
            del os.environ['NEO4J_PASSWORD']

        print("  ✅ PASS: Environment change triggered recreation")


if __name__ == "__main__":
    """Run the priority tests"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("ADR-63 MOUNT VALIDATION PRIORITY TESTS")
    print("These 3 tests MUST PASS to prevent the regression")
    print("="*60)

    test = TestADR63MountValidation()
    test.setup_class()

    try:
        # Run the 3 priority tests
        asyncio.run(test.test_priority1_mount_changes_force_recreation())
        asyncio.run(test.test_priority2_stale_container_wrong_mount())
        asyncio.run(test.test_priority3_env_var_change_forces_recreation())

        print("\n" + "="*60)
        print("✅ ALL PRIORITY TESTS PASSED")
        print("Mount validation regression is prevented")
        print("="*60 + "\n")

    except AssertionError as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED - REGRESSION DETECTED")
        print(str(e))
        print("="*60 + "\n")
        raise

    finally:
        test.teardown_class()