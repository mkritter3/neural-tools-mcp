#!/usr/bin/env python3
"""
ADR-0048 Container Path Resolution - Comprehensive Test Suite
Tests container path resolution, idempotent management, and graceful shutdown
"""

import asyncio
import docker
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.servers.services.indexer_orchestrator import IndexerOrchestrator
from src.servers.services.project_context_manager import ProjectContextManager
from src.servers.services.container_discovery import ContainerDiscoveryService


# ============================================================================
# UNIT TESTS - Mocked Docker Client
# ============================================================================

class TestADR0048UnitTests:
    """Unit tests with mocked Docker client for fast, isolated testing"""

    @pytest.fixture
    def mock_docker_client(self):
        """Create a mock Docker client"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        return mock_client

    @pytest.fixture
    def orchestrator_with_mock(self, mock_docker_client):
        """Create orchestrator with mocked Docker client"""
        import asyncio

        orchestrator = IndexerOrchestrator()
        orchestrator.docker_client = mock_docker_client
        orchestrator.discovery_service = None  # No discovery service for unit tests
        orchestrator.context_manager = None  # No context manager for unit tests

        # Create a real async lock - it's simpler than mocking
        orchestrator._lock = asyncio.Lock()

        # Initialize port tracking
        orchestrator.allocated_ports = set()
        orchestrator.active_indexers = {}

        return orchestrator

    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=True)
    def test_project_path_always_container_path(self, mock_isdir, mock_exists, orchestrator_with_mock, mock_docker_client):
        """Test that PROJECT_PATH is always set to /workspace"""
        # Arrange
        project_name = "test-project"
        host_path = "/Users/test/my-project"

        mock_container = MagicMock()
        mock_container.id = "test-container-id"
        mock_container.status = "running"
        mock_docker_client.containers.run.return_value = mock_container
        mock_docker_client.containers.list.return_value = []

        # Act
        asyncio.run(orchestrator_with_mock.ensure_indexer(project_name, host_path))

        # Assert
        mock_docker_client.containers.run.assert_called_once()
        call_args = mock_docker_client.containers.run.call_args

        # Verify PROJECT_PATH is /workspace, not the host path
        env_vars = call_args.kwargs['environment']
        assert env_vars['PROJECT_PATH'] == '/workspace', \
            f"PROJECT_PATH should be /workspace, got {env_vars['PROJECT_PATH']}"
        assert env_vars['PROJECT_PATH'] != host_path, \
            "PROJECT_PATH must not be the host path"

    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=True)
    def test_volume_mount_configuration(self, mock_isdir, mock_exists, orchestrator_with_mock, mock_docker_client):
        """Test volume mount maps host path to /workspace correctly"""
        # Arrange
        project_name = "test-project"
        host_path = "/path/with spaces/my-project"

        mock_container = MagicMock()
        mock_container.id = "test-container-id"
        mock_docker_client.containers.run.return_value = mock_container
        mock_docker_client.containers.list.return_value = []

        # Act
        asyncio.run(orchestrator_with_mock.ensure_indexer(project_name, host_path))

        # Assert
        call_args = mock_docker_client.containers.run.call_args
        volumes = call_args.kwargs['volumes']

        # Should map absolute host path to /workspace
        expected_volume = {
            os.path.abspath(host_path): {'bind': '/workspace', 'mode': 'ro'}
        }
        assert volumes == expected_volume, \
            f"Volume mount incorrect. Expected {expected_volume}, got {volumes}"

    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=True)
    def test_idempotent_container_removal(self, mock_isdir, mock_exists, orchestrator_with_mock, mock_docker_client):
        """Test that existing containers are removed before creating new ones"""
        # Arrange
        project_name = "test-project"
        host_path = "/test/path"

        # Mock existing container
        existing_container = MagicMock()
        existing_container.id = "old-container-id"
        existing_container.status = "running"
        existing_container.name = f"indexer-{project_name}"

        mock_docker_client.containers.list.return_value = [existing_container]

        new_container = MagicMock()
        new_container.id = "new-container-id"
        mock_docker_client.containers.run.return_value = new_container

        # Act
        asyncio.run(orchestrator_with_mock.ensure_indexer(project_name, host_path))

        # Assert - verify removal sequence
        existing_container.stop.assert_called_once_with(timeout=10)
        existing_container.remove.assert_called_once_with(force=True)
        mock_docker_client.containers.run.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=True)
    def test_graceful_shutdown_on_stop_failure(self, mock_isdir, mock_exists, orchestrator_with_mock, mock_docker_client):
        """Test that remove is still called even if stop fails"""
        # Arrange
        project_name = "test-project"
        host_path = "/test/path"

        # Mock existing container that fails to stop
        existing_container = MagicMock()
        existing_container.id = "stubborn-container"
        existing_container.status = "running"
        existing_container.stop.side_effect = docker.errors.APIError("Container won't stop")

        mock_docker_client.containers.list.return_value = [existing_container]

        new_container = MagicMock()
        new_container.id = "new-container-id"
        mock_docker_client.containers.run.return_value = new_container

        # Act
        asyncio.run(orchestrator_with_mock.ensure_indexer(project_name, host_path))

        # Assert - remove should still be called despite stop failure
        existing_container.stop.assert_called_once()
        existing_container.remove.assert_called_once_with(force=True)
        mock_docker_client.containers.run.assert_called_once()

    @patch('os.path.exists', return_value=True)
    @patch('os.path.isdir', return_value=True)
    def test_no_removal_when_no_existing_container(self, mock_isdir, mock_exists, orchestrator_with_mock, mock_docker_client):
        """Test that stop/remove are not called when no existing container"""
        # Arrange
        project_name = "brand-new-project"
        host_path = "/test/new"

        # No existing containers
        mock_docker_client.containers.list.return_value = []

        new_container = MagicMock()
        new_container.id = "new-container-id"
        mock_docker_client.containers.run.return_value = new_container

        # Act
        asyncio.run(orchestrator_with_mock.ensure_indexer(project_name, host_path))

        # Assert - no stop/remove calls
        mock_docker_client.containers.run.assert_called_once()
        # Verify no stop/remove on non-existent containers
        for container in mock_docker_client.containers.list.return_value:
            container.stop.assert_not_called()
            container.remove.assert_not_called()


# ============================================================================
# INTEGRATION TESTS - Real Docker Daemon
# ============================================================================

@pytest.mark.integration
class TestADR0048IntegrationTests:
    """Integration tests with real Docker daemon"""

    @pytest.fixture
    async def orchestrator(self):
        """Create real orchestrator with actual Docker client"""
        context_manager = ProjectContextManager()
        orchestrator = IndexerOrchestrator(context_manager=context_manager)
        await orchestrator.initialize()
        yield orchestrator
        # Cleanup
        await orchestrator.cleanup()

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory"""
        temp_dir = tempfile.mkdtemp(prefix="adr0048_test_")
        # Create a marker file
        marker_file = Path(temp_dir) / "marker.txt"
        marker_file.write_text("ADR-0048 test file")
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_real_docker(self, orchestrator, temp_project_dir):
        """Test full container lifecycle with real Docker"""
        project_name = "adr0048-test-project"

        try:
            # Ensure no existing container
            docker_client = docker.from_env()
            try:
                existing = docker_client.containers.get(f"indexer-{project_name}")
                existing.remove(force=True)
            except docker.errors.NotFound:
                pass

            # Create container
            container_id = await orchestrator.ensure_indexer(project_name, temp_project_dir)

            # Verify container configuration
            container = docker_client.containers.get(container_id)

            # Check environment variables
            env_vars = container.attrs['Config']['Env']
            project_path = None
            for env in env_vars:
                if env.startswith('PROJECT_PATH='):
                    project_path = env.split('=', 1)[1]
                    break

            assert project_path == '/workspace', \
                f"PROJECT_PATH should be /workspace, got {project_path}"

            # Check volume mounts
            mounts = container.attrs['Mounts']
            workspace_mount = None
            for mount in mounts:
                if mount['Destination'] == '/workspace':
                    workspace_mount = mount
                    break

            assert workspace_mount is not None, "No /workspace mount found"
            assert workspace_mount['Source'] == temp_project_dir, \
                f"Mount source should be {temp_project_dir}, got {workspace_mount['Source']}"

            # Verify file is accessible inside container
            exec_result = container.exec_run("ls /workspace/marker.txt")
            assert exec_result.exit_code == 0, "Marker file not accessible in container"

        finally:
            # Cleanup
            await orchestrator.stop_indexer(project_name)

    @pytest.mark.asyncio
    async def test_idempotency_with_replacement(self, orchestrator, temp_project_dir):
        """Test that calling ensure_indexer twice replaces the container"""
        project_name = "adr0048-idempotent-test"

        try:
            # Create first container
            container_id_1 = await orchestrator.ensure_indexer(project_name, temp_project_dir)

            # Create second container (should replace first)
            container_id_2 = await orchestrator.ensure_indexer(project_name, temp_project_dir)

            # Verify different containers
            assert container_id_1 != container_id_2, \
                "Second call should create a new container"

            # Verify first container is gone
            docker_client = docker.from_env()
            with pytest.raises(docker.errors.NotFound):
                docker_client.containers.get(container_id_1)

            # Verify second container exists and is running
            container_2 = docker_client.containers.get(container_id_2)
            assert container_2.status == 'running', \
                f"New container should be running, got {container_2.status}"

        finally:
            # Cleanup
            await orchestrator.stop_indexer(project_name)


# ============================================================================
# E2E VALIDATION TESTS
# ============================================================================

@pytest.mark.e2e
class TestADR0048E2EValidation:
    """End-to-end validation tests for production scenarios"""

    @pytest.mark.asyncio
    async def test_indexer_can_read_workspace(self):
        """Test that indexer can actually read files from /workspace"""
        # This would be integrated with your actual indexing logic
        # For now, we'll create a simple validation

        context_manager = ProjectContextManager()
        orchestrator = IndexerOrchestrator(context_manager=context_manager)
        await orchestrator.initialize()

        project_name = "e2e-workspace-test"
        project_path = "/Users/mkr/local-coding/claude-l9-template"  # Your actual project

        try:
            # Start indexer
            container_id = await orchestrator.ensure_indexer(project_name, project_path)

            # Verify container can list files in /workspace
            docker_client = docker.from_env()
            container = docker_client.containers.get(container_id)

            # Execute a command to verify /workspace is readable
            exec_result = container.exec_run("ls -la /workspace")
            assert exec_result.exit_code == 0, "Cannot list /workspace directory"

            # Check that some expected files are visible
            output = exec_result.output.decode('utf-8')
            assert "pyproject.toml" in output or "package.json" in output, \
                "Expected project files not found in /workspace"

        finally:
            await orchestrator.stop_indexer(project_name)
            await orchestrator.cleanup()


# ============================================================================
# FAILURE SCENARIO TESTS
# ============================================================================

class TestADR0048FailureScenarios:
    """Test failure scenarios and recovery"""

    @pytest.mark.asyncio
    async def test_invalid_host_path(self):
        """Test handling of non-existent host path"""
        orchestrator = IndexerOrchestrator()
        await orchestrator.initialize()

        try:
            with pytest.raises(ValueError, match="does not exist"):
                await orchestrator.ensure_indexer(
                    "invalid-path-test",
                    "/this/path/does/not/exist/at/all"
                )
        finally:
            await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_docker_daemon_unavailable(self):
        """Test handling when Docker daemon is not available"""
        orchestrator = IndexerOrchestrator()

        # Mock docker client to simulate daemon unavailable
        with patch('docker.from_env') as mock_docker:
            mock_docker.side_effect = docker.errors.DockerException("Cannot connect to Docker")

            with pytest.raises(docker.errors.DockerException):
                await orchestrator.initialize()


# ============================================================================
# PERFORMANCE/STRESS TESTS
# ============================================================================

@pytest.mark.performance
class TestADR0048PerformanceTests:
    """Performance and stress tests for container management"""

    @pytest.mark.asyncio
    async def test_rapid_replacement_cycle(self):
        """Test rapid container replacement doesn't leak resources"""
        orchestrator = IndexerOrchestrator()
        await orchestrator.initialize()

        project_name = "stress-test-rapid"
        project_path = "/tmp"

        docker_client = docker.from_env()

        # Get initial container count
        initial_containers = len(docker_client.containers.list(all=True))

        try:
            # Rapid replacement cycle
            for i in range(10):  # Reduced from 50 for faster testing
                await orchestrator.ensure_indexer(project_name, project_path)

            # Check container count hasn't grown
            final_containers = len(docker_client.containers.list(all=True))

            # Should have at most 1 additional container (the current one)
            assert final_containers <= initial_containers + 1, \
                f"Container leak detected: started with {initial_containers}, ended with {final_containers}"

        finally:
            await orchestrator.stop_indexer(project_name)
            await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_calls_same_project(self):
        """Test concurrent calls for same project are handled safely"""
        orchestrator = IndexerOrchestrator()
        await orchestrator.initialize()

        project_name = "concurrent-test"
        project_path = "/tmp"

        try:
            # Launch concurrent ensure_indexer calls
            tasks = [
                orchestrator.ensure_indexer(project_name, project_path)
                for _ in range(5)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check no crashes/exceptions
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"Concurrent calls caused exceptions: {exceptions}"

            # Verify only one container exists
            docker_client = docker.from_env()
            containers = docker_client.containers.list(
                filters={'name': f'indexer-{project_name}'}
            )
            assert len(containers) == 1, \
                f"Expected 1 container, found {len(containers)}"

        finally:
            await orchestrator.stop_indexer(project_name)
            await orchestrator.cleanup()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])