#!/usr/bin/env python3
"""
ADR-64: Mock-based unit tests for indexer mount validation
These tests run without Docker and verify the logic of container management
"""

import os
import sys
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, PropertyMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'neural-tools', 'src'))

from servers.services.indexer_orchestrator import IndexerOrchestrator, is_container_reusable


class TestMountValidation:
    """
    Mock-based tests for mount validation logic (ADR-63/64)
    These tests verify container recreation logic without requiring Docker
    """

    @pytest.fixture
    def mock_docker_client(self):
        """Create a mock Docker client"""
        client = MagicMock()
        client.ping.return_value = True
        return client

    @pytest.fixture
    def orchestrator(self, mock_docker_client):
        """Create orchestrator with mocked Docker client"""
        orch = IndexerOrchestrator()
        orch.docker_client = mock_docker_client
        orch.redis_client = None  # Use local locks
        return orch

    def test_is_container_reusable_matching_mount(self):
        """Test that containers with matching mounts are reusable"""
        container_attrs = {
            'Mounts': [
                {
                    'Source': '/Users/test/project',
                    'Destination': '/workspace',
                    'Mode': 'ro'
                }
            ],
            'Config': {
                'Env': ['PROJECT_NAME=test', 'PROJECT_PATH=/workspace']
            }
        }

        assert is_container_reusable(
            container_attrs,
            '/Users/test/project',
            {'PROJECT_NAME': 'test', 'PROJECT_PATH': '/workspace'}
        ) is True

    def test_is_container_reusable_different_mount(self):
        """Test that containers with different mounts are NOT reusable"""
        container_attrs = {
            'Mounts': [
                {
                    'Source': '/Users/test/old-project',
                    'Destination': '/workspace',
                    'Mode': 'ro'
                }
            ],
            'Config': {
                'Env': ['PROJECT_NAME=test', 'PROJECT_PATH=/workspace']
            }
        }

        assert is_container_reusable(
            container_attrs,
            '/Users/test/new-project',  # Different path
            {'PROJECT_NAME': 'test', 'PROJECT_PATH': '/workspace'}
        ) is False

    def test_is_container_reusable_env_var_mismatch(self):
        """Test that containers with different env vars are NOT reusable"""
        container_attrs = {
            'Mounts': [
                {
                    'Source': '/Users/test/project',
                    'Destination': '/workspace',
                    'Mode': 'ro'
                }
            ],
            'Config': {
                'Env': ['DEBUG=false', 'LOG_LEVEL=INFO']
            }
        }

        assert is_container_reusable(
            container_attrs,
            '/Users/test/project',
            {'DEBUG': 'true', 'LOG_LEVEL': 'DEBUG'}  # Different values
        ) is False

    @pytest.mark.asyncio
    async def test_ensure_indexer_removes_stale_container(self, orchestrator, mock_docker_client):
        """Test that stale containers with wrong mounts are removed"""
        # Mock path existence and directory checks
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True):
            # Mock existing container with wrong mount
            stale_container = MagicMock()
            stale_container.id = 'stale123'
            stale_container.name = 'indexer-test-old'
            stale_container.labels = {
                'com.l9.project': 'test-project',
                'com.l9.managed': 'true'
            }
            stale_container.attrs = {
                'Mounts': [
                    {
                        'Source': '/old/path',
                        'Destination': '/workspace'
                    }
                ],
                'Config': {
                    'Env': []
                }
            }

            # Mock discovery service with async method
            orchestrator.discovery_service = MagicMock()

            # Mock discover_project_container to return container info
            async def mock_discover_project(*args, **kwargs):
                return {
                    'container_id': stale_container.id,
                    'name': stale_container.name,
                    'labels': stale_container.labels
                }
            orchestrator.discovery_service.discover_project_container = mock_discover_project

            # Mock docker client to return stale container when getting by ID
            mock_docker_client.containers.get.return_value = stale_container

            # Mock container creation
            new_container = MagicMock()
            new_container.id = 'new456'
            mock_docker_client.containers.run.return_value = new_container

            # Request indexer with different path
            container_id = await orchestrator._ensure_indexer_internal('test-project', '/new/path')

            # Verify stale container was removed
            stale_container.remove.assert_called_once_with(force=True)

            # Verify new container was created
            mock_docker_client.containers.run.assert_called_once()
            assert container_id == 'new456'

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_paths(self, orchestrator, mock_docker_client):
        """Test concurrent requests with different paths create different containers"""
        # Mock path existence and directory checks
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isdir', return_value=True):
            # Mock discovery returning no existing containers
            orchestrator.discovery_service = MagicMock()
            async def mock_discover(*args, **kwargs):
                return []
            orchestrator.discovery_service.discover_by_project = mock_discover

            # Mock container creation to return unique containers
            container_ids = ['container1', 'container2', 'container3']
            mock_docker_client.containers.run.side_effect = [
                MagicMock(id=cid) for cid in container_ids
            ]

            # Concurrent requests for same project, different paths
            paths = ['/path1', '/path2', '/path3']
            tasks = [
                orchestrator._ensure_indexer_internal(f'concurrent-test-{i}', path)
                for i, path in enumerate(paths)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should create different containers
            assert len(set(results)) == 3
            assert mock_docker_client.containers.run.call_count == 3

    @pytest.mark.asyncio
    async def test_redis_fallback_still_works(self, mock_docker_client):
        """Test that orchestrator works even when Redis fails"""
        orch = IndexerOrchestrator()
        orch.docker_client = mock_docker_client

        # Simulate Redis failure
        with patch('servers.services.indexer_orchestrator.redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Redis connection failed")

            # Initialize should succeed despite Redis failure
            await orch.initialize()

            # Redis should be None (fallback mode)
            assert orch.redis_client is None

            # Discovery service should still be initialized
            assert orch.discovery_service is not None

    def test_mount_path_normalization(self):
        """Test that mount paths are properly normalized for comparison"""
        # Mock os.path.abspath to test normalization behavior
        with patch('os.path.abspath') as mock_abspath:
            # Make abspath normalize trailing slashes
            mock_abspath.side_effect = lambda p: p.rstrip('/')

            container_attrs = {
                'Mounts': [
                    {
                        'Source': '/Users/test/project',  # No trailing slash
                        'Destination': '/workspace'
                    }
                ],
                'Config': {'Env': []}
            }

            # Should match with normalized path
            assert is_container_reusable(
                container_attrs,
                '/Users/test/project/',  # Trailing slash - will be normalized
                {}
            ) is True

            # Should match without trailing slash
            assert is_container_reusable(
                container_attrs,
                '/Users/test/project',
                {}
            ) is True


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )
    sys.exit(result.returncode)