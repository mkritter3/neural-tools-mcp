#!/usr/bin/env python3
"""
Test Dynamic Port Discovery for Indexer Containers
Validates that MCP tools can discover actual container ports instead of using hardcoded values
"""

import asyncio
import docker
import pytest
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Any

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.servers.services.indexer_orchestrator import IndexerOrchestrator
from src.servers.services.container_discovery import ContainerDiscoveryService


class TestDynamicPortDiscovery:
    """Test suite for dynamic port discovery functionality"""

    @pytest.fixture
    def mock_docker_client(self):
        """Create a mock Docker client with running containers"""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        # Create mock containers with different ports
        mock_container_1 = MagicMock()
        mock_container_1.name = "indexer-claude-l9-template"
        mock_container_1.id = "container-123"
        mock_container_1.status = "running"
        mock_container_1.attrs = {
            'NetworkSettings': {
                'Ports': {
                    '8080/tcp': [{'HostPort': '48102'}]
                }
            }
        }

        mock_container_2 = MagicMock()
        mock_container_2.name = "indexer-northstar-finance"
        mock_container_2.id = "container-456"
        mock_container_2.status = "running"
        mock_container_2.attrs = {
            'NetworkSettings': {
                'Ports': {
                    '8080/tcp': [{'HostPort': '48100'}]
                }
            }
        }

        # Mock containers.list to return our containers
        mock_client.containers.list.return_value = [mock_container_1, mock_container_2]
        mock_client.containers.get.side_effect = lambda id: {
            "container-123": mock_container_1,
            "container-456": mock_container_2
        }.get(id)

        return mock_client

    @pytest.mark.asyncio
    async def test_orchestrator_get_indexer_port(self, mock_docker_client):
        """Test that orchestrator.get_indexer_port returns correct port"""
        orchestrator = IndexerOrchestrator()
        orchestrator.docker_client = mock_docker_client

        # Manually add to active_indexers (simulating ensure_indexer)
        orchestrator.active_indexers['claude-l9-template'] = {
            'container_id': 'container-123',
            'port': 48102
        }
        orchestrator.active_indexers['northstar-finance'] = {
            'container_id': 'container-456',
            'port': 48100
        }

        # Test get_indexer_port method
        port1 = orchestrator.get_indexer_port('claude-l9-template')
        assert port1 == 48102, f"Expected port 48102, got {port1}"

        port2 = orchestrator.get_indexer_port('northstar-finance')
        assert port2 == 48100, f"Expected port 48100, got {port2}"

        # Test non-existent project
        port3 = orchestrator.get_indexer_port('non-existent-project')
        assert port3 is None, f"Expected None for non-existent project, got {port3}"

    @pytest.mark.asyncio
    async def test_discovery_service_find_container(self, mock_docker_client):
        """Test that discovery service can find containers and their ports"""
        discovery = ContainerDiscoveryService(mock_docker_client)

        # Test discovering claude-l9-template container
        result = await discovery.discover_project_container('claude-l9-template')
        assert result is not None, "Should find claude-l9-template container"
        assert result['port'] == 48102, f"Expected port 48102, got {result['port']}"
        assert result['container_name'] == 'indexer-claude-l9-template'
        assert result['status'] == 'running'

        # Test discovering northstar-finance container
        result2 = await discovery.discover_project_container('northstar-finance')
        assert result2 is not None, "Should find northstar-finance container"
        assert result2['port'] == 48100, f"Expected port 48100, got {result2['port']}"
        assert result2['container_name'] == 'indexer-northstar-finance'

        # Test non-existent project
        result3 = await discovery.discover_project_container('non-existent')
        assert result3 is None, "Should not find non-existent container"

    @pytest.mark.asyncio
    async def test_mcp_tool_port_discovery_flow(self, mock_docker_client):
        """Test the complete flow as used by MCP tools"""
        # Simulate the MCP tool flow
        orchestrator = IndexerOrchestrator()
        orchestrator.docker_client = mock_docker_client

        # Initialize discovery service (as done in orchestrator.initialize())
        from src.servers.services.container_discovery import ContainerDiscoveryService
        orchestrator.discovery_service = ContainerDiscoveryService(mock_docker_client)

        project_name = 'claude-l9-template'

        # Step 1: Try to get port from active_indexers (will be None initially)
        indexer_port = orchestrator.get_indexer_port(project_name)
        assert indexer_port is None, "Initially should have no port in active_indexers"

        # Step 2: Fall back to discovery service
        if not indexer_port and orchestrator.discovery_service:
            discovered = await orchestrator.discovery_service.discover_project_container(project_name)
            if discovered and discovered.get('port'):
                indexer_port = discovered['port']

        assert indexer_port == 48102, f"Should discover port 48102, got {indexer_port}"

    @pytest.mark.asyncio
    async def test_port_allocation_range(self, mock_docker_client):
        """Test that new containers get allocated ports in correct range"""
        orchestrator = IndexerOrchestrator()
        orchestrator.docker_client = mock_docker_client

        # Check port allocation range
        assert orchestrator.port_base == 48100, "Port base should be 48100"
        # Note: port_range is computed, not stored as an attribute

        # Test _allocate_port method
        port = orchestrator._allocate_port()
        assert 48100 <= port < 48200, f"Port {port} not in range 48100-48199"

        # Allocate multiple ports
        allocated = set()
        for _ in range(10):
            port = orchestrator._allocate_port()
            assert port not in allocated, f"Port {port} already allocated"
            allocated.add(port)
            assert 48100 <= port < 48200, f"Port {port} not in range"
            # Track allocated ports to prevent duplicates
            orchestrator.allocated_ports.add(port)

    @pytest.mark.asyncio
    async def test_container_with_missing_port_mapping(self, mock_docker_client):
        """Test handling of containers without port mappings"""
        # Create container without port mapping
        mock_container = MagicMock()
        mock_container.name = "indexer-broken-project"
        mock_container.id = "container-789"
        mock_container.status = "running"
        mock_container.attrs = {
            'NetworkSettings': {'Ports': {}},  # No port mapping
            'HostConfig': {'PortBindings': {}}  # No port bindings either
        }

        mock_docker_client.containers.list.return_value = [mock_container]

        discovery = ContainerDiscoveryService(mock_docker_client)
        result = await discovery.discover_project_container('broken-project')

        # Should return None when no port is found
        assert result is None, "Should return None for container without port mapping"

    @pytest.mark.asyncio
    async def test_no_hardcoded_48080_fallback(self):
        """Ensure we're not falling back to hardcoded 48080 inappropriately"""
        orchestrator = IndexerOrchestrator()

        # With no Docker client and no active indexers
        port = orchestrator.get_indexer_port('any-project')
        assert port is None, "Should return None, not fall back to 48080"

        # This validates that the fix removes inappropriate hardcoded defaults


class TestIntegrationWithRealDocker:
    """Integration tests with real Docker (if available)"""

    @pytest.mark.skipif(not shutil.which('docker'), reason="Docker not available")
    @pytest.mark.asyncio
    async def test_real_container_discovery(self):
        """Test with real Docker daemon if available"""
        try:
            docker_client = docker.from_env()
            docker_client.ping()
        except:
            pytest.skip("Docker daemon not accessible")

        discovery = ContainerDiscoveryService(docker_client)

        # Get all running indexer containers
        containers = docker_client.containers.list(
            filters={'name': 'indexer-'}
        )

        if not containers:
            pytest.skip("No indexer containers running")

        # Test discovery for each running container
        for container in containers:
            project_name = container.name.replace('indexer-', '')
            result = await discovery.discover_project_container(project_name)

            if container.status == 'running':
                assert result is not None, f"Should find {container.name}"
                assert 'port' in result, "Should have port in result"
                assert isinstance(result['port'], int), "Port should be an integer"
                assert 48000 <= result['port'] < 48300, f"Port {result['port']} in expected range"
                print(f"✅ Found {container.name} on port {result['port']}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])