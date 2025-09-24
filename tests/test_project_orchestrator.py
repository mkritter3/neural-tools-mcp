#!/usr/bin/env python3
"""
Test Project Orchestrator - ADR-0097
Verify multi-project orchestration works without breaking legacy mode
"""

import os
import sys
import json
import time
import asyncio
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'neural-tools' / 'src'))

from servers.services.project_orchestrator import ProjectOrchestrator

class TestProjectOrchestrator(unittest.TestCase):
    """Test project orchestration functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.redis_mock = AsyncMock()
        self.docker_mock = MagicMock()
        self.orchestrator = ProjectOrchestrator(self.redis_mock, self.docker_mock)

    def test_project_id_generation(self):
        """Test deterministic project ID generation"""
        # Same path should generate same ID
        path1 = "/Users/test/project1"
        id1 = self.orchestrator.get_project_id(path1)
        id2 = self.orchestrator.get_project_id(path1)
        self.assertEqual(id1, id2)

        # Different paths should generate different IDs
        path2 = "/Users/test/project2"
        id3 = self.orchestrator.get_project_id(path2)
        self.assertNotEqual(id1, id3)

        # ID should be 12 characters (truncated SHA1)
        self.assertEqual(len(id1), 12)

        # Normalized paths should generate same ID
        path_with_slash = "/Users/test/project1/"
        path_without_slash = "/Users/test/project1"
        id4 = self.orchestrator.get_project_id(path_with_slash)
        id5 = self.orchestrator.get_project_id(path_without_slash)
        self.assertEqual(id4, id5)

    async def test_legacy_mode_preserved(self):
        """Test that legacy mode works when feature flag is off"""
        # Ensure we're in legacy mode
        with patch.dict(os.environ, {'MCP_MULTI_PROJECT_MODE': 'false'}):
            orchestrator = ProjectOrchestrator(self.redis_mock, self.docker_mock)

            # Mock legacy method
            orchestrator._legacy_start_indexer = AsyncMock(return_value={
                "port": 48100,
                "status": "legacy",
                "container_id": "legacy-container"
            })

            result = await orchestrator.get_or_create_indexer("/some/project")

            # Should use legacy method
            orchestrator._legacy_start_indexer.assert_called_once_with("/some/project")
            self.assertEqual(result["status"], "legacy")

    async def test_multi_project_mode_existing_container(self):
        """Test finding existing container in multi-project mode"""
        with patch.dict(os.environ, {'MCP_MULTI_PROJECT_MODE': 'true'}):
            orchestrator = ProjectOrchestrator(self.redis_mock, self.docker_mock)

            project_dir = "/Users/test/project1"
            project_id = orchestrator.get_project_id(project_dir)

            # Mock existing project info in Redis
            existing_info = {
                "container_id": "existing-container-123",
                "container_name": f"mcp-indexer-{project_id}",
                "port": 48105,
                "status": "running",
                "last_heartbeat": time.time(),
                "project_path": project_dir
            }
            self.redis_mock.hget.return_value = json.dumps(existing_info)

            # Mock container is running
            container_mock = MagicMock()
            container_mock.status = "running"
            self.docker_mock.containers.get.return_value = container_mock

            result = await orchestrator.get_or_create_indexer(project_dir)

            # Should return existing container info
            self.assertEqual(result["status"], "existing")
            self.assertEqual(result["port"], 48105)
            self.assertEqual(result["container_id"], "existing-container-123")

            # Should update heartbeat
            self.redis_mock.hset.assert_called()

    async def test_multi_project_mode_create_new_container(self):
        """Test creating new container in multi-project mode"""
        with patch.dict(os.environ, {'MCP_MULTI_PROJECT_MODE': 'true'}):
            orchestrator = ProjectOrchestrator(self.redis_mock, self.docker_mock)

            project_dir = "/Users/test/project2"
            project_id = orchestrator.get_project_id(project_dir)

            # Mock no existing project
            self.redis_mock.hget.return_value = None
            self.redis_mock.hlen.return_value = 2  # Under limit
            self.redis_mock.hgetall.return_value = {}
            self.redis_mock.lock.return_value.__aenter__ = AsyncMock()
            self.redis_mock.lock.return_value.__aexit__ = AsyncMock()

            # Mock container creation
            new_container = MagicMock()
            new_container.id = "new-container-456"
            self.docker_mock.containers.run.return_value = new_container
            self.docker_mock.containers.list.return_value = []

            result = await orchestrator.get_or_create_indexer(project_dir)

            # Should create new container
            self.assertEqual(result["status"], "created")
            self.assertIn("container_id", result)

            # Should call Docker run with correct params
            self.docker_mock.containers.run.assert_called_once()
            call_kwargs = self.docker_mock.containers.run.call_args.kwargs
            self.assertEqual(call_kwargs["name"], f"mcp-indexer-{project_id}")
            self.assertIn("PROJECT_PATH", call_kwargs["environment"])
            self.assertEqual(call_kwargs["environment"]["PROJECT_PATH"], project_dir)

    async def test_cleanup_idle_containers(self):
        """Test cleanup of idle containers"""
        with patch.dict(os.environ, {'MCP_MULTI_PROJECT_MODE': 'true'}):
            orchestrator = ProjectOrchestrator(self.redis_mock, self.docker_mock)

            # Mock two projects: one idle, one active
            old_time = time.time() - 7200  # 2 hours ago
            recent_time = time.time() - 600  # 10 minutes ago

            idle_project = {
                "container_id": "idle-container",
                "last_heartbeat": old_time
            }
            active_project = {
                "container_id": "active-container",
                "last_heartbeat": recent_time
            }

            self.redis_mock.hgetall.return_value = {
                "idle_proj": json.dumps(idle_project),
                "active_proj": json.dumps(active_project)
            }

            # Mock container operations
            idle_container = MagicMock()
            self.docker_mock.containers.get.return_value = idle_container

            await orchestrator._cleanup_idle_containers()

            # Should remove idle project from Redis
            self.redis_mock.hdel.assert_called_with("mcp:project_registry", "idle_proj")

            # Should stop idle container
            idle_container.stop.assert_called_once()
            idle_container.remove.assert_called_once()

    def test_max_concurrent_projects(self):
        """Test enforcement of max concurrent projects limit"""
        with patch.dict(os.environ, {
            'MCP_MULTI_PROJECT_MODE': 'true',
            'MCP_MAX_PROJECTS': '2'
        }):
            orchestrator = ProjectOrchestrator(self.redis_mock, self.docker_mock)

            # Mock registry at max capacity
            self.redis_mock.hlen.return_value = 2

            # Create should fail
            with self.assertRaises(RuntimeError) as context:
                asyncio.run(orchestrator._create_indexer("proj3", "/path/to/proj3"))

            self.assertIn("Max concurrent projects", str(context.exception))

    async def test_port_allocation(self):
        """Test finding free ports avoids conflicts"""
        orchestrator = ProjectOrchestrator(self.redis_mock, self.docker_mock)

        # Mock some ports already in use
        existing_projects = {
            "proj1": json.dumps({"port": 48100}),
            "proj2": json.dumps({"port": 48101})
        }
        self.redis_mock.hgetall.return_value = existing_projects

        # Mock Docker has no containers on 48102
        self.docker_mock.containers.list.return_value = []

        port = await orchestrator._find_free_port()

        # Should find next available port
        self.assertEqual(port, 48102)

    async def test_graceful_shutdown(self):
        """Test graceful shutdown stops all containers"""
        with patch.dict(os.environ, {'MCP_MULTI_PROJECT_MODE': 'true'}):
            orchestrator = ProjectOrchestrator(self.redis_mock, self.docker_mock)

            # Mock two running projects
            projects = {
                "proj1": json.dumps({"container_id": "container1"}),
                "proj2": json.dumps({"container_id": "container2"})
            }
            self.redis_mock.hgetall.return_value = projects

            # Mock containers
            container1 = MagicMock()
            container2 = MagicMock()
            self.docker_mock.containers.get.side_effect = [container1, container2]

            await orchestrator.shutdown()

            # Should stop both containers
            container1.stop.assert_called_once()
            container2.stop.assert_called_once()

            # Should clear Redis registry
            self.redis_mock.delete.assert_called_with("mcp:project_registry")


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    # For async tests
    import asyncio

    class AsyncTestRunner(unittest.TextTestRunner):
        def run(self, test):
            """Run async tests"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return super().run(test)
            finally:
                loop.close()

    unittest.main(testRunner=AsyncTestRunner(), verbosity=2)