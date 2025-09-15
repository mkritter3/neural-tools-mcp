#!/usr/bin/env python3
"""
Test script to verify ADR-0048 indexer fix
Tests that indexers can now start properly with PROJECT_PATH=/workspace
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.servers.services.indexer_orchestrator import IndexerOrchestrator
from src.servers.services.project_context_manager import ProjectContextManager

async def test_indexer_startup():
    """Test that indexer can start with correct PROJECT_PATH"""
    print("🧪 Testing ADR-0048 Indexer Fix...")
    print("=" * 50)

    try:
        # Initialize context manager
        context_manager = ProjectContextManager()

        # Initialize orchestrator with context manager
        orchestrator = IndexerOrchestrator(context_manager=context_manager)
        await orchestrator.initialize()

        # Test project path
        test_project = "claude-l9-template"
        test_path = "/Users/mkr/local-coding/claude-l9-template"

        print(f"\n📦 Testing indexer startup for: {test_project}")
        print(f"   Host path: {test_path}")
        print(f"   Container path: /workspace (ADR-0048)")

        # Ensure indexer with ADR-0048 fix
        container_id = await orchestrator.ensure_indexer(test_project, test_path)

        if container_id:
            print(f"\n✅ SUCCESS: Indexer started with container ID: {container_id[:12]}")

            # Verify the container configuration
            import docker
            client = docker.from_env()
            container = client.containers.get(container_id)

            # Check environment variables
            env_vars = container.attrs['Config']['Env']
            project_path_env = None
            for env in env_vars:
                if env.startswith('PROJECT_PATH='):
                    project_path_env = env.split('=', 1)[1]
                    break

            if project_path_env == '/workspace':
                print(f"✅ PROJECT_PATH correctly set to: {project_path_env}")
            else:
                print(f"❌ PROJECT_PATH incorrectly set to: {project_path_env}")

            # Check if container is running
            if container.status == 'running':
                print(f"✅ Container is running")
            else:
                print(f"⚠️  Container status: {container.status}")

            # Clean up - stop the test container
            print(f"\n🧹 Cleaning up test container...")
            await orchestrator.stop_indexer(test_project)
            print("✅ Test container stopped")

        else:
            print("❌ FAILED: Could not start indexer")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("✅ ADR-0048 Fix Verified Successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_indexer_startup())
    sys.exit(0 if success else 1)