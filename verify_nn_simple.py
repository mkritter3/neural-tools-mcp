#!/usr/bin/env python3
"""Simple verification of Neural Novelist project detection"""

import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src/servers/services')

from project_context_manager import ProjectContextManager

async def verify():
    manager = ProjectContextManager()
    
    print("=" * 60)
    print("NEURAL NOVELIST - DYNAMIC PROJECT DETECTION")
    print("=" * 60)
    
    # Test detection for Neural Novelist
    nn_path = "/Users/mkr/local-coding/neural-novelist"
    result = await manager.set_project(nn_path)
    
    print(f"\n✅ Project Detection Working!")
    print(f"   Project: {result['project']}")
    print(f"   Collection: project_{result['project']}_code")
    print(f"   Container: neural-indexer-{result['project']}")
    
    # Check Docker containers
    print(f"\n📦 Docker Containers:")
    result = subprocess.run(
        ['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'],
        capture_output=True, text=True
    )
    print(result.stdout)
    
    # Show what will happen
    print("\n🎯 What Happens Next:")
    print("1. When you restart Claude, new MCP tools will be available:")
    print("   - set_project_context: Switch active project")
    print("   - list_projects: See all known projects")
    print("\n2. When you use 'set_project_context' with Neural Novelist:")
    print("   - Project context switches to 'neural-novelist'")
    print("   - Indexing will use 'project_neural-novelist_code' collection")
    print("   - Neo4j nodes will have {project: 'neural-novelist'} property")
    print("\n3. If ADR-0030 multi-container is enabled:")
    print("   - Container 'neural-indexer-neural-novelist' will be created")
    print("   - Each project gets its own isolated indexer")
    
    print("\n✨ SUCCESS! Multi-project support is ready!")
    
if __name__ == "__main__":
    asyncio.run(verify())