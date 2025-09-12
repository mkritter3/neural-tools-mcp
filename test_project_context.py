#!/usr/bin/env python3
"""Test script for dynamic project context detection"""

import asyncio
import sys
from pathlib import Path

# Add the neural-tools paths
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src/servers/services')
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src/servers')

from project_context_manager import ProjectContextManager

async def test_project_detection():
    """Test the project context detection system"""
    
    manager = ProjectContextManager()
    
    print("=" * 60)
    print("Testing Dynamic Project Context Detection")
    print("=" * 60)
    
    # Test 1: Auto-detect current project
    print("\n1. Auto-detecting current project...")
    current = await manager.get_current_project()
    print(f"   Project: {current['project']}")
    print(f"   Path: {current['path']}")
    print(f"   Confidence: {current['confidence']:.1%}")
    print(f"   Method: {current['method']}")
    
    # Test 2: List all known projects
    print("\n2. Listing all known projects...")
    projects = await manager.list_projects()
    for proj in projects[:5]:  # Show first 5
        print(f"   - {proj['name']}: {proj['path']}")
    if len(projects) > 5:
        print(f"   ... and {len(projects) - 5} more")
    
    # Test 3: Try to detect a different project
    test_paths = [
        "/Users/mkr/local-coding/neural-novelist",
        "/Users/mkr/local-coding/vesikaa",
        "/Users/mkr/local-coding/claude-l9-template"
    ]
    
    print("\n3. Testing project detection for different paths...")
    for test_path in test_paths:
        if Path(test_path).exists():
            result = await manager.set_project(test_path)
            print(f"   {test_path}:")
            print(f"     -> {result['project']} (confidence: {result['confidence']:.1%})")
    
    # Test 4: Verify indexer orchestration would trigger
    print("\n4. Verifying indexer orchestration readiness...")
    # Check if switching projects would trigger container spawn
    result = await manager.set_project("/Users/mkr/local-coding/neural-novelist")
    print(f"   Set project to: {result['project']}")
    print(f"   Would trigger indexer: neural-indexer-{result['project']}")
    
    # Check current context
    current = await manager.get_current_project()
    print(f"   Current active: {current['project']}")
    print(f"   Container name: neural-indexer-{current['project']}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    

if __name__ == "__main__":
    asyncio.run(test_project_detection())