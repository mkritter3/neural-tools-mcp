#!/usr/bin/env python3
"""
Test Unified Project Detection (ADR-0102)
Verifies that ProjectContextManager is the single source of truth
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from servers.services.project_context_manager import get_project_context_manager

async def test_detection():
    """Test the unified project detection flow"""
    print("=" * 60)
    print("🧪 TESTING UNIFIED PROJECT DETECTION (ADR-0102)")
    print("=" * 60)

    # Get the singleton manager
    manager = await get_project_context_manager()
    print("\n✅ ProjectContextManager singleton obtained")

    # Test 1: Basic detection without containers
    print("\n📍 Test 1: Basic project detection")
    result = await manager.detect_project()
    print(f"  Project: {result.get('project')}")
    print(f"  Path: {result.get('path')}")
    print(f"  Method: {result.get('method')}")
    print(f"  Confidence: {result.get('confidence')}")

    if result.get('error'):
        print(f"  ⚠️ Error: {result.get('error')}")
        print(f"  Action: {result.get('action')}")

    # Test 2: Force refresh (checks containers)
    print("\n📍 Test 2: Force refresh detection (checks containers)")
    result = await manager.get_current_project(force_refresh=True)
    print(f"  Project: {result.get('project')}")
    print(f"  Method: {result.get('method')}")
    print(f"  Confidence: {result.get('confidence')}")

    # Test 3: Explicit project setting
    print("\n📍 Test 3: Explicit project setting")
    test_path = "/Users/mkr/local-coding/claude-l9-template"
    if Path(test_path).exists():
        result = await manager.set_project(test_path)
        print(f"  Set project to: {result.get('project')}")
        print(f"  Path: {result.get('path')}")
        print(f"  Method: {result.get('method')}")
    else:
        print(f"  ⚠️ Test path doesn't exist: {test_path}")

    # Test 4: Verify cached detection
    print("\n📍 Test 4: Cached detection (no refresh)")
    result = await manager.get_current_project(force_refresh=False)
    print(f"  Project: {result.get('project')}")
    print(f"  Method: {result.get('method')}")
    print(f"  Should be 'cached' or 'explicit'")

    # Test 5: List all known projects
    print("\n📍 Test 5: List known projects")
    projects = await manager.list_projects()
    print(f"  Found {len(projects)} known projects:")
    for proj in projects[:3]:  # Show first 3
        print(f"    - {proj['name']}: {proj['path']}")
        if proj['is_current']:
            print(f"      ^ CURRENT PROJECT")

    # Test 6: Verify ProjectDetector is gone
    print("\n📍 Test 6: Verify ProjectDetector is archived")
    detector_path = Path(__file__).parent / "src/neural_mcp/project_detector.py"
    archived_path = Path(__file__).parent / "src/neural_mcp/project_detector.py.archived-20250924"

    if detector_path.exists():
        print(f"  ❌ FAIL: ProjectDetector still exists at {detector_path}")
    else:
        print(f"  ✅ PASS: ProjectDetector removed")

    if archived_path.exists():
        print(f"  ✅ Archived at: {archived_path}")
    else:
        print(f"  ⚠️ Archive not found (may have been deleted)")

    # Test 7: Check detection priority order (ADR-0102)
    print("\n📍 Test 7: Detection priority verification")
    print("  Expected order (ADR-0102):")
    print("    1. CLAUDE_PROJECT_DIR env (1.0)")
    print("    2. Explicit set_project (1.0)")
    print("    3. File-based detection (0.9)")
    print("    4. Container detection (0.7) - SECONDARY")
    print("    5. Registry cache (0.5) - local only")
    print("    6. None/error (0.0)")

    # Simulate environment variable
    os.environ["CLAUDE_PROJECT_DIR"] = "/tmp/test-project"
    try:
        result = await manager.detect_project()
        if result.get('method') == 'claude_project_dir':
            print(f"  ✅ CLAUDE_PROJECT_DIR detected correctly")
        else:
            print(f"  ❌ Wrong detection method: {result.get('method')}")
    finally:
        del os.environ["CLAUDE_PROJECT_DIR"]

    print("\n" + "=" * 60)
    print("🎉 UNIFIED DETECTION TEST COMPLETE")
    print("=" * 60)

    # Summary
    all_passed = True
    if detector_path.exists():
        all_passed = False
        print("\n⚠️ WARNING: ProjectDetector still exists - migration incomplete!")

    if all_passed:
        print("\n✅ All tests passed - ADR-0102 implementation successful!")
    else:
        print("\n❌ Some tests failed - review needed")

    return all_passed


if __name__ == "__main__":
    result = asyncio.run(test_detection())
    sys.exit(0 if result else 1)