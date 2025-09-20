#!/usr/bin/env python3
"""
Test script for ADR-0058 indexer fix
Tests the circular dependency and race condition fixes
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "neural-tools" / "src"))

async def test_scenario_1_cold_start():
    """Test Scenario 1: Fresh MCP session (cold start)"""
    print("\n" + "="*60)
    print("TEST SCENARIO 1: Fresh MCP Session (Cold Start)")
    print("="*60)

    try:
        # Import after path setup
        from neural_mcp.neural_server_stdio import get_project_context, state

        # Simulate first call - PROJECT_CONTEXT should be None initially
        print("1. Calling get_project_context for first time...")
        start_time = time.time()

        project_name, container, _ = await get_project_context({})

        elapsed = time.time() - start_time
        print(f"   ✅ PROJECT_CONTEXT initialized in {elapsed:.2f}s")
        print(f"   Project: {project_name}")
        print(f"   Container exists: {container is not None}")

        # Check for circular dependency symptoms
        if container and hasattr(container, 'context_manager'):
            if container.context_manager is None:
                print("   ❌ FAILED: Container has None context_manager (circular dependency!)")
                return False
            else:
                print(f"   ✅ Container has valid context_manager: {type(container.context_manager).__name__}")

        # Check if discovery service is available
        if container and hasattr(container, 'indexer_orchestrator'):
            if not container.indexer_orchestrator:
                print("   ⚠️  IndexerOrchestrator not initialized yet")
            else:
                orchestrator = container.indexer_orchestrator
                if hasattr(orchestrator, 'discovery_service'):
                    if orchestrator.discovery_service:
                        print("   ✅ Discovery service available")
                    else:
                        print("   ❌ FAILED: No discovery service (would use removal logic!)")
                        return False

        print("   ✅ PASSED: Cold start successful, no race condition detected")
        return True

    except Exception as e:
        print(f"   ❌ FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_scenario_2_isinstance_check():
    """Test Scenario 2: isinstance check with module identity"""
    print("\n" + "="*60)
    print("TEST SCENARIO 2: isinstance Check Validation")
    print("="*60)

    try:
        from servers.services.project_context_manager import ProjectContextManager
        from servers.services.service_container import ServiceContainer

        # Create a real ProjectContextManager
        print("1. Creating ProjectContextManager...")
        context_mgr = ProjectContextManager()

        print("2. Testing ServiceContainer with valid context_manager...")
        try:
            container = ServiceContainer(context_manager=context_mgr)
            print("   ✅ ServiceContainer accepted valid ProjectContextManager")
        except TypeError as e:
            print(f"   ❌ FAILED: isinstance check rejected valid context_manager: {e}")
            return False

        print("3. Testing ServiceContainer with invalid type...")
        try:
            container = ServiceContainer(context_manager="not a ProjectContextManager")
            print("   ❌ FAILED: isinstance check didn't catch invalid type!")
            return False
        except TypeError as e:
            print(f"   ✅ isinstance check correctly rejected invalid type: {e}")

        print("   ✅ PASSED: isinstance check working correctly")
        return True

    except Exception as e:
        print(f"   ❌ FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_scenario_3_file_extensions():
    """Test Scenario 3: File extension coverage"""
    print("\n" + "="*60)
    print("TEST SCENARIO 3: File Extension Coverage")
    print("="*60)

    try:
        from servers.services.indexer_service import IncrementalIndexer

        # Create indexer service
        indexer = IncrementalIndexer("test-project", "/tmp/test")

        # Check for critical extensions
        critical_extensions = ['.dart', '.vue', '.svelte', '.kt', '.swift']
        missing = []

        print("Checking critical file extensions...")
        for ext in critical_extensions:
            if ext in indexer.watch_patterns:
                print(f"   ✅ {ext} supported")
            else:
                print(f"   ❌ {ext} MISSING")
                missing.append(ext)

        if missing:
            print(f"   ❌ FAILED: Missing extensions: {missing}")
            return False

        print(f"   ℹ️  Total extensions supported: {len(indexer.watch_patterns)}")
        print("   ✅ PASSED: All critical file extensions supported")
        return True

    except Exception as e:
        print(f"   ❌ FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all test scenarios"""
    print("\n" + "#"*60)
    print("# ADR-0058 INDEXER FIX VALIDATION TESTS")
    print("#"*60)

    results = []

    # Test 1: Cold start
    result1 = await test_scenario_1_cold_start()
    results.append(("Cold Start", result1))

    # Test 2: isinstance check
    result2 = await test_scenario_2_isinstance_check()
    results.append(("isinstance Check", result2))

    # Test 3: File extensions
    result3 = await test_scenario_3_file_extensions()
    results.append(("File Extensions", result3))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:20} {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! The fix appears to be working.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)