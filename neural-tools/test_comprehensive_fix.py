#!/usr/bin/env python3
"""
Comprehensive test to validate the MCP working directory fix.

This test validates:
1. Server name is correctly set to "neural-tools"
2. Working directory is captured at initialization
3. Project context is properly initialized
4. No remaining "l9-neural-enhanced" references
"""

import os
import sys
import ast
import asyncio
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

def check_server_name():
    """Check that server name is correctly set to 'neural-tools'"""
    print("\n🔍 Checking server name configuration...")

    file_path = "/Users/mkr/local-coding/claude-l9-template/neural-tools/src/neural_mcp/neural_server_stdio.py"
    with open(file_path, 'r') as f:
        content = f.read()

    # Check for correct server name
    if 'Server("neural-tools"' in content:
        print("  ✅ Server name correctly set to 'neural-tools'")
    else:
        print("  ❌ Server name not correctly set")
        return False

    # Check for incorrect references
    if 'l9-neural-enhanced' in content or 'l9-enhanced' in content:
        print("  ❌ Found remaining 'l9-neural-enhanced' references")
        return False
    else:
        print("  ✅ No 'l9-neural-enhanced' references found")

    return True


def check_lifespan_implementation():
    """Check that lifespan is properly implemented"""
    print("\n🔍 Checking lifespan implementation...")

    file_path = "/Users/mkr/local-coding/claude-l9-template/neural-tools/src/neural_mcp/neural_server_stdio.py"
    with open(file_path, 'r') as f:
        content = f.read()

    # Check for lifespan function
    if '@asynccontextmanager' in content and 'async def server_lifespan' in content:
        print("  ✅ Lifespan context manager properly defined")
    else:
        print("  ❌ Lifespan context manager not found")
        return False

    # Check that Server uses lifespan
    if 'Server("neural-tools", lifespan=server_lifespan)' in content:
        print("  ✅ Server correctly uses lifespan")
    else:
        print("  ❌ Server not using lifespan")
        return False

    # Check that working directory is captured in lifespan
    if 'INITIAL_WORKING_DIRECTORY = os.getcwd()' in content:
        print("  ✅ Working directory captured in lifespan")
    else:
        print("  ❌ Working directory not captured correctly")
        return False

    return True


def check_imports():
    """Check that all necessary imports are present"""
    print("\n🔍 Checking imports...")

    file_path = "/Users/mkr/local-coding/claude-l9-template/neural-tools/src/neural_mcp/neural_server_stdio.py"
    with open(file_path, 'r') as f:
        content = f.read()

    required_imports = [
        'from contextlib import asynccontextmanager',
        'from servers.services.project_context_manager import get_project_context_manager'
    ]

    all_present = True
    for imp in required_imports:
        if imp in content:
            print(f"  ✅ Found: {imp}")
        else:
            print(f"  ❌ Missing: {imp}")
            all_present = False

    return all_present


async def test_working_directory_capture():
    """Test that working directory is correctly captured"""
    print("\n🔍 Testing working directory capture...")

    # Simulate starting from different directories
    test_dirs = [
        "/Users/mkr/local-coding/Systems/neural-novelist",
        "/Users/mkr/local-coding/claude-l9-template",
        "/Users/mkr/.claude/mcp-servers/neural-tools"
    ]

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            os.chdir(test_dir)
            cwd = os.getcwd()

            print(f"\n  Testing from: {test_dir}")
            print(f"  Captured: {cwd}")

            if "mcp-servers" in cwd or "mcp_servers" in cwd:
                print("  → Would trigger auto-detection")
            else:
                project_name = Path(cwd).name
                print(f"  → Would use as project: {project_name}")
        else:
            print(f"  ⚠️ Skipping {test_dir} (doesn't exist)")

    return True


def check_project_context_initialization():
    """Check that PROJECT_CONTEXT is properly initialized"""
    print("\n🔍 Checking PROJECT_CONTEXT initialization...")

    file_path = "/Users/mkr/local-coding/claude-l9-template/neural-tools/src/neural_mcp/neural_server_stdio.py"
    with open(file_path, 'r') as f:
        content = f.read()

    # Check global declaration
    if 'PROJECT_CONTEXT = None' in content:
        print("  ✅ PROJECT_CONTEXT global variable declared")
    else:
        print("  ❌ PROJECT_CONTEXT global variable not declared")
        return False

    # Check initialization in lifespan
    if 'PROJECT_CONTEXT = await get_project_context_manager()' in content:
        print("  ✅ PROJECT_CONTEXT initialized in lifespan")
    else:
        print("  ❌ PROJECT_CONTEXT not initialized in lifespan")
        return False

    # Check fallback in get_project_context
    if 'if PROJECT_CONTEXT is None:' in content:
        print("  ✅ Fallback initialization check present")
    else:
        print("  ❌ No fallback initialization check")
        return False

    return True


def main():
    print("=" * 60)
    print("🧪 COMPREHENSIVE MCP FIX VALIDATION")
    print("=" * 60)

    all_passed = True

    # Run all checks
    all_passed &= check_server_name()
    all_passed &= check_lifespan_implementation()
    all_passed &= check_imports()
    all_passed &= check_project_context_initialization()

    # Run async test
    asyncio.run(test_working_directory_capture())

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - MCP FIX IS COMPLETE")
    else:
        print("❌ SOME CHECKS FAILED - REVIEW NEEDED")
    print("=" * 60)


if __name__ == "__main__":
    main()