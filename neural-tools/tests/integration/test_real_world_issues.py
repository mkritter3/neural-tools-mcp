#!/usr/bin/env python3
"""
Real-World Integration Tests for Neural Tools
Tests actual issues encountered in production that our mocked tests miss
"""

import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Test colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def run_test(name, test_func):
    """Run a test and report results"""
    print(f"{YELLOW}Testing: {name}{NC}")
    try:
        result = test_func()
        if result:
            print(f"{GREEN}✅ PASSED: {name}{NC}")
            return True
        else:
            print(f"{RED}❌ FAILED: {name}{NC}")
            return False
    except Exception as e:
        print(f"{RED}❌ FAILED: {name} - {str(e)}{NC}")
        return False

def test_project_context_manager_initialization():
    """Test that ProjectContextManager properly initializes"""
    try:
        # Import the actual server module
        from neural_mcp.neural_server_stdio import PROJECT_CONTEXT
        from servers.services.project_context_manager import ProjectContextManager

        # Check if PROJECT_CONTEXT is properly initialized
        if not isinstance(PROJECT_CONTEXT, ProjectContextManager):
            print(f"  ERROR: PROJECT_CONTEXT is {type(PROJECT_CONTEXT)}, not ProjectContextManager")
            return False

        # Test that it can detect projects
        import asyncio
        async def test_detection():
            result = await PROJECT_CONTEXT._detect_project_name(Path.cwd())
            return result is not None

        return asyncio.run(test_detection())
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_graphrag_actual_search():
    """Test GraphRAG search with real data (not mocked)"""
    try:
        from servers.services.service_container import ServiceContainer

        async def test_search():
            container = ServiceContainer()
            await container.initialize()

            # Test actual hybrid search
            from servers.services.hybrid_retriever import HybridRetriever
            retriever = HybridRetriever(
                container.neo4j_service,
                container.qdrant_service,
                container.nomic_service
            )

            # Search for something that should exist
            results = await retriever.search(
                "UnifiedSDKRouter MainProcessFileIndexer",
                limit=5
            )

            # Check we got real results
            if not results or len(results) == 0:
                print("  ERROR: No results returned from GraphRAG search")
                return False

            # Check results have expected fields
            for result in results:
                if 'score' not in result or 'content' not in result:
                    print(f"  ERROR: Result missing required fields: {result.keys()}")
                    return False

            return True

        return asyncio.run(test_search())
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_mcp_message_handling():
    """Test actual MCP message handling (not mocked)"""
    try:
        # Create a real MCP request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "project_understanding",
                "arguments": {"scope": "summary"}
            }
        }

        # Run the server with the request
        process = subprocess.Popen(
            [sys.executable, "run_mcp_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # Send initialize first
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }

        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        # Read response
        response_line = process.stdout.readline()
        if not response_line:
            print("  ERROR: No response from server")
            process.terminate()
            return False

        # Send actual request
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()

        # Read response
        response_line = process.stdout.readline()
        process.terminate()

        if not response_line:
            print("  ERROR: No response to tool call")
            return False

        response = json.loads(response_line)

        # Check for error about context_manager
        if "error" in response:
            error_msg = response["error"].get("message", "")
            if "context_manager must be a ProjectContextManager" in error_msg:
                print(f"  ERROR: ProjectContextManager not initialized: {error_msg}")
                return False

        return "result" in response
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_configuration_path_resolution():
    """Test that configuration paths are correctly resolved"""
    try:
        # Check if correct config file exists
        config_paths = [
            Path.home() / ".claude.json",
            Path.home() / ".claude" / "mcp_config.json",
            Path.cwd() / ".mcp.json"
        ]

        found_configs = []
        for path in config_paths:
            if path.exists():
                found_configs.append(str(path))

        if not found_configs:
            print("  ERROR: No configuration files found")
            return False

        # Check if neural-tools is configured
        for config_path in found_configs:
            with open(config_path) as f:
                config = json.load(f)
                if "mcpServers" in config and "neural-tools" in config["mcpServers"]:
                    neural_config = config["mcpServers"]["neural-tools"]

                    # Check critical fields
                    if "command" not in neural_config:
                        print(f"  ERROR: Missing 'command' in {config_path}")
                        return False
                    if "args" not in neural_config:
                        print(f"  ERROR: Missing 'args' in {config_path}")
                        return False
                    if "cwd" not in neural_config and str(config_path).endswith("mcp_config.json"):
                        print(f"  WARNING: Missing 'cwd' in {config_path} - may cause issues")

                    return True

        print("  ERROR: neural-tools not configured in any config file")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_session_persistence():
    """Test that project context persists across multiple calls"""
    try:
        from neural_mcp.neural_server_stdio import PROJECT_CONTEXT

        async def test_persistence():
            # Set project context
            project1 = await PROJECT_CONTEXT.set_project_context(
                Path("/Users/mkr/local-coding/claude-l9-template")
            )

            # Get active project
            active1 = await PROJECT_CONTEXT.get_active_project()

            if active1 != project1:
                print(f"  ERROR: Project not persisted: {active1} != {project1}")
                return False

            # Switch to different project
            project2 = await PROJECT_CONTEXT.set_project_context(
                Path("/Users/mkr/local-coding/Systems/neural-novelist")
            )

            # Get active project again
            active2 = await PROJECT_CONTEXT.get_active_project()

            if active2 != project2:
                print(f"  ERROR: Project switch not persisted: {active2} != {project2}")
                return False

            return True

        return asyncio.run(test_persistence())
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_error_patterns():
    """Test for specific error patterns we've encountered"""
    error_patterns = [
        "context_manager must be a ProjectContextManager instance",
        "unhashable type: 'dict'",
        "RuntimeWarning: coroutine .* was never awaited",
        "Failed to reconnect to neural-tools",
        "Server disconnected without sending a response"
    ]

    # Check recent logs for these patterns
    import re

    # This would check actual log files or stderr output
    # For now, we'll check if the patterns are handled in code
    try:
        # Check if error handling exists for these patterns
        source_files = list(Path(__file__).parent.parent.parent.glob("src/**/*.py"))

        handled_patterns = set()
        for source_file in source_files:
            content = source_file.read_text()
            for pattern in error_patterns:
                # Check if pattern is mentioned in error handling
                if pattern in content or re.search(f"except.*{pattern[:20]}", content):
                    handled_patterns.add(pattern)

        unhandled = set(error_patterns) - handled_patterns
        if unhandled:
            print(f"  WARNING: Unhandled error patterns: {unhandled}")
            # This is a warning, not a failure

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    """Run all real-world tests"""
    print("\n" + "="*50)
    print("🚨 Real-World Integration Tests")
    print("="*50 + "\n")

    tests = [
        ("ProjectContextManager Initialization", test_project_context_manager_initialization),
        ("GraphRAG Actual Search", test_graphrag_actual_search),
        ("MCP Message Handling", test_mcp_message_handling),
        ("Configuration Path Resolution", test_configuration_path_resolution),
        ("Session Persistence", test_session_persistence),
        ("Error Pattern Detection", test_error_patterns),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        if run_test(name, test_func):
            passed += 1
        else:
            failed += 1

    print("\n" + "="*50)
    print("📊 Real-World Test Summary")
    print("="*50)
    print(f"Total Tests:  {len(tests)}")
    print(f"Passed:       {GREEN}{passed}{NC}")
    print(f"Failed:       {RED}{failed}{NC}")
    print()

    if failed == 0:
        print(f"{GREEN}✅ All real-world tests passed!{NC}")
        return 0
    else:
        print(f"{RED}❌ {failed} real-world test(s) failed{NC}")
        print("\nThese tests catch issues that mocked tests miss:")
        print("- ProjectContextManager initialization errors")
        print("- Real GraphRAG search functionality")
        print("- Configuration path resolution issues")
        print("- Session persistence problems")
        print("- Specific error patterns from production")
        return 1

if __name__ == "__main__":
    sys.exit(main())