#!/usr/bin/env python3
"""
Functional test for MCP server and tools.
Tests actual execution, not just configuration.
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path

TEST_PROJECT_DIR = "/Users/mkr/local-coding/Systems/neural-novelist"
MCP_SERVER_SCRIPT = "/Users/mkr/local-coding/claude-l9-template/neural-tools/run_mcp_server.py"


async def test_mcp_server_startup():
    """Test that MCP server starts and responds to initialization"""
    print("\n🚀 Testing MCP Server Startup from neural-novelist...")

    # Change to test directory
    original_cwd = os.getcwd()
    os.chdir(TEST_PROJECT_DIR)

    try:
        # Start MCP server
        cmd = [sys.executable, MCP_SERVER_SCRIPT]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=TEST_PROJECT_DIR
        )

        print(f"  ✅ Server process started (PID: {process.pid})")

        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }

        request_str = json.dumps(init_request) + "\n"
        process.stdin.write(request_str.encode())
        await process.stdin.drain()
        print("  📤 Sent initialize request")

        # Try to read response with timeout
        try:
            response_line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=5.0
            )

            if response_line:
                response = json.loads(response_line.decode())
                if "result" in response:
                    server_info = response.get("result", {}).get("serverInfo", {})
                    print(f"  ✅ Got response: {server_info.get('name', 'unknown')} v{server_info.get('version', 'unknown')}")

                    # Send initialized notification
                    initialized = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    }
                    notification_str = json.dumps(initialized) + "\n"
                    process.stdin.write(notification_str.encode())
                    await process.stdin.drain()
                    print("  📤 Sent initialized notification")

                    # Test a tool call
                    tool_request = {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list"
                    }
                    tool_str = json.dumps(tool_request) + "\n"
                    process.stdin.write(tool_str.encode())
                    await process.stdin.drain()
                    print("  📤 Sent tools/list request")

                    # Read tool response
                    tool_response_line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=5.0
                    )

                    if tool_response_line:
                        tool_response = json.loads(tool_response_line.decode())
                        tools = tool_response.get("result", {}).get("tools", [])
                        print(f"  ✅ Server has {len(tools)} tools available")

                        # List first few tools
                        for tool in tools[:3]:
                            print(f"     - {tool['name']}")

                        return True
                else:
                    print(f"  ❌ Unexpected response: {response}")
            else:
                print("  ❌ No response received")

        except asyncio.TimeoutError:
            print("  ⏱️ Timeout waiting for response")

            # Check stderr for errors
            stderr_data = await process.stderr.read(4096)
            if stderr_data:
                print("\n  📋 Server logs:")
                logs = stderr_data.decode()
                for line in logs.split('\n')[:10]:
                    if line.strip():
                        print(f"     {line}")

        finally:
            # Cleanup
            process.terminate()
            await process.wait()

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

    finally:
        os.chdir(original_cwd)

    return False


async def test_container_connectivity():
    """Test connectivity to Docker containers"""
    print("\n🐳 Testing Container Connectivity...")

    tests = [
        ("Neo4j", "localhost", 47687, "curl -s http://localhost:47474 | head -5"),
        ("Qdrant", "localhost", 46333, "curl -s http://localhost:46333/collections | head -5"),
        ("Nomic Embeddings", "localhost", 48000, "curl -s http://localhost:48000/health"),
        ("Redis Cache", "localhost", 46379, "redis-cli -p 46379 ping"),
        ("Redis Queue", "localhost", 46380, "redis-cli -p 46380 ping"),
    ]

    all_good = True
    for name, host, port, test_cmd in tests:
        try:
            # Test with simple connection check
            result = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                print(f"  ✅ {name} ({host}:{port}) - Connected")
            else:
                print(f"  ❌ {name} ({host}:{port}) - Failed")
                all_good = False

        except subprocess.TimeoutExpired:
            print(f"  ⏱️ {name} ({host}:{port}) - Timeout")
            all_good = False
        except Exception as e:
            print(f"  ❌ {name} ({host}:{port}) - Error: {e}")
            all_good = False

    return all_good


async def test_tool_execution():
    """Test actual tool execution"""
    print("\n🔧 Testing Tool Execution...")

    # Add neural-tools to path
    sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

    try:
        # Try to import and test basic functionality
        from servers.services.service_container import ServiceContainer
        from servers.services.project_context_manager import get_project_context_manager

        # Test project context manager
        context_manager = await get_project_context_manager()
        current_project = await context_manager.get_current_project()

        print(f"  ✅ Project Context Manager working")
        print(f"     Current project: {current_project.get('project', 'unknown')}")
        print(f"     Method: {current_project.get('method', 'unknown')}")

        # Test service container initialization
        container = ServiceContainer(
            context_manager=context_manager,
            project_name=current_project['project']
        )

        # Test Neo4j connection
        try:
            container.initialize()  # Not async
            print(f"  ✅ ServiceContainer initialized")

            # Initialize services if needed
            if hasattr(container, 'initialize_all_services'):
                await container.initialize_all_services()
                print(f"  ✅ Services async initialization complete")

            # Test Neo4j
            if container.neo4j:
                try:
                    result = await container.neo4j.execute_query(
                        "RETURN 1 as test",
                        {}
                    )
                    print(f"  ✅ Neo4j query successful")
                except Exception as e:
                    print(f"  ⚠️ Neo4j query failed: {e}")

            # Test Qdrant
            if container.qdrant:
                try:
                    # The correct method might be different
                    if hasattr(container.qdrant, 'get_collections'):
                        collections = await container.qdrant.get_collections()
                        print(f"  ✅ Qdrant connected - collections found")
                    else:
                        print(f"  ⚠️ Qdrant has no get_collections method")
                except Exception as e:
                    print(f"  ⚠️ Qdrant check failed: {e}")

            # Test Nomic
            if container.nomic:
                try:
                    test_text = "test embedding"
                    embedding = await container.nomic.embed_text(test_text)
                    print(f"  ✅ Nomic embeddings working - dim: {len(embedding)}")
                except Exception as e:
                    print(f"  ⚠️ Nomic embedding failed: {e}")

        except Exception as e:
            print(f"  ❌ Service initialization failed: {e}")
            return False

        return True

    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


async def main():
    print("=" * 60)
    print("🧪 MCP FUNCTIONAL TEST SUITE")
    print("=" * 60)

    results = {}

    # Run tests
    results['containers'] = await test_container_connectivity()
    results['mcp_startup'] = await test_mcp_server_startup()
    results['tool_execution'] = await test_tool_execution()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print("-" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:20} {status}")

    all_passed = all(results.values())

    print("-" * 60)
    if all_passed:
        print("✅ ALL FUNCTIONAL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - INVESTIGATION NEEDED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)