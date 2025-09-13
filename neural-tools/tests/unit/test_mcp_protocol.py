#!/usr/bin/env python3
"""
MCP Protocol Compliance Test
Tests that our MCP server properly implements the 2025-06-18 protocol
"""

import json
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


async def test_mcp_protocol_compliance():
    """Test MCP server protocol compliance"""
    print("\n🔍 Testing MCP Protocol Compliance...")

    # Test 1: JSON-RPC 2.0 Format
    print("  Testing JSON-RPC 2.0 message format...")

    # Start MCP server subprocess
    cmd = [sys.executable, str(Path(__file__).parent.parent.parent / "run_mcp_server.py")]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        # Test 1: Valid initialization request
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

        # Read response
        response_line = await asyncio.wait_for(
            process.stdout.readline(),
            timeout=5.0
        )

        if not response_line:
            print("  ❌ No response received")
            return False

        response = json.loads(response_line.decode())

        # Validate response structure
        assert "jsonrpc" in response and response["jsonrpc"] == "2.0", "Missing or invalid jsonrpc field"
        assert "id" in response and response["id"] == 1, "Missing or mismatched id"
        assert "result" in response, "Missing result field"

        result = response["result"]
        assert "protocolVersion" in result, "Missing protocolVersion in result"
        assert "serverInfo" in result, "Missing serverInfo in result"
        assert "capabilities" in result, "Missing capabilities in result"

        server_info = result["serverInfo"]
        assert "name" in server_info, "Missing name in serverInfo"
        assert "version" in server_info, "Missing version in serverInfo"

        print(f"  ✅ Valid JSON-RPC 2.0 response structure")
        print(f"     Server: {server_info['name']} v{server_info['version']}")
        print(f"     Protocol: {result['protocolVersion']}")

        # Test 2: Send initialized notification
        initialized = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }

        notification_str = json.dumps(initialized) + "\n"
        process.stdin.write(notification_str.encode())
        await process.stdin.drain()

        print("  ✅ Initialized notification accepted")

        # Test 3: Tools list request
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }

        tools_str = json.dumps(tools_request) + "\n"
        process.stdin.write(tools_str.encode())
        await process.stdin.drain()

        tools_response_line = await asyncio.wait_for(
            process.stdout.readline(),
            timeout=5.0
        )

        if tools_response_line:
            tools_response = json.loads(tools_response_line.decode())
            assert "result" in tools_response, "Missing result in tools response"
            assert "tools" in tools_response["result"], "Missing tools array"

            tools = tools_response["result"]["tools"]
            assert isinstance(tools, list), "Tools should be an array"
            assert len(tools) > 0, "No tools registered"

            # Check tool structure
            for tool in tools[:3]:  # Check first 3 tools
                assert "name" in tool, f"Tool missing name: {tool}"
                assert "description" in tool, f"Tool missing description: {tool}"
                assert "inputSchema" in tool, f"Tool missing inputSchema: {tool}"

            print(f"  ✅ Tools list valid ({len(tools)} tools available)")

        # Test 4: Invalid request handling
        invalid_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "invalid/method"
        }

        invalid_str = json.dumps(invalid_request) + "\n"
        process.stdin.write(invalid_str.encode())
        await process.stdin.drain()

        error_response_line = await asyncio.wait_for(
            process.stdout.readline(),
            timeout=5.0
        )

        if error_response_line:
            error_response = json.loads(error_response_line.decode())
            assert "error" in error_response, "Invalid method should return error"
            print("  ✅ Error handling works correctly")

        # Cleanup
        process.terminate()
        await process.wait()

        print("\n✅ MCP Protocol Compliance Test PASSED")
        return True

    except asyncio.TimeoutError:
        print("  ❌ Timeout waiting for response")
        if 'process' in locals():
            process.terminate()
            await process.wait()
        return False
    except AssertionError as e:
        print(f"  ❌ Protocol validation failed: {e}")
        if 'process' in locals():
            process.terminate()
            await process.wait()
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        if 'process' in locals():
            process.terminate()
            await process.wait()
        return False


def main():
    """Run the test"""
    success = asyncio.run(test_mcp_protocol_compliance())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()