#!/usr/bin/env python3
"""
Test script to verify MCP working directory fix.

This script simulates what happens when Claude starts an MCP server:
1. Starts from a specific directory (like neural-novelist)
2. Launches the MCP server as a subprocess
3. Sends initialization messages
4. Verifies the server captures the correct working directory
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path

# Test directories
TEST_PROJECT_DIR = "/Users/mkr/local-coding/Systems/neural-novelist"
MCP_SERVER_PATH = "/Users/mkr/local-coding/claude-l9-template/neural-tools/run_mcp_server.py"


async def test_mcp_initialization():
    """Test that MCP server captures correct working directory"""

    print(f"📍 Starting test from directory: {TEST_PROJECT_DIR}")

    # Change to test project directory (simulating where Claude starts)
    original_cwd = os.getcwd()
    os.chdir(TEST_PROJECT_DIR)

    try:
        # Start MCP server as subprocess (like Claude does)
        print(f"🚀 Launching MCP server from: {os.getcwd()}")

        # Build the command
        cmd = [sys.executable, MCP_SERVER_PATH]

        # Start the process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=TEST_PROJECT_DIR  # Explicitly set working directory
        )

        print("✅ MCP server started")

        # Send initialization request (following MCP protocol)
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

        # Send request
        request_str = json.dumps(init_request) + "\n"
        process.stdin.write(request_str.encode())
        await process.stdin.drain()

        print("📤 Sent initialize request")

        # Read response
        response_line = await process.stdout.readline()
        response = json.loads(response_line.decode())

        print(f"📥 Got initialize response: {response.get('result', {}).get('serverInfo', {})}")

        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }

        notification_str = json.dumps(initialized_notification) + "\n"
        process.stdin.write(notification_str.encode())
        await process.stdin.drain()

        print("📤 Sent initialized notification")

        # Give server time to process
        await asyncio.sleep(2)

        # Read stderr to check logs
        print("\n📋 Server logs:")
        print("-" * 50)

        # Read some stderr output
        stderr_task = asyncio.create_task(process.stderr.read(4096))
        try:
            stderr_output = await asyncio.wait_for(stderr_task, timeout=1.0)
            logs = stderr_output.decode()

            # Check if correct directory was captured
            if TEST_PROJECT_DIR in logs:
                print("✅ SUCCESS: Server captured correct working directory!")
                print(f"   Found '{TEST_PROJECT_DIR}' in logs")
            elif "neural-novelist" in logs:
                print("✅ SUCCESS: Server detected neural-novelist project!")
            else:
                print("❌ FAILURE: Server did not capture correct directory")
                print("   Logs:")
                print(logs)
        except asyncio.TimeoutError:
            print("⏱️ Timeout reading logs")

        # Cleanup
        process.terminate()
        await process.wait()

    finally:
        # Restore original directory
        os.chdir(original_cwd)
        print(f"\n📍 Restored directory to: {os.getcwd()}")


if __name__ == "__main__":
    print("🧪 Testing MCP Working Directory Fix")
    print("=" * 50)

    # Check if test directory exists
    if not Path(TEST_PROJECT_DIR).exists():
        print(f"❌ Test directory does not exist: {TEST_PROJECT_DIR}")
        print("   Please update TEST_PROJECT_DIR in this script")
        sys.exit(1)

    # Run the test
    asyncio.run(test_mcp_initialization())

    print("\n✨ Test complete!")