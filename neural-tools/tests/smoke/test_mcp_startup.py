#!/usr/bin/env python3
"""
MCP Server Startup Smoke Test
Quick test to verify MCP server starts and responds
"""

import json
import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


async def test_mcp_startup():
    """Test that MCP server starts up correctly"""
    print("\n🚀 Testing MCP Server Startup...")

    start_time = time.time()

    # Start MCP server
    cmd = [sys.executable, str(Path(__file__).parent.parent.parent / "run_mcp_server.py")]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        print(f"  ✅ Process started (PID: {process.pid})")

        # Send minimal initialization
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "smoke-test", "version": "1.0"}
            }
        }

        request_str = json.dumps(init_request) + "\n"
        process.stdin.write(request_str.encode())
        await process.stdin.drain()

        # Wait for response (with timeout)
        response_line = await asyncio.wait_for(
            process.stdout.readline(),
            timeout=10.0  # Generous timeout for startup
        )

        if response_line:
            response = json.loads(response_line.decode())
            if "result" in response:
                startup_time = time.time() - start_time
                server_info = response["result"].get("serverInfo", {})
                print(f"  ✅ Server responded: {server_info.get('name', 'unknown')}")
                print(f"  ⏱️ Startup time: {startup_time:.2f}s")

                if startup_time > 5.0:
                    print(f"  ⚠️ Startup took longer than expected (>5s)")

                # Cleanup
                process.terminate()
                await process.wait()

                print("\n✅ MCP Startup Smoke Test PASSED")
                return True
            else:
                print(f"  ❌ Unexpected response: {response}")
        else:
            print("  ❌ No response received")

        process.terminate()
        await process.wait()
        return False

    except asyncio.TimeoutError:
        print("  ❌ Timeout waiting for server startup (>10s)")
        if 'process' in locals():
            process.terminate()
            await process.wait()
        return False
    except Exception as e:
        print(f"  ❌ Startup failed: {e}")
        if 'process' in locals():
            process.terminate()
            await process.wait()
        return False


def main():
    """Run the test"""
    success = asyncio.run(test_mcp_startup())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()