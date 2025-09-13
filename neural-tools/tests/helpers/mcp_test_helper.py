#!/usr/bin/env python3
"""
MPCTestHelper - Reusable helper for end-to-end MCP server testing
Implements L9 standards for production-grade testing via subprocess
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List
import time
from contextlib import asynccontextmanager

class MPCTestHelper:
    """
    Async context manager for MCP server subprocess lifecycle management.
    Provides clean setup/teardown and JSON-RPC communication.
    """

    def __init__(self, server_path: Optional[Path] = None, timeout: float = 10.0):
        """
        Initialize the MCP test helper.

        Args:
            server_path: Path to run_mcp_server.py (auto-detected if None)
            timeout: Default timeout for operations in seconds
        """
        if server_path is None:
            # Auto-detect server path relative to test location
            test_dir = Path(__file__).parent.parent
            server_path = test_dir.parent / "run_mcp_server.py"

        if not server_path.exists():
            raise FileNotFoundError(f"MCP server not found at {server_path}")

        self.server_path = server_path
        self.timeout = timeout
        self.process = None
        self.reader = None
        self.writer = None
        self.message_id = 0
        self.initialized = False
        self.stderr_log = []
        self._request_lock = asyncio.Lock()  # Add lock for concurrent requests

    async def __aenter__(self):
        """Async context manager entry - starts the server."""
        await self.start_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - stops the server."""
        await self.stop_server()

    async def start_server(self):
        """Start the MCP server subprocess using asyncio."""
        try:
            # Run from the claude-l9-template directory for proper project detection
            project_dir = Path.home() / "local-coding" / "claude-l9-template"
            if not project_dir.exists():
                # Fallback to running from neural-tools directory
                project_dir = self.server_path.parent

            # Use asyncio.create_subprocess_exec for native async support
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(self.server_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(project_dir)
            )

            # Initialize the server with protocol handshake
            await self._initialize_server()

            # Start stderr monitoring task
            asyncio.create_task(self._monitor_stderr())

        except Exception as e:
            raise RuntimeError(f"Failed to start MCP server: {e}")

    async def stop_server(self):
        """Stop the MCP server subprocess gracefully."""
        if self.process:
            try:
                # Send shutdown signal
                self.process.terminate()

                # Wait for graceful shutdown with timeout
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill if graceful shutdown fails
                    self.process.kill()
                    await self.process.wait()

            except Exception as e:
                print(f"Warning: Error stopping server: {e}")
            finally:
                self.process = None
                self.initialized = False

    async def _initialize_server(self):
        """Perform the MCP protocol initialization handshake."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "mcp-test-helper",
                    "version": "1.0.0"
                }
            }
        }

        response = await self._send_request(init_request)

        if "error" in response:
            raise RuntimeError(f"Server initialization failed: {response['error']}")

        # Send the "initialized" notification (no id, no response expected)
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }

        # Send notification without waiting for response
        if self.process:
            notification_str = json.dumps(initialized_notification) + "\n"
            self.process.stdin.write(notification_str.encode())
            await self.process.stdin.drain()

            # Give the server a moment to process the notification
            await asyncio.sleep(0.1)

        self.initialized = True
        return response

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool and return the response.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as a dictionary

        Returns:
            The tool response dictionary
        """
        if not self.initialized:
            raise RuntimeError("Server not initialized. Call start_server() first.")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        response = await self._send_request(request)

        # Check for error response
        if "error" in response:
            return response

        # Extract the result content if present
        if "result" in response:
            result = response["result"]

            # Handle the result based on its structure
            if isinstance(result, dict):
                # If result has content array
                if "content" in result and isinstance(result["content"], list):
                    contents = result["content"]
                    if len(contents) > 0 and "text" in contents[0]:
                        # Parse the JSON text content
                        try:
                            return json.loads(contents[0]["text"])
                        except json.JSONDecodeError:
                            # If not JSON, return as-is
                            return {"text": contents[0]["text"]}
                return result

            # MCP returns content as an array of TextContent objects
            elif isinstance(result, list) and len(result) > 0:
                # Get the first content item's text
                if isinstance(result[0], dict) and "text" in result[0]:
                    # Parse the JSON text content
                    try:
                        return json.loads(result[0]["text"])
                    except json.JSONDecodeError:
                        # If not JSON, return as-is
                        return {"text": result[0]["text"]}

            return result

        return response

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            request: The request dictionary

        Returns:
            The response dictionary
        """
        if not self.process:
            raise RuntimeError("Server process not running")

        # Use lock to prevent concurrent reads/writes
        async with self._request_lock:
            # Send request
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()

            # Read response with timeout
            try:
                response_bytes = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                # Dump stderr for debugging
                stderr_content = "\n".join(self.stderr_log[-20:])  # Last 20 lines
                raise TimeoutError(
                    f"Server timeout after {self.timeout}s. Request: {request['method']}\n"
                    f"Recent stderr:\n{stderr_content}"
                )

            if not response_bytes:
                raise RuntimeError("Server returned empty response")

            try:
                response = json.loads(response_bytes.decode())
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON response: {response_bytes.decode()[:200]}... Error: {e}")

            return response

    async def _monitor_stderr(self):
        """Monitor stderr for debugging information."""
        if not self.process or not self.process.stderr:
            return

        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                decoded_line = line.decode().strip()
                if decoded_line:
                    self.stderr_log.append(decoded_line)
                    # Keep only last 100 lines
                    if len(self.stderr_log) > 100:
                        self.stderr_log.pop(0)
        except Exception:
            pass  # Ignore errors in monitoring task

    def _next_id(self) -> int:
        """Get next message ID for JSON-RPC."""
        self.message_id += 1
        return self.message_id

    def get_stderr_log(self) -> List[str]:
        """Get the stderr log for debugging."""
        return self.stderr_log.copy()

    async def call_multiple_tools(self, tool_calls: List[tuple]) -> List[Dict[str, Any]]:
        """
        Call multiple tools sequentially.

        Args:
            tool_calls: List of (tool_name, arguments) tuples

        Returns:
            List of responses in the same order
        """
        results = []
        for tool_name, arguments in tool_calls:
            result = await self.call_tool(tool_name, arguments)
            results.append(result)
        return results

    async def call_tools_concurrently(self, tool_calls: List[tuple]) -> List[Dict[str, Any]]:
        """
        Call multiple tools concurrently using asyncio.gather.

        Args:
            tool_calls: List of (tool_name, arguments) tuples

        Returns:
            List of responses in the same order
        """
        tasks = [
            self.call_tool(tool_name, arguments)
            for tool_name, arguments in tool_calls
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def get_project_paths():
        """Get project paths dynamically based on environment."""
        # Use environment variables or relative paths
        base_dir = os.environ.get("TEST_BASE_DIR", str(Path.home() / "local-coding"))

        return {
            "claude-l9-template": Path(base_dir) / "claude-l9-template",
            "neural-novelist": Path(base_dir) / "Systems" / "neural-novelist"
        }


# Convenience function for simple testing
@asynccontextmanager
async def mcp_server_session(timeout: float = 10.0):
    """
    Async context manager for a complete MCP server session.

    Usage:
        async with mcp_server_session() as helper:
            result = await helper.call_tool("neural_system_status", {})
    """
    helper = MPCTestHelper(timeout=timeout)
    try:
        await helper.start_server()
        yield helper
    finally:
        await helper.stop_server()