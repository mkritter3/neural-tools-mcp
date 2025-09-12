#!/usr/bin/env python3
"""
Test script to validate MCP neural tools server functionality.
Tests SeaGOAT integration and other neural tools via MCP protocol.
"""

import json
import asyncio
import subprocess
import sys
from typing import Dict, Any, List

class MCPTester:
    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process = None
        self.request_id = 1

    async def start_server(self):
        """Start the MCP server process."""
        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print(f"✓ Started MCP server: {' '.join(self.server_command)}")

    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server."""
        if not self.process:
            raise RuntimeError("Server not started")

        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params:
            request["params"] = params

        self.request_id += 1
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from server")
        
        return json.loads(response_line.decode().strip())

    async def test_initialization(self) -> bool:
        """Test MCP server initialization."""
        try:
            print("\n=== Testing MCP Server Initialization ===")
            
            # Send initialize request
            response = await self.send_request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "roots": {
                        "listChanged": False
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "mcp-test-client",
                    "version": "1.0.0"
                }
            })
            
            if "result" in response:
                print("✓ Server initialization successful")
                print(f"  Server: {response['result'].get('serverInfo', {}).get('name', 'unknown')}")
                print(f"  Version: {response['result'].get('protocolVersion', 'unknown')}")
                return True
            else:
                print(f"✗ Server initialization failed: {response}")
                return False
                
        except Exception as e:
            print(f"✗ Initialization error: {e}")
            return False

    async def test_list_tools(self) -> List[str]:
        """Test listing available tools."""
        try:
            print("\n=== Testing Tool Listing ===")
            
            # Send initialized notification first
            await self.send_request("initialized", {})
            
            # Try different method names with proper parameters
            for method in ["tools/list", "listTools"]:
                print(f"  Trying method: {method}")
                response = await self.send_request(method, {})
                
                if "error" not in response:
                    if "result" in response:
                        if isinstance(response["result"], list):
                            # Direct list of tools
                            tools = response["result"]
                        elif "tools" in response["result"]:
                            # Tools wrapped in result object
                            tools = response["result"]["tools"]
                        else:
                            print(f"  Unexpected result format: {response['result']}")
                            continue
                            
                        tool_names = [tool["name"] for tool in tools]
                        print(f"✓ Found {len(tools)} tools using method '{method}':")
                        for tool in tools:
                            print(f"  - {tool['name']}: {tool['description']}")
                        return tool_names
                else:
                    print(f"  Method '{method}' failed: {response['error']['message']}")
            
            print("✗ All tool listing methods failed")
            return []
                
        except Exception as e:
            print(f"✗ Tool listing error: {e}")
            return []

    async def test_tool_call(self, tool_name: str, arguments: Dict[str, Any] = None) -> bool:
        """Test calling a specific tool."""
        try:
            print(f"\n=== Testing Tool: {tool_name} ===")
            
            params = {
                "name": tool_name
            }
            if arguments:
                params["arguments"] = arguments
            
            # Try different method names for tool calling
            for method in ["tools/call", "callTool"]:
                print(f"  Trying method: {method}")
                response = await self.send_request(method, params)
                
                if "error" not in response and "result" in response:
                    result = response["result"]
                    print(f"✓ Tool '{tool_name}' executed successfully using method '{method}'")
                    
                    # Display results
                    if "content" in result:
                        for content in result["content"]:
                            if content["type"] == "text":
                                # Try to parse as JSON for pretty printing
                                try:
                                    data = json.loads(content["text"])
                                    print("  Result:")
                                    print(json.dumps(data, indent=4))
                                except:
                                    print(f"  Result: {content['text'][:500]}...")
                    return True
                else:
                    if "error" in response:
                        print(f"  Method '{method}' failed: {response['error']['message']}")
                    else:
                        print(f"  Method '{method}' returned unexpected response: {response}")
            
            print(f"✗ All tool call methods failed for '{tool_name}'")
            return False
                
        except Exception as e:
            print(f"✗ Tool '{tool_name}' error: {e}")
            return False

    async def cleanup(self):
        """Cleanup server process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            print("\n✓ Server process terminated")

async def main():
    """Main test function."""
    print("🧪 MCP Neural Tools Server Test Suite")
    print("=" * 50)
    
    # Server command for Docker
    server_command = [
        "docker", "exec", "-i", "default-neural", 
        "python3", "-u", "/app/neural-tools-src/servers/neural_server_2025.py"
    ]
    
    tester = MCPTester(server_command)
    
    try:
        # Start server
        await tester.start_server()
        
        # Test initialization
        if not await tester.test_initialization():
            print("❌ Server initialization failed - aborting tests")
            return
        
        # Test tool listing
        tools = await tester.test_list_tools()
        
        # Test specific tools requested
        test_results = {}
        
        if "neural_system_status" in tools:
            test_results["neural_system_status"] = await tester.test_tool_call("neural_system_status")
        
        if "seagoat_server_status" in tools:
            test_results["seagoat_server_status"] = await tester.test_tool_call("seagoat_server_status")
        
        if "semantic_code_search" in tools:
            test_results["semantic_code_search"] = await tester.test_tool_call(
                "semantic_code_search", 
                {"query": "MCP neural tools", "limit": 5}
            )
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        passed = sum(1 for result in test_results.values() if result)
        total = len(test_results)
        print(f"  Passed: {passed}/{total}")
        
        for tool_name, result in test_results.items():
            status = "✓" if result else "✗"
            print(f"  {status} {tool_name}")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed - check logs above")
    
    except Exception as e:
        print(f"❌ Test suite error: {e}")
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())