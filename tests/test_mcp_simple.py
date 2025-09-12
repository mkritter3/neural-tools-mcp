#!/usr/bin/env python3
"""
Simple MCP test using direct JSON-RPC communication.
"""

import subprocess
import json
import time

def test_mcp_tools():
    """Test MCP tools via JSON-RPC."""
    print("=== Testing MCP Tools via JSON-RPC ===")
    
    # Start the MCP server
    process = subprocess.Popen([
        "docker", "exec", "-i", "default-neural", 
        "python3", "-u", "/app/neural-tools-src/servers/neural_server_2025.py"
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"roots": {"listChanged": False}, "sampling": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        print("Sending initialize request...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline().strip()
        if response_line:
            response = json.loads(response_line)
            print(f"✓ Initialize response: {response.get('result', {}).get('serverInfo', {}).get('name', 'unknown')}")
        
        # Send initialized notification
        initialized_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        print("Sending initialized notification...")
        process.stdin.write(json.dumps(initialized_notif) + "\n")
        process.stdin.flush()
        
        # List tools
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        print("Sending tools/list request...")
        process.stdin.write(json.dumps(list_tools_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline().strip()
        if response_line:
            response = json.loads(response_line)
            if "result" in response:
                tools = response["result"]["tools"] if "tools" in response["result"] else response["result"]
                print(f"✓ Found {len(tools)} tools")
                for tool in tools:
                    print(f"  - {tool['name']}")
            else:
                print(f"Tools list error: {response}")
        
        # Test neural_system_status tool
        status_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "neural_system_status",
                "arguments": {}
            }
        }
        
        print("\\nTesting neural_system_status tool...")
        process.stdin.write(json.dumps(status_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline().strip()
        if response_line:
            response = json.loads(response_line)
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                data = json.loads(content)
                print(f"✓ System status: {data.get('status', 'unknown')}")
                services = data.get('services', {})
                for service, active in services.items():
                    status = "✓" if active else "✗"
                    print(f"  {status} {service}")
            else:
                print(f"Status error: {response}")
        
        # Test seagoat_server_status tool
        seagoat_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "seagoat_server_status",
                "arguments": {}
            }
        }
        
        print("\\nTesting seagoat_server_status tool...")
        process.stdin.write(json.dumps(seagoat_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline().strip()
        if response_line:
            response = json.loads(response_line)
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                data = json.loads(content)
                print(f"✓ SeaGOAT status: {data.get('status', 'unknown')}")
                if "seagoat_response" in data:
                    seagoat_data = data["seagoat_response"]
                    print(f"  Version: {seagoat_data.get('version', 'unknown')}")
                    stats = seagoat_data.get('stats', {})
                    if 'chunks' in stats:
                        chunks = stats['chunks']
                        print(f"  Analyzed: {chunks.get('analyzed', 0)}, Unanalyzed: {chunks.get('unanalyzed', 0)}")
            else:
                print(f"SeaGOAT status error: {response}")
        
        # Test semantic_code_search tool
        search_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "semantic_code_search",
                "arguments": {
                    "query": "MCP neural tools",
                    "limit": 3
                }
            }
        }
        
        print("\\nTesting semantic_code_search tool...")
        process.stdin.write(json.dumps(search_request) + "\n")
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline().strip()
        if response_line:
            response = json.loads(response_line)
            if "result" in response and "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                data = json.loads(content)
                print(f"✓ Search status: {data.get('status', 'unknown')}")
                if "results" in data:
                    results = data["results"]
                    print(f"  Found {len(results)} results")
                    for i, result in enumerate(results[:2]):
                        path = result.get('path', 'unknown')
                        score = result.get('score', 0)
                        print(f"    {i+1}. {path} (score: {score:.3f})")
            else:
                print(f"Search error: {response}")
        
        return True
        
    except Exception as e:
        print(f"✗ MCP test error: {e}")
        return False
    finally:
        process.terminate()
        process.wait()

def main():
    """Run the test."""
    print("🧪 MCP Simple Test")
    print("=" * 30)
    
    success = test_mcp_tools()
    
    print("\\n" + "=" * 30)
    if success:
        print("🎉 MCP test completed!")
    else:
        print("⚠️ MCP test failed")

if __name__ == "__main__":
    main()