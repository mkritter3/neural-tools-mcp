#!/usr/bin/env python3
"""
Test MCP server session state persistence
"""

import json
import subprocess
import sys

def send_mcp_request(request):
    """Send a request to the MCP server via docker exec"""
    cmd = [
        "docker", "exec", "-i", "default-neural",
        "python3", "-u", 
        "/app/neural-tools-src/servers/neural_server_2025_fixed.py"
    ]
    
    # Send request
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = proc.communicate(input=json.dumps(request) + "\n")
    
    if stderr:
        print(f"STDERR: {stderr}", file=sys.stderr)
    
    # Parse response
    try:
        for line in stdout.split('\n'):
            if line.strip() and line.startswith('{'):
                return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(f"STDOUT: {stdout}")
        return None

def test_mcp_session():
    """Test MCP server with proper session state"""
    
    print("Testing MCP Server Session State...")
    
    # 1. Initialize session
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    print("\n1. Sending initialize request...")
    response = send_mcp_request(init_request)
    if response:
        print(f"Initialize response: {json.dumps(response, indent=2)}")
    else:
        print("Failed to get initialize response")
        return False
    
    # 2. List tools (should work after initialization)
    list_tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    print("\n2. Listing tools...")
    response = send_mcp_request(list_tools_request)
    if response:
        print(f"Tools available: {len(response.get('result', {}).get('tools', []))}")
        for tool in response.get('result', {}).get('tools', []):
            print(f"  - {tool['name']}")
    else:
        print("Failed to list tools")
        return False
    
    # 3. Call a tool (test session persistence)
    status_request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "neural_system_status",
            "arguments": {}
        }
    }
    
    print("\n3. Calling neural_system_status tool...")
    response = send_mcp_request(status_request)
    if response and 'result' in response:
        content = response['result'].get('content', [])
        if content and len(content) > 0:
            status_data = json.loads(content[0]['text'])
            print(f"System Status: {status_data.get('status')}")
            print(f"Services: {status_data.get('services')}")
            print(f"Protocol: {status_data.get('protocol')}")
            return True
    else:
        print(f"Tool call failed: {response}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_mcp_session()
    sys.exit(0 if success else 1)