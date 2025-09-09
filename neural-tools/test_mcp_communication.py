#!/usr/bin/env python3
"""
Test script for MCP communication with the neural server
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

async def test_mcp_communication():
    """Test basic MCP communication"""
    print("🧪 Testing MCP Neural Server Communication")
    
    # Environment setup for localhost connectivity
    env = {
        'QDRANT_HOST': 'localhost',
        'QDRANT_HTTP_PORT': '6681',
        'NEO4J_HOST': 'localhost', 
        'NEO4J_HTTP_PORT': '7475',
        'NEO4J_BOLT_PORT': '7688',
        'PROJECT_NAME': 'default'
    }
    
    # Start the server process
    server_path = Path(__file__).parent / "src/servers/neural_server_stdio.py"
    
    try:
        # Start server with proper environment
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(server_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**env, **dict(os.environ)} if 'os' in globals() else env
        )
        
        # Test 1: Initialize request
        initialize_request = {
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
        
        print("📡 Sending initialize request...")
        request_json = json.dumps(initialize_request) + "\n"
        process.stdin.write(request_json.encode())
        await process.stdin.drain()
        
        # Read response
        response_line = await process.stdout.readline()
        if response_line:
            response = json.loads(response_line.decode().strip())
            print(f"✅ Initialize response: {response.get('result', {}).get('serverInfo', {}).get('name', 'unknown')}")
        
        # Test 2: List tools
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        print("📋 Requesting tools list...")
        request_json = json.dumps(list_tools_request) + "\n"
        process.stdin.write(request_json.encode())
        await process.stdin.drain()
        
        # Read response
        response_line = await process.stdout.readline()
        if response_line:
            response = json.loads(response_line.decode().strip())
            tools = response.get('result', {}).get('tools', [])
            print(f"🔧 Available tools: {len(tools)}")
            for tool in tools[:3]:  # Show first 3 tools
                print(f"   - {tool.get('name')}: {tool.get('description', '')[:50]}...")
        
        # Test 3: Neural system status
        status_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "neural_system_status",
                "arguments": {}
            }
        }
        
        print("🏥 Checking system status...")
        request_json = json.dumps(status_request) + "\n"
        process.stdin.write(request_json.encode())
        await process.stdin.drain()
        
        # Read response
        response_line = await process.stdout.readline()
        if response_line:
            response = json.loads(response_line.decode().strip())
            content = response.get('result', {}).get('content', [{}])[0].get('text', '{}')
            status_data = json.loads(content)
            print(f"📊 System status: {status_data.get('status', 'unknown')}")
            print(f"🔌 Services: {status_data.get('healthy_services', 0)}/{status_data.get('total_services', 0)}")
            
            services = status_data.get('services', {})
            for service, healthy in services.items():
                status_emoji = "✅" if healthy else "❌"
                print(f"   {status_emoji} {service}")
        
        # Clean shutdown
        process.stdin.close()
        await process.wait()
        
        print("🎉 MCP communication test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if 'process' in locals():
            process.terminate()
            await process.wait()

if __name__ == "__main__":
    import os
    asyncio.run(test_mcp_communication())