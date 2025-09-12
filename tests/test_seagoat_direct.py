#!/usr/bin/env python3
"""
Direct test of SeaGOAT tools via MCP protocol.
"""

import subprocess
import json
import time

def test_seagoat_tools():
    """Test SeaGOAT tools specifically."""
    print("=== Testing SeaGOAT Tools Directly ===")
    
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
        
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()
        response_line = process.stdout.readline().strip()
        print(f"Initialize: {json.loads(response_line)['result']['serverInfo']['name']}")

        # Send initialized notification  
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        process.stdin.flush()
        
        # Test seagoat_server_status
        print("\n--- Testing seagoat_server_status ---")
        request = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "seagoat_server_status", "arguments": {}}
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline().strip()
        response = json.loads(response_line)
        
        if "result" in response:
            content = json.loads(response["result"]["content"][0]["text"])
            print(f"Status: {content.get('status')}")
            if 'seagoat_response' in content:
                sg_resp = content['seagoat_response']
                print(f"Version: {sg_resp.get('version')}")
                stats = sg_resp.get('stats', {})
                chunks = stats.get('chunks', {})
                print(f"Chunks: {chunks.get('analyzed', 0)} analyzed, {chunks.get('unanalyzed', 0)} unanalyzed")
        else:
            print(f"Error: {response}")
        
        # Test semantic_code_search
        print("\n--- Testing semantic_code_search ---")
        request = {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "semantic_code_search", "arguments": {"query": "MCP neural tools", "limit": 3}}
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline().strip()
        response = json.loads(response_line)
        
        if "result" in response:
            content = json.loads(response["result"]["content"][0]["text"])
            print(f"Status: {content.get('status')}")
            results = content.get('results', [])
            print(f"Results: {len(results)} found")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result.get('path', 'unknown')} (score: {result.get('score', 0):.3f})")
                print(f"     {result.get('snippet', '')[:80]}...")
        else:
            print(f"Error: {response}")
            
        # Test seagoat_index_project
        print("\n--- Testing seagoat_index_project ---")
        request = {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call", 
            "params": {"name": "seagoat_index_project", "arguments": {"force_reindex": False}}
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        response_line = process.stdout.readline().strip()
        response = json.loads(response_line)
        
        if "result" in response:
            content = json.loads(response["result"]["content"][0]["text"])
            print(f"Status: {content.get('status')}")
            print(f"Message: {content.get('message', 'No message')}")
        else:
            print(f"Error: {response}")

        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        process.terminate()
        process.wait()

def main():
    print("🧪 SeaGOAT Direct Test")
    print("=" * 30)
    
    success = test_seagoat_tools()
    
    print("\n" + "=" * 30)
    if success:
        print("🎉 SeaGOAT test completed!")
    else:
        print("⚠️ SeaGOAT test failed")

if __name__ == "__main__":
    main()