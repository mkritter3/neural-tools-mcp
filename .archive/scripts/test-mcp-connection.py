#!/usr/bin/env python3
"""Test MCP server connection locally"""

import subprocess
import json
import time

def test_mcp_server():
    """Test that MCP server responds to initialization"""
    
    # Start MCP server process
    env = {
        'PYTHONPATH': '/Users/mkr/local-coding/claude-l9-template/.claude/neural-system',
        'NEURAL_L9_MODE': '1',
        'USE_SINGLE_QODO_MODEL': '1'
    }
    
    process = subprocess.Popen(
        ['python3', '.claude/neural-system/mcp_neural_server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**env},
        text=True
    )
    
    # Send initialization request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {}
        }
    }
    
    try:
        # Send request
        process.stdin.write(json.dumps(request) + '\n')
        process.stdin.flush()
        
        # Wait for response
        time.sleep(1)
        
        # Try to read response
        process.stdin.close()
        stdout, stderr = process.communicate(timeout=2)
        
        if "Neural Flow systems initialized successfully" in stderr:
            print("✅ MCP server started successfully")
            print("✅ L9 Neural Flow system ready")
            return True
        else:
            print("❌ MCP server failed to initialize")
            print(f"Stderr: {stderr[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        process.terminate()
        return False
    finally:
        process.terminate()

if __name__ == "__main__":
    success = test_mcp_server()
    exit(0 if success else 1)