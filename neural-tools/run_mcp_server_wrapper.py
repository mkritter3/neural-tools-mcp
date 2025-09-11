#!/usr/bin/env python3
"""
Wrapper script for MCP server that ensures clean startup.
This replaces direct calls to run_mcp_server.py in .mcp.json
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def wait_for_services(max_wait=30):
    """Wait for Docker services to be ready"""
    import socket
    
    services = {
        "Neo4j": ("localhost", 47687),
        "Qdrant": ("localhost", 46333),
    }
    
    start_time = time.time()
    sys.stderr.write("⏳ Waiting for services to be ready...\n")
    
    while time.time() - start_time < max_wait:
        all_ready = True
        for service, (host, port) in services.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result != 0:
                all_ready = False
                break
        
        if all_ready:
            sys.stderr.write("✅ All services ready!\n")
            return True
        
        time.sleep(1)
    
    sys.stderr.write("⚠️  Services not ready after 30s, attempting anyway...\n")
    return False

def main():
    # Log startup
    pid = os.getpid()
    sys.stderr.write(f"🚀 Starting MCP wrapper (PID: {pid})\n")
    
    # Wait for services to be ready
    wait_for_services()
    
    # Launch the actual MCP server
    mcp_script = Path(__file__).parent / "run_mcp_server.py"
    
    # Pass through all environment variables and arguments
    env = os.environ.copy()
    
    # Ensure critical variables are set
    if 'NEO4J_PASSWORD' not in env:
        env['NEO4J_PASSWORD'] = 'graphrag-password'
    if 'PROJECT_NAME' not in env:
        env['PROJECT_NAME'] = 'default'
    
    # Execute the real MCP server (replaces this process)
    sys.stderr.write(f"🔄 Launching MCP server: {mcp_script}\n")
    os.execve(sys.executable, [sys.executable, str(mcp_script)] + sys.argv[1:], env)

if __name__ == "__main__":
    main()