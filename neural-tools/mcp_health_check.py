#!/usr/bin/env python3
"""
Health check and diagnostic tool for MCP server instances.
Helps diagnose why new Claude instances can't connect.
"""

import os
import sys
import json
import psutil
import socket
import subprocess
from pathlib import Path
import time

def check_service_ports():
    """Check if all required services are listening on expected ports"""
    ports = {
        "Neo4j": 47687,
        "Qdrant": 46333,
        "Redis Cache": 46379,
        "Redis Queue": 46380,
        "Nomic Embeddings": 48000,
        "Indexer": 48080
    }
    
    print("🔍 Checking service ports:")
    all_good = True
    for service, port in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"  ✅ {service:20} on port {port:5} - LISTENING")
        else:
            print(f"  ❌ {service:20} on port {port:5} - NOT AVAILABLE")
            all_good = False
    
    return all_good

def check_mcp_processes():
    """Check MCP server processes and their states"""
    print("\n🔍 Checking MCP processes:")
    
    mcp_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'status']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'run_mcp_server.py' in ' '.join(cmdline):
                mcp_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not mcp_procs:
        print("  ⚠️  No MCP server processes found")
        return False
    
    print(f"  Found {len(mcp_procs)} MCP server process(es):")
    
    for proc in mcp_procs:
        pid = proc.info['pid']
        status = proc.info['status']
        create_time = proc.info['create_time']
        age_seconds = time.time() - create_time
        age_str = f"{age_seconds/60:.1f}m" if age_seconds > 60 else f"{age_seconds:.0f}s"
        
        # Check connections
        connections = []
        try:
            for conn in proc.connections():
                if conn.status == 'ESTABLISHED':
                    connections.append(f"{conn.laddr.port}->{conn.raddr.port}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        conn_str = f"Connections: {', '.join(connections[:3])}" if connections else "No active connections"
        print(f"    PID {pid:5} - Status: {status:10} Age: {age_str:6} - {conn_str}")
    
    return len(mcp_procs) > 0

def check_neo4j_auth():
    """Test Neo4j authentication"""
    print("\n🔍 Testing Neo4j authentication:")
    
    try:
        result = subprocess.run([
            'docker', 'exec', 'claude-l9-template-neo4j-1',
            'cypher-shell', '-u', 'neo4j', '-p', 'graphrag-password',
            'RETURN 1 as test'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("  ✅ Neo4j authentication successful")
            return True
        else:
            print(f"  ❌ Neo4j authentication failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ❌ Neo4j authentication timed out")
        return False
    except Exception as e:
        print(f"  ❌ Neo4j authentication error: {e}")
        return False

def check_mcp_config():
    """Check MCP configuration file"""
    print("\n🔍 Checking MCP configuration:")
    
    config_path = Path("/Users/mkr/local-coding/claude-l9-template/.mcp.json")
    if not config_path.exists():
        print(f"  ❌ MCP config not found at {config_path}")
        return False
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        if 'neural-tools' in config.get('mcpServers', {}):
            server_config = config['mcpServers']['neural-tools']
            print(f"  ✅ MCP server configured: {server_config.get('command')} {' '.join(server_config.get('args', []))}")
            
            # Check critical environment variables
            env = server_config.get('env', {})
            critical_vars = ['NEO4J_PASSWORD', 'PROJECT_NAME', 'NEO4J_URI']
            for var in critical_vars:
                value = env.get(var, 'NOT SET')
                if var == 'NEO4J_PASSWORD':
                    value = '***' + value[-4:] if len(value) > 4 else '***'
                print(f"    {var}: {value}")
            
            return True
        else:
            print("  ❌ 'neural-tools' server not configured")
            return False
            
    except Exception as e:
        print(f"  ❌ Error reading config: {e}")
        return False

def suggest_fixes(issues):
    """Suggest fixes for identified issues"""
    if issues:
        print("\n💡 Suggested fixes:")
        
        if 'services' in issues:
            print("  • Start Docker services: docker-compose up -d")
        
        if 'mcp_multiple' in issues:
            print("  • Multiple MCP processes detected. Run: python3 cleanup_stale_mcp.py")
        
        if 'neo4j_auth' in issues:
            print("  • Check Neo4j password in docker-compose.yml matches .mcp.json")
        
        if 'config' in issues:
            print("  • Verify .mcp.json configuration, especially environment variables")

def main():
    print("=" * 60)
    print("MCP Server Health Check & Diagnostics")
    print("=" * 60)
    
    issues = []
    
    # Check services
    if not check_service_ports():
        issues.append('services')
    
    # Check MCP processes
    mcp_count = len([p for p in psutil.process_iter(['cmdline']) 
                     if p.info.get('cmdline') and 'run_mcp_server.py' in ' '.join(p.info['cmdline'])])
    if mcp_count > 2:
        issues.append('mcp_multiple')
    check_mcp_processes()
    
    # Check Neo4j auth
    if not check_neo4j_auth():
        issues.append('neo4j_auth')
    
    # Check config
    if not check_mcp_config():
        issues.append('config')
    
    # Summary
    print("\n" + "=" * 60)
    if not issues:
        print("✅ All checks passed! MCP should be able to connect.")
        print("\nIf new Claude instances still can't connect:")
        print("  1. Check if current MCP processes are zombies: ps aux | grep run_mcp_server")
        print("  2. Try reconnecting in Claude with: /mcp")
        print("  3. Check Claude logs for specific error messages")
    else:
        print(f"❌ Found {len(issues)} issue(s) that may prevent MCP connection")
        suggest_fixes(issues)
    
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main())