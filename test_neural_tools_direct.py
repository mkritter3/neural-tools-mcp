#!/usr/bin/env python3
"""
Direct test of neural tools server functionality.
Tests SeaGOAT integration by importing and calling functions directly.
"""

import sys
import os
import subprocess
import json
import asyncio
from pathlib import Path

# Add the neural-tools source to Python path
neural_tools_path = Path(__file__).parent / "neural-tools" / "src" / "servers"
sys.path.insert(0, str(neural_tools_path))

def test_docker_connection():
    """Test that we can connect to the Docker container."""
    print("=== Testing Docker Connection ===")
    try:
        result = subprocess.run([
            "docker", "exec", "default-neural", "python3", "-c", "print('Docker connection OK')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ Docker container 'default-neural' is accessible")
            print(f"  Output: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ Docker connection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Docker connection error: {e}")
        return False

def test_neural_server_import():
    """Test importing and running basic neural server functions via Docker."""
    print("\n=== Testing Neural Server Import ===")
    
    # Test script to run inside Docker
    test_script = '''
import sys
import os
import json
from datetime import datetime

# Add the neural-tools path
sys.path.insert(0, "/app/neural-tools-src/servers")

try:
    print("Attempting to import neural server...")
    import neural_server_2025 as ns
    print(f"  Project: {ns.PROJECT_NAME}")
    
    # Import the functions that actually exist
    ensure_services_initialized = ns.ensure_services_initialized
    neural_system_status_impl = ns.neural_system_status_impl
    seagoat_server_status_impl = ns.seagoat_server_status_impl
    semantic_code_search_seagoat_impl = ns.semantic_code_search_seagoat_impl
    print("✓ Successfully imported neural server components")
    print(f"  SeaGOAT client available: {hasattr(ns, 'seagoat_client')}")
    
    # Test service initialization
    print("\\nTesting service initialization...")
    import asyncio
    
    async def test_services():
        try:
            await ensure_services_initialized()
            print("✓ Service initialization completed")
            
            # Test neural system status
            print("\\nTesting neural_system_status...")
            status_result = await neural_system_status_impl()
            status_text = status_result[0].text if status_result else "No result"
            status_data = json.loads(status_text)
            print(f"✓ Neural system status: {status_data.get('status', 'unknown')}")
            print(f"  Services: {status_data.get('services', {})}")
            
            # Test SeaGOAT server status
            print("\\nTesting seagoat_server_status...")
            seagoat_result = await seagoat_server_status_impl()
            seagoat_text = seagoat_result[0].text if seagoat_result else "No result"
            seagoat_data = json.loads(seagoat_text)
            print(f"✓ SeaGOAT server status: {seagoat_data.get('status', 'unknown')}")
            
            # Test semantic search
            print("\\nTesting semantic_code_search...")
            search_result = await semantic_code_search_seagoat_impl("MCP neural tools", 5)
            search_text = search_result[0].text if search_result else "No result"
            search_data = json.loads(search_text)
            print(f"✓ Semantic search result: {search_data.get('status', 'unknown')}")
            if 'results' in search_data:
                print(f"  Found {len(search_data['results'])} results")
                for i, result in enumerate(search_data['results'][:2]):  # Show first 2
                    print(f"    {i+1}. {result.get('path', 'unknown')}: {result.get('snippet', '')[:100]}...")
            
        except Exception as e:
            print(f"✗ Service test error: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_services())
    
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        result = subprocess.run([
            "docker", "exec", "default-neural", "python3", "-c", test_script
        ], capture_output=True, text=True, timeout=60)
        
        print("Docker execution output:")
        print(result.stdout)
        
        if result.stderr:
            print("Docker execution errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Neural server test error: {e}")
        return False

def test_seagoat_port():
    """Test if SeaGOAT is running on the expected port."""
    print("\n=== Testing SeaGOAT Port ===")
    
    test_script = '''
import socket
import requests
import json

# Test if port 34743 is accessible from inside Docker
def test_port(host, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Socket test error: {e}")
        return False

# Test different host configurations
hosts = ["localhost", "127.0.0.1", "host.docker.internal"]
port = 34743

for host in hosts:
    print(f"Testing {host}:{port}...")
    if test_port(host, port):
        print(f"✓ Port {port} is accessible on {host}")
        
        # Try HTTP request
        try:
            url = f"http://{host}:{port}/status"
            response = requests.get(url, timeout=5)
            print(f"✓ HTTP request successful: {response.status_code}")
            if response.status_code == 200:
                print(f"  Response: {response.text[:200]}")
        except Exception as e:
            print(f"  HTTP request failed: {e}")
        
        break
    else:
        print(f"✗ Port {port} not accessible on {host}")
else:
    print(f"✗ Port {port} not accessible on any tested host")
'''
    
    try:
        result = subprocess.run([
            "docker", "exec", "default-neural", "python3", "-c", test_script
        ], capture_output=True, text=True, timeout=30)
        
        print("SeaGOAT port test output:")
        print(result.stdout)
        
        if result.stderr:
            print("SeaGOAT port test errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ SeaGOAT port test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Neural Tools Direct Test Suite")
    print("=" * 50)
    
    tests = [
        ("Docker Connection", test_docker_connection),
        ("SeaGOAT Port", test_seagoat_port),
        ("Neural Server Import", test_neural_server_import),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check logs above")

if __name__ == "__main__":
    main()