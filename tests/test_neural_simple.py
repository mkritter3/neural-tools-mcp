#!/usr/bin/env python3
"""
Simple test of neural tools server functionality.
Tests basic imports and service initialization.
"""

import subprocess
import json

def test_neural_service_init():
    """Test neural server service initialization."""
    print("=== Testing Neural Server Service Initialization ===")
    
    test_script = '''
import sys
import os
import json
import asyncio
from datetime import datetime

# Add the neural-tools path
sys.path.insert(0, "/app/neural-tools-src/servers")

try:
    print("Importing neural server...")
    import neural_server_2025 as ns
    print(f"✓ Neural server imported successfully")
    print(f"  Project: {ns.PROJECT_NAME}")
    
    # Check if services are available
    print(f"  Qdrant client: {ns.qdrant_client is not None}")
    print(f"  Neo4j client: {ns.neo4j_client is not None}")  
    print(f"  Nomic client: {ns.nomic_client is not None}")
    print(f"  SeaGOAT client: {ns.seagoat_client is not None}")
    
    # Test service initialization
    async def test_init():
        print("\\nInitializing services...")
        await ns.ensure_services_initialized()
        print("✓ Services initialized")
        
        # Check services after initialization
        print(f"  Qdrant client: {ns.qdrant_client is not None}")
        print(f"  Neo4j client: {ns.neo4j_client is not None}")  
        print(f"  Nomic client: {ns.nomic_client is not None}")
        print(f"  SeaGOAT client: {ns.seagoat_client is not None}")
        
        # Test neural system status
        print("\\nTesting neural system status...")
        status_result = await ns.neural_system_status_impl()
        if status_result:
            status_text = status_result[0].text
            status_data = json.loads(status_text)
            print(f"✓ System status: {status_data.get('status', 'unknown')}")
            services = status_data.get('services', {})
            for service, active in services.items():
                status = "✓" if active else "✗"
                print(f"  {status} {service}: {active}")
        
        # Test SeaGOAT directly if client exists
        if ns.seagoat_client:
            print("\\nTesting SeaGOAT directly...")
            try:
                response = await ns.seagoat_client.get("/status")
                print(f"✓ SeaGOAT status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print(f"  Version: {data.get('version', 'unknown')}")
                    stats = data.get('stats', {})
                    if 'chunks' in stats:
                        chunks = stats['chunks']
                        print(f"  Analyzed chunks: {chunks.get('analyzed', 0)}")
                        print(f"  Unanalyzed chunks: {chunks.get('unanalyzed', 0)}")
                        
                # Test search functionality
                print("\\nTesting SeaGOAT search...")
                search_response = await ns.seagoat_client.post("/query", json={
                    "query": "MCP neural tools",
                    "limit_results": 3
                })
                print(f"✓ SeaGOAT search: {search_response.status_code}")
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    results = search_data.get('results', [])
                    print(f"  Found {len(results)} results")
                    for i, result in enumerate(results[:2]):
                        path = result.get('path', 'unknown')
                        score = result.get('score', 0)
                        print(f"    {i+1}. {path} (score: {score:.3f})")
            except Exception as e:
                print(f"✗ SeaGOAT direct test failed: {e}")
        else:
            print("✗ SeaGOAT client not available")
    
    asyncio.run(test_init())
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        result = subprocess.run([
            "docker", "exec", "default-neural", "python3", "-c", test_script
        ], capture_output=True, text=True, timeout=60)
        
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Run the test."""
    print("🧪 Neural Tools Simple Test")
    print("=" * 40)
    
    success = test_neural_service_init()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Test completed successfully!")
    else:
        print("⚠️ Test encountered issues - check output above")

if __name__ == "__main__":
    main()