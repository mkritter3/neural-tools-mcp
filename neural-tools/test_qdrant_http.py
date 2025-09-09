#!/usr/bin/env python3
"""
Test the neural system with HTTP-only Qdrant for local development
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src/servers"))

# Import services
from services.qdrant_service import QdrantService
from services.nomic_service import NomicService
from services.neo4j_service import Neo4jService

class HTTPQdrantService(QdrantService):
    """Modified Qdrant service that uses HTTP instead of gRPC"""
    
    async def initialize(self):
        """Initialize Qdrant client with HTTP-only mode"""
        try:
            # Import here to avoid early import issues
            import httpx
            from qdrant_client import QdrantClient
            
            # Test basic connectivity using HTTP first
            async with httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(),
                timeout=10.0
            ) as client:
                response = await client.get(f"http://{self.host}:{self.http_port}/collections")
                
                if response.status_code != 200:
                    return {
                        "success": False, 
                        "message": f"Qdrant HTTP connectivity failed: {response.status_code}"
                    }
            
            # Initialize HTTP client (no gRPC)
            self.client = QdrantClient(
                host=self.host,
                port=self.http_port,  # Use HTTP port
                prefer_grpc=False,    # Force HTTP
                check_compatibility=False  # Skip version check
            )
            
            # Test client operations
            collections = self.client.get_collections()
            self.initialized = True
            
            return {
                "success": True,
                "message": "Qdrant service initialized successfully (HTTP mode)",
                "collections_count": len(collections.collections),
                "http_endpoint": f"{self.host}:{self.http_port}",
                "collections": [col.name for col in collections.collections]
            }
            
        except Exception as e:
            print(f"Qdrant initialization error: {e}")
            return {
                "success": False,
                "message": f"Qdrant initialization failed: {str(e)}"
            }

async def test_services():
    """Test neural services with HTTP Qdrant"""
    print("🧪 Testing Neural Services with HTTP Qdrant")
    
    # Set environment
    os.environ.update({
        'QDRANT_HOST': 'localhost',
        'QDRANT_HTTP_PORT': '6681',
        'NEO4J_HOST': 'localhost', 
        'NEO4J_HTTP_PORT': '7475',
        'NEO4J_BOLT_PORT': '7688',
        'PROJECT_NAME': 'default'
    })
    
    # Test Qdrant
    print("\n📊 Testing Qdrant service...")
    qdrant = HTTPQdrantService('default')
    qdrant_result = await qdrant.initialize()
    print(f"Qdrant: {'✅' if qdrant_result['success'] else '❌'} {qdrant_result['message']}")
    if qdrant_result['success']:
        print(f"  Collections: {qdrant_result['collections']}")
    
    # Test Nomic (will likely fail without API key)
    print("\n🤖 Testing Nomic service...")
    nomic = NomicService()
    nomic_result = await nomic.initialize()
    print(f"Nomic: {'✅' if nomic_result['success'] else '❌'} {nomic_result['message']}")
    
    # Test Neo4j
    print("\n🕸️ Testing Neo4j service...")
    neo4j = Neo4jService('default')
    neo4j_result = await neo4j.initialize()
    print(f"Neo4j: {'✅' if neo4j_result['success'] else '❌'} {neo4j_result['message']}")
    
    # If Qdrant works, test some basic operations
    if qdrant_result['success'] and qdrant.initialized:
        print("\n🔍 Testing Qdrant operations...")
        
        # Check collection details
        try:
            code_collection = "project_default_code"
            info = await qdrant.get_collection_info(code_collection)
            print(f"  Collection '{code_collection}': {info.get('vectors_count', 0)} vectors")
            
            # Try a sample search if vectors exist
            if info.get('vectors_count', 0) > 0:
                print("  🔎 Testing sample search...")
                # Create a dummy vector for search test
                import numpy as np
                dummy_vector = np.random.random(1536).tolist()  # Nomic v2 dimensions
                
                try:
                    results = await qdrant.search_vectors(
                        collection_name=code_collection,
                        query_vector=dummy_vector,
                        limit=3,
                        score_threshold=0.1
                    )
                    print(f"    Found {len(results)} results")
                    for i, result in enumerate(results[:2], 1):
                        file_path = result.get('file_path', 'unknown')
                        score = result.get('score', 0)
                        print(f"    {i}. {file_path} (score: {score:.3f})")
                except Exception as e:
                    print(f"    Search failed: {e}")
        except Exception as e:
            print(f"  Collection info failed: {e}")
    
    print(f"\n🎯 Summary:")
    print(f"  Qdrant: {'✅' if qdrant_result['success'] else '❌'}")
    print(f"  Nomic:  {'✅' if nomic_result['success'] else '❌'}")  
    print(f"  Neo4j:  {'✅' if neo4j_result['success'] else '❌'}")
    
    healthy_services = sum([
        qdrant_result['success'], 
        nomic_result['success'], 
        neo4j_result['success']
    ])
    print(f"  Overall: {healthy_services}/3 services operational")

if __name__ == "__main__":
    asyncio.run(test_services())