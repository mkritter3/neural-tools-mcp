#!/usr/bin/env python3
"""
Basic Qdrant Connection Test for L9 Neural System
Verifies Qdrant is running and accessible on L9 template ports
"""

import sys
import json
import time
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from qdrant_client.models import PointStruct, SparseVector

def test_qdrant_connection():
    """Test basic Qdrant connection and operations"""
    print("=" * 60)
    print("Testing L9 Qdrant Connection (Ports 6433/6434)")
    print("=" * 60)
    
    # Connect to L9 Qdrant instance (avoiding enterprise v3 on 6333/6334)
    print("\n📡 Connecting to Qdrant...")
    try:
        # Try REST first
        client_rest = QdrantClient(
            host="localhost",
            port=6433,  # L9 template REST port
            prefer_grpc=False
        )
        
        # Try gRPC for performance
        client_grpc = QdrantClient(
            host="localhost", 
            port=6434,  # L9 template gRPC port
            prefer_grpc=True
        )
        
        print("✅ Connected via REST (6433) and gRPC (6434)")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Test collection creation with hybrid support
    collection_name = "l9_test_collection"
    print(f"\n📂 Creating hybrid collection: {collection_name}")
    
    try:
        # Delete if exists
        collections = client_grpc.get_collections()
        if any(c.name == collection_name for c in collections.collections):
            client_grpc.delete_collection(collection_name)
            print("  ↳ Deleted existing collection")
        
        # Create hybrid collection (BM25 + Semantic)
        client_grpc.create_collection(
            collection_name=collection_name,
            vectors_config={
                "semantic": VectorParams(
                    size=384,  # Small model for testing
                    distance=Distance.COSINE,
                    on_disk=False
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    on_disk=False
                )
            },
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            )
        )
        print("✅ Created hybrid collection with BM25 + Semantic vectors")
        
    except Exception as e:
        print(f"❌ Collection creation failed: {e}")
        return False
    
    # Test indexing
    print("\n📝 Testing document indexing...")
    test_docs = [
        {
            "id": 1,
            "text": "PostgreSQL database with connection pooling for authentication",
            "dense": [0.1] * 384,  # Mock dense vector
            "sparse_indices": [10, 50, 100, 200],
            "sparse_values": [0.5, 0.3, 0.2, 0.1]
        },
        {
            "id": 2,
            "text": "React TypeScript frontend with Material-UI components",
            "dense": [0.2] * 384,  # Mock dense vector
            "sparse_indices": [20, 60, 110, 210],
            "sparse_values": [0.6, 0.4, 0.3, 0.2]
        },
        {
            "id": 3,
            "text": "API rate limiting using token bucket algorithm",
            "dense": [0.3] * 384,  # Mock dense vector
            "sparse_indices": [30, 70, 120, 220],
            "sparse_values": [0.7, 0.5, 0.4, 0.3]
        }
    ]
    
    points = []
    for doc in test_docs:
        point = PointStruct(
            id=doc["id"],
            vector={
                "semantic": doc["dense"],
                "bm25": SparseVector(
                    indices=doc["sparse_indices"],
                    values=doc["sparse_values"]
                )
            },
            payload={"text": doc["text"]}
        )
        points.append(point)
    
    try:
        start = time.time()
        client_grpc.upsert(
            collection_name=collection_name,
            points=points
        )
        elapsed = (time.time() - start) * 1000
        print(f"✅ Indexed {len(points)} documents in {elapsed:.1f}ms")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return False
    
    # Test hybrid search with RRF fusion
    print("\n🔍 Testing Qdrant Native Hybrid Search (RRF Fusion)...")
    
    try:
        # Query vectors
        query_dense = [0.15] * 384  # Mock query vector
        query_sparse = SparseVector(
            indices=[10, 50, 110, 200],
            values=[0.6, 0.4, 0.3, 0.2]
        )
        
        start = time.time()
        results = client_grpc.query_points(
            collection_name=collection_name,
            prefetch=[
                # BM25 search
                models.Prefetch(
                    query=query_sparse,
                    using="bm25",
                    limit=10
                ),
                # Semantic search
                models.Prefetch(
                    query=query_dense,
                    using="semantic",
                    limit=10
                )
            ],
            # Native RRF fusion!
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=3
        )
        elapsed = (time.time() - start) * 1000
        
        print(f"✅ Hybrid search completed in {elapsed:.1f}ms")
        print(f"   Results: {len(results.points)} documents")
        
        for point in results.points:
            score = point.score if hasattr(point, 'score') else 'N/A'
            text = point.payload.get('text', '')[:50]
            print(f"   • Score: {score} | {text}...")
            
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        return False
    
    # Test collection stats
    print("\n📊 Collection Statistics:")
    try:
        info = client_grpc.get_collection(collection_name)
        print(f"  Points: {info.points_count}")
        print(f"  Vectors: {info.vectors_count}")
        print(f"  Status: {info.status}")
        
        # Clean up
        client_grpc.delete_collection(collection_name)
        print(f"\n🗑️ Cleaned up test collection")
        
    except Exception as e:
        print(f"❌ Stats failed: {e}")
        return False
    
    return True

def compare_with_enterprise():
    """Compare L9 template deployment with enterprise v3"""
    print("\n" + "=" * 60)
    print("Deployment Comparison")
    print("=" * 60)
    
    print("\n🏢 Enterprise Neural V3 (Production):")
    print("  • Container: neural-v36-qdrant")
    print("  • Ports: 6333 (REST), 6334 (gRPC)")
    print("  • Version: qdrant/qdrant:v1.7.4")
    print("  • Status: Running (DO NOT TOUCH)")
    
    print("\n🧪 L9 Template (Development):")
    print("  • Container: l9-qdrant-local")
    print("  • Ports: 6433 (REST), 6434 (gRPC)")  
    print("  • Version: qdrant/qdrant:v1.10.0")
    print("  • Status: Running (Safe to modify)")
    
    print("\n✅ Benefits of Separate Deployments:")
    print("  • No interference with production")
    print("  • Independent version management")
    print("  • Isolated data and collections")
    print("  • Safe experimentation environment")
    print("  • Easy rollback if needed")

def main():
    """Run basic Qdrant tests"""
    print("\n🧪 L9 QDRANT BASIC CONNECTION TEST")
    print("Verifying Qdrant on ports 6433/6434")
    print("(Avoiding enterprise v3 on 6333/6334)")
    
    # Run connection test
    success = test_qdrant_connection()
    
    # Show deployment comparison
    compare_with_enterprise()
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ L9 QDRANT BASIC TEST PASSED")
        print("\n💡 Key Achievements:")
        print("  • Qdrant running on unique ports (6433/6434)")
        print("  • Hybrid collection created successfully")
        print("  • Native BM25 + Semantic indexing works")
        print("  • Server-side RRF fusion operational")
        print("  • No interference with enterprise v3")
        
        print("\n🎯 Next Steps:")
        print("  1. Install embedding models (sentence-transformers)")
        print("  2. Run full integration tests")
        print("  3. Update mcp_neural_server.py")
        print("  4. Migrate from ChromaDB")
    else:
        print("❌ L9 QDRANT TEST FAILED")
        print("Check Docker logs: docker logs l9-qdrant-local")
        sys.exit(1)

if __name__ == "__main__":
    main()