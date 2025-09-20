#!/usr/bin/env python3
"""Test Qdrant collection creation issue"""

import asyncio
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "neural-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "neural-tools" / "src"))

from servers.services.qdrant_service import QdrantService
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

async def test_qdrant():
    print("="*60)
    print("Testing Qdrant Collection Creation")
    print("="*60)

    # Test 1: Direct client connection
    print("\n1. Testing direct Qdrant client...")
    client = QdrantClient(host="localhost", port=46333)

    try:
        collections = client.get_collections()
        print(f"   Existing collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"   Error listing collections: {e}")

    # Test 2: Create collection directly
    print("\n2. Creating collection directly...")
    try:
        client.create_collection(
            collection_name="test_collection",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print("   ✅ Collection created successfully")
    except Exception as e:
        if "already exists" in str(e):
            print("   Collection already exists")
        else:
            print(f"   ❌ Error creating collection: {e}")

    # Test 3: List collections again
    collections = client.get_collections()
    print(f"   Collections after create: {[c.name for c in collections.collections]}")

    # Test 4: Test with QdrantService wrapper
    print("\n3. Testing QdrantService wrapper...")
    service = QdrantService()
    await service.connect(host="localhost", port=46333)

    result = await service.ensure_collection(
        collection_name="project_claude-l9-template_code",
        vector_size=768
    )
    print(f"   Ensure collection result: {result}")

    # Test 5: Check if sync_manager collection exists
    print("\n4. Checking sync_manager collection...")
    collections = client.get_collections()
    sync_collections = [c.name for c in collections.collections if "claude-l9" in c.name]
    if sync_collections:
        print(f"   ✅ Found sync collections: {sync_collections}")
    else:
        print("   ❌ No claude-l9-template collections found!")

    # Test 6: Try to upsert a test point
    print("\n5. Testing point upsert...")
    from qdrant_client.models import PointStruct

    try:
        # First ensure collection exists
        if "test_collection" in [c.name for c in collections.collections]:
            point = PointStruct(
                id=1,
                vector=[0.1] * 768,
                payload={"test": "data"}
            )
            client.upsert(
                collection_name="test_collection",
                points=[point]
            )
            print("   ✅ Point upserted successfully")

            # Check point count
            info = client.get_collection("test_collection")
            print(f"   Collection point count: {info.points_count}")
    except Exception as e:
        print(f"   ❌ Error upserting point: {e}")

    print("\n" + "="*60)
    print("Diagnosis:")
    if not sync_collections:
        print("❌ CRITICAL: The indexer is NOT creating Qdrant collections!")
        print("   This explains why vectors aren't being stored.")
        print("   The sync_manager expects collections to exist but they don't.")
    else:
        print("✅ Collections exist - issue might be in the write process")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_qdrant())