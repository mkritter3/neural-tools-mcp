#!/usr/bin/env python3
"""
Fix Qdrant indexing threshold for existing collections.
This resolves the issue where collections with <10,000 points don't return search results.
"""

from qdrant_client import QdrantClient
import sys

def fix_collection_indexing(collection_name: str, threshold: int = 100):
    """Update indexing threshold for an existing collection."""
    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=46333)

        # Check if collection exists
        collections = client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            print(f"❌ Collection '{collection_name}' not found")
            return False

        # Update collection optimizer config
        client.update_collection(
            collection_name=collection_name,
            optimizer_config={
                "indexing_threshold": threshold
            }
        )

        # Get collection info
        info = client.get_collection(collection_name)
        print(f"✅ Fixed {collection_name}:")
        print(f"   - Points count: {info.points_count}")
        print(f"   - Indexed vectors: {info.indexed_vectors_count}")
        print(f"   - New indexing threshold: {threshold}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Fix specific collection or all collections
    if len(sys.argv) > 1:
        collection_name = sys.argv[1]
        fix_collection_indexing(collection_name)
    else:
        # Fix common project collections
        collections_to_fix = [
            "project-north-star-aws-flow",
            "project-northstar-finance",
            "project-neural-novelist",
            "project-claude-l9-template"
        ]

        print("🔧 Fixing Qdrant indexing thresholds...")
        for collection in collections_to_fix:
            fix_collection_indexing(collection)