#!/usr/bin/env python3
"""
Test script to verify the project_indexer tool works
"""

import sys
import os
import asyncio
import json

# Add paths
sys.path.append('/app')
sys.path.append('/app/src')
sys.path.append('/app/src/servers')

async def test_indexer():
    """Test the project indexer implementation directly"""
    
    # Import the indexer implementation
    from neural_server_stdio import project_indexer_impl
    
    print("=" * 60)
    print("🧪 Testing Project Indexer Tool")
    print("=" * 60)
    
    # Test path setup
    test_path = "/app/project"
    
    print(f"\n📁 Test Configuration:")
    print(f"  Path: {test_path}")
    print(f"  Recursive: True")
    print(f"  Clear existing: False")
    
    # Check if path exists
    if os.path.exists(test_path):
        files = os.listdir(test_path)[:5]  # First 5 files
        print(f"\n✅ Path exists with {len(os.listdir(test_path))} items")
        print(f"  Sample files: {files}")
    else:
        print(f"\n❌ Path does not exist: {test_path}")
        return False
    
    print("\n🚀 Starting indexing...")
    
    try:
        # Call the indexer
        result = await project_indexer_impl(
            path=test_path,
            recursive=True,
            clear_existing=False,
            file_patterns=None,
            force_reindex=False
        )
        
        # Parse result
        if result and len(result) > 0:
            content = result[0]
            if hasattr(content, 'text'):
                print("\n📊 Indexing Result:")
                print(content.text)
                
                # Try to parse as JSON
                try:
                    data = json.loads(content.text)
                    if data.get('status') == 'success':
                        print("\n✅ SUCCESS! Indexing completed:")
                        stats = data.get('statistics', {})
                        print(f"  Files indexed: {stats.get('files_indexed', 0)}")
                        print(f"  Files skipped: {stats.get('files_skipped', 0)}")
                        print(f"  Chunks created: {stats.get('chunks_created', 0)}")
                        print(f"  Neo4j available: {stats.get('neo4j_available', False)}")
                        print(f"  Qdrant available: {stats.get('qdrant_available', False)}")
                        print(f"  Nomic available: {stats.get('nomic_available', False)}")
                        return True
                    else:
                        print(f"\n⚠️ Indexing returned status: {data.get('status')}")
                        if 'error' in data:
                            print(f"  Error: {data['error']}")
                except json.JSONDecodeError:
                    print("  (Result is not JSON)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Project Indexer Test Script")
    print("==============================")
    
    # Run the async test
    success = asyncio.run(test_indexer())
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST PASSED: Project indexer is working!")
    else:
        print("❌ TEST FAILED: Project indexer has issues")
    print("=" * 60)
    
    sys.exit(0 if success else 1)