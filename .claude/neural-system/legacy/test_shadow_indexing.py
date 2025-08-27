#!/usr/bin/env python3
"""
Test Shadow Indexing System for Phase 1 Implementation
Tests multiple embedding models and shadow indexing capabilities
"""

import os
import sys
from pathlib import Path

# Add neural-system to path
sys.path.insert(0, str(Path(__file__).parent))

from neural_embeddings import HybridEmbeddingSystem, get_neural_system

def test_shadow_indexing():
    """Test the shadow indexing system with multiple models"""
    
    print("🧠 Testing L9 Phase 1: Shadow Indexing System")
    print("=" * 60)
    
    # Initialize system
    system = get_neural_system()
    
    # Test code samples
    test_samples = [
        {
            'text': '''def fibonacci(n):
    """Calculate fibonacci number using recursion"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)''',
            'metadata': {'file_path': 'algorithms.py', 'file_type': 'CORE_CODE'}
        },
        {
            'text': '''class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None
    
    def connect(self):
        # Establish database connection
        pass''',
            'metadata': {'file_path': 'database.py', 'file_type': 'CORE_CODE'}
        },
        {
            'text': 'This is a simple text document about project documentation and how to use the API effectively.',
            'metadata': {'file_path': 'README.md', 'file_type': 'DOCS'}
        }
    ]
    
    print("📝 Testing embedding generation with multiple backends...")
    embeddings = []
    
    for i, sample in enumerate(test_samples):
        print(f"\nSample {i+1}:")
        embedding = system.generate_embedding(sample['text'], sample['metadata'])
        embeddings.append(embedding)
        
        print(f"  Backend: {embedding.metadata.get('backend', 'unknown')}")
        print(f"  Model: {embedding.model}")
        print(f"  Dimensions: {embedding.dimensions}")
        print(f"  Is Code: {embedding.metadata.get('is_code', False)}")
    
    # Test shadow indexing storage
    print(f"\n💾 Testing shadow indexing storage...")
    success = system.vector_store.add_embeddings(embeddings)
    print(f"Storage success: {success}")
    
    # Get stats to see collections
    print(f"\n📊 Vector Store Statistics:")
    stats = system.vector_store.get_stats()
    for key, value in stats.items():
        if key == 'collections':
            print(f"  Collections:")
            for coll_key, coll_stats in value.items():
                print(f"    {coll_key}: {coll_stats}")
        else:
            print(f"  {key}: {value}")
    
    # Test search with different preferences
    print(f"\n🔍 Testing search with model preferences...")
    query = "fibonacci recursive algorithm implementation"
    
    # Search with code preference
    results = system.semantic_search(query, n_results=3)
    print(f"\nQuery: '{query}'")
    print(f"Results ({len(results)} found):")
    
    for i, result in enumerate(results, 1):
        model = result.get('model', 'unknown')
        similarity = result.get('similarity_score', 0)
        preview = result['document'][:60].replace('\n', ' ') + "..."
        
        print(f"  {i}. Model: {model} | Similarity: {similarity:.3f}")
        print(f"     Preview: {preview}")
    
    # Test A/B testing mode if enabled
    ab_test_env = os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
    if ab_test_env:
        print(f"\n🧪 A/B Testing Mode Enabled")
        # Additional A/B testing would be here
    else:
        print(f"\n🧪 A/B Testing: Disabled (set ENABLE_AB_TESTING=true to enable)")
    
    return True

if __name__ == "__main__":
    try:
        success = test_shadow_indexing()
        if success:
            print("\n✅ Shadow indexing system test completed successfully!")
        else:
            print("\n❌ Shadow indexing system test failed!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()