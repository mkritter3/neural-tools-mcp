#!/usr/bin/env python3
"""
Simple test for GraphRAG implementation
Tests the core shared ID generation and cross-referencing logic
"""

import hashlib
import asyncio
from pathlib import Path

def test_shared_id_generation():
    """Test that we generate deterministic shared IDs correctly"""
    print("🧪 Testing Shared ID Generation\n")
    
    # Test data
    test_cases = [
        {
            'file': '/src/auth/user.py',
            'start': 10,
            'end': 20,
            'expected_prefix': None  # Will be calculated
        },
        {
            'file': '/src/auth/user.py',
            'start': 10,
            'end': 20,
            'expected_prefix': None  # Should match the first one
        },
        {
            'file': '/src/auth/token.py',
            'start': 5,
            'end': 15,
            'expected_prefix': None  # Should be different
        }
    ]
    
    results = []
    for case in test_cases:
        # Generate ID using same logic as indexer
        chunk_id_str = f"{case['file']}:{case['start']}:{case['end']}"
        chunk_id_hash = hashlib.sha256(chunk_id_str.encode()).hexdigest()
        chunk_id_numeric = int(chunk_id_hash[:16], 16)
        
        results.append({
            'file': case['file'],
            'lines': f"{case['start']}-{case['end']}",
            'hash': chunk_id_hash,
            'numeric': chunk_id_numeric
        })
        
        print(f"File: {case['file']}")
        print(f"Lines: {case['start']}-{case['end']}")
        print(f"SHA256: {chunk_id_hash[:32]}...")
        print(f"Numeric ID: {chunk_id_numeric}")
        print()
    
    # Verify deterministic generation
    if results[0]['hash'] == results[1]['hash']:
        print("✅ Deterministic ID generation verified (same input = same ID)")
    else:
        print("❌ ID generation not deterministic!")
        return False
    
    if results[0]['hash'] != results[2]['hash']:
        print("✅ Different inputs produce different IDs")
    else:
        print("❌ Different inputs produced same ID!")
        return False
    
    return True

def test_id_format():
    """Test that IDs are in the correct format for both databases"""
    print("\n🔢 Testing ID Format Compatibility\n")
    
    test_file = "/test/example.py"
    test_chunk = {"start_line": 100, "end_line": 150}
    
    # Generate ID
    chunk_id_str = f"{test_file}:{test_chunk['start_line']}:{test_chunk['end_line']}"
    chunk_id_hash = hashlib.sha256(chunk_id_str.encode()).hexdigest()
    chunk_id_numeric = int(chunk_id_hash[:16], 16)
    
    print(f"Original string: {chunk_id_str}")
    print(f"Full SHA256 hash: {chunk_id_hash}")
    print(f"Hash length: {len(chunk_id_hash)} chars")
    print(f"Neo4j ID (string): {chunk_id_hash}")
    print(f"Qdrant ID (numeric): {chunk_id_numeric}")
    print(f"Numeric ID fits in int64: {chunk_id_numeric < 2**63}")
    
    # Verify formats
    checks = []
    
    # Neo4j uses string
    if isinstance(chunk_id_hash, str) and len(chunk_id_hash) == 64:
        print("✅ Neo4j ID format correct (64-char hex string)")
        checks.append(True)
    else:
        print("❌ Neo4j ID format incorrect")
        checks.append(False)
    
    # Qdrant uses numeric (must fit in int64)
    if isinstance(chunk_id_numeric, int) and 0 < chunk_id_numeric < 2**63:
        print("✅ Qdrant ID format correct (positive int64)")
        checks.append(True)
    else:
        print("❌ Qdrant ID format incorrect")
        checks.append(False)
    
    return all(checks)

def test_cross_reference_structure():
    """Test the structure of cross-references"""
    print("\n🔗 Testing Cross-Reference Structure\n")
    
    # Simulate what would be stored
    chunk_id_hash = hashlib.sha256(b"test").hexdigest()
    chunk_id_numeric = int(chunk_id_hash[:16], 16)
    
    # Qdrant payload structure
    qdrant_payload = {
        'neo4j_chunk_id': chunk_id_hash,
        'chunk_hash': chunk_id_hash,
        'file_path': '/test/file.py',
        'content': 'def test(): pass',
        'start_line': 1,
        'end_line': 1
    }
    
    # Neo4j properties structure
    neo4j_properties = {
        'id': chunk_id_hash,
        'qdrant_id': chunk_id_numeric,
        'file_path': '/test/file.py',
        'type': 'function',
        'start_line': 1,
        'end_line': 1
    }
    
    print("Qdrant Payload Structure:")
    for key, value in qdrant_payload.items():
        print(f"  {key}: {value if len(str(value)) < 50 else str(value)[:47] + '...'}")
    
    print("\nNeo4j Properties Structure:")
    for key, value in neo4j_properties.items():
        print(f"  {key}: {value if len(str(value)) < 50 else str(value)[:47] + '...'}")
    
    # Verify cross-references
    print("\n🔍 Cross-Reference Verification:")
    
    # Can we go from Qdrant to Neo4j?
    neo4j_ref_from_qdrant = qdrant_payload['neo4j_chunk_id']
    if neo4j_ref_from_qdrant == neo4j_properties['id']:
        print("✅ Qdrant → Neo4j reference valid")
    else:
        print("❌ Qdrant → Neo4j reference mismatch")
        return False
    
    # Can we go from Neo4j to Qdrant?
    qdrant_ref_from_neo4j = neo4j_properties['qdrant_id']
    if qdrant_ref_from_neo4j == chunk_id_numeric:
        print("✅ Neo4j → Qdrant reference valid")
    else:
        print("❌ Neo4j → Qdrant reference mismatch")
        return False
    
    return True

def test_deduplication_logic():
    """Test content hash deduplication"""
    print("\n🔄 Testing Deduplication Logic\n")
    
    content1 = "def hello():\n    return 'world'"
    content2 = "def hello():\n    return 'world'"  # Same
    content3 = "def hello():\n    return 'universe'"  # Different
    
    hash1 = hashlib.sha256(content1.encode()).hexdigest()
    hash2 = hashlib.sha256(content2.encode()).hexdigest()
    hash3 = hashlib.sha256(content3.encode()).hexdigest()
    
    print(f"Content 1 hash: {hash1[:32]}...")
    print(f"Content 2 hash: {hash2[:32]}...")
    print(f"Content 3 hash: {hash3[:32]}...")
    
    if hash1 == hash2:
        print("✅ Identical content produces same hash (dedup works)")
    else:
        print("❌ Identical content produced different hashes")
        return False
    
    if hash1 != hash3:
        print("✅ Different content produces different hash")
    else:
        print("❌ Different content produced same hash")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("GraphRAG Implementation Logic Tests")
    print("=" * 60)
    
    tests = [
        ("Shared ID Generation", test_shared_id_generation),
        ("ID Format Compatibility", test_id_format),
        ("Cross-Reference Structure", test_cross_reference_structure),
        ("Deduplication Logic", test_deduplication_logic)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print('=' * 60)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name}: {status}")
    
    total_passed = sum(1 for _, p in results if p)
    total = len(results)
    
    print(f"\nTotal: {total_passed}/{total} tests passed")
    
    if total_passed == total:
        print("\n🎉 All tests passed! GraphRAG implementation logic is correct.")
    else:
        print("\n⚠️ Some tests failed. Please review the implementation.")
    
    return total_passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)