#!/usr/bin/env python3
"""
P0 Unit Tests - Test P0 fixes without external dependencies
"""

import sys
import logging
from pathlib import Path

# Add neural-tools/src to path
neural_tools_src = Path(__file__).parent / "src"
sys.path.insert(0, str(neural_tools_src))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_p0_1_method_exists():
    """Test P0-1: Verify QdrantService has search_vectors method"""
    logger.info("Testing P0-1: QdrantService method signature")
    
    try:
        from servers.services.qdrant_service import QdrantService
        
        # Check method exists
        assert hasattr(QdrantService, 'search_vectors'), "QdrantService should have search_vectors method"
        
        # Check it's callable
        method = getattr(QdrantService, 'search_vectors')
        assert callable(method), "search_vectors should be callable"
        
        logger.info("✅ P0-1: search_vectors method exists and is callable")
        return True
        
    except Exception as e:
        logger.error(f"❌ P0-1 failed: {e}")
        return False

def test_p0_2_delete_points_exists():
    """Test P0-2: Verify QdrantService has delete_points method"""
    logger.info("Testing P0-2: QdrantService delete_points method")
    
    try:
        from servers.services.qdrant_service import QdrantService
        
        # Check method exists
        assert hasattr(QdrantService, 'delete_points'), "QdrantService should have delete_points method"
        
        # Check it's callable
        method = getattr(QdrantService, 'delete_points')
        assert callable(method), "delete_points should be callable"
        
        logger.info("✅ P0-2: delete_points method exists and is callable")
        return True
        
    except Exception as e:
        logger.error(f"❌ P0-2 failed: {e}")
        return False

def test_p0_3_port_config():
    """Test P0-3: QdrantService HTTP port configuration"""
    logger.info("Testing P0-3: QdrantService HTTP port configuration")
    
    try:
        import os
        
        # Set test environment
        os.environ['QDRANT_HTTP_PORT'] = '6333'
        
        from servers.services.qdrant_service import QdrantService
        service = QdrantService("test")
        
        # Check that http_port is set correctly
        assert service.http_port == 6333, f"Expected http_port=6333, got {service.http_port}"
        
        logger.info("✅ P0-3: HTTP port configuration works")
        return True
        
    except Exception as e:
        logger.error(f"❌ P0-3 failed: {e}")
        return False

def test_p0_4_collection_manager_dimension():
    """Test P0-4: CollectionManager uses EMBED_DIM environment variable"""
    logger.info("Testing P0-4: CollectionManager dimension configuration")
    
    try:
        import os
        
        # Test EMBED_DIM override
        os.environ['EMBED_DIM'] = '1024'
        
        from servers.services.collection_config import CollectionManager
        manager = CollectionManager("test")
        
        assert manager.embedding_dimension == 1024, f"Expected 1024, got {manager.embedding_dimension}"
        
        # Test with explicit parameter
        manager2 = CollectionManager("test", embedding_dimension=512)
        assert manager2.embedding_dimension == 512, f"Expected 512, got {manager2.embedding_dimension}"
        
        logger.info("✅ P0-4: CollectionManager dimension configuration works")
        return True
        
    except Exception as e:
        logger.error(f"❌ P0-4 failed: {e}")
        return False

def main():
    """Run all P0 unit tests"""
    logger.info("Starting P0 Unit Tests")
    logger.info("=" * 40)
    
    tests = [
        ("P0-1: search_vectors method", test_p0_1_method_exists),
        ("P0-2: delete_points method", test_p0_2_delete_points_exists),
        ("P0-3: HTTP port config", test_p0_3_port_config),
        ("P0-4: Dimension config", test_p0_4_collection_manager_dimension)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("P0 UNIT TEST RESULTS")
    logger.info("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All P0 unit tests passed!")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())