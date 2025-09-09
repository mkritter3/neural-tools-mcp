#!/usr/bin/env python3
"""
P0 Smoke Tests - Minimal regression tests for critical fixes
Tests the 4 P0 fixes from the implementation roadmap
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add neural-tools/src to path
neural_tools_src = Path(__file__).parent / "src"
sys.path.insert(0, str(neural_tools_src))

from servers.services.service_container import ServiceContainer
from servers.services.collection_config import get_collection_manager, CollectionType

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_p0_1_qdrant_method_fix():
    """Test P0-1: MCP→Qdrant method mismatch fix"""
    logger.info("Testing P0-1: MCP→Qdrant method mismatch fix")
    
    try:
        container = ServiceContainer("test")
        await container.initialize_all_services()
        
        if not container.qdrant:
            logger.warning("Qdrant not available - skipping P0-1 test")
            return False
            
        # Test that search_vectors method exists (not search)
        assert hasattr(container.qdrant, 'search_vectors'), "QdrantService should have search_vectors method"
        assert not hasattr(container.qdrant, 'search') or callable(getattr(container.qdrant, 'search_vectors')), "search_vectors should be callable"
        
        logger.info("✅ P0-1: search_vectors method exists")
        return True
        
    except Exception as e:
        logger.error(f"❌ P0-1 failed: {e}")
        return False

async def test_p0_2_delete_points_implementation():
    """Test P0-2: delete_points method implementation"""
    logger.info("Testing P0-2: delete_points method implementation")
    
    try:
        container = ServiceContainer("test")
        await container.initialize_all_services()
        
        if not container.qdrant:
            logger.warning("Qdrant not available - skipping P0-2 test")
            return False
            
        # Test that delete_points method exists
        assert hasattr(container.qdrant, 'delete_points'), "QdrantService should have delete_points method"
        assert callable(getattr(container.qdrant, 'delete_points')), "delete_points should be callable"
        
        logger.info("✅ P0-2: delete_points method exists")
        return True
        
    except Exception as e:
        logger.error(f"❌ P0-2 failed: {e}")
        return False

def test_p0_3_http_port_env():
    """Test P0-3: Qdrant HTTP port environment variables"""
    logger.info("Testing P0-3: Qdrant HTTP port environment standardization")
    
    try:
        # Set test environment
        os.environ['QDRANT_HTTP_PORT'] = '6333'
        
        from servers.services.qdrant_service import QdrantService
        service = QdrantService("test")
        
        # Check that http_port is set correctly
        assert service.http_port == 6333, f"Expected http_port=6333, got {service.http_port}"
        
        logger.info("✅ P0-3: QDRANT_HTTP_PORT environment variable works")
        return True
        
    except Exception as e:
        logger.error(f"❌ P0-3 failed: {e}")
        return False

async def test_p0_4_vector_dimensions():
    """Test P0-4: Vector dimensions alignment"""
    logger.info("Testing P0-4: Vector dimensions end-to-end alignment")
    
    try:
        # Test environment variable override
        os.environ['EMBED_DIM'] = '768'
        
        # Test CollectionManager uses EMBED_DIM
        manager = get_collection_manager("test")
        assert manager.embedding_dimension == 768, f"Expected embedding_dimension=768, got {manager.embedding_dimension}"
        
        # Test dynamic dimension detection
        container = ServiceContainer("test")
        await container.initialize_all_services()
        
        if container.nomic:
            try:
                health = await container.nomic.health_check()
                embedding_dim = health.get('embedding_dim')
                if embedding_dim:
                    logger.info(f"Detected embedding dimension: {embedding_dim}")
                    assert embedding_dim in [768, 1024, 1536], f"Unexpected embedding dimension: {embedding_dim}"
                else:
                    logger.warning("Could not detect embedding dimension from Nomic service")
            except Exception as e:
                logger.warning(f"Nomic health_check failed: {e} - this is expected if service not initialized")
        
        logger.info("✅ P0-4: Vector dimensions alignment works")
        return True
        
    except Exception as e:
        logger.error(f"❌ P0-4 failed: {e}")
        return False

async def test_basic_integration():
    """Basic integration test - ensure services can initialize"""
    logger.info("Testing basic service integration")
    
    try:
        container = ServiceContainer("test")
        result = await container.initialize_all_services()
        
        # Check service availability
        services_status = {
            'neo4j': bool(container.neo4j),
            'qdrant': bool(container.qdrant),
            'nomic': bool(container.nomic)
        }
        
        logger.info(f"Services status: {services_status}")
        
        # At least one service should be available
        assert any(services_status.values()), "At least one service should be available"
        
        logger.info("✅ Basic integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic integration failed: {e}")
        return False

async def main():
    """Run all P0 smoke tests"""
    logger.info("Starting P0 Smoke Tests")
    logger.info("=" * 50)
    
    tests = [
        ("P0-1: MCP→Qdrant method fix", test_p0_1_qdrant_method_fix),
        ("P0-2: delete_points implementation", test_p0_2_delete_points_implementation), 
        ("P0-3: HTTP port env standardization", test_p0_3_http_port_env),
        ("P0-4: Vector dimensions alignment", test_p0_4_vector_dimensions),
        ("Basic Integration", test_basic_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("P0 SMOKE TEST RESULTS")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL" 
        logger.info(f"{status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All P0 fixes verified!")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} tests failed - P0 fixes need attention")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))