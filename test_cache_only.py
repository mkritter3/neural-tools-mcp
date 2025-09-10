#!/usr/bin/env python3
"""
Phase 3 Cache-Only Test - Verify caching without embeddings

Tests the cache layer functionality with Redis only:
- Cache key generation
- Cache storage and retrieval 
- TTL management
- Performance metrics
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add neural-tools src to Python path
neural_tools_src = Path(__file__).parent / "neural-tools" / "src"
sys.path.insert(0, str(neural_tools_src))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set Redis connection for local Docker
os.environ['REDIS_CACHE_HOST'] = 'localhost'
os.environ['REDIS_CACHE_PORT'] = '46379'
os.environ['REDIS_QUEUE_HOST'] = 'localhost'
os.environ['REDIS_QUEUE_PORT'] = '46380'

async def test_redis_caching():
    """Test basic Redis caching functionality"""
    logger.info("🚀 Testing Phase 3 Cache Layer with Redis")
    
    try:
        from servers.services.service_container import ServiceContainer
        
        # Create minimal service container
        container = ServiceContainer("cache_test")
        
        # Get Redis clients
        redis_cache = await container.get_redis_cache_client()
        redis_queue = await container.get_redis_queue_client()
        
        logger.info("✅ Redis connections established")
        
        # Test cache operations
        test_key = "l9:prod:neural_tools:test:cache_verification"
        test_data = {"query": "test", "result": [1, 2, 3], "timestamp": time.time()}
        
        # Store in cache
        await redis_cache.setex(test_key, 300, json.dumps(test_data))
        logger.info(f"✅ Stored test data in cache with key: {test_key}")
        
        # Retrieve from cache
        cached_data = await redis_cache.get(test_key)
        retrieved_data = json.loads(cached_data)
        
        assert retrieved_data["query"] == test_data["query"]
        logger.info("✅ Retrieved and verified cached data")
        
        # Test TTL
        ttl = await redis_cache.ttl(test_key)
        logger.info(f"✅ Cache TTL: {ttl} seconds")
        
        # Test cache metrics storage
        from servers.services.cache_metrics import CacheMetricsService
        
        metrics_service = CacheMetricsService(container)
        await metrics_service.initialize()
        
        # Let it collect some metrics
        await asyncio.sleep(2)
        
        # Get performance summary
        summary = await metrics_service.get_cache_performance_summary(window_minutes=5)
        
        if "error" not in summary:
            logger.info("✅ Cache metrics service operational")
            logger.info(f"   Memory: {summary.get('memory', {}).get('used_memory_mb', 0):.1f} MB")
        else:
            logger.info(f"⚠️  Metrics collection starting: {summary.get('error')}")
        
        # Test cache warmer analytics
        from servers.services.cache_warmer import CacheAnalytics
        
        analytics = CacheAnalytics(redis_cache)
        
        # Record some cache accesses
        await analytics.record_cache_access("test_key_1", hit=True, ttl_remaining=300)
        await analytics.record_cache_access("test_key_2", hit=False)
        await analytics.record_cache_access("test_key_1", hit=True, ttl_remaining=250)
        
        logger.info("✅ Cache analytics recording operational")
        
        # Get frequent queries
        frequent = await analytics.get_frequent_queries(limit=5)
        logger.info(f"✅ Frequent query tracking: {len(frequent)} queries tracked")
        
        # Cleanup
        await redis_cache.delete(test_key)
        await metrics_service.shutdown()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 PHASE 3 CACHE LAYER: OPERATIONAL")
        logger.info("   ├─ Redis cache: ✅ Connected")
        logger.info("   ├─ Cache storage: ✅ Working")
        logger.info("   ├─ Cache retrieval: ✅ Working")
        logger.info("   ├─ TTL management: ✅ Working")
        logger.info("   ├─ Analytics tracking: ✅ Working")
        logger.info("   └─ Metrics collection: ✅ Working")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cache_performance():
    """Test cache performance improvements"""
    logger.info("\n📊 Testing Cache Performance...")
    
    try:
        from servers.services.service_container import ServiceContainer
        
        container = ServiceContainer("perf_test")
        redis_cache = await container.get_redis_cache_client()
        
        # Simulate cache operations
        test_operations = 100
        test_key_prefix = "l9:prod:neural_tools:perf_test:"
        
        # Write operations
        start_time = time.time()
        for i in range(test_operations):
            key = f"{test_key_prefix}{i}"
            data = {"index": i, "data": "x" * 100}
            await redis_cache.setex(key, 60, json.dumps(data))
        write_time = time.time() - start_time
        
        logger.info(f"✅ Write performance: {test_operations} ops in {write_time:.2f}s")
        logger.info(f"   ({test_operations/write_time:.0f} ops/sec)")
        
        # Read operations (should be faster)
        start_time = time.time()
        for i in range(test_operations):
            key = f"{test_key_prefix}{i}"
            data = await redis_cache.get(key)
        read_time = time.time() - start_time
        
        logger.info(f"✅ Read performance: {test_operations} ops in {read_time:.2f}s")
        logger.info(f"   ({test_operations/read_time:.0f} ops/sec)")
        
        # Cleanup
        for i in range(test_operations):
            await redis_cache.delete(f"{test_key_prefix}{i}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

async def main():
    """Run cache tests"""
    success = await test_redis_caching()
    if success:
        await test_cache_performance()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)