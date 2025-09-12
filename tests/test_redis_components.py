#!/usr/bin/env python3
"""
Focused Redis Components Test

Tests core Redis resilience features without external service dependencies
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add neural-tools src to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_redis_components():
    """Test Redis components in isolation"""
    
    logger.info("🚀 Testing Redis Components")
    
    # Set environment
    os.environ['PROJECT_NAME'] = 'test'
    os.environ['REDIS_CACHE_HOST'] = 'localhost'
    os.environ['REDIS_CACHE_PORT'] = '46379'
    os.environ['REDIS_CACHE_PASSWORD'] = 'cache-secret-key'
    os.environ['REDIS_QUEUE_HOST'] = 'localhost'
    os.environ['REDIS_QUEUE_PORT'] = '46380'
    os.environ['REDIS_QUEUE_PASSWORD'] = 'queue-secret-key'
    
    try:
        # Test basic Redis connectivity
        import redis.asyncio as redis
        
        # Test cache Redis
        cache_redis = redis.Redis(
            host='localhost', port=46379, 
            password='cache-secret-key', decode_responses=True
        )
        await cache_redis.ping()
        logger.info("✅ Cache Redis connectivity OK")
        
        # Test queue Redis
        queue_redis = redis.Redis(
            host='localhost', port=46380,
            password='queue-secret-key', decode_responses=True
        )
        await queue_redis.ping()
        logger.info("✅ Queue Redis connectivity OK")
        
        # Test ARQ integration
        from arq import create_pool
        from arq.connections import RedisSettings
        
        redis_settings = RedisSettings(
            host='localhost', port=46380, 
            password='queue-secret-key', database=0
        )
        job_queue = await create_pool(redis_settings)
        logger.info("✅ ARQ job queue pool created")
        
        # Test Redis Streams for DLQ
        stream_name = "test:dlq:stream"
        await queue_redis.xadd(stream_name, {'test': 'data', 'timestamp': '123456'})
        stream_length = await queue_redis.xlen(stream_name)
        logger.info(f"✅ Redis Streams working (length: {stream_length})")
        
        # Cleanup test stream
        await queue_redis.delete(stream_name)
        
        # Test basic caching
        await cache_redis.setex('test:cache:key', 60, 'test_value')
        cached_value = await cache_redis.get('test:cache:key')
        assert cached_value == 'test_value'
        await cache_redis.delete('test:cache:key')
        logger.info("✅ Cache operations working")
        
        logger.info("\n🎉 All Redis components working correctly!")
        logger.info("✓ Separate Redis instances (cache + queue)")
        logger.info("✓ ARQ job queue integration")
        logger.info("✓ Redis Streams for DLQ")
        logger.info("✓ Basic cache operations")
        logger.info("\n📝 Phase 2 Redis Resilience Architecture is READY!")
        
        await job_queue.close()
        await cache_redis.close()
        await queue_redis.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis components test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_redis_components())
    exit(0 if success else 1)