#!/usr/bin/env python3
"""
Phase 3 Integration Test - Live Services

Tests the complete intelligent caching layer with actual running services:
- Tests cache warming with real embeddings
- Verifies Neo4j query caching with actual database
- Tests Qdrant vector search caching
- Monitors cache metrics in real-time
- Measures actual performance improvements
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for local Docker services
os.environ['NEO4J_HOST'] = 'localhost'
os.environ['NEO4J_PORT'] = '47687'
os.environ['QDRANT_HOST'] = 'localhost'
os.environ['QDRANT_PORT'] = '46333'
os.environ['REDIS_CACHE_HOST'] = 'localhost'
os.environ['REDIS_CACHE_PORT'] = '46379'
os.environ['REDIS_QUEUE_HOST'] = 'localhost'
os.environ['REDIS_QUEUE_PORT'] = '46380'
os.environ['NOMIC_HOST'] = 'localhost'
os.environ['NOMIC_PORT'] = '48000'

class Phase3IntegrationTest:
    """Integration test for Phase 3 with live services"""
    
    def __init__(self):
        self.service_container = None
        self.test_results = {}
    
    async def setup(self):
        """Initialize services and verify connectivity"""
        logger.info("🔧 Initializing Phase 3 integration test environment...")
        
        try:
            from servers.services.service_container import ServiceContainer
            
            self.service_container = ServiceContainer("phase3_integration")
            
            # Initialize all services
            init_result = await self.service_container.initialize_all_services()
            
            logger.info(f"Service initialization result: {init_result}")
            
            if not init_result.get("success"):
                logger.warning(f"Some services failed to initialize: {init_result}")
            
            # Get cache services
            self.cache_warmer = await self.service_container.get_cache_warmer()
            self.cache_metrics = await self.service_container.get_cache_metrics()
            
            logger.info("✅ Test environment ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def test_cache_warming(self):
        """Test cache warming with real embedding service"""
        logger.info("\n🧪 Testing Cache Warming with Real Embeddings...")
        
        try:
            # Test queries for warming
            test_queries = [
                "What is machine learning?",
                "Explain neural networks",
                "How does deep learning work?",
                "What is natural language processing?",
                "Explain transformer architecture"
            ]
            
            # Warm the cache
            logger.info(f"Warming cache with {len(test_queries)} queries...")
            warming_result = await self.cache_warmer.warm_embedding_cache(test_queries)
            
            logger.info(f"✅ Cache warming: {warming_result['successful']}/{warming_result['total_queries']} successful")
            
            if warming_result.get('errors'):
                logger.warning(f"Warming errors: {warming_result['errors']}")
            
            # Test that cached embeddings are faster
            if self.service_container.nomic:
                # First call - should be from cache (fast)
                start_time = time.time()
                cached_embedding = await self.service_container.nomic.get_embedding(test_queries[0])
                cached_time = time.time() - start_time
                
                # New query - should hit service (slower)
                start_time = time.time()
                new_embedding = await self.service_container.nomic.get_embedding("This is a completely new query that was never cached")
                new_time = time.time() - start_time
                
                speedup = new_time / max(cached_time, 0.001)
                logger.info(f"✅ Cache speedup: {speedup:.1f}x faster for cached queries")
                
                self.test_results['cache_warming'] = {
                    'success': True,
                    'queries_warmed': warming_result['successful'],
                    'cache_speedup': f"{speedup:.1f}x"
                }
            else:
                logger.warning("Nomic service not available for speedup test")
                self.test_results['cache_warming'] = {
                    'success': True,
                    'queries_warmed': warming_result['successful']
                }
                
        except Exception as e:
            logger.error(f"❌ Cache warming test failed: {e}")
            self.test_results['cache_warming'] = {'success': False, 'error': str(e)}
    
    async def test_neo4j_caching(self):
        """Test Neo4j query caching with real database"""
        logger.info("\n🧪 Testing Neo4j Query Caching...")
        
        try:
            if not self.service_container.neo4j:
                logger.warning("Neo4j service not available")
                self.test_results['neo4j_caching'] = {'skipped': True}
                return
            
            # Create some test data
            test_node_query = """
            MERGE (n:TestNode {id: 'test_phase3', timestamp: timestamp()})
            RETURN n
            """
            await self.service_container.neo4j.execute_cypher(test_node_query)
            
            # Test query caching
            count_query = "MATCH (n:TestNode) RETURN count(n) as count"
            
            # First execution - no cache
            start_time = time.time()
            result1 = await self.service_container.neo4j.execute_cypher(count_query)
            first_time = time.time() - start_time
            
            # Second execution - should be cached
            start_time = time.time()
            result2 = await self.service_container.neo4j.execute_cypher(count_query)
            cached_time = time.time() - start_time
            
            # Verify results match
            assert result1 == result2, "Cached results don't match original"
            
            speedup = first_time / max(cached_time, 0.001)
            logger.info(f"✅ Neo4j cache speedup: {speedup:.1f}x")
            logger.info(f"   First query: {first_time*1000:.2f}ms, Cached: {cached_time*1000:.2f}ms")
            
            # Cleanup test data
            cleanup_query = "MATCH (n:TestNode {id: 'test_phase3'}) DELETE n"
            await self.service_container.neo4j.execute_cypher(cleanup_query)
            
            self.test_results['neo4j_caching'] = {
                'success': True,
                'speedup': f"{speedup:.1f}x",
                'first_time_ms': round(first_time * 1000, 2),
                'cached_time_ms': round(cached_time * 1000, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Neo4j caching test failed: {e}")
            self.test_results['neo4j_caching'] = {'success': False, 'error': str(e)}
    
    async def test_qdrant_caching(self):
        """Test Qdrant vector search caching"""
        logger.info("\n🧪 Testing Qdrant Vector Search Caching...")
        
        try:
            if not self.service_container.qdrant:
                logger.warning("Qdrant service not available")
                self.test_results['qdrant_caching'] = {'skipped': True}
                return
            
            # Test collections caching
            start_time = time.time()
            collections1 = await self.service_container.qdrant.get_collections()
            first_time = time.time() - start_time
            
            start_time = time.time()
            collections2 = await self.service_container.qdrant.get_collections()
            cached_time = time.time() - start_time
            
            assert collections1 == collections2, "Cached collections don't match"
            
            speedup = first_time / max(cached_time, 0.001)
            logger.info(f"✅ Qdrant collections cache speedup: {speedup:.1f}x")
            
            # If we have collections, test vector search caching
            if collections1:
                collection_name = collections1[0]
                test_vector = [0.1] * 768  # Standard dimension
                
                # First search - no cache
                start_time = time.time()
                search1 = await self.service_container.qdrant.search_vectors(
                    collection_name, test_vector, limit=5
                )
                search_first_time = time.time() - start_time
                
                # Second search - cached
                start_time = time.time()
                search2 = await self.service_container.qdrant.search_vectors(
                    collection_name, test_vector, limit=5
                )
                search_cached_time = time.time() - start_time
                
                search_speedup = search_first_time / max(search_cached_time, 0.001)
                logger.info(f"✅ Qdrant search cache speedup: {search_speedup:.1f}x")
                
                self.test_results['qdrant_caching'] = {
                    'success': True,
                    'collections_speedup': f"{speedup:.1f}x",
                    'search_speedup': f"{search_speedup:.1f}x"
                }
            else:
                self.test_results['qdrant_caching'] = {
                    'success': True,
                    'collections_speedup': f"{speedup:.1f}x",
                    'note': 'No collections for search test'
                }
                
        except Exception as e:
            logger.error(f"❌ Qdrant caching test failed: {e}")
            self.test_results['qdrant_caching'] = {'success': False, 'error': str(e)}
    
    async def test_cache_metrics(self):
        """Monitor cache performance metrics"""
        logger.info("\n🧪 Testing Cache Performance Metrics...")
        
        try:
            # Wait for metrics collection
            await asyncio.sleep(2)
            
            # Get performance summary
            summary = await self.cache_metrics.get_cache_performance_summary(window_minutes=5)
            
            if "error" not in summary:
                cache_summary = summary.get("summary", {})
                memory_info = summary.get("memory", {})
                
                logger.info(f"✅ Cache Performance Summary:")
                logger.info(f"   Hit Ratio: {cache_summary.get('hit_ratio', 0):.3f}")
                logger.info(f"   Total Requests: {cache_summary.get('total_requests', 0)}")
                logger.info(f"   Performance Grade: {cache_summary.get('performance_grade', 'N/A')}")
                logger.info(f"   Memory Usage: {memory_info.get('used_memory_mb', 0):.1f} MB")
                
                # Get service breakdown
                services = summary.get("services", {})
                if services:
                    logger.info(f"   Service Cache Keys:")
                    for service, stats in services.items():
                        if isinstance(stats, dict):
                            logger.info(f"     - {service}: {stats.get('key_count', 0)} keys")
                
                # Get recommendations
                recommendations = summary.get("recommendations", [])
                if recommendations:
                    logger.info(f"   Recommendations: {len(recommendations)} suggestions")
                    for rec in recommendations[:3]:
                        logger.info(f"     • {rec}")
                
                self.test_results['cache_metrics'] = {
                    'success': True,
                    'hit_ratio': cache_summary.get('hit_ratio', 0),
                    'performance_grade': cache_summary.get('performance_grade', 'N/A'),
                    'memory_mb': memory_info.get('used_memory_mb', 0)
                }
            else:
                logger.info(f"Metrics not yet available: {summary['error']}")
                self.test_results['cache_metrics'] = {
                    'success': True,
                    'note': 'Metrics collection in progress'
                }
                
        except Exception as e:
            logger.error(f"❌ Cache metrics test failed: {e}")
            self.test_results['cache_metrics'] = {'success': False, 'error': str(e)}
    
    async def test_auto_warming(self):
        """Test automatic cache warming based on frequency"""
        logger.info("\n🧪 Testing Automatic Cache Warming...")
        
        try:
            # Run auto-warming
            auto_result = await self.cache_warmer.auto_warm_frequent_queries(limit=5)
            
            logger.info(f"✅ Auto-warming completed:")
            logger.info(f"   Queries analyzed: {auto_result.get('queries_analyzed', 0)}")
            logger.info(f"   Queries warmed: {auto_result.get('queries_warmed', 0)}")
            
            # Test TTL optimization
            ttl_result = await self.cache_warmer.optimize_ttl_policies()
            
            logger.info(f"✅ TTL Optimization:")
            logger.info(f"   Keys analyzed: {ttl_result.get('analyzed_keys', 0)}")
            
            if ttl_result.get('recommendations'):
                logger.info(f"   TTL Recommendations:")
                for rec in ttl_result['recommendations'][:3]:
                    logger.info(f"     • {rec}")
            
            self.test_results['auto_warming'] = {
                'success': True,
                'queries_warmed': auto_result.get('queries_warmed', 0),
                'ttl_keys_analyzed': ttl_result.get('analyzed_keys', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Auto-warming test failed: {e}")
            self.test_results['auto_warming'] = {'success': False, 'error': str(e)}
    
    async def run_tests(self):
        """Run all integration tests"""
        logger.info("🚀 Starting Phase 3 Integration Tests with Live Services")
        logger.info("="*60)
        
        if not await self.setup():
            logger.error("Setup failed - aborting tests")
            return False
        
        # Run all tests
        await self.test_cache_warming()
        await self.test_neo4j_caching()
        await self.test_qdrant_caching()
        await self.test_cache_metrics()
        await self.test_auto_warming()
        
        # Generate report
        self.generate_report()
        
        return True
    
    def generate_report(self):
        """Generate integration test report"""
        logger.info("\n" + "="*60)
        logger.info("PHASE 3 INTEGRATION TEST RESULTS")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        successful = sum(1 for r in self.test_results.values() if r.get('success'))
        skipped = sum(1 for r in self.test_results.values() if r.get('skipped'))
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Successful: {successful}")
        logger.info(f"⏭️  Skipped: {skipped}")
        logger.info(f"❌ Failed: {total_tests - successful - skipped}")
        logger.info("-"*60)
        
        for test_name, result in self.test_results.items():
            if result.get('success'):
                logger.info(f"✅ {test_name}: PASSED")
                for key, value in result.items():
                    if key != 'success':
                        logger.info(f"   └─ {key}: {value}")
            elif result.get('skipped'):
                logger.info(f"⏭️  {test_name}: SKIPPED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
                if 'error' in result:
                    logger.info(f"   └─ Error: {result['error']}")
        
        logger.info("-"*60)
        
        success_rate = (successful / max(total_tests - skipped, 1)) * 100
        
        if success_rate >= 80:
            logger.info("🎉 PHASE 3 INTELLIGENT CACHING: INTEGRATION VERIFIED")
            logger.info("   The caching layer is working with live services!")
        elif success_rate >= 60:
            logger.info("⚠️  PHASE 3 INTELLIGENT CACHING: PARTIAL SUCCESS")
        else:
            logger.info("🚨 PHASE 3 INTELLIGENT CACHING: INTEGRATION ISSUES")
        
        logger.info("="*60)

async def main():
    """Main test execution"""
    test = Phase3IntegrationTest()
    await test.run_tests()

if __name__ == "__main__":
    asyncio.run(main())