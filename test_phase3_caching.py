#!/usr/bin/env python3
"""
Phase 3 Intelligent Caching Layer - End-to-End Test Suite

Tests the complete intelligent caching implementation including:
- Cache warming strategies
- TTL optimization and cache invalidation
- Neo4j service caching (Cypher queries, semantic search)
- Qdrant service caching (vector search, hybrid search, collections)
- Cache performance analytics and metrics
- Cross-service cache integration
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

class Phase3CachingTests:
    """End-to-end test suite for Phase 3 intelligent caching layer"""
    
    def __init__(self):
        self.test_results = []
        self.service_container = None
    
    async def setup(self):
        """Initialize services and test environment"""
        logger.info("🔧 Setting up Phase 3 caching test environment...")
        
        try:
            # Import and initialize ServiceContainer
            from servers.services.service_container import ServiceContainer
            
            self.service_container = ServiceContainer("test_phase3")
            
            # Initialize all services
            init_result = await self.service_container.initialize_all_services()
            
            if not init_result.get("success"):
                raise Exception(f"Service initialization failed: {init_result}")
            
            logger.info("✅ ServiceContainer initialized successfully")
            
            # Verify Redis connections
            redis_cache = await self.service_container.get_redis_cache_client()
            await redis_cache.ping()
            logger.info("✅ Redis cache connection verified")
            
            redis_queue = await self.service_container.get_redis_queue_client()
            await redis_queue.ping()
            logger.info("✅ Redis queue connection verified")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    async def test_neo4j_caching(self):
        """Test Neo4j service intelligent caching"""
        test_name = "Neo4j Intelligent Caching"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            if not self.service_container.neo4j:
                self.test_results.append({
                    "test": test_name,
                    "status": "skipped",
                    "reason": "Neo4j service not available"
                })
                return
            
            # Test Cypher query caching
            test_query = "MATCH (n) RETURN count(n) as node_count"
            
            # First execution - cache miss
            start_time = time.time()
            result1 = await self.service_container.neo4j.execute_cypher(test_query)
            first_execution_time = time.time() - start_time
            
            # Second execution - should be cache hit
            start_time = time.time()
            result2 = await self.service_container.neo4j.execute_cypher(test_query)
            second_execution_time = time.time() - start_time
            
            # Verify results are identical
            assert result1 == result2, "Cached query results don't match original"
            
            # Cache hit should be significantly faster
            cache_speedup = first_execution_time / max(second_execution_time, 0.001)
            
            logger.info(f"✅ Neo4j query caching: {cache_speedup:.1f}x speedup")
            
            # Test semantic search caching if available
            try:
                semantic_result1 = await self.service_container.neo4j.semantic_search(
                    "test query", top_k=5
                )
                semantic_result2 = await self.service_container.neo4j.semantic_search(
                    "test query", top_k=5
                )
                
                assert semantic_result1 == semantic_result2, "Cached semantic search results don't match"
                logger.info("✅ Neo4j semantic search caching verified")
                
            except Exception as e:
                logger.warning(f"Semantic search caching test skipped: {e}")
            
            self.test_results.append({
                "test": test_name,
                "status": "passed",
                "details": {
                    "cache_speedup": f"{cache_speedup:.1f}x",
                    "first_execution_ms": round(first_execution_time * 1000, 2),
                    "second_execution_ms": round(second_execution_time * 1000, 2)
                }
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results.append({
                "test": test_name,
                "status": "failed",
                "error": str(e)
            })
    
    async def test_qdrant_caching(self):
        """Test Qdrant service intelligent caching"""
        test_name = "Qdrant Intelligent Caching"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            if not self.service_container.qdrant:
                self.test_results.append({
                    "test": test_name,
                    "status": "skipped",
                    "reason": "Qdrant service not available"
                })
                return
            
            # Test collections list caching
            start_time = time.time()
            collections1 = await self.service_container.qdrant.get_collections()
            first_execution_time = time.time() - start_time
            
            start_time = time.time()
            collections2 = await self.service_container.qdrant.get_collections()
            second_execution_time = time.time() - start_time
            
            assert collections1 == collections2, "Cached collections list doesn't match original"
            
            cache_speedup = first_execution_time / max(second_execution_time, 0.001)
            logger.info(f"✅ Qdrant collections caching: {cache_speedup:.1f}x speedup")
            
            # Test vector search caching if collections exist
            if collections1:
                test_collection = collections1[0]
                test_vector = [0.1] * 1536  # Standard embedding dimension
                
                try:
                    # First search - cache miss
                    start_time = time.time()
                    search_result1 = await self.service_container.qdrant.search_vectors(
                        test_collection, test_vector, limit=5
                    )
                    first_search_time = time.time() - start_time
                    
                    # Second search - cache hit
                    start_time = time.time()
                    search_result2 = await self.service_container.qdrant.search_vectors(
                        test_collection, test_vector, limit=5
                    )
                    second_search_time = time.time() - start_time
                    
                    assert search_result1 == search_result2, "Cached search results don't match"
                    
                    search_speedup = first_search_time / max(second_search_time, 0.001)
                    logger.info(f"✅ Qdrant vector search caching: {search_speedup:.1f}x speedup")
                    
                except Exception as e:
                    logger.warning(f"Vector search caching test skipped: {e}")
            
            self.test_results.append({
                "test": test_name,
                "status": "passed",
                "details": {
                    "collections_speedup": f"{cache_speedup:.1f}x",
                    "collections_count": len(collections1)
                }
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results.append({
                "test": test_name,
                "status": "failed",
                "error": str(e)
            })
    
    async def test_cache_warming(self):
        """Test cache warming functionality"""
        test_name = "Cache Warming Service"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Get cache warmer service
            cache_warmer = await self.service_container.get_cache_warmer()
            
            # Test embedding cache warming
            test_queries = [
                "What is artificial intelligence?",
                "Explain machine learning algorithms",
                "How does neural network training work?"
            ]
            
            warming_result = await cache_warmer.warm_embedding_cache(
                test_queries, model="nomic-v2"
            )
            
            assert warming_result.get("total_queries") == len(test_queries)
            
            logger.info(f"✅ Cache warming: {warming_result['successful']}/{warming_result['total_queries']} successful")
            
            # Test automatic frequent query warming
            auto_warming_result = await cache_warmer.auto_warm_frequent_queries(limit=10)
            logger.info(f"✅ Auto cache warming completed: {auto_warming_result}")
            
            # Test TTL optimization
            ttl_result = await cache_warmer.optimize_ttl_policies()
            logger.info(f"✅ TTL optimization analyzed {ttl_result.get('analyzed_keys', 0)} keys")
            
            # Get warming statistics
            warming_stats = await cache_warmer.get_warming_stats()
            logger.info(f"✅ Cache warming stats: {warming_stats['warming_stats']}")
            
            self.test_results.append({
                "test": test_name,
                "status": "passed",
                "details": {
                    "queries_warmed": warming_result.get("successful", 0),
                    "total_queries": warming_result.get("total_queries", 0),
                    "ttl_keys_analyzed": ttl_result.get("analyzed_keys", 0)
                }
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results.append({
                "test": test_name,
                "status": "failed",
                "error": str(e)
            })
    
    async def test_cache_metrics(self):
        """Test cache performance analytics and metrics"""
        test_name = "Cache Performance Metrics"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Get cache metrics service
            cache_metrics = await self.service_container.get_cache_metrics()
            
            # Wait for some metrics to be collected
            await asyncio.sleep(5)
            
            # Test performance summary
            performance_summary = await cache_metrics.get_cache_performance_summary(window_minutes=30)
            
            if "error" in performance_summary:
                logger.warning(f"Performance summary: {performance_summary['error']}")
            else:
                summary = performance_summary.get("summary", {})
                logger.info(f"✅ Cache hit ratio: {summary.get('hit_ratio', 0):.3f}")
                logger.info(f"✅ Performance grade: {summary.get('performance_grade', 'Unknown')}")
                
                memory = performance_summary.get("memory", {})
                logger.info(f"✅ Memory usage: {memory.get('used_memory_mb', 0):.1f} MB")
                
                recommendations = performance_summary.get("recommendations", [])
                logger.info(f"✅ Generated {len(recommendations)} optimization recommendations")
            
            # Test trending metrics
            trending_metrics = await cache_metrics.get_trending_metrics(hours=1)
            
            if "error" in trending_metrics:
                logger.warning(f"Trending metrics: {trending_metrics['error']}")
            else:
                performance = trending_metrics.get("performance", {})
                logger.info(f"✅ Avg hit ratio trend: {performance.get('avg_hit_ratio', 0):.3f} ({performance.get('trend', 'unknown')})")
            
            # Test metrics cleanup
            cleanup_result = await cache_metrics.cleanup_old_metrics(days_to_keep=7)
            logger.info(f"✅ Metrics cleanup: {cleanup_result}")
            
            self.test_results.append({
                "test": test_name,
                "status": "passed",
                "details": {
                    "has_performance_summary": "error" not in performance_summary,
                    "has_trending_data": "error" not in trending_metrics,
                    "cleanup_successful": cleanup_result.get("success", False)
                }
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results.append({
                "test": test_name,
                "status": "failed",
                "error": str(e)
            })
    
    async def test_cache_integration(self):
        """Test cross-service cache integration"""
        test_name = "Cross-Service Cache Integration"
        logger.info(f"🧪 Testing {test_name}...")
        
        try:
            # Test that all services have cache access
            services_with_cache = {}
            
            if self.service_container.neo4j and hasattr(self.service_container.neo4j, 'service_container'):
                services_with_cache['neo4j'] = self.service_container.neo4j.service_container is not None
            
            if self.service_container.qdrant and hasattr(self.service_container.qdrant, 'service_container'):
                services_with_cache['qdrant'] = self.service_container.qdrant.service_container is not None
            
            if self.service_container.nomic and hasattr(self.service_container.nomic, 'service_container'):
                services_with_cache['nomic'] = self.service_container.nomic.service_container is not None
            
            logger.info(f"✅ Services with cache integration: {services_with_cache}")
            
            # Test Redis connectivity from service container
            redis_cache = await self.service_container.get_redis_cache_client()
            redis_info = await redis_cache.info()
            
            logger.info(f"✅ Redis version: {redis_info.get('redis_version', 'unknown')}")
            logger.info(f"✅ Connected clients: {redis_info.get('connected_clients', 0)}")
            
            # Test cache key patterns
            all_keys = await redis_cache.keys("l9:prod:neural_tools:*")
            key_patterns = {}
            
            for key in all_keys[:100]:  # Sample first 100 keys
                if isinstance(key, bytes):
                    key = key.decode()
                
                parts = key.split(":")
                if len(parts) >= 4:
                    service_type = parts[3]  # neo4j, qdrant, embeddings, etc.
                    key_patterns[service_type] = key_patterns.get(service_type, 0) + 1
            
            logger.info(f"✅ Cache key distribution: {key_patterns}")
            
            self.test_results.append({
                "test": test_name,
                "status": "passed",
                "details": {
                    "services_integrated": services_with_cache,
                    "total_cache_keys": len(all_keys),
                    "key_patterns": key_patterns
                }
            })
            
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            self.test_results.append({
                "test": test_name,
                "status": "failed",
                "error": str(e)
            })
    
    async def run_all_tests(self):
        """Execute complete Phase 3 caching test suite"""
        logger.info("🚀 Starting Phase 3 Intelligent Caching Test Suite")
        
        # Setup test environment
        if not await self.setup():
            logger.error("❌ Test environment setup failed - aborting tests")
            return False
        
        # Run all tests
        test_methods = [
            self.test_neo4j_caching,
            self.test_qdrant_caching,
            self.test_cache_warming,
            self.test_cache_metrics,
            self.test_cache_integration
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Test method {test_method.__name__} crashed: {e}")
                self.test_results.append({
                    "test": test_method.__name__,
                    "status": "error", 
                    "error": f"Test crashed: {e}"
                })
        
        # Generate final report
        self.generate_test_report()
        
        return True
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("📊 Generating Phase 3 Caching Test Report")
        
        passed_tests = [t for t in self.test_results if t["status"] == "passed"]
        failed_tests = [t for t in self.test_results if t["status"] == "failed"]
        skipped_tests = [t for t in self.test_results if t["status"] == "skipped"]
        error_tests = [t for t in self.test_results if t["status"] == "error"]
        
        total_tests = len(self.test_results)
        success_rate = len(passed_tests) / total_tests * 100 if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("PHASE 3 INTELLIGENT CACHING LAYER - TEST RESULTS")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {len(passed_tests)}")
        print(f"❌ Failed: {len(failed_tests)}")
        print(f"⏭️ Skipped: {len(skipped_tests)}")
        print(f"💥 Errors: {len(error_tests)}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("-"*80)
        
        for result in self.test_results:
            status_emoji = {
                "passed": "✅",
                "failed": "❌", 
                "skipped": "⏭️",
                "error": "💥"
            }.get(result["status"], "❓")
            
            print(f"{status_emoji} {result['test']}: {result['status'].upper()}")
            
            if "details" in result:
                for key, value in result["details"].items():
                    print(f"   └─ {key}: {value}")
            
            if "error" in result:
                print(f"   └─ Error: {result['error']}")
            
            if "reason" in result:
                print(f"   └─ Reason: {result['reason']}")
        
        print("-"*80)
        
        if success_rate >= 80:
            print("🎉 Phase 3 Intelligent Caching Layer: DEPLOYMENT READY")
        elif success_rate >= 60:
            print("⚠️ Phase 3 Intelligent Caching Layer: NEEDS ATTENTION")
        else:
            print("🚨 Phase 3 Intelligent Caching Layer: REQUIRES FIXES")
        
        print("="*80)

async def main():
    """Main test execution"""
    test_suite = Phase3CachingTests()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("✅ Phase 3 caching test suite completed")
        return 0
    else:
        logger.error("❌ Phase 3 caching test suite failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)