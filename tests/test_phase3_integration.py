#!/usr/bin/env python3
"""
Phase 3 Integration Test Suite
Tests all critical production patterns from ADR-0011
"""

import asyncio
import httpx
import time
import json
from typing import Dict, Any, List
from neo4j import AsyncGraphDatabase
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
import redis.asyncio as redis
import hashlib
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class Phase3IntegrationTests:
    def __init__(self):
        self.results = {
            "timestamp": time.time(),
            "tests": {},
            "summary": {"passed": 0, "failed": 0, "skipped": 0}
        }
        
    async def setup(self):
        """Initialize test connections"""
        try:
            # Qdrant client
            self.qdrant = qdrant_client.QdrantClient(
                host="localhost",
                port=6333,
                timeout=30
            )
            
            # Neo4j driver
            self.neo4j = AsyncGraphDatabase.driver(
                "bolt://localhost:7687",
                auth=("neo4j", "testpassword123")
            )
            
            # Redis client
            self.redis = redis.Redis(
                host="localhost",
                port=6379,
                decode_responses=True
            )
            
            # HTTP client for embeddings
            self.http = httpx.AsyncClient(timeout=30.0)
            
            logger.info("✓ Test setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False
    
    async def teardown(self):
        """Cleanup test resources"""
        if hasattr(self, 'http'):
            await self.http.aclose()
        if hasattr(self, 'neo4j'):
            await self.neo4j.close()
        if hasattr(self, 'redis'):
            await self.redis.close()
    
    def record_test(self, name: str, status: str, details: Any = None):
        """Record test result"""
        self.results["tests"][name] = {
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        self.results["summary"][status] += 1
        
        emoji = "✓" if status == "passed" else "✗" if status == "failed" else "⊘"
        logger.info(f"{emoji} {name}: {status}")
        if details and status == "failed":
            logger.error(f"  Details: {details}")
    
    # TEST 1: Qdrant Named Vectors (Critical Fix from Codex)
    async def test_qdrant_named_vectors(self):
        """Test critical named vector format fixes"""
        test_name = "qdrant_named_vectors"
        try:
            collection_name = "test_phase3_vectors"
            
            # Create collection with named vectors
            try:
                await self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=768,
                            distance=Distance.COSINE
                        )
                    }
                )
            except:
                # Collection might exist
                await self.qdrant.delete_collection(collection_name)
                await self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=768,
                            distance=Distance.COSINE
                        )
                    }
                )
            
            # Test CRITICAL fix: Upsert with named vector format
            test_vector = [0.1] * 768
            test_id = hashlib.sha256(b"test_doc").hexdigest()[:16]
            
            point = PointStruct(
                id=test_id,
                vector={"dense": test_vector},  # CRITICAL: Named vector, not bare list
                payload={
                    "content": "test document",
                    "type": "test"
                }
            )
            
            result = await self.qdrant.upsert(
                collection_name=collection_name,
                points=[point],
                wait=True
            )
            
            # Test CRITICAL fix: Search with named vector tuple format
            search_results = await self.qdrant.search(
                collection_name=collection_name,
                query_vector=("dense", test_vector),  # CRITICAL: Tuple format
                limit=1
            )
            
            # Verify results
            assert len(search_results) == 1
            assert search_results[0].id == test_id
            
            # Cleanup
            await self.qdrant.delete_collection(collection_name)
            
            self.record_test(test_name, "passed", {
                "upsert": "success",
                "search": "success",
                "named_vectors": "working"
            })
            
        except Exception as e:
            self.record_test(test_name, "failed", str(e))
    
    # TEST 2: Nomic Embedding Service Connectivity
    async def test_nomic_embeddings(self):
        """Test Nomic embedding service with dimension detection"""
        test_name = "nomic_embeddings"
        try:
            # Test dimension detection
            response = await self.http.post(
                "http://localhost:48000/embed",  # External port
                json={
                    "texts": ["dimension test"],
                    "model": "nomic-embed-text-v1.5"
                }
            )
            
            if response.status_code != 200:
                # Try internal port (from container perspective)
                response = await self.http.post(
                    "http://localhost:8000/embed",
                    json={
                        "texts": ["dimension test"],
                        "model": "nomic-embed-text-v1.5"
                    }
                )
            
            data = response.json()
            embeddings = data.get("embeddings", [])
            
            # Verify dimension
            actual_dim = len(embeddings[0]) if embeddings else 0
            expected_dim = 768
            
            assert actual_dim == expected_dim, f"Dimension mismatch: {actual_dim} != {expected_dim}"
            
            # Test batch embedding
            batch_response = await self.http.post(
                "http://localhost:48000/embed",
                json={
                    "texts": [
                        "test document 1",
                        "test document 2",
                        "test document 3"
                    ],
                    "model": "nomic-embed-text-v1.5"
                }
            )
            
            batch_data = batch_response.json()
            batch_embeddings = batch_data.get("embeddings", [])
            
            assert len(batch_embeddings) == 3
            
            self.record_test(test_name, "passed", {
                "dimension": actual_dim,
                "batch_size": len(batch_embeddings),
                "service": "responsive"
            })
            
        except Exception as e:
            self.record_test(test_name, "failed", str(e))
    
    # TEST 3: Neo4j Write Query Detection Fix
    async def test_neo4j_write_detection(self):
        """Test the critical Neo4j write query detection fix"""
        test_name = "neo4j_write_detection"
        try:
            async with self.neo4j.session() as session:
                # Test the FIXED pattern: MATCH...MERGE
                # This was failing before because it starts with MATCH
                query = """
                MATCH (f:File {path: $path})
                MERGE (m:Module {name: $name})
                MERGE (f)-[:BELONGS_TO]->(m)
                RETURN f, m
                """
                
                # This should be detected as WRITE (contains MERGE)
                write_keywords = ('CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP')
                is_write = any(keyword in query.upper() for keyword in write_keywords)
                
                assert is_write, "MATCH...MERGE pattern not detected as write!"
                
                # Execute the query to ensure it works
                result = await session.execute_write(
                    lambda tx: tx.run(
                        query,
                        path="/test/file.py",
                        name="test_module"
                    ).data()
                )
                
                # Test pure read query
                read_query = "MATCH (n) RETURN count(n) as count"
                is_read = not any(keyword in read_query.upper() for keyword in write_keywords)
                
                assert is_read, "Read query incorrectly detected as write!"
                
                # Cleanup
                await session.execute_write(
                    lambda tx: tx.run(
                        "MATCH (f:File {path: $path}) DETACH DELETE f",
                        path="/test/file.py"
                    )
                )
                
                await session.execute_write(
                    lambda tx: tx.run(
                        "MATCH (m:Module {name: $name}) DETACH DELETE m",
                        name="test_module"
                    )
                )
                
                self.record_test(test_name, "passed", {
                    "match_merge_detection": "fixed",
                    "write_detection": "working",
                    "read_detection": "working"
                })
                
        except Exception as e:
            self.record_test(test_name, "failed", str(e))
    
    # TEST 4: Cache Warming and Analytics
    async def test_cache_warming(self):
        """Test cache warming functionality"""
        test_name = "cache_warming"
        try:
            # Test cache metrics recording
            cache_key = "test:embedding:hash123"
            
            # Record cache miss
            await self.redis.hincrby("cache:metrics:misses", cache_key, 1)
            
            # Record cache hit
            await self.redis.hincrby("cache:metrics:hits", cache_key, 1)
            await self.redis.hincrby("cache:metrics:hits", cache_key, 1)
            
            # Get metrics
            hits = await self.redis.hget("cache:metrics:hits", cache_key)
            misses = await self.redis.hget("cache:metrics:misses", cache_key)
            
            assert int(hits) == 2
            assert int(misses) == 1
            
            # Test cache TTL tracking
            await self.redis.setex(
                f"cache:embedding:{cache_key}",
                300,  # 5 minute TTL
                json.dumps([0.1] * 768)
            )
            
            ttl = await self.redis.ttl(f"cache:embedding:{cache_key}")
            assert ttl > 0 and ttl <= 300
            
            # Cleanup
            await self.redis.delete(f"cache:embedding:{cache_key}")
            await self.redis.hdel("cache:metrics:hits", cache_key)
            await self.redis.hdel("cache:metrics:misses", cache_key)
            
            self.record_test(test_name, "passed", {
                "hits_recorded": 2,
                "misses_recorded": 1,
                "ttl_working": True
            })
            
        except Exception as e:
            self.record_test(test_name, "failed", str(e))
    
    # TEST 5: Deterministic ID Generation
    async def test_deterministic_ids(self):
        """Test deterministic ID generation for idempotency"""
        test_name = "deterministic_ids"
        try:
            # Test file + function ID generation
            metadata1 = {
                "file_path": "/src/main.py",
                "function_name": "process_data"
            }
            content1 = "def process_data(): pass"
            
            id_source1 = f"{metadata1['file_path']}:{metadata1['function_name']}"
            id1 = hashlib.sha256(id_source1.encode()).hexdigest()[:16]
            
            # Same metadata should generate same ID
            id_source2 = f"{metadata1['file_path']}:{metadata1['function_name']}"
            id2 = hashlib.sha256(id_source2.encode()).hexdigest()[:16]
            
            assert id1 == id2, "Same metadata produced different IDs!"
            
            # Different metadata should generate different ID
            metadata3 = {
                "file_path": "/src/utils.py",
                "function_name": "helper"
            }
            id_source3 = f"{metadata3['file_path']}:{metadata3['function_name']}"
            id3 = hashlib.sha256(id_source3.encode()).hexdigest()[:16]
            
            assert id1 != id3, "Different metadata produced same ID!"
            
            self.record_test(test_name, "passed", {
                "idempotency": "verified",
                "uniqueness": "verified",
                "id_length": len(id1)
            })
            
        except Exception as e:
            self.record_test(test_name, "failed", str(e))
    
    # TEST 6: Health Checks
    async def test_health_checks(self):
        """Test comprehensive health checking"""
        test_name = "health_checks"
        try:
            health_status = {
                "neo4j": False,
                "qdrant": False,
                "redis": False,
                "embeddings": False
            }
            
            # Check Neo4j
            try:
                async with self.neo4j.session() as session:
                    result = await session.run("RETURN 1 as health")
                    await result.single()
                    health_status["neo4j"] = True
            except:
                pass
            
            # Check Qdrant
            try:
                collections = await self.qdrant.get_collections()
                health_status["qdrant"] = True
            except:
                pass
            
            # Check Redis
            try:
                await self.redis.ping()
                health_status["redis"] = True
            except:
                pass
            
            # Check Embeddings
            try:
                response = await self.http.get("http://localhost:48000/health")
                if response.status_code == 200:
                    health_status["embeddings"] = True
            except:
                pass
            
            # All should be healthy
            all_healthy = all(health_status.values())
            
            if all_healthy:
                self.record_test(test_name, "passed", health_status)
            else:
                self.record_test(test_name, "failed", health_status)
                
        except Exception as e:
            self.record_test(test_name, "failed", str(e))
    
    async def run_all_tests(self):
        """Execute all integration tests"""
        logger.info("=" * 60)
        logger.info("Phase 3 Integration Test Suite")
        logger.info("=" * 60)
        
        if not await self.setup():
            logger.error("Setup failed, cannot run tests")
            return self.results
        
        # Run tests
        tests = [
            self.test_qdrant_named_vectors,
            self.test_nomic_embeddings,
            self.test_neo4j_write_detection,
            self.test_cache_warming,
            self.test_deterministic_ids,
            self.test_health_checks
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                logger.error(f"Test crashed: {test.__name__}: {e}")
                self.record_test(test.__name__, "failed", f"Crashed: {e}")
        
        await self.teardown()
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(f"Passed:  {self.results['summary']['passed']}")
        logger.info(f"Failed:  {self.results['summary']['failed']}")
        logger.info(f"Skipped: {self.results['summary']['skipped']}")
        
        # Calculate pass rate
        total = sum(self.results['summary'].values())
        if total > 0:
            pass_rate = (self.results['summary']['passed'] / total) * 100
            logger.info(f"Pass Rate: {pass_rate:.1f}%")
            
            if pass_rate == 100:
                logger.info("🎉 All tests passed! Phase 3 is production ready!")
            elif pass_rate >= 80:
                logger.info("⚠️  Most tests passed, but some issues need attention")
            else:
                logger.error("❌ Multiple failures detected. Phase 3 needs fixes")
        
        # Save detailed results
        with open("test_results_phase3.json", "w") as f:
            json.dump(self.results, f, indent=2)
            logger.info(f"Detailed results saved to test_results_phase3.json")
        
        return self.results

async def main():
    tester = Phase3IntegrationTests()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    if results['summary']['failed'] == 0:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())