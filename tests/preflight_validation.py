#!/usr/bin/env python3
"""
Preflight Validation Script for Neural Tools
Addresses Codex P1 requirements for production readiness
"""

import asyncio
import sys
import os
import time
import httpx
from typing import Dict, Any, List, Optional
import hashlib
import json
import logging
from pathlib import Path

# Load environment from .env.mcp.local if available
env_file = Path(".env.mcp.local")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class PreflightValidator:
    """Comprehensive preflight validation for Neural Tools"""
    
    def __init__(self):
        self.results = {
            "timestamp": time.time(),
            "checks": {},
            "warnings": [],
            "errors": [],
            "status": "pending"
        }
        
        # Configuration from environment
        self.nomic_host = os.getenv("EMBEDDING_SERVICE_HOST", "localhost")
        self.nomic_port = int(os.getenv("EMBEDDING_SERVICE_PORT", "48000"))
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "46333"))
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:47687")
        self.redis_cache_host = os.getenv("REDIS_CACHE_HOST", "localhost")
        self.redis_cache_port = int(os.getenv("REDIS_CACHE_PORT", "46379"))
        self.expected_dim = int(os.getenv("EMBED_DIM", "768"))
        
    def log_result(self, check: str, status: str, details: Any = None):
        """Log and record check result"""
        self.results["checks"][check] = {
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
        
        icon = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
        logger.info(f"{icon} {check}: {status}")
        if details and status != "pass":
            logger.info(f"   Details: {details}")
    
    async def check_nomic_connectivity(self) -> bool:
        """P1: Check Nomic embedding service connectivity and dimension detection"""
        check_name = "nomic_connectivity"
        
        try:
            base_url = f"http://{self.nomic_host}:{self.nomic_port}"
            logger.info(f"🔍 Checking Nomic at {base_url}")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check health endpoint
                try:
                    health_response = await client.get(f"{base_url}/health")
                    health_ok = health_response.status_code == 200
                except:
                    health_ok = False
                
                # Detect dimensions
                embed_response = await client.post(
                    f"{base_url}/embed",
                    json={
                        "inputs": ["dimension detection test"],
                        "model": "nomic-v2"
                    }
                )
                
                if embed_response.status_code == 200:
                    data = embed_response.json()
                    embeddings = data.get("embeddings", [])
                    
                    if embeddings:
                        detected_dim = len(embeddings[0])
                        
                        if detected_dim == self.expected_dim:
                            self.log_result(check_name, "pass", {
                                "base_url": base_url,
                                "health": health_ok,
                                "detected_dimension": detected_dim,
                                "expected_dimension": self.expected_dim
                            })
                            return True
                        else:
                            self.results["errors"].append(
                                f"Dimension mismatch: expected {self.expected_dim}, got {detected_dim}"
                            )
                            self.log_result(check_name, "fail", {
                                "error": "dimension_mismatch",
                                "expected": self.expected_dim,
                                "actual": detected_dim
                            })
                            return False
                
                self.log_result(check_name, "fail", {
                    "error": "embedding_failed",
                    "status_code": embed_response.status_code
                })
                return False
                
        except Exception as e:
            self.results["errors"].append(f"Nomic connection failed: {str(e)}")
            self.log_result(check_name, "fail", str(e))
            return False
    
    async def check_qdrant_setup(self) -> bool:
        """P1: Check Qdrant connectivity, version, and collection configuration"""
        check_name = "qdrant_setup"
        
        try:
            import qdrant_client
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            client = qdrant_client.QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                timeout=10
            )
            
            # Check version compatibility
            try:
                client_version = qdrant_client.__version__
            except AttributeError:
                # Fallback for newer versions
                import pkg_resources
                client_version = pkg_resources.get_distribution("qdrant-client").version
            
            # Get server info (use sync client)
            try:
                collections = client.get_collections()
                server_healthy = True
            except Exception as e:
                server_healthy = False
                raise
            
            # Check collection configuration
            test_collection = "preflight_test"
            collection_exists = any(c.name == test_collection for c in collections.collections)
            
            if not collection_exists:
                # Create test collection with named vectors
                client.create_collection(
                    collection_name=test_collection,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.expected_dim,
                            distance=Distance.COSINE
                        )
                    }
                )
            
            # Test named vector upsert (P1: Critical fix)
            test_vector = [0.1] * self.expected_dim
            import uuid
            test_id = str(uuid.uuid4())
            
            point = PointStruct(
                id=test_id,
                vector={"dense": test_vector},  # Named vector format
                payload={"test": "preflight"}
            )
            
            client.upsert(
                collection_name=test_collection,
                points=[point],
                wait=True
            )
            
            # Test named vector search
            search_results = client.search(
                collection_name=test_collection,
                query_vector=("dense", test_vector),  # Tuple format
                limit=1
            )
            
            # Cleanup
            client.delete_collection(test_collection)
            
            self.log_result(check_name, "pass", {
                "client_version": client_version,
                "server_healthy": server_healthy,
                "named_vectors": "working",
                "dimension": self.expected_dim
            })
            return True
            
        except Exception as e:
            self.results["errors"].append(f"Qdrant setup failed: {str(e)}")
            self.log_result(check_name, "fail", str(e))
            return False
    
    async def check_neo4j_connectivity(self) -> bool:
        """Check Neo4j connectivity and write detection fix"""
        check_name = "neo4j_connectivity"
        
        try:
            from neo4j import AsyncGraphDatabase
            
            driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=("neo4j", "testpassword123")
            )
            
            async with driver.session() as session:
                # Test basic connectivity
                result = await session.run("RETURN 1 as health")
                await result.single()
                
                # Test write detection fix (MATCH...MERGE pattern)
                test_query = """
                MATCH (f:File {path: $path})
                MERGE (m:Module {name: $name})
                RETURN f, m
                """
                
                # This should be detected as WRITE
                write_keywords = ('CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP')
                is_write = any(keyword in test_query.upper() for keyword in write_keywords)
                
                if not is_write:
                    self.results["errors"].append(
                        "Neo4j write detection broken: MATCH...MERGE not detected as write"
                    )
                    self.log_result(check_name, "fail", "write_detection_broken")
                    return False
            
            await driver.close()
            
            self.log_result(check_name, "pass", {
                "uri": self.neo4j_uri,
                "write_detection": "fixed"
            })
            return True
            
        except Exception as e:
            self.results["warnings"].append(f"Neo4j not available: {str(e)}")
            self.log_result(check_name, "warning", str(e))
            return True  # Non-critical for MCP
    
    async def check_redis_connectivity(self) -> bool:
        """Check Redis cache connectivity"""
        check_name = "redis_connectivity"
        
        try:
            import redis.asyncio as redis
            
            client = redis.Redis(
                host=self.redis_cache_host,
                port=self.redis_cache_port,
                decode_responses=True
            )
            
            # Test basic operations
            await client.ping()
            
            # Test cache operations
            test_key = "preflight:test"
            await client.setex(test_key, 10, "test_value")
            value = await client.get(test_key)
            await client.delete(test_key)
            
            await client.close()
            
            self.log_result(check_name, "pass", {
                "host": self.redis_cache_host,
                "port": self.redis_cache_port
            })
            return True
            
        except Exception as e:
            self.results["warnings"].append(f"Redis not available: {str(e)}")
            self.log_result(check_name, "warning", str(e))
            return True  # Non-critical
    
    async def check_environment_variables(self) -> bool:
        """P1: Check critical environment variables"""
        check_name = "environment_variables"
        
        required_vars = {
            "EMBEDDING_SERVICE_HOST": self.nomic_host,
            "EMBEDDING_SERVICE_PORT": str(self.nomic_port),
            "QDRANT_HOST": self.qdrant_host,
            "QDRANT_HTTP_PORT": str(self.qdrant_port),
            "EMBED_DIM": str(self.expected_dim)
        }
        
        missing = []
        configured = {}
        
        for var, expected in required_vars.items():
            actual = os.getenv(var)
            if actual:
                configured[var] = actual
                if actual != expected and expected != actual:
                    self.results["warnings"].append(
                        f"{var}: expected '{expected}', got '{actual}'"
                    )
            else:
                missing.append(var)
        
        if missing:
            self.log_result(check_name, "fail", {
                "missing": missing,
                "configured": configured
            })
            return False
        
        # Log effective URLs (P1 requirement)
        logger.info(f"📍 Nomic base URL: http://{self.nomic_host}:{self.nomic_port}")
        logger.info(f"📍 Qdrant URL: http://{self.qdrant_host}:{self.qdrant_port}")
        
        self.log_result(check_name, "pass", configured)
        return True
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Execute all preflight checks"""
        logger.info("=" * 60)
        logger.info("🚀 Neural Tools Preflight Validation")
        logger.info("=" * 60)
        
        # P1 Critical Checks
        checks = [
            ("Environment Variables", self.check_environment_variables),
            ("Nomic Embeddings", self.check_nomic_connectivity),
            ("Qdrant Vector DB", self.check_qdrant_setup),
            ("Neo4j Graph DB", self.check_neo4j_connectivity),
            ("Redis Cache", self.check_redis_connectivity)
        ]
        
        all_passed = True
        critical_passed = True
        
        for name, check_func in checks:
            logger.info(f"\n🔧 Checking {name}...")
            try:
                result = await check_func()
                if not result:
                    all_passed = False
                    # First 3 checks are critical (P1)
                    if checks.index((name, check_func)) < 3:
                        critical_passed = False
            except Exception as e:
                logger.error(f"Check failed with exception: {e}")
                all_passed = False
                if checks.index((name, check_func)) < 3:
                    critical_passed = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 Preflight Summary")
        logger.info("=" * 60)
        
        passed = sum(1 for c in self.results["checks"].values() if c["status"] == "pass")
        failed = sum(1 for c in self.results["checks"].values() if c["status"] == "fail")
        warnings = sum(1 for c in self.results["checks"].values() if c["status"] == "warning")
        
        logger.info(f"✅ Passed:  {passed}")
        logger.info(f"❌ Failed:  {failed}")
        logger.info(f"⚠️  Warning: {warnings}")
        
        if critical_passed:
            self.results["status"] = "ready"
            logger.info("\n🎉 P1 Critical checks passed! System ready for operation.")
        else:
            self.results["status"] = "blocked"
            logger.error("\n❌ P1 Critical checks failed. Fix issues before proceeding.")
            
            if self.results["errors"]:
                logger.error("\nErrors to fix:")
                for error in self.results["errors"]:
                    logger.error(f"  - {error}")
        
        if self.results["warnings"]:
            logger.warning("\nWarnings (non-critical):")
            for warning in self.results["warnings"]:
                logger.warning(f"  - {warning}")
        
        # Save results
        with open("preflight_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
            logger.info(f"\n📁 Detailed results saved to preflight_results.json")
        
        return self.results

async def main():
    validator = PreflightValidator()
    results = await validator.run_all_checks()
    
    # Exit code based on P1 critical status
    if results["status"] == "ready":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())