#!/usr/bin/env python3
"""
Full Integration Test - Verify all components are connected
Tests the complete Neural Tools ecosystem with Phase 3 caching
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add neural-tools src to Python path
neural_tools_src = Path(__file__).parent / "neural-tools" / "src"
sys.path.insert(0, str(neural_tools_src))

# Configure for Docker services
os.environ['NEO4J_HOST'] = 'localhost'
os.environ['NEO4J_PORT'] = '47687'
os.environ['QDRANT_HOST'] = 'localhost'
os.environ['QDRANT_PORT'] = '46333'
os.environ['REDIS_CACHE_HOST'] = 'localhost'
os.environ['REDIS_CACHE_PORT'] = '46379'
os.environ['REDIS_QUEUE_HOST'] = 'localhost'
os.environ['REDIS_QUEUE_PORT'] = '46380'

async def test_integration():
    """Test full integration of all components"""
    print("🔬 FULL INTEGRATION TEST")
    print("="*60)
    
    integration_status = {
        "indexer_to_neo4j": False,
        "indexer_to_qdrant": False,
        "cache_to_redis": False,
        "neo4j_cache_integration": False,
        "qdrant_cache_integration": False,
        "service_container_connected": False
    }
    
    try:
        from servers.services.service_container import ServiceContainer
        
        # Initialize service container
        container = ServiceContainer("integration_test")
        init_result = await container.initialize_all_services()
        
        print(f"\n1️⃣ Service Container Initialization:")
        print(f"   Status: {init_result}")
        
        if init_result.get("success") or any(init_result.get("services", {}).values()):
            integration_status["service_container_connected"] = True
            print(f"   ✅ Service container initialized")
        
        # Test Redis Cache Integration
        print(f"\n2️⃣ Cache Layer Integration:")
        try:
            redis_cache = await container.get_redis_cache_client()
            
            # Test cache operations
            test_key = "l9:prod:neural_tools:integration:test"
            await redis_cache.setex(test_key, 60, json.dumps({"test": "data"}))
            cached = await redis_cache.get(test_key)
            
            if cached:
                integration_status["cache_to_redis"] = True
                print(f"   ✅ Redis cache layer: INTEGRATED")
                
                # Check for cached keys from services
                neo4j_keys = await redis_cache.keys("l9:prod:neural_tools:neo4j:*")
                qdrant_keys = await redis_cache.keys("l9:prod:neural_tools:qdrant:*")
                
                print(f"   📊 Neo4j cached queries: {len(neo4j_keys)}")
                print(f"   📊 Qdrant cached searches: {len(qdrant_keys)}")
                
                if neo4j_keys:
                    integration_status["neo4j_cache_integration"] = True
                if qdrant_keys:
                    integration_status["qdrant_cache_integration"] = True
                    
        except Exception as e:
            print(f"   ❌ Cache integration failed: {e}")
        
        # Check Indexer Integration
        print(f"\n3️⃣ Indexer Integration:")
        
        # Check indexer health endpoint
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                health = await client.get("http://localhost:48080/health", timeout=5.0)
                if health.status_code == 200:
                    health_data = health.json()
                    print(f"   ✅ Indexer health: {health_data.get('status', 'unknown')}")
                    
                    # Check if indexer has processed files
                    if health_data.get("files_indexed", 0) > 0:
                        integration_status["indexer_to_neo4j"] = True
                        print(f"   📁 Files indexed: {health_data.get('files_indexed', 0)}")
                        
                    if health_data.get("embeddings_created", 0) > 0:
                        integration_status["indexer_to_qdrant"] = True
                        print(f"   🔤 Embeddings created: {health_data.get('embeddings_created', 0)}")
                        
        except Exception as e:
            print(f"   ⚠️  Indexer health check failed: {e}")
        
        # Test Neo4j Integration (if available)
        if container.neo4j:
            print(f"\n4️⃣ Neo4j GraphRAG Integration:")
            try:
                # Test cached query
                result = await container.neo4j.execute_cypher(
                    "MATCH (n) RETURN count(n) as total"
                )
                if result:
                    print(f"   ✅ Neo4j query executed (possibly cached)")
                    print(f"   📊 Total nodes: {result[0].get('total', 0) if result else 0}")
                    
                    # Check if it was cached
                    cache_key = container.neo4j._generate_cache_key(
                        "MATCH (n) RETURN count(n) as total"
                    )
                    cached_result = await redis_cache.get(cache_key)
                    if cached_result:
                        print(f"   ✅ Query result was CACHED")
                        integration_status["neo4j_cache_integration"] = True
                        
            except Exception as e:
                print(f"   ⚠️  Neo4j integration: {e}")
        
        # Test Qdrant Integration (if available)
        if container.qdrant:
            print(f"\n5️⃣ Qdrant Vector Integration:")
            try:
                collections = await container.qdrant.get_collections()
                print(f"   ✅ Qdrant connected")
                print(f"   📊 Collections: {collections}")
                
                # Check if collections list was cached
                cache_key = container.qdrant._generate_cache_key("get_collections")
                cached_collections = await redis_cache.get(cache_key)
                if cached_collections:
                    print(f"   ✅ Collections list was CACHED")
                    integration_status["qdrant_cache_integration"] = True
                    
            except Exception as e:
                print(f"   ⚠️  Qdrant integration: {e}")
        
        # Test Cache Metrics
        print(f"\n6️⃣ Cache Metrics Integration:")
        try:
            cache_metrics = await container.get_cache_metrics()
            summary = await cache_metrics.get_cache_performance_summary(window_minutes=5)
            
            if "error" not in summary:
                print(f"   ✅ Cache metrics operational")
                print(f"   📊 Hit ratio: {summary.get('summary', {}).get('hit_ratio', 0):.2%}")
                print(f"   📊 Memory: {summary.get('memory', {}).get('used_memory_mb', 0):.1f} MB")
            
            await cache_metrics.shutdown()
            
        except Exception as e:
            print(f"   ⚠️  Metrics integration: {e}")
        
    except Exception as e:
        print(f"\n❌ Integration test error: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"🔍 INTEGRATION STATUS SUMMARY")
    print("="*60)
    
    integrated_count = sum(1 for v in integration_status.values() if v)
    total_checks = len(integration_status)
    
    for component, status in integration_status.items():
        emoji = "✅" if status else "❌"
        component_name = component.replace("_", " ").title()
        print(f"{emoji} {component_name}: {'INTEGRATED' if status else 'NOT INTEGRATED'}")
    
    print("-"*60)
    integration_percentage = (integrated_count / total_checks) * 100
    print(f"Integration Score: {integrated_count}/{total_checks} ({integration_percentage:.0f}%)")
    
    if integration_percentage >= 80:
        print("\n🎉 FULL INTEGRATION VERIFIED!")
        print("The Neural Tools ecosystem with Phase 3 caching is fully integrated.")
        print("\nKey Integration Points:")
        print("  • Indexer → Neo4j → Cache Layer → Client")
        print("  • Indexer → Qdrant → Cache Layer → Client")
        print("  • All queries/searches are cached intelligently")
        print("  • Metrics and monitoring are operational")
    elif integration_percentage >= 50:
        print("\n⚠️  PARTIAL INTEGRATION")
        print("Core components are integrated but some connections need attention.")
    else:
        print("\n🚨 INTEGRATION ISSUES DETECTED")
        print("Multiple components are not properly connected.")
    
    print("="*60)
    
    return integration_percentage >= 80

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)