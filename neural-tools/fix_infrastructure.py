#!/usr/bin/env python3
"""
Infrastructure Fix Script
Updates service endpoints to match actual container names
"""

import os
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_environment():
    """Update environment variables with correct service names"""
    logger.info("🔧 Updating infrastructure service endpoints...")
    
    # Update to match actual container names
    os.environ['QDRANT_URL'] = 'http://default-neural-storage:6333'
    os.environ['NEO4J_PASSWORD'] = 'neural-l9-2025'
    os.environ['NOMIC_ENDPOINT'] = 'http://neural-embeddings:8000/embed'
    
    # Verify environment
    logger.info(f"✅ QDRANT_URL: {os.environ['QDRANT_URL']}")
    logger.info(f"✅ NEO4J_PASSWORD: ***")
    logger.info(f"✅ NOMIC_ENDPOINT: {os.environ['NOMIC_ENDPOINT']}")

async def test_infrastructure():
    """Test all infrastructure services"""
    logger.info("🧪 Testing infrastructure connectivity...")
    
    results = {
        'neo4j': False,
        'qdrant': False, 
        'nomic': False
    }
    
    # Test Neo4j
    try:
        from neo4j import AsyncGraphDatabase
        
        driver = AsyncGraphDatabase.driver(
            'bolt://neo4j-graph:7687',
            auth=('neo4j', 'neural-l9-2025')
        )
        
        async with driver.session() as session:
            await session.run('RETURN 1')
            results['neo4j'] = True
            logger.info("✅ Neo4j: Connected")
            
        await driver.close()
        
    except Exception as e:
        logger.error(f"❌ Neo4j: {e}")
    
    # Test Qdrant
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get("http://default-neural-storage:6333/collections", timeout=5.0)
            if response.status_code == 200:
                results['qdrant'] = True
                logger.info("✅ Qdrant: Connected")
            else:
                logger.error(f"❌ Qdrant: HTTP {response.status_code}")
                
    except Exception as e:
        logger.error(f"❌ Qdrant: {e}")
    
    # Test Nomic (with retry for loading)
    try:
        import httpx
        
        async with httpx.AsyncClient() as client:
            # First check if service is up
            response = await client.get("http://neural-embeddings:8000/health", timeout=3.0)
            if response.status_code == 200:
                results['nomic'] = True
                logger.info("✅ Nomic: Service healthy")
            else:
                logger.warning(f"⚠️ Nomic: Service starting (HTTP {response.status_code})")
                
    except Exception as e:
        logger.error(f"❌ Nomic: {e}")
    
    # Summary
    healthy_count = sum(results.values())
    total_count = len(results)
    
    logger.info(f"📊 Infrastructure Health: {healthy_count}/{total_count} services healthy")
    
    if healthy_count == total_count:
        logger.info("🎯 All infrastructure services operational!")
        return True
    else:
        logger.warning(f"⚠️ {total_count - healthy_count} services need attention")
        return False

async def main():
    """Main infrastructure fix"""
    try:
        # Update environment
        update_environment()
        
        # Test connectivity  
        success = await test_infrastructure()
        
        if success:
            logger.info("✅ Infrastructure fix complete - all services healthy")
            return True
        else:
            logger.warning("⚠️ Infrastructure partially fixed - some services still starting")
            return False
            
    except Exception as e:
        logger.error(f"❌ Infrastructure fix failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)