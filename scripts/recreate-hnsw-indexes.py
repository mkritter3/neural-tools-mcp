#!/usr/bin/env python3
"""
ADR-0090: Recreate HNSW indexes with optimized parameters
This script drops existing vector indexes and recreates them with:
- M=24 (increased from default 16)
- ef_construction=150 (increased from default 100)
- int8 quantization for 2x speed improvement
"""

import asyncio
from neo4j import AsyncGraphDatabase
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO4J_URI = "neo4j://localhost:47687"
NEO4J_PASSWORD = "graphrag-password"

async def recreate_indexes():
    """Drop and recreate HNSW indexes with optimized parameters"""

    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=("neo4j", NEO4J_PASSWORD)
    )

    try:
        async with driver.session() as session:
            # Drop existing indexes
            logger.info("Dropping existing vector indexes...")

            try:
                await session.run("DROP INDEX chunk_embeddings_index IF EXISTS")
                logger.info("Dropped chunk_embeddings_index")
            except Exception as e:
                logger.warning(f"Could not drop chunk index: {e}")

            try:
                await session.run("DROP INDEX file_embeddings_index IF EXISTS")
                logger.info("Dropped file_embeddings_index")
            except Exception as e:
                logger.warning(f"Could not drop file index: {e}")

            # Wait for indexes to be fully dropped
            await asyncio.sleep(2)

            # Create optimized chunk index
            logger.info("Creating optimized chunk_embeddings_index...")
            start_time = time.time()

            await session.run("""
                CREATE VECTOR INDEX chunk_embeddings_index IF NOT EXISTS
                FOR (c:Chunk) ON c.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine',
                        `vector.hnsw.m`: 24,
                        `vector.hnsw.ef_construction`: 150,
                        `vector.quantization.enabled`: true
                    }
                }
            """)

            chunk_time = time.time() - start_time
            logger.info(f"Created chunk_embeddings_index in {chunk_time:.2f}s")

            # Create optimized file index
            logger.info("Creating optimized file_embeddings_index...")
            start_time = time.time()

            await session.run("""
                CREATE VECTOR INDEX file_embeddings_index IF NOT EXISTS
                FOR (f:File) ON f.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine',
                        `vector.hnsw.m`: 24,
                        `vector.hnsw.ef_construction`: 150,
                        `vector.quantization.enabled`: true
                    }
                }
            """)

            file_time = time.time() - start_time
            logger.info(f"Created file_embeddings_index in {file_time:.2f}s")

            # Verify indexes
            logger.info("\nVerifying indexes...")
            result = await session.run("SHOW INDEXES WHERE type = 'VECTOR'")
            indexes = await result.data()

            for index in indexes:
                logger.info(f"Index: {index['name']}")
                if 'options' in index:
                    logger.info(f"  Configuration: {index['options']}")

            # Count vectors
            result = await session.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) as count")
            data = await result.single()
            chunk_count = data['count'] if data else 0

            result = await session.run("MATCH (f:File) WHERE f.embedding IS NOT NULL RETURN count(f) as count")
            data = await result.single()
            file_count = data['count'] if data else 0

            logger.info(f"\nVector statistics:")
            logger.info(f"  Chunks with embeddings: {chunk_count}")
            logger.info(f"  Files with embeddings: {file_count}")

            # Performance benchmark
            if chunk_count > 0:
                logger.info("\nRunning performance benchmark...")

                # Get a sample embedding
                result = await session.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN c.embedding as embedding LIMIT 1")
                data = await result.single()
                sample_embedding = data['embedding'] if data else None

                if sample_embedding:
                    # Test search performance
                    start_time = time.time()
                    result = await session.run("""
                        CALL db.index.vector.queryNodes('chunk_embeddings_index', 10, $embedding)
                        YIELD node, score
                        RETURN node.content, score
                        LIMIT 10
                    """, embedding=sample_embedding)

                    search_time = (time.time() - start_time) * 1000
                    results = await result.data()

                    logger.info(f"  Vector search time: {search_time:.2f}ms")
                    logger.info(f"  Results returned: {len(results)}")

                    if search_time < 50:
                        logger.info("  ✅ EXCELLENT: <50ms (Elite performance)")
                    elif search_time < 100:
                        logger.info("  ✅ GOOD: <100ms (Target met)")
                    elif search_time < 500:
                        logger.info("  ⚠️ OK: <500ms (Baseline)")
                    else:
                        logger.info("  ❌ SLOW: >500ms (Needs optimization)")

            logger.info("\n✅ HNSW indexes recreated with ADR-0090 optimizations!")
            logger.info("Expected improvements:")
            logger.info("  - 2x faster searches with int8 quantization")
            logger.info("  - Better recall with M=24")
            logger.info("  - More thorough indexing with ef_construction=150")

    finally:
        await driver.close()

if __name__ == "__main__":
    asyncio.run(recreate_indexes())