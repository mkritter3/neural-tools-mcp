#!/usr/bin/env python3
"""
Add performance-enhancing constraints and indexes to Neo4j.
These ensure data integrity and improve query performance.
"""

import asyncio
import os
from neo4j import AsyncGraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection details
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:47687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'graphrag-password')


async def add_constraints_and_indexes():
    """Add constraints and indexes for better Neo4j performance."""

    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    try:
        async with driver.session() as session:

            # 1. Composite constraint for File nodes (project + path uniqueness)
            logger.info("Creating composite constraint for File nodes...")
            try:
                await session.run("""
                    CREATE CONSTRAINT file_project_path_unique IF NOT EXISTS
                    FOR (f:File)
                    REQUIRE (f.project, f.path) IS UNIQUE
                """)
                logger.info("✅ File composite constraint created")
            except Exception as e:
                logger.warning(f"File constraint may already exist: {e}")

            # 2. Composite constraint for CodeChunk nodes
            logger.info("Creating composite constraint for CodeChunk nodes...")
            try:
                await session.run("""
                    CREATE CONSTRAINT chunk_project_id_unique IF NOT EXISTS
                    FOR (c:CodeChunk)
                    REQUIRE (c.project, c.chunk_id) IS UNIQUE
                """)
                logger.info("✅ CodeChunk composite constraint created")
            except Exception as e:
                logger.warning(f"CodeChunk constraint may already exist: {e}")

            # 3. Index on File.file_type for type-based queries
            logger.info("Creating index on File.file_type...")
            try:
                await session.run("""
                    CREATE INDEX file_type_index IF NOT EXISTS
                    FOR (f:File)
                    ON (f.file_type)
                """)
                logger.info("✅ File.file_type index created")
            except Exception as e:
                logger.warning(f"File type index may already exist: {e}")

            # 4. Index on CodeChunk.type for finding specific chunk types
            logger.info("Creating index on CodeChunk.type...")
            try:
                await session.run("""
                    CREATE INDEX chunk_type_index IF NOT EXISTS
                    FOR (c:CodeChunk)
                    ON (c.type)
                """)
                logger.info("✅ CodeChunk.type index created")
            except Exception as e:
                logger.warning(f"Chunk type index may already exist: {e}")

            # 5. Composite index for project + file_type queries
            logger.info("Creating composite index for project + file_type...")
            try:
                await session.run("""
                    CREATE INDEX file_project_type_index IF NOT EXISTS
                    FOR (f:File)
                    ON (f.project, f.file_type)
                """)
                logger.info("✅ File project+type composite index created")
            except Exception as e:
                logger.warning(f"Composite index may already exist: {e}")

            # 6. Text index for code search (if APOC is available)
            logger.info("Creating full-text search index...")
            try:
                await session.run("""
                    CALL db.index.fulltext.createNodeIndex(
                        'codeSearchIndex',
                        ['CodeChunk'],
                        ['content', 'summary'],
                        {analyzer: 'standard-no-stop-words'}
                    )
                """)
                logger.info("✅ Full-text search index created")
            except Exception as e:
                logger.warning(f"Full-text index may already exist or APOC not available: {e}")

            # 7. Relationship indexes for traversal performance
            logger.info("Creating relationship indexes...")
            try:
                # Index for CONTAINS relationship
                await session.run("""
                    CREATE INDEX rel_contains_index IF NOT EXISTS
                    FOR ()-[r:CONTAINS]->()
                    ON (r.project)
                """)

                # Index for IMPORTS relationship
                await session.run("""
                    CREATE INDEX rel_imports_index IF NOT EXISTS
                    FOR ()-[r:IMPORTS]->()
                    ON (r.project)
                """)

                # Index for CALLS relationship
                await session.run("""
                    CREATE INDEX rel_calls_index IF NOT EXISTS
                    FOR ()-[r:CALLS]->()
                    ON (r.project)
                """)

                logger.info("✅ Relationship indexes created")
            except Exception as e:
                logger.warning(f"Relationship indexes may already exist: {e}")

            # 8. Verify all constraints and indexes
            logger.info("\n📊 Verifying constraints and indexes...")

            # Show constraints
            result = await session.run("SHOW CONSTRAINTS")
            constraints = await result.values()
            logger.info(f"Total constraints: {len(constraints)}")
            for c in constraints[:5]:  # Show first 5
                logger.info(f"  - {c[1]} on {c[2]}")

            # Show indexes
            result = await session.run("SHOW INDEXES")
            indexes = await result.values()
            logger.info(f"Total indexes: {len(indexes)}")
            for idx in indexes[:5]:  # Show first 5
                logger.info(f"  - {idx[1]} on {idx[2]}")

            logger.info("\n✅ All Neo4j performance optimizations applied successfully!")

    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(add_constraints_and_indexes())