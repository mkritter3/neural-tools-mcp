#!/usr/bin/env python3
"""
ADR-0050 Recovery Script: Backfill Neo4j Chunk nodes from existing Qdrant data
One-time migration to synchronize existing chunks between databases
"""

import asyncio
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Set, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from servers.services.service_container import ServiceContainer
from servers.services.project_context_manager import ProjectContextManager
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkBackfiller:
    """
    Backfills Neo4j Chunk nodes from existing Qdrant data
    Implements ADR-0050 synchronization recovery
    """

    def __init__(self, project_name: str = None):
        self.project_name = project_name
        self.container = None
        self.stats = {
            'total_qdrant': 0,
            'total_neo4j_before': 0,
            'migrated': 0,
            'skipped': 0,
            'failed': 0,
            'relationships_created': 0
        }

    async def initialize(self):
        """Initialize services and connections"""
        logger.info("Initializing service container...")

        # Initialize container with project context manager
        from servers.services.project_context_manager import ProjectContextManager
        context_manager = ProjectContextManager()

        # Get the current project first
        if not self.project_name:
            project_info = await context_manager.get_current_project()
            self.project_name = project_info.get('project', 'claude-l9-template')

        logger.info(f"Using project: {self.project_name}")

        # Initialize container with the context manager
        self.container = ServiceContainer(context_manager=context_manager)
        # initialize() returns bool, not awaitable
        initialized = self.container.initialize()
        if not initialized:
            raise RuntimeError("Failed to initialize service container")

        # Verify connections
        await self.verify_connections()

    async def verify_connections(self):
        """Verify both Neo4j and Qdrant are accessible"""
        # Test Neo4j - use direct driver since we don't have async wrapper
        try:
            with self.container.neo4j_client.session() as session:
                result = session.run("RETURN 1 as test")
                test_val = result.single()['test']
                if test_val != 1:
                    raise ConnectionError(f"Neo4j test query failed")
            logger.info("✅ Neo4j connection verified")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise

        # Test Qdrant
        try:
            collections = self.container.qdrant_client.get_collections()
            logger.info(f"✅ Qdrant connection verified ({len(collections.collections)} collections found)")
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {e}")
            raise

    async def count_existing_chunks(self) -> Dict[str, int]:
        """Count chunks in both databases before migration"""
        collection_name = f"project-{self.project_name}"

        # Count Qdrant chunks
        try:
            from qdrant_client.models import CountFilter
            count_result = self.container.qdrant_client.count(
                collection_name=collection_name,
                count_filter=CountFilter()
            )
            qdrant_count = count_result.count
        except Exception as e:
            logger.warning(f"Could not count Qdrant chunks: {e}")
            qdrant_count = 0

        # Count Neo4j Chunk nodes
        query = """
        MATCH (c:Chunk) WHERE c.project = $project
        RETURN count(c) as count
        """

        neo4j_count = 0
        with self.container.neo4j_client.session() as session:
            result = session.run(query, {'project': self.project_name})
            neo4j_count = result.single()['count']

        self.stats['total_qdrant'] = qdrant_count
        self.stats['total_neo4j_before'] = neo4j_count

        logger.info(f"Current state:")
        logger.info(f"  Qdrant chunks: {qdrant_count}")
        logger.info(f"  Neo4j chunks: {neo4j_count}")
        logger.info(f"  Sync rate: {(neo4j_count/qdrant_count*100) if qdrant_count > 0 else 0:.1f}%")

        return {'qdrant': qdrant_count, 'neo4j': neo4j_count}

    async def backfill_chunks(self, batch_size: int = 100, dry_run: bool = False):
        """
        Main backfill process - migrate chunks from Qdrant to Neo4j
        """
        collection_name = f"project-{self.project_name}"

        logger.info(f"Starting backfill from collection: {collection_name}")
        if dry_run:
            logger.info("🔍 DRY RUN MODE - No changes will be made")

        offset = None
        batch_num = 0

        while True:
            batch_num += 1
            logger.info(f"Processing batch {batch_num} (size: {batch_size})...")

            try:
                # Scroll through Qdrant collection
                scroll_result = self.container.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # Don't need vectors for Neo4j
                )

                points, next_offset = scroll_result

                if not points:
                    logger.info("No more points to process")
                    break

                # Process batch
                await self.process_batch(points, dry_run)

                # Update offset for next iteration
                offset = next_offset
                if not offset:
                    break

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                if not dry_run:
                    # Continue with next batch even if one fails
                    continue
                else:
                    break

    async def process_batch(self, points: List, dry_run: bool):
        """Process a batch of Qdrant points"""

        # Prepare chunk data for Neo4j
        chunks_to_create = []

        for point in points:
            payload = point.payload or {}

            # Extract chunk data
            chunk_id = str(point.id) if not isinstance(point.id, str) else point.id

            # Handle different ID formats
            if 'chunk_id' in payload:
                chunk_id = payload['chunk_id']

            chunk_data = {
                'chunk_id': chunk_id,
                'content': payload.get('content', ''),
                'file_path': payload.get('file_path', ''),
                'start_line': payload.get('start_line', 0),
                'end_line': payload.get('end_line', 0),
                'chunk_type': payload.get('chunk_type', 'code'),
                'metadata': json.dumps(payload.get('metadata', {})),
                'qdrant_id': str(point.id),
                'project': self.project_name
            }

            chunks_to_create.append(chunk_data)

        if dry_run:
            logger.info(f"  [DRY RUN] Would create {len(chunks_to_create)} chunks")
            for chunk in chunks_to_create[:3]:  # Show first 3 as examples
                logger.info(f"    - {chunk['chunk_id'][:16]}... from {chunk['file_path']}")
            return

        # Batch create in Neo4j using UNWIND
        if chunks_to_create:
            await self.create_chunks_in_neo4j(chunks_to_create)

    async def create_chunks_in_neo4j(self, chunks: List[Dict]):
        """Create Chunk nodes and relationships in Neo4j"""

        # Use UNWIND for efficient batch creation
        cypher = """
        UNWIND $chunks as chunk
        MERGE (c:Chunk {chunk_id: chunk.chunk_id, project: chunk.project})
        SET c.content = chunk.content,
            c.start_line = chunk.start_line,
            c.end_line = chunk.end_line,
            c.chunk_type = chunk.chunk_type,
            c.file_path = chunk.file_path,
            c.qdrant_id = chunk.qdrant_id,
            c.metadata = chunk.metadata,
            c.backfilled = true,
            c.backfilled_at = datetime(),
            c.embedding_status = 'completed'
        WITH c, chunk
        MATCH (f:File {path: chunk.file_path, project: chunk.project})
        MERGE (f)-[:HAS_CHUNK]->(c)
        RETURN count(c) as created
        """

        try:
            with self.container.neo4j_client.session() as session:
                result = session.run(cypher, {'chunks': chunks})
                created = result.single()['created']
                self.stats['migrated'] += created
                logger.info(f"  Created {created} Chunk nodes")

        except Exception as e:
            logger.error(f"  Error creating chunks: {e}")
            self.stats['failed'] += len(chunks)

    async def verify_migration(self) -> bool:
        """Verify the migration was successful"""
        logger.info("\n" + "="*60)
        logger.info("MIGRATION VERIFICATION")
        logger.info("="*60)

        # Count final state
        final_counts = await self.count_existing_chunks()

        # Calculate success metrics
        neo4j_added = final_counts['neo4j'] - self.stats['total_neo4j_before']
        sync_rate = (final_counts['neo4j'] / final_counts['qdrant'] * 100) if final_counts['qdrant'] > 0 else 0

        logger.info(f"\nMigration Statistics:")
        logger.info(f"  Qdrant chunks: {final_counts['qdrant']}")
        logger.info(f"  Neo4j chunks before: {self.stats['total_neo4j_before']}")
        logger.info(f"  Neo4j chunks after: {final_counts['neo4j']}")
        logger.info(f"  Chunks added: {neo4j_added}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"  Final sync rate: {sync_rate:.1f}%")

        # Sample validation - check some chunks exist in both
        sample_validation = await self.validate_sample_chunks()

        success = sync_rate >= 95.0

        if success:
            logger.info("\n✅ Migration SUCCESSFUL!")
            logger.info(f"   Achieved {sync_rate:.1f}% synchronization")
        else:
            logger.warning("\n⚠️ Migration INCOMPLETE")
            logger.warning(f"   Only {sync_rate:.1f}% synchronization (target: 95%)")
            logger.warning("   Manual investigation required")

        return success

    async def validate_sample_chunks(self, sample_size: int = 10) -> bool:
        """Validate that sample chunks have matching content"""
        collection_name = f"project-{self.project_name}"

        logger.info(f"\nValidating {sample_size} random chunks...")

        try:
            # Get sample from Qdrant
            scroll_result = self.container.qdrant_client.scroll(
                collection_name=collection_name,
                limit=sample_size,
                with_payload=True,
                with_vectors=False
            )

            points, _ = scroll_result

            mismatches = 0
            for point in points:
                chunk_id = point.payload.get('chunk_id', str(point.id))
                qdrant_content = point.payload.get('content', '')

                # Check in Neo4j
                query = """
                MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
                RETURN c.content as content
                """

                with self.container.neo4j_client.session() as session:
                    result = session.run(query, {'chunk_id': chunk_id, 'project': self.project_name})
                    record = result.single()

                    if record:
                        neo4j_content = record.get('content', '')
                        if qdrant_content != neo4j_content:
                            mismatches += 1
                            logger.warning(f"  Content mismatch for chunk {chunk_id[:16]}...")
                    else:
                        mismatches += 1
                        logger.warning(f"  Chunk {chunk_id[:16]}... not found in Neo4j")

            if mismatches == 0:
                logger.info(f"  ✅ All {sample_size} sample chunks validated successfully")
                return True
            else:
                logger.warning(f"  ⚠️ {mismatches}/{sample_size} chunks have issues")
                return False

        except Exception as e:
            logger.error(f"Sample validation failed: {e}")
            return False

    async def create_missing_file_nodes(self):
        """Create File nodes for orphaned chunks if needed"""
        logger.info("\nChecking for missing File nodes...")

        # Find unique file paths from chunks
        query = """
        MATCH (c:Chunk) WHERE c.project = $project
        WITH DISTINCT c.file_path as file_path
        WHERE file_path IS NOT NULL AND file_path <> ''
        OPTIONAL MATCH (f:File {path: file_path, project: $project})
        WHERE f IS NULL
        RETURN file_path
        LIMIT 100
        """

        with self.container.neo4j_client.session() as session:
            result = session.run(query, {'project': self.project_name})
            missing_files = [r['file_path'] for r in result if r.get('file_path')]

            if missing_files:
                logger.info(f"Found {len(missing_files)} missing File nodes, creating...")

                for file_path in missing_files:
                    create_query = """
                    MERGE (f:File {path: $path, project: $project})
                    SET f.backfilled = true,
                        f.created_at = datetime()
                    """

                    session.run(create_query, {'path': file_path, 'project': self.project_name})

                logger.info(f"  Created {len(missing_files)} File nodes")
            else:
                logger.info("  No missing File nodes found")


async def main():
    """Main entry point for backfill script"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Backfill Neo4j Chunk nodes from Qdrant (ADR-0050)'
    )
    parser.add_argument(
        '--project',
        type=str,
        help='Project name (default: auto-detect)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for processing (default: 100)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate migration without making changes'
    )
    parser.add_argument(
        '--create-files',
        action='store_true',
        help='Create missing File nodes for orphaned chunks'
    )

    args = parser.parse_args()

    # Create and run backfiller
    backfiller = ChunkBackfiller(project_name=args.project)

    try:
        # Initialize
        await backfiller.initialize()

        # Show current state
        await backfiller.count_existing_chunks()

        # Create missing files if requested
        if args.create_files:
            await backfiller.create_missing_file_nodes()

        # Run backfill
        await backfiller.backfill_chunks(
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )

        # Verify results
        if not args.dry_run:
            success = await backfiller.verify_migration()
            sys.exit(0 if success else 1)
        else:
            logger.info("\n🔍 DRY RUN completed - no changes made")
            logger.info("Run without --dry-run to perform actual migration")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())