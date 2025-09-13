#!/usr/bin/env python3
"""
Migration script to rename Qdrant collections to new naming standard (ADR-0039)
Removes _code suffix and standardizes to clean project-{name} format
"""

import sys
import asyncio
import logging
from pathlib import Path
from qdrant_client import QdrantClient

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "neural-tools" / "src"))
from servers.config.collection_naming import collection_naming

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def migrate_collections():
    """Migrate existing collections to new naming standard without _code suffix"""

    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=46333)

    try:
        # Get all collections
        response = client.get_collections()
        collections = [col.name for col in response.collections]

        logger.info(f"Found {len(collections)} collections")

        migrations_needed = []

        for old_name in collections:
            # Check if this collection needs migration
            if collection_naming.validate_migration_needed(old_name):
                try:
                    # Parse project name from old format
                    project_name = collection_naming.parse_project_name(old_name)

                    # Get new canonical name (without _code suffix)
                    new_name = collection_naming.get_collection_name(project_name)

                    if old_name != new_name:
                        migrations_needed.append((old_name, new_name, project_name))
                        logger.info(f"Migration needed: {old_name} -> {new_name}")
                except ValueError as e:
                    logger.warning(f"Skipping unknown collection format: {old_name} ({e})")

        if not migrations_needed:
            logger.info("No migrations needed - all collections already use standard naming")
            return

        # Perform migrations
        logger.info(f"Starting migration of {len(migrations_needed)} collections...")

        for old_name, new_name, project_name in migrations_needed:
            logger.info(f"Migrating: {old_name} -> {new_name}")

            # Check if new name already exists
            if new_name in collections:
                logger.warning(f"Target collection {new_name} already exists, skipping")
                continue

            try:
                # Get collection info
                old_collection = client.get_collection(old_name)

                # Get all points from old collection
                logger.info(f"Reading points from {old_name}...")
                points = []
                offset = None
                while True:
                    result = client.scroll(
                        collection_name=old_name,
                        limit=100,
                        offset=offset
                    )
                    points.extend(result[0])
                    offset = result[1]
                    if offset is None:
                        break

                logger.info(f"Found {len(points)} points to migrate")

                # Create new collection with same config
                from qdrant_client.models import Distance, VectorParams

                # Get vector config from old collection
                vectors_config = old_collection.config.params.vectors

                # Create new collection
                client.create_collection(
                    collection_name=new_name,
                    vectors_config=VectorParams(
                        size=768,  # Nomic embedding dimension
                        distance=Distance.COSINE
                    ) if not vectors_config else vectors_config
                )
                logger.info(f"Created new collection: {new_name}")

                # Copy all points to new collection
                if points:
                    from qdrant_client.models import PointStruct

                    # Convert Record objects to PointStruct format
                    point_structs = []
                    for point in points:
                        point_structs.append(PointStruct(
                            id=point.id,
                            vector=point.vector,
                            payload=point.payload if hasattr(point, 'payload') else {}
                        ))

                    client.upsert(
                        collection_name=new_name,
                        points=point_structs
                    )
                    logger.info(f"Copied {len(point_structs)} points to {new_name}")

                # Keep old collection for safety (can delete manually later)
                logger.info(f"Migration complete. Old collection {old_name} preserved for rollback.")
                logger.info(f"To delete old collection, run: client.delete_collection('{old_name}')")

            except Exception as e:
                logger.error(f"Failed to migrate {old_name}: {e}")
                continue

        logger.info("Migration complete!")

        # Show final state
        response = client.get_collections()
        final_collections = [col.name for col in response.collections]
        logger.info(f"Final collections: {final_collections}")

        # Show aliases
        for col in response.collections:
            if col.aliases:
                logger.info(f"Collection {col.name} has aliases: {col.aliases}")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(migrate_collections())