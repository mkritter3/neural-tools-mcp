#!/usr/bin/env python3
"""
Migrate old CodeChunk nodes to new Chunk nodes per ADR-0050
One-time migration script to update node labels and relationships
"""

from neo4j import GraphDatabase
import sys

def migrate_codechunks():
    """Migrate CodeChunk nodes to Chunk nodes"""

    driver = GraphDatabase.driver('bolt://localhost:47687', auth=('neo4j', 'graphrag-password'))

    with driver.session() as session:
        # Count existing CodeChunk nodes
        result = session.run("MATCH (c:CodeChunk) RETURN count(c) as count")
        codechunk_count = result.single()['count']
        print(f"Found {codechunk_count} CodeChunk nodes to migrate")

        if codechunk_count == 0:
            print("No CodeChunk nodes to migrate")
            return

        # Step 1: Add Chunk label to all CodeChunk nodes (in batches to handle constraints)
        print("Step 1: Adding Chunk label to CodeChunk nodes...")

        # Process in batches to avoid constraint violations
        batch_size = 100
        total_migrated = 0
        offset = 0

        while True:
            result = session.run("""
                MATCH (c:CodeChunk)
                WHERE NOT c:Chunk
                WITH c SKIP $offset LIMIT $batch_size
                SET c:Chunk
                SET c.chunk_id = COALESCE(c.chunk_id, c.id, c.qdrant_id, toString(id(c)))
                RETURN count(c) as migrated
            """, offset=offset, batch_size=batch_size)

            batch_migrated = result.single()['migrated']
            total_migrated += batch_migrated

            if batch_migrated == 0:
                break

            print(f"  Migrated batch: {batch_migrated} nodes (total: {total_migrated})")
            offset += batch_size

        print(f"  Added Chunk label to {total_migrated} nodes total")

        # Step 2: Update relationships from PART_OF to HAS_CHUNK
        print("Step 2: Creating HAS_CHUNK relationships...")
        result = session.run("""
            MATCH (c:Chunk)-[r:PART_OF]->(f:File)
            CREATE (f)-[:HAS_CHUNK]->(c)
            DELETE r
            RETURN count(c) as updated
        """)
        updated = result.single()['updated']
        print(f"  Updated {updated} relationships from PART_OF to HAS_CHUNK")

        # Step 3: Remove CodeChunk label (optional, keep for audit)
        # Uncomment if you want to remove the old label
        # print("Step 3: Removing CodeChunk label...")
        # result = session.run("""
        #     MATCH (c:CodeChunk)
        #     REMOVE c:CodeChunk
        #     RETURN count(c) as cleaned
        # """)
        # cleaned = result.single()['cleaned']
        # print(f"  Removed CodeChunk label from {cleaned} nodes")

        # Verify migration
        print("\nVerifying migration:")
        result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
        chunk_count = result.single()['count']
        print(f"  Chunk nodes: {chunk_count}")

        result = session.run("MATCH (f:File)-[:HAS_CHUNK]->(c:Chunk) RETURN count(*) as count")
        rel_count = result.single()['count']
        print(f"  HAS_CHUNK relationships: {rel_count}")

        print("\n✅ Migration complete!")

    driver.close()

if __name__ == "__main__":
    try:
        migrate_codechunks()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)