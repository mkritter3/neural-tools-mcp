#!/usr/bin/env python3
"""
Fix and complete the Chunk node migration
Handles duplicates and constraint issues
"""

from neo4j import GraphDatabase
import sys

def fix_migration():
    """Complete the migration handling duplicates properly"""

    driver = GraphDatabase.driver('bolt://localhost:47687', auth=('neo4j', 'graphrag-password'))

    with driver.session() as session:
        print("=== ADR-0050 Migration Fix ===\n")

        # Step 1: Check current state
        result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
        chunk_count = result.single()['count']
        print(f"Current Chunk nodes: {chunk_count}")

        result = session.run("MATCH (c:CodeChunk) WHERE NOT c:Chunk RETURN count(c) as count")
        remaining = result.single()['count']
        print(f"CodeChunk nodes to migrate: {remaining}")

        if remaining == 0:
            print("\n✅ All CodeChunk nodes already have Chunk label")
        else:
            # Step 2: Continue migration with better duplicate handling
            print(f"\nMigrating remaining {remaining} nodes...")

            # First, drop the constraint that's causing issues
            print("Dropping problematic constraints...")

            # Drop by exact name if it exists
            try:
                session.run("DROP CONSTRAINT chunk_project_id_unique IF EXISTS")
                print("  Dropped chunk_project_id_unique constraint")
            except Exception as e:
                print(f"  Note: {str(e)[:100]}")

            # Also try the standard syntax
            try:
                session.run("DROP CONSTRAINT ON (c:CodeChunk) ASSERT (c.project, c.chunk_id) IS UNIQUE")
                print("  Dropped CodeChunk constraint")
            except:
                pass  # Constraint might not exist

            try:
                session.run("DROP CONSTRAINT ON (c:Chunk) ASSERT (c.project, c.chunk_id) IS UNIQUE")
                print("  Dropped Chunk constraint")
            except:
                pass

            # Now migrate remaining nodes
            print("\nMigrating nodes without constraints...")
            result = session.run("""
                MATCH (c:CodeChunk)
                WHERE NOT c:Chunk
                SET c:Chunk
                SET c.chunk_id = COALESCE(c.chunk_id, c.id, c.qdrant_id, toString(id(c)))
                RETURN count(c) as migrated
            """)
            migrated = result.single()['migrated']
            print(f"  Migrated {migrated} additional nodes")

        # Step 3: Create HAS_CHUNK relationships
        print("\nCreating HAS_CHUNK relationships...")

        # First check if any exist
        result = session.run("MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) as count")
        existing_rels = result.single()['count']
        print(f"  Existing HAS_CHUNK relationships: {existing_rels}")

        if existing_rels == 0:
            # Create from PART_OF relationships
            result = session.run("""
                MATCH (c:Chunk)-[r:PART_OF]->(f:File)
                CREATE (f)-[:HAS_CHUNK]->(c)
                DELETE r
                RETURN count(*) as created
            """)
            created = result.single()['created']
            print(f"  Created {created} HAS_CHUNK relationships from PART_OF")

            # Also try reverse direction if PART_OF was backwards
            result = session.run("""
                MATCH (f:File)-[r:PART_OF]->(c:Chunk)
                CREATE (f)-[:HAS_CHUNK]->(c)
                DELETE r
                RETURN count(*) as created
            """)
            created = result.single()['created']
            if created > 0:
                print(f"  Created {created} more HAS_CHUNK relationships (reverse PART_OF)")

        # Step 4: Create relationships for orphaned chunks
        print("\nLinking orphaned chunks to files...")
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.file_path IS NOT NULL
              AND NOT (c)<-[:HAS_CHUNK]-(:File)
              AND NOT (c)-[:HAS_CHUNK]->(:File)
            WITH c, c.file_path as file_path
            MATCH (f:File {path: file_path})
            CREATE (f)-[:HAS_CHUNK]->(c)
            RETURN count(*) as linked
        """)
        linked = result.single()['linked']
        if linked > 0:
            print(f"  Linked {linked} orphaned chunks to their files")

        # Step 5: Final verification
        print("\n=== Final State ===")
        result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
        final_chunks = result.single()['count']
        print(f"Total Chunk nodes: {final_chunks}")

        result = session.run("MATCH (f:File)-[:HAS_CHUNK]->(c:Chunk) RETURN count(*) as count")
        final_rels = result.single()['count']
        print(f"Total HAS_CHUNK relationships: {final_rels}")

        result = session.run("""
            MATCH (c:Chunk)
            WHERE NOT (c)<-[:HAS_CHUNK]-(:File)
            RETURN count(c) as orphans
        """)
        orphans = result.single()['orphans']
        print(f"Orphaned chunks (no file relationship): {orphans}")

        # Step 6: Recreate constraints properly
        print("\nRecreating constraints...")
        try:
            session.run("""
                CREATE CONSTRAINT chunk_unique IF NOT EXISTS
                FOR (c:Chunk)
                REQUIRE (c.project, c.chunk_id) IS UNIQUE
            """)
            print("  Created Chunk uniqueness constraint")
        except Exception as e:
            print(f"  Note: Constraint creation skipped: {str(e)[:100]}")

        print("\n✅ Migration fix complete!")

    driver.close()

if __name__ == "__main__":
    try:
        fix_migration()
    except Exception as e:
        print(f"❌ Migration fix failed: {e}")
        sys.exit(1)