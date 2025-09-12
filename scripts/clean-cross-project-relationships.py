#!/usr/bin/env python3
"""
Clean up cross-project relationships in Neo4j that violate ADR-0029.
"""

import asyncio
from neo4j import AsyncGraphDatabase

# Neo4j connection settings
NEO4J_URI = "bolt://localhost:47687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag-password"

async def clean_cross_project_relationships():
    """Remove all relationships between nodes of different projects."""
    print("\n🧹 Cleaning Cross-Project Relationships")
    print("="*50)
    
    driver = AsyncGraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    
    try:
        async with driver.session() as session:
            # First, count cross-project relationships
            count_result = await session.run("""
                MATCH (n1)-[r]-(n2)
                WHERE n1.project IS NOT NULL 
                AND n2.project IS NOT NULL 
                AND n1.project <> n2.project
                RETURN COUNT(r) AS count
            """)
            count_record = await count_result.single()
            total_count = count_record["count"] if count_record else 0
            
            if total_count == 0:
                print("✅ No cross-project relationships found!")
                return
            
            print(f"⚠️  Found {total_count} cross-project relationships to remove")
            
            # Delete all cross-project relationships
            delete_result = await session.run("""
                MATCH (n1)-[r]-(n2)
                WHERE n1.project IS NOT NULL 
                AND n2.project IS NOT NULL 
                AND n1.project <> n2.project
                DELETE r
                RETURN COUNT(r) AS deleted
            """)
            delete_record = await delete_result.single()
            deleted_count = delete_record["deleted"] if delete_record else 0
            
            print(f"🗑️  Deleted {deleted_count} cross-project relationships")
            
            # Verify cleanup
            verify_result = await session.run("""
                MATCH (n1)-[r]-(n2)
                WHERE n1.project IS NOT NULL 
                AND n2.project IS NOT NULL 
                AND n1.project <> n2.project
                RETURN COUNT(r) AS remaining
            """)
            verify_record = await verify_result.single()
            remaining = verify_record["remaining"] if verify_record else 0
            
            if remaining == 0:
                print("✅ Successfully cleaned all cross-project relationships!")
            else:
                print(f"⚠️  Warning: {remaining} cross-project relationships still remain")
            
    finally:
        await driver.close()
    
    print("="*50)
    print("Cleanup complete!\n")

if __name__ == "__main__":
    asyncio.run(clean_cross_project_relationships())
