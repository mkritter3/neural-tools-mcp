#!/usr/bin/env python3
"""
Migration script to convert Symbol nodes to properly typed Class/Function/Method nodes.
Fixes ADR-0094: Schema mismatch between Symbol storage and relationship expectations.
"""

import asyncio
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "servers" / "services"))

from neo4j_service import Neo4jService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate_symbols_to_typed_nodes():
    """
    Convert Symbol nodes to Class/Function/Method nodes based on their type.
    This fixes the schema mismatch where relationships expect typed nodes
    but the storage creates generic Symbol nodes.
    """

    # Connect to Neo4j
    neo4j_service = Neo4jService()
    await neo4j_service.connect()

    logger.info("Starting Symbol -> Typed Node migration...")

    # 1. Migrate Symbol nodes with type='class' to Class nodes
    class_migration = """
    MATCH (s:Symbol)
    WHERE s.type = 'class'
    CREATE (c:Class {
        name: s.name,
        qualified_name: s.qualified_name,
        file_path: s.file_path,
        start_line: s.start_line,
        end_line: s.end_line,
        language: s.language,
        docstring: s.docstring,
        project: s.project,
        indexed_at: s.indexed_at
    })
    WITH s, c
    // Transfer any relationships from Symbol to Class
    MATCH (s)-[r]->(target)
    CREATE (c)-[newRel:RELATIONSHIP]->(target)
    SET newRel = properties(r)
    WITH s, c
    MATCH (source)-[r]->(s)
    CREATE (source)-[newRel:RELATIONSHIP]->(c)
    SET newRel = properties(r)
    WITH s
    DETACH DELETE s
    RETURN count(s) as migrated_classes
    """

    result = await neo4j_service.execute_cypher(class_migration, {})
    if result['status'] == 'success' and result['result']:
        count = result['result'][0].get('migrated_classes', 0)
        logger.info(f"✓ Migrated {count} Symbol nodes to Class nodes")

    # 2. Migrate Symbol nodes with type='function' to Function nodes
    function_migration = """
    MATCH (s:Symbol)
    WHERE s.type = 'function'
    CREATE (f:Function {
        name: s.name,
        qualified_name: s.qualified_name,
        file_path: s.file_path,
        start_line: s.start_line,
        end_line: s.end_line,
        language: s.language,
        docstring: s.docstring,
        project: s.project,
        indexed_at: s.indexed_at
    })
    WITH s, f
    // Transfer any relationships from Symbol to Function
    MATCH (s)-[r]->(target)
    CREATE (f)-[newRel:RELATIONSHIP]->(target)
    SET newRel = properties(r)
    WITH s, f
    MATCH (source)-[r]->(s)
    CREATE (source)-[newRel:RELATIONSHIP]->(f)
    SET newRel = properties(r)
    WITH s
    DETACH DELETE s
    RETURN count(s) as migrated_functions
    """

    result = await neo4j_service.execute_cypher(function_migration, {})
    if result['status'] == 'success' and result['result']:
        count = result['result'][0].get('migrated_functions', 0)
        logger.info(f"✓ Migrated {count} Symbol nodes to Function nodes")

    # 3. Migrate Symbol nodes with type='method' to Method nodes
    method_migration = """
    MATCH (s:Symbol)
    WHERE s.type = 'method'
    CREATE (m:Method {
        name: s.name,
        qualified_name: s.qualified_name,
        file_path: s.file_path,
        start_line: s.start_line,
        end_line: s.end_line,
        language: s.language,
        docstring: s.docstring,
        project: s.project,
        indexed_at: s.indexed_at,
        class_name: CASE
            WHEN s.qualified_name CONTAINS '.'
            THEN split(s.qualified_name, '.')[0]
            ELSE null
        END
    })
    WITH s, m
    // Transfer any relationships from Symbol to Method
    MATCH (s)-[r]->(target)
    CREATE (m)-[newRel:RELATIONSHIP]->(target)
    SET newRel = properties(r)
    WITH s, m
    MATCH (source)-[r]->(s)
    CREATE (source)-[newRel:RELATIONSHIP]->(m)
    SET newRel = properties(r)
    WITH s
    DETACH DELETE s
    RETURN count(s) as migrated_methods
    """

    result = await neo4j_service.execute_cypher(method_migration, {})
    if result['status'] == 'success' and result['result']:
        count = result['result'][0].get('migrated_methods', 0)
        logger.info(f"✓ Migrated {count} Symbol nodes to Method nodes")

    # 4. Create indexes for better performance
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (c:Class) ON (c.name)",
        "CREATE INDEX IF NOT EXISTS FOR (f:Function) ON (f.name)",
        "CREATE INDEX IF NOT EXISTS FOR (m:Method) ON (m.name)",
        "CREATE INDEX IF NOT EXISTS FOR (c:Class) ON (c.project, c.name)",
        "CREATE INDEX IF NOT EXISTS FOR (f:Function) ON (f.project, f.name)",
        "CREATE INDEX IF NOT EXISTS FOR (m:Method) ON (m.project, m.name)"
    ]

    for index_query in indexes:
        await neo4j_service.execute_cypher(index_query, {})
    logger.info("✓ Created indexes for typed nodes")

    # 5. Verify migration success
    verification = """
    MATCH (s:Symbol)
    RETURN count(s) as remaining_symbols
    """

    result = await neo4j_service.execute_cypher(verification, {})
    if result['status'] == 'success' and result['result']:
        remaining = result['result'][0].get('remaining_symbols', 0)
        if remaining == 0:
            logger.info("✅ Migration complete! All Symbol nodes converted to typed nodes.")
        else:
            logger.warning(f"⚠️  {remaining} Symbol nodes remain (may be other types)")

    await neo4j_service.close()


if __name__ == "__main__":
    asyncio.run(migrate_symbols_to_typed_nodes())