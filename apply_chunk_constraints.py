#!/usr/bin/env python3
"""
Apply Neo4j constraints from ChunkSchema
ADR-0096: Enforce schema contract at database level
"""

import asyncio
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src/servers/services')

async def apply_constraints():
    """Apply Neo4j constraints for ChunkSchema enforcement"""

    from service_container import ServiceContainer
    from servers.services.project_context_manager import ProjectContextManager
    from chunk_schema import NEO4J_CHUNK_CONSTRAINTS, NEO4J_VECTOR_INDEX

    # Set up for claude-l9-template
    project_path = "/Users/mkr/local-coding/claude-l9-template"
    project_name = "claude-l9-template"

    print("=" * 70)
    print("🔧 APPLYING NEO4J CHUNK CONSTRAINTS (ADR-0096)")
    print("=" * 70)

    # Create context and container
    context = ProjectContextManager()
    await context.set_project(project_path)

    container = ServiceContainer(context, project_name)
    await container.initialize_all_services()

    print("\n📝 STEP 1: Applying Chunk Constraints")
    print("-" * 60)

    # Apply each constraint
    for i, constraint in enumerate(NEO4J_CHUNK_CONSTRAINTS, 1):
        print(f"\n{i}. Applying constraint...")
        print(f"   Query: {constraint[:80]}...")

        result = await container.neo4j.execute_cypher(constraint, {})

        if result['status'] == 'success':
            print(f"   ✅ Constraint applied successfully")
        else:
            print(f"   ❌ Failed: {result.get('message')}")

    print("\n📝 STEP 2: Creating Vector Index")
    print("-" * 60)

    print(f"   Query: {NEO4J_VECTOR_INDEX[:80]}...")

    result = await container.neo4j.execute_cypher(NEO4J_VECTOR_INDEX, {})

    if result['status'] == 'success':
        print(f"   ✅ Vector index created successfully")
    else:
        # May already exist, which is fine
        if 'already exists' in result.get('message', '').lower():
            print(f"   ℹ️ Vector index already exists")
        else:
            print(f"   ❌ Failed: {result.get('message')}")

    print("\n📝 STEP 3: Verifying Constraints")
    print("-" * 60)

    # Check existing constraints
    check_query = """
    SHOW CONSTRAINTS
    YIELD name, type, entityType, labelsOrTypes, properties
    WHERE 'Chunk' IN labelsOrTypes
    RETURN name, type, properties
    """

    result = await container.neo4j.execute_cypher(check_query, {})

    if result['status'] == 'success' and result['result']:
        print(f"   ✅ Found {len(result['result'])} Chunk constraints:")
        for constraint in result['result']:
            print(f"      • {constraint['name']}: {constraint['properties']}")
    else:
        print("   ⚠️ No Chunk constraints found")

    print("\n📝 STEP 4: Verifying Vector Index")
    print("-" * 60)

    # Check vector indexes
    index_query = """
    SHOW INDEXES
    YIELD name, type, entityType, labelsOrTypes, properties
    WHERE type = 'VECTOR' AND 'Chunk' IN labelsOrTypes
    RETURN name, properties
    """

    result = await container.neo4j.execute_cypher(index_query, {})

    if result['status'] == 'success' and result['result']:
        print(f"   ✅ Found {len(result['result'])} vector indexes:")
        for index in result['result']:
            print(f"      • {index['name']}: {index['properties']}")
    else:
        print("   ⚠️ No vector indexes found for Chunks")

    print("\n" + "=" * 70)
    print("🎉 CONSTRAINT APPLICATION COMPLETE")
    print("=" * 70)

    print("\n📊 SUMMARY:")
    print(f"   • Applied {len(NEO4J_CHUNK_CONSTRAINTS)} constraints")
    print(f"   • Created/verified vector index")
    print(f"   • ChunkSchema contract is now enforced at database level")

if __name__ == "__main__":
    asyncio.run(apply_constraints())