#!/usr/bin/env python3
"""
Test Neo4j-only Architecture (ADR-0075)
Verifies new modular architecture works without Qdrant dependency
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "neural-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "neural-tools" / "src"))

# Test the new modular tools
from neural_mcp.shared.connection_pool import get_shared_neo4j_service
from neural_mcp.tools.neural_system_status import execute as status_execute
from neural_mcp.tools.semantic_search import execute as search_execute

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_neo4j_only_architecture():
    """Test that new modular architecture works with Neo4j-only"""

    project_name = "claude-l9-template"

    print("============================================================")
    print("🧪 Testing Neo4j-Only Modular Architecture (ADR-0075)")
    print("============================================================")

    try:
        # Test 1: Connection pooling
        print("\n1️⃣ Testing ADR-0075 connection pooling...")
        neo4j_service = await get_shared_neo4j_service(project_name)
        result = await neo4j_service.execute_cypher("MATCH (n) RETURN count(n) as total LIMIT 1", {})
        if result.get('status') == 'success':
            print(f"✅ Neo4j connection pool: {result['result'][0]['total']} nodes")
        else:
            print("❌ Neo4j connection failed")
            return False

        # Test 2: System status tool
        print("\n2️⃣ Testing neural_system_status tool...")
        status_result = await status_execute({})
        if status_result and len(status_result) > 0:
            print("✅ System status tool working")
        else:
            print("❌ System status tool failed")
            return False

        # Test 3: Semantic search tool (Neo4j-only)
        print("\n3️⃣ Testing semantic_search tool (Neo4j-only)...")
        search_result = await search_execute({"query": "test", "limit": 1})
        if search_result and len(search_result) > 0:
            print("✅ Semantic search tool working")
        else:
            print("❌ Semantic search tool failed")
            return False

        print("\n✅ ALL TESTS PASSED - Neo4j-only architecture is functional!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_neo4j_only_architecture())
    sys.exit(0 if success else 1)