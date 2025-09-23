#!/usr/bin/env python3
"""
Simple test of graph-aware MCP functionality
"""

import asyncio
import sys
import json
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from neural_mcp.neural_server_stdio import enrich_with_graph_context
from servers.services.neo4j_service import Neo4jService

async def test_graph_enrichment():
    """Test graph context enrichment with mock search results"""
    print("🧪 Testing Graph Context Enrichment")
    print("="*50)

    # Initialize Neo4j
    neo4j = Neo4jService("claude-l9-template")
    await neo4j.initialize()

    # Create mock search results based on actual chunks
    mock_search_results = [
        {
            "node": {
                "chunk_id": "service_container.py:chunk:0",
                "file_path": "neural-tools/src/servers/services/service_container.py",
                "path": "neural-tools/src/servers/services/service_container.py",
                "text": "Sample chunk content"
            },
            "similarity_score": 0.85
        }
    ]

    print(f"🔍 Testing with mock chunk: {mock_search_results[0]['node']['chunk_id']}")

    # Test graph enrichment
    graph_context = await enrich_with_graph_context(neo4j, mock_search_results, "claude-l9-template")

    print(f"📊 Graph context results: {len(graph_context)} enriched chunks")

    for chunk_id, context in graph_context.items():
        print(f"\n🕸️  Context for {chunk_id}:")
        print(f"   Functions: {context.get('related_functions', [])[:3]}")
        print(f"   Modules: {context.get('imported_modules', [])[:3]}")
        print(f"   Files: {context.get('related_files', [])[:3]}")

    print("\n✅ Graph context enrichment test completed!")

if __name__ == "__main__":
    asyncio.run(test_graph_enrichment())