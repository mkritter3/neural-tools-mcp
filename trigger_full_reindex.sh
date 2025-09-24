#!/bin/bash
# Script to trigger full reindexing with the fixed chunking

echo "🔄 Triggering Full Reindex with Fixed Chunking"
echo "================================================"

# Option 1: Clear all chunks and trigger reindex via Docker
echo "Option 1: Docker restart method"
echo "--------------------------------"
echo "Run these commands:"
echo ""
echo "# 1. Clear old chunks (optional but recommended)"
echo "docker exec <neo4j-container> cypher-shell -u neo4j -p graphrag-password \"MATCH (c:Chunk) DETACH DELETE c\""
echo ""
echo "# 2. Restart indexer to trigger full reindex"
echo "docker ps | grep indexer"
echo "docker restart <indexer-container-id>"
echo ""

# Option 2: Direct indexing for specific directories
echo "Option 2: Direct reindex via MCP (if indexer is running)"
echo "---------------------------------------------------------"
echo "Use Claude to run:"
echo ""
echo "mcp__neural-tools__project_operations("
echo "  operation='reindex',"
echo "  path='/',"
echo "  recursive=true"
echo ")"
echo ""

# Option 3: Manual Python script
echo "Option 3: Manual Python reindex"
echo "--------------------------------"
cat << 'EOF' > manual_reindex.py
#!/usr/bin/env python3
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, 'neural-tools/src')
sys.path.insert(0, 'neural-tools/src/servers/services')

async def manual_reindex():
    from indexer_service import IncrementalIndexer
    from service_container import ServiceContainer
    from servers.services.project_context_manager import ProjectContextManager

    project_path = os.getcwd()
    project_name = "claude-l9-template"

    print(f"🔄 Starting full reindex for {project_path}")

    # Create indexer with fixed chunking
    context = ProjectContextManager()
    container = ServiceContainer(context, project_name)
    await container.initialize_all_services()

    indexer = IncrementalIndexer(project_path, project_name, container)
    indexer.pending_queue = asyncio.Queue(maxsize=1000)

    # Initialize and run
    await indexer.initialize_services()

    print("📂 Starting initial index of all files...")
    await indexer.initial_index()

    print("✅ Reindexing complete!")

    # Show results
    result = await container.neo4j.execute_cypher("""
        MATCH (f:File)
        OPTIONAL MATCH (f)-[:HAS_CHUNK]->(c:Chunk)
        RETURN
            count(DISTINCT f) as files,
            count(DISTINCT c) as chunks,
            count(DISTINCT CASE WHEN c.embedding IS NOT NULL THEN c END) as chunks_with_embeddings
    """, {})

    if result['status'] == 'success' and result['result']:
        data = result['result'][0]
        print(f"\n📊 Final Statistics:")
        print(f"  Files: {data['files']}")
        print(f"  Chunks: {data['chunks']}")
        print(f"  With embeddings: {data['chunks_with_embeddings']}")

if __name__ == "__main__":
    asyncio.run(manual_reindex())
EOF

echo ""
echo "To run manual reindex:"
echo "python3 manual_reindex.py"
echo ""
echo "⏱️  Estimated time: 5-10 minutes for full codebase"
echo "📈 Expected result: ~10,000+ chunks with embeddings"