#!/bin/bash
# L9 MCP Docker Setup Script

set -e

echo "🚀 L9 MCP Docker Setup"
echo "====================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Set project name
PROJECT_NAME="${1:-default}"
echo "📦 Project: $PROJECT_NAME"

# Build MCP server image
echo "🔨 Building MCP server image..."
docker build -f docker/Dockerfile.mcp -t l9-mcp-server:latest .

# Start services
echo "🎯 Starting MCP server and Qdrant..."
PROJECT_NAME=$PROJECT_NAME docker compose -f docker/docker-compose.mcp.yml up -d

# Wait for health checks
echo "⏳ Waiting for services to be healthy..."
sleep 5

# Check status
echo "✅ Checking service status..."
docker compose -f docker/docker-compose.mcp.yml ps

# Test MCP connection
echo "🧪 Testing MCP connection..."
docker compose -f docker/docker-compose.mcp.yml run --rm mcp-server python3 -c "
import asyncio
from qdrant_client import QdrantClient

async def test():
    client = QdrantClient(host='qdrant', port=6334, prefer_grpc=True)
    collections = client.get_collections()
    print(f'✅ Connected to Qdrant. Collections: {len(collections.collections)}')

asyncio.run(test())
"

echo ""
echo "✨ MCP Docker setup complete!"
echo ""
echo "To use with Claude Code:"
echo "  1. The .mcp.json file is already configured"
echo "  2. Restart Claude Code to pick up the new MCP server"
echo "  3. Use the MCP tools: memory_store, memory_search, code_index, etc."
echo ""
echo "To test manually:"
echo "  docker compose -f docker/docker-compose.mcp.yml run --rm mcp-server"
echo ""