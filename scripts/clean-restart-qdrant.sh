#!/bin/bash
# Clean restart of Qdrant with new collection naming
# WARNING: This will DELETE all existing data!

echo "⚠️  WARNING: This will DELETE all Qdrant data and restart with clean collections!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

echo "Step 1: Stopping Qdrant container..."
docker-compose stop qdrant

echo "Step 2: Removing Qdrant volume (deleting all data)..."
docker volume rm claude-l9-template_qdrant_data || true

echo "Step 3: Pulling latest Qdrant image..."
docker pull qdrant/qdrant:latest

echo "Step 4: Starting Qdrant with latest version..."
docker-compose up -d qdrant

echo "Step 5: Waiting for Qdrant to be healthy..."
sleep 5

echo "Step 6: Verifying Qdrant is running..."
curl -s http://localhost:46333/collections | python3 -m json.tool

echo "✅ Qdrant restarted with clean state and latest version"
echo ""
echo "Next steps:"
echo "1. Trigger reindexing to create collections with new naming"
echo "2. Test semantic search functionality"