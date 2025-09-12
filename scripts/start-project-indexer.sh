#!/bin/bash
# Start single-project indexer following ADR-0028 auto-detection pattern
# This mounts the project as a subdirectory of /workspace for auto-detection

PROJECT_PATH="${PROJECT_PATH:-$(pwd)}"
PROJECT_NAME="$(basename "$PROJECT_PATH")"

echo "Starting indexer for project: $PROJECT_NAME"
echo "Project path: $PROJECT_PATH"

# Stop any existing indexer
docker stop l9-project-indexer 2>/dev/null || true
docker rm l9-project-indexer 2>/dev/null || true

# Start indexer with project mounted as subdirectory (ADR-0028)
docker run -d \
  --name l9-project-indexer \
  --network l9-graphrag-network \
  -p 48080:8080 \
  -v "$PROJECT_PATH:/workspace/$PROJECT_NAME:ro" \
  -e INITIAL_INDEX=true \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=graphrag-password \
  -e QDRANT_HOST=qdrant \
  -e QDRANT_PORT=6333 \
  -e REDIS_CACHE_HOST=redis-cache \
  -e REDIS_CACHE_PORT=6379 \
  -e REDIS_QUEUE_HOST=redis-queue \
  -e REDIS_QUEUE_PORT=6379 \
  -e EMBEDDING_SERVICE_HOST=neural-flow-nomic-v2-production \
  -e EMBEDDING_SERVICE_PORT=8000 \
  l9-neural-indexer:adr-0028

echo "Indexer started for $PROJECT_NAME (auto-detected from mount)"
echo "Monitoring logs..."
docker logs -f l9-project-indexer