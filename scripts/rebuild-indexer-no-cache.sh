#!/bin/bash
# Force rebuild indexer with no cache to ensure new entrypoint.py is copied

echo "Stopping existing container..."
docker stop l9-project-indexer 2>/dev/null || true
docker rm l9-project-indexer 2>/dev/null || true

echo "Removing old images..."
docker rmi l9-neural-indexer:adr-0028 2>/dev/null || true
docker rmi l9-neural-indexer:production 2>/dev/null || true

echo "Building with --no-cache to force fresh copy of entrypoint.py..."
docker build --no-cache -f docker/Dockerfile.indexer -t l9-neural-indexer:adr-0028 .

echo "Build complete! Image tagged as l9-neural-indexer:adr-0028"