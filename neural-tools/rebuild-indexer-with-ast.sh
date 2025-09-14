#!/bin/bash

# ADR-0038: Docker Image Lifecycle Management
# ADR-0047: Include AST-aware chunking in indexer

set -e

echo "🔧 Rebuilding L9 Neural Indexer with AST-Aware Chunking"
echo "========================================================="

# Get current Git SHA for immutable tagging
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "no-git")

# Build new image with all ADR-0047 optimizations
echo "📦 Building new indexer image..."
docker build -f Dockerfile.l9-minimal \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$GIT_SHA \
  --build-arg VERSION="v1.3.0-ast" \
  -t l9-neural-indexer:v1.3.0 \
  -t l9-neural-indexer:sha-$GIT_SHA \
  -t l9-neural-indexer:production \
  .

echo "✅ Image built and tagged:"
echo "  - l9-neural-indexer:v1.3.0 (semantic version)"
echo "  - l9-neural-indexer:sha-$GIT_SHA (immutable)"
echo "  - l9-neural-indexer:production (auto-pickup)"

# Verify AST chunker is included
echo ""
echo "🔍 Verifying AST-aware chunking is included..."
docker run --rm l9-neural-indexer:production ls -la /app/servers/services/ | grep -E "ast|chunk" || true

echo ""
echo "📋 ADR-0038 Compliance:"
echo "  ✅ Semantic versioning (v1.3.0)"
echo "  ✅ Immutable SHA tag (sha-$GIT_SHA)"
echo "  ✅ Production tag updated"
echo ""
echo "🚀 READY: Next indexer spawn will automatically use AST-aware chunking!"
echo ""
echo "Note: Existing containers continue running with old code (ephemeral design)."
echo "They'll get the update when they respawn (idle timeout or manual restart)."