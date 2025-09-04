# L9 Enhanced MCP Server - Neo4j Production Version
# Features: Neo4j GraphRAG + Tree-sitter + Qdrant (enhanced dependencies)
# Strategy: Full-featured build with Neo4j client-server architecture

FROM python:3.11-slim as base

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies 
COPY requirements/base.txt requirements/prod.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r prod.txt

# Copy organized source structure
COPY src/ ./src/

# Copy neural-tools configuration  
COPY neural-tools/config/ ./config/

# Copy source for backward compatibility paths
COPY src/graphrag/services/ ./neural-tools-src/services/
COPY src/mcp/ ./neural-tools-src/servers/

# Fix import paths for GraphRAG (hybrid_retriever needs to be accessible from servers/)
RUN mkdir -p /app/src/servers/services && \
    cp /app/src/graphrag/core/hybrid_retriever.py /app/src/servers/services/

# Create required directories
RUN mkdir -p /app/data /app/neo4j-workspace /app/models /var/log /var/run /workspace

# Set environment variables for organized structure
ENV PYTHONPATH=/app:/app/src:/app/src/servers:/app/neural-tools-src
ENV PYTHONUNBUFFERED=1
ENV L9_VERSION=neo4j-production-2025
ENV PROJECT_NAME=default

# Production configuration - use external embedding service
ENV USE_EXTERNAL_EMBEDDING=true
ENV EMBEDDING_SERVICE_HOST=neural-embeddings
ENV EMBEDDING_SERVICE_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; import fastmcp; import neo4j; sys.exit(0)"

# Run supervisord to manage both MCP server and indexer
CMD ["/usr/bin/supervisord", "-c", "/app/config/supervisord.conf"]