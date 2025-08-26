# Neural Flow MCP Server - Multi-stage Docker Build
# Optimized for local development with dependency isolation

FROM python:3.13-slim as base

# System dependencies for neural processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM base as builder

# Install Python dependencies in isolated layer  
COPY requirements/ /tmp/requirements/
RUN pip install --user --no-cache-dir -r /tmp/requirements/requirements-base.txt

# Install sentence-transformers for Qodo-Embed support
RUN pip install --user --no-cache-dir sentence-transformers>=2.2.0

FROM base as runtime

# Copy only installed packages (smaller final image)
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy neural system and MCP server
COPY .claude/neural-system /app/neural-system
COPY .claude/mcp-server /app/mcp-server
WORKDIR /app

# Create volume mount points
RUN mkdir -p /app/data /app/models /app/project

# MCP server entry point
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import neural_system; print('healthy')" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]