# Neural Flow MCP Server - Optimized Multi-stage Docker Build
# L9-grade performance optimization for vibe coder workflows

# Stage 1: Model cache layer (shared across rebuilds)
FROM python:3.13-slim as model-cache

# Pre-download and cache models to improve container startup time
RUN pip install --no-cache-dir sentence-transformers>=2.2.0
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qodo/Qodo-Embed-1-1.5B')"

# Stage 2: Base system with optimizations
FROM python:3.13-slim as base

# Install system dependencies with optimization flags
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    git \
    curl \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python optimizations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Stage 3: Dependencies builder
FROM base as builder

# Install Python dependencies with optimizations
COPY requirements/ /tmp/requirements/
RUN pip install --user --no-cache-dir --compile -r /tmp/requirements/requirements-base.txt

# Copy pre-cached models from model-cache stage
COPY --from=model-cache /root/.cache/huggingface /root/.cache/huggingface

# Install additional ML dependencies optimized for CPU
RUN pip install --user --no-cache-dir \
    sentence-transformers>=2.2.0 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/

# Stage 4: Optimized runtime
FROM base as runtime

# Copy dependencies and models
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Set optimized PATH and Python settings
ENV PATH=/root/.local/bin:$PATH \
    TRANSFORMERS_CACHE=/app/models/.cache/huggingface \
    HF_HOME=/app/models/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Copy application code
COPY .claude/neural-system /app/neural-system
WORKDIR /app

# Create optimized volume mount points with proper permissions
RUN mkdir -p /app/data /app/models /app/project \
    /app/models/.cache/huggingface \
    /app/data/benchmarks \
    /app/data/chroma_db \
    && chmod -R 755 /app

# Copy entrypoint script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Enhanced health check with performance validation
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 -c "import sys, time; start = time.time(); \
    try: \
        from neural_embeddings import CodeSpecificEmbedder; \
        embedder = CodeSpecificEmbedder(); \
        load_time = time.time() - start; \
        print(f'healthy - load_time: {load_time:.2f}s'); \
        sys.exit(0 if load_time < 15.0 else 1); \
    except Exception as e: \
        print(f'unhealthy: {e}'); \
        sys.exit(1)"

# Container performance and resource limits
LABEL maintainer="Neural Flow L9 Team" \
      version="2.0.0-optimized" \
      description="High-performance Neural Flow MCP Server"

ENTRYPOINT ["/app/docker-entrypoint.sh"]