# Shared Model Server - Multi-stage Docker Build
# Serves Qodo-Embed-1.5B and BM25 models to all projects
# Saves 90%+ memory by centralizing models

# Stage 1: Model pre-loading and caching
FROM python:3.12-slim as model-cache

# Install sentence transformers and pre-cache Qodo model
RUN pip install --no-cache-dir \
    sentence-transformers>=3.0.0 \
    scikit-learn>=1.3.0 \
    numpy>=1.24.0

# Pre-download Qodo-Embed model to cache
RUN python3 -c "from sentence_transformers import SentenceTransformer; print('📥 Pre-caching Qodo-Embed-1.5B model...'); model = SentenceTransformer('Qodo/Qodo-Embed-1-1.5B'); print('✅ Qodo model cached')"

# Stage 2: Dependencies builder
FROM python:3.12-slim as model-builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    curl \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies for model server
RUN pip install --user --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    sentence-transformers>=3.0.0 \
    scikit-learn>=1.3.0 \
    numpy>=1.24.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.5.0

# Copy cached models from model-cache stage
COPY --from=model-cache /root/.cache/huggingface /root/.cache/huggingface

# Stage 3: Model Server runtime (optimized)
FROM python:3.12-slim as model-server

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy dependencies and cached models
COPY --from=model-builder /root/.local /root/.local
COPY --from=model-builder /root/.cache/huggingface /root/.cache/huggingface

# Ensure Python can find installed packages
ENV PATH=/root/.local/bin:$PATH

# Model server optimized environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    MODEL_CACHE_DIR=/app/models

# Create app directory
WORKDIR /app

# Copy shared model server
COPY shared-model-server.py /app/shared-model-server.py

# Create models cache directory
RUN mkdir -p /app/models

# Create entrypoint script
RUN echo '#!/bin/bash' > /app/docker-entrypoint-model-server.sh && \
    echo 'set -e' >> /app/docker-entrypoint-model-server.sh && \
    echo '' >> /app/docker-entrypoint-model-server.sh && \
    echo 'echo "🤖 Starting Shared Model Server..."' >> /app/docker-entrypoint-model-server.sh && \
    echo 'echo "📦 Models: Qodo-Embed-1.5B + BM25/TF-IDF"' >> /app/docker-entrypoint-model-server.sh && \
    echo 'echo "🌐 Serving on: http://0.0.0.0:8080"' >> /app/docker-entrypoint-model-server.sh && \
    echo 'echo "💾 Memory optimization: 90%+ savings vs per-project models"' >> /app/docker-entrypoint-model-server.sh && \
    echo '' >> /app/docker-entrypoint-model-server.sh && \
    echo '# Start model server' >> /app/docker-entrypoint-model-server.sh && \
    echo 'exec python3 /app/shared-model-server.py' >> /app/docker-entrypoint-model-server.sh && \
    chmod +x /app/docker-entrypoint-model-server.sh

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Container metadata
LABEL maintainer="Neural Memory Team" \
      description="Shared Model Server for Neural Memory System" \
      version="1.0.0" \
      models="Qodo-Embed-1.5B,BM25" \
      port="8080"

# Expose model server port
EXPOSE 8080

# Use the model-server entrypoint
ENTRYPOINT ["/app/docker-entrypoint-model-server.sh"]