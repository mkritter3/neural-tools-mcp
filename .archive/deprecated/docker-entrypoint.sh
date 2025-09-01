#!/bin/bash
# Neural Flow MCP Server - Docker Entry Point
# L9-grade initialization with performance optimizations

set -e

# Performance optimizations
echo "Setting performance optimizations..." >&2
export MALLOC_MMAP_THRESHOLD_=131072
export MALLOC_TRIM_THRESHOLD_=131072
export MALLOC_TOP_PAD_=131072
export PYTHONOPTIMIZE=2

# Ensure data directories exist
mkdir -p /app/data/chroma /app/data/memory /app/data/benchmarks /app/models

# Ensure model cache is available
if [ ! -d "/app/models/.cache/huggingface" ]; then
    mkdir -p /app/models/.cache/huggingface
    # Link pre-cached models if available
    if [ -d "/root/.cache/huggingface" ]; then
        ln -sf /root/.cache/huggingface/* /app/models/.cache/huggingface/ 2>/dev/null || true
    fi
fi

# Set up Python path
export PYTHONPATH="/app/neural-system:$PYTHONPATH"

# Configure logging
export PYTHONUNBUFFERED=1

# MCP server startup message
echo "🔮 Neural Flow MCP Server starting..." >&2
echo "   Project: ${PROJECT_NAME:-default}" >&2
echo "   Qodo-Embed: ${USE_QODO_EMBED:-false}" >&2
echo "   A/B Testing: ${ENABLE_AB_TESTING:-false}" >&2

# Validate neural system can load
python3 -c "
import sys
sys.path.insert(0, '/app/neural-system')
try:
    from neural_embeddings import get_neural_system
    system = get_neural_system()
    print('✅ Neural system initialized successfully', file=sys.stderr)
except Exception as e:
    print(f'❌ Neural system initialization failed: {e}', file=sys.stderr)
    sys.exit(1)
"

# Check if we should run benchmarks on startup
if [ "$RUN_BENCHMARKS_ON_START" = "true" ]; then
    echo "Running initial performance benchmarks..." >&2
    python3 -O /app/neural-system/performance_benchmarks.py --data-dir /app/data --report-only || true
fi

# Start MCP server with optimizations
echo "🚀 Starting MCP server on stdio transport..." >&2
cd /app/neural-system
exec python3 -O -u mcp_neural_server.py