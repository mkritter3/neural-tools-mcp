#!/bin/bash
# Neural Flow MCP Server - Docker Entry Point
# Handles MCP server startup with proper environment configuration

set -e

# Ensure data directories exist
mkdir -p /app/data/chroma /app/data/memory /app/models

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

# Start MCP server
echo "🚀 Starting MCP server on stdio transport..." >&2
cd /app/neural-system
exec python3 -m mcp_neural_server