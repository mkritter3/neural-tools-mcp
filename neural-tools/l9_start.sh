#!/bin/bash
# L9 Simple Startup Script
# Starts Neural Tools with proper configuration

set -e  # Exit on error

echo "L9 Neural Tools Startup"
echo "======================"

# Check if we're in Docker or local
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    ENV_PREFIX=""
else
    echo "Running locally"
    ENV_PREFIX="export "
fi

# Set core environment variables (single source of truth)
${ENV_PREFIX}PROJECT_NAME="${PROJECT_NAME:-claude-l9-template}"
${ENV_PREFIX}PROJECT_DIR="${PROJECT_DIR:-/app/project}"

# Service endpoints
${ENV_PREFIX}NEO4J_HOST="${NEO4J_HOST:-localhost}"
${ENV_PREFIX}NEO4J_PORT="${NEO4J_PORT:-7688}"
${ENV_PREFIX}NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
${ENV_PREFIX}NEO4J_PASSWORD="${NEO4J_PASSWORD:-neural-l9-2025}"

${ENV_PREFIX}QDRANT_HOST="${QDRANT_HOST:-localhost}"
${ENV_PREFIX}QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6681}"
${ENV_PREFIX}QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6681}"

${ENV_PREFIX}EMBEDDING_SERVICE_HOST="${EMBEDDING_SERVICE_HOST:-localhost}"
${ENV_PREFIX}EMBEDDING_SERVICE_PORT="${EMBEDDING_SERVICE_PORT:-8081}"

# Python path
${ENV_PREFIX}PYTHONPATH="${PROJECT_DIR}/neural-tools/src:${PYTHONPATH}"
${ENV_PREFIX}PYTHONUNBUFFERED=1

# Validate configuration
echo "Validating configuration..."
python3 ${PROJECT_DIR}/neural-tools/config/l9_config.py

if [ $? -ne 0 ]; then
    echo "Configuration validation failed!"
    exit 1
fi

# Run health check
echo ""
echo "Running health check..."
python3 ${PROJECT_DIR}/neural-tools/src/servers/health_check.py

if [ $? -ne 0 ]; then
    echo "Health check failed!"
    echo "Services may not be running or configured correctly."
    exit 1
fi

echo ""
echo "✓ Neural Tools ready!"
echo ""
echo "To start MCP server:"
echo "  python3 ${PROJECT_DIR}/neural-tools/run_mcp_server.py"
echo ""
echo "To run indexer:"
echo "  python3 ${PROJECT_DIR}/neural-tools/src/servers/services/indexer_service.py"
