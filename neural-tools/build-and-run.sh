#!/bin/bash
# Build and run Neural Tools with latest changes
# This ensures all code changes are included in the container

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Neural Tools Build & Run Script"
echo "=================================="

# Parse arguments
BUILD_ONLY=false
DEV_MODE=false
REBUILD=false
CLEAN_FIRST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --clean)
            CLEAN_FIRST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --build-only  Build images without starting services"
            echo "  --dev         Run in development mode with source mounted"
            echo "  --rebuild     Force rebuild even if image exists"
            echo "  --clean       Run Docker cleanup before building"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Load environment variables
if [ -f "config/.env" ]; then
    export $(cat config/.env | grep -v '^#' | xargs)
fi

# Set default project name if not set
export PROJECT_NAME=${PROJECT_NAME:-default}

echo "📦 Project: $PROJECT_NAME"

# Run cleanup if requested
if [ "$CLEAN_FIRST" = true ]; then
    echo ""
    echo "🧹 Running Docker cleanup..."
    if [ -f "./docker-cleanup.sh" ]; then
        ./docker-cleanup.sh
    else
        echo "  Warning: docker-cleanup.sh not found"
    fi
fi

# Build the Docker images
echo ""
echo "🏗️  Building Docker images..."

if [ "$REBUILD" = true ]; then
    echo "   Force rebuilding all images..."
    docker-compose -f config/docker-compose.neural-tools.yml build --no-cache
else
    docker-compose -f config/docker-compose.neural-tools.yml build
fi

# Build the embedding service image separately if needed
if ! docker images | grep -q "neural-flow:nomic-v2-production"; then
    echo ""
    echo "🧠 Building Nomic embedding service..."
    docker build -t neural-flow:nomic-v2-production -f Dockerfile.neural-embeddings .
fi

if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo "✅ Build complete! (--build-only flag set)"
    exit 0
fi

# Stop existing services
echo ""
echo "🛑 Stopping existing services..."
docker-compose -f config/docker-compose.neural-tools.yml down

# Start services
echo ""
if [ "$DEV_MODE" = true ]; then
    echo "🚀 Starting services in DEVELOPMENT mode (source code mounted)..."
    docker-compose \
        -f config/docker-compose.neural-tools.yml \
        -f config/docker-compose.neural-tools.dev.yml \
        up -d
else
    echo "🚀 Starting services in PRODUCTION mode..."
    docker-compose -f config/docker-compose.neural-tools.yml up -d
fi

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to initialize..."
sleep 5

# Check service health
echo ""
echo "🔍 Checking service status..."
docker-compose -f config/docker-compose.neural-tools.yml ps

# Show logs for troubleshooting
echo ""
echo "📋 Recent logs from neural-tools-server:"
docker-compose -f config/docker-compose.neural-tools.yml logs --tail=20 neural-tools-server

echo ""
echo "✅ Neural Tools is running!"
echo ""
echo "📊 Service URLs:"
echo "   Neo4j:  http://localhost:${NEO4J_DEBUG_HTTP_PORT:-7475}"
echo "   Qdrant: http://localhost:${QDRANT_DEBUG_PORT:-6681}/dashboard"
echo "   Nomic:  http://localhost:${NOMIC_DEBUG_PORT:-8081}/docs"
echo ""
echo "🔧 Management commands:"
echo "   View logs:    docker-compose -f config/docker-compose.neural-tools.yml logs -f"
echo "   Stop:         docker-compose -f config/docker-compose.neural-tools.yml down"
echo "   Restart:      docker-compose -f config/docker-compose.neural-tools.yml restart"
if [ "$DEV_MODE" = true ]; then
    echo ""
    echo "⚠️  Development mode: Source code is mounted - changes reflect immediately"
fi