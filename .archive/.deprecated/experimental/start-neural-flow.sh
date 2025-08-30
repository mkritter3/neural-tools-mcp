#!/bin/bash
# Neural Flow L9 - Portable Container Startup Script
# This script ensures anyone can run the system without managing dependencies

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_NAME="${PROJECT_NAME:-claude-l9-template}"
CONTAINER_NAME="neural-flow-${PROJECT_NAME}"
IMAGE_NAME="neural-flow:l9-production"

echo -e "${GREEN}🚀 Neural Flow L9 Container Manager${NC}"
echo "================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker daemon is not running. Please start Docker.${NC}"
    exit 1
fi

# Function to build image if needed
build_image() {
    echo -e "${YELLOW}🔨 Building Neural Flow L9 Docker image...${NC}"
    echo "This may take 5-10 minutes on first run (downloading dependencies)"
    
    if docker build -f Dockerfile.l9 -t ${IMAGE_NAME} . ; then
        echo -e "${GREEN}✅ Image built successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Build failed. Check Dockerfile.l9${NC}"
        return 1
    fi
}

# Check if image exists
if ! docker images | grep -q "${IMAGE_NAME}"; then
    echo -e "${YELLOW}⚠️  Neural Flow L9 image not found.${NC}"
    build_image || exit 1
else
    echo -e "${GREEN}✅ Neural Flow L9 image found${NC}"
fi

# Stop any existing container with same name
if docker ps -a | grep -q "${CONTAINER_NAME}"; then
    echo -e "${YELLOW}🛑 Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME} > /dev/null 2>&1 || true
    docker rm ${CONTAINER_NAME} > /dev/null 2>&1 || true
fi

# Start the container
echo -e "${GREEN}🚀 Starting Neural Flow L9 container...${NC}"
docker run -d \
    --name ${CONTAINER_NAME} \
    -v $(pwd)/.claude:/app/data \
    -e PROJECT_NAME=${PROJECT_NAME} \
    -e NEURAL_L9_MODE=1 \
    -e USE_SINGLE_QODO_MODEL=1 \
    -e ENABLE_AUTO_SAFETY=1 \
    -e L9_PROTECTION_LEVEL=maximum \
    --restart unless-stopped \
    ${IMAGE_NAME}

# Wait for container to be healthy
echo -n "⏳ Waiting for container to be ready"
for i in {1..30}; do
    if docker exec ${CONTAINER_NAME} python3 -c "import mcp; import chromadb; print('OK')" &> /dev/null; then
        echo ""
        echo -e "${GREEN}✅ Container is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Check final status
if docker ps | grep -q ${CONTAINER_NAME}; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Neural Flow L9 is running!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Container: ${CONTAINER_NAME}"
    echo "MCP Server: Ready for connections"
    echo ""
    echo "To view logs: docker logs -f ${CONTAINER_NAME}"
    echo "To stop: docker stop ${CONTAINER_NAME}"
    echo "To restart: ./start-neural-flow.sh"
else
    echo -e "${RED}❌ Container failed to start${NC}"
    echo "Check logs: docker logs ${CONTAINER_NAME}"
    exit 1
fi