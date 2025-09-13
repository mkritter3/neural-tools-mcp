#!/bin/bash

# L9 Neural GraphRAG - One-Click Setup Script
# Sets up the entire system for new users

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       L9 Neural GraphRAG MCP - Quick Setup              ║${NC}"
echo -e "${BLUE}║       Pattern-based metadata + Neural embeddings        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker Desktop first.${NC}"
    echo "   Visit: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites met${NC}"

# Step 1: Install Python dependencies
echo ""
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip3 install --quiet --upgrade pip
pip3 install --quiet \
    neo4j==5.22.0 \
    qdrant-client==1.15.1 \
    redis==5.0.1 \
    aiohttp \
    mcp \
    watchdog \
    pydantic

echo -e "${GREEN}✅ Python dependencies installed${NC}"

# Step 2: Build required Docker images (only if they don't exist)
echo ""
echo -e "${BLUE}🐳 Checking Docker images...${NC}"

# Check/Build Nomic embedding service
if docker images | grep -q "neural-flow.*nomic-v2-production"; then
    echo -e "${GREEN}✓ Nomic embedding service image already exists${NC}"
else
    if [ -f "neural-tools/Dockerfile.neural-embeddings" ]; then
        echo "Building Nomic embedding service..."
        # Build from neural-tools directory as context since it expects config/ and src/ there
        docker build -f neural-tools/Dockerfile.neural-embeddings -t neural-flow:nomic-v2-production neural-tools/ --quiet
    fi
fi

# Check/Build indexer service (with ADR-0029 multi-project isolation)
if docker images | grep -q "l9-neural-indexer.*production"; then
    echo -e "${GREEN}✓ Indexer service image already exists${NC}"
else
    if [ -f "docker/Dockerfile.indexer" ]; then
        echo "Building indexer service with multi-project isolation..."
        docker build -f docker/Dockerfile.indexer -t l9-neural-indexer:production . --quiet
    fi
fi

echo -e "${GREEN}✅ Docker images ready${NC}"

# Step 3: Start Docker services
echo ""
echo -e "${BLUE}🚀 Starting Docker services...${NC}"
docker-compose up -d

# Wait for services to be healthy
echo -e "${BLUE}⏳ Waiting for services to be healthy...${NC}"
sleep 5

# Check service health
SERVICES=("neo4j" "qdrant" "redis-cache" "redis-queue" "embeddings")
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    ALL_HEALTHY=true
    
    for service in "${SERVICES[@]}"; do
        if ! docker-compose ps | grep $service | grep -q "Up"; then
            ALL_HEALTHY=false
            break
        fi
    done
    
    if $ALL_HEALTHY; then
        echo -e "${GREEN}✅ All services healthy${NC}"
        break
    fi
    
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}❌ Services failed to start. Check docker-compose logs.${NC}"
    exit 1
fi

# Step 4: Configure MCP for Claude
echo ""
echo -e "${BLUE}🔧 Configuring MCP for Claude...${NC}"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    CLAUDE_CONFIG_DIR="$HOME/.config/claude"
else
    echo -e "${YELLOW}⚠️  Unknown OS. Please configure Claude manually.${NC}"
    CLAUDE_CONFIG_DIR=""
fi

if [ -n "$CLAUDE_CONFIG_DIR" ]; then
    # Create config directory if it doesn't exist
    mkdir -p "$CLAUDE_CONFIG_DIR"
    
    # Get current directory (where setup.sh is run from)
    PROJECT_DIR=$(pwd)
    
    # Create or update Claude config
    python3 - <<EOF
import json
import os

config_file = "$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
project_dir = "$PROJECT_DIR"

# Read existing config or create new
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    config = {"mcpServers": {}}

# Add/update neural-tools configuration
config['mcpServers']['neural-tools'] = {
    "command": "python3",
    "args": [f"{project_dir}/neural-tools/run_mcp_server.py"],
    "env": {
        "PYTHONPATH": f"{project_dir}/neural-tools/src",
        "AUTO_DETECT_PROJECT": "true",
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "EMBEDDING_SERVICE_HOST": "localhost",
        "EMBEDDING_SERVICE_PORT": "48000",
        "REDIS_CACHE_HOST": "localhost",
        "REDIS_CACHE_PORT": "46379",
        "REDIS_QUEUE_HOST": "localhost",
        "REDIS_QUEUE_PORT": "46380",
        "EMBED_DIM": "768",
        "PYTHONUNBUFFERED": "1"
    }
}

# Write config
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Claude configuration updated")
EOF
fi

# Step 5: Display success message
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  🎉 Setup Complete! 🎉                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 System Status:${NC}"
docker-compose ps --format "table {{.Name}}\t{{.Status}}"
echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "  1. Restart Claude Desktop to load the MCP"
echo "  2. Open any project in Claude"
echo "  3. Neural-tools will auto-detect and index it"
echo ""
echo -e "${BLUE}📚 Available MCP Tools:${NC}"
echo "  • semantic_code_search - Search by meaning"
echo "  • graphrag_hybrid_search - Graph + vector search"
echo "  • project_understanding - Get codebase overview"
echo "  • neural_system_status - Check system health"
echo ""
echo -e "${BLUE}⚙️  Management Commands:${NC}"
echo "  • Start services:  docker-compose up -d"
echo "  • Stop services:   docker-compose down"
echo "  • View logs:       docker-compose logs -f"
echo "  • Check status:    docker-compose ps"
echo ""
echo -e "${GREEN}Ready to use! Open Claude and start searching your code.${NC}"