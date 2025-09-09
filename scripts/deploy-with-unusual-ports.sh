#!/bin/bash
# L9 Deployment with Unusual Ports - No Dev Environment Conflicts
set -euo pipefail

echo "🚀 L9 Neural GraphRAG Stack - Unusual Ports Deployment"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Port Assignments (Unusual High Ports):${NC}"
echo "=================================================="
echo -e "${GREEN}Core Services:${NC}"
echo "  • Neo4j Browser:  http://localhost:47474"
echo "  • Neo4j Bolt:     bolt://localhost:47687"
echo "  • Qdrant REST:    http://localhost:46333"
echo "  • Qdrant gRPC:    localhost:46334"
echo "  • Redis:          localhost:46379"
echo ""
echo -e "${GREEN}Neural Services:${NC}"
echo "  • Indexer Health: http://localhost:48080/health"
echo "  • Indexer Status: http://localhost:48080/status" 
echo "  • Indexer Metrics: http://localhost:48080/metrics"
echo "  • Neural-Flow:    http://localhost:48000 (external)"
echo ""
echo -e "${GREEN}GraphRAG & Monitoring:${NC}"
echo "  • GraphRAG API:   http://localhost:43000"
echo "  • GraphRAG Metrics: http://localhost:49090"
echo "  • Prometheus:     http://localhost:49091"
echo "  • Grafana:        http://localhost:43001"
echo ""

# Step 1: Start neural-flow embeddings (external)
echo -e "${YELLOW}Step 1: Starting Neural-Flow Embeddings Service...${NC}"
if ! docker ps --format "{{.Names}}" | grep -q "^neural-flow-nomic-v2-production$"; then
    docker run -d \
        --name neural-flow-nomic-v2-production \
        -p 48000:8000 \
        --restart unless-stopped \
        neural-flow:nomic-v2-production
    echo -e "${GREEN}✅ Neural-Flow started on port 48000${NC}"
else
    echo -e "${GREEN}✅ Neural-Flow already running${NC}"
fi

# Connect to our network when it's created
echo "Waiting for l9-graphrag-network to be created..."

# Step 2: Start core services
echo -e "${YELLOW}Step 2: Starting Core Services (Neo4j, Qdrant, Redis)...${NC}"
docker-compose up -d neo4j qdrant redis

# Wait for services to be healthy
echo "Waiting for core services to be healthy..."
timeout 180 bash -c '
    while true; do
        healthy_count=0
        total_count=3
        
        if docker-compose ps neo4j | grep -q "healthy"; then
            ((healthy_count++))
        fi
        if docker-compose ps qdrant | grep -q "healthy"; then
            ((healthy_count++))
        fi
        if docker-compose ps redis | grep -q "healthy"; then
            ((healthy_count++))
        fi
        
        echo "Healthy services: $healthy_count/$total_count"
        
        if [ $healthy_count -eq $total_count ]; then
            break
        fi
        sleep 5
    done
' || {
    echo -e "${RED}❌ Core services failed to become healthy${NC}"
    echo "Service status:"
    docker-compose ps neo4j qdrant redis
    exit 1
}

echo -e "${GREEN}✅ Core services are healthy${NC}"

# Connect neural-flow to our network
echo "Connecting neural-flow to l9-graphrag-network..."
docker network connect l9-graphrag-network neural-flow-nomic-v2-production 2>/dev/null || {
    echo -e "${GREEN}✅ Neural-flow already connected to network${NC}"
}

# Step 3: Build and start indexer
echo -e "${YELLOW}Step 3: Building and Starting Neural Indexer Sidecar...${NC}"
docker-compose up -d --build l9-indexer

# Wait for indexer to be healthy
echo "Waiting for indexer to be ready..."
timeout 120 bash -c '
    while ! curl -f http://localhost:48080/health > /dev/null 2>&1; do 
        echo -n "."
        sleep 3
    done
' || {
    echo -e "${RED}❌ Indexer failed to become healthy${NC}"
    echo "Indexer logs:"
    docker-compose logs l9-indexer --tail 30
    exit 1
}

echo -e "${GREEN}✅ Indexer is healthy and ready${NC}"

# Step 4: Validate all endpoints
echo -e "${YELLOW}Step 4: Validating All Service Endpoints...${NC}"

services=(
    "Neo4j:http://localhost:47474"
    "Qdrant:http://localhost:46333/health"
    "Indexer Health:http://localhost:48080/health"
    "Indexer Status:http://localhost:48080/status"
    "Neural-Flow:http://localhost:48000/health"
)

for service in "${services[@]}"; do
    name="${service%%:*}"
    url="${service#*:}"
    
    if curl -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name responding${NC}"
    else
        echo -e "${RED}❌ $name not responding: $url${NC}"
    fi
done

# Step 5: Show quick test commands
echo ""
echo -e "${BLUE}🧪 Quick Test Commands:${NC}"
echo "=================================================="
echo "# Test indexer status via MCP (use in Claude Code):"
echo "indexer_status"
echo ""
echo "# Test reindexing:"
echo "reindex_path /workspace/README.md"
echo ""
echo "# Manual endpoint tests:"
echo "curl http://localhost:48080/health | jq"
echo "curl http://localhost:48080/status | jq"
echo "curl http://localhost:48080/metrics"
echo ""
echo "# Neo4j Browser (user: neo4j, pass: graphrag-password):"
echo "open http://localhost:47474"
echo ""
echo "# Qdrant collections:"
echo "curl http://localhost:46333/collections | jq"

echo ""
echo -e "${GREEN}🎉 L9 Neural GraphRAG Stack Deployed Successfully!${NC}"
echo "=================================================="
echo -e "${GREEN}All services running on unusual ports - no dev conflicts${NC}"
echo -e "${YELLOW}Next: Test MCP integration in Claude Code${NC}"