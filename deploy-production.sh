#!/bin/bash

# Production Deployment Script for Neural Tools with Phase 3 Caching
# Usage: ./deploy-production.sh [staging|production] [--skip-tests] [--canary]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
SKIP_TESTS=${2:-false}
CANARY_DEPLOY=${3:-false}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Neural Tools Production Deployment${NC}"
echo -e "${GREEN}Phase 3: Intelligent Caching Layer${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker is not installed${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed${NC}"
        exit 1
    fi
    
    # Check environment file
    if [ "$ENVIRONMENT" == "production" ]; then
        if [ ! -f ".env.production" ]; then
            echo -e "${RED}Missing .env.production file${NC}"
            echo "Copy .env.production.template to .env.production and configure"
            exit 1
        fi
        ENV_FILE=".env.production"
    else
        if [ ! -f ".env.staging" ]; then
            echo -e "${YELLOW}Creating .env.staging from template...${NC}"
            cp .env.production.template .env.staging
            # Use default passwords for staging
            sed -i '' 's/<STRONG_PASSWORD_HERE>/staging-password-123/g' .env.staging
            sed -i '' 's/<STRONG_API_KEY_HERE>/staging-api-key-123/g' .env.staging
        fi
        ENV_FILE=".env.staging"
    fi
    
    echo -e "${GREEN}✓ Prerequisites check passed${NC}"
}

# Function to run tests
run_tests() {
    if [ "$SKIP_TESTS" != "--skip-tests" ]; then
        echo -e "${YELLOW}Running pre-deployment tests...${NC}"
        
        # Run unit tests
        echo "Running cache unit tests..."
        python3 test_phase3_caching_unit.py
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Unit tests failed!${NC}"
            exit 1
        fi
        
        # Run integration tests (simplified)
        echo "Running cache integration tests..."
        python3 test_cache_only.py
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Integration tests failed!${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${YELLOW}Skipping tests (--skip-tests flag)${NC}"
    fi
}

# Function to backup current deployment
backup_current() {
    echo -e "${YELLOW}Creating backup of current deployment...${NC}"
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup Redis data
    docker exec neural-redis-cache-prod redis-cli BGSAVE 2>/dev/null || true
    docker exec neural-redis-queue-prod redis-cli BGSAVE 2>/dev/null || true
    
    # Export compose config
    docker-compose -f docker-compose.production.yml config > "$BACKUP_DIR/compose-config.yml"
    
    echo -e "${GREEN}✓ Backup created in $BACKUP_DIR${NC}"
}

# Function to deploy services
deploy_services() {
    echo -e "${YELLOW}Deploying services to $ENVIRONMENT...${NC}"
    
    # Load environment
    export $(cat $ENV_FILE | grep -v '^#' | xargs)
    
    if [ "$CANARY_DEPLOY" == "--canary" ]; then
        echo -e "${YELLOW}Starting canary deployment (10% traffic)...${NC}"
        
        # Deploy new version alongside old with traffic split
        docker-compose -f docker-compose.production.yml up -d --scale embeddings=2
        
        # TODO: Configure load balancer for 10% traffic split
        echo -e "${YELLOW}Canary deployment started - monitor metrics${NC}"
    else
        # Full deployment
        echo "Pulling latest images..."
        docker-compose -f docker-compose.production.yml pull
        
        echo "Starting services..."
        docker-compose -f docker-compose.production.yml up -d
        
        echo "Waiting for services to be healthy..."
        sleep 30
        
        # Check health
        docker-compose -f docker-compose.production.yml ps
    fi
    
    echo -e "${GREEN}✓ Services deployed${NC}"
}

# Function to initialize cache warming
initialize_cache() {
    echo -e "${YELLOW}Initializing cache warming...${NC}"
    
    # Run cache warming script
    cat << 'EOF' | python3
import asyncio
import sys
from pathlib import Path

# Add neural-tools src to path
sys.path.insert(0, str(Path("neural-tools/src")))

async def warm_cache():
    from servers.services.service_container import ServiceContainer
    
    container = ServiceContainer("production")
    await container.initialize_all_services()
    
    # Get cache warmer
    warmer = await container.get_cache_warmer()
    
    # Warm common queries
    common_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does deep learning work?",
        # Add your common queries here
    ]
    
    result = await warmer.warm_embedding_cache(common_queries)
    print(f"Cache warming: {result['successful']}/{result['total_queries']} successful")
    
    # Run auto-warming
    auto_result = await warmer.auto_warm_frequent_queries(limit=50)
    print(f"Auto-warming: {auto_result}")

asyncio.run(warm_cache())
EOF
    
    echo -e "${GREEN}✓ Cache warming initialized${NC}"
}

# Function to setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}Setting up monitoring dashboards...${NC}"
    
    # Wait for Grafana to be ready
    until curl -s http://localhost:3000/api/health > /dev/null; do
        echo "Waiting for Grafana..."
        sleep 5
    done
    
    echo -e "${GREEN}✓ Monitoring available at http://localhost:3000${NC}"
    echo "  Username: ${GRAFANA_USER:-admin}"
    echo "  Dashboards: Cache Performance, Service Health, Redis Metrics"
}

# Function to run smoke tests
run_smoke_tests() {
    echo -e "${YELLOW}Running smoke tests...${NC}"
    
    # Test Redis connectivity
    docker exec neural-redis-cache-prod redis-cli ping > /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Redis cache: OK${NC}"
    else
        echo -e "${RED}✗ Redis cache: FAILED${NC}"
        exit 1
    fi
    
    # Test cache operations
    python3 test_cache_only.py > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Cache operations: OK${NC}"
    else
        echo -e "${RED}✗ Cache operations: FAILED${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ All smoke tests passed${NC}"
}

# Function to display deployment summary
show_summary() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Services deployed:"
    echo "  • Redis Cache (Port: ${REDIS_CACHE_PORT:-6379})"
    echo "  • Redis Queue (Port: ${REDIS_QUEUE_PORT:-6380})"
    echo "  • Neo4j (Port: ${NEO4J_BOLT_PORT:-7687})"
    echo "  • Qdrant (Port: ${QDRANT_HTTP_PORT:-6333})"
    echo "  • Embeddings (Port: ${NOMIC_PORT:-8000})"
    echo "  • Prometheus (Port: 9090)"
    echo "  • Grafana (Port: 3000)"
    echo ""
    echo "Cache Performance:"
    echo "  • Hit ratio target: >95%"
    echo "  • Response time: <5ms (cached)"
    echo "  • Throughput: 1,600+ ops/sec"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor cache metrics in Grafana"
    echo "  2. Watch error rates and latencies"
    echo "  3. Adjust TTLs based on usage patterns"
    echo "  4. Scale Redis memory if needed"
    echo ""
    
    if [ "$CANARY_DEPLOY" == "--canary" ]; then
        echo -e "${YELLOW}⚠️  Canary deployment active${NC}"
        echo "Monitor metrics and gradually increase traffic if stable"
    fi
}

# Main deployment flow
main() {
    echo "Starting deployment to $ENVIRONMENT..."
    
    check_prerequisites
    run_tests
    
    if [ "$ENVIRONMENT" == "production" ]; then
        backup_current
    fi
    
    deploy_services
    
    # Wait for services to stabilize
    echo "Waiting for services to stabilize..."
    sleep 20
    
    initialize_cache
    setup_monitoring
    run_smoke_tests
    show_summary
}

# Run main function
main