#!/bin/bash
# Neural Docker Management Script
# Manages unified neural-flow and Qdrant containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default project name if not provided
PROJECT_NAME=${PROJECT_NAME:-$(basename "$PWD")}

echo -e "${GREEN}🚀 Neural Docker Manager${NC}"
echo -e "Project: ${YELLOW}$PROJECT_NAME${NC}"
echo ""

case "$1" in
    start|up)
        echo -e "${GREEN}Starting containers...${NC}"
        docker-compose -f docker-compose.unified.yml up -d
        echo -e "${GREEN}✅ Containers started${NC}"
        echo ""
        echo "Services:"
        echo "  • Neural Flow: neural-flow-$PROJECT_NAME"
        echo "  • Qdrant: qdrant-$PROJECT_NAME (ports ${QDRANT_REST_PORT:-6678}/${QDRANT_GRPC_PORT:-6679})"
        ;;
        
    stop|down)
        echo -e "${YELLOW}Stopping containers...${NC}"
        docker-compose -f docker-compose.unified.yml down
        echo -e "${GREEN}✅ Containers stopped${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}Restarting containers...${NC}"
        docker-compose -f docker-compose.unified.yml restart
        echo -e "${GREEN}✅ Containers restarted${NC}"
        ;;
        
    logs)
        service=${2:-}
        if [ -z "$service" ]; then
            docker-compose -f docker-compose.unified.yml logs -f
        else
            docker-compose -f docker-compose.unified.yml logs -f $service
        fi
        ;;
        
    status|ps)
        echo -e "${GREEN}Container Status:${NC}"
        docker-compose -f docker-compose.unified.yml ps
        ;;
        
    clean)
        echo -e "${RED}⚠️  Warning: This will delete all project data!${NC}"
        echo -n "Are you sure? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            docker-compose -f docker-compose.unified.yml down -v
            rm -rf .docker/qdrant/$PROJECT_NAME/
            echo -e "${GREEN}✅ Project data cleaned${NC}"
        else
            echo "Cancelled"
        fi
        ;;
        
    shell)
        service=${2:-neural-flow}
        echo -e "${GREEN}Opening shell in $service...${NC}"
        docker-compose -f docker-compose.unified.yml exec $service /bin/bash
        ;;
        
    test)
        echo -e "${GREEN}Running tests...${NC}"
        python3 .claude/neural-system/config_manager.py
        echo ""
        python3 .claude/neural-system/project_isolation.py
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|clean|shell|test}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all containers"
        echo "  stop     - Stop all containers"
        echo "  restart  - Restart all containers"
        echo "  logs     - View logs (optional: specify service)"
        echo "  status   - Show container status"
        echo "  clean    - Remove containers and data (careful!)"
        echo "  shell    - Open shell in container"
        echo "  test     - Run configuration tests"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs qdrant"
        echo "  $0 shell neural-flow"
        exit 1
        ;;
esac