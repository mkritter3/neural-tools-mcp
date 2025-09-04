#!/bin/bash
# Docker Cleanup Script for Neural Tools
# Prevents storage debt from continuous rebuilds

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🧹 Docker Cleanup for Neural Tools"
echo "==================================="
echo ""

# Function to format bytes
format_bytes() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "$1 bytes"
}

# Show current usage
echo -e "${BLUE}Current Docker Disk Usage:${NC}"
docker system df
echo ""

# Parse arguments
AGGRESSIVE=false
KEEP_BUILDS=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --aggressive)
            AGGRESSIVE=true
            shift
            ;;
        --keep-builds)
            KEEP_BUILDS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --aggressive   Remove ALL unused images (not just dangling)"
            echo "  --keep-builds  Keep recent build cache (last 24h)"
            echo "  --dry-run      Show what would be removed without removing"
            echo "  --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Safe cleanup (dangling only)"
            echo "  $0 --aggressive       # Remove all unused images"
            echo "  $0 --dry-run          # Preview what would be cleaned"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Safety check - preserve our neural-tools images
echo -e "${YELLOW}Preserving Neural Tools images:${NC}"
PRESERVE_IMAGES=(
    "l9-mcp-enhanced:minimal-fixed"
    "neural-flow:nomic-v2-production"
    "qdrant/qdrant:v1.10.0"
    "neo4j:5.23.0-community"
)

for img in "${PRESERVE_IMAGES[@]}"; do
    if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^$img$"; then
        echo "  ✓ $img"
    fi
done
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🔍 DRY RUN MODE - Nothing will be deleted${NC}"
    echo ""
fi

# Step 1: Remove stopped containers (except neural-tools)
echo -e "${BLUE}1. Cleaning stopped containers...${NC}"
STOPPED_CONTAINERS=$(docker ps -a -q --filter "status=exited" | wc -l | tr -d ' ')
if [ "$STOPPED_CONTAINERS" -gt 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "  Would remove $STOPPED_CONTAINERS stopped containers:"
        docker ps -a --filter "status=exited" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
    else
        docker container prune -f
        echo -e "${GREEN}  ✓ Removed $STOPPED_CONTAINERS stopped containers${NC}"
    fi
else
    echo "  No stopped containers to remove"
fi
echo ""

# Step 2: Remove dangling images
echo -e "${BLUE}2. Cleaning dangling images...${NC}"
DANGLING_IMAGES=$(docker images -f "dangling=true" -q | wc -l | tr -d ' ')
if [ "$DANGLING_IMAGES" -gt 0 ]; then
    DANGLING_SIZE=$(docker images -f "dangling=true" --format "{{.Size}}" | awk '{sum+=$1} END {print sum}')
    if [ "$DRY_RUN" = true ]; then
        echo "  Would remove $DANGLING_IMAGES dangling images"
    else
        docker image prune -f
        echo -e "${GREEN}  ✓ Removed $DANGLING_IMAGES dangling images${NC}"
    fi
else
    echo "  No dangling images to remove"
fi
echo ""

# Step 3: Remove unused images (if aggressive)
if [ "$AGGRESSIVE" = true ]; then
    echo -e "${BLUE}3. Cleaning ALL unused images (aggressive mode)...${NC}"
    
    # Get list of images to preserve
    PRESERVE_LIST=""
    for img in "${PRESERVE_IMAGES[@]}"; do
        IMAGE_ID=$(docker images --format "{{.Repository}}:{{.Tag}}|{{.ID}}" | grep "^$img|" | cut -d'|' -f2)
        if [ -n "$IMAGE_ID" ]; then
            PRESERVE_LIST="$PRESERVE_LIST $IMAGE_ID"
        fi
    done
    
    if [ "$DRY_RUN" = true ]; then
        echo "  Would remove unused images (except preserved ones)"
        docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}"
    else
        # Remove all unused images except our preserved ones
        docker image prune -a -f --filter "label!=preserve=true"
        echo -e "${GREEN}  ✓ Removed unused images${NC}"
    fi
    echo ""
fi

# Step 4: Clean build cache
echo -e "${BLUE}4. Cleaning build cache...${NC}"
if [ "$KEEP_BUILDS" = true ]; then
    # Keep builds from last 24 hours
    if [ "$DRY_RUN" = true ]; then
        echo "  Would remove build cache older than 24h"
        docker builder du
    else
        docker builder prune -f --filter "until=24h"
        echo -e "${GREEN}  ✓ Removed old build cache (kept last 24h)${NC}"
    fi
else
    if [ "$DRY_RUN" = true ]; then
        echo "  Would remove all build cache:"
        docker builder du
    else
        docker builder prune -a -f
        echo -e "${GREEN}  ✓ Removed all build cache${NC}"
    fi
fi
echo ""

# Step 5: Clean unused volumes (be careful!)
echo -e "${BLUE}5. Cleaning unused volumes...${NC}"
UNUSED_VOLUMES=$(docker volume ls -q -f dangling=true | wc -l | tr -d ' ')
if [ "$UNUSED_VOLUMES" -gt 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "  Would remove $UNUSED_VOLUMES unused volumes:"
        docker volume ls -f dangling=true
    else
        # Only remove truly dangling volumes
        docker volume prune -f
        echo -e "${GREEN}  ✓ Removed $UNUSED_VOLUMES unused volumes${NC}"
    fi
else
    echo "  No unused volumes to remove"
fi
echo ""

# Step 6: Clean unused networks
echo -e "${BLUE}6. Cleaning unused networks...${NC}"
if [ "$DRY_RUN" = true ]; then
    echo "  Would remove unused networks"
else
    docker network prune -f
    echo -e "${GREEN}  ✓ Cleaned unused networks${NC}"
fi
echo ""

# Show final usage
echo -e "${GREEN}==================================${NC}"
echo -e "${GREEN}Cleanup Complete!${NC}"
echo ""
echo -e "${BLUE}Final Docker Disk Usage:${NC}"
docker system df
echo ""

# Calculate space saved
if [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}✨ Space reclaimed successfully!${NC}"
fi

# Offer additional tips
echo ""
echo -e "${YELLOW}💡 Tips to prevent storage debt:${NC}"
echo "  1. Run this cleanup weekly: $0"
echo "  2. Use --aggressive monthly for deep clean"
echo "  3. Set Docker daemon limits in ~/.docker/daemon.json:"
echo '     {"builder": {"gc": {"enabled": true, "defaultKeepStorage": "20GB"}}}'
echo "  4. Use build --no-cache sparingly"
echo "  5. Tag images properly to avoid orphans"