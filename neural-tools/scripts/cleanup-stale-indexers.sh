#!/bin/bash
# cleanup-stale-indexers.sh
# Implements ADR-0048 immediate remediation for stale indexer containers
# Date: 2025-09-15

echo "🔍 ADR-0048: Cleaning up stale indexer containers..."
echo "=================================================="

# Count existing indexer containers
TOTAL_COUNT=$(docker ps -a --filter "name=indexer-" --format '{{.Names}}' | wc -l)
echo "Found $TOTAL_COUNT indexer containers total"

# First, remove any exited indexer containers
echo ""
echo "Step 1: Removing exited indexer containers..."
EXITED_COUNT=$(docker ps -a --filter "ancestor=l9-neural-indexer:production" --filter "status=exited" -q | wc -l)
if [ "$EXITED_COUNT" -gt 0 ]; then
    docker ps -a --filter "ancestor=l9-neural-indexer:production" --filter "status=exited" -q | xargs -r docker rm
    echo "✅ Removed $EXITED_COUNT exited containers"
else
    echo "✅ No exited containers found"
fi

# Second, check for misconfigured containers (those with host paths)
echo ""
echo "Step 2: Checking for misconfigured containers (host paths instead of container paths)..."
MISCONFIGURED_COUNT=0

docker ps -a --filter "name=indexer-" --format '{{.ID}} {{.Names}}' | while read id name; do
    # Check if container has PROJECT_PATH set to host path (e.g., /Users/)
    if docker inspect "$id" 2>/dev/null | grep -q '"PROJECT_PATH".*"/Users/'; then
        echo "  ⚠️  Found misconfigured container: $name (ID: ${id:0:12})"
        echo "     PROJECT_PATH uses host path instead of /workspace"

        # Check if running
        STATUS=$(docker inspect "$id" --format '{{.State.Status}}' 2>/dev/null)
        if [ "$STATUS" = "running" ]; then
            echo "     Gracefully stopping container..."
            docker stop "$id" --time=10 2>/dev/null
        fi

        echo "     Removing container..."
        docker rm -f "$id" 2>/dev/null
        ((MISCONFIGURED_COUNT++))
    fi
done

if [ "$MISCONFIGURED_COUNT" -gt 0 ]; then
    echo "✅ Removed $MISCONFIGURED_COUNT misconfigured containers"
else
    echo "✅ No misconfigured containers found"
fi

# Third, list any remaining indexer containers (these should be properly configured)
echo ""
echo "Step 3: Listing remaining indexer containers..."
REMAINING=$(docker ps -a --filter "name=indexer-" --format "table {{.Names}}\t{{.Status}}\t{{.ID}}")
if [ -n "$REMAINING" ]; then
    echo "$REMAINING"
else
    echo "✅ No indexer containers remaining"
fi

echo ""
echo "=================================================="
echo "✅ ADR-0048 cleanup complete!"
echo ""
echo "Next steps:"
echo "1. New indexer containers will be created with correct PROJECT_PATH=/workspace"
echo "2. Monitor for any new startup failures"
echo "3. If issues persist, check docker logs for specific container"
echo ""
echo "To verify a specific project's indexer configuration:"
echo "  docker inspect indexer-<project-name> | grep PROJECT_PATH"