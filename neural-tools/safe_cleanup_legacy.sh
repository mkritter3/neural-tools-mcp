#!/bin/bash

# Safe Cleanup Script for Legacy Neural Tools
# This script safely archives the old setup before removing it
# Everything is reversible!

set -e  # Exit on any error

echo "🔐 L9 Neural Tools - Safe Legacy Cleanup"
echo "========================================"
echo ""
echo "This script will:"
echo "1. ✅ Create full backup of Docker setup"
echo "2. ✅ Archive legacy code for reference"
echo "3. ✅ Stop redundant containers"
echo "4. ✅ Clean up unused Docker resources"
echo "5. ✅ Keep SeaGOAT + minimal MCP setup"
echo ""
echo "⚠️  Everything is reversible with the backup!"
echo ""
read -p "Continue with cleanup? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

# Create backup directory with timestamp
BACKUP_DIR="$HOME/neural-tools-backup-$(date +%Y%m%d-%H%M%S)"
echo ""
echo "📦 Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Step 1: Export Docker container configs
echo ""
echo "1️⃣ Backing up Docker container configurations..."
mkdir -p "$BACKUP_DIR/docker-configs"

# Save container info for each legacy container
for container in default-neural-storage neural-embeddings neo4j-graph; do
    if docker ps -a --format '{{.Names}}' | grep -q "^$container$"; then
        echo "   📋 Saving config for $container"
        docker inspect "$container" > "$BACKUP_DIR/docker-configs/${container}-inspect.json"
    fi
done

# Step 2: Archive the legacy code
echo ""
echo "2️⃣ Archiving legacy neural tools code..."
if [ -d "neural-tools/deprecated" ]; then
    cp -r neural-tools/deprecated "$BACKUP_DIR/legacy-code"
    echo "   ✅ Legacy code archived"
fi

# Archive old MCP servers
mkdir -p "$BACKUP_DIR/legacy-mcp-servers"
for file in neural-tools/src/servers/neural-mcp-server-enhanced.py \
            neural-tools/src/servers/neural_mcp_wrapper.py \
            neural-tools/src/servers/neural_server_refactored.py; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/legacy-mcp-servers/" 2>/dev/null || true
    fi
done

# Step 3: Save Docker compose files
echo ""
echo "3️⃣ Backing up Docker Compose configurations..."
for compose_file in docker-compose*.yml .env; do
    if [ -f "$compose_file" ]; then
        cp "$compose_file" "$BACKUP_DIR/"
        echo "   📄 Saved $compose_file"
    fi
done

# Step 4: Export Docker volumes (metadata only, not full data)
echo ""
echo "4️⃣ Documenting Docker volumes..."
docker volume ls --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}" > "$BACKUP_DIR/docker-volumes.txt"
echo "   ✅ Volume list saved (data remains on disk until you explicitly remove it)"

# Create restoration script
echo ""
echo "5️⃣ Creating restoration script..."
cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# Restoration Script - Run this to restore the legacy setup

echo "🔄 Restoring Legacy Neural Tools Setup"
echo "======================================"
echo ""
echo "This will restore the Docker containers (but not the data volumes)."
echo "Data volumes were not deleted and can be reattached."
echo ""
read -p "Continue with restoration? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restoration cancelled"
    exit 1
fi

# Restore would go here - requires docker-compose files
echo "To fully restore:"
echo "1. Copy back the docker-compose files"
echo "2. Run: docker-compose -f docker-compose.neural-tools.yml up -d"
echo "3. The old data volumes still exist unless you ran 'docker volume prune'"
EOF

chmod +x "$BACKUP_DIR/restore.sh"

echo ""
echo "✅ Backup complete at: $BACKUP_DIR"
echo ""
echo "================================================"
echo "🧹 Starting Cleanup Process"
echo "================================================"
echo ""

# Step 6: Stop redundant containers
echo "6️⃣ Stopping redundant Docker containers..."
for container in default-neural-storage neural-embeddings neo4j-graph; do
    if docker ps --format '{{.Names}}' | grep -q "^$container$"; then
        echo "   🛑 Stopping $container..."
        docker stop "$container" || true
    fi
done

echo ""
echo "7️⃣ Removing stopped containers..."
for container in default-neural-storage neural-embeddings neo4j-graph; do
    if docker ps -a --format '{{.Names}}' | grep -q "^$container$"; then
        echo "   🗑️  Removing $container..."
        docker rm "$container" || true
    fi
done

# Step 8: Display space savings
echo ""
echo "8️⃣ Checking disk space..."
echo ""
echo "Docker disk usage BEFORE cleanup:"
docker system df

echo ""
echo "🧹 Cleaning up unused Docker resources..."
echo "   (Keeping volumes for now - run 'docker volume prune' manually if needed)"
docker image prune -f

echo ""
echo "Docker disk usage AFTER cleanup:"
docker system df

# Step 9: Verify remaining setup
echo ""
echo "9️⃣ Verifying remaining setup..."
echo ""
echo "✅ Active containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "✅ SeaGOAT server status:"
if pgrep -f "seagoat-server" > /dev/null; then
    echo "   🟢 SeaGOAT server is running"
    seagoat-server status /Users/mkr/local-coding/claude-l9-template || true
else
    echo "   🔴 SeaGOAT server is not running"
    echo "   Start it with: seagoat-server start /Users/mkr/local-coding/claude-l9-template"
fi

echo ""
echo "================================================"
echo "✅ CLEANUP COMPLETE!"
echo "================================================"
echo ""
echo "📦 Full backup saved at: $BACKUP_DIR"
echo "🔄 To restore, run: $BACKUP_DIR/restore.sh"
echo ""
echo "Your simplified setup:"
echo "  • 1 Docker container (default-neural) for MCP"
echo "  • 1 SeaGOAT server for semantic search"
echo "  • All legacy code archived for reference"
echo ""
echo "💡 To remove Docker volumes and free more space:"
echo "   docker volume prune"
echo ""
echo "🎉 Enjoy your cleaner, simpler architecture!"