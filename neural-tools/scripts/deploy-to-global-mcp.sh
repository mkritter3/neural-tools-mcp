#!/bin/bash
# Deploy Neural Tools to Global MCP
# Implements ADR-0034: MCP Development & Deployment Workflow

set -e  # Exit on error

echo "🚀 Neural Tools Global MCP Deployment"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GLOBAL_MCP_DIR="$HOME/.claude/mcp-servers/neural-tools"
BACKUP_DIR="$HOME/.claude/mcp-servers/neural-tools-backup-$(date +%Y%m%d-%H%M%S)"

echo "📁 Deployment Paths:"
echo "   Source: $PROJECT_DIR"
echo "   Target: $GLOBAL_MCP_DIR"
echo "   Backup: $BACKUP_DIR"
echo ""

# Step 1: Run CI/CD tests
echo "========================================="
echo "🧪 Step 1: Running CI/CD Tests (L9 Standards)"
echo "========================================="
echo ""

# Check for pytest-asyncio requirement
if ! python3 -c "import pytest_asyncio" 2>/dev/null; then
    echo -e "${YELLOW}⚠️ Installing pytest-asyncio for L9 tests...${NC}"
    pip3 install pytest-asyncio > /dev/null 2>&1
fi

# Run the comprehensive test suite
if [ -f "$SCRIPT_DIR/run-cicd-tests.sh" ]; then
    echo "Running comprehensive test suite..."
    echo "  - Unit tests (MCP protocol compliance)"
    echo "  - Integration tests (22 tools)"
    echo "  - E2E tests (subprocess JSON-RPC)"
    echo "  - Failure mode tests (error handling)"
    echo "  - Concurrency tests (race conditions)"
    echo ""

    if "$SCRIPT_DIR/run-cicd-tests.sh"; then
        echo -e "${GREEN}✅ All tests passed including L9 standards${NC}"
    else
        echo -e "${RED}❌ Tests failed! Aborting deployment.${NC}"
        echo ""
        echo "Common failure reasons:"
        echo "  - Docker services not running (Neo4j, Qdrant)"
        echo "  - Missing pytest-asyncio dependency"
        echo "  - MCP server initialization errors"
        echo ""
        echo "Run tests manually for details:"
        echo "  python3 -m pytest tests/integration/test_e2e_all_tools.py -v"
        echo "  python3 -m pytest tests/integration/test_failure_modes.py -v"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️ No CI/CD tests found, proceeding with deployment${NC}"
    echo "WARNING: Deploying without L9 standard validation!"
fi

# Step 2: Create backup
echo ""
echo "========================================="
echo "💾 Step 2: Creating Backup"
echo "========================================="
echo ""

if [ -d "$GLOBAL_MCP_DIR" ]; then
    echo "Backing up existing deployment..."
    cp -r "$GLOBAL_MCP_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}✅ Backup created at: $BACKUP_DIR${NC}"
else
    echo "No existing deployment to backup"
fi

# Step 3: Deploy to global
echo ""
echo "========================================="
echo "📦 Step 3: Deploying to Global MCP"
echo "========================================="
echo ""

# Create parent directory if needed
mkdir -p "$(dirname "$GLOBAL_MCP_DIR")"

# Remove old deployment if exists
if [ -d "$GLOBAL_MCP_DIR" ]; then
    echo "Removing old deployment..."
    rm -rf "$GLOBAL_MCP_DIR"
fi

# Copy new deployment
echo "Copying files..."
cp -r "$PROJECT_DIR" "$GLOBAL_MCP_DIR"

# Remove test directories and other development files from production
echo "Cleaning production deployment..."
rm -rf "$GLOBAL_MCP_DIR/.git"
rm -rf "$GLOBAL_MCP_DIR/.pytest_cache"
rm -rf "$GLOBAL_MCP_DIR/__pycache__"
rm -rf "$GLOBAL_MCP_DIR/scripts/run-cicd-tests.sh"
find "$GLOBAL_MCP_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$GLOBAL_MCP_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

echo -e "${GREEN}✅ Files deployed successfully${NC}"

# Step 4: Verify deployment
echo ""
echo "========================================="
echo "✔️  Step 4: Verifying Deployment"
echo "========================================="
echo ""

# Check critical files exist
CRITICAL_FILES=(
    "src/neural_mcp/neural_server_stdio.py"
    "run_mcp_server.py"
    "src/servers/services/service_container.py"
)

all_files_present=true
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$GLOBAL_MCP_DIR/$file" ]; then
        echo -e "${GREEN}✅ Found: $file${NC}"
    else
        echo -e "${RED}❌ Missing: $file${NC}"
        all_files_present=false
    fi
done

if [ "$all_files_present" = false ]; then
    echo -e "${RED}❌ Deployment verification failed!${NC}"
    echo "Rolling back..."
    if [ -d "$BACKUP_DIR" ]; then
        rm -rf "$GLOBAL_MCP_DIR"
        mv "$BACKUP_DIR" "$GLOBAL_MCP_DIR"
        echo -e "${YELLOW}↩️  Rolled back to previous version${NC}"
    fi
    exit 1
fi

# Step 5: Update global MCP configuration
echo ""
echo "========================================="
echo "⚙️  Step 5: Updating MCP Configuration"
echo "========================================="
echo ""

MCP_CONFIG="$HOME/.claude/mcp_config.json"

if [ -f "$MCP_CONFIG" ]; then
    echo "Global MCP configuration found"

    # Create a temporary Python script to update the JSON properly
    cat > /tmp/update_mcp_config.py << 'PYTHON_EOF'
import json
import sys

config_file = sys.argv[1]
global_mcp_dir = sys.argv[2]

with open(config_file, 'r') as f:
    config = json.load(f)

# Update neural-tools configuration
config['mcpServers']['neural-tools'] = {
    "command": "python3",
    "args": [f"{global_mcp_dir}/run_mcp_server.py"],
    "cwd": global_mcp_dir,
    "env": {
        "PYTHONPATH": f"{global_mcp_dir}/src",
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "EMBEDDING_SERVICE_HOST": "localhost",
        "EMBEDDING_SERVICE_PORT": "48000",
        "NOMIC_API_URL": "http://localhost:48000",
        "EMBED_DIM": "768",
        "PYTHONUNBUFFERED": "1",
        "REDIS_CACHE_HOST": "localhost",
        "REDIS_CACHE_PORT": "46379",
        "REDIS_CACHE_PASSWORD": "cache-secret-key",
        "REDIS_QUEUE_HOST": "localhost",
        "REDIS_QUEUE_PORT": "46380",
        "REDIS_QUEUE_PASSWORD": "queue-secret-key"
    }
}

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Updated neural-tools configuration")
PYTHON_EOF

    # Run the Python script to update the config
    python3 /tmp/update_mcp_config.py "$MCP_CONFIG" "$GLOBAL_MCP_DIR"
    rm /tmp/update_mcp_config.py

    echo -e "${GREEN}✅ neural-tools configuration updated in global config${NC}"
else
    echo -e "${YELLOW}⚠️ No global MCP config found at $MCP_CONFIG${NC}"
    echo "Creating new configuration..."

    # Create a new config file
    cat > "$MCP_CONFIG" << EOF
{
  "mcpServers": {
    "neural-tools": {
      "command": "python3",
      "args": ["$GLOBAL_MCP_DIR/run_mcp_server.py"],
      "cwd": "$GLOBAL_MCP_DIR",
      "env": {
        "PYTHONPATH": "$GLOBAL_MCP_DIR/src",
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "EMBEDDING_SERVICE_HOST": "localhost",
        "EMBEDDING_SERVICE_PORT": "48000",
        "NOMIC_API_URL": "http://localhost:48000",
        "EMBED_DIM": "768",
        "PYTHONUNBUFFERED": "1",
        "REDIS_CACHE_HOST": "localhost",
        "REDIS_CACHE_PORT": "46379",
        "REDIS_CACHE_PASSWORD": "cache-secret-key",
        "REDIS_QUEUE_HOST": "localhost",
        "REDIS_QUEUE_PORT": "46380",
        "REDIS_QUEUE_PASSWORD": "queue-secret-key"
      }
    }
  }
}
EOF
    echo -e "${GREEN}✅ Created new global MCP configuration${NC}"
fi

# Step 6: Create deployment manifest
echo ""
echo "========================================="
echo "📄 Step 6: Creating Deployment Manifest"
echo "========================================="
echo ""

MANIFEST_FILE="$GLOBAL_MCP_DIR/.deployment-manifest.json"
cat > "$MANIFEST_FILE" << EOF
{
    "deployment_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "deployed_from": "$PROJECT_DIR",
    "deployed_by": "$USER",
    "git_commit": "$(cd "$PROJECT_DIR" && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(cd "$PROJECT_DIR" && git branch --show-current 2>/dev/null || echo 'unknown')",
    "backup_location": "$BACKUP_DIR"
}
EOF

echo -e "${GREEN}✅ Deployment manifest created${NC}"

# Summary
echo ""
echo "========================================="
echo "🎉 Deployment Complete!"
echo "========================================="
echo ""
echo -e "${GREEN}✅ neural-tools successfully deployed to global MCP${NC}"
echo ""
echo "Next steps:"
echo "1. Exit this directory: cd ~"
echo "2. Restart Claude to use the global deployment"
echo "3. Test that neural-tools commands work"
echo ""
echo "Rollback command (if needed):"
echo "  rm -rf $GLOBAL_MCP_DIR && mv $BACKUP_DIR $GLOBAL_MCP_DIR"
echo ""