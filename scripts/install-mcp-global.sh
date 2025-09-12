#!/bin/bash

# Install Neural-Tools MCP globally in Claude Desktop
# This makes it available to ALL projects automatically

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🌐 Installing Neural-Tools MCP globally...${NC}"

# Check if Docker containers are running
echo -e "${BLUE}Checking required services...${NC}"

MISSING_SERVICES=()

# Check each required service
if ! docker ps | grep -q "claude-l9-template-neo4j"; then
    MISSING_SERVICES+=("Neo4j")
fi

if ! docker ps | grep -q "claude-l9-template-qdrant"; then
    MISSING_SERVICES+=("Qdrant")
fi

if ! docker ps | grep -q "neural-flow-nomic"; then
    MISSING_SERVICES+=("Nomic Embeddings")
fi

if [ ${#MISSING_SERVICES[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing required services: ${MISSING_SERVICES[*]}${NC}"
    echo ""
    echo "Please start Docker services first:"
    echo "  cd /Users/mkr/local-coding/claude-l9-template"
    echo "  docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✅ All required services running${NC}"

# Update global Claude configuration
echo -e "${BLUE}Updating global Claude configuration...${NC}"

python3 - <<'EOF'
import json
import os

config_file = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
mcp_base_path = "/Users/mkr/local-coding/claude-l9-template"

# Read existing config
with open(config_file, 'r') as f:
    config = json.load(f)

# Update neural-tools (already exists but we'll improve it)
config['mcpServers']['neural-tools'] = {
    "command": "python3",
    "args": [f"{mcp_base_path}/neural-tools/run_mcp_server.py"],
    "env": {
        "PYTHONPATH": f"{mcp_base_path}/neural-tools/src",
        "AUTO_DETECT_PROJECT": "true",
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "EMBEDDING_SERVICE_HOST": "localhost",
        "EMBEDDING_SERVICE_PORT": "48000",
        "EMBED_DIM": "768",
        "PYTHONUNBUFFERED": "1"
    }
}

# Write updated config
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Updated global configuration")
EOF

echo ""
echo -e "${GREEN}🎉 Neural-Tools MCP configured globally!${NC}"
echo ""
echo -e "${BLUE}Available in ALL projects with:${NC}"
echo "  • Pattern-based metadata extraction"
echo "  • Semantic code search"
echo "  • GraphRAG hybrid search"
echo ""
echo -e "${GREEN}✅ Ready to use!${NC}"
