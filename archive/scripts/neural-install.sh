#!/bin/bash
# Neural Tools - Docker MCP Installation Script
# Simple installer for L9 Neural Tools system

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo -e "${BLUE}🧠 Neural Tools Installation${NC}"
echo "=================================="
echo

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running${NC}"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo -e "${GREEN}✅ Docker is ready${NC}"

# Parse arguments
SCOPE=""
PROJECT_PATH=""
PROJECT_NAME=""
INTERACTIVE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --scope)
            SCOPE="$2"
            INTERACTIVE=false
            shift 2
            ;;
        --project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options] [project-path]"
            echo
            echo "Options:"
            echo "  --scope project|user    Installation scope (default: asks interactively)"
            echo "  --project-name NAME     Project name for containers (default: directory name)"
            echo "  --help, -h             Show this help"
            echo
            echo "Examples:"
            echo "  $0                           # Interactive installation"
            echo "  $0 /path/to/project         # Install in specific project"
            echo "  $0 --scope user             # Install globally for all projects"
            echo "  $0 --project-name myapp     # Custom project name"
            exit 0
            ;;
        *)
            PROJECT_PATH="$1"
            shift
            ;;
    esac
done

# Interactive scope selection if not provided
if [[ -z "$SCOPE" ]]; then
    echo -e "${BLUE}Choose installation scope:${NC}"
    echo "  1) Project - Install for a specific project (recommended)"
    echo "  2) User    - Install globally for all projects"
    echo
    read -p "Select [1-2]: " choice
    case $choice in
        1)
            SCOPE="project"
            ;;
        2)
            SCOPE="user"
            ;;
        *)
            echo -e "${RED}Invalid choice. Using project scope.${NC}"
            SCOPE="project"
            ;;
    esac
    echo
fi

# Get project path if needed and not provided
if [[ "$SCOPE" == "project" && -z "$PROJECT_PATH" ]]; then
    echo -e "${BLUE}Enter project path (or press Enter for current directory):${NC}"
    read -p "Path [$(pwd)]: " input_path
    if [[ -z "$input_path" ]]; then
        PROJECT_PATH="$(pwd)"
    else
        PROJECT_PATH="$input_path"
    fi
    echo
fi

# Default to current directory if still not set
if [[ -z "$PROJECT_PATH" ]]; then
    PROJECT_PATH="$(pwd)"
fi

# Determine project name
if [[ -z "$PROJECT_NAME" ]]; then
    PROJECT_NAME=$(basename "$PROJECT_PATH")
fi

# Sanitize project name (Docker-compatible)
PROJECT_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g')

echo -e "${BLUE}📦 Installing Neural Tools${NC}"
echo "  Scope: $SCOPE"
echo "  Project: $PROJECT_NAME"
echo "  Path: $PROJECT_PATH"
echo

# Step 1: Deploy Docker containers
echo -e "${BLUE}1. Deploying Neural Tools containers...${NC}"
cd "$SCRIPT_DIR/neural-tools"

# Stop existing containers if they exist
if docker ps -a --format '{{.Names}}' | grep -q "${PROJECT_NAME}-neural"; then
    echo "  Stopping existing containers..."
    PROJECT_NAME="$PROJECT_NAME" docker-compose -f docker-compose.neural-tools.yml down 2>/dev/null || true
fi

# Start containers
PROJECT_NAME="$PROJECT_NAME" PROJECT_DIR="$PROJECT_PATH" \
    docker-compose -f docker-compose.neural-tools.yml up -d

# Wait for services to be healthy
echo -e "${BLUE}2. Waiting for services to be ready...${NC}"
for i in {1..30}; do
    if docker exec "${PROJECT_NAME}-neural" python3 -c "print('MCP server ready')" 2>/dev/null &&
       docker exec "${PROJECT_NAME}-neural-storage" curl -s "http://localhost:6333/health" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Services are healthy${NC}"
        break
    fi
    echo -n "."
    sleep 2
done
echo

# Step 2: Configure MCP
echo -e "${BLUE}3. Configuring MCP client...${NC}"

# Determine config file location  
if [[ "$SCOPE" == "user" ]]; then
    CONFIG_FILE="$HOME/.claude/mcp_config.json"  # Correct Claude Code CLI config
    CONFIG_DIR="$HOME/.claude"
else
    CONFIG_FILE="$PROJECT_PATH/.mcp.json"
    CONFIG_DIR="$PROJECT_PATH"
fi

mkdir -p "$CONFIG_DIR"

# Check if config exists
if [[ -f "$CONFIG_FILE" ]]; then
    echo "  Updating existing configuration..."
    # Backup existing config
    cp "$CONFIG_FILE" "${CONFIG_FILE}.backup.$(date +%Y%m%d-%H%M%S)"
else
    echo "  Creating new configuration..."
    echo '{"mcpServers": {}}' > "$CONFIG_FILE"
fi

# Create MCP configuration
MCP_CONFIG=$(cat << EOF
{
  "neural-tools-${PROJECT_NAME}": {
    "command": "docker",
    "args": [
      "exec",
      "-i",
      "${PROJECT_NAME}-neural",
      "python3",
      "-u",
      "/app/neural-mcp-server-enhanced.py"
    ],
    "env": {
      "PROJECT_NAME": "${PROJECT_NAME}",
      "PROJECT_DIR": "${PROJECT_PATH}",
      "QDRANT_HOST": "neural-data-storage",
      "QDRANT_GRPC_PORT": "6334",
      "EMBEDDING_SERVICE_HOST": "neural-embeddings",
      "EMBEDDING_SERVICE_PORT": "8000",
      "USE_EXTERNAL_EMBEDDING": "true",
      "GRAPHRAG_ENABLED": "true",
      "HYBRID_SEARCH_MODE": "enhanced",
      "NEO4J_URI": "bolt://neo4j-graph:7687",
      "NEO4J_USERNAME": "neo4j",
      "NEO4J_PASSWORD": "neural-l9-2025",
      "KUZU_DB_PATH": "/app/kuzu",
      "PYTHONUNBUFFERED": "1"
    },
    "alwaysAllow": [
      "mcp__neural-tools-${PROJECT_NAME}__memory_store_enhanced",
      "mcp__neural-tools-${PROJECT_NAME}__memory_search_enhanced", 
      "mcp__neural-tools-${PROJECT_NAME}__kuzu_graph_query",
      "mcp__neural-tools-${PROJECT_NAME}__neo4j_graph_query",
      "mcp__neural-tools-${PROJECT_NAME}__neo4j_semantic_graph_search",
      "mcp__neural-tools-${PROJECT_NAME}__neo4j_code_dependencies",
      "mcp__neural-tools-${PROJECT_NAME}__neo4j_migration_status",
      "mcp__neural-tools-${PROJECT_NAME}__neo4j_index_code_graph",
      "mcp__neural-tools-${PROJECT_NAME}__semantic_code_search",
      "mcp__neural-tools-${PROJECT_NAME}__project_auto_index",
      "mcp__neural-tools-${PROJECT_NAME}__neural_system_status"
    ],
    "description": "Neural Tools for ${PROJECT_NAME} - L9 Enhanced with Neo4j GraphRAG + 9 vibe coder tools"
  }
}
EOF
)

# Add to config using Python for proper JSON handling
python3 << PYTHON_END
import json
import sys

config_file = "$CONFIG_FILE"
new_server = $MCP_CONFIG

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except:
    config = {"mcpServers": {}}

if "mcpServers" not in config:
    config["mcpServers"] = {}

# Add new server config
config["mcpServers"].update(new_server)

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("  ✅ MCP configuration added")
PYTHON_END

echo
echo -e "${GREEN}🎉 Neural Tools Installation Complete!${NC}"
echo
echo -e "${BLUE}📚 Available MCP Tools:${NC}"
echo "  • memory_store_enhanced - Store with GraphRAG + embeddings"
echo "  • memory_search_enhanced - Hybrid search with RRF fusion"
echo "  • neo4j_graph_query - Execute Neo4j Cypher queries"
echo "  • neo4j_semantic_graph_search - Semantic search across code graph"
echo "  • neo4j_code_dependencies - Get dependency graph for files"
echo "  • neo4j_migration_status - Check Neo4j migration status"
echo "  • neo4j_index_code_graph - Index code files into Neo4j"
echo "  • neural_system_status - System performance monitoring"
echo "  • semantic_code_search - Find code by meaning"
echo "  • project_auto_index - Auto-index project files"
echo
echo -e "${YELLOW}🚀 Next Steps:${NC}"
if [[ "$SCOPE" == "project" ]]; then
    echo "  1. Open this project in Claude Code"
    echo "  2. The Neural Tools MCP server is already configured"
    echo "  3. Start using the tools in your conversations!"
else
    echo "  1. Restart Claude Code to load the new configuration"
    echo "  2. Neural Tools will be available in all projects"
    echo "  3. Each project gets its own isolated data"
fi
echo
echo -e "${BLUE}📋 Container Status:${NC}"
docker ps --filter "name=${PROJECT_NAME}-" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
echo
echo -e "${BLUE}💡 Management Commands:${NC}"
echo "  View logs:    docker logs ${PROJECT_NAME}-neural"
echo "  Stop:         cd $SCRIPT_DIR/neural-tools && PROJECT_NAME=$PROJECT_NAME docker-compose -f docker-compose.neural-tools.yml down"
echo "  Restart:      cd $SCRIPT_DIR/neural-tools && PROJECT_NAME=$PROJECT_NAME docker-compose -f docker-compose.neural-tools.yml restart"
echo