#!/bin/bash

# MCP Add Script - Adds neural-tools MCP to a project
# Usage: mcp-add [project-path] [project-name]
#        mcp-add  # Auto-detects from current directory

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Auto-detect or use provided arguments
if [ $# -eq 0 ]; then
    # Auto-detect from current directory
    PROJECT_PATH=$(pwd)
    
    # Try to extract project name from:
    # 1. package.json (Node.js projects)
    # 2. Cargo.toml (Rust projects)
    # 3. pyproject.toml or setup.py (Python projects)
    # 4. Directory name as fallback
    
    PROJECT_NAME=""
    
    # Check package.json
    if [ -f "$PROJECT_PATH/package.json" ] && command -v jq &> /dev/null; then
        PROJECT_NAME=$(jq -r '.name // empty' "$PROJECT_PATH/package.json" 2>/dev/null || true)
    fi
    
    # Check Cargo.toml
    if [ -z "$PROJECT_NAME" ] && [ -f "$PROJECT_PATH/Cargo.toml" ]; then
        PROJECT_NAME=$(grep '^name' "$PROJECT_PATH/Cargo.toml" | head -1 | sed 's/.*= *"\(.*\)"/\1/' 2>/dev/null || true)
    fi
    
    # Check pyproject.toml
    if [ -z "$PROJECT_NAME" ] && [ -f "$PROJECT_PATH/pyproject.toml" ]; then
        PROJECT_NAME=$(grep '^name' "$PROJECT_PATH/pyproject.toml" | head -1 | sed 's/.*= *"\(.*\)"/\1/' 2>/dev/null || true)
    fi
    
    # Fallback to directory name
    if [ -z "$PROJECT_NAME" ]; then
        PROJECT_NAME=$(basename "$PROJECT_PATH")
    fi
    
    # Sanitize project name (remove special chars, lowercase)
    PROJECT_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/_/g')
    
    # Check if we're in a Git repository
    if [ -d "$PROJECT_PATH/.git" ]; then
        echo -e "${BLUE}🔍 Auto-detected project configuration:${NC}"
        echo "  Path: $PROJECT_PATH"
        echo "  Name: $PROJECT_NAME"
        echo ""
        echo -e "${YELLOW}Press Enter to continue or Ctrl+C to cancel...${NC}"
        read -r
    else
        echo -e "${YELLOW}⚠️  Current directory doesn't appear to be a project directory${NC}"
        echo "  Path: $PROJECT_PATH"
        echo ""
        echo "Not a Git repository and no package.json/Cargo.toml/pyproject.toml found."
        echo ""
        echo "Usage: mcp-add [project-path] [project-name]"
        echo "       mcp-add  # Auto-detects from current directory"
        echo ""
        echo "Examples:"
        echo "  mcp-add /path/to/project   # Add to specific project"
        echo "  mcp-add . my-app           # Current dir with custom name"
        exit 1
    fi
    
elif [ $# -eq 1 ]; then
    # One argument: use as project path, auto-detect name
    PROJECT_PATH="$1"
    PROJECT_PATH=$(cd "$PROJECT_PATH" 2>/dev/null && pwd || echo "$PROJECT_PATH")
    
    if [ ! -d "$PROJECT_PATH" ]; then
        echo -e "${RED}Error: Project path does not exist: $PROJECT_PATH${NC}"
        exit 1
    fi
    
    PROJECT_NAME=$(basename "$PROJECT_PATH" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/_/g')
    
elif [ $# -eq 2 ]; then
    # Two arguments: use as provided
    PROJECT_PATH="$1"
    PROJECT_NAME="$2"
    
    # Convert to absolute path
    PROJECT_PATH=$(cd "$PROJECT_PATH" 2>/dev/null && pwd || echo "$PROJECT_PATH")
    
    if [ ! -d "$PROJECT_PATH" ]; then
        echo -e "${RED}Error: Project path does not exist: $PROJECT_PATH${NC}"
        exit 1
    fi
    
else
    echo "Usage: mcp-add [project-path] [project-name]"
    echo "       mcp-add  # Auto-detects from current directory"
    echo ""
    echo "Examples:"
    echo "  mcp-add                    # Auto-detect everything"
    echo "  mcp-add /path/to/project   # Auto-detect name"
    echo "  mcp-add . my-app           # Current dir with custom name"
    exit 1
fi

# Sanitize project name one more time
PROJECT_NAME=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/_/g')

# MCP installation path (where this script lives)
MCP_BASE_PATH="/Users/mkr/local-coding/claude-l9-template"

echo -e "${BLUE}🔧 Adding Neural-Tools MCP to project...${NC}"
echo "  Project: $PROJECT_NAME"
echo "  Path: $PROJECT_PATH"

# Check if .mcp.json exists
MCP_CONFIG_FILE="$PROJECT_PATH/.mcp.json"

if [ -f "$MCP_CONFIG_FILE" ]; then
    echo -e "${YELLOW}⚠️  Found existing .mcp.json${NC}"
    
    # Check if neural-tools already exists
    if grep -q '"neural-tools"' "$MCP_CONFIG_FILE"; then
        echo -e "${YELLOW}Updating existing neural-tools configuration...${NC}"
        # Backup existing file
        cp "$MCP_CONFIG_FILE" "$MCP_CONFIG_FILE.backup-$(date +%Y%m%d-%H%M%S)"
    else
        echo -e "${GREEN}Adding neural-tools to existing MCP configuration...${NC}"
        # We need to merge configurations - using Python for JSON manipulation
        python3 - <<EOF
import json
import sys

config_file = "$MCP_CONFIG_FILE"

# Read existing config
with open(config_file, 'r') as f:
    config = json.load(f)

# Ensure mcpServers exists
if 'mcpServers' not in config:
    config['mcpServers'] = {}

# Add neural-tools configuration
config['mcpServers']['neural-tools'] = {
    "command": "python3",
    "args": ["$MCP_BASE_PATH/neural-tools/run_mcp_server.py"],
    "cwd": "$PROJECT_PATH",
    "env": {
        "PYTHONPATH": "$MCP_BASE_PATH/neural-tools/src",
        "PROJECT_NAME": "$PROJECT_NAME",
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "EMBED_DIM": "768",
        "PYTHONUNBUFFERED": "1",
        "REDIS_CACHE_HOST": "localhost",
        "REDIS_CACHE_PORT": "46379",
        "REDIS_QUEUE_HOST": "localhost",
        "REDIS_QUEUE_PORT": "46380"
    }
}

# Write updated config
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Successfully added neural-tools to existing .mcp.json")
EOF
        exit 0
    fi
else
    echo -e "${GREEN}Creating new .mcp.json...${NC}"
    
    # Create new .mcp.json
    cat > "$MCP_CONFIG_FILE" <<EOF
{
  "mcpServers": {
    "neural-tools": {
      "command": "python3",
      "args": ["$MCP_BASE_PATH/neural-tools/run_mcp_server.py"],
      "cwd": "$PROJECT_PATH",
      "env": {
        "PYTHONPATH": "$MCP_BASE_PATH/neural-tools/src",
        "PROJECT_NAME": "$PROJECT_NAME",
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "EMBED_DIM": "768",
        "PYTHONUNBUFFERED": "1",
        "REDIS_CACHE_HOST": "localhost",
        "REDIS_CACHE_PORT": "46379",
        "REDIS_QUEUE_HOST": "localhost",
        "REDIS_QUEUE_PORT": "46380"
      }
    }
  }
}
EOF
fi

echo -e "${GREEN}✅ MCP configuration added successfully!${NC}"
echo ""
echo -e "${BLUE}📦 Project will use:${NC}"
echo "  - Collection: project_${PROJECT_NAME}_code"
echo "  - Shared indexer: l9-neural-indexer"
echo "  - Data isolation: Complete"
echo ""
echo -e "${YELLOW}🚀 Next steps:${NC}"
echo "  1. Open $PROJECT_PATH in Claude"
echo "  2. Claude will automatically connect to neural-tools MCP"
echo "  3. To trigger indexing, run:"
echo "     docker exec l9-neural-indexer python3 -c \"print('Reindex request for $PROJECT_PATH')\""
echo ""
echo -e "${GREEN}Done! Your project is ready for semantic search.${NC}"