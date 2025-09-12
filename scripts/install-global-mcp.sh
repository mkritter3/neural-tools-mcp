#!/bin/bash

# L9 Neural Tools - Global MCP Installation for Claude Code CLI
# Makes neural-tools MCP available in ALL projects

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     L9 Neural Tools - Global MCP Installation           ║${NC}"
echo -e "${BLUE}║     Making neural-tools available everywhere            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define paths
MCP_GLOBAL_DIR="$HOME/.claude/mcp-servers/neural-tools"
CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
CLAUDE_CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

# Step 1: Create global MCP directory
echo -e "${BLUE}📁 Creating global MCP directory...${NC}"
mkdir -p "$MCP_GLOBAL_DIR"

# Step 2: Copy neural-tools to global location
echo -e "${BLUE}📦 Installing neural-tools to global location...${NC}"
cp -r "$PROJECT_ROOT/neural-tools/"* "$MCP_GLOBAL_DIR/"

# Step 3: Create wrapper script for project detection
echo -e "${BLUE}🔧 Creating project detection wrapper...${NC}"
cat > "$MCP_GLOBAL_DIR/wrapper.py" << 'EOF'
#!/usr/bin/env python3
"""
Global MCP wrapper that detects current project context.
Allows neural-tools to work in any project directory.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get current working directory from Claude
    cwd = os.environ.get('PWD', os.getcwd())
    
    # Set project name from directory
    project_name = Path(cwd).name
    
    # Set environment for project context
    env = os.environ.copy()
    env['PROJECT_PATH'] = cwd
    env['PROJECT_NAME'] = project_name
    
    # Add neural-tools src to Python path
    mcp_base = Path(__file__).parent
    env['PYTHONPATH'] = str(mcp_base / 'src')
    
    # Preserve all MCP environment variables
    env.setdefault('AUTO_DETECT_PROJECT', 'true')
    env.setdefault('NEO4J_URI', 'bolt://localhost:47687')
    env.setdefault('NEO4J_USERNAME', 'neo4j')
    env.setdefault('NEO4J_PASSWORD', 'graphrag-password')
    env.setdefault('QDRANT_HOST', 'localhost')
    env.setdefault('QDRANT_PORT', '46333')
    env.setdefault('EMBEDDING_SERVICE_HOST', 'localhost')
    env.setdefault('EMBEDDING_SERVICE_PORT', '48000')
    env.setdefault('REDIS_CACHE_HOST', 'localhost')
    env.setdefault('REDIS_CACHE_PORT', '46379')
    env.setdefault('REDIS_QUEUE_HOST', 'localhost')
    env.setdefault('REDIS_QUEUE_PORT', '46380')
    env.setdefault('EMBED_DIM', '768')
    env.setdefault('PYTHONUNBUFFERED', '1')
    
    # Launch actual MCP server
    server_path = mcp_base / 'run_mcp_server.py'
    subprocess.run([sys.executable, str(server_path)], env=env)

if __name__ == '__main__':
    main()
EOF

chmod +x "$MCP_GLOBAL_DIR/wrapper.py"

# Step 4: Update Claude configuration
echo -e "${BLUE}🔄 Updating Claude configuration...${NC}"

# Create config directory if it doesn't exist
mkdir -p "$CLAUDE_CONFIG_DIR"

# Use Python to update the configuration
python3 - <<EOF
import json
import os
from pathlib import Path

config_file = "$CLAUDE_CONFIG_FILE"
mcp_global_dir = "$MCP_GLOBAL_DIR"

# Read existing config or create new
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    config = {"mcpServers": {}}

# Update neural-tools configuration to use global wrapper
config['mcpServers']['neural-tools'] = {
    "command": "python3",
    "args": [f"{mcp_global_dir}/wrapper.py"],
    "env": {
        "AUTO_DETECT_PROJECT": "true",
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "EMBEDDING_SERVICE_HOST": "localhost",
        "EMBEDDING_SERVICE_PORT": "48000",
        "REDIS_CACHE_HOST": "localhost",
        "REDIS_CACHE_PORT": "46379",
        "REDIS_QUEUE_HOST": "localhost",
        "REDIS_QUEUE_PORT": "46380",
        "EMBED_DIM": "768",
        "PYTHONUNBUFFERED": "1"
    }
}

# Write updated config
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Claude configuration updated")
EOF

# Step 5: Verify installation
echo ""
echo -e "${BLUE}🔍 Verifying installation...${NC}"

if [ -f "$MCP_GLOBAL_DIR/wrapper.py" ] && [ -f "$MCP_GLOBAL_DIR/run_mcp_server.py" ]; then
    echo -e "${GREEN}✅ Global MCP installation successful!${NC}"
else
    echo -e "${RED}❌ Installation may have issues. Please check the files.${NC}"
    exit 1
fi

# Step 6: Display success message
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           🎉 Installation Complete! 🎉                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📍 Installation Details:${NC}"
echo "   • Global MCP Location: $MCP_GLOBAL_DIR"
echo "   • Config File: $CLAUDE_CONFIG_FILE"
echo ""
echo -e "${BLUE}🚀 What's New:${NC}"
echo "   • neural-tools MCP now works in ANY project directory"
echo "   • Auto-detects current project for isolation"
echo "   • No per-project configuration needed"
echo ""
echo -e "${BLUE}📚 Available MCP Tools (works everywhere):${NC}"
echo "   • semantic_code_search - Search by meaning"
echo "   • graphrag_hybrid_search - Graph + vector search"
echo "   • project_understanding - Get codebase overview"
echo "   • neural_system_status - Check system health"
echo ""
echo -e "${YELLOW}⚠️  Important:${NC}"
echo "   • Restart Claude Code CLI for changes to take effect"
echo "   • Docker services must be running (docker-compose up -d)"
echo ""
echo -e "${GREEN}You can now use neural-tools MCP in any project!${NC}"