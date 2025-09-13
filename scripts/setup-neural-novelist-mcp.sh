#!/bin/bash
# Setup script to configure MCP for neural-novelist project
# This adds PROJECT_PATH to the environment so the MCP server knows where the user is working

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up MCP for neural-novelist project${NC}"
echo "=================================================="

# Project paths
NEURAL_NOVELIST_PATH="/Users/mkr/local-coding/Systems/neural-novelist"
MCP_CONFIG_PATH="$HOME/.claude/mcp_config.json"

# Check if neural-novelist exists
if [[ ! -d "$NEURAL_NOVELIST_PATH" ]]; then
    echo -e "${RED}❌ Neural-novelist not found at: $NEURAL_NOVELIST_PATH${NC}"
    exit 1
fi

# Create backup of current config
if [[ -f "$MCP_CONFIG_PATH" ]]; then
    cp "$MCP_CONFIG_PATH" "$MCP_CONFIG_PATH.backup.$(date +%Y%m%d-%H%M%S)"
    echo -e "${GREEN}✅ Backed up existing MCP config${NC}"
fi

# Update MCP config with PROJECT_PATH for neural-novelist
echo -e "${YELLOW}📝 Updating MCP configuration...${NC}"

# Use Python to update JSON properly
python3 << EOF
import json
import sys

config_path = "$MCP_CONFIG_PATH"

# Load existing config
with open(config_path, 'r') as f:
    config = json.load(f)

# Update neural-tools environment with PROJECT_PATH
if 'neural-tools' in config['mcpServers']:
    if 'env' not in config['mcpServers']['neural-tools']:
        config['mcpServers']['neural-tools']['env'] = {}

    # Add PROJECT_PATH and PROJECT_NAME
    config['mcpServers']['neural-tools']['env']['PROJECT_PATH'] = "$NEURAL_NOVELIST_PATH"
    config['mcpServers']['neural-tools']['env']['PROJECT_NAME'] = "neural-novelist"

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Updated neural-tools with PROJECT_PATH")
else:
    print("❌ neural-tools not found in MCP config")
    sys.exit(1)
EOF

echo ""
echo -e "${GREEN}✅ MCP configured for neural-novelist!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Restart Claude to apply the new configuration"
echo "2. Navigate to: $NEURAL_NOVELIST_PATH"
echo "3. Neural tools should now detect the correct project"
echo ""
echo -e "${YELLOW}To revert:${NC}"
echo "cp $MCP_CONFIG_PATH.backup.* $MCP_CONFIG_PATH"
echo ""
echo -e "${GREEN}🎯 Configuration complete!${NC}"