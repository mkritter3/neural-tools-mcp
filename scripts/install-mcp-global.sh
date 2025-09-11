#!/bin/bash

# Install mcp-add as a global command
# This script installs the mcp-add command globally

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running with sufficient permissions
if [ "$EUID" -ne 0 ] && [ ! -w "/usr/local/bin" ]; then 
    echo -e "${YELLOW}⚠️  This script needs permission to write to /usr/local/bin${NC}"
    echo "Please run with: sudo $0"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MCP_SCRIPT="$SCRIPT_DIR/mcp-add.sh"

if [ ! -f "$MCP_SCRIPT" ]; then
    echo -e "${RED}Error: mcp-add.sh not found at $MCP_SCRIPT${NC}"
    exit 1
fi

echo -e "${BLUE}🚀 Installing mcp-add globally...${NC}"

# Create a wrapper script that knows where the MCP base is
cat > /tmp/mcp-add << 'EOF'
#!/bin/bash

# Global mcp-add wrapper
# This script calls the actual mcp-add.sh with the correct MCP_BASE_PATH

# MCP installation path (update this if you move the L9 template)
export MCP_BASE_PATH="/Users/mkr/local-coding/claude-l9-template"

# Check if MCP_BASE_PATH exists
if [ ! -d "$MCP_BASE_PATH" ]; then
    echo "Error: MCP base path not found at $MCP_BASE_PATH"
    echo "Please update the MCP_BASE_PATH in /usr/local/bin/mcp-add"
    exit 1
fi

# Call the actual script
exec "$MCP_BASE_PATH/scripts/mcp-add.sh" "$@"
EOF

# Install the wrapper
if [ -w "/usr/local/bin" ]; then
    mv /tmp/mcp-add /usr/local/bin/mcp-add
    chmod +x /usr/local/bin/mcp-add
    echo -e "${GREEN}✅ mcp-add installed to /usr/local/bin/mcp-add${NC}"
else
    sudo mv /tmp/mcp-add /usr/local/bin/mcp-add
    sudo chmod +x /usr/local/bin/mcp-add
    echo -e "${GREEN}✅ mcp-add installed to /usr/local/bin/mcp-add${NC}"
fi

# Test the installation
if command -v mcp-add &> /dev/null; then
    echo -e "${GREEN}✅ mcp-add is now available globally!${NC}"
    echo ""
    echo -e "${BLUE}Usage:${NC}"
    echo "  mcp-add              # Auto-detect project in current directory"
    echo "  mcp-add /path/to/project  # Add to specific project"
    echo "  mcp-add . my-app     # Current directory with custom name"
    echo ""
    echo -e "${YELLOW}Try it out:${NC}"
    echo "  cd /path/to/your/project"
    echo "  mcp-add"
else
    echo -e "${RED}Warning: mcp-add not found in PATH${NC}"
    echo "You may need to add /usr/local/bin to your PATH"
fi