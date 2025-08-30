#!/bin/bash
# L9 Neural Flow - One-Command Installer
# Clean, simple installation for the L9 template

set -e

# Get absolute path to template directory
TEMPLATE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔮 L9 Neural Flow Installation"
echo "=============================="
echo "📁 Template: $TEMPLATE_DIR"
echo

# 1. Check Python
echo -e "${BLUE}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found${NC}"

# 2. Check Docker
echo -e "${BLUE}Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker not found. Please install Docker${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker found${NC}"

# 3. Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
if [[ -f "$TEMPLATE_DIR/core/requirements.txt" ]]; then
    pip3 install -r "$TEMPLATE_DIR/core/requirements.txt" -q
else
    pip3 install -q qdrant-client chromadb sentence-transformers tiktoken onnxruntime aiohttp httpx pyyaml python-dotenv
fi

# 4. Set up local bin directory
echo -e "${BLUE}Setting up commands...${NC}"
mkdir -p ~/.local/bin

# 5. Create mcp command that uses this specific template
cat > ~/.local/bin/mcp << EOF
#!/bin/bash
# MCP wrapper for L9 Neural Flow
L9_TEMPLATE_DIR="$TEMPLATE_DIR"

case "\$1" in
    add)
        shift
        exec "\$L9_TEMPLATE_DIR/mcp-add" "\$@"
        ;;
    *)
        if command -v claude &> /dev/null; then
            exec claude mcp "\$@"
        else
            echo "Claude Code not found. Install Claude Code first."
            exit 1
        fi
        ;;
esac
EOF
chmod +x ~/.local/bin/mcp

# 6. Create neural-flow command
cat > ~/.local/bin/neural-flow << EOF
#!/bin/bash
L9_TEMPLATE_DIR="$TEMPLATE_DIR"
exec "\$L9_TEMPLATE_DIR/scripts/neural-flow.sh" "\$@"
EOF
chmod +x ~/.local/bin/neural-flow

echo -e "${GREEN}✅ Installation complete!${NC}"
echo
echo "🎯 Quick Start:"
echo "  1. Add to your PATH (if not already):"
echo "     export PATH=\"\$HOME/.local/bin:\$PATH\""
echo
echo "  2. Add L9 to any project:"
echo "     cd /path/to/project"
echo "     mcp add"
echo
echo "  3. Or add globally for all projects:"
echo "     mcp add --scope user"
echo
echo "📚 Documentation: docs/QUICK-START.md"
echo "🔧 Template location: $TEMPLATE_DIR"