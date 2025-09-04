#!/bin/bash
# Neural SDK Simple Installation Script
# Installs a simplified JavaScript version of Neural SDK

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🧠 Neural SDK Simple Installation${NC}"
echo "===================================="
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_SIMPLE_DIR="$SCRIPT_DIR/neural-sdk-simple"

# Check Node.js
echo -e "${BLUE}1. Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found. Please install from: https://nodejs.org${NC}"
    exit 1
fi

NODE_VERSION=$(node -v | sed 's/v//')
echo -e "${GREEN}✅ Node.js v$NODE_VERSION${NC}"

# Check if simplified SDK exists
if [ ! -d "$SDK_SIMPLE_DIR" ]; then
    echo -e "${RED}❌ Neural SDK Simple not found at $SDK_SIMPLE_DIR${NC}"
    exit 1
fi

# Install dependencies
echo -e "${BLUE}2. Installing dependencies...${NC}"
cd "$SDK_SIMPLE_DIR"
npm install --silent

# Make executable
chmod +x neural.js

# Install globally using npm link
echo -e "${BLUE}3. Installing globally...${NC}"
npm link --silent

# Verify installation
echo -e "${BLUE}4. Verifying installation...${NC}"
if command -v neural &> /dev/null; then
    echo -e "${GREEN}✅ 'neural' command available${NC}"
    
    # Test version
    neural --version 2>/dev/null && echo -e "${GREEN}✅ Version check passed${NC}"
else
    echo -e "${RED}❌ Installation failed${NC}"
    exit 1
fi

echo
echo -e "${GREEN}🎉 Neural SDK Simple Installation Complete!${NC}"
echo
echo -e "${BLUE}📚 Available Commands:${NC}"
echo "  neural init      - Initialize Neural Tools in current project"  
echo "  neural status    - Check system status"
echo "  neural test      - Test MCP connection"
echo
echo -e "${YELLOW}🔧 Prerequisites:${NC}"
echo "  1. Make sure Docker is running"
echo "  2. Make sure neural-multi-project-final container exists"
echo "     (Run your existing container setup first)"
echo
echo -e "${BLUE}🚀 Quick Test:${NC}"
echo "  cd /path/to/any/project"
echo "  neural init"
echo
echo -e "${YELLOW}💡 This simplified version works with your existing container!${NC}"