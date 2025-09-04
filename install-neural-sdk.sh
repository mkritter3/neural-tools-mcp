#!/bin/bash
# Neural SDK Local Installation Script
# Installs the Neural SDK locally without needing npm publish

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧠 Neural SDK Local Installation${NC}"
echo "======================================"
echo

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_DIR="$SCRIPT_DIR/neural-sdk"

# Check Node.js version
echo -e "${BLUE}1. Checking Node.js version...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found${NC}"
    echo "Please install Node.js 18+ from: https://nodejs.org"
    exit 1
fi

NODE_VERSION=$(node -v | sed 's/v//')
MAJOR_VERSION=$(echo $NODE_VERSION | cut -d. -f1)

if [ $MAJOR_VERSION -lt 18 ]; then
    echo -e "${RED}❌ Node.js 18+ required${NC}"
    echo "Current version: v$NODE_VERSION"
    echo "Please upgrade from: https://nodejs.org"
    exit 1
fi

echo -e "${GREEN}✅ Node.js v$NODE_VERSION${NC}"

# Check if SDK directory exists
if [ ! -d "$SDK_DIR" ]; then
    echo -e "${RED}❌ Neural SDK source not found at $SDK_DIR${NC}"
    exit 1
fi

# Build the SDK
echo -e "${BLUE}2. Building Neural SDK...${NC}"
cd "$SDK_DIR"

# Install dependencies
echo "  📦 Installing dependencies..."
npm install --silent

# Build TypeScript
echo "  🔨 Compiling TypeScript..."
npm run build --silent

echo -e "${GREEN}✅ SDK built successfully${NC}"

# Create global installation
echo -e "${BLUE}3. Installing globally...${NC}"

# Use npm link to make it globally available
npm link --silent

echo -e "${GREEN}✅ Neural SDK installed globally${NC}"

# Verify installation
echo -e "${BLUE}4. Verifying installation...${NC}"
if command -v neural &> /dev/null; then
    echo -e "${GREEN}✅ 'neural' command available${NC}"
    
    # Test version
    NEURAL_VERSION=$(neural --version 2>/dev/null || echo "unknown")
    echo "   Version: $NEURAL_VERSION"
else
    echo -e "${RED}❌ Installation verification failed${NC}"
    echo "Try running: export PATH=\"\$PATH:$(npm root -g)/../bin\""
    exit 1
fi

echo
echo -e "${GREEN}🎉 Neural SDK Installation Complete!${NC}"
echo
echo -e "${BLUE}📚 Quick Start:${NC}"
echo "  cd your-project"
echo "  neural init"
echo
echo -e "${BLUE}📋 Available Commands:${NC}"
echo "  neural init      - Initialize Neural Tools in current project"
echo "  neural status    - Check system status"
echo "  neural watch     - Start file watching"
echo "  neural stop      - Stop services"
echo "  neural logs      - View logs"
echo "  neural reset     - Reset everything"
echo
echo -e "${YELLOW}💡 Next Steps:${NC}"
echo "  1. Navigate to any project directory"
echo "  2. Run 'neural init' to get started"
echo "  3. The SDK will auto-detect your project and set everything up!"
echo

# Optional: Create a test project
read -p "Create a test project to try Neural SDK? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    TEST_DIR="$HOME/neural-sdk-test"
    echo -e "${BLUE}5. Creating test project...${NC}"
    
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    
    # Create a simple Node.js project
    cat > package.json << EOF
{
  "name": "neural-sdk-test",
  "version": "1.0.0",
  "description": "Test project for Neural SDK",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  }
}
EOF

    cat > index.js << EOF
// Neural SDK Test Project
console.log('Hello from Neural SDK test project!');

function greetUser(name) {
  return \`Welcome \${name} to Neural Tools!\`;
}

function calculateFibonacci(n) {
  if (n <= 1) return n;
  return calculateFibonacci(n - 1) + calculateFibonacci(n - 2);
}

module.exports = { greetUser, calculateFibonacci };
EOF

    cat > README.md << EOF
# Neural SDK Test Project

This is a test project created during Neural SDK installation.

## Files:
- \`package.json\` - Node.js project configuration
- \`index.js\` - Sample JavaScript with functions
- \`README.md\` - This file

## Try Neural SDK:
\`\`\`bash
neural init    # Initialize Neural Tools
neural status  # Check status
neural watch   # Start file watching
\`\`\`

The SDK will automatically:
- Detect this as a Node.js project
- Index all files for semantic search
- Set up MCP integration with Claude Code
- Enable GraphRAG capabilities
\`\`\`
EOF

    echo -e "${GREEN}✅ Test project created at: $TEST_DIR${NC}"
    echo
    echo -e "${BLUE}🚀 Test Neural SDK:${NC}"
    echo "  cd $TEST_DIR"
    echo "  neural init"
    echo
fi

echo -e "${BLUE}🔧 Troubleshooting:${NC}"
echo "  If 'neural' command not found, try:"
echo "  export PATH=\"\$PATH:$(npm root -g)/../bin\""
echo "  source ~/.bashrc  # or ~/.zshrc"
echo