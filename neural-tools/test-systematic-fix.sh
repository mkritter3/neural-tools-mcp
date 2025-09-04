#!/bin/bash
# Test script to verify systematic fix for project indexing
# This confirms changes persist in both dev and production modes

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧪 Testing Systematic Fix for Project Indexing"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if tool exists
check_tool_exists() {
    local mode=$1
    echo "Checking if project_indexer tool exists in $mode mode..."
    
    docker exec -it ${PROJECT_NAME:-default}-neural python3 -c "
import sys
sys.path.append('/app')
try:
    from src.servers.neural_server_stdio import NeuralMCPServer
    server = NeuralMCPServer()
    tools = [t.name for t in server.list_tools()]
    if 'project_indexer' in tools:
        print('✅ project_indexer tool found!')
        sys.exit(0)
    else:
        print('❌ project_indexer tool NOT found')
        print('Available tools:', tools)
        sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
" 2>/dev/null
    
    return $?
}

# Test 1: Production Mode (Rebuilt)
echo -e "${YELLOW}Test 1: Production Mode with Rebuild${NC}"
echo "--------------------------------------"
echo "Building and starting in production mode..."
./build-and-run.sh --rebuild --build-only >/dev/null 2>&1
./build-and-run.sh >/dev/null 2>&1

# Wait for services
sleep 10

if check_tool_exists "production"; then
    echo -e "${GREEN}✅ Test 1 PASSED: Tool exists in production build${NC}"
else
    echo -e "${RED}❌ Test 1 FAILED: Tool missing in production build${NC}"
    exit 1
fi

# Clean up
docker-compose -f config/docker-compose.neural-tools.yml down >/dev/null 2>&1

echo ""

# Test 2: Development Mode (Live Mount)
echo -e "${YELLOW}Test 2: Development Mode with Live Mount${NC}"
echo "-----------------------------------------"
echo "Starting in development mode..."
./build-and-run.sh --dev >/dev/null 2>&1

# Wait for services
sleep 10

if check_tool_exists "development"; then
    echo -e "${GREEN}✅ Test 2 PASSED: Tool exists in development mode${NC}"
else
    echo -e "${RED}❌ Test 2 FAILED: Tool missing in development mode${NC}"
    exit 1
fi

echo ""

# Test 3: Live Code Update in Dev Mode
echo -e "${YELLOW}Test 3: Live Code Update Test${NC}"
echo "------------------------------"
echo "Adding test comment to verify live updates..."

# Add a test comment
cp src/servers/neural_server_stdio.py src/servers/neural_server_stdio.py.bak
echo "# TEST_MARKER_12345" >> src/servers/neural_server_stdio.py

# Check if change is reflected
docker exec -it ${PROJECT_NAME:-default}-neural bash -c "grep TEST_MARKER_12345 /app/src/servers/neural_server_stdio.py" >/dev/null 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Test 3 PASSED: Live code updates work${NC}"
else
    echo -e "${RED}❌ Test 3 FAILED: Live code updates not working${NC}"
fi

# Restore original file
mv src/servers/neural_server_stdio.py.bak src/servers/neural_server_stdio.py

# Clean up
docker-compose -f config/docker-compose.neural-tools.yml down >/dev/null 2>&1

echo ""

# Test 4: Persistence After Restart
echo -e "${YELLOW}Test 4: Persistence After Container Restart${NC}"
echo "-------------------------------------------"
echo "Rebuilding in production mode..."
./build-and-run.sh --rebuild >/dev/null 2>&1

# Wait for services
sleep 10

echo "Stopping containers..."
docker-compose -f config/docker-compose.neural-tools.yml stop >/dev/null 2>&1

echo "Restarting containers..."
docker-compose -f config/docker-compose.neural-tools.yml start >/dev/null 2>&1

# Wait for services
sleep 10

if check_tool_exists "production-restart"; then
    echo -e "${GREEN}✅ Test 4 PASSED: Tool persists after restart${NC}"
else
    echo -e "${RED}❌ Test 4 FAILED: Tool lost after restart${NC}"
    exit 1
fi

# Final cleanup
docker-compose -f config/docker-compose.neural-tools.yml down >/dev/null 2>&1

echo ""
echo -e "${GREEN}=============================================="
echo "🎉 All Systematic Tests PASSED!"
echo "=============================================="
echo ""
echo "Summary:"
echo "✅ Production build includes project_indexer tool"
echo "✅ Development mode mounts source code correctly"
echo "✅ Live code updates work in development mode"
echo "✅ Changes persist after container restart"
echo ""
echo "The systematic fix is complete and verified!"