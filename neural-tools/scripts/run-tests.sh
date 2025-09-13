#!/bin/bash
# L9 Neural Tools Comprehensive Test Suite
# Implements ADR-0047: CI/CD Validation Framework
# Based on Grok-4 and Gemini-2.5-pro consensus analysis

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURAL_TOOLS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$NEURAL_TOOLS_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧪 Neural Tools Test Suite"
echo "=========================="

# Function to run a test category
run_test_category() {
    local category=$1
    local test_path=$2

    echo -e "\n${YELLOW}Running $category tests...${NC}"

    if [ -d "$test_path" ] || [ -f "$test_path" ]; then
        cd "$NEURAL_TOOLS_DIR"
        if $PYTHON_CMD -m pytest "$test_path" -v --tb=short; then
            echo -e "${GREEN}✅ $category tests passed${NC}"
            return 0
        else
            echo -e "${RED}❌ $category tests failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  No $category tests found at $test_path${NC}"
        return 0
    fi
}

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    local check_cmd=$3

    if eval "$check_cmd" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $service is running on port $port${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  $service not available on port $port (some tests may skip)${NC}"
        return 0
    fi
}

# Check Python version
echo -e "\n📋 Environment Check"
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "Python version: $python_version"

# Export Python path for consistency
export PYTHON_CMD=$(which python3)

# Check required services
echo -e "\n📋 Service Availability"
check_service "Neo4j" 47687 "nc -z localhost 47687"
check_service "Qdrant" 46333 "curl -s http://localhost:46333/health"
check_service "Redis" 46379 "redis-cli -p 46379 ping"

# Install test dependencies
echo -e "\n📦 Installing test dependencies..."
cd "$NEURAL_TOOLS_DIR"
pip install -q pytest pytest-asyncio pytest-cov pytest-timeout

# Run linting
echo -e "\n${YELLOW}Running linting checks...${NC}"
if command -v ruff &> /dev/null; then
    if ruff check src/ --fix; then
        echo -e "${GREEN}✅ Linting passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Linting issues found (non-blocking)${NC}"
    fi
else
    echo "⚠️  Ruff not installed, skipping linting"
fi

# Initialize test results
FAILED_TESTS=()

# Run unit tests
if ! run_test_category "Unit" "tests/unit"; then
    FAILED_TESTS+=("Unit")
fi

# Run integration tests (if services available)
if nc -z localhost 47687 && curl -s http://localhost:46333/health >/dev/null; then
    if ! run_test_category "Integration" "tests/integration"; then
        FAILED_TESTS+=("Integration")
    fi
else
    echo -e "${YELLOW}⚠️  Skipping integration tests (services not available)${NC}"
fi

# Run ADR validation tests
echo -e "\n${YELLOW}Running ADR validation tests...${NC}"
cd "$NEURAL_TOOLS_DIR"

# Test ADR-0043
if [ -f "tests/test_adr_0043.py" ]; then
    if $PYTHON_CMD tests/test_adr_0043.py; then
        echo -e "${GREEN}✅ ADR-0043 validation passed${NC}"
    else
        echo -e "${RED}❌ ADR-0043 validation failed${NC}"
        FAILED_TESTS+=("ADR-0043")
    fi
fi

# Test ADR-0044
if [ -f "tests/test_adr_0044.py" ]; then
    if $PYTHON_CMD tests/test_adr_0044.py; then
        echo -e "${GREEN}✅ ADR-0044 validation passed${NC}"
    else
        echo -e "${RED}❌ ADR-0044 validation failed${NC}"
        FAILED_TESTS+=("ADR-0044")
    fi
fi

# Run deployment validation
echo -e "\n${YELLOW}Running deployment validation...${NC}"
if $PYTHON_CMD scripts/validate-deployment.py; then
    echo -e "${GREEN}✅ Deployment validation passed${NC}"
else
    echo -e "${RED}❌ Deployment validation failed${NC}"
    FAILED_TESTS+=("Deployment")
fi

# Check for breaking changes
echo -e "\n${YELLOW}Checking for breaking changes...${NC}"
if $PYTHON_CMD scripts/check-breaking-changes.py; then
    echo -e "${GREEN}✅ No breaking changes detected${NC}"
else
    echo -e "${YELLOW}⚠️  Breaking changes detected (review required)${NC}"
fi

# Generate test report
echo -e "\n"
echo "======================================"
echo "📊 TEST SUITE RESULTS"
echo "======================================"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo "Ready for deployment to global MCP"
    exit 0
else
    echo -e "${RED}❌ TESTS FAILED:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  • $test"
    done
    echo ""
    echo "Fix failing tests before deployment"
    exit 1
fi