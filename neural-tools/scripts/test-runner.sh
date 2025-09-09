#!/bin/bash
# Simple production test runner for Neural Tools

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_TYPE="${1:-unit}"

echo -e "${BLUE}Neural Tools Test Suite${NC}"
echo "========================"
echo "Test type: $TEST_TYPE"
echo ""

case "$TEST_TYPE" in
    "unit")
        echo -e "${BLUE}Running unit tests...${NC}"
        python3 -m pytest tests/unit/ -v --tb=short
        ;;
    "integration")
        if [[ -z "$ANTHROPIC_API_KEY" ]]; then
            echo -e "${RED}Error: ANTHROPIC_API_KEY required for integration tests${NC}"
            exit 1
        fi
        echo -e "${BLUE}Running integration tests...${NC}"
        python3 -m pytest tests/integration/ -v -m integration --tb=short
        ;;
    "performance")
        if [[ -z "$ANTHROPIC_API_KEY" ]]; then
            echo -e "${RED}Error: ANTHROPIC_API_KEY required for performance tests${NC}"
            exit 1
        fi
        echo -e "${BLUE}Running performance benchmarks...${NC}"
        python3 -m pytest tests/performance/ -v -m benchmark --tb=short
        ;;
    "contract")
        if [[ -z "$ANTHROPIC_API_KEY" ]]; then
            echo -e "${RED}Error: ANTHROPIC_API_KEY required for contract tests${NC}"
            exit 1
        fi
        echo -e "${BLUE}Running contract tests...${NC}"
        python3 -m pytest tests/contract/ -v -m contract --tb=short
        ;;
    *)
        echo "Usage: $0 [unit|integration|performance|contract]"
        exit 1
        ;;
esac

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}✅ Tests passed!${NC}"
else
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi