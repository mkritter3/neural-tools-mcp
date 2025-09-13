#!/bin/bash
# CI/CD Test Suite for Neural Tools MCP Server
# Implements ADR-0047: CI/CD Validation Framework
# Based on Grok-4 and Gemini-2.5-pro consensus

set -e  # Exit on error

echo "🚀 Neural Tools CI/CD Test Suite"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2

    echo -e "${YELLOW}Running: $test_name${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if python3 "$test_file" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        echo "    Error output:"
        tail -n 10 /tmp/test_output.log | sed 's/^/    /'
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    echo ""
}

# Check if running from correct directory
if [ ! -f "src/neural_mcp/neural_server_stdio.py" ]; then
    echo "❌ Error: Must run from neural-tools directory"
    exit 1
fi

# Phase 1: Unit Tests
echo "========================================="
echo "📋 Phase 1: Unit Tests"
echo "========================================="
echo ""

if [ -f "tests/unit/test_mcp_protocol.py" ]; then
    run_test "MCP Protocol Compliance" "tests/unit/test_mcp_protocol.py"
fi

# Phase 2: Smoke Tests
echo "========================================="
echo "🔥 Phase 2: Smoke Tests"
echo "========================================="
echo ""

if [ -f "tests/smoke/test_basic_connectivity.py" ]; then
    run_test "Basic Service Connectivity" "tests/smoke/test_basic_connectivity.py"
fi

if [ -f "tests/smoke/test_mcp_startup.py" ]; then
    run_test "MCP Server Startup" "tests/smoke/test_mcp_startup.py"
fi

# Phase 3: Integration Tests
echo "========================================="
echo "🔗 Phase 3: Integration Tests"
echo "========================================="
echo ""

if [ -f "tests/integration/test_container_health.py" ]; then
    run_test "Container Health Check" "tests/integration/test_container_health.py"
fi

if [ -f "tests/integration/test_graphrag_e2e.py" ]; then
    run_test "GraphRAG End-to-End" "tests/integration/test_graphrag_e2e.py"
fi

if [ -f "tests/integration/test_data_integrity.py" ]; then
    run_test "Neo4j/Qdrant Data Integrity" "tests/integration/test_data_integrity.py"
fi

# Real-world issue tests
if [ -f "tests/integration/test_real_world_issues.py" ]; then
    run_test "Real-World Issue Tests" "tests/integration/test_real_world_issues.py"
fi

# Comprehensive test for all 22 tools (OLD - direct imports)
if [ -f "tests/integration/test_all_22_tools.py" ]; then
    run_test "All 22 Tools Direct Import Test" "tests/integration/test_all_22_tools.py"
fi

# L9 Standard End-to-End tests (NEW - subprocess)
if [ -f "tests/integration/test_e2e_all_tools.py" ]; then
    echo -e "${YELLOW}Running: End-to-End All Tools Test (L9 Standard)${NC}"
    if python3 -m pytest tests/integration/test_e2e_all_tools.py -v > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✅ PASSED: End-to-End All Tools Test${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED: End-to-End All Tools Test${NC}"
        tail -n 20 /tmp/test_output.log | sed 's/^/    /'
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
fi

# Failure mode and invalid input tests
if [ -f "tests/integration/test_failure_modes.py" ]; then
    echo -e "${YELLOW}Running: Failure Mode Tests (L9 Standard)${NC}"
    if python3 -m pytest tests/integration/test_failure_modes.py -v > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✅ PASSED: Failure Mode Tests${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}❌ FAILED: Failure Mode Tests${NC}"
        tail -n 20 /tmp/test_output.log | sed 's/^/    /'
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
fi

# Phase 4: Performance Tests (if implemented)
if [ -d "tests/performance" ]; then
    echo "========================================="
    echo "⚡ Phase 4: Performance Tests"
    echo "========================================="
    echo ""

    # Run specific performance tests (skip haiku which has dependencies)
    if [ -f "tests/performance/test_mcp_performance.py" ]; then
        run_test "MCP Server Performance" "tests/performance/test_mcp_performance.py"
    fi

    if [ -f "tests/performance/test_file_watcher_performance.py" ]; then
        run_test "File Watcher Performance" "tests/performance/test_file_watcher_performance.py"
    fi
fi

# Phase 5: Rollback Tests (if implemented)
if [ -d "tests/rollback" ]; then
    echo "========================================="
    echo "↩️  Phase 5: Rollback Tests"
    echo "========================================="
    echo ""

    for test_file in tests/rollback/*.py; do
        if [ -f "$test_file" ]; then
            test_name=$(basename "$test_file" .py | sed 's/_/ /g' | sed 's/test //')
            run_test "$test_name" "$test_file"
        fi
    done
fi

# Summary
echo "========================================="
echo "📊 Test Summary"
echo "========================================="
echo ""
echo "Total Tests:  $TOTAL_TESTS"
echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All CI/CD tests passed!${NC}"
    echo "Ready for deployment to global MCP."
    exit 0
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    echo "Please fix the failing tests before deployment."
    exit 1
fi