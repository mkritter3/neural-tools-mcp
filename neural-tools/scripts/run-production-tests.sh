#!/bin/bash
"""
Production Test Suite Runner for Neural Tools

Runs comprehensive test suites for production readiness validation:
- Unit tests (fast, mocked dependencies)
- Integration tests (real API calls with ANTHROPIC_API_KEY)
- Performance benchmarks (latency, throughput, memory)
- Contract tests (API schema validation)

Usage:
  ./scripts/run-production-tests.sh [test-type]
  
Test types:
  unit        - Fast unit tests only (default for CI)
  integration - Real API integration tests (requires API key)
  performance - Performance benchmarks (requires API key)
  contract    - API contract validation (requires API key)
  all         - All test suites (requires API key)

Environment:
  ANTHROPIC_API_KEY - Required for integration/performance/contract tests
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_TYPE="${1:-unit}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$PROJECT_ROOT/tests"

echo -e "${BLUE}🧪 Neural Tools Production Test Suite${NC}"
echo -e "${BLUE}======================================${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Test type: $TEST_TYPE"
echo ""

# Check if we're in the right directory
if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    echo -e "${RED}❌ Error: Not in neural-tools directory${NC}"
    echo "Please run from neural-tools directory"
    exit 1
fi

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: python3 not found${NC}"
    exit 1
fi

# Check pytest installation
if ! python3 -c "import pytest" &> /dev/null; then
    echo -e "${RED}❌ Error: pytest not installed${NC}"
    echo "Run: pip install pytest pytest-asyncio"
    exit 1
fi

# Function to check API key for integration tests
check_api_key() {
    if [[ -z "$ANTHROPIC_API_KEY" ]]; then
        echo -e "${YELLOW}⚠️  Warning: ANTHROPIC_API_KEY not set${NC}"
        echo "   Integration/performance/contract tests will be skipped"
        echo "   Set ANTHROPIC_API_KEY environment variable to run real API tests"
        return 1
    fi
    
    # Basic API key format validation
    if [[ ! "$ANTHROPIC_API_KEY" =~ ^sk-.+ ]] && [[ ${#ANTHROPIC_API_KEY} -lt 20 ]]; then
        echo -e "${YELLOW}⚠️  Warning: API key format looks invalid${NC}"
        echo "   Expected format: sk-... or longer than 20 characters"
        return 1
    fi
    
    return 0
}

# Function to run specific test suite
run_test_suite() {
    local suite_name="$1"
    local pytest_args="$2"
    local description="$3"
    
    echo -e "${BLUE}🔍 Running $suite_name tests${NC}"
    echo "Description: $description"
    echo "Command: python3 -m pytest $pytest_args"
    echo ""
    
    if python3 -m pytest $pytest_args; then
        echo -e "${GREEN}✅ $suite_name tests passed${NC}"
        return 0
    else
        echo -e "${RED}❌ $suite_name tests failed${NC}"
        return 1
    fi
}

# Change to project directory
cd "$PROJECT_ROOT"

# Track test results
FAILED_SUITES=()

case "$TEST_TYPE" in
    "unit")
        echo -e "${BLUE}📋 Running unit tests (fast, mocked)${NC}"
        echo ""
        
        # Run unit tests
        if run_test_suite "Unit" \
            "tests/unit/ -v --tb=short -x" \
            "Fast unit tests with mocked dependencies"; then
            echo -e "${GREEN}🎉 Unit tests completed successfully${NC}"
        else
            FAILED_SUITES+=("unit")
        fi
        ;;
        
    "integration")
        echo -e "${BLUE}🌐 Running integration tests (real API)${NC}"
        echo ""
        
        if ! check_api_key; then
            echo -e "${RED}❌ Cannot run integration tests without valid ANTHROPIC_API_KEY${NC}"
            exit 1
        fi
        
        # Run integration tests with real API
        if run_test_suite "Integration" \
            "tests/integration/ -v -m integration --tb=short" \
            "Integration tests with real Anthropic API calls"; then
            echo -e "${GREEN}🎉 Integration tests completed successfully${NC}"
        else
            FAILED_SUITES+=("integration")
        fi
        ;;
        
    "performance")
        echo -e "${BLUE}⚡ Running performance benchmarks${NC}"
        echo ""
        
        if ! check_api_key; then
            echo -e "${RED}❌ Cannot run performance tests without valid ANTHROPIC_API_KEY${NC}"
            exit 1
        fi
        
        # Run performance benchmarks
        if run_test_suite "Performance" \
            "tests/performance/ -v -m benchmark --tb=short --durations=10" \
            "Performance benchmarks with real API latency measurements"; then
            echo -e "${GREEN}🎉 Performance benchmarks completed successfully${NC}"
        else
            FAILED_SUITES+=("performance")
        fi
        ;;
        
    "contract")
        echo -e "${BLUE}📄 Running contract validation tests${NC}"
        echo ""
        
        if ! check_api_key; then
            echo -e "${RED}❌ Cannot run contract tests without valid ANTHROPIC_API_KEY${NC}"
            exit 1
        fi
        
        # Run contract tests
        if run_test_suite "Contract" \
            "tests/contract/ -v -m contract --tb=short" \
            "API contract validation against Anthropic endpoints"; then
            echo -e "${GREEN}🎉 Contract tests completed successfully${NC}"
        else
            FAILED_SUITES+=("contract")
        fi
        ;;
        
    "all")
        echo -e "${BLUE}🚀 Running comprehensive test suite${NC}"
        echo ""
        
        # Check API key for comprehensive testing
        if ! check_api_key; then
            echo -e "${RED}❌ Cannot run comprehensive tests without valid ANTHROPIC_API_KEY${NC}"
            exit 1
        fi
        
        # Run all test suites in order
        echo -e "${YELLOW}📋 Phase 1: Unit Tests${NC}"
        if ! run_test_suite "Unit" \
            "tests/unit/ -v --tb=short" \
            "Unit tests with mocked dependencies"; then
            FAILED_SUITES+=("unit")
        fi
        
        echo -e "${YELLOW}📋 Phase 2: Integration Tests${NC}" 
        if ! run_test_suite "Integration" \
            "tests/integration/ -v -m integration --tb=short" \
            "Integration tests with real API"; then
            FAILED_SUITES+=("integration")
        fi
        
        echo -e "${YELLOW}📋 Phase 3: Performance Benchmarks${NC}"
        if ! run_test_suite "Performance" \
            "tests/performance/ -v -m benchmark --tb=short --durations=5" \
            "Performance benchmarks"; then
            FAILED_SUITES+=("performance")
        fi
        
        echo -e "${YELLOW}📋 Phase 4: Contract Validation${NC}"
        if ! run_test_suite "Contract" \
            "tests/contract/ -v -m contract --tb=short" \
            "API contract validation"; then
            FAILED_SUITES+=("contract")
        fi
        ;;
        
    *)
        echo -e "${RED}❌ Error: Unknown test type '$TEST_TYPE'${NC}"
        echo ""
        echo "Valid test types:"
        echo "  unit        - Unit tests only (fast, default)"
        echo "  integration - Integration tests (requires API key)" 
        echo "  performance - Performance benchmarks (requires API key)"
        echo "  contract    - Contract validation (requires API key)"
        echo "  all         - All test suites (requires API key)"
        exit 1
        ;;
esac

# Final results
echo ""
echo -e "${BLUE}📊 Test Results Summary${NC}"
echo -e "${BLUE}======================${NC}"

if [[ ${#FAILED_SUITES[@]} -eq 0 ]]; then
    echo -e "${GREEN}🎉 All test suites passed successfully!${NC}"
    echo -e "${GREEN}✅ System is production ready${NC}"
    exit 0
else
    echo -e "${RED}❌ Failed test suites: ${FAILED_SUITES[*]}${NC}"
    echo -e "${RED}🚫 System is NOT production ready${NC}"
    echo ""
    echo "Please fix failing tests before deployment"
    exit 1
fi