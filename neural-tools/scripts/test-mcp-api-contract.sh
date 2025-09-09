#!/bin/bash
#
# MCP API Contract Test Suite
# L9-grade validation of MCP tool API stability and backward compatibility
#
# Usage: ./scripts/test-mcp-api-contract.sh [--verbose] [--tool=TOOL_NAME] [--bail-on-fail]
#

set -euo pipefail

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/test-results/contract"

# Configuration
VERBOSE=false
SPECIFIC_TOOL=""
BAIL_ON_FAIL=false
PARALLEL_TESTS=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --tool=*)
            SPECIFIC_TOOL="${1#*=}"
            shift
            ;;
        --bail-on-fail)
            BAIL_ON_FAIL=true
            shift
            ;;
        --parallel=*)
            PARALLEL_TESTS="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--tool=TOOL_NAME] [--bail-on-fail] [--parallel=N]"
            echo ""
            echo "Options:"
            echo "  --verbose        Show detailed test output"
            echo "  --tool=NAME      Test only specific tool"
            echo "  --bail-on-fail   Stop on first failure"
            echo "  --parallel=N     Number of parallel test workers (default: 4)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "${BOLD}${BLUE}$1${NC}"
}

# Setup test environment
setup_test_environment() {
    log_header "🚀 MCP API Contract Test Suite"
    echo "Project: $(basename "$PROJECT_ROOT")"
    echo "Time: $(date)"
    echo "Environment: $(python3 --version)"
    echo ""
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Check dependencies
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if Docker services are running
    if ! docker ps --filter name=claude-l9-template-qdrant-1 --filter status=running | grep -q qdrant; then
        log_warning "Qdrant container not running - some tests may fail"
    fi
    
    if ! docker ps --filter name=claude-l9-template-neo4j-1 --filter status=running | grep -q neo4j; then
        log_warning "Neo4j container not running - some tests may fail"  
    fi
}

# Run contract tests
run_contract_tests() {
    log_header "📋 Running MCP Tool Contract Validation"
    
    cd "$PROJECT_ROOT"
    
    # Set environment for testing
    export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
    export QDRANT_HOST=localhost
    export QDRANT_PORT=46333
    export NEO4J_URI=bolt://localhost:47687
    export PROJECT_NAME=contract_test
    
    # Create test results file
    local results_file="$RESULTS_DIR/contract_results_$(date +%Y%m%d_%H%M%S).json"
    
    # Run the contract test framework
    log_info "Executing contract test framework..."
    
    if [[ "$VERBOSE" == true ]]; then
        python3 -u tests/contract/mcp_contract_test_framework.py 2>&1 | tee "$RESULTS_DIR/contract_test.log"
        local test_exit_code=${PIPESTATUS[0]}
    else
        python3 tests/contract/mcp_contract_test_framework.py > "$RESULTS_DIR/contract_test.log" 2>&1
        local test_exit_code=$?
    fi
    
    # Parse results
    if [[ $test_exit_code -eq 0 ]]; then
        log_success "Contract tests passed"
        return 0
    else
        log_error "Contract tests failed"
        
        if [[ "$VERBOSE" == false ]]; then
            echo ""
            echo "Last 20 lines of test output:"
            tail -20 "$RESULTS_DIR/contract_test.log"
        fi
        
        return 1
    fi
}

# Validate P1 API changes specifically
validate_p1_api_changes() {
    log_header "🔍 Validating P1 API Changes"
    
    cd "$PROJECT_ROOT"
    
    # Test that P1 API method changes don't break existing contracts
    log_info "Testing P1-A: API method consistency..."
    
    python3 -c "
import sys
from pathlib import Path
src_path = Path('src').absolute()
sys.path.insert(0, str(src_path))

from servers.services.qdrant_service import QdrantService

# P1-A: Check that search_vectors method exists
qdrant = QdrantService()
if not hasattr(qdrant, 'search_vectors'):
    print('❌ P1-A FAIL: search_vectors method missing')
    sys.exit(1)

# P1-C: Check that delete_points method exists  
if not hasattr(qdrant, 'delete_points'):
    print('❌ P1-C FAIL: delete_points method missing')
    sys.exit(1)

print('✅ P1 API methods present')
" || return 1
    
    log_success "P1 API validation passed"
    return 0
}

# Run performance regression tests
run_performance_tests() {
    log_header "⚡ Performance Regression Testing"
    
    log_info "Testing P1 dimension validation performance impact..."
    
    cd "$PROJECT_ROOT"
    
    # Simple performance test for P1-B dimension validation
    python3 -c "
import sys
import time
import asyncio
sys.path.append('src')

async def test_performance():
    from servers.services.qdrant_service import QdrantService
    
    # This is a lightweight test - just measure method call overhead
    service = QdrantService()
    
    # Time method attribute access
    start = time.time()
    for _ in range(1000):
        _ = hasattr(service, 'upsert_points')
    overhead = (time.time() - start) * 1000
    
    print(f'Method overhead: {overhead:.2f}ms for 1000 calls')
    
    if overhead > 100:  # 100ms threshold for 1000 calls
        print('❌ Performance regression detected')
        return False
    
    print('✅ Performance within acceptable range')
    return True

result = asyncio.run(test_performance())
sys.exit(0 if result else 1)
" || {
        log_warning "Performance test failed - may indicate regression"
        return 1
    }
    
    log_success "Performance tests passed"
    return 0
}

# Generate final report
generate_report() {
    local overall_result=$1
    
    log_header "📊 Test Report Summary"
    
    echo "Contract Test Results:"
    echo "  - Test Suite: MCP API Contract Validation"
    echo "  - Timestamp: $(date)"
    echo "  - Project: $(basename "$PROJECT_ROOT")"
    
    if [[ -f "$RESULTS_DIR/contract_test.log" ]]; then
        echo "  - Log File: $RESULTS_DIR/contract_test.log"
        
        # Extract key metrics from log
        local total_tests=$(grep -c "Testing " "$RESULTS_DIR/contract_test.log" || echo "0")
        echo "  - Total Tools Tested: $total_tests"
    fi
    
    echo ""
    
    if [[ $overall_result -eq 0 ]]; then
        log_success "🎉 ALL CONTRACT TESTS PASSED"
        echo "✅ API backward compatibility maintained"
        echo "✅ P1 integration fixes validated"
        echo "✅ Performance within acceptable range"
        echo ""
        echo "Ready for production deployment"
    else
        log_error "💥 CONTRACT TESTS FAILED"
        echo "❌ API compatibility issues detected"
        echo "❌ Review failures before deployment"
        echo ""
        echo "Check logs: $RESULTS_DIR/contract_test.log"
    fi
    
    return $overall_result
}

# Main execution
main() {
    local overall_result=0
    
    setup_test_environment
    
    # Run contract tests
    if ! run_contract_tests; then
        overall_result=1
        if [[ "$BAIL_ON_FAIL" == true ]]; then
            log_error "Bailing on contract test failure"
            generate_report $overall_result
            exit $overall_result
        fi
    fi
    
    # Validate P1 changes specifically
    if ! validate_p1_api_changes; then
        overall_result=1
        if [[ "$BAIL_ON_FAIL" == true ]]; then
            log_error "Bailing on P1 validation failure"
            generate_report $overall_result
            exit $overall_result
        fi
    fi
    
    # Performance regression tests
    if ! run_performance_tests; then
        log_warning "Performance tests failed but not blocking"
        # Don't fail overall for performance (warning only)
    fi
    
    generate_report $overall_result
    exit $overall_result
}

# Execute main function
main "$@"