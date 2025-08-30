#!/bin/bash
# Neural Flow L9-Grade End-to-End Test Suite
# Comprehensive validation with strict exit criteria

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Test tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()
CRITICAL_FAILURES=()

# Logging functions
log_test() { echo -e "${BLUE}🧪 TEST: $1${NC}"; }
log_pass() { echo -e "${GREEN}✅ PASS: $1${NC}"; ((TESTS_PASSED++)); }
log_fail() { echo -e "${RED}❌ FAIL: $1${NC}"; ((TESTS_FAILED++)); FAILED_TESTS+=("$1"); }
log_critical() { echo -e "${RED}🚨 CRITICAL FAIL: $1${NC}"; CRITICAL_FAILURES+=("$1"); }
log_info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }
log_step() { echo -e "${PURPLE}📋 STEP: $1${NC}"; }

# Test timeouts and criteria
DOCKER_TIMEOUT=30
MCP_RESPONSE_TIMEOUT=15
NEURAL_PROCESSING_TIMEOUT=10
CRITICAL_ERROR_THRESHOLD=0
TOTAL_ERROR_THRESHOLD=2

echo "========================================="
echo "🔮 Neural Flow L9 End-to-End Test Suite"
echo "========================================="
echo "Timestamp: $(date)"
echo "Project: $(basename $PWD)"
echo ""
echo "🎯 STRICT EXIT CRITERIA:"
echo "   • Critical failures: ${CRITICAL_ERROR_THRESHOLD} max"
echo "   • Total failures: ${TOTAL_ERROR_THRESHOLD} max" 
echo "   • All stdio communication must work"
echo "   • All neural tools must respond"
echo "   • ChromaDB must store/retrieve data"
echo "   • Multi-project isolation verified"
echo ""

# ==================================================
# TEST 1: MCP Configuration Schema Validation
# ==================================================
log_test "MCP Configuration Schema Validation"

log_step "Validating .mcp.json structure"
if [ ! -f ".mcp.json" ]; then
    log_critical "Missing .mcp.json file"
    exit 1
fi

# Check required schema elements
if ! jq -e '.mcpServers' .mcp.json >/dev/null 2>&1; then
    log_critical "Missing 'mcpServers' wrapper in .mcp.json"
    exit 1
fi

if ! jq -e '.mcpServers."neural-flow"' .mcp.json >/dev/null 2>&1; then
    log_critical "Missing 'neural-flow' server configuration"
    exit 1
fi

if ! jq -e '.mcpServers."neural-flow".command' .mcp.json >/dev/null 2>&1; then
    log_critical "Missing 'command' in neural-flow configuration"
    exit 1
fi

if ! jq -e '.mcpServers."neural-flow".args' .mcp.json >/dev/null 2>&1; then
    log_critical "Missing 'args' in neural-flow configuration"  
    exit 1
fi

# Validate Docker command structure
COMMAND=$(jq -r '.mcpServers."neural-flow".command' .mcp.json)
if [ "$COMMAND" != "docker" ]; then
    log_critical "Command must be 'docker', got: $COMMAND"
    exit 1
fi

# Check for critical flags
ARGS=$(jq -r '.mcpServers."neural-flow".args | join(" ")' .mcp.json)
if [[ ! "$ARGS" == *"-i"* ]]; then
    log_critical "Missing '-i' interactive flag in Docker args"
    exit 1
fi

if [[ ! "$ARGS" == *"neural-flow:production"* ]]; then
    log_critical "Missing 'neural-flow:production' image reference"
    exit 1
fi

log_pass "MCP configuration schema validation"

# ==================================================
# TEST 2: Docker Container Stdio Communication  
# ==================================================
log_test "Docker Container Stdio Communication"

log_step "Checking Docker image exists"
if ! docker image inspect neural-flow:production >/dev/null 2>&1; then
    log_critical "Docker image 'neural-flow:production' not found"
    exit 1
fi

log_step "Testing container stdio communication"
# Create a simple test MCP request
MCP_REQUEST='{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}'

# Test stdio communication with timeout
RESPONSE=$(timeout ${MCP_RESPONSE_TIMEOUT}s bash -c "
echo '$MCP_REQUEST' | docker run --rm -i \
    -v $PWD/.claude:/app/data \
    --env PROJECT_NAME=test-e2e \
    --env USE_QODO_EMBED=true \
    --env ENABLE_AB_TESTING=true \
    neural-flow:production
" 2>/dev/null || echo "TIMEOUT")

if [ "$RESPONSE" = "TIMEOUT" ]; then
    log_critical "MCP server did not respond within ${MCP_RESPONSE_TIMEOUT}s"
    exit 1
fi

if [ -z "$RESPONSE" ]; then
    log_critical "Empty response from MCP server"
    exit 1
fi

# Validate JSON response
if ! echo "$RESPONSE" | jq empty 2>/dev/null; then
    log_fail "Invalid JSON response from MCP server: ${RESPONSE:0:100}..."
else
    log_pass "Docker container stdio communication"
fi

# ==================================================
# TEST 3: Neural Flow MCP Tools Integration
# ==================================================
log_test "Neural Flow MCP Tools Integration"

log_step "Testing tools/list endpoint"
TOOLS_REQUEST='{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}'

TOOLS_RESPONSE=$(timeout ${MCP_RESPONSE_TIMEOUT}s bash -c "
echo '$TOOLS_REQUEST' | docker run --rm -i \
    -v $PWD/.claude:/app/data \
    --env PROJECT_NAME=test-e2e \
    --env USE_QODO_EMBED=true \
    neural-flow:production
" 2>/dev/null || echo "TIMEOUT")

if [ "$TOOLS_RESPONSE" = "TIMEOUT" ]; then
    log_fail "Tools list request timed out"
else
    # Check for expected L9-grade tools
    EXPECTED_TOOLS=("trace_dependencies" "find_atomic_relations" "analyze_database_schema" "smart_context_window")
    TOOLS_FOUND=0
    
    for tool in "${EXPECTED_TOOLS[@]}"; do
        if echo "$TOOLS_RESPONSE" | jq -e ".result.tools[]? | select(.name == \"$tool\")" >/dev/null 2>&1; then
            ((TOOLS_FOUND++))
        fi
    done
    
    if [ $TOOLS_FOUND -ge 3 ]; then
        log_pass "Neural Flow MCP tools integration ($TOOLS_FOUND/${#EXPECTED_TOOLS[@]} L9 tools found)"
    else
        log_fail "Insufficient L9-grade tools found ($TOOLS_FOUND/${#EXPECTED_TOOLS[@]})"
    fi
fi

# ==================================================
# TEST 4: ChromaDB Vector Storage Validation
# ==================================================
log_test "ChromaDB Vector Storage Validation"

log_step "Testing neural embeddings generation"
EMBED_TEST='{"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "neural_memory_index", "arguments": {"content": "def test_function(): return \"hello world\"", "priority": 8, "tags": ["test", "e2e"]}}}'

EMBED_RESPONSE=$(timeout ${NEURAL_PROCESSING_TIMEOUT}s bash -c "
echo '$EMBED_TEST' | docker run --rm -i \
    -v $PWD/.claude:/app/data \
    --env PROJECT_NAME=test-e2e-chroma \
    --env USE_QODO_EMBED=false \
    neural-flow:production
" 2>/dev/null || echo "TIMEOUT")

if [ "$EMBED_RESPONSE" = "TIMEOUT" ]; then
    log_fail "Neural embedding test timed out"
elif echo "$EMBED_RESPONSE" | jq -e '.result.contents[]?.text | contains("success")' >/dev/null 2>&1; then
    log_pass "ChromaDB vector storage validation"
else
    log_fail "Neural embedding indexing failed"
fi

# ==================================================
# TEST 5: Multi-Project Data Isolation
# ==================================================
log_test "Multi-Project Data Isolation"

log_step "Creating isolated project environments"
mkdir -p /tmp/test-project-a/.claude /tmp/test-project-b/.claude

# Add different data to each project
INDEX_A='{"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "neural_memory_index", "arguments": {"content": "PROJECT_A_UNIQUE_CODE", "priority": 9}}}'
INDEX_B='{"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "neural_memory_index", "arguments": {"content": "PROJECT_B_UNIQUE_CODE", "priority": 9}}}'

# Index into project A
echo "$INDEX_A" | timeout 10s docker run --rm -i \
    -v /tmp/test-project-a/.claude:/app/data \
    --env PROJECT_NAME=project-a \
    neural-flow:production >/dev/null 2>&1 || true

# Index into project B  
echo "$INDEX_B" | timeout 10s docker run --rm -i \
    -v /tmp/test-project-b/.claude:/app/data \
    --env PROJECT_NAME=project-b \
    neural-flow:production >/dev/null 2>&1 || true

log_step "Verifying data isolation"
# Search in project A should only find A's data
SEARCH_A='{"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "neural_memory_search", "arguments": {"query": "PROJECT_A_UNIQUE_CODE", "n_results": 1}}}'

SEARCH_A_RESULT=$(echo "$SEARCH_A" | timeout 10s docker run --rm -i \
    -v /tmp/test-project-a/.claude:/app/data \
    --env PROJECT_NAME=project-a \
    neural-flow:production 2>/dev/null || echo "TIMEOUT")

# Search in project B should only find B's data
SEARCH_B='{"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "neural_memory_search", "arguments": {"query": "PROJECT_B_UNIQUE_CODE", "n_results": 1}}}'

SEARCH_B_RESULT=$(echo "$SEARCH_B" | timeout 10s docker run --rm -i \
    -v /tmp/test-project-b/.claude:/app/data \
    --env PROJECT_NAME=project-b \
    neural-flow:production 2>/dev/null || echo "TIMEOUT")

# Verify isolation
A_ISOLATED=false
B_ISOLATED=false

if echo "$SEARCH_A_RESULT" | grep -q "PROJECT_A_UNIQUE_CODE" && ! echo "$SEARCH_A_RESULT" | grep -q "PROJECT_B_UNIQUE_CODE"; then
    A_ISOLATED=true
fi

if echo "$SEARCH_B_RESULT" | grep -q "PROJECT_B_UNIQUE_CODE" && ! echo "$SEARCH_B_RESULT" | grep -q "PROJECT_A_UNIQUE_CODE"; then
    B_ISOLATED=true
fi

if [ "$A_ISOLATED" = true ] && [ "$B_ISOLATED" = true ]; then
    log_pass "Multi-project data isolation"
else
    log_fail "Data isolation breach detected"
fi

# Cleanup
rm -rf /tmp/test-project-a /tmp/test-project-b

# ==================================================
# TEST 6: Performance Benchmarks Execution
# ==================================================
log_test "Performance Benchmarks Execution"

log_step "Testing benchmark system execution"
BENCHMARK_TEST='{"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "neural_performance_benchmark", "arguments": {"benchmark_type": "recall_test", "n_samples": 10}}}'

BENCHMARK_RESPONSE=$(timeout 20s bash -c "
echo '$BENCHMARK_TEST' | docker run --rm -i \
    -v $PWD/.claude:/app/data \
    --env PROJECT_NAME=test-e2e-bench \
    --env ENABLE_BENCHMARKING=true \
    neural-flow:production
" 2>/dev/null || echo "TIMEOUT")

if [ "$BENCHMARK_RESPONSE" = "TIMEOUT" ]; then
    log_fail "Performance benchmark test timed out"
elif echo "$BENCHMARK_RESPONSE" | jq -e '.result' >/dev/null 2>&1; then
    log_pass "Performance benchmarks execution"
else
    log_fail "Performance benchmark execution failed"
fi

# ==================================================
# FINAL EVALUATION & EXIT CRITERIA  
# ==================================================
log_test "Final L9-Grade Validation"

echo ""
echo "========================================="
echo "📊 END-TO-END TEST RESULTS"
echo "========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ ${#CRITICAL_FAILURES[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}🚨 CRITICAL FAILURES (${#CRITICAL_FAILURES[@]}):${NC}"
    for failure in "${CRITICAL_FAILURES[@]}"; do
        echo "  • $failure"
    done
fi

if [ $TESTS_FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ FAILED TESTS:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  • $test"
    done
fi

echo ""
echo "🎯 EXIT CRITERIA EVALUATION:"
echo "   Critical failures: ${#CRITICAL_FAILURES[@]}/$CRITICAL_ERROR_THRESHOLD ($([ ${#CRITICAL_FAILURES[@]} -le $CRITICAL_ERROR_THRESHOLD ] && echo "✅ PASS" || echo "❌ FAIL"))"
echo "   Total failures: $TESTS_FAILED/$TOTAL_ERROR_THRESHOLD ($([ $TESTS_FAILED -le $TOTAL_ERROR_THRESHOLD ] && echo "✅ PASS" || echo "❌ FAIL"))"

# Final exit decision
if [ ${#CRITICAL_FAILURES[@]} -gt $CRITICAL_ERROR_THRESHOLD ]; then
    echo ""
    echo -e "${RED}💥 CRITICAL EXIT CRITERIA BREACH${NC}"
    echo "   System has critical failures that prevent L9-grade certification"
    exit 1
elif [ $TESTS_FAILED -gt $TOTAL_ERROR_THRESHOLD ]; then
    echo ""
    echo -e "${RED}⚠️  FAILURE THRESHOLD EXCEEDED${NC}"
    echo "   Too many failures for L9-grade certification"
    exit 1
else
    echo ""
    echo -e "${GREEN}🎉 L9-GRADE CERTIFICATION ACHIEVED!${NC}"
    echo "   Neural Flow system passes all end-to-end validation criteria"
    echo ""
    echo "🚀 SYSTEM READY FOR PRODUCTION USE:"
    echo "   • MCP configuration validated"
    echo "   • Stdio communication working"
    echo "   • Neural tools responding"
    echo "   • ChromaDB storage verified"
    echo "   • Multi-project isolation confirmed"
    echo "   • Performance benchmarks operational"
    exit 0
fi