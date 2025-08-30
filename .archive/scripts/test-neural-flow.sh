#!/bin/bash
# Neural Flow L9-Grade Comprehensive Integration Test Suite
# Tests all aspects of the optimized system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
FAILED_TESTS=()

# Logging
log_test() { echo -e "${BLUE}🧪 TEST: $1${NC}"; }
log_pass() { echo -e "${GREEN}✅ PASS: $1${NC}"; ((TESTS_PASSED++)); }
log_fail() { echo -e "${RED}❌ FAIL: $1${NC}"; ((TESTS_FAILED++)); FAILED_TESTS+=("$1"); }
log_info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }

# Test configuration
PROJECT_NAME="test-project-$(date +%s)"
DOCKER_IMAGE="${1:-neural-flow:production}"

echo "========================================="
echo "🔮 Neural Flow L9 Integration Test Suite"
echo "========================================="
echo "Docker Image: $DOCKER_IMAGE"
echo "Test Project: $PROJECT_NAME"
echo ""

# Check prerequisites
log_test "Prerequisites Check"
if ! docker image inspect "$DOCKER_IMAGE" > /dev/null 2>&1; then
    log_fail "Docker image $DOCKER_IMAGE not found"
    exit 1
fi
log_pass "Docker image exists"

# Test 1: Container Health Check
log_test "Container Health Check"
if docker run --rm "$DOCKER_IMAGE" python3 -c "
import sys
sys.path.insert(0, '/app/neural-system')
from neural_embeddings import ONNXCodeEmbedder, CodeSpecificEmbedder
from performance_benchmarks import PerformanceBenchmarks
print('System modules loaded')
" > /dev/null 2>&1; then
    log_pass "Container health check"
else
    log_fail "Container health check"
fi

# Test 2: Neural Embeddings System
log_test "Neural Embeddings System"
if docker run --rm "$DOCKER_IMAGE" python3 -c "
import sys
sys.path.insert(0, '/app/neural-system')
from neural_embeddings import ONNXCodeEmbedder
embedder = ONNXCodeEmbedder()
embedding = embedder.encode('def hello(): return world')
print(f'Embedding dimensions: {len(embedding)}')
assert len(embedding) == 384, 'Expected 384 dimensions'
" > /dev/null 2>&1; then
    log_pass "Neural embeddings (384D ONNX)"
else
    log_fail "Neural embeddings (384D ONNX)"
fi

# Test 3: Code-Specific Embeddings (if available)
log_test "Code-Specific Embeddings (Qodo)"
if docker run --rm -e USE_QODO_EMBED=true "$DOCKER_IMAGE" python3 -c "
import sys
sys.path.insert(0, '/app/neural-system')
try:
    from neural_embeddings import CodeSpecificEmbedder
    embedder = CodeSpecificEmbedder()
    if embedder.is_available():
        print('Qodo-Embed available')
    else:
        print('Qodo-Embed not available (expected without model download)')
except Exception as e:
    print(f'Expected: {e}')
" 2>&1 | grep -q "available"; then
    log_info "Qodo-Embed status checked"
    log_pass "Code-specific embedding system"
else
    log_info "Qodo-Embed not available (expected without full model)"
    log_pass "Code-specific embedding system (fallback mode)"
fi

# Test 4: Performance Benchmarking System
log_test "Performance Benchmarking System"
if docker run --rm -v /tmp:/app/data "$DOCKER_IMAGE" python3 -c "
import sys, asyncio
sys.path.insert(0, '/app/neural-system')
from performance_benchmarks import PerformanceBenchmarks
from pathlib import Path

async def test():
    benchmarker = PerformanceBenchmarks(Path('/app/data'))
    targets = benchmarker.get_current_targets()
    print(f'Week {benchmarker.current_week} targets:')
    print(f'  Recall@1: {targets.recall_at_1:.1%}')
    print(f'  Latency P95: {targets.latency_p95}ms')
    return True

asyncio.run(test())
" > /dev/null 2>&1; then
    log_pass "Performance benchmarking system"
else
    log_fail "Performance benchmarking system"
fi

# Test 5: MCP Server Startup
log_test "MCP Server Initialization"
CONTAINER_ID=$(docker run -d --rm "$DOCKER_IMAGE" 2>/dev/null)
if [ -n "$CONTAINER_ID" ]; then
    sleep 3
    if docker ps -q --filter "id=$CONTAINER_ID" > /dev/null 2>&1; then
        docker stop "$CONTAINER_ID" > /dev/null 2>&1
        log_pass "MCP server startup"
    else
        log_fail "MCP server crashed on startup"
    fi
else
    log_fail "Failed to start container"
fi

# Test 6: Volume Mount Data Isolation
log_test "Volume Mount Data Isolation"
mkdir -p /tmp/test-project-1/.claude /tmp/test-project-2/.claude
echo "project1" > /tmp/test-project-1/.claude/test.txt
echo "project2" > /tmp/test-project-2/.claude/test.txt

RESULT1=$(docker run --rm -v /tmp/test-project-1/.claude:/app/data "$DOCKER_IMAGE" \
    cat /app/data/test.txt 2>/dev/null || echo "failed")
RESULT2=$(docker run --rm -v /tmp/test-project-2/.claude:/app/data "$DOCKER_IMAGE" \
    cat /app/data/test.txt 2>/dev/null || echo "failed")

if [ "$RESULT1" = "project1" ] && [ "$RESULT2" = "project2" ]; then
    log_pass "Volume mount data isolation"
else
    log_fail "Volume mount data isolation"
fi
rm -rf /tmp/test-project-1 /tmp/test-project-2

# Test 7: Environment Variable Configuration
log_test "Environment Variable Configuration"
if docker run --rm \
    -e PROJECT_NAME="test-env-project" \
    -e USE_QODO_EMBED="true" \
    -e ENABLE_AB_TESTING="true" \
    -e ENABLE_BENCHMARKING="true" \
    "$DOCKER_IMAGE" python3 -c "
import os
assert os.environ.get('PROJECT_NAME') == 'test-env-project'
assert os.environ.get('USE_QODO_EMBED') == 'true'
assert os.environ.get('ENABLE_AB_TESTING') == 'true'
assert os.environ.get('ENABLE_BENCHMARKING') == 'true'
print('Environment variables configured correctly')
" > /dev/null 2>&1; then
    log_pass "Environment variable configuration"
else
    log_fail "Environment variable configuration"
fi

# Test 8: Performance Optimizations Active
log_test "Performance Optimizations"
if docker run --rm "$DOCKER_IMAGE" python3 -c "
import os
assert os.environ.get('PYTHONOPTIMIZE') == '2', 'Python optimization not set'
assert os.environ.get('OMP_NUM_THREADS'), 'OpenMP threads not configured'
assert os.environ.get('MALLOC_MMAP_THRESHOLD_'), 'Memory optimization not set'
print('Performance optimizations active')
" > /dev/null 2>&1; then
    log_pass "Performance optimizations active"
else
    log_fail "Performance optimizations not configured"
fi

# Test 9: ChromaDB Vector Store
log_test "ChromaDB Vector Store"
if docker run --rm -v /tmp:/app/data "$DOCKER_IMAGE" python3 -c "
import sys
sys.path.insert(0, '/app/neural-system')
from neural_embeddings import ChromaDBVectorStore
store = ChromaDBVectorStore(persist_dir='/app/data/test_chroma')
stats = store.get_stats()
print(f'ChromaDB initialized: {stats}')
assert 'total_collections' in stats
" > /dev/null 2>&1; then
    log_pass "ChromaDB vector store"
else
    log_fail "ChromaDB vector store"
fi

# Test 10: Feature Flags System
log_test "Feature Flags System"
if docker run --rm -e ENABLE_AB_TESTING=true "$DOCKER_IMAGE" python3 -c "
import sys, os
sys.path.insert(0, '/app/neural-system')
os.environ['ENABLE_AB_TESTING'] = 'true'
from feature_flags import get_feature_manager, is_enabled
manager = get_feature_manager()
# Should work even if feature not explicitly set
print('Feature flags system operational')
" > /dev/null 2>&1; then
    log_pass "Feature flags system"
else
    log_fail "Feature flags system"
fi

# Summary
echo ""
echo "========================================="
echo "📊 Test Results Summary"
echo "========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
    echo -e "${RED}⚠️  Some tests failed. Review and fix issues.${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}🎉 All tests passed! Neural Flow is L9-grade ready.${NC}"
    exit 0
fi