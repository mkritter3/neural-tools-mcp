#!/bin/bash
# L9 Integration Test Suite - Non-Docker Comprehensive Validation
# Tests all L9 components working together without requiring Docker containers

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "======================================="
echo "🔮 L9 Neural Flow Integration Test"
echo "======================================="
echo "Testing L9 systems without Docker dependency"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0

test_pass() { echo -e "${GREEN}✅ PASS: $1${NC}"; ((TESTS_PASSED++)); }
test_fail() { echo -e "${RED}❌ FAIL: $1${NC}"; ((TESTS_FAILED++)); }
test_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Test 1: L9 Certification Suite (Most comprehensive)
test_info "Running L9 Certification Suite..."
if USE_QODO_EMBED=true ENABLE_AB_TESTING=true python3 /Users/mkr/local-coding/claude-l9-template/.claude/neural-system/l9_certification_suite.py > /tmp/l9_cert.log 2>&1; then
    if grep -q "✅ L9 CERTIFICATION ACHIEVED" /tmp/l9_cert.log; then
        test_pass "L9 Certification Suite (100% success required)"
    else
        test_fail "L9 Certification Suite (not achieving 100%)"
    fi
else
    test_fail "L9 Certification Suite (execution failed)"
fi

# Test 2: Auto-Safety System
test_info "Testing Auto-Safety System..."
if python3 -c "
from l9_auto_safety import L9AutoSafetySystem
system = L9AutoSafetySystem()
result = system.scan_project_for_risks('.')
assert len(result.sensitive_files) >= 1, 'Should find sensitive files'
assert len(result.dangerous_commands) >= 40, 'Should find dangerous commands'
assert result.protection_level == 'maximum', 'Should have maximum protection'
print('Auto-safety validation passed')
" 2>/dev/null; then
    test_pass "Auto-Safety System"
else
    test_fail "Auto-Safety System"
fi

# Test 3: ChromaDB Integration
test_info "Testing ChromaDB Integration..."
if python3 -c "
import sqlite3
conn = sqlite3.connect('.claude/chroma/chroma.sqlite3')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM collections')
collections = cursor.fetchone()[0]
assert collections > 0, 'Should have collections'
print('ChromaDB validation passed')
conn.close()
" 2>/dev/null; then
    test_pass "ChromaDB Integration"
else
    test_fail "ChromaDB Integration"
fi

# Test 4: Project Structure Validation
test_info "Testing Project Structure..."
REQUIRED_L9_FILES=(
    ".claude/neural-system/l9_single_model_system.py"
    ".claude/neural-system/l9_auto_safety.py"
    ".claude/neural-system/l9_hybrid_search.py"
    ".claude/neural-system/l9_certification_suite.py"
    ".claude/neural-system/mcp_neural_server.py"
    ".mcp.json"
    "Dockerfile.l9-production"
    "requirements/requirements-l9.txt"
)

MISSING_FILES=0
for file in "${REQUIRED_L9_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  Missing: $file"
        ((MISSING_FILES++))
    fi
done

if [ $MISSING_FILES -eq 0 ]; then
    test_pass "Project Structure (all L9 files present)"
else
    test_fail "Project Structure ($MISSING_FILES missing files)"
fi

# Test 5: MCP Configuration Validation
test_info "Testing MCP Configuration..."
if python3 -c "
import json
with open('.mcp.json', 'r') as f:
    config = json.load(f)
assert 'mcpServers' in config
assert 'neural-flow' in config['mcpServers']
assert 'neural-flow-l9' in config['mcpServers']
l9_config = config['mcpServers']['neural-flow-l9']
assert l9_config['command'] == 'docker'
assert 'NEURAL_L9_MODE=1' in str(l9_config['args'])
print('MCP configuration validation passed')
" 2>/dev/null; then
    test_pass "MCP Configuration"
else
    test_fail "MCP Configuration"
fi

# Test 6: Safety Settings Auto-Generation
test_info "Testing Safety Settings Auto-Generation..."
if [ -f ".claude/settings.json" ]; then
    if python3 -c "
import json
with open('.claude/settings.json', 'r') as f:
    settings = json.load(f)
assert 'permissions' in settings
assert 'deny' in settings['permissions']
assert len(settings['permissions']['deny']) >= 40
assert 'env' in settings
assert settings['env'].get('NEURAL_L9_MODE') == '1'
print('Safety settings validation passed')
" 2>/dev/null; then
        test_pass "Safety Settings Auto-Generation"
    else
        test_fail "Safety Settings Auto-Generation (invalid format)"
    fi
else
    test_fail "Safety Settings Auto-Generation (file missing)"
fi

# Test 7: Documentation & Legacy Structure
test_info "Testing Documentation Structure..."
DOC_FILES=(
    ".claude/neural-system/README.md"
    ".claude/neural-system/legacy/README.md"
    "PROJECT_STRUCTURE.md"
    "docs/adr/0001-l9-neural-flow-optimization-2025.md"
)

DOC_MISSING=0
for file in "${DOC_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        ((DOC_MISSING++))
    fi
done

if [ $DOC_MISSING -eq 0 ]; then
    test_pass "Documentation Structure"
else
    test_fail "Documentation Structure ($DOC_MISSING missing docs)"
fi

# Final Results
echo ""
echo "======================================="
echo "📊 L9 INTEGRATION TEST RESULTS"
echo "======================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"

SUCCESS_RATE=$(( (TESTS_PASSED * 100) / (TESTS_PASSED + TESTS_FAILED) ))
echo "Success Rate: $SUCCESS_RATE%"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL L9 INTEGRATION TESTS PASSED!${NC}"
    echo ""
    echo "✅ L9 Neural Flow System is fully integrated and operational"
    echo "✅ All components working together correctly"
    echo "✅ Auto-safety, certification, and data systems validated"
    echo "✅ MCP configuration and project structure verified"
    echo ""
    echo "🚀 System ready for production deployment"
    exit 0
elif [ $TESTS_FAILED -le 2 ]; then
    echo -e "${YELLOW}⚠️  MINOR INTEGRATION ISSUES DETECTED${NC}"
    echo "   Most systems working correctly but some components need attention"
    exit 1
else
    echo -e "${RED}❌ MAJOR INTEGRATION FAILURES${NC}"
    echo "   Multiple systems failing - requires investigation"
    exit 2
fi