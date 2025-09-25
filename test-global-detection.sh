#!/bin/bash
# Test script for global MCP project detection

echo "=================================================="
echo "Testing Global MCP Project Detection (ADR-0102)"
echo "=================================================="
echo ""
echo "This script will test project detection from different directories"
echo ""

# Test 1: From claude-l9-template
echo "Test 1: From claude-l9-template directory"
cd /Users/mkr/local-coding/claude-l9-template
echo "  Current dir: $(pwd)"
echo "  Expected: Should detect 'claude-l9-template'"
echo ""

# Test 2: From neural-novelist
echo "Test 2: From neural-novelist directory"
if [ -d "/Users/mkr/local-coding/Systems/neural-novelist" ]; then
    cd /Users/mkr/local-coding/Systems/neural-novelist
    echo "  Current dir: $(pwd)"
    echo "  Expected: Should detect 'neural-novelist'"
else
    echo "  ⚠️ neural-novelist directory not found"
fi
echo ""

# Test 3: From home directory (no project)
echo "Test 3: From home directory (no project)"
cd ~
echo "  Current dir: $(pwd)"
echo "  Expected: Should return None or require set_project_context"
echo ""

echo "=================================================="
echo "INSTRUCTIONS:"
echo "1. Launch Claude from each directory above"
echo "2. Run: neural_system_status"
echo "3. Check if 'Active Project' matches expected"
echo ""
echo "If detection fails:"
echo "- Use set_project_context tool to manually set"
echo "- Check for running indexer containers"
echo "- Verify package.json or pyproject.toml exists"
echo "=================================================="