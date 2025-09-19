#!/bin/bash
# Quick validation script to ensure modular workflows are valid

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔍 Validating GitHub Actions Modular Workflows..."
echo "================================================"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${YELLOW}⚠️  GitHub CLI not installed. Install with: brew install gh${NC}"
    echo "Skipping workflow validation"
    exit 0
fi

# Validate workflow syntax for all YAML files
WORKFLOW_DIR=".github/workflows"
ERRORS=0

echo -e "\n${YELLOW}Checking main orchestrator...${NC}"
if gh workflow view main.yml &> /dev/null; then
    echo -e "${GREEN}✅ main.yml is valid${NC}"
else
    echo -e "${RED}❌ main.yml has errors${NC}"
    ERRORS=$((ERRORS + 1))
fi

echo -e "\n${YELLOW}Checking modular workflows...${NC}"
for workflow in $WORKFLOW_DIR/modules/*.yml; do
    basename=$(basename "$workflow")
    # GitHub CLI doesn't directly validate reusable workflows, so we check YAML syntax
    if python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null; then
        echo -e "${GREEN}✅ $basename is valid YAML${NC}"
    else
        echo -e "${RED}❌ $basename has YAML errors${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

echo -e "\n${YELLOW}Checking test matrix configuration...${NC}"
if python3 -c "import yaml; yaml.safe_load(open('$WORKFLOW_DIR/test-matrix.yml'))" 2>/dev/null; then
    echo -e "${GREEN}✅ test-matrix.yml is valid YAML${NC}"
else
    echo -e "${RED}❌ test-matrix.yml has YAML errors${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Summary
echo -e "\n================================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All workflows are valid!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Push to GitHub to trigger workflows"
    echo "2. Check Actions tab for execution"
    echo "3. Use 'gh run list --workflow=main.yml' to see runs"
else
    echo -e "${RED}❌ Found $ERRORS workflow errors${NC}"
    echo "Fix the errors above before pushing"
    exit 1
fi