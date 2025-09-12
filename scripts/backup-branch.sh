#!/bin/bash

# L9 Branch Backup Script
# Creates a local backup of the current branch before merging

set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

CURRENT_BRANCH=$(git branch --show-current)
BACKUP_BRANCH="${CURRENT_BRANCH}-backup-$(date +%Y%m%d-%H%M%S)"

echo -e "${YELLOW}L9 Branch Backup Tool${NC}"
echo "========================"
echo ""
echo "Current branch: ${GREEN}${CURRENT_BRANCH}${NC}"
echo "Backup branch:  ${YELLOW}${BACKUP_BRANCH}${NC}"
echo ""

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}⚠️  Warning: You have uncommitted changes${NC}"
    echo "Please commit or stash your changes first."
    exit 1
fi

# Create backup branch
echo "Creating backup branch..."
git branch "${BACKUP_BRANCH}"
echo -e "${GREEN}✓ Backup created: ${BACKUP_BRANCH}${NC}"

# Show branch list
echo ""
echo "Your branches:"
git branch -v

echo ""
echo -e "${GREEN}✓ Backup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. To push to remote: git push origin ${CURRENT_BRANCH}"
echo "2. To merge to main:  git checkout main && git merge ${CURRENT_BRANCH}"
echo "3. Your backup is at: ${BACKUP_BRANCH}"
echo ""
echo "To restore from backup if needed:"
echo "  git checkout ${BACKUP_BRANCH}"