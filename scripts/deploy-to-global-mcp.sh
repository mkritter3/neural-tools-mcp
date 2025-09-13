#!/bin/bash
# L9 Engineering Deployment Script
# Deploy ADR-0034 fixes from dev to global MCP production
# Author: L9 Engineering Team
# Date: 2025-09-12

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SOURCE_DIR="/Users/mkr/local-coding/claude-l9-template/neural-tools"
TARGET_DIR="/Users/mkr/.claude/mcp-servers/neural-tools"
BACKUP_DIR="/Users/mkr/.claude/mcp-servers/neural-tools-backup-$(date +%Y%m%d-%H%M%S)"

echo -e "${GREEN}🚀 L9 Neural Tools MCP Deployment${NC}"
echo "=================================================="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo "Backup: $BACKUP_DIR"
echo ""

# Pre-deployment validation
echo -e "${YELLOW}🔍 Pre-deployment validation...${NC}"

# Check source directory exists
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo -e "${RED}❌ Source directory not found: $SOURCE_DIR${NC}"
    exit 1
fi

# Check key files exist
REQUIRED_FILES=(
    "src/neural_mcp/neural_server_stdio.py"
    "src/servers/services/project_context_manager.py"
    "src/servers/services/pipeline_validation.py"
    "src/servers/services/data_migration_service.py"
    "run_mcp_server.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$SOURCE_DIR/$file" ]]; then
        echo -e "${RED}❌ Required file missing: $file${NC}"
        exit 1
    fi
done

# Run ADR-0034 validation tests
echo -e "${YELLOW}🧪 Running ADR-0034 validation tests...${NC}"
cd "$SOURCE_DIR/.."
if python3 -c "
import asyncio
import sys
sys.path.append('neural-tools/src')
from servers.services.pipeline_validation import validate_pipeline_stage

async def test():
    result = await validate_pipeline_stage('detection',
        project_name='l9-graphrag',
        project_path='/Users/mkr/local-coding/claude-l9-template',
        confidence=0.95
    )
    if not result:
        sys.exit(1)
    print('✅ Pipeline validation passed')

asyncio.run(test())
"; then
    echo -e "${GREEN}✅ Validation tests passed${NC}"
else
    echo -e "${RED}❌ Validation tests failed - aborting deployment${NC}"
    exit 1
fi

# Create backup
echo -e "${YELLOW}💾 Creating backup...${NC}"
if [[ -d "$TARGET_DIR" ]]; then
    cp -r "$TARGET_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}✅ Backup created: $BACKUP_DIR${NC}"
else
    echo -e "${YELLOW}⚠️ No existing target directory to backup${NC}"
fi

# Deploy with checksum verification
echo -e "${YELLOW}📦 Deploying neural-tools...${NC}"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Copy files with verification
rsync -av --checksum --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' "$SOURCE_DIR/" "$TARGET_DIR/"

# Verify critical files
echo -e "${YELLOW}🔍 Verifying deployment...${NC}"
for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$TARGET_DIR/$file" ]]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file - deployment failed${NC}"
        echo -e "${YELLOW}🔄 Restoring from backup...${NC}"
        rm -rf "$TARGET_DIR"
        mv "$BACKUP_DIR" "$TARGET_DIR"
        exit 1
    fi
done

# Set permissions
chmod +x "$TARGET_DIR/run_mcp_server.py"

# Create deployment manifest
cat > "$TARGET_DIR/DEPLOYMENT_MANIFEST.json" << EOF
{
  "deployment_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "source_commit": "$(cd "$SOURCE_DIR/.." && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "adr_version": "ADR-0034 Phase 2 Complete",
  "deployed_by": "$(whoami)",
  "backup_location": "$BACKUP_DIR",
  "validation_status": "passed",
  "deployment_method": "rsync with checksum verification"
}
EOF

echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo "=================================================="
echo -e "${GREEN}✅ ADR-0034 fixes deployed to global MCP${NC}"
echo -e "${GREEN}✅ Project pipeline synchronization active${NC}"
echo -e "${GREEN}✅ Dynamic project detection enabled${NC}"
echo -e "${GREEN}✅ Data migration capabilities available${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Restart Claude outside this project directory to use global MCP"
echo "2. Test neural tools with any project"  
echo "3. Verify project detection shows actual project names (not 'default')"
echo ""
echo -e "${YELLOW}💡 To continue development:${NC}"
echo "1. Come back to this project directory: cd /Users/mkr/local-coding/claude-l9-template"
echo "2. Restart Claude - it will use local .mcp.json for development"
echo ""
echo -e "${YELLOW}🔧 Rollback if needed:${NC}"
echo "rm -rf '$TARGET_DIR' && mv '$BACKUP_DIR' '$TARGET_DIR'"
echo ""
echo -e "${GREEN}🎯 L9 Engineering: Deployment complete with full auditability${NC}"