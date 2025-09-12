#!/bin/bash
# Update global neural-tools MCP installation with latest changes

echo "🔄 Updating global neural-tools MCP installation..."

# Source and destination
SRC_DIR="/Users/mkr/local-coding/claude-l9-template/neural-tools"
DEST_DIR="/Users/mkr/.claude/mcp-servers/neural-tools"

# Backup current global installation
BACKUP_DIR="${DEST_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
echo "📦 Creating backup at: $BACKUP_DIR"
cp -r "$DEST_DIR" "$BACKUP_DIR"

# Files and directories to sync
echo "📋 Syncing updated files..."

# Core MCP server files
cp -v "$SRC_DIR/src/neural_mcp/neural_server_stdio.py" "$DEST_DIR/src/neural_mcp/"

# Service files with new canon/metadata functionality  
cp -v "$SRC_DIR/src/servers/services/pattern_extractor.py" "$DEST_DIR/src/servers/services/"
cp -v "$SRC_DIR/src/servers/services/git_extractor.py" "$DEST_DIR/src/servers/services/"
cp -v "$SRC_DIR/src/servers/services/canon_manager.py" "$DEST_DIR/src/servers/services/"
cp -v "$SRC_DIR/src/servers/services/metadata_backfiller.py" "$DEST_DIR/src/servers/services/"

# Updated indexer with metadata extraction
cp -v "$SRC_DIR/src/servers/services/indexer_service.py" "$DEST_DIR/src/servers/services/"

# Verify the update
echo ""
echo "✅ Verifying update..."
NEW_TOOLS=$(grep -c "canon_understanding\|backfill_metadata" "$DEST_DIR/src/neural_mcp/neural_server_stdio.py")
echo "Found $NEW_TOOLS references to new tools in global installation"

if [ "$NEW_TOOLS" -gt 0 ]; then
    echo "✨ Global neural-tools MCP successfully updated!"
    echo ""
    echo "📝 New tools added:"
    echo "  - canon_understanding: Get canonical knowledge insights"
    echo "  - backfill_metadata: Add metadata to existing indexed data"
    echo ""
    echo "🔄 Please restart Claude to use the updated MCP server"
else
    echo "❌ Update may have failed - new tools not found"
    echo "Restoring from backup..."
    rm -rf "$DEST_DIR"
    mv "$BACKUP_DIR" "$DEST_DIR"
fi