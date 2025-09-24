#!/bin/bash
# Disable the diamondrag MCP server to avoid confusion with neural-tools

echo "🔧 Disabling diamondrag MCP server..."

# Backup current config
cp ~/.claude/mcp_config.json ~/.claude/mcp_config.json.backup-$(date +%Y%m%d-%H%M%S)

# Remove diamondrag from config using Python
python3 << 'EOF'
import json
import sys

config_path = '/Users/mkr/.claude/mcp_config.json'

try:
    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'mcpServers' in config and 'diamondrag' in config['mcpServers']:
        del config['mcpServers']['diamondrag']

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print("✅ Removed diamondrag from MCP config")
        print("   Neural-tools is now the only GraphRAG MCP server")
    else:
        print("⚠️  diamondrag not found in config")

except Exception as e:
    print(f"❌ Error updating config: {e}")
    sys.exit(1)
EOF

echo ""
echo "📝 Next steps:"
echo "1. Restart Claude to reload MCP configuration"
echo "2. You should now see only 11 tools from neural-tools"
echo ""
echo "To re-enable diamondrag later:"
echo "  cp ~/.claude/mcp_config.json.backup-* ~/.claude/mcp_config.json"