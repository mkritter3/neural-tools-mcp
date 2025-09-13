# L9 Neural Tools MCP: Dev/Prod Configuration

This document outlines the L9 engineering approach for managing MCP server configurations across development and production environments.

## Quick Start

### Use Development Neural Tools (This Project)
```bash
# Set environment variable (add to your shell profile for persistence)
export CLAUDE_MCP_CONFIG_PATH="$HOME/.claude/mcp-dev-config.json"

# Start Claude - it will use the dev neural tools with all ADR-0034 fixes
```

### Use Production Neural Tools (Global)
```bash
# Remove or unset the environment variable
unset CLAUDE_MCP_CONFIG_PATH

# Start Claude - it will use the global production neural tools
```

## Configuration Files

### Development Config
- **Location**: `~/.claude/mcp-dev-config.json`
- **Neural Tools Path**: `/Users/mkr/local-coding/claude-l9-template/neural-tools/`
- **Purpose**: Testing and development with latest fixes
- **Features**: ADR-0034 pipeline synchronization, dynamic project detection, data migration

### Production Config  
- **Location**: `~/.claude/mcp_config.json`
- **Neural Tools Path**: `/Users/mkr/.claude/mcp-servers/neural-tools/`
- **Purpose**: Stable version for all projects
- **Deployment**: Via `scripts/deploy-to-global-mcp.sh`

## L9 Engineering Workflow

### Development Phase
1. Work in `/Users/mkr/local-coding/claude-l9-template/neural-tools/`
2. Use dev config: `export CLAUDE_MCP_CONFIG_PATH="$HOME/.claude/mcp-dev-config.json"`
3. Test changes with Claude using dev neural tools
4. Run validation: `python3 -c "...validation tests..."`

### Production Deployment
1. Validate all tests pass in development
2. Run deployment script: `./scripts/deploy-to-global-mcp.sh`
3. Switch to production config: `unset CLAUDE_MCP_CONFIG_PATH`
4. Restart Claude to load production neural tools
5. Verify functionality across multiple projects

### Rollback (If Needed)
```bash
# The deployment script creates automatic backups
rm -rf /Users/mkr/.claude/mcp-servers/neural-tools
mv /Users/mkr/.claude/mcp-servers/neural-tools-backup-YYYYMMDD-HHMMSS /Users/mkr/.claude/mcp-servers/neural-tools
```

## Environment Variables

| Variable | Development | Production |
|----------|-------------|------------|
| `CLAUDE_MCP_CONFIG_PATH` | `~/.claude/mcp-dev-config.json` | *(unset)* |
| `MCP_ENV` | `development` | *(not set)* |
| `PROJECT_VALIDATION_ENABLED` | `true` | *(not set)* |

## Benefits of This Approach

✅ **Isolation**: Dev changes don't affect production until explicitly deployed  
✅ **Safety**: Automatic backups and validation before deployment  
✅ **Auditability**: Deployment manifest tracks all changes  
✅ **Rollback**: Easy reversion if issues arise  
✅ **Velocity**: Fast iteration in dev environment  
✅ **L9 Standards**: Proper dev/test/prod separation  

## Current Status

- ✅ **ADR-0034 Phase 2 Complete**: Project pipeline synchronization working
- ✅ **Development Environment**: Ready for testing
- ⏳ **Production Deployment**: Run `./scripts/deploy-to-global-mcp.sh` when ready

## Troubleshooting

### Claude Not Using Dev Config
```bash
# Verify environment variable is set
echo $CLAUDE_MCP_CONFIG_PATH

# Should output: /Users/mkr/.claude/mcp-dev-config.json
```

### Project Detection Still Shows "default" 
1. Ensure you're using the dev config with ADR-0034 fixes
2. Restart Claude after changing config
3. Check that neural-tools is loading from the correct path

### Deployment Issues
- Check deployment script output for specific errors
- Verify source files exist and are valid
- Review backup location in case rollback is needed

---

**L9 Engineering**: This approach balances development velocity with production stability, following industry best practices for configuration management and immutable deployments.