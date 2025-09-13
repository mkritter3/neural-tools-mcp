# L9 Neural Tools MCP: Dev/Prod Configuration

This document outlines the L9 engineering approach for managing MCP server configurations across development and production environments.

## Quick Start

### Use Development Neural Tools (This Project)
```bash
# Simply work from this project directory
cd /Users/mkr/local-coding/claude-l9-template

# Restart Claude - it automatically uses local .mcp.json with all ADR-0034 fixes
```

### Use Production Neural Tools (Global)
```bash
# Work from any other directory 
cd ~/

# Restart Claude - it uses global ~/.claude/mcp_config.json
```

## Configuration Files

### Development Config
- **Location**: `.mcp.json` (local to this project)
- **Neural Tools Path**: `/Users/mkr/local-coding/claude-l9-template/neural-tools/`
- **Purpose**: Testing and development with latest fixes
- **Features**: ADR-0034 pipeline synchronization, dynamic project detection, data migration
- **Activation**: Automatic when Claude starts from this project directory

### Production Config  
- **Location**: `~/.claude/mcp_config.json` (global)
- **Neural Tools Path**: `/Users/mkr/.claude/mcp-servers/neural-tools/`
- **Purpose**: Stable version for all projects
- **Deployment**: Via `scripts/deploy-to-global-mcp.sh`
- **Activation**: Automatic when Claude starts from any other directory

## L9 Engineering Workflow

### Development Phase
1. Work in `/Users/mkr/local-coding/claude-l9-template/neural-tools/`
2. Ensure Claude is started from project directory (uses local `.mcp.json`)
3. Test changes with Claude using dev neural tools
4. Run validation: `python3 -c "...validation tests..."`

### Production Deployment
1. Validate all tests pass in development
2. Run deployment script: `./scripts/deploy-to-global-mcp.sh`
3. Change to any directory outside the project: `cd ~/`
4. Restart Claude to load production neural tools
5. Verify functionality across multiple projects

### Rollback (If Needed)
```bash
# The deployment script creates automatic backups
rm -rf /Users/mkr/.claude/mcp-servers/neural-tools
mv /Users/mkr/.claude/mcp-servers/neural-tools-backup-YYYYMMDD-HHMMSS /Users/mkr/.claude/mcp-servers/neural-tools
```

## Configuration Selection

| Context | Config Used | Neural Tools Path |
|---------|-------------|------------------|
| **Development** (in project dir) | `.mcp.json` (local) | This project's neural-tools |
| **Production** (outside project) | `~/.claude/mcp_config.json` (global) | Deployed neural-tools |

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
# Verify you're in the project directory
pwd
# Should output: /Users/mkr/local-coding/claude-l9-template

# Check local .mcp.json exists
ls -la .mcp.json
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