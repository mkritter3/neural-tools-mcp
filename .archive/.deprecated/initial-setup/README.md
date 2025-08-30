# Legacy Components

This directory contains legacy components from the pre-Docker architecture. These files are maintained for compatibility but are not part of the current recommended workflow.

## Current Status

**✅ Active System**: Docker-based containerization with CLI/SDK
**📍 Current Workflow**: `neural-init → ./neural-start.sh → Claude Code`

## Legacy Components

### Pre-Docker Architecture (Historical)
- `SETUP.md` - Original setup instructions for direct Python installation
- `install.py` - Environment-aware installer for Python dependencies
- `start_preloaded_server.sh` - Direct MCP server startup script
- `test_mcp_server.py` - Legacy test suite
- `phase1-completion-report.md` - Historical Phase 1 implementation report

### When to Use Legacy Components

**Use Docker Architecture (Recommended):**
- ✅ New projects
- ✅ Multi-project development  
- ✅ Dependency isolation needed
- ✅ Team development

**Use Legacy Architecture (Special Cases):**
- ⚠️  Docker not available
- ⚠️  Direct Python integration required
- ⚠️  Debugging neural system components
- ⚠️  Development of neural system itself

## Migration Path

If you're using legacy components:

```bash
# 1. Install Docker
# 2. Use new workflow
./scripts/neural-init my-project python
cd my-project
./neural-start.sh
```

## Support

Legacy components are maintained for compatibility but may not receive updates. For issues with legacy components, consider migrating to the Docker architecture.