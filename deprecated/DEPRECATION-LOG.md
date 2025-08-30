# Deprecation Log

## Date: 2025-08-29

### Files Deprecated
The following files were moved to the `deprecated/` directory as part of the transition to the Neural Tools Docker-based architecture:

#### Install Scripts (replaced by `neural-install.sh`)
- `install.sh` - Old local Python installation
- `install-global.sh` - Old global installation  
- `mcp-add` - Old MCP configuration tool

#### Docker Files (replaced by `neural-tools/Dockerfile.*`)
- `Dockerfile` - Old monolithic MCP server build
- `docker-entrypoint.sh` - Old Docker entry point
- `requirements.txt` - Old Python dependencies

#### Documentation (content moved to README.md)
- `DEPRECATED-FILES.md` - Already deprecated files list
- `L9-ENHANCED-MCP-SETUP.md` - Old setup instructions
- `TEMPLATE-REORGANIZATION-PLAN.md` - Completed reorganization plan
- `TEMPLATE-STRUCTURE.md` - Old structure documentation

#### Backups & Exports
- `docker-backup-20250829-103920/` - Docker backup from reorganization
- `l9-export.json` - Old configuration export
- `.mcp.json.backup` - MCP configuration backup

### Reason for Deprecation
These files were part of the old Python-based local installation system. The project has been migrated to a fully containerized Docker architecture with the Neural Tools system, which provides:
- No local Python dependencies
- Complete isolation between projects
- Shared embedding model for efficiency
- 9 powerful MCP tools for vibe coders
- Simple one-command installation via `neural-install.sh`

### Current Active System
- **Installer**: `/neural-install.sh`
- **Docker System**: `/neural-tools/`
- **Documentation**: `/README.md`
- **Configuration**: `/.mcp.json` (auto-managed by installer)