# L9 Neural Flow - Archived Components

This directory contains all components that were moved from the local machine to preserve history while transitioning to **container-pure architecture**.

## Archive Structure

### `/neural-system/`
**Original L9 neural components** - Now integrated into Docker containers
- `tree_sitter_ast.py` → Copied to `docker/neural-mcp-server-v2.py`
- `neural_embeddings.py` → Replaced by `docker/embedding_server.py`
- `memory_system.py` → Container-based memory management
- `l9_*.py` → L9 certification and hybrid search components

### `/scripts/`
**Management and setup scripts** - Replaced by docker-compose orchestration
- `neural-mcp-manager.py` → Copied to Docker containers for MCP tools
- `test-*.py` → Testing utilities (still useful for debugging)
- Shell scripts → Replaced by `docker-compose` commands

### `/requirements/`
**Multiple Python requirements** - Consolidated into `docker/requirements-mcp.txt`
- `requirements-l9.txt` → Core L9 dependencies
- `requirements-docker.txt` → Docker-specific packages
- Platform-specific requirements → Handled by Docker base images

### `/data/`
**Local data directories** - Now managed by Docker volumes
- `qdrant_storage/` → Docker volume `qdrant-data`  
- `qdrant_snapshots/` → Docker volume `qdrant-snapshots`
- `projects/` → Per-project Docker volumes
- `backups/` → Container backup functionality

### `/core/`
**Original local server** - Replaced by containerized MCP server
- Local FastAPI server → `docker/neural-mcp-server-v2.py`
- Direct file access → Container volume mounts

### `/examples/` & `/templates/`
**Usage examples** - Moved to documentation
- Basic setup examples → Integrated into main README
- Multi-project examples → Docker-compose examples

### `/legacy/`
**Historical components** - Preserved for reference
- Phase 1 completion reports
- Original server implementations

## Migration Summary

**Before (Local Architecture):**
- Multiple Python environments and requirements
- Local data directories mixed with code
- Scripts for manual management
- Direct file system access

**After (Container-Pure Architecture):**
```
claude-l9-template/
├── docker/                    # Complete containerized system
├── docs/                      # Documentation only
├── README.md                  # Usage guide
├── CLAUDE.md                  # Project instructions  
└── .archive/                  # This preserved history
```

**Benefits of Container Architecture:**
1. **Zero Local Setup** - Only Docker required
2. **Perfect Isolation** - Each project gets clean containers
3. **Easy Sharing** - Just share `docker/` directory  
4. **Consistent Environment** - Same behavior everywhere
5. **Scalable** - Easy to add more container services

## Recovery Instructions

If you need to reference or restore any archived component:

1. **For Neural System Components:**
   ```bash
   # Components are already integrated into containers
   # See docker/neural-mcp-server-v2.py for current implementation
   ```

2. **For Scripts:**
   ```bash
   # Use docker-compose instead:
   cd docker && docker-compose -f docker-compose.mcp.yml up -d
   ```

3. **For Requirements:**
   ```bash
   # Single clean requirements file:
   # See docker/requirements-mcp.txt
   ```

4. **For Data Recovery:**
   ```bash
   # Data now lives in Docker volumes
   docker volume ls  # List volumes
   docker run --rm -v volume_name:/data alpine tar -czf - /data > backup.tar.gz
   ```

---

**Archive Date:** August 28, 2025  
**Reason:** Transition to container-pure L9 architecture  
**Status:** All functionality preserved in Docker containers