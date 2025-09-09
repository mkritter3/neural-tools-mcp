# Systematic Build & Development Guide

## Overview
This guide ensures that all code changes to the Neural Tools MCP server are systematically applied and persist across container rebuilds. The solution provides both development (live code mounting) and production (baked into image) workflows.

## The Problem Solved
When modifying the MCP server code (e.g., adding the `project_indexer` tool), changes made only to local files wouldn't appear in running containers. This guide provides a systematic approach to ensure changes persist.

## Two Workflow Modes

### 1. Development Mode (Live Code Updates)
For rapid development with immediate code changes:

```bash
cd neural-tools
./build-and-run.sh --dev
```

**How it works:**
- Mounts `src/` directory into container at `/app/src` and `/app/neural-tools-src`
- Changes to Python files reflect immediately without rebuilding
- Perfect for testing and iterating on MCP tools

**Files involved:**
- `config/docker-compose.neural-tools.yml` (base configuration)
- `config/docker-compose.neural-tools.dev.yml` (development override)
- Source code in `src/` directory

### 2. Production Mode (Baked Code)
For stable deployments with code baked into image:

```bash
cd neural-tools
./build-and-run.sh --rebuild
```

**How it works:**
- Builds Docker image with code copied into container
- Changes require rebuild to take effect
- Ensures consistent deployment across environments

**Files involved:**
- `Dockerfile.l9-minimal` (defines image build)
- `config/docker-compose.neural-tools.yml` (orchestrates services)

## Script Options

The `build-and-run.sh` script provides systematic control:

```bash
# Build only (don't start services)
./build-and-run.sh --build-only

# Development mode with live code mounting
./build-and-run.sh --dev

# Force rebuild even if image exists
./build-and-run.sh --rebuild

# Combine options
./build-and-run.sh --dev --rebuild
```

## Making Code Changes Systematic

### Step 1: Modify Source Code
Edit files in `neural-tools/src/`:
```bash
# Example: Adding a new MCP tool
vim src/mcp/neural_server_stdio.py
```

### Step 2: Test in Development Mode
```bash
# Start with live code mounting
./build-and-run.sh --dev

# Test your changes
# Changes reflect immediately
```

### Step 3: Commit to Production
```bash
# Rebuild image with changes baked in
./build-and-run.sh --rebuild

# Verify changes persist
docker exec -it default-neural python3 -c "
from src.servers.neural_server_stdio import NeuralMCPServer
server = NeuralMCPServer()
print([t.name for t in server.list_tools()])
"
```

### Step 4: Verify Persistence
```bash
# Stop services
docker-compose -f config/docker-compose.neural-tools.yml down

# Restart without dev mode
./build-and-run.sh

# Confirm changes still present
```

## File Structure for Systematic Updates

```
neural-tools/
├── src/                          # Source code (edit here)
│   ├── servers/
│   │   ├── neural_server_stdio.py  # MCP server with tools (canonical)
│   │   └── services/             # Service integrations
│   └── services/
│       └── indexer_service.py   # IncrementalIndexer
├── config/
│   ├── docker-compose.neural-tools.yml      # Base config
│   ├── docker-compose.neural-tools.dev.yml  # Dev override
│   └── requirements-l9-enhanced.txt         # Dependencies
├── Dockerfile.l9-minimal         # Production image
└── build-and-run.sh             # Systematic build script
```

## Critical Changes Made

### 1. Added `project_indexer` Tool
Location: `src/mcp/neural_server_stdio.py`
- Tool definition in `__init__` method
- Implementation in `project_indexer_impl` method
- Proper service initialization

### 2. Created Development Override
File: `config/docker-compose.neural-tools.dev.yml`
- Mounts source code for live updates
- Enables debug logging
- Disables health checks during development

### 3. Systematic Build Script
File: `build-and-run.sh`
- Handles both dev and production modes
- Ensures consistent builds
- Provides clear feedback

## Verification Commands

### Check Tool Availability
```bash
# List all MCP tools
docker exec -it default-neural python3 -c "
import sys
sys.path.append('/app')
from src.servers.neural_server_stdio import NeuralMCPServer
server = NeuralMCPServer()
for tool in server.list_tools():
    print(f'- {tool.name}')
"
```

### Test Indexing
```bash
# Run indexing via MCP tool (from MCP client)
{
  "tool": "project_indexer",
  "arguments": {
    "path": "/app/project",
    "recursive": true
  }
}
```

### Check Service Status
```bash
# View all services
docker-compose -f config/docker-compose.neural-tools.yml ps

# Check logs
docker-compose -f config/docker-compose.neural-tools.yml logs neural-tools-server
```

## Troubleshooting Systematic Issues

### Changes Not Appearing
1. **In Dev Mode**: Check volume mounts
   ```bash
   docker inspect default-neural | grep -A 5 Mounts
   ```

2. **In Production**: Ensure rebuild
   ```bash
   ./build-and-run.sh --rebuild
   ```

### Import Errors
- Check PYTHONPATH in Dockerfile
- Verify file locations match import paths
- Ensure `__init__.py` files exist

### Service Connection Issues
- Verify service names in docker-compose
- Check network configuration
- Ensure environment variables are set

## Best Practices for Systematic Development

1. **Always test in dev mode first**
   - Faster iteration
   - Immediate feedback
   - No rebuild delays

2. **Document changes**
   - Update this guide for new tools
   - Add docstrings to functions
   - Include usage examples

3. **Version control**
   - Commit working changes
   - Tag production releases
   - Track configuration changes

4. **Monitor builds**
   - Check build logs for errors
   - Verify image sizes
   - Track build times

## Summary

The systematic approach ensures:
- **Development flexibility**: Live code mounting for rapid iteration
- **Production stability**: Baked images for consistent deployment
- **Clear workflows**: Script automation for common tasks
- **Persistence**: Changes survive container restarts
- **Traceability**: Clear file structure and documentation

This systematic solution addresses the core issue: "did you only fix this in the running container or the dockerfile / compose as well?" by providing both immediate (dev) and permanent (production) workflows.
