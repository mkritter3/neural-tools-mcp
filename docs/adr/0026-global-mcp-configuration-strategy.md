# ADR-0026: Global MCP Configuration Strategy

**Status:** Accepted  
**Date:** September 11, 2025  
**Author:** L9 Engineering Team

## Context

Currently, the neural-tools MCP server is configured with an absolute path pointing to a specific project directory:
```
/Users/mkr/local-coding/claude-l9-template/neural-tools/run_mcp_server.py
```

This creates a problem: the MCP only works when Claude is opened in that exact project. When opening Claude in other projects, the neural-tools MCP is not available because the hardcoded path doesn't exist in those directories.

## Decision

Implement a **project-agnostic global MCP configuration** that:
1. Installs the MCP server code in a central location
2. Uses a wrapper script that auto-detects the current working directory
3. Maintains `AUTO_DETECT_PROJECT=true` for dynamic project context
4. Preserves all existing Docker service connections

## Architecture

### Installation Structure
```
~/.claude/
├── mcp-servers/
│   └── neural-tools/
│       ├── run_mcp_server.py      # Main MCP server
│       ├── wrapper.py              # Project detection wrapper
│       └── src/                    # Full source tree copy
│           ├── neural_mcp/
│           └── servers/
│               └── services/
```

### Wrapper Script Design
```python
#!/usr/bin/env python3
"""
Global MCP wrapper that detects current project context
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get current working directory from Claude
    cwd = os.getcwd()
    
    # Set project name from directory
    project_name = Path(cwd).name
    
    # Set environment for project context
    env = os.environ.copy()
    env['PROJECT_PATH'] = cwd
    env['PROJECT_NAME'] = project_name
    
    # Add neural-tools src to Python path
    mcp_base = Path(__file__).parent
    env['PYTHONPATH'] = str(mcp_base / 'src')
    
    # Launch actual MCP server
    server_path = mcp_base / 'run_mcp_server.py'
    subprocess.run([sys.executable, str(server_path)], env=env)

if __name__ == '__main__':
    main()
```

### Claude Configuration Update
```json
{
  "mcpServers": {
    "neural-tools": {
      "command": "python3",
      "args": ["~/.claude/mcp-servers/neural-tools/wrapper.py"],
      "env": {
        "AUTO_DETECT_PROJECT": "true",
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "EMBEDDING_SERVICE_HOST": "localhost",
        "EMBEDDING_SERVICE_PORT": "48000",
        "REDIS_CACHE_HOST": "localhost",
        "REDIS_CACHE_PORT": "46379",
        "REDIS_QUEUE_HOST": "localhost",
        "REDIS_QUEUE_PORT": "46380",
        "EMBED_DIM": "768",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## Benefits

1. **Universal Availability**: MCP works in any project directory
2. **Project Isolation**: Each project gets its own collections/graphs
3. **Zero Configuration**: No per-project setup required
4. **Backward Compatible**: Existing project data remains intact
5. **Single Source of Truth**: One MCP installation to maintain

## Implementation Plan

### Phase 1: Create Installation Script
```bash
#!/bin/bash
# install-mcp-global.sh

# Create MCP directory structure
mkdir -p ~/.claude/mcp-servers/neural-tools

# Copy neural-tools source
cp -r neural-tools/* ~/.claude/mcp-servers/neural-tools/

# Create wrapper script
cat > ~/.claude/mcp-servers/neural-tools/wrapper.py << 'EOF'
[wrapper script content]
EOF

# Update Claude configuration
python3 scripts/update_claude_config.py --global
```

### Phase 2: Update MCP Server Code
Modify `run_mcp_server.py` to:
- Accept PROJECT_PATH from environment
- Use PROJECT_NAME for collection naming
- Default to cwd if not provided

### Phase 3: Migration Support
- Detect existing project-specific configs
- Offer migration to global config
- Preserve existing data in Qdrant/Neo4j

## Consequences

### Positive
- MCP available in all projects immediately
- No more "MCP not found" errors
- Simplified onboarding for team members
- Easier updates (single location)

### Negative
- Initial setup slightly more complex
- Need to maintain wrapper script
- Potential version conflicts if projects need different versions

### Mitigation
- Clear documentation for setup process
- Version checking in wrapper script
- Option to override with project-specific config

## Alternatives Considered

1. **Symlink Approach**: Create symlinks in each project
   - Rejected: Requires per-project setup
   
2. **System Package**: Install as system Python package
   - Rejected: Harder to update, permission issues
   
3. **Docker Container**: Run MCP in container
   - Rejected: Added complexity, slower startup

## Decision Rationale

The global configuration with project detection provides the best balance of:
- User convenience (works everywhere)
- Technical simplicity (Python wrapper)
- Maintenance ease (single installation)
- Flexibility (project isolation maintained)

## References

- [MCP Protocol Specification 2025-06-18](https://modelcontextprotocol.io/spec)
- [Claude Desktop Configuration Guide](https://docs.anthropic.com/claude/docs/mcp-config)
- ADR-0016: MCP Container Connectivity Architecture

**Confidence: 95%**
Assumptions: 
- Claude passes cwd to MCP servers
- ~/.claude directory is standard location