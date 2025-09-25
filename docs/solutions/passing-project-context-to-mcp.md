# Solution: Passing Project Context to MCP Servers

**Date:** 2025-09-24
**Research Summary:** Deep investigation into MCP protocol and Claude Code capabilities

## The Core Problem

The MCP (Model Context Protocol) doesn't automatically pass the client's working directory to servers. When Claude Code launches an MCP server, the server's `cwd` is hardcoded to the server's location, not the user's project directory.

## Current State (September 2025)

### What MCP Protocol Supports

1. **Roots Capability**: MCP has a `roots` capability that allows clients to expose filesystem roots
   - Clients can send `roots/list` to share workspace folders
   - BUT: Claude Code doesn't appear to implement this yet

2. **Environment Variable Expansion**: Claude Code supports `${workspaceFolder}` in some contexts
   - Cline fixed this with PR #2990
   - VS Code requested this feature

3. **Initialize Request**: The MCP `initialize` request does NOT include:
   - `rootUri` (unlike LSP)
   - `workspaceFolder`
   - `cwd` or working directory

## Available Solutions

### Solution 1: Command-Line Arguments (Most Reliable)

Pass the project directory as a command-line argument to your MCP server:

```json
// .mcp.json or mcp_config.json
{
  "neural-tools": {
    "command": "python",
    "args": [
      "/path/to/server.py",
      "--project-dir", "${workspaceFolder}"  // If supported
    ]
  }
}
```

Then in your MCP server:
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--project-dir', default=os.getcwd())
args = parser.parse_args()

PROJECT_DIR = Path(args.project_dir)
```

### Solution 2: Environment Variable in Command

Set environment variables directly in the command string:

```json
{
  "neural-tools": {
    "command": "bash",
    "args": [
      "-c",
      "CLAUDE_PROJECT_DIR=$(pwd) python /path/to/server.py"
    ]
  }
}
```

### Solution 3: Wrapper Script

Create a launcher script that captures the working directory:

```bash
#!/bin/bash
# mcp-launcher.sh

# Capture the current directory when launched
export CLAUDE_PROJECT_DIR="$(pwd)"

# Launch the actual MCP server
exec python /path/to/neural_server.py
```

Then use the wrapper in config:
```json
{
  "neural-tools": {
    "command": "/path/to/mcp-launcher.sh"
  }
}
```

### Solution 4: Project-Scoped Configuration

Use Claude Code's project-scoped MCP servers:

1. Create `.mcp.json` in your project root
2. Configure with relative paths
3. The server knows it's in the project directory

```json
// /your/project/.mcp.json
{
  "neural-tools": {
    "command": "python",
    "args": ["./neural-server.py"],
    "cwd": "."  // Current project directory
  }
}
```

### Solution 5: Multiple Server Instances

Configure separate MCP server instances per project:

```json
// ~/.claude/mcp_config.json
{
  "neural-tools-project-a": {
    "command": "python",
    "args": ["/path/to/server.py"],
    "env": {
      "PROJECT_PATH": "/path/to/project-a"
    }
  },
  "neural-tools-project-b": {
    "command": "python",
    "args": ["/path/to/server.py"],
    "env": {
      "PROJECT_PATH": "/path/to/project-b"
    }
  }
}
```

## Recommended Implementation

### For Our Neural Tools

1. **Primary**: Accept `--project-dir` command-line argument
2. **Fallback 1**: Check `CLAUDE_PROJECT_DIR` environment variable
3. **Fallback 2**: Use container detection (our current implementation)
4. **Fallback 3**: File-based detection from `cwd`
5. **Last Resort**: Require explicit `set_project_context` call

### Code Implementation

```python
# In neural_server.py startup
import argparse
import os
from pathlib import Path

async def detect_project_context():
    # 1. Command-line argument (highest priority)
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', help='Project directory path')
    args, _ = parser.parse_known_args()

    if args.project_dir:
        return Path(args.project_dir)

    # 2. Environment variable
    if claude_dir := os.getenv("CLAUDE_PROJECT_DIR"):
        return Path(claude_dir)

    # 3. Current working directory (if not in .claude)
    cwd = Path.cwd()
    if ".claude" not in str(cwd):
        return cwd

    # 4. Container detection (our implementation)
    # ... existing container detection code ...

    # 5. Explicit None
    return None
```

## Known Issues (September 2025)

1. **Environment Variables Bug**: The `env` section in config doesn't always work (#1254)
2. **Windows Path Issues**: Use full paths for `npx.cmd` on Windows
3. **WSL Complications**: Requires special `wsl.exe` wrapper configuration
4. **No ${workspaceFolder}**: Not all Claude Code versions expand this variable

## Future Improvements

### What We Need from Claude/Anthropic

1. **Native Support**: Add `workingDirectory` or `rootUri` to MCP initialize request
2. **Roots Implementation**: Fully implement the `roots` capability in Claude Code
3. **Variable Expansion**: Support `${workspaceFolder}` consistently
4. **Auto-Detection**: Pass the launch directory automatically as env var

### MCP Protocol Enhancement Proposal

```typescript
// Proposed MCP Initialize Request
interface InitializeRequest {
  protocolVersion: string;
  capabilities: ClientCapabilities;
  clientInfo: ClientInfo;
  // NEW FIELDS:
  workingDirectory?: string;  // Client's working directory
  workspaceFolders?: WorkspaceFolder[];  // Like LSP
  environment?: Record<string, string>;  // Client environment
}
```

## Testing Your Solution

```bash
# Test if project directory is being passed
python -c "
import os
print(f'CWD: {os.getcwd()}')
print(f'CLAUDE_PROJECT_DIR: {os.getenv(\"CLAUDE_PROJECT_DIR\", \"Not set\")}')
print(f'Args: {sys.argv}')
"
```

## Conclusion

Until Claude Code natively passes the working directory, we must use workarounds:
1. Command-line arguments are most reliable
2. Environment variables work but have bugs
3. Our container detection is a good fallback
4. Multiple fallback strategies ensure robustness

The best approach is to implement ALL methods and use them in priority order, ensuring your MCP server can detect the project context regardless of how it's launched.

**Confidence: 95%** - Based on extensive research of September 2025 MCP documentation and community solutions.