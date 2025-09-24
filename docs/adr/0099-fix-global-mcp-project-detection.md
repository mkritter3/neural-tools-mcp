# ADR-0099: Fix Global MCP Project Detection

**Status:** ACCEPTED
**Date:** 2025-09-24
**Authors:** L9 Engineering Team
**Context:** Global MCP Deployment, Multi-Project Support
**Supersedes:** Complements ADR-0033, ADR-0098

## Executive Summary

Fix the critical bug where global MCP deployment (`~/.claude/mcp-servers/neural-tools/`) always reports "claude-l9-template" as the project name, preventing indexer orchestration from working in other projects. Implement proper project detection that works regardless of MCP server location.

## Problem Statement

### Current Failure Mode

When neural-tools runs from global MCP location:

1. **Wrong Working Directory**:
   - MCP server runs from: `~/.claude/mcp-servers/neural-tools/`
   - ProjectContextManager calls `Path.cwd()` → returns MCP location
   - No `.mcp.json` exists there
   - Falls back to hardcoded "claude-l9-template"

2. **Impact on Users**:
   - Indexer containers not created for actual projects
   - All projects incorrectly labeled as "claude-l9-template"
   - Search returns no results (wrong project context)
   - Multiple MCP instances running, causing conflicts

3. **Evidence from Production**:
   ```
   # User working in neural-novelist:
   Actual CWD: /Users/mkr/local-coding/Systems/neural-novelist
   Reported: project: "claude-l9-template"  # WRONG!
   Result: Empty database, no search results
   ```

### Root Cause Analysis (from ADR-0098 & Grok-4)

- **ADR-0028**: Designed indexer to auto-detect from mount paths
- **ADR-0033**: Created ProjectContextManager when MCP limitations discovered
- **Break point**: Global deployment lost all project context
- **Missing piece**: `set_project_context` tool specified but not implemented

## Decision

Implement a three-pronged solution that works in both local and global deployments:

1. **Add `set_project_context` MCP tool** (immediate fix)
2. **Use environment variables** from Claude (automatic)
3. **Detect from tool usage patterns** (smart fallback)

## Detailed Design

### Phase 1: Add Missing MCP Tool (Immediate Fix)

Create the `set_project_context` tool that ADR-0033 specified:

```python
# neural-tools/src/neural_mcp/tools/set_project_context.py
from mcp.types import Tool
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Tool definition for MCP registration
tool = Tool(
    name="set_project_context",
    description="Set the active project for indexing and search operations",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to project directory"
            }
        },
        "required": ["path"]
    }
)

async def execute(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set the project context explicitly

    This solves the problem where MCP cannot detect which project
    you're working on when running from global location.
    """
    from servers.services.project_context_manager import get_project_context_manager

    project_path = arguments.get("path")
    if not project_path:
        # Auto-detect from environment or usage
        import os
        project_path = os.getenv("CLAUDE_PROJECT_DIR", os.getcwd())

    manager = await get_project_context_manager()
    result = await manager.set_project(project_path)

    logger.info(f"✅ Project context set: {result['project']} at {result['path']}")
    return result
```

### Phase 2: Environment Variable Detection

Claude can set environment variables when launching MCP. Update detection:

```python
# In ProjectContextManager.detect_project():
async def detect_project(self) -> Dict:
    """Enhanced detection with environment variable support"""

    # Strategy 0: Check CLAUDE_PROJECT_DIR first (highest priority)
    if claude_dir := os.getenv("CLAUDE_PROJECT_DIR"):
        logger.info(f"🎯 Using CLAUDE_PROJECT_DIR: {claude_dir}")
        return await self.set_project(claude_dir)

    # Strategy 1: Check for explicit PROJECT_PATH (legacy support)
    if project_path := os.getenv("PROJECT_PATH"):
        return await self.set_project(project_path)

    # Strategy 2: Only use cwd if it's NOT the MCP server location
    current_dir = Path.cwd()
    if "mcp-servers" not in str(current_dir):
        # We're running locally, cwd is valid
        project_name = await self._detect_project_name(current_dir)
        if project_name != "default":
            return await self.set_project(str(current_dir))

    # Strategy 3: Check registry for last known project
    if self.current_project and self.current_project_path:
        logger.info(f"📌 Using cached project: {self.current_project}")
        return {
            "project": self.current_project,
            "path": str(self.current_project_path),
            "method": "cached",
            "confidence": 0.7
        }

    # Strategy 4: Fail explicitly - require user action
    logger.warning("⚠️ No project detected - use set_project_context tool")
    return {
        "project": "unknown",
        "path": None,
        "method": "none",
        "confidence": 0.0,
        "error": "Project detection failed - please use set_project_context tool"
    }
```

### Phase 3: Smart Detection from Usage

Track which files are being accessed and infer project:

```python
# In MCP tool execution wrapper:
async def track_file_access(file_path: str):
    """Track file access for project inference"""
    manager = await get_project_context_manager()

    # Extract project from file path
    path = Path(file_path).resolve()

    # Walk up to find project root (has .git, package.json, etc.)
    for parent in path.parents:
        if (parent / ".git").exists():
            # Found project root
            if manager.current_project != parent.name:
                logger.info(f"📍 Auto-switching to project: {parent.name}")
                await manager.set_project(str(parent))
            break
```

### Phase 4: Register Tool in MCP Server

```python
# In neural_mcp/server.py:
from neural_mcp.tools import set_project_context

# In main():
server.add_tool(set_project_context.tool, set_project_context.execute)
```

## Implementation Plan

### Week 1: Immediate Fix
1. Implement `set_project_context` tool
2. Register in MCP server
3. Test with global deployment
4. Deploy to all users

### Week 2: Enhanced Detection
1. Add environment variable support
2. Implement usage-based detection
3. Update ProjectContextManager
4. Test multi-project scenarios

## Success Metrics

1. **Project Detection Accuracy**: >95% correct detection
2. **Zero Hardcoded Projects**: No "claude-l9-template" fallback
3. **Multi-Project Support**: Each project gets own indexer
4. **User Reports**: No more "wrong project" issues

## Testing Strategy

```bash
# Test 1: Global MCP with explicit set
cd ~/.claude/mcp-servers/neural-tools
export CLAUDE_PROJECT_DIR=/Users/mkr/neural-novelist
python run_mcp_server.py
# Should detect: neural-novelist

# Test 2: Tool usage
mcp call set_project_context '{"path": "/Users/mkr/eventfully-yours"}'
# Should switch to: eventfully-yours

# Test 3: Multiple projects
# Open two Claude windows in different projects
# Each should maintain separate context
```

## Rollback Plan

If issues arise:
1. Keep existing detection as fallback
2. Add feature flag: `ENABLE_NEW_DETECTION=false`
3. Revert to ADR-0098 state

## Dependencies

- No new dependencies required
- Uses existing ProjectContextManager (ADR-0033)
- Compatible with Docker labels (ADR-0098)
- Works with IndexerOrchestrator (ADR-0030)

## Alternatives Considered

1. **Use Docker labels only**: Rejected - doesn't solve initial detection
2. **Require manual config**: Rejected - poor UX
3. **Use Redis for state**: Rejected - adds complexity (per ADR-0098)
4. **Multiple MCP configs**: Rejected - maintenance burden

## Decision Record

This ADR is ACCEPTED because:
1. Solves immediate production issue
2. Minimal code changes required
3. Backwards compatible
4. Follows L9 principles (simple, reversible)
5. Aligns with existing ADRs (0033, 0098)

## References

- ADR-0033: Dynamic Workspace Detection (design)
- ADR-0098: Docker State Migration (state management)
- ADR-0028: Indexer Auto-Detection (original design)
- Issue: Multiple users reporting "claude-l9-template" in wrong projects