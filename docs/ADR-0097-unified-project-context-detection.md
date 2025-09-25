# ADR-0097: Unified Project Context Detection Architecture

**Status:** Proposed
**Date:** 2025-09-24
**Author:** L9 Engineering Team

## Executive Summary

Critical architectural issue: Multiple overlapping project detection mechanisms causing cross-project data contamination when using global MCP. MCP tools cannot reliably detect which directory Claude was launched from, leading to wrong project contexts and failed operations.

## The Problem (Final Correct Understanding)

### The Real Issue: MCP Protocol Doesn't Pass Client Working Directory

**The fundamental problem:** The MCP protocol doesn't provide a way for Claude to tell MCP servers which directory it was launched from.

**What happens:**
1. User launches Claude from `neural-novelist` directory
2. MCP server starts in `~/.claude/mcp-servers/neural-tools`
3. `get_consistent_project_path()` calls `os.getcwd()` → returns MCP server dir
4. Finds `.git` marker → incorrectly detects `claude-l9-template`
5. ALL tools use wrong project: search, indexing, everything

### Critical Insight: Container Detection is NOT the Solution

**Why the user is right:**
- Search tools connect to static Neo4j container (port 47687)
- They just need to pass correct `project: "neural-novelist"` parameter
- Container detection creates circular logic: containers are created AFTER project detection
- Only the reindex tool needs to find containers (to connect to their API)

**Actual flow:**
```
Project Detection → Create Indexer Container → Search with Project Filter
       ↑                                              ↑
  THE PROBLEM                                   NOT THE PROBLEM
```

### Current State Analysis

We have **17 files** using project detection across the codebase with **two competing systems**:

1. **ProjectDetector** (`neural_mcp/project_detector.py`)
   - Uses `PROJECT_PATH` and `PROJECT_NAME` env vars
   - Detects from Docker containers
   - Has global MCP safety check
   - Returns `Tuple[str, Path]`

2. **ProjectContextManager** (`servers/services/project_context_manager.py`)
   - Uses `CLAUDE_PROJECT_DIR` env var
   - Has registry persistence
   - Includes detection hints from file operations
   - Singleton pattern with `get_project_context_manager()`
   - Returns structured project info dict

### Root Cause

The MCP protocol (2025-06-18 specification) lacks a mechanism to pass the client's launch directory to servers:

```json
// MCP Initialize Request - no cwd field
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": { ... },
    "clientInfo": { ... }
    // NO working directory field!
  }
}
```

The MCP config hardcodes `cwd` to the MCP server location:
```json
// ~/.claude/mcp_config.json
{
  "neural-tools": {
    "cwd": "/Users/mkr/.claude/mcp-servers/neural-tools",  // Wrong!
    // Should be: Claude's launch directory
  }
}
```

### Impact Analysis

#### 1. Search Tools Break
- `elite_search` and `fast_search` query Neo4j with wrong project filter
- Example: Launched from `neural-novelist` but searches `claude-l9-template` data

#### 2. Reindex Operations Fail
- `project_operations.reindex` creates `ServiceContainer` with project name
- `ServiceContainer.get_indexer_url()` calls `get_consistent_project_path()`
- Returns wrong path → container mount mismatch → operation fails

#### 3. Cross-Project Contamination
- Registry fallback in global mode causes project bleed
- Example: Working in project A, getting results from project B

#### 4. Container Discovery Issues
- Multiple indexer containers running
- No clear selection criteria
- Race conditions in detection

## Industry Standards Research (2025)

### MCP Protocol Limitations
From official specification analysis:
- **No built-in client context**: Unlike LSP's `rootUri`, MCP lacks workspace context
- **Roots capability**: Exists but requires client to implement `roots/list` handler
- **Environment variables**: Primary workaround mechanism

### Best Practices from Similar Protocols

1. **Language Server Protocol (LSP)**
   ```typescript
   interface InitializeParams {
     rootUri: string;  // Mandatory project root
     workspaceFolders?: WorkspaceFolder[];
   }
   ```

2. **Debug Adapter Protocol (DAP)**
   ```typescript
   interface LaunchRequest {
     cwd: string;  // Working directory for debuggee
   }
   ```

3. **2025 MCP Implementations**
   - **Cline**: Fixed with `${workspaceFolder}` expansion (PR #2990)
   - **Cursor**: Requests `VSCODE_CWD` environment variable support
   - **K2view**: Uses enterprise data context tokens

## Proposed Solution

### Core Fix: Explicit Project Context Setting

**Key Insight:** Since Claude can't pass working directory, we need explicit project context setting.

### Phase 1: Require Explicit Project Setting

**Decision:** Make `set_project_context` the primary mechanism in global MCP mode

```python
# ADR-0097: Unified ProjectContextManager
class ProjectContextManager:
    """Single source of truth for project context detection and management"""

    async def detect_project(self) -> ProjectContext:
        """
        Unified detection strategy (CORRECTED):

        1. CLAUDE_PROJECT_DIR env (confidence: 1.0) 🔥 PRIMARY
           - Would be authoritative if Claude set it
           - Currently not implemented by Claude

        2. Explicit user setting (confidence: 1.0)
           - Via set_project_context tool
           - Most reliable in global mode

        3. Active containers as HINT (confidence: 0.8)
           - If indexer container running, suggests active project
           - NOT primary - containers are created AFTER detection
           - Useful for session continuity

        4. Registry cache (confidence: 0.5, local only)
           - Last known active project
           - DISABLED in global mode to prevent contamination

        5. Current directory (confidence: 0.3, local only)
           - Path.cwd() only works for local servers
           - Fails in global mode (returns MCP server dir)

        6. Explicit None (safest failure)
           - Force user to call set_project_context
           - Better than wrong project
        """

    async def get_current_project(self, force_refresh: bool = True) -> ProjectInfo:
        """
        Get current project context.

        Args:
            force_refresh: If True, ALWAYS re-detect from containers.
                          This ensures we catch newly created containers.
        """
        if force_refresh or self._cached_project is None:
            self._cached_project = await self.detect_project()
        return self._cached_project
```

### Phase 2: Fix Container Discovery

**Problem:** Current implementation can't parse container names correctly

```python
# Current (broken) - assumes simple format
project_name = container.name.replace("indexer-", "")

# Fixed - handles indexer-{project}-{timestamp}-{random}
parts = container.name.split("-")
if len(parts) >= 4:
    project_name = "-".join(parts[1:-2])  # Everything between prefix and suffix
```

**Enhancement:** Select most recently started container

```python
indexer_containers.sort(key=lambda x: x["started"], reverse=True)
most_recent = indexer_containers[0]
```

### Phase 3: Fix Path Resolution

**Problem:** `get_consistent_project_path()` uses wrong context

```python
# Current implementation issues:
1. Uses os.getcwd() which returns MCP server directory in global mode
2. Caches first resolution forever (even if wrong)
3. No project-specific resolution
```

**Solution:** Make path resolution project-aware

```python
async def get_project_path(project_name: str) -> Path:
    """Get path for specific project, not MCP server location"""

    # 1. Check if indexer container has mount info
    container = find_indexer_container(project_name)
    if container:
        mount = get_workspace_mount(container)
        if mount:
            return Path(mount["Source"])

    # 2. Check project registry
    registry = load_project_registry()
    if project_name in registry["projects"]:
        return Path(registry["projects"][project_name])

    # 3. Return None - don't guess
    return None
```

### Phase 4: Update MCP Tools to Handle Missing Context

**Critical:** MCP tools must handle missing project context gracefully

```python
# elite_search.py - IMPROVED
context_manager = await get_project_context_manager()
project_info = await context_manager.get_current_project()

if not project_info or not project_info.get("project"):
    return {
        "error": "No project context detected",
        "message": "Please use set_project_context tool first to specify your working project"
    }

project_name = project_info["project"]
# Now safe to query Neo4j with correct project filter
```

**Tools requiring this pattern:**
- `elite_search.py` - Fail gracefully without project
- `fast_search.py` - Same pattern
- `semantic_search.py` - Same pattern
- `project_operations.py` - Especially critical for reindex
- `dependency_analysis.py` - Must have correct project
- `schema_management.py` - Schema is per-project
- All other MCP tools that query data

### Phase 5: Client-Side Fix

**Critical:** Modify Claude's launch script to set environment variable

```bash
# In Claude's MCP launcher (pseudocode)
export CLAUDE_PROJECT_DIR="$(pwd)"  # User's actual directory
exec python3 /path/to/mcp_server.py
```

## Migration Plan

### Step 1: Add Compatibility Layer
```python
# Temporary backward compatibility
async def get_user_project():
    """Deprecated: Use get_project_context_manager()"""
    manager = await get_project_context_manager()
    project_info = await manager.get_current_project()
    return project_info["project"], Path(project_info["path"])
```

### Step 2: Update All Consumers
Files requiring updates (17 total):
- `elite_search.py` - Use manager.get_current_project()
- `fast_search.py` - Same update
- `project_operations.py` - Pass project path to reindex
- `schema_backfill.py` - Use unified context
- `set_project_context.py` - Update to use manager
- (12 more files...)

### Step 3: Remove Deprecated Code
- Delete `neural_mcp/project_detector.py`
- Remove compatibility shims
- Update tests

## Testing Requirements

### Unit Tests
```python
async def test_project_detection_priority():
    """Verify detection strategy order"""
    # Set CLAUDE_PROJECT_DIR
    # Should use it even if containers exist

async def test_global_mode_safety():
    """Ensure no registry fallback in global"""
    # Simulate global MCP context
    # Verify returns None instead of wrong project

async def test_container_name_parsing():
    """Test complex container name formats"""
    # indexer-neural-novelist-1758777614-4cb1
    # Should extract: neural-novelist
```

### Integration Tests
```python
async def test_cross_project_isolation():
    """Verify no data bleed between projects"""
    # Create two projects with indexers
    # Query from each context
    # Verify correct results

async def test_reindex_with_correct_path():
    """Ensure reindex uses right container"""
    # Set project context
    # Call reindex operation
    # Verify correct indexer receives request
```

## Security Considerations

### 1. Cross-Project Data Isolation
- **Risk:** User sees data from wrong project
- **Mitigation:** Fail-safe to None when uncertain

### 2. Container Hijacking
- **Risk:** Malicious container pretends to be indexer
- **Mitigation:** Verify container labels and ownership

### 3. Path Traversal
- **Risk:** Malicious project paths escape boundaries
- **Mitigation:** Validate paths, use Path.resolve()

## Performance Impact

### Positive
- Single detection per session (cached)
- Fewer Docker API calls
- Reduced registry I/O

### Negative
- Initial detection may take 100-200ms
- Container inspection adds latency

### Mitigation
- Cache detection result for session
- Background refresh after 5 minutes
- Parallel container inspection

## Rollback Plan

If issues arise:
1. Revert to dual-system temporarily
2. Add feature flag: `USE_UNIFIED_CONTEXT=false`
3. Restore `ProjectDetector` from git
4. Monitor and fix issues
5. Re-attempt migration

## Success Metrics

- **Zero** cross-project data contamination incidents
- **100%** of reindex operations succeed
- **<200ms** project detection latency
- **95%+** correct project detection rate

## Timeline

- **Day 1-2:** Implement unified manager
- **Day 3:** Update all consumers
- **Day 4:** Testing and validation
- **Day 5:** Deploy to global MCP
- **Week 2:** Remove deprecated code

## Decision

**REVISED APPROACH:** Require explicit project context setting in global MCP mode.

**Rationale:**
1. Container detection creates circular dependencies
2. MCP protocol lacks working directory passing
3. Explicit setting is more reliable than guessing
4. Only reindex tool needs container discovery (for API connection)

## References

- MCP Specification 2025-06-18
- Cline PR #2990 (workspace folder fix)
- LSP Specification (initialization pattern)
- ADR-0060 (container naming)
- ADR-0029 (project isolation)