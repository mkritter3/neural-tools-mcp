# ADR-052: Automatic Indexer Initialization

**Status**: Proposed
**Date**: September 15, 2025
**Author**: L9 Engineering Team

## Context

The neural-tools MCP indexer currently requires a fragmented 3-step manual initialization process before it can index files in a new project directory:

1. **set_project_context** - Establishes which project to work on
2. **schema_init** - Detects and initializes project type schema
3. **reindex_path** - Finally able to index files

This violates L9 engineering principles of seamless operation and creates poor user experience. When users attempt to index a project, they receive cryptic empty error messages and must discover the correct sequence through trial and error.

### Current User Experience

```
User: "Index this project"
> reindex_path("./")
Error: {"status": "error", "error": ""}

User: "Why didn't that work? Let me try setting context..."
> set_project_context("/path/to/project")
Success

User: "Now let me try indexing again..."
> reindex_path("./")
Error: {"status": "error", "error": ""}

User: "Still failing? Maybe I need schema..."
> schema_init(auto_detect=true)
Success

User: "Finally, let me try indexing..."
> reindex_path("./")
Success: Indexing queued
```

## Problem Analysis

### Root Cause
The global `PROJECT_CONTEXT` variable in `neural_server_stdio.py` is initialized to `None` and only gets created when `set_project_context` is explicitly called. When `reindex_path_impl` tries to use it:

```python
# Line 2025 in neural_server_stdio.py
project_context = await PROJECT_CONTEXT.get_current_project()  # AttributeError if None
```

This causes an AttributeError that's caught but returns empty error messages, providing no guidance to users.

### Architecture Issues Identified

1. **Global State Dependency**: Heavy reliance on global `PROJECT_CONTEXT` variable
2. **Fragmented Initialization**: Three separate tools must be called in correct order
3. **Silent Failures**: Empty error messages provide no actionable guidance
4. **No Auto-Detection**: Despite having detection logic, it's not triggered automatically
5. **Missing Lazy Initialization**: No fallback when prerequisites aren't met

## Decision

Implement **safe lazy initialization** in the operation phase to handle all prerequisites transparently WITHOUT breaking Claude's MCP lifecycle management. The system should:

1. **Respect Claude's lifecycle** - No changes to initialize/initialized handshake
2. **Use lazy initialization** - Initialize PROJECT_CONTEXT on first tool use
3. **Leverage existing captures** - Use INITIAL_WORKING_DIRECTORY already captured
4. **Auto-initialize schema** if not present for the project
5. **Provide clear error messages** if initialization fails

## Critical Constraint: MCP Lifecycle

Per the MCP 2025-06-18 specification, the lifecycle is strictly defined:
1. **Initialization Phase**: Claude sends `initialize` request, server responds, Claude sends `initialized` notification
2. **Operation Phase**: Normal tool operations begin
3. **Shutdown Phase**: Connection termination

**We MUST NOT modify the initialization phase** - our MCP is working and we don't want to break it.

## Solution Design

### The Safe Fix: Lazy Initialization in Operation Phase

After deep analysis with Grok 4, we identified the safest initialization point: **`get_project_context`** which is called from `handle_call_tool` (line 999) during the **operation phase**, AFTER Claude's handshake completes.

```python
# In neural_server_stdio.py around line 614
async def get_project_context(arguments: Dict[str, Any]):
    """Get or auto-initialize project context - SAFE lazy initialization"""
    global PROJECT_CONTEXT

    # Safe lazy initialization - happens AFTER Claude handshake
    if PROJECT_CONTEXT is None:
        from servers.services.project_context_manager import ProjectContextManager
        PROJECT_CONTEXT = ProjectContextManager()

        # Use the directory captured at startup (already exists at line 565)
        if INITIAL_WORKING_DIRECTORY and '/.claude/mcp-servers/' not in str(INITIAL_WORKING_DIRECTORY):
            # Valid project directory from launch
            project_path = Path(INITIAL_WORKING_DIRECTORY)
            project_name = project_path.name
            await PROJECT_CONTEXT.set_project(project_path, project_name)
            logger.info(f"✅ Auto-initialized PROJECT_CONTEXT from launch dir: {INITIAL_WORKING_DIRECTORY}")

            # Auto-initialize schema too
            container = await state.get_service_container(project_name)
            if hasattr(container, 'schema_manager'):
                schema_mgr = container.schema_manager
                if not await schema_mgr.has_schema():
                    await schema_mgr.auto_init()
                    logger.info(f"✅ Auto-initialized schema for {project_name}")
        else:
            # Fall back to auto-detection for global installs
            detected = await PROJECT_CONTEXT.detect_project()
            logger.info(f"✅ Auto-detected project: {detected.get('project')}")

    # Now PROJECT_CONTEXT is guaranteed to exist
    try:
        project_context = await PROJECT_CONTEXT.get_current_project()
        return project_context
    except Exception as e:
        logger.error(f"Failed to get project context: {e}")
        return {
            "project": "default",
            "path": str(Path.cwd()),
            "error": str(e)
        }
```

### Enhanced reindex_path Implementation

```python
async def reindex_path_impl(path: str) -> List[types.TextContent]:
    """Enhanced reindex with automatic initialization"""

    if not path:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "Path is required for reindexing"}, indent=2)
        )]

    try:
        # get_project_context returns tuple: (project_name, container, retriever)
        # This will auto-initialize PROJECT_CONTEXT if needed (safe lazy init)
        project_name, container, _ = await get_project_context({})

        if not container:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "error": "Failed to initialize service container",
                    "suggestion": "Try set_project_context with explicit path"
                }, indent=2)
            )]

        # Get project path from PROJECT_CONTEXT
        if PROJECT_CONTEXT:
            context = await PROJECT_CONTEXT.get_current_project()
            project_path = context.get('path', os.getcwd())
        else:
            project_path = os.getcwd()

        # Container already obtained from get_project_context

        # Continue with normal reindex flow...
        logger.info(f"Reindexing {path} for project: {project_name}")
        # ... rest of implementation
```

### Why This Solution is 100% Safe

1. **No Lifecycle Interference**: Initialization happens during operation phase, not during Claude's handshake
2. **Uses Existing Infrastructure**: INITIAL_WORKING_DIRECTORY is already captured in server_lifespan (line 565)
3. **Atomic Operation**: PROJECT_CONTEXT null check prevents race conditions
4. **Backward Compatible**: Existing manual set_project_context still works
5. **Zero Breaking Changes**: All existing code continues to function

## Implementation Plan

### Phase 1: Quick Win (1-2 hours)
- [ ] Add auto-initialization to `reindex_path_impl`
- [ ] Improve error messages with actionable suggestions
- [ ] Add logging for debugging initialization flow

### Phase 2: MCP Startup Integration (2-4 hours)
- [ ] Capture launch directory in `server_lifespan`
- [ ] Initialize PROJECT_CONTEXT at startup
- [ ] Auto-detect and cache project from launch directory
- [ ] Pre-initialize schema if project detected

### Phase 3: Refactor Global State (1 day)
- [ ] Replace global PROJECT_CONTEXT with dependency injection
- [ ] Pass context explicitly through function parameters
- [ ] Add proper async context management
- [ ] Improve testability with context mocking

### Phase 4: Enhanced Auto-Detection (2 days)
- [ ] Implement parent directory traversal for project roots
- [ ] Add support for monorepo detection
- [ ] Cache successful detections in ~/.graphrag/
- [ ] Support multiple project contexts in same session

## Benefits

### User Experience
- **Zero-step initialization**: Just call `reindex_path` and it works
- **Clear error messages**: Users know exactly what went wrong
- **Automatic schema detection**: No manual project type specification
- **Launch-aware context**: MCP knows your project from where you start Claude

### Technical Benefits
- **Reduced coupling**: Components self-initialize as needed
- **Better testability**: No global state dependencies
- **Improved maintainability**: Clear initialization flow
- **Future-proof**: Supports automation and CI/CD pipelines

## Risks and Mitigations

### Risk: Auto-detection might choose wrong project
**Mitigation**: Provide clear logging of detected project and easy override via `set_project_context`

### Risk: Schema auto-init might pick wrong type
**Mitigation**: Use comprehensive detection heuristics and allow manual `schema_init` override

### Risk: Performance impact from repeated checks
**Mitigation**: Cache initialization state and use lazy initialization patterns

## Implementation Notes (Critical Bug Fixed)

### TypeError Issue (Fixed Sept 19, 2025)

**Bug**: Initial implementation incorrectly treated `get_project_context()` return value as a dictionary when it actually returns a tuple.

**Error**: `TypeError: tuple indices must be integers or slices, not str`

**Root Cause**: The `get_project_context()` function returns `(project_name, container, retriever)` tuple, not a dictionary with 'project' and 'path' keys.

**Fix**: Updated all functions to correctly unpack the tuple:
```python
# WRONG (caused TypeError)
project_context = await get_project_context({})
project_name = project_context['project']  # TypeError!

# CORRECT (fixed)
project_name, container, _ = await get_project_context({})
```

This bug affected all indexing operations and was documented in multiple project reports before being fixed.

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Steps to index new project | 3 | 1 |
| Empty error messages | Common | None |
| User confusion reports | High | Low |
| Automation compatibility | Poor | Excellent |
| Time to first successful index | 2-5 min | <30 sec |

## Decision Outcome

**Accepted** - The current 3-step manual process is a critical UX failure that violates L9 engineering standards. Automatic initialization will dramatically improve user experience while maintaining project isolation and schema flexibility. The implementation is straightforward and low-risk.

## References

- Issue: Users report "cryptic errors" when trying to index projects
- Grok 4 Analysis: Identified PROJECT_CONTEXT initialization as root cause
- Related ADRs:
  - ADR-0033: Dynamic Project Context Management
  - ADR-0020: Per-Project Custom GraphRAG Schemas
  - ADR-0050: Neo4j-Qdrant Chunk Synchronization

## Future Considerations

### Multi-Project Sessions
Consider supporting multiple active projects in a single MCP session, switching context based on file paths being indexed.

### Cloud/Remote Projects
Extend auto-detection to support remote repositories and cloud storage projects.

### Project Templates
Create initialization templates for common project types to speed up schema setup.