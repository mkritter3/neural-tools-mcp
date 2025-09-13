# ADR-0043: MCP Server Project Context Lifecycle Management

**Status**: Proposed
**Date**: September 13, 2025
**Author**: L9 Engineering Team

## Context

Our MCP (Model Context Protocol) server is a long-running process that maintains persistent connections to Neo4j and Qdrant databases. When users switch between projects by changing directories and restarting Claude (the client), the server retains its previous project context, leading to critical failures:

1. **Wrong Project Detection**: Server reports the previous project instead of current directory
2. **Indexing Failures**: Reindexing crashes with "Server disconnected without sending a response"
3. **Database Connection Confusion**: Async connections remain bound to the old project's context
4. **Manual Intervention Required**: Users must manually call `set_project_context` every time

### Evidence of the Problem

When a user switched from `claude-l9-template` to `neural-novelist`:
```
User: "can you use the neural tools to get project understanding?"
Result: Shows "claude-l9-template" (wrong project!)

User manually runs: set_project_context(path: "/Users/mkr/local-coding/Systems/neural-novelist")
Result: Shows "neural-novelist" but with 0 files indexed

User tries: reindex_path(...)
Result: "Server disconnected without sending a response" (server crash!)
```

## Problem Analysis

### Root Cause (Identified by Gemini & Grok)

The MCP server maintains persistent state across client connections:

1. **Server Lifecycle Mismatch**: The MCP server process outlives client sessions
2. **Shallow Context Updates**: `set_project_context` only updates the project name variable
3. **Connection Persistence**: Neo4j AsyncGraphDatabase and Qdrant connections remain bound to old project
4. **State Inconsistency**: Project name changes but database connections don't

### The Cascade Failure

```
1. User changes directory to new project
2. User restarts Claude (client only)
3. MCP server (still running) retains old project state
4. Client connects, server reports wrong project
5. User forces context switch via set_project_context
6. Project name updates but connections don't
7. Indexer tries to write with mismatched context
8. Async operation fails → unhandled exception → server crash
```

## Decision

Implement comprehensive project context lifecycle management with proper teardown and rebuild of all stateful connections when switching projects.

## Solution Architecture

### Files to Update (CRITICAL - Update EXISTING files only)

1. **neural-tools/src/servers/services/service_container.py** - Add teardown method
2. **neural-tools/src/servers/services/project_context_manager.py** - Add switch_project methods
3. **neural-tools/src/neural_mcp/neural_server_stdio.py** - Update set_project_context handler

⚠️ **DO NOT CREATE NEW FILES** - All changes go into existing files

### 1. Project Detection Strategy

**On Server Startup**: Detect from current working directory
**On Client Connection**: Validate project matches, auto-switch if different
**On Manual Switch**: Full teardown and rebuild

### 2. ServiceContainer Teardown Method (VALIDATED by Gemini & Grok)

```python
# FILE: neural-tools/src/servers/services/service_container.py
# ADD this method to EXISTING ServiceContainer class

async def teardown(self):
    """Gracefully close all managed connections and resources"""
    logger.info("Tearing down service container...")

    # Close Neo4j async driver (validated by Gemini)
    try:
        if self.neo4j and hasattr(self.neo4j, 'client') and self.neo4j.client.driver:
            await self.neo4j.client.driver.close()
            logger.info("Neo4j connection closed")
    except Exception as e:
        logger.error(f"Failed to close Neo4j connection: {e}")

    # Close Qdrant client (Grok: handle both sync and async)
    try:
        if hasattr(self, 'qdrant') and self.qdrant:
            if hasattr(self.qdrant, 'close'):
                # Check if async client
                if 'Async' in str(type(self.qdrant)):
                    await self.qdrant.close()
                else:
                    self.qdrant.close()
                logger.info("Qdrant client closed")
    except Exception as e:
        logger.error(f"Failed to close Qdrant client: {e}")

    # Clear caches
    if hasattr(self, 'cache'):
        self.cache.clear()
        logger.info("Caches cleared")
```

### 3. ProjectContextManager Updates (VALIDATED by Gemini & Grok)

```python
# FILE: neural-tools/src/servers/services/project_context_manager.py
# ADD these methods to EXISTING ProjectContextManager class

def __init__(self):
    # ... existing init code ...
    self.switch_lock = asyncio.Lock()  # Add lock for concurrent safety
    self.container = None  # Store ServiceContainer here (Gemini: correct pattern)

async def switch_project(self, new_project_path: str):
    """Complete project context switch with full teardown/rebuild"""
    async with self.switch_lock:  # Grok: Critical for preventing race conditions
        # Phase 1: Teardown
        await self._teardown_current_context()

        # Phase 2: Update
        self.current_project = self._detect_project(new_project_path)

        # Phase 3: Rebuild
        await self._rebuild_context()

        # Gemini: Must return project dict for handler
        return self.current_project

async def _teardown_current_context(self):
    """Delegate teardown to ServiceContainer for encapsulation"""
    if self.container:
        await self.container.teardown()  # Gemini: Better encapsulation
        self.container = None

async def _rebuild_context(self):
    """Initialize fresh ServiceContainer with validation"""
    self.container = ServiceContainer(
        project_name=self.current_project['name'],
        project_path=self.current_project['path']
    )
    await self.container.initialize()

    # Grok: Validate connections after rebuild
    if hasattr(self.container, 'verify_connections'):
        if not await self.container.verify_connections():
            raise RuntimeError(f"Failed to initialize connections for {self.current_project['name']}")
```

### 3. Client Connection Validation

```python
async def handle_client_connection(client_info):
    """Validate project context on each client connection"""
    client_cwd = client_info.get('cwd')
    client_project = detect_project_from_path(client_cwd)

    if client_project != context_manager.current_project['name']:
        logger.info(f"Project mismatch detected: {context_manager.current_project['name']} -> {client_project}")
        await context_manager.switch_project(client_cwd)
```

### 4. Enhanced set_project_context Handler (VALIDATED)

```python
# FILE: neural-tools/src/neural_mcp/neural_server_stdio.py
# UPDATE the EXISTING handle_set_project_context function

@server.call_handler("set_project_context")
async def handle_set_project_context(params):
    """Handle manual project context switch with full lifecycle management"""
    try:
        path = params.get('path')
        if not path:
            # Auto-detect from current working directory
            path = os.getcwd()

        # Use the manager's switch_project method (returns project dict)
        new_project = await context_manager.switch_project(path)

        # Return detailed status
        return {
            "status": "success",
            "project": new_project['name'],  # Gemini: new_project is dict
            "path": new_project['path'],
            "connections": {
                "neo4j": "reconnected",
                "qdrant": "reconnected",
                "indexer": "ready"
            },
            "message": f"Successfully switched to project: {new_project['name']}"
        }

    except Exception as e:
        logger.error(f"Failed to switch project context: {e}", exc_info=True)
        # Grok: Graceful error handling
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to switch project. Server may need restart."
        }
```

### 5. Indexer Integration

```python
async def reindex_path(params):
    """Reindex with proper context validation"""
    # Verify we have valid context
    if not context_manager.container:
        return {"status": "error", "message": "No active project context"}

    # Use the current container's connections
    indexer = context_manager.container.indexer

    try:
        await indexer.index_path(params['path'], params.get('recursive', True))
        return {"status": "success", "message": "Indexing completed"}
    except Exception as e:
        # Don't crash the server - handle gracefully
        logger.error(f"Indexing failed: {e}", exc_info=True)
        return {"status": "error", "message": f"Indexing failed: {str(e)}"}
```

## Implementation Plan

### Phase 1: Immediate Fix (Critical)
1. Implement proper teardown in `set_project_context`
2. Add connection closing for AsyncGraphDatabase
3. Clear all cached state

### Phase 2: Robust Context Management
1. Implement ProjectContextManager class
2. Add client connection validation
3. Auto-switch on project mismatch

### Phase 3: Error Recovery
1. Add graceful error handling
2. Implement connection retry logic
3. Add health checks after switch

## Testing Strategy

### Unit Tests
```python
async def test_project_context_switch():
    """Test complete teardown and rebuild"""
    manager = ProjectContextManager()

    # Initial project
    await manager.switch_project("/path/to/project1")
    container1 = manager.container

    # Switch project
    await manager.switch_project("/path/to/project2")
    container2 = manager.container

    # Verify different containers
    assert container1 != container2

    # Verify old connections closed
    assert container1.neo4j.client.driver.closed
```

### Integration Tests
1. Start server in project A
2. Connect client from project B
3. Verify auto-switch occurs
4. Test indexing works after switch
5. Verify no data leakage between projects

## Consequences

### Positive
- ✅ Automatic project detection on directory change
- ✅ No manual intervention required
- ✅ Clean connection management
- ✅ No resource leaks
- ✅ Prevents data confusion between projects
- ✅ Indexing works reliably after project switch

### Negative
- ⚠️ Slight latency on project switch (connection rebuild)
- ⚠️ More complex state management
- ⚠️ Potential for brief unavailability during switch

### Risks
- Connection rebuild failures could leave server in broken state
- Rapid project switching could cause thrashing
- Async teardown must be carefully managed to avoid deadlocks

## Migration Path

1. **Deploy fix to global MCP**: Update global neural-tools with new context management
2. **Add backward compatibility**: Ensure old clients still work
3. **Monitor for issues**: Log all context switches and failures
4. **Gradual rollout**: Test with power users first

## Success Metrics

- Zero manual `set_project_context` calls required
- 100% successful indexing after project switches
- No "Server disconnected" errors
- Project detection accuracy: 100%
- Context switch time: <2 seconds

## AI Validation Feedback

### Gemini's Validation ✅
1. **Correct files identified** - Yes, updating existing files is correct approach
2. **Async teardown correct** - Neo4j AsyncGraphDatabase.close() is properly awaited
3. **Lock is critical** - Prevents concurrent switches and race conditions
4. **Qdrant needs close** - Should explicitly close to prevent resource leaks
5. **Container in manager** - Correct pattern for encapsulation
6. **Add ServiceContainer.teardown()** - Better encapsulation than manager reaching into internals
7. **Return value fix** - switch_project must return project dict for handler

### Grok's Validation ✅
1. **Qdrant close handling** - Added check for sync vs async client types
2. **Graceful error handling** - Try-catch blocks prevent partial failures
3. **Container storage pattern** - Confirmed correct to store in ProjectContextManager
4. **Connection validation** - Added post-rebuild verification
5. **Race condition prevention** - Lock sufficient, but watch for stray async tasks
6. **Logging enhancement** - Added context to teardown/rebuild logs

### Critical Implementation Notes
- ⚠️ **DO NOT CREATE NEW FILES** - All changes go into existing files
- ⚠️ **Graceful teardown** - Failures in one component shouldn't block others
- ⚠️ **Validate after rebuild** - Ensure connections are working before proceeding

## References

- ADR-0029: Project Isolation via Properties
- ADR-0034: Dynamic Project Detection
- ADR-0037: Configuration Priority Standard
- ADR-0042: GraphRAG Architecture (async Neo4j fix)
- Gemini's analysis of stale state issue
- Grok's recommendations for async connection management

---

*Confidence: 98%* (Validated by both Gemini and Grok)
*Assumptions: MCP server process persists across client restarts, AsyncGraphDatabase requires explicit close*