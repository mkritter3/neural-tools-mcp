# ADR-0058: Indexer Race Condition from Circular Initialization

**Status**: Accepted
**Date**: September 20, 2025
**Author**: L9 Engineering Team

## Executive Summary

The indexer regression is caused by a **circular dependency race condition** introduced September 19, 2025 when fixing import issues. The isinstance check removal exposed a timing bug where PROJECT_CONTEXT initialization creates a ServiceContainer that needs PROJECT_CONTEXT, leading to containers without discovery service, causing the removal logic to run instead of reuse logic.

## Context

### What Was Working (ADR-0052 - Sept 15, 2025)

ADR-0052 established automatic indexer initialization with:
- Lazy initialization on first use
- Automatic project detection from launch directory
- Self-initializing PROJECT_CONTEXT
- Zero-step indexing (just call `reindex_path`)
- Clean container lifecycle management

**Success metrics from ADR-0052:**
- Time to first index: <30 seconds
- Steps required: 1 (down from 3)
- User confusion: Low
- Container management: Simple and working

### The Regression (Sept 13-19, 2025)

Multiple ADRs and implementations introduced conflicting container management strategies:

1. **ADR-0044 (Sept 13)**: Container Discovery Service - "reuse existing containers"
2. **ADR-0048 (Sept 19)**: Idempotent management - "remove existing containers"
3. **ADR-0049 (Sept 19)**: Dynamic port discovery - adds more discovery logic

## Problem Analysis

### Root Cause: Circular Dependency in Initialization

The actual root cause is NOT competing strategies - the code has proper branching:

**IndexerOrchestrator.ensure_indexer (lines 115-184):**
```python
if self.discovery_service and self.context_manager:
    # Use discovery service (lines 133-155)
    container_info = await self.discovery_service.get_or_create_container(...)
    return container_info['container_id']  # RETURNS HERE!

# Only runs if NO discovery service (lines 161-184)
logger.warning("⚠️ No discovery service, using legacy container creation")
# Removal logic here...
```

The removal logic (161-184) only runs when discovery_service is None!

### The REAL Race Condition

The circular dependency occurs during PROJECT_CONTEXT initialization:

```python
# neural_server_stdio.py line 622-640
if PROJECT_CONTEXT is None:
    PROJECT_CONTEXT = ProjectContextManager()  # Creates new instance
    # ... initialization ...
    container = await state.get_service_container(project_name)  # Line 640!

# get_project_container line 272
container = ServiceContainer(context_manager=PROJECT_CONTEXT, ...)
```

**The Problem Timeline:**
1. First tool call triggers `get_project_context()`
2. PROJECT_CONTEXT is None, so creates new ProjectContextManager
3. During initialization, calls `state.get_service_container()` (line 640)
4. This creates ServiceContainer with PROJECT_CONTEXT (line 272)
5. But PROJECT_CONTEXT is still being initialized!
6. ServiceContainer gets None or partially initialized context_manager
7. IndexerOrchestrator created without proper context_manager
8. No discovery_service, falls back to removal logic
9. Container conflicts!

### Timeline of Breaking Changes

```
Sept 13, 2025: ADR-0044 introduces ContainerDiscoveryService ✅ Working
Sept 15, 2025: ADR-0052 achieves automatic initialization ✅ Working
Sept 19, 12:22: ADR-0054 adds event sourcing with relative imports
                from .event_store import SyncEventStore ✅ Still working
Sept 19, 20:23: Commit 36fb169 "fix: Resolve import chain issues"
                ⚠️ BREAKING: Commented out isinstance check
                This exposed the circular dependency race condition!
Sept 19, 22:27: Commit c419f7f "fix: Correct import paths"
Sept 20, 08:56: ADR-0057 collection naming (unrelated)
Sept 20, Morning: User reports indexer failures
```

### The Module Identity Crisis That Led to the Bug

**What Actually Happened:**

1. **ADR-0054 introduced relative imports** (`from .event_store import ...`)
2. **Module identity conflict occurred** because services use `sys.path.insert()`:
   ```python
   # Same class loaded twice by Python:
   from servers.services.project_context_manager import ProjectContextManager  # Type A
   from .project_context_manager import ProjectContextManager                  # Type B
   # Python considers these DIFFERENT classes!
   ```

3. **isinstance check started failing**:
   ```python
   # PROJECT_CONTEXT created with absolute import (Type A)
   PROJECT_CONTEXT = ProjectContextManager()

   # isinstance check used relative import (Type B)
   from .project_context_manager import ProjectContextManager
   isinstance(context_manager, ProjectContextManager)  # FALSE! Different types!
   ```

4. **Quick fix removed the safety guard**:
   - Comment: "Skip isinstance check - causes module identity issues"
   - This "fixed" deployment but exposed the latent circular dependency bug
   - The isinstance check was actually PROTECTING against bad initialization!

### Why It Worked Before

The system worked until Sept 19 evening because:
1. **The isinstance check was a safety guard** - It prevented ServiceContainer from accepting partially initialized PROJECT_CONTEXT
2. **When removed, the circular dependency was exposed** - No type checking = silent failures
3. **Race condition became active** - Sometimes works if PROJECT_CONTEXT initializes fast enough
4. **Intermittent failures** - Different Claude sessions trigger different initialization orders

## Additional Issues Introduced

### 1. Missing File Types

**Regression**: `.dart` files not in watch_patterns (Flutter support lost)

```python
# indexer_service.py lines 252-256
self.watch_patterns = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go',
    '.rs', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php',
    '.md', '.yaml', '.yml', '.json', '.toml', '.txt', '.sql'
    # Missing: .dart, .kt, .swift, .vue, .svelte
}
```

### 2. Misleading Status Messages

**Regression**: "Indexing queued" shown even when container fails to start

The system reports success before verifying container actually started, leading to user confusion when nothing happens.

### 3. Configuration Drift

**Regression**: Hardcoded values override environment variables

Multiple services now have different defaults for the same configuration, breaking ADR-0037 (Configuration Priority Standard).

## Decision

**Fix both the import consistency AND the circular dependency:**

### The Complete Fix Requires Three Parts

1. **Fix import consistency** to prevent module identity issues
2. **Restore the isinstance check** as a safety guard
3. **Break the circular dependency** in initialization

### Part 1: Fix Import Consistency

```python
# Use absolute imports everywhere - no mixing!
# service_container.py
from servers.services.project_context_manager import ProjectContextManager
from servers.services.indexer_orchestrator import IndexerOrchestrator

# NOT: from .project_context_manager import ...
```

### Part 2: Restore isinstance Check

```python
# service_container.py - with consistent imports
if context_manager:
    from servers.services.project_context_manager import ProjectContextManager
    if not isinstance(context_manager, ProjectContextManager):
        raise TypeError("context_manager must be a ProjectContextManager instance")
    self.context_manager = context_manager
```

### Part 3: Break Circular Dependency

```python
# In get_project_context() - neural_server_stdio.py
if PROJECT_CONTEXT is None:
    PROJECT_CONTEXT = ProjectContextManager()
    # Complete ALL initialization first
    if INITIAL_WORKING_DIRECTORY:
        await PROJECT_CONTEXT.set_project(...)
    # DO NOT create container here! (Remove line 640)

# Later, when container is actually needed:
container = await state.get_service_container(project_name)
# Now PROJECT_CONTEXT is fully initialized
```

**Recommendation**: Implement all three parts for a robust fix.

## Solution Design

### Immediate Fix (Break Circular Dependency)

1. **Remove container creation from PROJECT_CONTEXT initialization** (line 640)
2. **Ensure PROJECT_CONTEXT is fully initialized before use**
3. **Add missing file extensions** (.dart, .vue, .svelte, etc.)
4. **Fix status reporting** to verify container started

### Correct Implementation

```python
# neural_server_stdio.py - get_project_context()
async def get_project_context(arguments: Dict[str, Any]):
    global PROJECT_CONTEXT

    if PROJECT_CONTEXT is None:
        from servers.services.project_context_manager import ProjectContextManager
        PROJECT_CONTEXT = ProjectContextManager()

        if INITIAL_WORKING_DIRECTORY:
            project_path = Path(INITIAL_WORKING_DIRECTORY)
            project_name = project_path.name
            await PROJECT_CONTEXT.set_project(project_path, project_name)
            logger.info(f"✅ PROJECT_CONTEXT initialized: {project_name}")

        # DO NOT CREATE CONTAINER HERE! (Remove line 640)
        # Let it be created lazily when actually needed

    # Now get the context
    project_context = await PROJECT_CONTEXT.get_current_project()

    # Get container separately (it will have proper context now)
    project_name = project_context.get("project", DEFAULT_PROJECT_NAME)
    container = await state.get_service_container(project_name)

    return project_name, container, None
```

### File Type Completeness

```python
# Comprehensive file watching
WATCH_PATTERNS = {
    # Languages
    '.py', '.js', '.ts', '.jsx', '.tsx',   # Python, JavaScript, TypeScript
    '.java', '.kt', '.scala',               # JVM languages
    '.go', '.rs', '.c', '.cpp', '.h',       # Systems languages
    '.cs', '.vb', '.fs',                    # .NET languages
    '.rb', '.php', '.perl',                 # Scripting languages
    '.swift', '.m', '.dart',                # Mobile (iOS, Flutter)
    '.r', '.jl', '.m',                      # Data science

    # Web frameworks
    '.vue', '.svelte', '.astro',            # Modern web components

    # Config/Data
    '.json', '.yaml', '.yml', '.toml',      # Configuration
    '.xml', '.ini', '.env',                 # Legacy config

    # Documentation
    '.md', '.rst', '.txt',                  # Docs

    # Database
    '.sql', '.graphql', '.prisma'           # Query languages
}
```

## Implementation Priority

### Phase 1: Stop the Bleeding (1 hour)
- [ ] Disable conflicting removal logic
- [ ] Add .dart to watch_patterns
- [ ] Fix status message timing

### Phase 2: Architectural Decision (2 hours)
- [ ] Choose ephemeral OR reuse (not both)
- [ ] Remove unused service (Discovery or Orchestrator cleanup)
- [ ] Update all callers to use single service

### Phase 3: Restore Full Functionality (4 hours)
- [ ] Implement complete file type list
- [ ] Add proper error reporting
- [ ] Verify ADR-0052 metrics restored

## Metrics to Track

| Metric | Current (Broken) | Target (ADR-0052) |
|--------|------------------|-------------------|
| Container conflicts | Common | Zero |
| Indexing success rate | ~60% | >95% |
| Time to first index | 2-5 min or timeout | <30 sec |
| File types supported | 22 | 40+ |
| User confusion | High | Low |

## Lessons Learned

### 1. Module Identity Issues Are Dangerous

Mixing relative imports with `sys.path` manipulation creates subtle bugs where Python loads the same module twice as different types. This breaks isinstance checks and type safety.

### 2. Safety Guards Should Not Be Removed

The isinstance check was protecting against a latent circular dependency bug. Removing it to "fix" deployment exposed a worse problem. Always understand WHY a check exists before removing it.

### 3. Circular Dependencies in Initialization Are Race Conditions

When initialization of A requires B, but B needs A, you have a race condition. Sometimes it works (if A initializes fast), sometimes it fails. This makes bugs intermittent and hard to debug.

### 4. Quick Fixes Can Expose Latent Bugs

The quick fix (removing isinstance check) solved the immediate problem but exposed a deeper architectural issue that had been hidden. Always investigate root causes.

### 5. Import Consistency Is Critical

All modules should use the same import style (absolute vs relative). Mixing styles with sys.path manipulation leads to module identity crises.

## Decision Outcome

**Accepted** - The root cause is a combination of module identity issues and a circular dependency:

1. **Module identity crisis** caused isinstance check to fail (ADR-0054's relative imports)
2. **Quick fix removed the safety guard** (commenting out isinstance check)
3. **Exposed latent circular dependency** (PROJECT_CONTEXT creates container that needs PROJECT_CONTEXT)
4. **Results in race condition** (sometimes works, sometimes fails)

The fix requires addressing both issues: consistent imports AND breaking the circular dependency. The discovery service and removal logic are actually working correctly - they just need proper initialization to function.

## References

- ADR-0052: Automatic Indexer Initialization (working state)
- ADR-0044: Container Discovery Service (introduced reuse)
- ADR-0048: Idempotent Container Management (introduced removal conflict)
- ADR-0049: Dynamic Port Discovery (added complexity)
- Issue Report: vesikaa project indexing failures (Sept 20, 2025)

## Testing Criteria & Exit Conditions

### Test Scenarios

#### Scenario 1: Fresh MCP Session (Cold Start)
```bash
# Kill all MCP processes
# Start Claude from project directory
# First command: reindex_path("./")
```
**Pass Criteria**:
- ✅ No "container marked for removal" errors
- ✅ Indexing completes within 30 seconds
- ✅ Qdrant shows points added
- ✅ Neo4j shows nodes created

#### Scenario 2: Multiple Rapid Reindex Calls
```python
# Call reindex 5 times in succession
for i in range(5):
    reindex_path("./")
```
**Pass Criteria**:
- ✅ No 409 Conflict errors
- ✅ Each call completes successfully
- ✅ No orphaned containers left running

#### Scenario 3: Project Switch Test
```python
# Test project context switching
set_project_context("/project/A")
reindex_path("./")
set_project_context("/project/B")
reindex_path("./")
```
**Pass Criteria**:
- ✅ Each project gets separate container
- ✅ No cross-project data contamination
- ✅ PROJECT_CONTEXT properly initialized for each

#### Scenario 4: isinstance Check Validation
```python
# Verify type safety is restored
# In service_container.py __init__
assert isinstance(context_manager, ProjectContextManager)
```
**Pass Criteria**:
- ✅ No TypeError on valid context_manager
- ✅ Raises TypeError on invalid type
- ✅ No module identity issues

#### Scenario 5: File Type Coverage
```bash
# Create test files with all extensions
touch test.{dart,vue,svelte,kt,swift}
reindex_path("./")
```
**Pass Criteria**:
- ✅ All file types detected and indexed
- ✅ Specifically: .dart files are indexed

### Exit Conditions Checklist

#### Code Changes Complete
- [ ] Line 640 removed from neural_server_stdio.py (no container creation in PROJECT_CONTEXT init)
- [ ] isinstance check restored in service_container.py
- [ ] All imports use absolute paths (no relative imports with sys.path)
- [ ] Missing file extensions added to watch_patterns
- [ ] Status reporting fixed to verify container started

#### Verification Tests Pass
- [ ] Scenario 1: Fresh MCP session works (no race condition)
- [ ] Scenario 2: Multiple reindex calls work (no conflicts)
- [ ] Scenario 3: Project switching works (proper isolation)
- [ ] Scenario 4: Type safety restored (isinstance works)
- [ ] Scenario 5: All file types indexed (including .dart)

#### Integration Validation
- [ ] Run full indexing on vesikaa project
- [ ] Run full indexing on claude-l9-template project
- [ ] Verify GraphRAG hybrid search returns results
- [ ] No "discovery service not found" warnings in logs
- [ ] No container removal logs when discovery should work

#### Performance Metrics Met
- [ ] Time to first index: <30 seconds (was 2-5 min timeout)
- [ ] Success rate: >95% (was ~60%)
- [ ] Container reuse: Working when discovery enabled
- [ ] No orphaned containers after 1 hour idle

### Monitoring After Fix

#### Log Patterns to Watch
```bash
# Good patterns (should see):
"✅ Using injected context"
"🔍 Using discovery service"
"♻️ Reusing existing container"

# Bad patterns (should NOT see):
"⚠️ No discovery service, using legacy"
"[ADR-0048] Removing existing indexer"
"409 Conflict"
"module identity issues"
```

#### Metrics to Track (First 48 Hours)
1. **Container lifecycle events** - Count create vs reuse vs remove
2. **Initialization timing** - PROJECT_CONTEXT init duration
3. **Race condition indicators** - Count of None context_managers
4. **Error rates** - 409 conflicts, timeouts, failures
5. **File indexing completeness** - % of files actually indexed

### Rollback Plan

If issues persist after fix:
1. **Immediate**: Revert to commit before 36fb169
2. **Restore isinstance check** with workaround for module identity
3. **Temporarily disable** discovery service (force legacy mode)
4. **Add explicit delays** to prevent race conditions
5. **Document known issues** for next sprint

## Action Items

### Phase 1: Immediate Fix (1 hour)
- [ ] Remove line 640 from neural_server_stdio.py
- [ ] Fix imports to use absolute paths consistently
- [ ] Restore isinstance check with proper imports
- [ ] Add .dart and other missing extensions

### Phase 2: Testing (2 hours)
- [ ] Run all 5 test scenarios
- [ ] Document results in test log
- [ ] Fix any failing scenarios
- [ ] Verify exit conditions met

### Phase 3: Deploy & Monitor (4 hours)
- [ ] Deploy to local environment
- [ ] Run integration validation
- [ ] Monitor for 2 hours
- [ ] Deploy to global MCP if stable
- [ ] Continue monitoring for 48 hours

## Evidence of Regression

### Before (ADR-0052):
```
User: reindex_path("./")
System: ✅ Indexing started for 1,247 files
Status: Processing... [actual progress shown]
Result: Indexed successfully in 28 seconds
```

### After (Current):
```
User: reindex_path("./")
System: ✅ Indexing queued for project vesikaa
Container: 409 Conflict - marked for removal
Actual: Nothing happens, user confused
Debug: 2 services fighting over same container
```

This regression must be fixed to restore L9 quality standards.