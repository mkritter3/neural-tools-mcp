# ADR-0098: Gradual Migration to Docker as Single Source of Truth

**Status:** Proposed
**Date:** September 2025
**Author:** L9 Engineering Team
**Supersedes:** ADR-0097

## Context

Deep analysis with Gemini revealed that our system has **four different sources of truth** for container state:

1. **IndexerOrchestrator.active_indexers** - In-memory dict
2. **ProjectContextManager.container_registry** - JSON file persistence
3. **Docker daemon** - Actual container state with labels
4. **ADR-0097's proposal** - Would have added Redis as a 4th source

This violates the Single Source of Truth principle and causes:
- State synchronization bugs (impossible to keep 4 sources in sync)
- Orphaned containers after crashes
- Port allocation conflicts between components
- Resource leaks when state diverges

## Decision

**Gradually migrate to Docker as the ONLY source of truth** through a careful, phased approach with observability-first validation.

### Why Gradual Migration?

- **No breaking changes** - Each phase is backward compatible
- **Observable validation** - Measure divergence before changing behavior
- **Easy rollback** - Can stop at any phase if issues arise
- **Minimal risk** - Small, reversible steps per L9 principles

## Implementation Plan

### Phase 0: Observability Only (1 week, ZERO RISK)

**Goal:** Understand state divergence without changing any behavior.

Add logging to compare Docker state with internal dicts:

```python
# New docker_observability.py module
class DockerStateObserver:
    def check_state_divergence(project_name, dict_state, source):
        docker_state = query_docker(project_name)
        if docker_state != dict_state:
            logger.warning(f"State divergence: {source} vs Docker")
```

**Integration:** Add observer calls to existing code without changing logic:
```python
# In IndexerOrchestrator.ensure_indexer()
self.active_indexers[project_name] = {...}  # Existing code
observer.check_state_divergence(...)  # New observability
```

**Success Metrics:**
- Divergence rate < 5%
- No functional changes
- Metrics dashboard shows state health

### Phase 1: Enhanced Docker Labels (1 week, LOW RISK)

**Goal:** Add metadata to Docker labels without breaking existing code.

Enhance container creation with additional labels:

```python
labels = {
    'com.l9.managed': 'true',           # Already exists
    'com.l9.project': project_name,      # Already exists
    'com.l9.created': timestamp,         # Already exists
    'com.l9.project_hash': sha256(path), # NEW: Stable ID
    'com.l9.port': str(port),           # NEW: Port in labels
    'com.l9.project_path': project_path  # NEW: Full path
}
```

**Key Points:**
- Keep writing to existing dicts (no behavior change)
- Labels are additional metadata only
- Existing code continues working exactly as before

**Success Metrics:**
- 100% of new containers have enhanced labels
- Existing state management unchanged
- Zero breaking changes

### Phase 2: Docker Primary, Dicts as Fallback (2 weeks, LOW RISK)

**Goal:** Read from Docker first, fall back to dicts if needed.

```python
def get_container_info(project_name):
    # Try Docker first (new)
    container = docker_query_by_label(project_name)
    if container:
        logger.debug("Using Docker state")
        return extract_info_from_container(container)

    # Fall back to dict (existing behavior)
    logger.warning("Docker query failed, using dict fallback")
    return self.active_indexers.get(project_name)
```

**Key Points:**
- Still write to both Docker AND dicts
- Log when fallback is used
- Monitor fallback rate

**Success Metrics:**
- Docker queries succeed > 95% of time
- Fallback usage < 5%
- No functional regressions

### Phase 3: Write-Through Cache (1 week, MEDIUM RISK)

**Goal:** Dicts become pure caches of Docker state.

```python
async def sync_state_from_docker():
    """On startup and periodically"""
    containers = docker.list(label='com.l9.managed=true')

    # Populate dicts FROM Docker (reverse of current)
    self.active_indexers = {}
    for container in containers:
        project = container.labels['com.l9.project']
        self.active_indexers[project] = {
            'container_id': container.id,
            'port': container.labels['com.l9.port']
        }
```

**Key Points:**
- Docker is source, dicts are cache
- On startup: populate dicts from Docker
- On change: update Docker first, then dicts
- Single source of truth achieved

**Success Metrics:**
- Zero state divergence
- Crash recovery works correctly
- Port conflicts eliminated

### Phase 4: Remove Redundant State (Future, Optional)

**Goal:** Eliminate dicts entirely (only after Phase 3 proven stable).

- Remove `active_indexers` dict
- Remove `container_registry` JSON persistence
- Query Docker directly for all operations
- ~200 lines of code removed

**When to Consider:**
- Phase 3 stable for > 4 weeks
- Performance metrics acceptable
- Team consensus on simplification

## Benefits Over ADR-0097

| Aspect | ADR-0097 | ADR-0098 |
|--------|----------|----------|
| Risk | HIGH (big bang) | LOW (gradual) |
| Sources of Truth | Adds 4th (Redis) | Reduces to 1 (Docker) |
| Rollback | Difficult | Easy (stop at any phase) |
| Validation | Hope it works | Observable at each phase |
| Code Changes | Major refactor | Incremental additions |

## Performance Considerations

**Docker Query Performance:**
- Label queries: ~5-10ms (acceptable for orchestration)
- Full container list: ~20-30ms (use sparingly)
- Port allocation check: ~15ms (on-demand only)

**If Performance Issues Arise:**
- Short-lived in-memory cache (< 5 seconds)
- Docker events API for reactive updates
- But NOT persistent state (that recreates the problem)

## Migration Coordination

### What Changes Where

**IndexerOrchestrator:**
- Phase 0: Add observability calls
- Phase 1: Enhanced labels on creation
- Phase 2: Query Docker before checking active_indexers
- Phase 3: Populate active_indexers from Docker
- Phase 4: Remove active_indexers entirely

**ProjectContextManager:**
- Phase 0: Add observability calls
- Phase 1: No changes (doesn't create containers)
- Phase 2: Query Docker before checking container_registry
- Phase 3: Populate container_registry from Docker
- Phase 4: Remove container_registry entirely

**ServiceContainer:**
- No changes needed (uses the above services)

## Rollback Plan

At any phase, if issues arise:

1. **Phase 0:** Remove observability calls (no functional impact)
2. **Phase 1:** New labels ignored by old code (harmless)
3. **Phase 2:** Remove Docker queries, rely on dicts (instant)
4. **Phase 3:** Revert to dict-first logic (config flag)

## Testing Strategy

**Phase 0 Tests:**
```python
def test_observability_no_side_effects():
    # Verify observer doesn't change behavior
    # Check metrics are collected correctly
```

**Phase 1 Tests:**
```python
def test_enhanced_labels_present():
    # Verify new containers have all labels
    # Verify old code still works
```

**Phase 2 Tests:**
```python
def test_docker_primary_dict_fallback():
    # Test successful Docker queries
    # Test fallback when Docker unavailable
    # Verify no behavior changes
```

**Phase 3 Tests:**
```python
def test_crash_recovery():
    # Kill orchestrator
    # Restart and verify state recovered from Docker
    # No orphaned containers or ports
```

## Decision

**Approved for Phase 0 implementation.**

Start with observability to validate assumptions. Each subsequent phase requires metrics review before proceeding.

## Why This Approach

✅ **L9 Compliant:**
- Small steps (phases can stop anytime)
- Reversible (easy rollback at each phase)
- Observable (metrics before changes)
- Simple > Complex (removes state, doesn't add)
- Truth > Comfort (addresses real architectural flaw)

❌ **What We're NOT Doing:**
- Not adding Redis (4th source of truth)
- Not doing big-bang migration
- Not breaking existing deployments
- Not requiring all phases immediately

## References

- ADR-0097: Multi-Project Orchestration (superseded)
- ADR-0060: Container naming and discovery patterns
- ADR-0029: Neo4j project isolation
- Gemini Analysis: September 2025
- Docker Best Practices: Labels as metadata

## Tracking

- Phase 0: ⏳ Ready to implement
- Phase 1: 📋 Planned
- Phase 2: 📋 Planned
- Phase 3: 📋 Planned
- Phase 4: 🔮 Future consideration

**Confidence: 95%** - Gradual approach minimizes risk while solving core architectural issue.