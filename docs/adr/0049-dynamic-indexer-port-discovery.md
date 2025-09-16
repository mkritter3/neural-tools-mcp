# ADR-0049: Dynamic Indexer Port Discovery

**Date:** September 15, 2025
**Status:** Implemented (Partial)
**Tags:** infrastructure, containers, networking, mcp

## Context

The MCP neural tools were failing to connect to indexer containers because they used a hardcoded port (48080) while actual containers run on dynamically allocated ports in the range 48100-48199. This caused "connection refused" errors when users tried to use indexer-related MCP tools.

### Problem Statement

1. **Hardcoded Port 48080**: MCP tools had `localhost:48080` hardcoded in multiple places
2. **Dynamic Container Ports**: Actual indexer containers run on ports like 48100 (northstar-finance), 48102 (claude-l9-template)
3. **No Discovery Mechanism**: No way for MCP tools to discover the actual port of a running container
4. **Container Startup Race**: Tools tried to connect immediately without waiting for containers to be ready

## Decision

Implement dynamic port discovery for indexer containers with proper startup synchronization and error handling.

### Implementation Details

#### 1. Dynamic Port Discovery Chain
```python
# Priority order for port discovery:
1. orchestrator.get_indexer_port(project_name)     # Check cached active_indexers
2. discovery_service.discover_project_container()    # Scan Docker containers
3. Return error (no fallback to 48080)              # Fail clearly if not found
```

#### 2. Container Startup Wait
```python
# Wait for container to be ready before connecting
is_ready = await _wait_for_indexer_ready(indexer_port, timeout=10)
if not is_ready:
    return "Container starting" error
```

#### 3. Exact Container Name Matching
```python
# Prevent "api" from matching "api-v2"
if container.name == f"indexer-{project_name}":  # Exact match only
    return container
elif f"indexer-{project_name}" in container.name:
    log_warning("Partial match rejected")
    return None
```

## Implementation Status

### ✅ Completed

1. **Removed hardcoded port 48080** from:
   - `indexer_status_impl()`
   - `reindex_path_impl()`

2. **Added dynamic port discovery**:
   - Uses `IndexerOrchestrator.get_indexer_port()`
   - Falls back to `ContainerDiscoveryService.discover_project_container()`
   - No fallback to non-working ports

3. **Added container readiness check**:
   - `_wait_for_indexer_ready()` waits up to 10s
   - Prevents `ReadError` from containers still starting

4. **Fixed container name matching**:
   - Exact match required (`indexer-api` != `indexer-api-v2`)
   - Prevents project name collision bugs

5. **Improved error messages**:
   - Shows actual exception types
   - Includes helpful suggestions ("Run set_project_context")
   - Logs full tracebacks for debugging

### ⚠️ Known Issues (Identified by Gemini Analysis)

1. **Dual Port Allocation Mechanisms** (HIGH RISK)
   - `IndexerOrchestrator._allocate_port()` uses local `self.allocated_ports` set
   - `ContainerDiscoveryService` uses `context_manager.container_registry`
   - **Risk**: Same port could be allocated twice, causing collisions

2. **Stale Cache Problem** (MEDIUM RISK)
   - `active_indexers` cache doesn't validate container health
   - Container could be stopped externally but still in cache
   - **Risk**: Returns invalid ports for stopped containers

3. **Distributed Logic** (LOW RISK)
   - Port discovery logic duplicated in each MCP tool
   - Should be centralized in orchestrator

## Future Work (TODO)

### Phase 1: Fix Critical Issues
1. **Unify Port Allocation**
   - Remove `_allocate_port()` from `IndexerOrchestrator`
   - Use only `ContainerDiscoveryService` for all port management
   - Single source of truth for allocated ports

2. **Add Cache Validation**
   ```python
   async def get_indexer_port(self, project_name: str) -> Optional[int]:
       if project_name in self.active_indexers:
           port = self.active_indexers[project_name].get('port')
           if await self._validate_port_health(port):
               return port
           else:
               del self.active_indexers[project_name]  # Invalidate stale entry

       # Fall back to discovery...
   ```

### Phase 2: Improve Architecture
1. **Centralize Port Discovery**
   - Move all logic to `IndexerOrchestrator.get_valid_port()`
   - MCP tools make single call instead of implementing fallback chain

2. **Add Port Registry Persistence**
   - Store port allocations in Redis or file
   - Survive MCP server restarts

3. **Implement Port Recycling**
   - Track when containers stop
   - Return ports to available pool

## Testing

### Test Coverage Added
- `test_indexer_dynamic_port_discovery.py` - 7 tests
  - Port discovery from orchestrator
  - Discovery service container finding
  - Port allocation range validation
  - Missing port mapping handling
  - No hardcoded 48080 fallback
  - Real Docker integration test

### E2E Test Added
- `test_dynamic_port_discovery()` in `test_e2e_all_tools.py`
  - Verifies actual container ports used
  - Ensures no 48080 references

## Consequences

### Positive
- ✅ MCP indexer tools now work with actual container ports
- ✅ Clear error messages when containers not found
- ✅ No more connection refused errors from wrong ports
- ✅ Handles container startup timing correctly

### Negative
- ⚠️ Dual port allocation could cause collisions (needs fixing)
- ⚠️ Stale cache could return invalid ports (needs validation)
- ⚠️ More complex than hardcoded port (but necessary)

### Neutral
- 10-second wait on first connection to new container
- Requires Docker access for port discovery

## Code Locations

- **MCP Tool Changes**: `src/neural_mcp/neural_server_stdio.py`
- **Discovery Service**: `src/servers/services/container_discovery.py`
- **Orchestrator**: `src/servers/services/indexer_orchestrator.py`
- **Tests**: `tests/test_indexer_dynamic_port_discovery.py`

## References

- Original issue: "It was trying to connect to it on 48080, but the docker port is on 48100"
- Gemini code review identified brittleness points
- Related ADRs:
  - ADR-0030: Multi-Container Indexer Architecture
  - ADR-0037: Container Configuration Priority
  - ADR-0048: Container Path Resolution

## Decision Outcome

**Partial Success** - The dynamic port discovery is working and deployed, but Gemini's analysis revealed architectural issues that need addressing to make it production-robust. The system works for the happy path but has edge cases that could cause failures.