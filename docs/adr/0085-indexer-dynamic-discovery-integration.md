# ADR-0085: Indexer Dynamic Discovery Integration

**Status:** Proposed
**Date:** 2025-09-23
**Author:** L9 Neural Team

## Context

The neural indexer system uses a sophisticated orchestrator (IndexerOrchestrator) that manages per-project containers with dynamic port allocation (ADR-0030, ADR-0060). However, the MCP tools are hardcoded to use port 48121, creating a disconnect that prevents proper indexing and vector search functionality.

### Current Architecture

1. **IndexerOrchestrator** (Working Correctly):
   - Creates one container per project
   - Dynamically allocates ports (48100-48200 range)
   - Uses label-based discovery: `com.l9.project={name}`
   - Provides `get_indexer_endpoint()` for discovery
   - Caches endpoints in Redis for 30s

2. **MCP Tools** (Broken):
   - Hardcoded to port 48121: `os.getenv('INDEXER_PORT', '48121')`
   - No integration with orchestrator
   - No service discovery mechanism
   - Results in 404 errors

### Root Cause Analysis

The disconnect occurred because:
1. Orchestrator was updated to use dynamic ports (ADR-0060)
2. MCP tools were never updated to use the discovery mechanism
3. Test containers accidentally work on 48121 (misleading success)
4. No integration tests between MCP and dynamic indexer

## Decision

Integrate MCP tools with the IndexerOrchestrator's dynamic discovery mechanism to properly locate and communicate with per-project indexer containers.

## Solution Design

### 1. MCP Tool Integration

Update `project_operations.py` to use orchestrator:

```python
async def _execute_reindex_path(neo4j_service, arguments, project_name):
    """Execute path reindexing with dynamic discovery"""

    # Get ServiceContainer and orchestrator
    from servers.services.service_container import ServiceContainer
    container = ServiceContainer(project_name=project_name)
    orchestrator = await container.get_indexer_orchestrator()

    # Ensure indexer is running for this project
    project_path = os.getcwd()  # Or from context
    container_id = await orchestrator.ensure_indexer(
        project_name=project_name,
        project_path=project_path
    )

    # Get the dynamic endpoint
    endpoint = await orchestrator.get_indexer_endpoint(project_name)
    if not endpoint:
        return {
            "status": "error",
            "message": f"No indexer found for project {project_name}"
        }

    # Use discovered endpoint instead of hardcoded port
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{endpoint}/reindex-path",
            json={"path": path, "recursive": recursive}
        )
```

### 2. Service Container Enhancement

Add helper method to ServiceContainer:

```python
class ServiceContainer:
    async def get_indexer_endpoint(self) -> Optional[str]:
        """Get indexer endpoint for current project"""
        orchestrator = await self.get_indexer_orchestrator()

        # Ensure indexer is running
        if self.context_manager:
            project_path = self.context_manager.get_project_path()
        else:
            project_path = os.getcwd()

        await orchestrator.ensure_indexer(
            project_name=self.project_name,
            project_path=project_path
        )

        # Return discovered endpoint
        return await orchestrator.get_indexer_endpoint(self.project_name)
```

### 3. Fallback Strategy

For backwards compatibility:

```python
# Try dynamic discovery first
endpoint = await container.get_indexer_endpoint()

# Fallback to env var if discovery fails
if not endpoint:
    indexer_host = os.getenv('INDEXER_HOST', 'localhost')
    indexer_port = os.getenv('INDEXER_PORT', '48121')
    endpoint = f"http://{indexer_host}:{indexer_port}"
    logger.warning(f"Using fallback endpoint: {endpoint}")
```

## Implementation Steps

### Phase 1: Service Container Enhancement
1. Add `get_indexer_endpoint()` helper method
2. Ensure proper project context propagation
3. Add logging for discovery process

### Phase 2: MCP Tool Updates
1. Update `_execute_reindex_path()` to use discovery
2. Update `_execute_indexer_status()` similarly
3. Add proper error handling for discovery failures

### Phase 3: Testing
1. Create test for dynamic discovery
2. Verify multi-project scenarios
3. Test fallback behavior

### Phase 4: Cleanup
1. Remove hardcoded port references
2. Update documentation
3. Add deprecation warnings for env vars

## Benefits

1. **Correct Integration**: MCP tools properly find their project's indexer
2. **Multi-Project Support**: Each project gets its own indexer
3. **Dynamic Scaling**: Ports allocated as needed
4. **Cache Performance**: Redis caching for endpoint discovery
5. **Backwards Compatible**: Fallback to env vars if needed

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Discovery failure | Can't index | Fallback to env vars |
| Orchestrator not initialized | No indexer | Initialize on demand |
| Port exhaustion | Can't create indexer | Expand port range |
| Redis unavailable | Slower discovery | Local discovery fallback |

## Testing Strategy

1. **Unit Tests**:
   - Mock orchestrator responses
   - Test discovery with/without Redis
   - Verify fallback behavior

2. **Integration Tests**:
   - Multi-project indexing
   - Dynamic port allocation
   - Container lifecycle

3. **E2E Tests**:
   - Full MCP → Orchestrator → Indexer flow
   - Vector search after indexing
   - Graph context retrieval

## Success Metrics

- MCP tools successfully index files
- Vector search returns results
- Graph context properly populated
- No hardcoded port references
- Multi-project isolation verified

## Migration Path

1. **Week 1**: Implement service container enhancement
2. **Week 2**: Update MCP tools with discovery
3. **Week 3**: Testing and validation
4. **Week 4**: Remove deprecated code

## Code Examples

### Before (Broken)
```python
# Hardcoded port - doesn't work with orchestrator
indexer_port = os.getenv('INDEXER_PORT', '48121')
indexer_url = f"http://localhost:{indexer_port}"
```

### After (Fixed)
```python
# Dynamic discovery - works with orchestrator
endpoint = await container.get_indexer_endpoint()
if not endpoint:
    raise RuntimeError("No indexer available")
```

## Related ADRs

- ADR-0030: Multi-Container Indexer Architecture
- ADR-0060: Graceful Ephemeral Containers
- ADR-0044: Dependency Injection Pattern
- ADR-0076: Modular Service Architecture

## Conclusion

This ADR fixes the critical disconnect between MCP tools and the indexer orchestrator, enabling proper vector search and graph context functionality. The solution maintains backwards compatibility while enabling the full power of the dynamic, per-project indexer architecture.