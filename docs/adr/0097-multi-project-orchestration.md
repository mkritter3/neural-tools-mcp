# ADR-0097: Multi-Project Orchestration Enhancement

**Status:** Proposed
**Date:** September 2025
**Author:** L9 Engineering Team

## Context

The current system has excellent foundations:
- **ProjectContextManager** (ADR-0033) - Handles project detection and switching
- **IndexerOrchestrator** (ADR-0030) - Manages container lifecycle
- **ServiceContainer** (ADR-0044) - Dependency injection pattern

However, these components need enhanced coordination for true multi-project support when Claude opens in different directories.

### Current Problems
1. Components exist but lack full integration for multi-project scenarios
2. `$CLAUDE_PROJECT_DIR` not consistently used across all tools
3. Container lifecycle not fully tied to project switches
4. Incomplete state persistence across Claude sessions

## Decision

**ENHANCE EXISTING COMPONENTS** - Do NOT create parallel systems!
We will enhance the existing orchestration with better coordination:

## Implementation Plan

### Phase 0: Safety & Reversibility
```python
# Feature flag for gradual rollout
MCP_MULTI_PROJECT_MODE = os.getenv("MCP_MULTI_PROJECT_MODE", "false").lower() == "true"

def get_project_id(project_dir: str) -> str:
    """Generate deterministic project ID from directory path"""
    normalized_path = os.path.abspath(os.path.normpath(project_dir))
    return hashlib.sha1(normalized_path.encode('utf-8')).hexdigest()[:12]
```

### Phase 1: Redis State Registry
Expand Redis from just locking to full state management:

**Schema:**
```
Key: mcp:project_registry
Field: {project_id}
Value: {
    "container_id": "...",
    "container_name": "mcp-indexer-{project_id}",
    "port": 48101,
    "status": "running",
    "last_heartbeat": 1234567890,
    "project_path": "/full/path/to/project"
}
```

### Phase 2: Core Orchestration Logic
```python
async def get_or_create_indexer(project_dir: str):
    if not MCP_MULTI_PROJECT_MODE:
        return legacy_start_logic()  # Existing functionality preserved

    project_id = get_project_id(project_dir)

    # Check registry for existing container
    project_info = await redis.hget("mcp:project_registry", project_id)
    if project_info and is_container_running(project_info['container_id']):
        await update_heartbeat(project_id)
        return {"port": project_info['port'], "status": "existing"}

    # Create new container with distributed lock
    async with redis.lock(f"mcp:create:{project_id}", timeout=30):
        container_name = f"mcp-indexer-{project_id}"
        port = await find_free_port(start=48100)

        container = await docker.run(
            image="indexer:production",
            name=container_name,
            ports={'48100/tcp': port},
            environment={
                "PROJECT_PATH": project_dir,
                "PROJECT_ID": project_id
            },
            labels={
                "com.l9.project": project_id,
                "com.l9.managed": "true"
            },
            mem_limit="2g",
            cpu_shares=512
        )

        await redis.hset("mcp:project_registry", project_id, {
            "container_id": container.id,
            "container_name": container_name,
            "port": port,
            "status": "running",
            "last_heartbeat": time.time(),
            "project_path": project_dir
        })

        return {"port": port, "status": "created"}
```

### Phase 3: Lifecycle Management

**Heartbeat Mechanism:**
```python
async def heartbeat(project_dir: str):
    """Called periodically by MCP tools when active"""
    project_id = get_project_id(project_dir)
    await redis.hset("mcp:project_registry", project_id,
                     "last_heartbeat", time.time())
```

**Janitor Process:**
```python
async def cleanup_idle_containers():
    """Runs every 5 minutes"""
    IDLE_THRESHOLD = 3600  # 1 hour
    now = time.time()

    registry = await redis.hgetall("mcp:project_registry")
    for project_id, info in registry.items():
        if now - info['last_heartbeat'] > IDLE_THRESHOLD:
            await stop_container(info['container_id'])
            await redis.hdel("mcp:project_registry", project_id)
            logger.info(f"Cleaned up idle container for {project_id}")
```

### Phase 4: API Changes

MCP tools must pass project context:
```python
# In each MCP tool's execute function
async def execute(arguments: dict) -> List[types.TextContent]:
    project_dir = os.getenv("CLAUDE_PROJECT_DIR", os.getcwd())

    # Ensure indexer is running for this project
    indexer_info = await get_or_create_indexer(project_dir)

    # Continue with tool logic using project-specific indexer
    ...
```

## Advantages

1. **Incremental & Safe:** Feature flag allows instant rollback
2. **Zero Breaking Changes:** Legacy mode preserves all existing behavior
3. **Resource Efficient:** Idle containers auto-stop after 1 hour
4. **Simple:** No new infrastructure (K8s, systemd, compose files)
5. **Portable:** Works on any platform with Docker
6. **Observable:** Redis registry provides clear state visibility

## Rollback Plan

1. Set `MCP_MULTI_PROJECT_MODE=false`
2. Restart MCP server
3. System immediately reverts to legacy single-project mode

## Resource Limits

Per-container limits to prevent resource exhaustion:
- Memory: 2GB
- CPU: 512 shares (relative weight)
- Max containers: 10 concurrent projects (configurable)

## Testing Strategy

1. **Unit Tests:** Mock Docker/Redis, test state transitions
2. **Integration Tests:** Real containers, project switching
3. **Load Tests:** 10 concurrent projects
4. **Chaos Tests:** Kill containers, restart MCP, verify recovery

## Migration Path

Week 1: Implement Phase 0-1 (feature flag + Redis registry)
Week 2: Phase 2 (core orchestration) with extensive testing
Week 3: Phase 3 (lifecycle management)
Week 4: Phase 4 (API changes) and production rollout

## Decision Rationale

- **Not Kubernetes:** Over-engineering for current scale
- **Not Docker Compose:** Decentralized config = maintenance nightmare
- **Not Systemd:** Platform-specific, breaks portability
- **Yes Enhanced MCP:** Minimal change, maximum compatibility

## References

- ADR-0060: Container naming conflicts resolution
- ADR-0052: Automatic indexer initialization
- MCP 2025-06-18 specification
- L9 Engineering Contract

## Status Updates

- 2025-09-24: Initial proposal