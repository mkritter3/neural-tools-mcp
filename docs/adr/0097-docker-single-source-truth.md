# ADR-0097: Docker as Single Source of Truth for Container Orchestration

**Status:** Proposed (COMPLETE REWRITE)
**Date:** September 2025
**Author:** L9 Engineering Team + Gemini Analysis

## Context

Gemini's deep analysis revealed a critical architectural flaw: We have **FOUR different sources of truth** for container state:

1. **ProjectContextManager** - JSON file with `container_registry: Dict[str, int]`
2. **IndexerOrchestrator** - In-memory `active_indexers: Dict[str, dict]`
3. **Docker Daemon** - Actual container state
4. **Original ADR-97** - Proposed adding Redis (4th source!)

This violates the Single Source of Truth principle and causes:
- State synchronization bugs (impossible to keep in sync)
- Orphaned containers after crashes
- Port allocation conflicts between components
- Resource leaks when state diverges

## Decision

**Use Docker as the ONLY source of truth.** Query Docker API in real-time for all container state.

## Implementation

### 1. Docker Labels for Metadata

Store ALL metadata as Docker labels on containers:

```python
labels = {
    "com.l9.managed": "true",                    # Identifies our containers
    "com.l9.project": "project-name",            # Project name
    "com.l9.project_hash": "sha256_hash",        # Hash of project path
    "com.l9.port": "48101",                      # Allocated port
    "com.l9.created": "2025-09-24T10:00:00Z",   # Creation timestamp
}
```

### 2. Simplify Components

#### ProjectContextManager
- **REMOVE**: `container_registry` completely (-50 lines)
- **KEEP**: Project paths and detection logic only
- No container state tracking

#### IndexerOrchestrator
- **REMOVE**: `active_indexers` dict (-100 lines)
- **CHANGE**: Always query Docker for state:

```python
async def get_indexer_for_project(self, project_path: str):
    """Query Docker for container by project"""
    project_hash = hashlib.sha256(project_path.encode()).hexdigest()

    containers = self.docker_client.containers.list(filters={
        'label': [
            'com.l9.managed=true',
            f'com.l9.project_hash={project_hash}'
        ],
        'status': 'running'
    })

    if containers:
        container = containers[0]
        return {
            'container_id': container.id,
            'port': container.labels['com.l9.port'],
            'status': 'existing'
        }
    return None
```

#### ServiceContainer
- **SIMPLIFY**: `ensure_indexer_running()` to only check Docker:

```python
async def ensure_indexer_running(self, project_path: str):
    """Ensure indexer container exists for project"""
    # 1. Hash the project path for stable ID
    project_hash = hashlib.sha256(project_path.encode()).hexdigest()

    # 2. Check Docker for existing container
    containers = self.docker_client.containers.list(filters={
        'label': f'com.l9.project_hash={project_hash}',
        'status': 'running'
    })

    if containers:
        # Container exists, return its info
        port = containers[0].labels['com.l9.port']
        return {'port': port, 'status': 'existing'}

    # 3. No container, create new one
    port = self._allocate_port()  # Check Docker for used ports

    container = self.docker_client.containers.run(
        image="indexer:production",
        detach=True,
        labels={
            'com.l9.managed': 'true',
            'com.l9.project_hash': project_hash,
            'com.l9.project': project_name,
            'com.l9.port': str(port),
            'com.l9.created': datetime.now(timezone.utc).isoformat()
        },
        ports={'48100/tcp': port},
        # ... other config
    )

    return {'port': port, 'status': 'created'}
```

### 3. Garbage Collection

Use Docker's built-in timestamps, not custom labels:

```python
async def cleanup_idle_containers(self):
    """Remove containers idle for > 1 hour"""
    containers = self.docker_client.containers.list(
        filters={'label': 'com.l9.managed=true'},
        all=True  # Include stopped containers
    )

    for container in containers:
        # Use Docker's State.StartedAt, not custom labels
        started = container.attrs['State']['StartedAt']
        uptime = datetime.now() - parse_datetime(started)

        if uptime > timedelta(hours=1):
            container.stop()
            container.remove()
```

### 4. CLAUDE_PROJECT_DIR Integration

Handle consistently at startup:

```python
# At MCP server initialization
async def initialize():
    claude_dir = os.getenv("CLAUDE_PROJECT_DIR")
    if claude_dir:
        # This just validates and records the project
        # Container creation is lazy on first use
        await context_manager.set_project(claude_dir)
```

## Benefits

### Eliminates ALL Synchronization Issues
- No state to sync (Docker is the state)
- No stale data possible
- Crash recovery automatic

### Simpler Code
- Remove ~200 lines of state management
- No JSON file I/O
- No in-memory dictionaries to maintain

### Better Performance
- Docker API calls are milliseconds (acceptable for orchestration)
- No Redis dependency
- Native Docker caching

### L9 Compliance
- **Simple > Complex**: Removes 3 state stores
- **Single Responsibility**: Docker manages Docker containers
- **Fail-Fast**: Can't have stale state
- **Reversible**: Can add caching layer later if needed
- **Truth > Comfort**: Acknowledges current design is broken

## Migration Path

### Phase 1: Add Docker Labels (Safe)
1. Update container creation to add labels
2. Existing state mechanisms continue working
3. Test label-based queries in parallel

### Phase 2: Read from Docker (Gradual)
1. Update read methods to query Docker
2. Fall back to old state if Docker query fails
3. Log discrepancies for debugging

### Phase 3: Remove Old State (Clean)
1. Delete `container_registry` from ProjectContextManager
2. Delete `active_indexers` from IndexerOrchestrator
3. Remove all associated persistence code

## Testing

```python
# Test: Container discovery by project
def test_find_container_by_project():
    project_path = "/test/project"
    container = create_test_container(project_path)

    # Should find by label
    found = orchestrator.get_indexer_for_project(project_path)
    assert found['container_id'] == container.id

    # Should not find different project
    other = orchestrator.get_indexer_for_project("/other/project")
    assert other is None

# Test: No orphaned state after crash
def test_crash_recovery():
    # Create container
    container = create_test_container("/test")

    # Simulate crash (no cleanup)
    orchestrator = None

    # New instance should find container
    new_orchestrator = IndexerOrchestrator()
    found = new_orchestrator.get_indexer_for_project("/test")
    assert found is not None
```

## Rejected Alternatives

### Redis as State Store (Original ADR-97)
- Adds 4th source of truth (makes problem worse)
- Requires Redis dependency
- Still needs synchronization with Docker

### Keep Current Multi-State Design
- Fundamentally broken (can't sync 3+ sources reliably)
- Causes production bugs
- Violates single source of truth

### In-Memory Cache Layer
- Premature optimization
- Docker queries are fast enough for orchestration
- Can add later if metrics show need

## References

- Gemini's Deep Analysis (September 2025)
- Docker Best Practices: Labels for Metadata
- L9 Engineering Principles: Simple > Complex
- Single Source of Truth Pattern

## Decision

**Approved.** Docker as single source of truth eliminates entire classes of bugs while simplifying the codebase significantly.