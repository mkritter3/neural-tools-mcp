# ADR-0044: Container Orchestration Architecture - Robust Systematic Fix

**Status**: Proposed
**Date**: September 13, 2025
**Author**: L9 Engineering Team

## Context

Our MCP server's container orchestration is fundamentally broken due to state management fragmentation. When users switch projects, the system creates wrong containers, ignores existing ones, and fails to track port allocations properly.

### Critical Evidence (September 13, 2025)

User in `/Users/mkr/local-coding/Systems/neural-novelist`:
```bash
# Docker shows existing container
indexer-neural-novelist running on port 48101

# But MCP tries to create wrong container
"Creating container: indexer-claude-l9-template"  # WRONG PROJECT!
"Could not determine port for neural-novelist indexer"  # Can't find 48101!
```

### Root Architectural Problems

1. **State Fragmentation**: Multiple ProjectContextManager instances with no shared state
2. **No Discovery Protocol**: System blindly spawns containers, ignores existing ones
3. **Registry Desync**: Memory state not persisted to disk
4. **Missing Dependency Injection**: Components create their own dependencies

## Problem Analysis

### The State Management Disaster

```python
# In neural_server_stdio.py
context_manager = ProjectContextManager()  # Global instance
await context_manager.switch_project("/path/to/neural-novelist")  # Updates THIS instance

# But in service_container.py, line 843
async def ensure_indexer_running(self):
    manager = ProjectContextManager()  # Creates NEW instance!
    # This loads from ~/.graphrag/projects.json which still has old project
    project_name = manager.current_project  # Returns "claude-l9-template" 😱
```

### Why This Is Architecturally Broken

1. **Violates Single Source of Truth**: Multiple managers = conflicting state
2. **No Dependency Injection**: Components create dependencies = no control
3. **No Service Discovery**: Can't find existing containers by port
4. **No State Persistence**: Registry file never updated on switches
5. **No Port Tracking**: Lost track of which container uses which port

## Decision: Complete Architectural Refactoring

Implement a **robust, systematic fix** with proper dependency injection, singleton patterns, container discovery, and persistent state management.

## Solution Architecture

### 1. Singleton ProjectContextManager

**Transform ProjectContextManager into a true singleton with module-level instance:**

```python
# FILE: neural-tools/src/servers/services/project_context_manager.py

# Module-level singleton instance
_global_context_manager = None
_manager_lock = asyncio.Lock()

async def get_project_context_manager() -> ProjectContextManager:
    """Get or create the singleton ProjectContextManager instance"""
    global _global_context_manager

    async with _manager_lock:
        if _global_context_manager is None:
            _global_context_manager = ProjectContextManager()
            await _global_context_manager.initialize()
        return _global_context_manager

class ProjectContextManager:
    def __init__(self):
        self.current_project = None
        self.current_project_path = None
        self.registry_path = Path.home() / ".graphrag" / "projects.json"
        self.container_registry = {}  # Track container->port mappings
        self._lock = asyncio.Lock()

    async def switch_project(self, project_path: str):
        """Switch project with full state persistence"""
        async with self._lock:
            # ... teardown logic ...

            # Update state
            project_info = self._detect_project(project_path)
            self.current_project = project_info['name']
            self.current_project_path = Path(project_info['path'])

            # CRITICAL: Persist to registry immediately
            await self._persist_registry()

            # ... rebuild logic ...

    async def _persist_registry(self):
        """Persist current state to registry file"""
        registry = {
            "active_project": self.current_project,
            "active_path": str(self.current_project_path),
            "projects": self._list_known_projects(),
            "container_ports": self.container_registry,
            "last_updated": datetime.now().isoformat()
        }

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
```

### 2. Dependency Injection Architecture

**Pass ProjectContextManager through constructors, never create new instances:**

```python
# FILE: neural-tools/src/servers/services/service_container.py

class ServiceContainer:
    def __init__(self, context_manager: ProjectContextManager):
        """Initialize with injected ProjectContextManager"""
        self.context_manager = context_manager  # INJECTED, not created!
        self.project_name = context_manager.current_project
        self.project_path = context_manager.current_project_path

        # Initialize services
        self.neo4j = None
        self.qdrant = None
        self.indexer_orchestrator = None

    async def ensure_indexer_running(self) -> IndexerOrchestrator:
        """Use injected context manager, don't create new one"""
        if not self.indexer_orchestrator:
            # Pass context to orchestrator
            self.indexer_orchestrator = IndexerOrchestrator(
                context_manager=self.context_manager  # Propagate injection
            )

        return await self.indexer_orchestrator.ensure_running()
```

### 3. Container Discovery Service

**Discover existing containers before spawning new ones:**

```python
# FILE: neural-tools/src/servers/services/container_discovery.py

class ContainerDiscoveryService:
    """Service for discovering and managing Docker containers"""

    def __init__(self, docker_client):
        self.docker = docker_client

    async def discover_project_container(self, project_name: str) -> Optional[dict]:
        """Find existing container for project"""
        try:
            containers = self.docker.containers.list(all=True)

            for container in containers:
                # Match by name pattern
                if f"indexer-{project_name}" in container.name:
                    # Get port mapping
                    ports = container.attrs['NetworkSettings']['Ports']
                    for internal, bindings in ports.items():
                        if bindings and '8080/tcp' in internal:
                            host_port = int(bindings[0]['HostPort'])
                            return {
                                'container': container,
                                'port': host_port,
                                'status': container.status
                            }
        except Exception as e:
            logger.error(f"Container discovery failed: {e}")

        return None

    async def get_or_create_container(self, project_name: str,
                                     context_manager: ProjectContextManager) -> dict:
        """Get existing or create new container with port tracking"""

        # First, try to discover existing
        existing = await self.discover_project_container(project_name)
        if existing and existing['status'] == 'running':
            # Update registry with discovered port
            context_manager.container_registry[project_name] = existing['port']
            await context_manager._persist_registry()
            return existing

        # Allocate new port (48100-48199 range)
        used_ports = set(context_manager.container_registry.values())
        for port in range(48100, 48200):
            if port not in used_ports:
                new_port = port
                break
        else:
            raise RuntimeError("No available ports in range 48100-48199")

        # Create new container
        container = self.docker.containers.run(
            image='l9-neural-indexer:production',
            name=f'indexer-{project_name}',
            ports={'8080/tcp': new_port},
            environment={
                'PROJECT_NAME': project_name,
                'PROJECT_PATH': str(context_manager.current_project_path)
            },
            detach=True
        )

        # Update registry
        context_manager.container_registry[project_name] = new_port
        await context_manager._persist_registry()

        return {
            'container': container,
            'port': new_port,
            'status': 'running'
        }
```

### 4. Updated IndexerOrchestrator

**Use discovery service and injected context:**

```python
# FILE: neural-tools/src/servers/services/indexer_orchestrator.py

class IndexerOrchestrator:
    def __init__(self, context_manager: ProjectContextManager):
        """Initialize with injected context manager"""
        self.context_manager = context_manager  # INJECTED
        self.docker = docker.from_env()
        self.discovery = ContainerDiscoveryService(self.docker)

    async def ensure_running(self) -> dict:
        """Ensure indexer is running for current project"""
        project_name = self.context_manager.current_project

        # Use discovery service
        container_info = await self.discovery.get_or_create_container(
            project_name,
            self.context_manager
        )

        return {
            'project': project_name,
            'port': container_info['port'],
            'status': container_info['status'],
            'container_id': container_info['container'].id
        }
```

### 5. Updated MCP Server Integration

**Create singleton once, inject everywhere:**

```python
# FILE: neural-tools/src/neural_mcp/neural_server_stdio.py

# Global singleton
context_manager = None
service_container = None

async def initialize_global_services():
    """Initialize singleton services once"""
    global context_manager, service_container

    # Get singleton context manager
    context_manager = await get_project_context_manager()

    # Create service container with injection
    service_container = ServiceContainer(context_manager)
    await service_container.initialize()

@server.call_handler("set_project_context")
async def handle_set_project_context(params):
    """Switch project using singleton"""
    global context_manager, service_container

    path = params.get('path', os.getcwd())

    # Switch using singleton
    await context_manager.switch_project(path)

    # Recreate service container with new context
    if service_container:
        await service_container.teardown()

    service_container = ServiceContainer(context_manager)
    await service_container.initialize()

    return {
        "status": "success",
        "project": context_manager.current_project,
        "path": str(context_manager.current_project_path),
        "container_port": context_manager.container_registry.get(
            context_manager.current_project
        )
    }
```

## Implementation Plan

### Phase 1: Singleton Pattern (Immediate)
1. Convert ProjectContextManager to singleton
2. Add get_project_context_manager() factory
3. Update all references to use singleton

### Phase 2: Dependency Injection (Day 1)
1. Add context_manager parameter to ServiceContainer
2. Pass through to IndexerOrchestrator
3. Remove all ProjectContextManager() instantiations

### Phase 3: Container Discovery (Day 2)
1. Implement ContainerDiscoveryService
2. Add discovery before container creation
3. Track port allocations in registry

### Phase 4: State Persistence (Day 2)
1. Add _persist_registry() method
2. Update registry on every state change
3. Load registry on initialization

### Phase 5: Testing & Validation (Day 3)
1. Integration tests for project switching
2. Container discovery tests
3. Port allocation tests
4. State persistence verification

## Testing Strategy

### Integration Tests

```python
async def test_project_switching_with_containers():
    """Test complete project switch with container orchestration"""

    # Get singleton
    manager = await get_project_context_manager()

    # Switch to project A
    await manager.switch_project("/path/to/projectA")
    container_a = ServiceContainer(manager)
    await container_a.initialize()
    orchestrator_a = await container_a.ensure_indexer_running()
    port_a = orchestrator_a['port']

    # Switch to project B
    await manager.switch_project("/path/to/projectB")
    container_b = ServiceContainer(manager)
    await container_b.initialize()
    orchestrator_b = await container_b.ensure_indexer_running()
    port_b = orchestrator_b['port']

    # Verify different ports
    assert port_a != port_b

    # Switch back to A - should reuse container
    await manager.switch_project("/path/to/projectA")
    container_a2 = ServiceContainer(manager)
    await container_a2.initialize()
    orchestrator_a2 = await container_a2.ensure_indexer_running()

    # Should get same port (container reused)
    assert orchestrator_a2['port'] == port_a
```

### Container Discovery Tests

```python
async def test_container_discovery():
    """Test discovery of existing containers"""

    discovery = ContainerDiscoveryService(docker.from_env())

    # Create a test container manually
    docker_client = docker.from_env()
    test_container = docker_client.containers.run(
        image='l9-neural-indexer:production',
        name='indexer-test-project',
        ports={'8080/tcp': 48150},
        detach=True
    )

    # Discover it
    found = await discovery.discover_project_container('test-project')
    assert found is not None
    assert found['port'] == 48150

    # Cleanup
    test_container.stop()
    test_container.remove()
```

## Consequences

### Positive
- ✅ **Single Source of Truth**: One ProjectContextManager instance
- ✅ **Container Reuse**: Discovers and reuses existing containers
- ✅ **Port Tracking**: Maintains port allocations across sessions
- ✅ **State Persistence**: Registry survives restarts
- ✅ **Dependency Injection**: Clean, testable architecture
- ✅ **Robust Solution**: Not a patch - architectural fix

### Negative
- ⚠️ **Breaking Change**: Requires updating all components
- ⚠️ **Migration Complexity**: Existing deployments need update
- ⚠️ **Testing Overhead**: More integration tests needed

### Risks
- Singleton pattern could become bottleneck under load
- Port allocation could exhaust range (100 ports)
- Registry file corruption needs recovery mechanism

## Migration Path

1. **Deploy singleton first** - Backward compatible
2. **Update components gradually** - Use feature flags
3. **Add discovery service** - Run in parallel initially
4. **Cut over completely** - Remove old code paths
5. **Monitor for issues** - Track container creation metrics

## Success Metrics

- Zero duplicate containers for same project
- 100% container reuse rate on project switch
- Port allocation never fails (within range)
- Registry accurately reflects running state
- No "wrong project" container creation

## Comparison: Patch vs Systematic Fix

### Quick Patch (What We're NOT Doing)
```python
# Just update the registry file
await manager._save_to_registry()  # Band-aid
```

### Systematic Fix (What We ARE Doing)
- Singleton pattern for state management
- Dependency injection architecture
- Container discovery protocol
- Persistent state with validation
- Complete test coverage

## AI Expert Validation

**Gemini & Grok confirm this is the correct architectural approach:**
- "Singleton pattern essential for state consistency"
- "Dependency injection provides proper control flow"
- "Container discovery prevents resource waste"
- "Registry persistence enables recovery"

---

**Confidence: 95%**
*Assumptions: Docker API available, port range 48100-48199 sufficient, registry file atomic writes*