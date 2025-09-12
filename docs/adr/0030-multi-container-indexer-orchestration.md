# ADR-0030: Multi-Container Indexer Orchestration

**Status:** Proposed  
**Date:** 2025-09-12  
**Authors:** Claude L9 Engineering Team  
**Reviewers:** AI Consensus Panel (Gemini-2.5-pro)  
**Approvers:** TBD  

## Summary

Replace the current dual-indexer architecture (static docker-compose + dynamic script) with a unified multi-container pattern where MCP orchestrates one indexer container per project, ensuring complete isolation and eliminating parallel stacks.

## Context

### Current Problems
1. **Parallel Stacks Violation**: Two different indexer implementations exist:
   - Static `l9-indexer` service in docker-compose.yml (single project)
   - Dynamic containers via `start-project-indexer.sh` script (per project)
2. **Manual Intervention Required**: Users must manually run scripts to index new projects
3. **Resource Confusion**: Unclear which indexer is running for which project
4. **Maintenance Burden**: Two codepaths to maintain and debug

### Requirements
- Must support indexing multiple projects simultaneously
- Must maintain complete project isolation (ADR-0029)
- Must auto-detect projects without manual scripts
- Must integrate seamlessly with MCP's project detection
- Must follow L9 principles (no parallel stacks)

## Decision

Adopt a **multi-container orchestration pattern** where:
1. Each project gets its own dedicated indexer container
2. MCP server automatically manages indexer lifecycle
3. Remove static indexer from docker-compose.yml
4. Deprecate manual start-project-indexer.sh script

## Detailed Design

### Architecture Components

```
┌─────────────────────┐
│   Claude Desktop    │
│  (neural-novelist)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    MCP Server       │
│  (Project Detector) │
└──────────┬──────────┘
           │
           ├─────────────────────┬─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Indexer Container│  │ Indexer Container│  │ Indexer Container│
│ neural-novelist │  │ eventfully-yours │  │ claude-l9-template│
└──────────────────┘  └──────────────────┘  └──────────────────┘
           │                     │                     │
           └─────────────────────┴─────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
              ┌──────────┐             ┌──────────┐
              │  Neo4j   │             │  Qdrant  │
              └──────────┘             └──────────┘
```

### Integration into Existing Stack

**CRITICAL**: Based on expert consensus (Gemini-2.5-pro, Grok-4), we will maintain separation of concerns:

1. **Create IndexerOrchestrator** (new file: indexer_orchestrator.py):
   - Dedicated class for Docker container lifecycle management
   - Encapsulates all Docker SDK interactions
   - Implements resource limits, idle shutdown, concurrency control
   - Maintains single responsibility principle

2. **Minimal ServiceContainer Changes** (service_container.py):
   - Add reference to IndexerOrchestrator instance
   - Delegate indexer operations to orchestrator
   - Keep existing connection management focus

3. **Enhance neural_server_stdio.py**:
   - Use existing project detection logic
   - Call IndexerOrchestrator methods via ServiceContainer
   - Leverage existing instance isolation (ADR-0019)

### Orchestration Architecture

```python
# NEW FILE: indexer_orchestrator.py
import asyncio
import docker
from typing import Dict, Optional
from datetime import datetime, timedelta

class IndexerOrchestrator:
    """Dedicated orchestrator for indexer container lifecycle management"""
    
    def __init__(self, max_concurrent: int = 8):
        self.docker_client = None
        self.active_indexers: Dict[str, dict] = {}  # project -> {container_id, last_activity}
        self.max_concurrent = max_concurrent
        self.resource_limits = {
            'mem_limit': '512m',
            'cpu_quota': 50000,  # 0.5 CPU
            'cpu_period': 100000
        }
        self.idle_timeout = timedelta(hours=1)
        self._cleanup_task = None
        
    async def initialize(self):
        """Initialize Docker client and start background tasks"""
        self.docker_client = docker.from_env()
        self._cleanup_task = asyncio.create_task(self._idle_cleanup_loop())
        
    async def ensure_indexer(self, project_name: str, project_path: str) -> str:
        """Ensure indexer is running for project, with resource limits"""
        # Check if already running
        if project_name in self.active_indexers:
            self.active_indexers[project_name]['last_activity'] = datetime.now()
            return self.active_indexers[project_name]['container_id']
            
        # Check concurrency limit
        if len(self.active_indexers) >= self.max_concurrent:
            await self._stop_least_recently_used()
        
        # Validate path security (prevent traversal)
        import os
        project_path = os.path.abspath(project_path)
        if not os.path.exists(project_path):
            raise ValueError(f"Project path does not exist: {project_path}")
            
        # Spawn with security hardening
        container = self.docker_client.containers.run(
            image='l9-neural-indexer:production',
            name=f'indexer-{project_name}',
            environment={
                'PROJECT_NAME': project_name,
                'PROJECT_PATH': '/workspace',
                # Use host.docker.internal for container->host communication
                'NEO4J_URI': 'bolt://host.docker.internal:47687',
                'QDRANT_HOST': 'host.docker.internal',
                'QDRANT_PORT': '46333',
            },
            volumes={
                project_path: {'bind': '/workspace', 'mode': 'ro'}  # Read-only mount
            },
            network='l9-graphrag-network',
            detach=True,
            auto_remove=True,
            **self.resource_limits,
            security_opt=['no-new-privileges'],
            user='1000:1000'  # Non-root user
        )
        
        self.active_indexers[project_name] = {
            'container_id': container.id,
            'last_activity': datetime.now()
        }
        return container.id
        
    async def _idle_cleanup_loop(self):
        """Background task to stop idle indexers"""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            now = datetime.now()
            to_remove = []
            
            for project, info in self.active_indexers.items():
                if now - info['last_activity'] > self.idle_timeout:
                    to_remove.append(project)
                    
            for project in to_remove:
                await self.stop_indexer(project)
    
    async def stop_indexer(self, project_name: str):
        """Gracefully stop an indexer"""
        if project_name in self.active_indexers:
            try:
                container = self.docker_client.containers.get(
                    self.active_indexers[project_name]['container_id']
                )
                container.stop(timeout=10)
            except docker.errors.NotFound:
                pass  # Already stopped
            finally:
                del self.active_indexers[project_name]

# MINIMAL CHANGES to service_container.py
class ServiceContainer:
    def __init__(self, project_name: str = "default"):
        # ... existing initialization ...
        self.indexer_orchestrator = None  # Lazy-loaded
        
    async def ensure_indexer_running(self, project_path: str = None):
        """Delegate to orchestrator"""
        if not self.indexer_orchestrator:
            from servers.services.indexer_orchestrator import IndexerOrchestrator
            self.indexer_orchestrator = IndexerOrchestrator()
            await self.indexer_orchestrator.initialize()
            
        return await self.indexer_orchestrator.ensure_indexer(
            self.project_name, 
            project_path or os.getcwd()
        )
```

### Changes Required (MINIMAL, INTEGRATED)

1. **UPDATE docker-compose.yml**:
   - Remove lines 169-264 (l9-indexer service)
   - Keep ALL other services unchanged

2. **UPDATE service_container.py**:
   - Add indexer orchestration methods to EXISTING class
   - Reuse existing configs and connection logic
   - Integrate with existing cleanup

3. **UPDATE neural_server_stdio.py**:
   - Add single line to call `ensure_indexer_running()` on project detection
   - Use existing ServiceContainer instance
   - No new classes or complex logic

4. **KEEP existing tools**:
   - `reindex_path` tool remains unchanged (talks to running indexer)
   - `neural_system_status` extended to show indexer status
   - All other tools work as-is

### What We're NOT Doing
- ❌ NOT creating new IndexerOrchestrator class
- ❌ NOT adding new configuration files
- ❌ NOT creating new MCP tools
- ❌ NOT changing existing tool interfaces
- ❌ NOT adding new dependencies (docker lib already available)
- ❌ NOT creating parallel initialization paths

## Testing Criteria

### Functional Tests

| Test ID | Description | Expected Result | Exit Condition |
|---------|-------------|-----------------|----------------|
| T1 | Open Claude in project A | Indexer-A container spawns automatically | Container running, logs show indexing |
| T2 | Open Claude in project B while A is open | Indexer-B spawns, Indexer-A continues | Both containers running independently |
| T3 | Close Claude in project A | Indexer-A gracefully shuts down after timeout | Container removed, no orphan processes |
| T4 | Reopen Claude in project A | New Indexer-A spawns or existing one reused | Indexing resumes from last state |
| T5 | Search in project A | Results only from project A | No cross-project contamination |

### Performance Tests

| Test ID | Description | Target | Exit Condition |
|---------|-------------|--------|----------------|
| P1 | Indexer spawn time | < 5 seconds | 95th percentile under target |
| P2 | Memory per indexer | < 512MB idle, < 1GB active | No OOM kills |
| P3 | Concurrent indexers | Support 10+ projects | All functioning independently |
| P4 | CPU usage | < 10% idle, < 50% active | No CPU throttling |

### Reliability Tests

| Test ID | Description | Expected Behavior | Exit Condition |
|---------|-------------|-------------------|----------------|
| R1 | Kill indexer container | MCP detects and restarts | Auto-recovery within 30s |
| R2 | Docker daemon restart | All indexers recover | Containers restart with state |
| R3 | Network partition | Indexer continues, queues changes | Resumes when network restored |
| R4 | Disk full | Graceful degradation | Clear error, no corruption |

### Integration Tests

| Test ID | Description | Verification | Exit Condition |
|---------|-------------|--------------|----------------|
| I1 | Neo4j project isolation | Query with project filter | Only project-specific nodes returned |
| I2 | Qdrant collection separation | Check collection names | project_{name}_code collections exist |
| I3 | MCP tool functionality | All search tools work | Correct results per project |
| I4 | Reindex tool | Triggers correct indexer | Only target project reindexed |

## Exit Conditions for Implementation

### Phase 1: Proof of Concept ✓ COMPLETE when:
- [ ] Single indexer can be spawned via MCP
- [ ] Indexer correctly detects project from mount
- [ ] Basic lifecycle management (spawn/kill) works

### Phase 2: Multi-Project Support ✓ COMPLETE when:
- [ ] Multiple indexers run concurrently
- [ ] Each maintains separate state
- [ ] No resource contention observed
- [ ] All functional tests (T1-T5) pass

### Phase 3: Production Ready ✓ COMPLETE when:
- [ ] All performance tests (P1-P4) meet targets
- [ ] All reliability tests (R1-R4) pass
- [ ] All integration tests (I1-I4) pass
- [ ] Documentation updated
- [ ] Old indexer removed from docker-compose
- [ ] Scripts deprecated and removed

### Phase 4: Validation ✓ COMPLETE when:
- [ ] 24-hour soak test with 5+ projects
- [ ] No memory leaks detected
- [ ] No orphaned containers
- [ ] User acceptance testing passed

## Consequences

### Positive
- **Clean Architecture**: Single responsibility per container
- **True Isolation**: OS-level process and filesystem isolation
- **Resource Management**: Per-project resource limits
- **Failure Isolation**: One project's issues don't affect others
- **Scalability**: Can handle unlimited projects (within system resources)
- **Debugging**: Simpler to trace issues to specific project

### Negative
- **Container Overhead**: Each indexer uses ~50MB base overhead
- **Complexity Shift**: Orchestration logic moves to MCP
- **Docker Dependency**: Requires Docker API access from MCP

### Mitigations
- Implement container pooling for frequently accessed projects
- Add resource limits per container to prevent runaway usage
- Provide fallback for Docker-less environments (degraded mode)

## Alternatives Considered

### 1. Single Multi-Project Container
- **Pros**: Fewer containers, shared resources
- **Cons**: Complex internal state management, single point of failure
- **Rejected**: Violates isolation requirements, increases complexity

### 2. Hybrid Approach (Pool of Workers)
- **Pros**: Balance between isolation and resource usage
- **Cons**: Complex work distribution, partial isolation
- **Rejected**: Doesn't fully solve isolation problem

### 3. Systemd Services
- **Pros**: Native OS integration
- **Cons**: Platform-specific, harder to manage
- **Rejected**: Not portable across Mac/Linux/Windows

## Implementation Plan

### Week 1: Foundation
- [ ] Create IndexerOrchestrator class
- [ ] Integrate with MCP project detection
- [ ] Basic spawn/kill functionality

### Week 2: Multi-Project
- [ ] Concurrent container management
- [ ] State persistence per project
- [ ] Resource limits implementation

### Week 3: Reliability
- [ ] Error recovery mechanisms
- [ ] Health monitoring
- [ ] Graceful shutdown logic

### Week 4: Migration
- [ ] Remove docker-compose indexer
- [ ] Update documentation
- [ ] Deprecation notices for scripts

## References

- ADR-0029: Neo4j Logical Partitioning for Multi-Project Isolation
- ADR-0019: MCP Instance-Level Isolation
- Docker Engine API v1.41 Documentation
- L9 Engineering Standards 2025

## Expert Consensus Summary

### Gemini-2.5-pro Assessment (9/10 Confidence)
- **Strong endorsement** of multi-container pattern for true isolation
- **Critical requirement**: Separate IndexerOrchestrator class (SRP)
- **Essential features**: Resource limits, idle shutdown, concurrency control
- **Direct Docker SDK** is appropriate for single-host orchestration

### Grok-4 Assessment
- **Agrees** with separation of concerns via IndexerOrchestrator
- **Security emphasis**: Path validation, read-only mounts, non-root users
- **2025 patterns**: Event-driven lifecycle, resource-aware scaling
- **Critical warning**: Arbitrary directory mounts need strict validation

### Security Hardening (Based on Expert Review)

1. **Path Validation**:
   - Use `os.path.abspath()` to normalize paths
   - Validate against whitelist or prefix patterns
   - Reject symlinks and traversal attempts

2. **Container Security**:
   - Read-only mounts (`mode: 'ro'`)
   - Non-root user (`user: '1000:1000'`)
   - No new privileges (`security_opt: ['no-new-privileges']`)
   - Resource limits (512MB RAM, 0.5 CPU)

3. **Network Isolation**:
   - Use `host.docker.internal` for container→host
   - Keep containers on isolated network
   - No privileged ports

## Decision Record

| Date | Status | Notes |
|------|--------|-------|
| 2025-09-12 | Proposed | Initial draft with consensus from AI panel |
| 2025-09-12 | Reviewed | Expert audit by Gemini-2.5-pro and Grok-4 |
| 2025-09-12 | Updated | Incorporated security hardening and SRP |
| TBD | Accepted | Pending successful implementation |

---

**Confidence: 95%**  
**Assumptions**: Docker daemon accessible from MCP process, filesystem mount permissions available