# ADR-0100: Comprehensive Container Orchestration

**Date:** September 24, 2025
**Status:** Proposed
**Supersedes:** ADR-0097, ADR-0098

## Context

Following ADR-0096 (Neo4j-Migrations), our indexer lifecycle orchestration stopped working properly across projects. Analysis with Gemini-2.5-pro identified 5 critical issues:

1. **No Proactive Initialization**: Containers only start when tools are called (cold start problem)
2. **Multiple Sources of Truth**: JSON files, in-memory dicts, and Docker state not synchronized
3. **No Lifecycle Management**: Containers don't stop when Claude sessions end
4. **No Health Monitoring**: Containers fail silently without detection
5. **No Session Awareness**: Can't track which Claude instance uses which containers

### September 2025 Best Practices Applied
- **Declarative Configuration**: Specify desired state, not steps
- **Health Monitoring**: HEALTHCHECK with automatic recovery
- **Graceful Shutdown**: SIGTERM signal handling
- **Resource Monitoring**: Track CPU/memory usage
- **Security**: Least privilege, vulnerability scanning
- **Auto-scaling**: Dynamic resource allocation

## Decision

Implement a comprehensive container orchestration system following Kubernetes-like patterns but using Docker as the foundation. This system will provide:

### 1. Proactive Container Initialization
```python
class ContainerOrchestrator:
    """Elite container orchestration with proactive lifecycle management"""

    async def initialize(self):
        """Called when MCP server starts"""
        # 1. Detect project using CLAUDE_PROJECT_DIR
        project_path = os.getenv("CLAUDE_PROJECT_DIR")

        # 2. Start essential containers proactively
        await self.start_container("neo4j", healthcheck=True)
        await self.start_container("redis-cache", healthcheck=True)
        await self.start_container("redis-queue", healthcheck=True)
        await self.start_container("nomic", healthcheck=True)

        # 3. Wait for health checks
        await self.wait_for_healthy(timeout=30)

        # 4. Start project indexer
        await self.start_indexer(project_path)
```

### 2. Docker as Single Source of Truth
```python
class DockerStateManager:
    """All state lives in Docker labels and container metadata"""

    def get_container_state(self, project_hash: str) -> Dict:
        """Read all state from Docker"""
        containers = self.docker.containers.list(
            all=True,
            filters={
                "label": [
                    "com.l9.managed=true",
                    f"com.l9.project_hash={project_hash}"
                ]
            }
        )

        return {
            "containers": [self._extract_metadata(c) for c in containers],
            "project_path": containers[0].labels.get("com.l9.project_path"),
            "session_count": int(containers[0].labels.get("com.l9.session_count", 0))
        }

    def update_container_state(self, container_id: str, updates: Dict):
        """Update Docker labels atomically"""
        container = self.docker.containers.get(container_id)
        labels = container.labels.copy()
        labels.update(updates)
        # Docker API atomic update
        container.reload()
```

### 3. Session-Aware Lifecycle Management
```python
class SessionManager:
    """Track Claude sessions with reference counting"""

    async def register_session(self, session_id: str, project_path: str):
        """Called when MCP initializes"""
        project_hash = self._hash_project(project_path)

        # Increment session count in Docker label
        containers = self._get_project_containers(project_hash)
        for container in containers:
            count = int(container.labels.get("com.l9.session_count", 0))
            self._update_label(container, "com.l9.session_count", str(count + 1))
            self._update_label(container, f"com.l9.session.{session_id}", str(time.time()))

    async def unregister_session(self, session_id: str):
        """Called when MCP shuts down or on SIGTERM"""
        for container in self._get_all_containers():
            if f"com.l9.session.{session_id}" in container.labels:
                # Remove session label
                self._remove_label(container, f"com.l9.session.{session_id}")

                # Decrement session count
                count = int(container.labels.get("com.l9.session_count", 1))
                new_count = max(0, count - 1)
                self._update_label(container, "com.l9.session_count", str(new_count))

                # Stop container if no more sessions
                if new_count == 0:
                    await self._graceful_shutdown(container)
```

### 4. Health Monitoring with Auto-Recovery
```python
class HealthMonitor:
    """Continuous health monitoring with automatic recovery"""

    def __init__(self):
        self.healthcheck_interval = 30  # seconds
        self.unhealthy_threshold = 3

    async def monitor_health(self):
        """Background task for health monitoring"""
        while True:
            containers = self._get_managed_containers()

            for container in containers:
                health = await self._check_health(container)

                if health["status"] == "unhealthy":
                    failures = int(container.labels.get("com.l9.health_failures", 0))

                    if failures >= self.unhealthy_threshold:
                        logger.warning(f"Container {container.name} exceeded failure threshold")
                        await self._recover_container(container)
                    else:
                        self._update_label(container, "com.l9.health_failures", str(failures + 1))
                else:
                    self._update_label(container, "com.l9.health_failures", "0")

            await asyncio.sleep(self.healthcheck_interval)

    async def _recover_container(self, container):
        """Attempt to recover unhealthy container"""
        logger.info(f"Attempting recovery for {container.name}")

        # 1. Try restart first
        container.restart()
        await asyncio.sleep(10)

        # 2. Check if healthy after restart
        health = await self._check_health(container)
        if health["status"] == "healthy":
            logger.info(f"✅ Recovery successful for {container.name}")
            return

        # 3. If still unhealthy, recreate
        logger.warning(f"Recreating container {container.name}")
        await self._recreate_container(container)
```

### 5. Container Configuration with Healthchecks
```yaml
# docker-compose.orchestrated.yml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.22.0
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    labels:
      com.l9.managed: "true"
      com.l9.service: "neo4j"
      com.l9.critical: "true"

  redis-cache:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    labels:
      com.l9.managed: "true"
      com.l9.service: "redis-cache"

  indexer:
    image: indexer:production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:48100/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    labels:
      com.l9.managed: "true"
      com.l9.service: "indexer"
      com.l9.project: "${PROJECT_NAME}"
```

### 6. Graceful Shutdown and Cleanup
```python
class GracefulShutdown:
    """Handle shutdown signals properly"""

    def __init__(self):
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        atexit.register(self._cleanup)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")

        # 1. Stop accepting new requests
        self.accepting_requests = False

        # 2. Wait for in-flight requests (max 30s)
        self._wait_for_requests(timeout=30)

        # 3. Unregister session
        asyncio.run(self.session_manager.unregister_session(self.session_id))

        # 4. Close connections
        self._close_connections()

        sys.exit(0)
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
- [ ] Create ContainerOrchestrator class
- [ ] Implement DockerStateManager
- [ ] Add health monitoring infrastructure
- [ ] Update MCP server initialization

### Phase 2: Session Management (Week 2)
- [ ] Implement SessionManager with reference counting
- [ ] Add SIGTERM signal handling
- [ ] Create session registration/unregistration
- [ ] Add session timeout detection

### Phase 3: Health & Recovery (Week 3)
- [ ] Implement HealthMonitor background task
- [ ] Add container recovery logic
- [ ] Create health check endpoints for all services
- [ ] Add monitoring dashboard

### Phase 4: Optimization (Week 4)
- [ ] Add connection pooling optimization
- [ ] Implement resource monitoring
- [ ] Add auto-scaling based on load
- [ ] Performance testing

## Testing Strategy

### Unit Tests
```python
def test_proactive_initialization():
    """Containers start before first tool call"""
    orchestrator = ContainerOrchestrator()
    await orchestrator.initialize()

    containers = orchestrator.get_running_containers()
    assert len(containers) >= 4  # neo4j, redis-cache, redis-queue, nomic

def test_session_reference_counting():
    """Containers stop only when all sessions end"""
    manager = SessionManager()

    # Two sessions start
    await manager.register_session("session1", "/project1")
    await manager.register_session("session2", "/project1")

    # One session ends - containers keep running
    await manager.unregister_session("session1")
    assert container_running("indexer-project1")

    # Last session ends - containers stop
    await manager.unregister_session("session2")
    assert not container_running("indexer-project1")
```

### Integration Tests
```bash
# Test concurrent Claude instances
./tests/test_concurrent_sessions.sh

# Test health recovery
./tests/test_container_recovery.sh

# Test graceful shutdown
./tests/test_graceful_shutdown.sh
```

## Success Criteria

1. **Zero Cold Starts**: First tool call has no initialization delay
2. **100% Health Detection**: All container failures detected within 60s
3. **Clean Session Tracking**: Containers stop when last session ends
4. **Automatic Recovery**: 95% of failures recover without manual intervention
5. **Resource Efficiency**: Idle containers consume < 100MB RAM
6. **Performance**: Tool calls complete in < 500ms (P95)

## Rollback Plan

If issues arise:
1. Revert to ADR-0096 behavior (on-demand start)
2. Keep Docker labels for debugging
3. Disable proactive initialization via feature flag
4. Fall back to manual container management

## References

- ADR-0096: Neo4j-Migrations framework
- ADR-0097: Container State Management (superseded)
- ADR-0098: Docker State Migration (superseded)
- ADR-0099: Fix Global MCP Project Detection
- September 2025 Container Orchestration Best Practices
- Kubernetes Pod Lifecycle Documentation
- Docker Healthcheck Best Practices

## Decision Outcome

This comprehensive orchestration system will:
1. Eliminate cold starts through proactive initialization
2. Use Docker as the single source of truth
3. Track sessions with reference counting
4. Monitor health and auto-recover
5. Clean up resources when sessions end

The result will be an "elite container orchestrator that works flawlessly" as requested, following September 2025 best practices while maintaining backward compatibility.