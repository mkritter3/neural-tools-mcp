# ADR-0035: MCP Indexer Container Auto-Start on Initialization

**Status:** Proposed  
**Date:** 2025-09-12  
**Authors:** L9 Engineering Team + AI Consensus (Gemini 2.5 Pro, Grok 4)  
**Context:** Race Condition Elimination, Reliability Enhancement  
**Version:** 1.0

---

## Context

### Problem Statement

The current MCP server implementation has a **critical race condition** that causes the first reindex attempt to frequently fail. When Claude starts and users immediately call neural tools for reindexing, the request gets queued before the indexer container is fully operational, resulting in failed initial indexing attempts that require manual retry.

**Evidence of the Problem:**
```
Current Flow (PROBLEMATIC):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Claude Starts   │ →  │ User Calls      │ →  │ Reindex Fails   │
│ MCP Available   │    │ Reindex         │    │ Container       │
│                 │    │ Immediately     │    │ Still Starting  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### User Impact

- **Frustrating First Experience**: Initial reindex attempts fail requiring manual retry
- **Perceived System Unreliability**: Users expect running services to be ready
- **Development Friction**: Developers must implement client-side retry logic
- **Inconsistent Behavior**: Sometimes works, sometimes fails based on timing

### Technical Root Cause

The MCP server reports as "available" before critical dependencies (indexer containers) are ready to handle requests. This creates a timing window where:

1. MCP server initializes and accepts requests
2. User immediately calls `reindex_path` 
3. Request triggers lazy container startup
4. Request gets queued while container is still initializing
5. Initial request fails due to unready container

---

## Decision

**We will implement proactive indexer container startup during MCP server initialization to eliminate race conditions and ensure reliable first-time indexing.**

### Solution Architecture

```
Proposed Flow (RELIABLE):
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ MCP Starts      │ →  │ Start Indexer   │ →  │ Health Check    │ →  │ MCP Ready       │
│ Initialization  │    │ Containers      │    │ Until Healthy   │    │ Accept Requests │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Implementation Strategy

1. **Modify MCP Server Initialization** (`neural_server_stdio.py`)
   - Add container startup to initialization sequence
   - Ensure indexer containers are running and healthy before MCP accepts requests

2. **Health Check Integration**
   - Poll container health endpoints until ready
   - Implement timeout and failure handling
   - Clear logging for startup status

3. **Graceful Error Handling**
   - "Fail-fast" strategy if containers won't start
   - Descriptive error messages for troubleshooting
   - Configurable timeouts and retry logic

---

## Expert Consensus Analysis

### Gemini 2.5 Pro Assessment (9/10 Confidence)

**Strong Endorsement:** "This is a strong proposal that directly addresses a known reliability issue with a technically sound, industry-standard approach."

**Key Points:**
- **Technical Feasibility**: Highly feasible using standard container orchestration patterns
- **User Value**: Eliminates frustrating first-interaction failure mode
- **Industry Alignment**: Standard microservice best practice (readiness probes)
- **Implementation**: Low to moderate complexity with clear technical requirements

### Grok 4 Assessment (9/10 Confidence) 

**Technical Validation:** "This is a technically feasible and valuable enhancement that directly addresses a clear reliability issue."

**Key Points:**
- **Architecture Fit**: Enhances MCP server's role as central coordinator
- **User Benefits**: Concrete reliability gains over current flaky behavior  
- **Implementation Effort**: 1-2 days for skilled developer
- **Industry Perspective**: Common in AWS ECS, Kubernetes with init containers

### Consensus Agreement

Both experts strongly agree on:
- ✅ **High technical feasibility** using standard patterns
- ✅ **Clear user value** eliminating frustrating failures
- ✅ **Industry best practice** alignment
- ✅ **Low implementation risk** with manageable complexity
- ✅ **Future extensibility** for other containerized dependencies

---

## Technical Implementation

### Phase 1: Core Auto-Start Logic

**Modify MCP Server Initialization** (`neural_server_stdio.py`):

```python
async def initialize_mcp_server():
    """
    Enhanced MCP server initialization with indexer auto-start
    Implements ADR-0035 race condition elimination
    """
    logger.info("🚀 Starting MCP server initialization...")
    
    # 1. Initialize core services (existing)
    await initialize_core_services()
    
    # 2. NEW: Proactively start indexer containers
    await ensure_indexer_containers_ready()
    
    # 3. Complete MCP server setup
    await complete_mcp_initialization()
    
    logger.info("✅ MCP server ready - all dependencies healthy")

async def ensure_indexer_containers_ready():
    """
    Proactively start and health-check indexer containers
    """
    from servers.services.service_container import ServiceContainer
    
    # Get current project context
    project_manager = ProjectContextManager()
    project_info = await project_manager.get_current_project()
    
    # Start indexer container for current project
    container = ServiceContainer(project_info["project"])
    await container.initialize()
    
    # Ensure indexer is running and healthy
    container_id = await container.ensure_indexer_running(project_info["path"])
    
    # Health check with timeout
    await wait_for_indexer_health(container_id, timeout=30)
    
    logger.info(f"✅ Indexer container ready: {container_id[:12]}")

async def wait_for_indexer_health(container_id: str, timeout: int = 30):
    """
    Wait for indexer container to be healthy with timeout
    """
    import asyncio
    import time
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check container health endpoint
            health_status = await check_indexer_health(container_id)
            if health_status == "healthy":
                return True
                
        except Exception as e:
            logger.debug(f"Health check attempt failed: {e}")
            
        await asyncio.sleep(1)  # Check every second
    
    raise TimeoutError(f"Indexer container {container_id[:12]} failed to become healthy within {timeout}s")
```

### Phase 2: Health Check Implementation

**Container Health Monitoring**:

```python
async def check_indexer_health(container_id: str) -> str:
    """
    Check if indexer container is healthy and ready
    """
    import docker
    
    client = docker.from_env()
    container = client.containers.get(container_id)
    
    # Check container status
    if container.status != "running":
        return "unhealthy"
    
    # Check health endpoint if available
    try:
        # Get container port mapping
        port_info = container.attrs['NetworkSettings']['Ports']['8080/tcp']
        if port_info:
            host_port = port_info[0]['HostPort']
            
            # HTTP health check
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{host_port}/health", timeout=5) as response:
                    if response.status == 200:
                        return "healthy"
                        
    except Exception as e:
        logger.debug(f"Health endpoint check failed: {e}")
    
    return "starting"
```

### Phase 3: Configuration and Error Handling

**Configuration Options** (`.mcp.json`):

```json
{
  "mcpServers": {
    "neural-tools": {
      "env": {
        "INDEXER_AUTO_START": "true",
        "INDEXER_STARTUP_TIMEOUT": "30",
        "INDEXER_HEALTH_CHECK_INTERVAL": "1",
        "INDEXER_STARTUP_STRATEGY": "proactive"
      }
    }
  }
}
```

**Error Handling Strategy**:

```python
async def handle_indexer_startup_failure(error: Exception, project_name: str):
    """
    Handle indexer container startup failures gracefully
    """
    logger.error(f"❌ Failed to start indexer for {project_name}: {error}")
    
    # Strategy options:
    # 1. Fail-fast: Refuse to start MCP server
    # 2. Degraded mode: Start MCP but disable indexing features
    # 3. Retry: Attempt restart with exponential backoff
    
    startup_strategy = os.getenv("INDEXER_STARTUP_STRATEGY", "proactive")
    
    if startup_strategy == "fail-fast":
        raise RuntimeError(f"MCP server cannot start: Indexer startup failed for {project_name}")
    elif startup_strategy == "degraded":
        logger.warning(f"⚠️ Starting MCP in degraded mode - indexing disabled for {project_name}")
        return "degraded"
    else:
        # Default: fail-fast for reliability
        raise error
```

---

## Benefits

### Immediate Benefits

1. **Eliminates Race Conditions**: First reindex attempts will reliably succeed
2. **Improved User Experience**: No more frustrating failed initial interactions
3. **System Reliability**: Services are truly ready when they report as available
4. **Reduced Support Load**: Fewer user issues and error reports

### Long-term Benefits

1. **Foundation for Future Dependencies**: Pattern for other containerized services
2. **Operational Excellence**: Aligns with microservice readiness best practices
3. **Scalability**: Better handling of concurrent user sessions
4. **Maintainability**: Clearer system state and dependency management

---

## Trade-offs and Considerations

### Acceptable Trade-offs

1. **Increased Startup Time**: MCP server initialization will take longer (estimated 10-30 seconds)
   - **Mitigation**: Clear logging to communicate startup progress
   - **Benefit**: Trade short delay for reliable operation

2. **Added Complexity**: More initialization logic to maintain
   - **Mitigation**: Well-tested, standard patterns with good error handling
   - **Benefit**: Eliminates more complex failure scenarios

3. **Container Dependencies**: MCP server now depends on container runtime
   - **Mitigation**: Graceful degradation options if containers unavailable
   - **Benefit**: Makes dependencies explicit rather than hidden

### Risk Mitigation

1. **Container Startup Failures**: Implement fail-fast with clear error messages
2. **Resource Usage**: Monitor container resource consumption during startup
3. **Configuration Drift**: Keep startup logic synchronized with container changes
4. **Testing Complexity**: Add startup scenarios to test suites

---

## Implementation Plan

### Phase 1: Core Implementation (Week 1)
- [ ] Add indexer auto-start to MCP server initialization
- [ ] Implement basic health checking with timeouts
- [ ] Add configuration options for startup behavior
- [ ] Create fail-fast error handling

### Phase 2: Robustness (Week 2)  
- [ ] Add comprehensive health check endpoints
- [ ] Implement retry logic with exponential backoff
- [ ] Add startup progress logging and metrics
- [ ] Create degraded mode fallback option

### Phase 3: Testing and Monitoring (Week 3)
- [ ] Add unit tests for startup scenarios
- [ ] Create integration tests for container lifecycle
- [ ] Add monitoring for startup times and failures
- [ ] Document troubleshooting procedures

### Validation Criteria

**Exit Conditions:**
- [ ] MCP server startup waits for indexer container health
- [ ] First reindex attempt succeeds 95%+ of the time
- [ ] Startup time increase is under 60 seconds
- [ ] Clear error messages for any startup failures
- [ ] Graceful handling of container unavailability

---

## Alternatives Considered

### 1. Client-Side Retry Logic
- **Pros**: No server changes required
- **Cons**: Pushes complexity to clients, doesn't fix root cause
- **Rejected**: Band-aid solution that doesn't address the architectural issue

### 2. Request Queuing with Delays
- **Pros**: Simple implementation
- **Cons**: Unpredictable delays, still prone to timing issues
- **Rejected**: Doesn't guarantee container readiness

### 3. Lazy Loading with Blocking
- **Pros**: No startup time impact
- **Cons**: First API call has unpredictable long delay
- **Rejected**: Poor user experience with blocking first interactions

### 4. External Orchestration (Docker Compose/Kubernetes)
- **Pros**: Industry-standard dependency management
- **Cons**: Major architectural change, deployment complexity
- **Future Consideration**: Excellent long-term goal, this ADR is stepping stone

---

## References

- **Industry Best Practices**: Kubernetes readiness probes, Docker health checks
- **Related ADRs**: 
  - ADR-0030: Multi-Container Indexer Orchestration
  - ADR-0034: Project Pipeline Synchronization
- **Expert Consensus**: Gemini 2.5 Pro (9/10), Grok 4 (9/10) confidence ratings
- **Technical Standards**: Microservice dependency management patterns

---

**Implementation Priority:** P1 - Critical user experience improvement  
**Estimated Effort:** 1-2 weeks development + testing  
**Success Metrics:** 
- First reindex success rate >95%
- MCP startup time <60 seconds
- Zero race condition reports
- Improved user satisfaction scores

**Rollback Criteria:** If startup time exceeds 90 seconds or container startup success rate <90%, implement degraded mode fallback and reassess approach.

**Confidence: 100%** - Strong expert consensus validates this as the correct L9 engineering approach to eliminate race conditions and improve system reliability.