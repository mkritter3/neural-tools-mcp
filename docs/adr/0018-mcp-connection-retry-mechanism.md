# ADR 0018: MCP Connection Retry Mechanism

**Date:** September 11, 2025  
**Status:** Proposed  
**Author:** L9 Engineering Team  

## Context

The MCP (Model Context Protocol) server experiences consistent connection failures on initial startup, requiring users to manually reconnect. This occurs because:

1. **Race Condition**: MCP server launches immediately when Claude invokes a tool
2. **Service Startup Time**: Docker containers need 10-30 seconds to fully initialize
3. **No Retry Logic**: Current implementation attempts connection once and fails
4. **Health Check Delays**: Services have `start_period` of 30-60 seconds before becoming healthy

### Current Behavior
- First MCP tool invocation fails with "Failed to reconnect to neural-tools"
- User must manually reconnect using `/mcp` command
- Second connection attempt succeeds (services are ready by then)

### Evidence
- Neo4j health check: 30s start_period
- Qdrant: No health check configured (commented out)
- ServiceContainer: 3-second timeout, no retries
- Connection attempt happens immediately on MCP server launch

## Decision

Implement a **robust retry mechanism with exponential backoff** in the ServiceContainer initialization to handle service startup delays gracefully.

## Solution Design

### Phase 1: Basic Retry Logic (Immediate Fix)
Add retry logic to `ServiceContainer.initialize()`:

```python
async def initialize_with_retry(self) -> bool:
    """Initialize services with retry logic for container startup"""
    max_retries = 5
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Try to connect to all services
            neo4j_ok = self.ensure_neo4j_client()
            qdrant_ok = self.ensure_qdrant_client()
            
            if neo4j_ok and qdrant_ok:
                logger.info(f"✅ Services connected on attempt {attempt + 1}")
                return self._complete_initialization()
            
            # Calculate exponential backoff
            delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16, 32 seconds
            
            if attempt < max_retries - 1:
                logger.info(f"⏳ Services not ready, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    
    return False
```

### Phase 2: Health Check Verification
Add explicit health checks before connection:

```python
async def wait_for_services_healthy(self, timeout: int = 60) -> bool:
    """Wait for all services to pass health checks"""
    start_time = time.time()
    
    health_checks = {
        'neo4j': self._check_neo4j_health,
        'qdrant': self._check_qdrant_health,
        'redis_cache': self._check_redis_cache_health,
        'redis_queue': self._check_redis_queue_health,
        'nomic': self._check_nomic_health
    }
    
    while time.time() - start_time < timeout:
        all_healthy = True
        
        for service, check_func in health_checks.items():
            if not await check_func():
                logger.debug(f"Service {service} not yet healthy")
                all_healthy = False
                break
        
        if all_healthy:
            logger.info("✅ All services healthy and ready")
            return True
            
        await asyncio.sleep(2)
    
    logger.error(f"Services failed to become healthy within {timeout}s")
    return False
```

### Phase 3: Progressive Connection Strategy
Implement progressive connection with partial functionality:

```python
class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    PARTIAL = "partial"  # Some services connected
    DEGRADED = "degraded"  # Core services only
    FULL = "full"  # All services connected

async def progressive_initialization(self):
    """Initialize with progressive service availability"""
    essential_services = ['neo4j', 'qdrant']  # Must have
    optional_services = ['redis_cache', 'redis_queue', 'nomic']  # Nice to have
    
    # First, get essential services
    if not await self._connect_essential_services(timeout=30):
        return ConnectionState.DISCONNECTED
    
    # Then try optional services with shorter timeout
    connected_optional = await self._connect_optional_services(timeout=10)
    
    if connected_optional == len(optional_services):
        return ConnectionState.FULL
    elif connected_optional > 0:
        return ConnectionState.PARTIAL
    else:
        return ConnectionState.DEGRADED
```

### Phase 4: Circuit Breaker Integration
Add circuit breaker to prevent connection storms:

```python
class ServiceConnectionBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError(f"Circuit open, retry after {self.recovery_timeout}s")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

## Implementation Plan

### Week 1: Basic Retry (Phase 1)
- [ ] Add retry logic to ServiceContainer.initialize()
- [ ] Add exponential backoff calculation
- [ ] Add logging for retry attempts
- [ ] Test with various startup scenarios

### Week 2: Health Checks (Phase 2)
- [ ] Implement health check functions for each service
- [ ] Add wait_for_services_healthy method
- [ ] Enable Qdrant health check in docker-compose.yml
- [ ] Test health check timing

### Week 3: Progressive Connection (Phase 3)
- [ ] Implement ConnectionState enum
- [ ] Add progressive_initialization method
- [ ] Update MCP server to handle partial connections
- [ ] Add fallback behavior for missing services

### Week 4: Circuit Breaker (Phase 4)
- [ ] Implement ServiceConnectionBreaker class
- [ ] Integrate with ServiceContainer
- [ ] Add metrics for circuit breaker events
- [ ] Test failure scenarios

## Success Metrics

1. **Connection Success Rate**: >99% on first attempt (after retry logic)
2. **Average Connection Time**: <10 seconds (down from 30+ seconds)
3. **User Interventions**: 0 manual reconnections required
4. **Partial Availability**: Can operate with degraded functionality if optional services fail

## Testing Strategy

1. **Cold Start Test**: All containers stopped, then started together
2. **Rolling Start Test**: Services started one by one with delays
3. **Failure Injection**: Random service failures during initialization
4. **Network Partition**: Simulate network delays and timeouts
5. **Load Test**: Multiple MCP sessions starting simultaneously

## Rollback Plan

If the retry mechanism causes issues:

1. **Immediate**: Set `DISABLE_RETRY=true` environment variable
2. **Short-term**: Revert to single connection attempt
3. **Document**: Add user instructions for manual retry

## Alternatives Considered

1. **Pre-warming Services**: Start services long before MCP
   - **Rejected**: Requires changing user workflow

2. **Lazy Connection**: Connect only when service is needed
   - **Rejected**: Would cause delays during operations

3. **Docker Healthcheck Dependencies**: Use depends_on with condition
   - **Rejected**: Not all services have reliable health checks

4. **Synchronous Polling**: Block until services ready
   - **Rejected**: Would freeze Claude's UI

## References

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Exponential Backoff Algorithm](https://en.wikipedia.org/wiki/Exponential_backoff)
- [Docker Compose Health Checks](https://docs.docker.com/compose/compose-file/05-services/#healthcheck)
- MCP Protocol Specification 2025-06-18

## Decision Outcome

**Approved for Implementation**: Phase 1 (Basic Retry) immediately, Phases 2-4 as time permits.

The retry mechanism will eliminate the need for manual reconnection while maintaining system responsiveness.

---

**Confidence: 95%**  
**Assumptions**: Docker containers continue to stabilize within 30 seconds, MCP protocol supports async initialization