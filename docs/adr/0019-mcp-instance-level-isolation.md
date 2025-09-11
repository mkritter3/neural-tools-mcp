# ADR-0019: MCP Instance-Level Session Isolation

## Status
Proposed

## Context

We discovered that multiple Claude instances connecting to our MCP server are sharing the same session state, leading to:
- Cross-talk between different Claude instances
- Prompts from one instance appearing in another
- Shared file system access without isolation
- No privacy/security boundaries between sessions

Current state:
- We have project-level isolation (MultiProjectServiceState)
- But all Claude instances share the same MCP server process
- STDIO transport is designed for 1:1 communication
- Environment variable `ENABLE_SESSION_ISOLATION=true` exists but isn't enforced

## Decision

Implement a **three-tier isolation strategy** that maintains backward compatibility:

### Tier 1: Instance ID via Environment (Immediate, Non-Breaking)
```python
# Each Claude instance gets unique INSTANCE_ID in .mcp.json
"env": {
    "INSTANCE_ID": "${CLAUDE_INSTANCE_ID}",  # Unique per Claude window
    "PROJECT_NAME": "default",
    ...
}
```

### Tier 2: Enhanced Session Management (Short-term, Non-Breaking)
```python
class EnhancedMultiProjectServiceState:
    def __init__(self):
        self.instance_containers = {}  # instance_id -> project_containers
        self.instance_id = os.getenv('INSTANCE_ID', 'default')
        
    async def get_instance_container(self, instance_id: str = None):
        if not instance_id:
            instance_id = self.instance_id
        
        if instance_id not in self.instance_containers:
            self.instance_containers[instance_id] = {
                'project_containers': {},
                'project_retrievers': {},
                'session_started': datetime.now(),
                'last_activity': datetime.now()
            }
        return self.instance_containers[instance_id]
```

### Tier 3: Process-Level Isolation (Long-term, Claude-side change)
```json
// Future .mcp.json enhancement
{
  "mcpServers": {
    "neural-tools": {
      "command": "python3",
      "args": ["..."],
      "isolation": "per-instance",  // New field
      "portRange": [50000, 50100]   // Dynamic port allocation
    }
  }
}
```

## Implementation Plan (Non-Breaking)

### Phase 1: Instance Detection (Immediate)
```python
# In neural_server_stdio.py
import uuid
import hashlib

def get_instance_id():
    """Get or generate instance ID"""
    # Priority order:
    # 1. Environment variable (if Claude provides it)
    instance_id = os.getenv('INSTANCE_ID')
    if instance_id:
        return instance_id
    
    # 2. Process-based ID (fallback)
    pid = os.getpid()
    ppid = os.getppid()
    
    # 3. Hash of stdin/stdout file descriptors (unique per connection)
    stdin_stat = os.fstat(sys.stdin.fileno())
    unique_string = f"{pid}:{ppid}:{stdin_stat.st_ino}:{stdin_stat.st_dev}"
    
    return hashlib.md5(unique_string.encode()).hexdigest()[:8]
```

### Phase 2: Isolated State Management
```python
class IsolatedServiceState:
    def __init__(self):
        self.instance_id = get_instance_id()
        self.instances = {}
        logger.info(f"🔐 MCP Instance ID: {self.instance_id}")
    
    async def get_container(self, project_name: str = None):
        """Get container isolated by instance AND project"""
        if not project_name:
            project_name = self.detect_project_from_context()
        
        key = f"{self.instance_id}:{project_name}"
        
        if key not in self.instances:
            self.instances[key] = {
                'container': ServiceContainer(project_name),
                'created': datetime.now(),
                'last_used': datetime.now()
            }
        
        instance = self.instances[key]
        instance['last_used'] = datetime.now()
        
        # Lazy init as before
        if not instance['container'].initialized:
            await instance['container'].initialize_all_services()
        
        return instance['container']
```

### Phase 3: Resource Cleanup
```python
async def cleanup_stale_instances():
    """Clean up instances inactive for >1 hour"""
    now = datetime.now()
    stale_threshold = timedelta(hours=1)
    
    for key, instance in list(state.instances.items()):
        if now - instance['last_used'] > stale_threshold:
            logger.info(f"🧹 Cleaning stale instance: {key}")
            # Close connections
            container = instance['container']
            if container.neo4j_driver:
                container.neo4j_driver.close()
            if container.qdrant_client:
                container.qdrant_client.close()
            del state.instances[key]
```

### Phase 4: Instance Identification in Responses
```python
@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    # Add instance ID to all responses for debugging
    instance_id = state.instance_id
    
    try:
        result = await execute_tool(name, arguments)
        
        # Add metadata if verbose mode
        if os.getenv('MCP_VERBOSE') == 'true':
            result['_metadata'] = {
                'instance_id': instance_id,
                'project': current_project,
                'timestamp': datetime.now().isoformat()
            }
        
        return result
    except Exception as e:
        logger.error(f"[Instance {instance_id}] Tool error: {e}")
        raise
```

## Migration Strategy (Zero Downtime)

1. **Week 1**: Deploy instance detection code (backward compatible)
   - Falls back to current behavior if no instance ID
   - Logs instance IDs for monitoring

2. **Week 2**: Enable instance isolation for new connections
   - Existing connections continue using shared state
   - New connections get isolated state

3. **Week 3**: Add cleanup and monitoring
   - Stale instance cleanup
   - Metrics on instance count and resource usage

4. **Week 4**: Full rollout
   - All connections use instance isolation
   - Deprecate shared state code path

## Consequences

### Positive
- Complete isolation between Claude instances
- No cross-talk or shared prompts
- Better security and privacy
- Easier debugging (instance IDs in logs)
- Resource cleanup for stale sessions

### Negative
- Slightly higher memory usage (multiple containers)
- Need to track instance lifecycle
- Potential for resource leaks if cleanup fails

### Neutral
- Each instance maintains its own connection pools
- Project-level isolation still works within each instance
- Backward compatible with existing code

## Testing Strategy

```python
# Test file: test_instance_isolation.py
async def test_instance_isolation():
    # Simulate two instances
    os.environ['INSTANCE_ID'] = 'test-instance-1'
    state1 = IsolatedServiceState()
    container1 = await state1.get_container('project-a')
    
    os.environ['INSTANCE_ID'] = 'test-instance-2'
    state2 = IsolatedServiceState()
    container2 = await state2.get_container('project-a')
    
    # Verify they're different instances
    assert container1 is not container2
    assert state1.instance_id != state2.instance_id
```

## Monitoring

Add metrics for:
- Active instance count
- Instance creation rate
- Instance cleanup rate
- Resource usage per instance
- Cross-talk detection (log analysis)

## Security Considerations

- Instance IDs should not be guessable
- Log instance IDs for audit trails
- Consider rate limiting per instance
- Implement instance timeout (configurable)

## References

- MCP Specification 2025-06-18
- Issue: Multiple Claude instances sharing state
- OAuth 2.1 Resource Server patterns
- Docker container isolation patterns

## Decision Outcome

Implement instance-level isolation using a phased approach that maintains backward compatibility while providing immediate benefits through instance detection and isolated state management.

**Target completion: 2 weeks for basic isolation, 4 weeks for full implementation**