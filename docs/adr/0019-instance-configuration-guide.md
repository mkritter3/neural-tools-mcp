# ADR-19 Instance Configuration Guide

## Claude-Side Configuration Options

Since we control the `.mcp.json` configuration that Claude reads, we can implement significant parts of Phase 4 (Process-Level Isolation) through configuration.

## Configuration Files

### 1. Standard Configuration (`.mcp.json`)
The basic configuration with instance isolation support:

```json
{
  "mcpServers": {
    "neural-tools": {
      "env": {
        // Instance isolation (Phase 1-2)
        "INSTANCE_ID": "${CLAUDE_INSTANCE_ID:-}",
        "INSTANCE_TIMEOUT_HOURS": "1",
        "CLEANUP_INTERVAL_MINUTES": "10",
        "ENABLE_AUTO_CLEANUP": "true",
        
        // Monitoring (Phase 3)
        "MCP_VERBOSE": "false",
        "INCLUDE_INSTANCE_METADATA": "false"
      }
    }
  }
}
```

### 2. Advanced Configuration (`.mcp.advanced.json`)
Enhanced configuration with full isolation features and metadata.

## Environment Variables Claude Could Provide

While Claude doesn't currently provide these, our configuration is ready for them:

| Variable | Purpose | Fallback |
|----------|---------|----------|
| `${CLAUDE_INSTANCE_ID}` | Unique ID per Claude window | Process-based ID |
| `${CLAUDE_WINDOW_ID}` | Window/tab identifier | Process-based ID |
| `${CLAUDE_PROJECT_PATH}` | Current project directory | Working directory |
| `${MCP_DEBUG}` | Debug mode toggle | false |

## Configuration Modes

### Development Mode
```bash
# Enable verbose debugging
export MCP_DEBUG=true

# Short cleanup intervals
INSTANCE_TIMEOUT_HOURS=1
CLEANUP_INTERVAL_MINUTES=10
```

### Production Mode
```bash
# Disable verbose output
export MCP_DEBUG=false

# Longer cleanup intervals
INSTANCE_TIMEOUT_HOURS=4
CLEANUP_INTERVAL_MINUTES=30
```

### Debug Mode
```bash
# Maximum verbosity
MCP_VERBOSE=true
INCLUDE_INSTANCE_METADATA=true
LOG_LEVEL=DEBUG
```

## Per-Instance Isolation Strategies

### Strategy 1: Process-Based (Current)
Each Claude instance spawns its own MCP subprocess:
- ✅ Complete isolation
- ✅ No configuration needed
- ❌ Can't share resources between windows

### Strategy 2: Instance ID Based (Implemented)
Shared process with instance-level state isolation:
- ✅ Resource sharing possible
- ✅ Automatic cleanup
- ✅ Migration support
- ⚠️ Requires instance ID

### Strategy 3: Port-Based (Future)
Each instance on different port:
```json
{
  "mcpServers": {
    "neural-tools-1": {
      "command": "python3",
      "args": ["server.py", "--port", "50001"],
      "env": {"INSTANCE_ID": "instance-1"}
    },
    "neural-tools-2": {
      "command": "python3",
      "args": ["server.py", "--port", "50002"],
      "env": {"INSTANCE_ID": "instance-2"}
    }
  }
}
```

## Testing Instance Isolation

### 1. Verify Instance ID
```bash
# Check instance ID in tool response
MCP_VERBOSE=true claude

# Look for instance_id in _metadata field
```

### 2. Monitor Instance Metrics
Use the `instance_metrics` tool to see:
- Current instance ID
- Total active instances
- Stale instances pending cleanup
- Cleanup statistics

### 3. Test Cross-Instance Isolation
1. Open two Claude windows
2. Create test files in each:
   ```bash
   # Window 1
   echo "INSTANCE_A" > /tmp/test_a.txt
   
   # Window 2
   echo "INSTANCE_B" > /tmp/test_b.txt
   ```
3. Verify each can only see its own file

## Troubleshooting

### Issue: Instances Share State
**Solution**: Ensure `INSTANCE_ID` is set uniquely:
```json
"INSTANCE_ID": "${CLAUDE_INSTANCE_ID:-${RANDOM}}"
```

### Issue: Stale Instances Not Cleaned
**Solution**: Check cleanup configuration:
```json
"ENABLE_AUTO_CLEANUP": "true",
"INSTANCE_TIMEOUT_HOURS": "1"
```

### Issue: Can't See Instance Information
**Solution**: Enable verbose mode:
```json
"MCP_VERBOSE": "true",
"INCLUDE_INSTANCE_METADATA": "true"
```

## Best Practices

1. **Production Settings**
   - Keep `MCP_VERBOSE=false` to reduce overhead
   - Set reasonable cleanup intervals (30+ minutes)
   - Use longer timeouts (4+ hours)

2. **Development Settings**
   - Enable `MCP_VERBOSE=true` for debugging
   - Use short cleanup intervals for testing
   - Monitor with `instance_metrics` tool

3. **Security**
   - Each instance has isolated state
   - No cross-instance data access
   - Automatic cleanup of abandoned sessions

## Migration Between Instances

If needed, state can be migrated:

1. Export from old instance:
   ```python
   state = await export_instance_state()
   ```

2. Import to new instance:
   ```python
   await import_instance_state(state, merge=False)
   ```

## Future Enhancements

### Claude Could Provide:
1. `CLAUDE_INSTANCE_ID` environment variable
2. `CLAUDE_PROJECT_CONTEXT` for project detection
3. Instance lifecycle hooks (on_start, on_stop)
4. Multi-server coordination support

### We're Ready For:
- ✅ Unique instance IDs
- ✅ Instance-specific configuration
- ✅ Automatic resource management
- ✅ State migration
- ✅ Verbose debugging
- ✅ Metrics and monitoring

## Summary

The current configuration provides:
- **Complete isolation** between Claude instances
- **Automatic cleanup** of stale resources
- **Production-ready** monitoring and debugging
- **Zero configuration** needed (works with defaults)
- **Future-proof** for Claude enhancements