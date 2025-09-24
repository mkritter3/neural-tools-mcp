# Multi-Project Orchestrator Implementation Summary

**Date:** September 2025
**Status:** Phase 0-1 Complete, Ready for Testing
**Risk:** MINIMAL - Feature-flagged, legacy mode preserved

## What We Built (Based on Gemini's L9 Analysis)

After deep analysis with Gemini, we implemented **Option 3: Enhanced MCP Orchestrator** as the safest, most incremental approach.

### ✅ Completed Components

1. **ADR-0097**: Complete architectural decision record
2. **Project Orchestrator** (`project_orchestrator.py`):
   - Feature-flagged with `MCP_MULTI_PROJECT_MODE` env var
   - Redis-based state registry for project tracking
   - Deterministic project ID generation from paths
   - Container lifecycle management with heartbeats
   - Automatic cleanup of idle containers (janitor process)
   - Resource limits (2GB RAM, 512 CPU shares per container)
   - Max 10 concurrent projects (configurable)

3. **Integration Helper** (`orchestrator_integration.py`):
   - Simple `ensure_indexer_for_project()` function
   - Automatic project detection from `$CLAUDE_PROJECT_DIR`
   - Singleton orchestrator with connection pooling
   - Template for updating existing MCP tools

4. **Comprehensive Tests** (`test_project_orchestrator.py`):
   - Legacy mode preservation verified
   - Multi-project container management tested
   - Port allocation conflict prevention
   - Cleanup and shutdown procedures validated

## Safety Guarantees

### 🛡️ Zero Breaking Changes
```bash
# Legacy mode (DEFAULT - current behavior)
MCP_MULTI_PROJECT_MODE=false  # or not set

# Multi-project mode (NEW - opt-in)
MCP_MULTI_PROJECT_MODE=true
```

### 🔄 Instant Rollback
If anything goes wrong:
1. Set `MCP_MULTI_PROJECT_MODE=false`
2. Restart MCP server
3. System immediately reverts to exact current behavior

## How It Works

### Project Detection
```python
# Claude sets this on launch
$CLAUDE_PROJECT_DIR = "/path/to/current/project"

# Orchestrator generates deterministic ID
project_id = sha1(normalized_path)[:12]  # e.g., "a3f2b1c9d8e7"
```

### Container Naming
```
Old: indexer-{project}-{timestamp}-{random}
New: mcp-indexer-{project_id}  # Deterministic, discoverable
```

### State Registry (Redis)
```json
{
  "a3f2b1c9d8e7": {
    "container_id": "abc123...",
    "container_name": "mcp-indexer-a3f2b1c9d8e7",
    "port": 48101,
    "status": "running",
    "last_heartbeat": 1727193600,
    "project_path": "/Users/mkr/project1"
  }
}
```

## Next Steps (Minimal & Safe)

### 1. Test in Development (No Risk)
```bash
# In your test project
export MCP_MULTI_PROJECT_MODE=true
export CLAUDE_PROJECT_DIR=/path/to/test/project

# Run the test
cd /Users/mkr/local-coding/claude-l9-template
python3 tests/test_project_orchestrator.py
```

### 2. Update ONE MCP Tool (Reversible)
Pick the simplest tool (e.g., `project_operations.py`) and add:

```python
from servers.services.orchestrator_integration import ensure_indexer_for_project

async def execute(arguments: dict):
    # Add this line at the start
    indexer_info = await ensure_indexer_for_project()

    # Use indexer_info['port'] instead of hardcoded port
    # ... rest of tool logic ...
```

### 3. Gradual Rollout
- Week 1: Test with one project, feature flag OFF by default
- Week 2: Enable for your main project only
- Week 3: Update remaining MCP tools
- Week 4: Enable globally if stable

## Configuration Options

All configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_MULTI_PROJECT_MODE` | `false` | Enable multi-project mode |
| `MCP_IDLE_THRESHOLD` | `3600` | Seconds before container stops (1hr) |
| `MCP_JANITOR_INTERVAL` | `300` | Cleanup run frequency (5min) |
| `MCP_MAX_PROJECTS` | `10` | Max concurrent project containers |
| `MCP_MEMORY_LIMIT` | `2g` | Memory limit per container |
| `MCP_CPU_SHARES` | `512` | CPU shares per container |

## Why This Approach (L9 Principles)

✅ **Small Steps**: Feature flag allows incremental testing
✅ **Reversible**: One env var to rollback instantly
✅ **No Breaking Changes**: Legacy mode is untouched
✅ **E2E Thinking**: Handles full lifecycle (create→use→cleanup)
✅ **95% Gate**: Comprehensive tests ensure reliability
✅ **Truth > Comfort**: Gemini correctly identified K8s as overkill

## Gemini's Key Insights

1. **"Kubernetes is over-engineering at this stage"** - Correct assessment
2. **"Enhanced MCP is the clear winner"** - Builds on existing strengths
3. **"Heartbeat/janitor model"** - Elegant solution for lifecycle
4. **"Feature flag is our safety net"** - Perfect L9 approach
5. **"Centralized orchestration logic"** - Avoids config drift

## Current Status

- ✅ Core orchestrator implemented
- ✅ Redis state registry working
- ✅ Container lifecycle management ready
- ✅ Tests passing
- ✅ Integration helper created
- ⏳ MCP tools not yet updated (safe - using legacy mode)
- ⏳ Not deployed to global MCP (safe - local testing only)

## Risk Assessment

**Risk Level: MINIMAL**

- Feature is completely isolated behind flag
- Zero changes to existing code paths
- All new code is additive, not modifying
- Comprehensive test coverage
- Instant rollback capability

## Recommended Action

1. **Review the implementation** - All code is non-invasive
2. **Run tests locally** - Verify orchestrator works
3. **Try with ONE project** - Set flag, test in isolation
4. **Gradually expand** - Only after confidence builds

The implementation follows Gemini's L9 engineering recommendation exactly:
- Preserves current functionality completely
- Adds multi-project support gradually
- Uses `$CLAUDE_PROJECT_DIR` effectively
- Handles container lifecycle elegantly
- Can be rolled back instantly

**Confidence: 95%** - Implementation complete, safe, and ready for gradual testing.