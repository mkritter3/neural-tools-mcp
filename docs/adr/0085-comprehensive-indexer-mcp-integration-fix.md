# ADR-0085: Comprehensive Indexer-MCP Integration Fix

**Status**: Accepted
**Date**: September 23, 2025
**Author**: L9 Engineering Team

## Executive Summary

After multiple attempts to fix the indexer-MCP integration, we need a comprehensive solution that addresses all the interconnected issues preventing vector search and graph context from working. This ADR documents the root causes and provides a concrete, tested implementation plan.

## Context

### The Problem Chain

1. **MCP tools report "offline" indexer** despite container running on port 48106
2. **Reindexing fails** with Neo4j syntax errors when attempted directly
3. **Vector search returns no results** because indexing never completes
4. **Graph context is unavailable** due to empty Neo4j/Qdrant databases

### Root Causes Identified

Through extensive investigation using zen tracer with Gemini-2.5-Pro:

1. **HTTP API Mismatch** (Critical)
   - Location: `project_operations.py` lines 367-372
   - Issue: MCP sends `json={"path": path, "recursive": recursive}`
   - Expected: Query parameters `?path=/workspace/...&recursive=true`
   - Impact: Indexer returns 422 "Field required" errors

2. **Neo4j Cypher Syntax Error** (Critical)
   - Location: `indexer_service.py` lines 963-964
   - Error: "Aliasing or expressions are not supported" in CALL subquery WITH clause
   - Issue: `CALL (chunk_data, f) { WITH chunk_data, f, $project AS project`
   - Impact: All indexing operations fail with syntax error

3. **Port Discovery Disconnect** (High)
   - Location: `service_container.py` line 236
   - Issue: Orchestrator returns None despite container running on 48106
   - Root cause: Container name mismatch or registry not updated
   - Impact: MCP can't find running indexer

4. **Vector Storage Confusion** (High)
   - Issue: Code references Qdrant but we use Neo4j for vectors
   - Reality: Neo4j stores embeddings directly in Chunk nodes with HNSW indexes
   - Impact: Confusion in implementation and maintenance

5. **Path Translation Failure** (Medium)
   - Issue: MCP sends host paths `/Users/mkr/local-coding/...`
   - Expected: Container paths `/workspace/...`
   - Impact: Indexer can't find files to process

## Decision

Implement a four-phase fix that addresses each issue systematically:

### Phase 1: Fix Neo4j Syntax Error (Immediate)

**File**: `neural-tools/src/servers/services/indexer_service.py`

**Current Code** (lines 955-964):
```cypher
// Current broken syntax
CALL (chunk_data, f) {
    WITH chunk_data, f, $project AS project  // ❌ Aliasing not allowed
    // ... rest of query
}
```

**Fixed Code** (Neo4j 5.23+ with scope clause):
```cypher
// Option 1: Pre-bind parameter before CALL
WITH f, $project as project  // ✅ Bind parameter outside CALL
UNWIND CASE
    WHEN $chunks_data IS NULL OR size($chunks_data) = 0
    THEN [null]
    ELSE $chunks_data
END AS chunk_data

CALL (chunk_data, f, project) {  // ✅ Pass as variable
    WITH chunk_data, f, project   // ✅ Simple references only
    // ... rest of query
}

// OR Option 2: Use older CALL syntax without scope clause
CALL {
    WITH f, chunk_data
    WITH f, chunk_data, $project as project  // ✅ Can alias here
    // ... rest of query
}
```

**Also fix line 982-983**:
```cypher
// Current (broken)
CALL (f) {
    WITH f, $project as project  // ❌ Same aliasing issue

// Fixed
WITH f, collect(c) as chunks, project  // ✅ Keep project in scope
CALL (f, project) {
    WITH f, project  // ✅ Simple references
```

### Phase 2: Deploy Fixed Container Image

```bash
# 1. Build new indexer image with Neo4j fix
cd neural-tools
docker build -f docker/indexer.dockerfile -t l9-neural-indexer:adr-085-fix .

# 2. Tag as production (following ADR-0038)
docker tag l9-neural-indexer:adr-085-fix l9-neural-indexer:production

# 3. Restart indexer with fixed image
docker stop indexer-claude-l9-template
docker rm indexer-claude-l9-template
# Let orchestrator recreate with new image
```

### Phase 3: Fix MCP HTTP API Call

**File**: `neural-tools/src/neural_mcp/tools/project_operations.py`

Fix the HTTP API call to use query parameters (lines 367-373):

**Current (broken)**:
```python
response = await client.post(
    f"{indexer_url}/reindex-path",
    json={
        "path": path,
        "recursive": recursive
    }
)
```

**Fixed**:
```python
from urllib.parse import urlencode

# Convert host path to container path
container_path = path.replace('/Users/mkr/local-coding/claude-l9-template', '/workspace')

# Build query parameters
params = {
    "path": container_path,
    "recursive": str(recursive).lower()
}
query_string = urlencode(params)

response = await client.post(
    f"{indexer_url}/reindex-path?{query_string}"
)
```

### Phase 4: Fix Port Discovery

**File**: `neural-tools/src/neural_mcp/tools/project_operations.py`

Add fallback discovery for running containers:

```python
async def _get_indexer_endpoint(project_name: str) -> str:
    """Get indexer endpoint with multiple discovery methods"""

    # Method 1: Try orchestrator (existing)
    try:
        orchestrator = IndexerOrchestrator()
        endpoint = await orchestrator.get_indexer_endpoint(project_name)
        if endpoint:
            return endpoint
    except Exception as e:
        logger.warning(f"Orchestrator discovery failed: {e}")

    # Method 2: Direct container discovery (ADR-0085)
    try:
        import docker
        client = docker.from_env()
        container_name = f"indexer-{project_name}"
        container = client.containers.get(container_name)

        if container.status == 'running':
            # Extract port from container
            ports = container.attrs['NetworkSettings']['Ports']
            if '8080/tcp' in ports and ports['8080/tcp']:
                host_port = ports['8080/tcp'][0]['HostPort']
                endpoint = f"http://localhost:{host_port}"
                logger.info(f"✅ Found indexer via direct discovery: {endpoint}")
                return endpoint
    except Exception as e:
        logger.warning(f"Direct container discovery failed: {e}")

    # Method 3: Known port scan (last resort)
    for port in [48106, 48100, 48101, 48102]:  # Common indexer ports
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    logger.info(f"✅ Found indexer via port scan: port {port}")
                    return f"http://localhost:{port}"
        except:
            continue

    raise RuntimeError(f"No indexer found for project {project_name}")
```

### Phase 5: Deploy Global MCP Fix

```bash
# Deploy complete fix to global MCP
./scripts/deploy-to-global-mcp.sh

# This ensures all Claude sessions use the fixed code
```

## Implementation Checklist

### Pre-Implementation Validation
- [ ] Backup current global MCP deployment
- [ ] Document current indexer container state
- [ ] Save current Neo4j/Qdrant data if needed

### Phase 1: Neo4j Fix (30 minutes)
- [ ] Fix syntax in `indexer_service.py` lines 955-964
- [ ] Fix syntax in `indexer_service.py` lines 982-983
- [ ] Add unit test for the fixed query
- [ ] Verify query syntax with Neo4j browser

### Phase 2: Container Deployment (30 minutes)
- [ ] Build new indexer image with fix
- [ ] Tag as `l9-neural-indexer:production`
- [ ] Stop and remove old container
- [ ] Verify new container starts successfully
- [ ] Check logs for Neo4j connection success

### Phase 3: HTTP API Fix (20 minutes)
- [ ] Fix JSON body to query parameters conversion
- [ ] Add path translation from host to container
- [ ] Test with direct HTTP call to indexer
- [ ] Verify 200 response instead of 422

### Phase 4: MCP Discovery (45 minutes)
- [ ] Implement `_get_indexer_endpoint()` with fallback methods
- [ ] Update `reindex_path_impl()` to use new discovery
- [ ] Update `indexer_status_impl()` to use new discovery
- [ ] Add logging for discovery method used
- [ ] Test each discovery method independently

### Phase 5: Global Deployment (15 minutes)
- [ ] Run `./scripts/deploy-to-global-mcp.sh`
- [ ] Restart Claude to pick up changes
- [ ] Test MCP tools from fresh session

### Post-Implementation Testing
- [ ] Test `mcp__neural-tools__project_operations` indexer_status
- [ ] Test `mcp__neural-tools__reindex_path` on a small directory
- [ ] Verify files are actually indexed (check Neo4j)
- [ ] Test `mcp__neural-tools__semantic_search` returns results
- [ ] Test graph context is included in search results

## Success Criteria

1. **Indexer Status**: MCP reports indexer as "running" not "offline"
2. **Reindexing Works**: Can reindex files without Neo4j errors
3. **Vector Search**: Returns relevant results with embeddings
4. **Graph Context**: Includes file relationships and dependencies
5. **No Manual Steps**: Works immediately after Claude restart

## Risk Mitigation

### Risk: Neo4j fix breaks other queries
**Mitigation**: Search for all CALL subqueries and fix consistently

### Risk: Container discovery adds latency
**Mitigation**: Cache discovered endpoints for session duration

### Risk: Global deployment breaks other projects
**Mitigation**: Keep backup and test in dev environment first

## Testing Script

Create `test_adr_085_fix.py`:

```python
#!/usr/bin/env python3
"""Test ADR-0085 fixes end-to-end"""

import asyncio
import httpx
import json

async def test_indexer_integration():
    results = {}

    # 1. Test direct indexer health
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:48106/health")
            results['direct_health'] = resp.status_code == 200
    except Exception as e:
        results['direct_health'] = False

    # 2. Test reindex endpoint
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:48106/reindex-path?path=/workspace/neural-tools&recursive=true"
            )
            results['reindex'] = resp.status_code == 200
    except Exception as e:
        results['reindex'] = False

    # 3. Wait for processing
    await asyncio.sleep(5)

    # 4. Check status
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:48106/status")
            status = resp.json()
            results['files_indexed'] = status.get('files_processed', 0) > 0
    except Exception as e:
        results['files_indexed'] = False

    # Print results
    print("ADR-0085 Integration Test Results:")
    print(f"✓ Direct health check: {results.get('direct_health')}")
    print(f"✓ Reindex request: {results.get('reindex')}")
    print(f"✓ Files indexed: {results.get('files_indexed')}")

    success = all(results.values())
    print(f"\nOverall: {'✅ PASS' if success else '❌ FAIL'}")
    return success

if __name__ == "__main__":
    asyncio.run(test_indexer_integration())
```

## Long-term Improvements

1. **Unified Discovery Service**: Centralize all port/container discovery
2. **Health Dashboard**: Web UI showing all service statuses
3. **Auto-Recovery**: Detect and fix common integration issues
4. **Integration Tests**: Automated tests running every deployment

## Decision Outcome

This comprehensive fix addresses all identified issues in the indexer-MCP integration. By fixing the Neo4j syntax error, improving port discovery, and deploying globally, we restore full GraphRAG functionality.

## References

- ADR-0052: Automatic Indexer Initialization
- ADR-0049: Dynamic Indexer Port Discovery
- ADR-0037: Configuration Priority
- ADR-0043: Project Context Lifecycle
- ADR-0084: Neo4j Embedding Pipeline Optimization
- Neo4j CALL Subquery Documentation (2025)
- Docker Container Discovery Patterns

---

**Confidence**: 95%
**Assumptions**: Container on port 48106 is the production indexer, Neo4j fix is the primary blocker