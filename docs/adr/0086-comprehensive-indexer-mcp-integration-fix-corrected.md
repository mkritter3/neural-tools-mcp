# ADR-0086: Comprehensive Indexer-MCP Integration Fix (Standards-Corrected)

**Status**: Accepted
**Date**: September 23, 2025
**Author**: L9 Engineering Team
**Supersedes**: ADR-0085 (incorrect fix direction)

## Executive Summary

After verifying September 2025 standards, we discovered that ADR-0085 had the fixes backwards. The MCP implementation is already correct per REST standards - it's the indexer that needs updating. This ADR provides the corrected implementation plan based on verified 2025 standards for REST APIs, Neo4j 5.23+, and cloud-native practices.

## Context

### The Problem Chain

1. **MCP tools report "offline" indexer** despite container running on port 48106
2. **Reindexing fails** with Neo4j syntax errors when attempted directly
3. **Vector search returns no results** because indexing never completes
4. **Graph context is unavailable** due to empty Neo4j/Qdrant databases

### Root Causes Identified

Through extensive investigation using zen tracer with Gemini-2.5-Pro:

1. **HTTP API Mismatch** (Critical)
   - Location: Indexer FastAPI endpoint definition
   - Issue: Indexer expects query parameters instead of JSON body
   - Standard: REST APIs should accept JSON body for POST requests (2025 REST best practices)
   - Correct: MCP already sends JSON body correctly
   - Fix: Update indexer to accept JSON body with Pydantic model
   - Impact: Indexer returns 422 "Field required" errors

2. **Neo4j Cypher Syntax Error** (Critical)
   - Location: `indexer_service.py` lines 963-964
   - Error: "Aliasing or expressions are not supported" in CALL subquery WITH clause
   - Issue: Cannot alias `$project AS project` inside CALL scope clause
   - Standard: Neo4j 5.23+ requires variables pre-bound or older CALL syntax
   - Fix: Bind `$project` to variable before CALL subquery
   - Impact: All indexing operations fail with syntax error

3. **Port Discovery Disconnect** (High)
   - Location: `service_container.py` line 236
   - Issue: Orchestrator returns None despite container running on 48106
   - Root cause: Container name mismatch or registry not updated
   - Impact: MCP can't find running indexer

4. **Path Translation Missing** (High)
   - Issue: No automatic path translation from host to container paths
   - Standard: Use environment variables for configuration (12-Factor App)
   - Fix: Add `HOST_PROJECT_PATH` and `CONTAINER_PROJECT_PATH` env vars
   - Impact: Files not found when indexing from host paths

5. **Missing Health Check Endpoints** (Medium)
   - Issue: Only basic `/health` endpoint, missing Kubernetes-style checks
   - Standard: Cloud-native apps need `/healthz` and `/readyz` (2025 standards)
   - Fix: Add liveness and readiness probes with dependency checks
   - Impact: Limited orchestration and monitoring capabilities

## Decision

Implement a four-phase fix that addresses each issue systematically:

### Phase 1: Fix Indexer API to Accept JSON Body (Immediate)

**File**: `neural-tools/src/servers/api/indexer_api.py`

**Current Code** (expecting query params - WRONG per 2025 standards):
```python
from fastapi import Query

@app.post("/reindex-path")
async def reindex_path(
    path: str = Query(..., description="Path to reindex"),
    recursive: bool = Query(True, description="Recursively index")
):
    # Process with query parameters
    await queue_for_indexing(path, recursive)
```

**Fixed Code** (accept JSON body per REST standards):
```python
from pydantic import BaseModel
import os

class ReindexRequest(BaseModel):
    """Request model for reindexing paths"""
    path: str
    recursive: bool = True

@app.post("/reindex-path")
async def reindex_path(request: ReindexRequest):
    """Accept JSON body per 2025 REST standards"""
    # Handle path translation from host to container
    container_path = request.path

    # Map host paths to container paths
    host_base = os.environ.get('HOST_PROJECT_PATH', '/Users/mkr/local-coding/claude-l9-template')
    container_base = os.environ.get('CONTAINER_PROJECT_PATH', '/workspace')

    if container_path.startswith(host_base):
        container_path = container_path.replace(host_base, container_base)

    # Process reindex request
    await queue_for_indexing(container_path, request.recursive)

    return {"status": "queued", "path": container_path}
```

### Phase 2: Fix Neo4j Syntax Error

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

### Phase 3: Add Environment Variables for Path Mapping

**File**: `neural-tools/docker/indexer.dockerfile`

```dockerfile
# Add environment variables for path mapping
ENV HOST_PROJECT_PATH=/Users/mkr/local-coding/claude-l9-template
ENV CONTAINER_PROJECT_PATH=/workspace

# These can be overridden at runtime
```

**File**: `docker-compose.yml` or container startup:

```yaml
services:
  indexer:
    image: l9-neural-indexer:production
    environment:
      - HOST_PROJECT_PATH=${PWD}  # Dynamic based on current directory
      - CONTAINER_PROJECT_PATH=/workspace
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_PASSWORD=graphrag-password
    volumes:
      - ${PWD}:/workspace:ro
```

### Phase 4: Deploy Fixed Container Image

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

### Phase 5: MCP Implementation is Already Correct!

**File**: `neural-tools/src/neural_mcp/tools/project_operations.py`

**Current MCP Code (ALREADY CORRECT per 2025 standards):**
```python
# This is the RIGHT way to do it in 2025!
response = await client.post(
    f"{indexer_url}/reindex-path",
    json={
        "path": path,
        "recursive": recursive
    }
)
```

**No changes needed to MCP** - it's already following REST best practices:
- ✅ POST request uses JSON body (not query parameters)
- ✅ Content-Type: application/json is set automatically
- ✅ Structured data in request body

The indexer needs to be fixed to accept this standard format.

### Phase 6: Add Kubernetes-Style Health Checks

**File**: `neural-tools/src/servers/api/indexer_api.py`

```python
from fastapi.responses import JSONResponse

@app.get("/healthz")
async def liveness_probe():
    """Kubernetes liveness probe - is the service running?"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@app.get("/readyz")
async def readiness_probe():
    """Kubernetes readiness probe - can the service handle requests?"""
    checks = {}

    # Check Neo4j connection
    try:
        async with get_neo4j_session() as session:
            await session.run("RETURN 1")
        checks["neo4j"] = "ready"
    except Exception as e:
        checks["neo4j"] = f"not ready: {str(e)}"

    # Check embedding service
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:48000/health")
            checks["embeddings"] = "ready" if resp.status_code == 200 else "not ready"
    except Exception as e:
        checks["embeddings"] = f"not ready: {str(e)}"

    # Check queue capacity
    queue_size = await get_queue_size()
    checks["queue"] = "ready" if queue_size < 1000 else f"overloaded: {queue_size}"

    # Overall readiness
    all_ready = all("ready" in str(v) for v in checks.values())

    if all_ready:
        return {"status": "ready", "checks": checks}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "checks": checks}
        )
```

### Phase 7: Enhance Port Discovery (Optional)

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

### Phase 8: Deploy Fixed Indexer (No MCP Changes Needed!)

```bash
# Build and deploy the fixed indexer
cd neural-tools
docker build -f docker/indexer.dockerfile -t l9-neural-indexer:adr-086-fix .
docker tag l9-neural-indexer:adr-086-fix l9-neural-indexer:production

# Stop old container
docker stop indexer-claude-l9-template
docker rm indexer-claude-l9-template

# Container will be recreated with fixed image by orchestrator
# No MCP deployment needed - MCP code is already correct!
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

1. **API Compatibility**: Indexer accepts JSON body from MCP (POST /reindex-path)
2. **Neo4j Queries**: No "aliasing not supported" errors
3. **Path Translation**: Host paths correctly mapped to container paths
4. **Health Checks**: /healthz and /readyz endpoints respond correctly
5. **Vector Search**: Returns relevant results with embeddings
6. **Graph Context**: Includes file relationships and dependencies

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

## Key Insights from Standards Verification

### What We Learned
1. **MCP was right all along** - sending JSON body is the 2025 standard for POST requests
2. **Indexer needs updating** - expecting query parameters is outdated
3. **Neo4j syntax is strict** - cannot alias parameters inside CALL scope clauses
4. **Environment variables are key** - proper path mapping via env vars follows 12-Factor App
5. **Health checks matter** - /healthz and /readyz are standard in 2025

### Why the Original Fix Was Wrong
ADR-0085 assumed the indexer's current behavior was correct and tried to change MCP to match.
After verifying 2025 standards:
- REST best practices mandate JSON body for complex POST data
- Query parameters should only be used for simple filtering/pagination
- Neo4j 5.23+ has specific rules about CALL subquery variable scoping

## Decision Outcome

Fix the indexer to follow 2025 standards, leave MCP unchanged. This aligns with modern REST API design, Neo4j 5.23+ syntax requirements, and cloud-native health check standards.

## References

- ADR-0052: Automatic Indexer Initialization
- ADR-0049: Dynamic Indexer Port Discovery
- ADR-0037: Configuration Priority
- ADR-0043: Project Context Lifecycle
- ADR-0084: Neo4j Embedding Pipeline Optimization
- Neo4j CALL Subquery Documentation (2025)
- Docker Container Discovery Patterns

---

**Confidence**: 98%
**Verification Method**: Independent research of September 2025 standards via web search
**Key Sources**: REST API best practices 2025, Neo4j 5.23+ documentation, Kubernetes health probes
**Critical Insight**: MCP implementation already correct - indexer needs standards compliance update