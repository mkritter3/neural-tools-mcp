# ADR-0065: Neural Indexing Pipeline Resilience and Error Handling

**Date:** September 21, 2025
**Status:** APPROVED - CRITICAL
**Deciders:** L9 Engineering Team
**Priority:** P0 (Blocking)

## Context

Deep analysis of the neural indexing pipeline revealed a critical architectural flaw causing silent data loss. The system reports "59 files processed" but creates 0 Neo4j nodes, resulting in complete search functionality failure without user awareness.

### Web Research Findings (September 2025 Standards)

Based on comprehensive research of Neo4j 5.22, Qdrant 1.15+, and Redis 7 standards:

- **Neo4j 5.22 Best Practices**: Mandatory error status validation, circuit breakers, property-based access control
- **Qdrant Vector DB**: GPU-accelerated indexing, binary quantization, multi-tenancy with proper error boundaries
- **Redis 7 Cluster**: Node timeout configuration, ACL permissions, authentication timeout handling
- **Modern Distributed Systems**: Circuit breaker patterns, observability metrics, graceful degradation

## Problem Statement

### Critical Issue: Dual-Pipeline Silent Failure Propagation

**Complete Root Cause Chain Analysis:**

The neural indexing system operates with dual storage pipelines:
1. **Semantic Pipeline**: File → Nomic Embeddings → Qdrant Vector Storage
2. **Graph Pipeline**: File → AST Analysis → Neo4j Graph Storage

**Both pipelines fail silently**, causing the critical "59 files processed, 0 nodes created" issue.

#### Pipeline 1: Semantic Storage Failures (Nomic → Qdrant)

**indexer_service.py:1012-1016** - Nomic embedding failures ignored:
```python
embeddings = await self.container.nomic.get_embeddings(texts)

if not embeddings:
    logger.warning(f"No embeddings generated for {file_path}")
    return  # ❌ SILENT FAILURE - Function exits but success=True still set
```

**indexer_service.py:1145** - Qdrant storage errors ignored:
```python
await self.container.qdrant.upsert_points(collection_name, points)
# ❌ NO STATUS CHECK - Qdrant returns {"status": "error"} but it's ignored
```

**qdrant_service.py:381-392** - Proper error structure returned but ignored:
```python
return {
    "status": "success",  # or "error"
    "operation_id": operation_info.operation_id,
    "points_count": len(points)
}
# vs error case:
return {
    "status": "error",
    "message": str(e)
}
```

#### Pipeline 2: Graph Storage Failures (Neo4j)

**indexer_service.py:1321** - Missing error validation:
```python
result = await self.container.neo4j.execute_cypher(cypher, params)
# ❌ NO STATUS CHECK - Errors are ignored
```

**indexer_service.py:1324** - Inadequate error handling:
```python
if result.get('status') == 'success' and result.get('result') and len(result['result']) > 0:
    # Only checks success path, ignores error status
```

**neo4j_service.py:129-140** - Proper error structure returned but ignored:
```python
except AuthError as auth_error:
    return {
        "status": "error",
        "message": f"Neo4j authentication failed: {str(auth_error)}",
        "error_type": "authentication_error"
    }
```

#### Success Metric Corruption

**indexer_service.py:484-494** - False success tracking:
```python
# Semantic indexing
if not self.degraded_mode['qdrant']:
    if not self.degraded_mode['nomic']:
        await self._index_semantic(file_path, relative_path, chunks)
        success = True  # ❌ Set even if _index_semantic fails silently

# Graph indexing
if not self.degraded_mode['neo4j']:
    await self._index_graph(file_path, relative_path, content)
    success = True  # ❌ Set even if _index_graph fails silently
```

**indexer_service.py:505** - Files counted as indexed despite storage failures:
```python
if success:  # ❌ success=True even when both storage pipelines fail
    self.metrics['files_indexed'] += 1
```

### Architectural Anti-Patterns Identified

1. **Silent Failure Tolerance**: System operates in "fail-open" mode without circuit breakers
2. **Missing Error Boundaries**: No bulkhead isolation between service failures
3. **Ineffective Degraded Mode**: Only set at startup, never during runtime failures
4. **Container Orchestration Conflicts**: Dual management systems causing port conflicts
5. **Poor Observability**: No error rate metrics or failure indicators

## Decision

Implement comprehensive neural indexing pipeline resilience with mandatory error validation, proper degraded mode activation, and September 2025 standard compliance.

## Solution Architecture

### 1. Mandatory Error Validation (CRITICAL - P0)

**All storage operations MUST validate status and return success/failure:**

#### 1.1 Semantic Pipeline Error Validation

```python
# BEFORE (Silent Failure in _index_semantic)
embeddings = await self.container.nomic.get_embeddings(texts)
if not embeddings:
    logger.warning(f"No embeddings generated for {file_path}")
    return  # ❌ Function exits, but parent still sets success=True

await self.container.qdrant.upsert_points(collection_name, points)
# ❌ No status check on Qdrant response

# AFTER (Mandatory Validation)
async def _index_semantic(self, file_path: str, relative_path: Path, chunks: List[Dict]) -> bool:
    try:
        # Step 1: Validate Nomic embedding generation
        embeddings = await self.container.nomic.get_embeddings(texts)
        if not embeddings:
            logger.error(f"Nomic embedding failed for {relative_path}")
            self._handle_service_failure('nomic', {"message": "No embeddings generated"})
            self.degraded_mode['nomic'] = True
            return False

        # Step 2: Validate Qdrant storage
        qdrant_result = await self.container.qdrant.upsert_points(collection_name, points)
        if qdrant_result.get('status') != 'success':
            logger.error(f"Qdrant storage failed for {relative_path}: {qdrant_result.get('message')}")
            self._handle_service_failure('qdrant', qdrant_result)
            self.degraded_mode['qdrant'] = True
            return False

        logger.info(f"✅ Semantic indexing successful: {len(points)} vectors stored")
        return True

    except Exception as e:
        logger.error(f"Semantic indexing failed for {file_path}: {e}")
        self.metrics['service_failures']['semantic_pipeline'] += 1
        return False
```

#### 1.2 Graph Pipeline Error Validation

```python
# BEFORE (Silent Failure in _index_graph)
result = await self.container.neo4j.execute_cypher(cypher, params)
# ❌ NO STATUS CHECK

# AFTER (Mandatory Validation)
async def _index_graph(self, file_path: str, relative_path: Path, content: str) -> bool:
    try:
        result = await self.container.neo4j.execute_cypher(cypher, params)
        if result.get('status') != 'success':
            logger.error(f"Neo4j operation failed for {relative_path}: {result.get('message')}")
            self._handle_service_failure('neo4j', result)
            self.degraded_mode['neo4j'] = True
            return False

        logger.info(f"✅ Graph indexing successful: {result.get('result', [])} nodes created")
        return True

    except Exception as e:
        logger.error(f"Graph indexing failed for {file_path}: {e}")
        self.metrics['service_failures']['neo4j'] += 1
        return False
```

#### 1.3 Corrected Success Tracking

```python
# BEFORE (False Success)
if not self.degraded_mode['qdrant']:
    if not self.degraded_mode['nomic']:
        await self._index_semantic(file_path, relative_path, chunks)
        success = True  # ❌ Always True

if not self.degraded_mode['neo4j']:
    await self._index_graph(file_path, relative_path, content)
    success = True  # ❌ Always True

# AFTER (Actual Success Validation)
semantic_success = False
graph_success = False

# Semantic indexing with actual result validation
if not self.degraded_mode['qdrant'] and not self.degraded_mode['nomic']:
    semantic_success = await self._index_semantic(file_path, relative_path, chunks)

# Graph indexing with actual result validation
if not self.degraded_mode['neo4j']:
    graph_success = await self._index_graph(file_path, relative_path, content)

# Success only if at least one pipeline succeeds
success = semantic_success or graph_success

# Track detailed metrics
if semantic_success:
    self.metrics['semantic_files_indexed'] += 1
if graph_success:
    self.metrics['graph_files_indexed'] += 1

# Overall success counter only incremented for actual storage success
if success:
    self.metrics['files_indexed'] += 1
else:
    self.metrics['files_failed'] += 1
    logger.error(f"❌ Complete indexing failure for {file_path}: semantic={semantic_success}, graph={graph_success}")
```

### 2. Runtime Degraded Mode Activation

**Immediate degraded mode on connection errors:**

```python
def _handle_service_failure(self, service_name: str, error_result: Dict):
    """Handle service failure with immediate degraded mode activation"""
    error_type = error_result.get('error_type', 'unknown')

    # Immediate degradation for connection-level errors
    if error_type in ['authentication_error', 'connection_error', 'service_unavailable']:
        self.degraded_mode[service_name] = True
        logger.warning(f"Service {service_name} entering degraded mode: {error_result.get('message')}")

    # Increment failure counter
    self.metrics['service_failures'][service_name] += 1

    # Schedule recovery attempt
    self._schedule_service_recovery(service_name)
```

### 3. Circuit Breaker Pattern Implementation

**Prevent cascade failures with timeout policies:**

```python
class ServiceCircuitBreaker:
    """Circuit breaker for external service calls"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open

    async def call(self, service_func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
            else:
                raise CircuitBreakerOpenError("Service circuit breaker is open")

        try:
            result = await service_func(*args, **kwargs)
            if result.get('status') == 'success':
                self.failure_count = 0
                self.state = 'closed'
            else:
                self._handle_failure()
            return result
        except Exception as e:
            self._handle_failure()
            raise
```

### 4. Container Orchestration Consolidation

**Single source of truth for container management:**

- **DEPRECATE**: `ContainerDiscoveryService` (simple naming scheme)
- **STANDARDIZE**: `IndexerOrchestrator` (robust with timestamps, labels, resource limits)
- **IMPLEMENT**: Proper port conflict detection and resolution

```python
class UnifiedContainerManager:
    """Single container management service replacing discovery/orchestrator duality"""

    async def ensure_indexer_container(self, project_name: str) -> Dict[str, Any]:
        # Check for existing containers by label (not name)
        existing = await self._find_by_labels({'com.l9.project': project_name})

        if existing and existing['status'] == 'running':
            return existing

        # Allocate free port with conflict detection
        port = await self._allocate_free_port(48100, 48199)

        # Create with unique name but discoverable labels
        container_name = f"indexer-{project_name}-{int(time.time())}-{secrets.token_hex(4)}"

        return await self._create_container(container_name, port, project_name)
```

### 5. Comprehensive Observability

**Error rate metrics and failure indicators:**

```python
class IndexingMetrics:
    """Centralized metrics for dual-pipeline indexing health"""

    def __init__(self):
        self.counters = {
            # Overall pipeline metrics
            'files_processed_total': 0,
            'files_indexed_successfully': 0,
            'files_failed_completely': 0,

            # Semantic pipeline (Nomic → Qdrant)
            'semantic_files_indexed': 0,
            'semantic_chunks_stored': 0,
            'nomic_embedding_failures': 0,
            'qdrant_storage_failures': 0,

            # Graph pipeline (Neo4j)
            'graph_files_indexed': 0,
            'graph_nodes_created': 0,
            'neo4j_operations_failed': 0,

            # Container/infrastructure
            'container_start_failures': 0,
            'service_degradations': 0
        }

        self.gauges = {
            'active_degraded_services': set(),
            'semantic_pipeline_health': 1.0,  # 0.0-1.0
            'graph_pipeline_health': 1.0,     # 0.0-1.0
            'processing_latency_p95': 0.0,
            'error_rate_5min': 0.0
        }

    def record_dual_pipeline_success(self, file_path: str, semantic_success: bool, graph_success: bool):
        """Record detailed dual-pipeline indexing results"""
        self.counters['files_processed_total'] += 1

        if semantic_success:
            self.counters['semantic_files_indexed'] += 1
            logger.info(f"✅ Semantic indexing successful: {file_path}")

        if graph_success:
            self.counters['graph_files_indexed'] += 1
            logger.info(f"✅ Graph indexing successful: {file_path}")

        if semantic_success or graph_success:
            self.counters['files_indexed_successfully'] += 1
            logger.info(f"✅ At least one pipeline successful: {file_path}")
        else:
            self.counters['files_failed_completely'] += 1
            logger.error(f"❌ Complete dual-pipeline failure: {file_path}")

    def record_pipeline_failure(self, pipeline: str, service: str, operation: str, error: str):
        """Record pipeline-specific failures with context"""
        counter_key = f'{service}_operations_failed' if service in ['neo4j'] else f'{service}_{operation}_failures'
        if counter_key in self.counters:
            self.counters[counter_key] += 1

        logger.error(f"❌ {pipeline} pipeline - {service} {operation} failed: {error}")

        # Update pipeline health metrics
        self._update_pipeline_health(pipeline, service)
        self._update_error_rate()

    def _update_pipeline_health(self, pipeline: str, failed_service: str):
        """Update pipeline health scores based on recent failures"""
        if pipeline == 'semantic':
            failure_rate = (self.counters['nomic_embedding_failures'] +
                          self.counters['qdrant_storage_failures']) / max(1, self.counters['files_processed_total'])
            self.gauges['semantic_pipeline_health'] = max(0.0, 1.0 - failure_rate)
        elif pipeline == 'graph':
            failure_rate = self.counters['neo4j_operations_failed'] / max(1, self.counters['files_processed_total'])
            self.gauges['graph_pipeline_health'] = max(0.0, 1.0 - failure_rate)
```

## Implementation Plan

### Phase 1: Critical Error Handling (Week 1)
1. **Dual-Pipeline Status Validation** - Add mandatory error validation for:
   - All `execute_cypher()` calls (Neo4j graph pipeline)
   - All `get_embeddings()` calls (Nomic semantic pipeline)
   - All `upsert_points()` calls (Qdrant vector storage)
2. **Function Return Value Fixes** - Make `_index_semantic()` and `_index_graph()` return boolean success/failure
3. **Success Metrics Correction** - Only increment `files_indexed` when actual storage succeeds
4. **Immediate degraded mode activation** for both pipelines
5. Fix container port conflict detection

### Phase 2: Resilience Patterns (Week 2)
1. Implement circuit breaker pattern for all external services
2. Add timeout policies and retry logic
3. Consolidate container management (deprecate discovery service)
4. Add health check endpoints for monitoring

### Phase 3: Observability (Week 3)
1. Implement centralized metrics collection
2. Add error rate and latency monitoring
3. Create alerting for degraded mode activation
4. Add dashboard for pipeline health visualization

## Testing Strategy

### Unit Tests
- Error handling path validation
- Circuit breaker state transitions
- Degraded mode activation triggers
- Metrics collection accuracy

### Integration Tests
- Service failure simulation and recovery
- Container lifecycle management
- End-to-end indexing pipeline validation
- Multi-project isolation verification

### Regression Tests
- Ensure existing functionality preserved
- Validate performance characteristics maintained
- Confirm no new silent failure modes introduced

## Monitoring and Alerting

### Key Metrics

#### Dual-Pipeline Health Metrics
- **Overall Success Rate**: (Semantic OR Graph successful) / Files processed (target: >95%)
- **Semantic Pipeline Success**: Successful Nomic→Qdrant operations / Attempts (target: >99%)
- **Graph Pipeline Success**: Successful Neo4j operations / Attempts (target: >99%)
- **Complete Failure Rate**: Both pipelines failed / Files processed (target: <1%)

#### Service-Level Metrics
- **Nomic Embedding Success**: Successful embeddings / Requests (target: >99%)
- **Qdrant Storage Success**: Successful vector storage / Requests (target: >99%)
- **Neo4j Graph Success**: Successful graph storage / Requests (target: >99%)
- **Recovery Time**: Time to recover from degraded mode (target: <60s)
- **Container Health**: Successful container starts / Total attempts (target: >95%)

### Alerts

#### P0 - Critical (Immediate Response)
- **Complete Pipeline Failure**: Both semantic AND graph pipelines failing (>1% failure rate)
- **Silent Failure Detection**: Files processed but zero storage in BOTH pipelines
- **Container Orchestration Collapse**: Unable to start indexer containers for >5 minutes

#### P1 - High (5-minute Response)
- **Single Pipeline Degraded**: Either semantic OR graph pipeline in degraded mode >5 minutes
- **Nomic Embedding Service Down**: Embedding generation failures >10% over 5 minutes
- **Qdrant Vector Storage Failing**: Vector storage failures >5% over 5 minutes
- **Neo4j Graph Storage Failing**: Graph storage failures >5% over 5 minutes

#### P2 - Medium (15-minute Response)
- **Elevated Error Rate**: Overall error rate exceeds 5% over 5-minute window
- **Degraded Mode Flip-Flopping**: Services entering/exiting degraded mode >3 times/hour
- **Cross-Reference Sync Issues**: Neo4j↔Qdrant ID synchronization failures

#### P3 - Low (1-hour Response)
- **Container Start Failures**: Container start failures exceed 2 per hour
- **Performance Degradation**: P95 latency >2x baseline for 15+ minutes

## Security Considerations

- **Authentication Failure Logging**: Proper audit trail for auth issues
- **Error Message Sanitization**: Prevent sensitive data leakage in logs
- **Service Isolation**: Circuit breakers prevent cascade authentication attacks
- **Privilege Separation**: Container processes run with minimal required permissions

## Performance Impact

### Expected Improvements
- **Reduced Resource Waste**: Stop processing files early when services fail
- **Faster Error Detection**: Immediate failure indication vs delayed discovery
- **Better Throughput**: Circuit breakers prevent cascade slowdowns
- **Improved UX**: Clear error messages instead of silent failures

### Overhead
- **CPU**: <2% additional overhead for error checking and metrics
- **Memory**: ~10MB for circuit breaker state and metrics storage
- **Network**: Minimal additional logging traffic
- **Storage**: Increased log volume for comprehensive error tracking

## Migration Strategy

### Backward Compatibility
- Maintain existing API contracts during transition
- Gradual rollout with feature flags for new error handling
- Parallel operation of old/new container management during deprecation

### Rollback Plan
- Feature flags allow instant disable of new error handling
- Container orchestrator fallback to discovery service if needed
- Comprehensive test suite validates rollback scenarios

## Success Criteria

### Dual-Pipeline Success Requirements

1. **Zero Silent Failures**: All indexing failures in BOTH pipelines must be detected and reported
   - No more "59 files processed, 0 nodes created" scenarios
   - Failed semantic indexing (Nomic→Qdrant) must be logged and tracked
   - Failed graph indexing (Neo4j) must be logged and tracked

2. **Pipeline Independence**: Either pipeline can fail without affecting the other
   - Semantic pipeline failures don't block graph indexing
   - Graph pipeline failures don't block semantic indexing
   - System remains functional with single-pipeline operation

3. **Accurate Success Metrics**: Files only counted as "indexed" when storage actually succeeds
   - `files_indexed` counter only incremented for actual storage success
   - Separate tracking: `semantic_files_indexed`, `graph_files_indexed`, `files_failed_completely`
   - Clear distinction between "processed" vs "successfully stored"

4. **<60s Recovery Time**: Services must recover from degraded mode within 1 minute
   - Automatic retry and circuit breaker recovery
   - Health check restoration for both pipelines

5. **>95% Overall Success Rate**: At least one pipeline succeeds for 95%+ of files
   - Individual pipeline targets: >99% success rate
   - Complete failure rate: <1% (both pipelines failing simultaneously)

6. **Complete Observability**: All failure modes visible in monitoring
   - Dual-pipeline health dashboards
   - Service-specific error rates and recovery times
   - Clear alerting for degraded mode activation

7. **Operational Confidence**: Clear error messages and recovery guidance
   - Specific pipeline failure identification
   - Actionable error messages for ops teams
   - Automated degraded mode recovery where possible

## Related ADRs

- **ADR-0060**: Graceful Ephemeral Containers (container orchestration)
- **ADR-0029**: Neo4j Project Isolation (database error handling)
- **ADR-0063**: Container Mount Validation (container lifecycle)
- **ADR-0044**: Container Discovery Service (deprecation target)

---

**Decision Status**: APPROVED
**Implementation Priority**: P0 (Blocking all neural tools deployments)
**Review Date**: October 1, 2025
**Success Metrics Review**: Weekly during implementation, monthly post-deployment