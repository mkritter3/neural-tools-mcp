# ADR-0098: Testing Criteria & Exit Conditions

**Status:** Active
**Date:** September 2025
**Purpose:** Define clear success metrics and exit criteria for each phase

## Phase 0: Observability (✅ IMPLEMENTED)

### Testing Criteria
1. **Zero Functional Changes**
   - [ ] All existing tests still pass
   - [ ] No behavior changes in normal operation
   - [ ] Observability can be disabled without impact

2. **Divergence Detection**
   - [ ] Logs warning when Docker state differs from dicts
   - [ ] Correctly identifies orphaned containers
   - [ ] Detects port mismatches
   - [ ] Reports metrics accurately

3. **Performance Impact**
   - [ ] Observability adds < 10ms latency
   - [ ] No memory leaks from observer
   - [ ] Log volume reasonable (< 100 lines/hour)

### Exit Conditions for Phase 0 → Phase 1
✅ **MUST HAVE** (All required):
- Observability deployed for ≥ 1 week
- Divergence rate measured and documented
- No production issues caused by observability
- Metrics show < 5% divergence rate

⚠️ **SHOULD HAVE**:
- At least one real divergence caught and logged
- Performance metrics collected
- Team consensus on proceeding

❌ **STOP if**:
- Divergence rate > 20%
- Performance degradation detected
- Observability causing errors

### Test Script for Phase 0
```bash
#!/bin/bash
# test_phase0_observability.sh

echo "=== Phase 0: Observability Testing ==="

# Test 1: Verify observability doesn't break normal operation
echo "Test 1: Normal operation..."
python3 -c "
from neural_tools.servers.services.indexer_orchestrator import IndexerOrchestrator
orchestrator = IndexerOrchestrator()
# Should work normally with observability
"

# Test 2: Create divergence and check detection
echo "Test 2: Divergence detection..."
docker run -d --name test-orphan-container \
    --label com.l9.managed=true \
    --label com.l9.project=orphan-project \
    alpine sleep 3600

# Run a tool that triggers observability
python3 -c "
from neural_tools.servers.services.docker_observability import observer
observer.check_state_divergence('orphan-project', {}, 'test')
metrics = observer.report_metrics()
assert metrics['divergence_count'] > 0, 'Should detect orphaned container'
print(f'✅ Divergence detected: {metrics}')
"

# Cleanup
docker stop test-orphan-container
docker rm test-orphan-container

echo "=== Phase 0 Tests Complete ==="
```

---

## Phase 1: Enhanced Docker Labels

### Testing Criteria
1. **Label Presence**
   - [ ] All new containers have required labels
   - [ ] Labels include: project_hash, port, project_path
   - [ ] Labels survive container restart

2. **Backward Compatibility**
   - [ ] Old containers without labels still work
   - [ ] Mixed environment (old + new) functions correctly
   - [ ] No breaking changes to APIs

3. **Label Accuracy**
   - [ ] project_hash consistent for same path
   - [ ] port label matches actual port binding
   - [ ] Labels updated when state changes

### Exit Conditions for Phase 1 → Phase 2
✅ **MUST HAVE**:
- 100% of new containers have enhanced labels
- Zero breaking changes confirmed
- Labels verified accurate in 50+ containers
- Backward compatibility tested

⚠️ **SHOULD HAVE**:
- Automated label validation in CI
- Documentation updated
- Team training completed

❌ **STOP if**:
- Labels causing container creation failures
- Performance impact > 5%
- Label data corruption detected

### Test Script for Phase 1
```python
# test_phase1_labels.py
import docker
import hashlib
import os

client = docker.from_env()

def test_enhanced_labels():
    """Verify new containers have all required labels"""

    # Create test container with enhanced labels
    project_path = "/test/project"
    project_hash = hashlib.sha256(project_path.encode()).hexdigest()

    container = client.containers.run(
        image="alpine",
        command="sleep 3600",
        labels={
            'com.l9.managed': 'true',
            'com.l9.project': 'test-project',
            'com.l9.project_hash': project_hash,
            'com.l9.port': '48100',
            'com.l9.project_path': project_path
        },
        detach=True,
        name="test-phase1-labels"
    )

    # Verify labels
    assert container.labels['com.l9.project_hash'] == project_hash
    assert container.labels['com.l9.port'] == '48100'
    print("✅ Enhanced labels verified")

    # Cleanup
    container.stop()
    container.remove()

if __name__ == "__main__":
    test_enhanced_labels()
```

---

## Phase 2: Docker Primary, Dicts Fallback

### Testing Criteria
1. **Query Order**
   - [ ] Docker queried first for all lookups
   - [ ] Fallback to dicts only on Docker failure
   - [ ] Fallback usage logged and measured

2. **Consistency**
   - [ ] Docker and dict state match > 95% of time
   - [ ] Discrepancies logged with details
   - [ ] No data loss during transition

3. **Failure Handling**
   - [ ] Graceful handling of Docker API errors
   - [ ] Fallback works correctly
   - [ ] Recovery without manual intervention

### Exit Conditions for Phase 2 → Phase 3
✅ **MUST HAVE**:
- Docker queries succeed > 95% of time
- Fallback usage < 5% of requests
- Zero data loss or corruption
- 2 weeks stable operation

⚠️ **SHOULD HAVE**:
- Docker query performance acceptable
- Monitoring dashboards updated
- Runbook for troubleshooting

❌ **STOP if**:
- Docker query success rate < 90%
- Fallback causing inconsistencies
- Performance degradation > 10%

### Test Script for Phase 2
```python
# test_phase2_docker_primary.py
def test_docker_primary_lookup():
    """Verify Docker is queried first, dict is fallback"""

    # Mock scenario where Docker has container but dict doesn't
    docker_state = query_docker_for_container("project-a")
    dict_state = {}  # Empty dict simulates out-of-sync

    # System should use Docker state
    result = get_container_info("project-a")  # Should query Docker first

    assert result == docker_state
    assert "Using Docker state" in logs
    print("✅ Docker primary lookup working")

    # Test fallback when Docker fails
    with mock_docker_failure():
        result = get_container_info("project-b")
        assert "Docker query failed, using dict fallback" in logs
        print("✅ Fallback working correctly")
```

---

## Phase 3: Write-Through Cache

### Testing Criteria
1. **Cache Population**
   - [ ] Dicts populated from Docker on startup
   - [ ] Cache updates after Docker changes
   - [ ] No manual dict updates

2. **Single Source of Truth**
   - [ ] Docker state always authoritative
   - [ ] Zero divergence measured
   - [ ] Crash recovery works correctly

3. **Performance**
   - [ ] Startup time acceptable (< 5 seconds)
   - [ ] Cache hit rate > 90%
   - [ ] Memory usage reasonable

### Exit Conditions for Phase 3 → Phase 4
✅ **MUST HAVE**:
- Zero state divergence for 4 weeks
- Successful crash recovery tested 10+ times
- All state changes go through Docker first
- Performance metrics acceptable

⚠️ **SHOULD HAVE**:
- Automated divergence monitoring
- Cache invalidation strategy proven
- Team confidence high

❌ **STOP if**:
- Any state divergence detected
- Performance worse than Phase 2
- Crash recovery failures

### Test Script for Phase 3
```python
# test_phase3_cache.py
def test_write_through_cache():
    """Verify dicts are pure caches of Docker state"""

    # Start fresh - dicts should populate from Docker
    orchestrator = IndexerOrchestrator()
    await orchestrator.sync_state_from_docker()

    # Verify dict matches Docker exactly
    docker_containers = list_docker_containers()
    for project, info in orchestrator.active_indexers.items():
        docker_info = get_docker_container(project)
        assert info['container_id'] == docker_info['id']
        assert info['port'] == docker_info['port']

    print("✅ Cache correctly populated from Docker")

    # Test crash recovery
    orchestrator = None  # Simulate crash
    orchestrator = IndexerOrchestrator()
    await orchestrator.sync_state_from_docker()

    # Should recover state from Docker
    assert len(orchestrator.active_indexers) == len(docker_containers)
    print("✅ Crash recovery successful")
```

---

## Phase 4: Remove Redundant State (Optional)

### Testing Criteria
1. **Code Removal**
   - [ ] active_indexers dict removed
   - [ ] container_registry removed
   - [ ] JSON persistence removed
   - [ ] ~200 lines deleted

2. **Docker-Only Operation**
   - [ ] All queries go directly to Docker
   - [ ] No intermediate state stores
   - [ ] Labels contain all needed metadata

3. **Production Stability**
   - [ ] 4 weeks without issues
   - [ ] Performance acceptable
   - [ ] Team consensus achieved

### Exit Conditions for Phase 4 Completion
✅ **SUCCESS CRITERIA**:
- All tests passing
- Zero state-related bugs for 4 weeks
- Performance equal or better than Phase 3
- Code significantly simplified
- Team unanimously approves

⚠️ **ROLLBACK if**:
- Performance degradation > 15%
- Docker API reliability issues
- Team lacks confidence

---

## Master Test Suite

```bash
#!/bin/bash
# run_adr098_tests.sh

PHASE=${1:-0}

echo "=== Running ADR-0098 Phase $PHASE Tests ==="

case $PHASE in
    0)
        ./test_phase0_observability.sh
        python3 test_phase0_metrics.py
        ;;
    1)
        python3 test_phase1_labels.py
        ./test_backward_compatibility.sh
        ;;
    2)
        python3 test_phase2_docker_primary.py
        ./test_fallback_scenarios.sh
        ;;
    3)
        python3 test_phase3_cache.py
        ./test_crash_recovery.sh
        ;;
    4)
        python3 test_phase4_docker_only.py
        ./test_performance_benchmark.sh
        ;;
    all)
        for p in 0 1 2 3 4; do
            ./run_adr098_tests.sh $p
        done
        ;;
    *)
        echo "Usage: $0 [0|1|2|3|4|all]"
        exit 1
        ;;
esac

echo "=== Phase $PHASE Tests Complete ==="
```

---

## Go/No-Go Decision Matrix

| Phase | Min Duration | Success Rate | Rollback Time | Risk Level |
|-------|-------------|--------------|---------------|------------|
| 0→1 | 1 week | Observability working | Instant | **Zero** |
| 1→2 | 1 week | 100% labels present | Instant | **Low** |
| 2→3 | 2 weeks | >95% Docker success | <1 hour | **Low** |
| 3→4 | 4 weeks | Zero divergence | <1 day | **Medium** |
| 4 Complete | ∞ | Simplified system | Complex | **Committed** |

## Monitoring Dashboard Requirements

```yaml
Phase 0 Metrics:
  - divergence_rate: < 5%
  - observability_errors: 0
  - performance_impact: < 10ms

Phase 1 Metrics:
  - containers_with_labels: 100%
  - label_accuracy: 100%
  - backward_compat_failures: 0

Phase 2 Metrics:
  - docker_query_success: > 95%
  - fallback_usage: < 5%
  - query_latency_p99: < 50ms

Phase 3 Metrics:
  - state_divergence: 0
  - cache_hit_rate: > 90%
  - crash_recovery_success: 100%

Phase 4 Metrics:
  - code_lines_removed: > 200
  - docker_api_errors: < 0.1%
  - system_uptime: > 99.9%
```

## Rollback Procedures

### Phase 0 Rollback
```bash
# Remove observability imports
grep -r "docker_observability" --include="*.py" | \
  awk -F: '{print $1}' | xargs sed -i '/docker_observability/d'
```

### Phase 1 Rollback
```bash
# No rollback needed - labels are harmless
echo "Labels are backward compatible - no action needed"
```

### Phase 2 Rollback
```python
# Revert to dict-first lookup
DOCKER_PRIMARY_ENABLED = False  # Feature flag
```

### Phase 3 Rollback
```python
# Revert to independent state management
CACHE_MODE = "independent"  # Not write-through
```

### Phase 4 Rollback
```bash
# This requires git revert
git revert <phase4-commit>
# Restore dict-based state management
```

**Confidence: 95%** - Comprehensive testing and clear exit criteria ensure safe migration.