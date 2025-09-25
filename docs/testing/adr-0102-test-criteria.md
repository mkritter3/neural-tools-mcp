# ADR-0102 Testing Criteria & Exit Conditions

**Date:** 2025-09-24
**Status:** Test Plan
**Scope:** Unified Project Detection Architecture

## Exit Conditions (Must ALL Pass)

### 🔴 HARD STOP Conditions
If ANY of these fail, implementation CANNOT proceed:

1. **No Circular Dependencies**
   - Indexer starts without requiring existing container
   - Tools work without creating containers
   - Container creation doesn't depend on tools

2. **Zero Cross-Project Contamination**
   - Search in project A returns ZERO results from project B
   - Neo4j queries with project filter return correct data only
   - No data leakage between projects

3. **Exact String Match Consistency**
   - Project name identical across: Manager → Container → Neo4j → Tools
   - No normalization, no case changes, no transformations

## Testing Criteria

### Level 1: Unit Tests (Component Isolation)
**Exit: 100% pass rate required**

```python
# test_project_context_manager.py
✅ test_singleton_instance()           # Same instance across imports
✅ test_detection_priority_order()      # Env > User > Files > Container > Cache
✅ test_file_based_detection()          # Detects from package.json, pyproject.toml
✅ test_container_detection_secondary() # Container is confirmation, not primary
✅ test_explicit_failure_no_context()   # Returns None, not guessing
✅ test_project_name_validation()       # Rejects invalid characters
✅ test_thread_safety()                 # Concurrent access safe
```

```python
# test_naming_contract.py
✅ test_exact_string_preservation()     # "neural-novelist" → "neural-novelist"
✅ test_container_name_format()          # indexer-{project}-{timestamp}-{random}
✅ test_neo4j_property_format()          # Exact project string
✅ test_invalid_name_rejection()         # Special chars rejected
✅ test_empty_name_rejection()           # Empty/None rejected
```

### Level 2: Integration Tests (Component Interaction)
**Exit: 100% pass rate required**

```python
# test_integration_flow.py
✅ test_manager_to_indexer_flow()       # Manager → Indexer uses same project
✅ test_indexer_to_container_flow()     # Indexer → Container name correct
✅ test_container_to_neo4j_flow()       # Container → Neo4j properties match
✅ test_tools_use_manager_flow()        # Tools → Manager → Same project
✅ test_no_container_detection_primary() # Tools work without containers
```

### Level 3: System Tests (End-to-End)
**Exit: 95% pass rate minimum**

```bash
# test_e2e_project_flow.sh
✅ Test 1: Fresh Environment
   - No containers running
   - Set project context: "test-project-alpha"
   - Create indexer → Verify container name
   - Run elite_search → Verify project filter
   - Check Neo4j → Verify project property

✅ Test 2: Multiple Projects
   - Create 3 projects with different data
   - Switch between projects
   - Verify zero cross-contamination
   - Each search returns only its project data

✅ Test 3: Container Failure Recovery
   - Set project context
   - Start indexing (create container)
   - Kill container mid-process
   - Tools still work (use Manager, not container)
   - Restart indexer → Correct project resumed

✅ Test 4: Global MCP Mode
   - Deploy to ~/.claude/mcp-servers
   - Launch Claude from project-A directory
   - Verify detection works without local context
   - Switch to project-B directory
   - Verify project switch detection
```

### Level 4: Stress Tests (Reliability)
**Exit: 90% success rate under load**

```python
# test_stress_detection.py
✅ test_concurrent_project_detection()  # 100 parallel detections
✅ test_rapid_context_switching()       # Switch projects 1000 times
✅ test_memory_leak_detection()         # No memory growth over time
✅ test_container_storm()               # 50 containers, correct selection
```

### Level 5: Regression Tests (Backward Compatibility)
**Exit: Must not break existing functionality**

```python
# test_regression_suite.py
✅ test_existing_search_tools_work()    # elite_search, fast_search unchanged
✅ test_indexer_api_compatibility()     # Indexer endpoints still work
✅ test_neo4j_queries_compatible()      # Existing queries still function
✅ test_container_labels_preserved()    # Container discovery still works
```

## Validation Checklist

### Pre-Implementation
- [ ] ADR-0102 reviewed and approved
- [ ] No conflicts with existing ADRs
- [ ] Test environment prepared
- [ ] Rollback plan documented

### During Implementation
- [ ] Unit tests written BEFORE code
- [ ] Each component tested in isolation
- [ ] Integration points verified
- [ ] Logging added for debugging

### Post-Implementation
- [ ] All unit tests pass (100%)
- [ ] All integration tests pass (100%)
- [ ] System tests pass (≥95%)
- [ ] Stress tests pass (≥90%)
- [ ] No regression failures
- [ ] Performance metrics acceptable

## Performance Exit Criteria

| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| Project detection latency | <100ms | ___ | [ ] |
| Manager singleton init | <50ms | ___ | [ ] |
| Container detection time | <200ms | ___ | [ ] |
| Project switch time | <150ms | ___ | [ ] |
| Memory per project context | <10MB | ___ | [ ] |
| CPU during detection | <5% | ___ | [ ] |

## Error Handling Exit Criteria

| Scenario | Expected Behavior | Pass? |
|----------|------------------|-------|
| No project detected | Return explicit error message | [ ] |
| Invalid project name | Reject with clear message | [ ] |
| Container not found | Fall back to file detection | [ ] |
| Multiple containers | Select most recent | [ ] |
| Manager init failure | Graceful degradation | [ ] |
| Concurrent access | Thread-safe operations | [ ] |

## Security Exit Criteria

| Check | Requirement | Pass? |
|-------|------------|-------|
| Project name injection | Sanitized, no SQL/Cypher injection | [ ] |
| Path traversal | Paths validated and confined | [ ] |
| Container hijacking | Container ownership verified | [ ] |
| Cross-project access | Strict isolation enforced | [ ] |

## Monitoring & Observability

### Required Metrics
```python
# Prometheus metrics to track
project_detection_duration_seconds     # Histogram
project_detection_method               # Counter per method
project_detection_failures             # Counter
project_name_validation_errors         # Counter
container_detection_attempts           # Counter
manager_singleton_initializations      # Counter
```

### Required Logs
```python
# Structured logging required
INFO:  "Project detected: {name} via {method}"
WARN:  "Project detection failed, falling back to {method}"
ERROR: "Circular dependency detected in {component}"
DEBUG: "Container detection found {count} containers"
```

## Test Execution Plan

### Phase 1: Local Development (Day 1-2)
1. Run unit tests after each component change
2. Run integration tests after connecting components
3. Fix all failures before proceeding

### Phase 2: Local Full Stack (Day 3)
1. Start all services (Neo4j, Redis, Nomic)
2. Run system tests
3. Monitor logs for issues
4. Validate performance metrics

### Phase 3: Global MCP Testing (Day 4)
1. Deploy to ~/.claude/mcp-servers
2. Test from multiple project directories
3. Verify cross-project isolation
4. Run stress tests

### Phase 4: Production Validation (Day 5)
1. Deploy to production
2. Monitor error rates
3. Check performance metrics
4. Validate user workflows

## Go/No-Go Decision Matrix

| Criteria | Weight | Score (0-10) | Weighted | Min Required |
|----------|--------|--------------|----------|--------------|
| No circular deps | 30% | ___ | ___ | 10 |
| Zero contamination | 30% | ___ | ___ | 10 |
| String consistency | 20% | ___ | ___ | 10 |
| Performance | 10% | ___ | ___ | 7 |
| Error handling | 10% | ___ | ___ | 8 |
| **TOTAL** | 100% | ___ | ___ | **9.5** |

**EXIT DECISION: Proceed only if total ≥ 9.5**

## Rollback Triggers

Automatic rollback if ANY occur:
1. Cross-project data contamination detected
2. Circular dependency errors in logs
3. Project detection success rate <90%
4. Performance degradation >2x baseline
5. Memory leak detected (>100MB growth/hour)

## Sign-offs Required

- [ ] Engineering Lead: Architecture approved
- [ ] QA Lead: Test coverage sufficient
- [ ] Security: No vulnerabilities identified
- [ ] DevOps: Deployment plan validated
- [ ] Product: User impact acceptable

---

**Confidence: 100%** - Comprehensive testing criteria with clear exit conditions.