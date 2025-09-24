# Deployment Safeguards Documentation

## ADR-0096: Robust Deployment with Contract Validation

**Status:** Implemented
**Date:** September 24, 2025
**Author:** L9 Engineering Team

## Overview

This document describes the comprehensive safeguards implemented to prevent deployment of broken code and ensure system contracts are maintained across all components.

## Key Principles

1. **Never Break Contracts** - Schema contracts are enforced at every level
2. **Fail Fast** - Problems detected early in deployment process
3. **No Bypass** - Critical tests cannot be skipped or overridden
4. **Automated Validation** - Machines verify correctness, not humans
5. **Rollback Ready** - Every deployment creates a backup for quick recovery

## Three-Phase Validation Process

### Phase 1: Pre-Deployment Validation

**File:** `neural-tools/tests/pre_deployment_validation.py`

Runs BEFORE any files are copied:

1. **Schema Contract Validation**
   - ChunkSchema consistency
   - Vector embedding dimensions (768)
   - No DateTime objects in JSON
   - File paths always present

2. **Vector Search Validation**
   - Literal index names (not parameterized)
   - Project isolation filters
   - Proper Neo4j query format

3. **MCP Tool Validation**
   - Fast search returns valid JSON
   - Elite search includes graph context
   - Error responses properly formatted

4. **Data Consistency Validation**
   - No nested Maps in Neo4j (ADR-0036)
   - ISO string timestamps
   - Project property always present (ADR-0029)

5. **Integration Validation**
   - Indexer uses ChunkSchema
   - Retrieval matches storage format
   - Cache serialization works

### Phase 2: Deployment Script Validation

**File:** `scripts/deploy-to-global-mcp.sh`

Built-in safeguards:

1. **Blocked Bypass Flags**
   ```bash
   --force, --skip-tests, --no-validation, --bypass, --yolo
   ```
   All result in immediate deployment failure

2. **Critical Test Requirements**
   - `test_indexer_mount_validation.py` (ADR-64)
   - `test_contract_validation.py` (ADR-96)
   - `pre_deployment_validation.py` (ADR-96)

3. **Timeout Protection**
   - 5-minute timeout for all tests
   - Prevents hanging deployments
   - Clear timeout reporting

4. **Backup Creation**
   - Automatic timestamped backup
   - Easy rollback command provided
   - Backup path in manifest

### Phase 3: Post-Deployment Smoke Tests

**File:** `neural-tools/tests/post_deployment_smoke_tests.py`

Runs AFTER deployment to verify:

1. **MCP Server Startup**
   - Server can be created
   - No import errors
   - Basic initialization works

2. **Search Functionality**
   - Fast search returns real files
   - Elite search includes graph context
   - Reasonable similarity scores

3. **Schema Validation**
   - ChunkSchema still works
   - JSON serialization succeeds
   - Neo4j dict conversion works

## Contract Specifications

### ChunkSchema Contract (ADR-0096)

```python
@dataclass
class ChunkSchema:
    chunk_id: str          # Format: "file_path:chunk:index"
    file_path: str         # Always present, never None
    content: str           # Actual content
    embedding: List[float] # Exactly 768 dimensions
    project: str           # Project isolation
    created_at: str        # ISO string, not DateTime
```

### Vector Search Contract

```cypher
# Literal index name, not parameterized
CALL db.index.vector.queryNodes('chunk_embeddings_index', $limit, $embedding)
YIELD node, score
WHERE node.project = $project AND score >= $min_score
```

### MCP Tool Response Contract

```json
{
  "status": "success|error|no_results",
  "query": "original query",
  "results": [...],
  "performance": {
    "query_time_ms": 123.45,
    "cache_hit": false
  }
}
```

## Regression Prevention

### Known Regressions Prevented

1. **Mount Validation Regression (ADR-64)**
   - Container reuse with wrong paths
   - Projects only indexing README files
   - Solution: Strict mount validation tests

2. **Vector Search Brittleness (ADR-96)**
   - Parameterized index names failing
   - Missing file_path in results
   - Solution: Schema contracts + literal queries

3. **DateTime Serialization (ADR-96)**
   - JSON/Redis cache failures
   - Solution: ISO strings only

4. **Project Isolation Failures (ADR-29)**
   - Cross-project data contamination
   - Solution: Mandatory project property

## Deployment Commands

### Standard Deployment (with all checks)
```bash
./scripts/deploy-to-global-mcp.sh
```

### Run Pre-Deployment Tests Only
```bash
cd neural-tools/tests
python3 pre_deployment_validation.py
```

### Run Contract Tests Only
```bash
cd neural-tools/tests
python3 test_contract_validation.py
```

### Run Post-Deployment Verification
```bash
cd neural-tools/tests
python3 post_deployment_smoke_tests.py
```

### Emergency Rollback
```bash
# Rollback to previous version
rm -rf ~/.claude/mcp-servers/neural-tools
mv ~/.claude/mcp-servers/neural-tools-backup-* ~/.claude/mcp-servers/neural-tools

# Verify rollback
cd ~/
# Restart Claude and test
```

## Monitoring and Alerts

### Success Criteria

1. **Pre-Deployment**: 100% tests must pass
2. **Deployment**: All critical files verified
3. **Post-Deployment**: No critical failures
4. **Runtime**: >95% success rate

### Failure Actions

1. **Pre-Deployment Failure**
   - Deployment blocked
   - Fix issues before retry
   - No bypass allowed

2. **Deployment Failure**
   - Automatic rollback
   - Backup restored
   - Error logged

3. **Post-Deployment Failure**
   - Warning issued
   - Consider rollback
   - Investigate issues

## Testing the Safeguards

### Verify Contract Enforcement
```python
# This should fail validation
from servers.services.chunk_schema import ChunkSchema

# Wrong embedding dimension
chunk = ChunkSchema(
    chunk_id="test.py:chunk:0",
    file_path="test.py",
    content="test",
    embedding=[0.1] * 500,  # Should be 768
    project="test"
)
# Raises ValueError
```

### Verify Deployment Blocking
```bash
# These should all fail
./scripts/deploy-to-global-mcp.sh --force
./scripts/deploy-to-global-mcp.sh --skip-tests
./scripts/deploy-to-global-mcp.sh --yolo
```

### Verify Search Contracts
```python
# Should return properly formatted results
from neural_mcp.tools.fast_search import execute
result = await execute({"query": "test"})
data = json.loads(result[0].text)
assert "status" in data
assert "file" in data["results"][0]
assert "path" in data["results"][0]["file"]
```

## Maintenance

### Adding New Contracts

1. Define contract in `chunk_schema.py`
2. Add validation to `test_contract_validation.py`
3. Add to pre-deployment checks
4. Update this documentation

### Updating Existing Contracts

1. Update schema with migration plan
2. Test backward compatibility
3. Update all validation tests
4. Deploy with extra monitoring

## Lessons Learned

1. **Contracts Prevent Brittleness** - Clear interfaces between components
2. **Validation Must Be Automatic** - Humans forget, machines don't
3. **No Bypass Options** - Remove temptation to skip tests
4. **Test the Tests** - Validation code needs validation too
5. **Document Everything** - Future developers need context

## References

- ADR-0096: Robust Vector Search Contract
- ADR-0064: Indexer Mount Validation
- ADR-0036: Neo4j Primitive Types Only
- ADR-0029: Project Isolation
- ADR-0060: Container Conflict Prevention

---

**Confidence:** 100% - Comprehensive safeguards implemented and tested