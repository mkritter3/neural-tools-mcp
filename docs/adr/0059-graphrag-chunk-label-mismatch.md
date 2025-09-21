# ADR-0059: GraphRAG Chunk Label Consistency - Preventive Validation

**Date:** September 21, 2025
**Status:** Accepted
**Tags:** graphrag, neo4j, validation, monitoring, preventive

## Context

During investigation of why GraphRAG returns empty graph context, we discovered that while the label consistency between WriteSynchronizationManager and hybrid_retriever was correct (both use `Chunk`), the actual issue was that Chunk nodes had never been created in Neo4j due to data being indexed before WriteSynchronizationManager was fully implemented.

### Discovery Process

1. **User Report**: "Does the graph actually give you contextual awareness?"
2. **Initial Theory**: Suspected label mismatch between `Chunk` and `CodeChunk`
3. **Investigation**: Found hybrid_retriever already uses correct `Chunk` label
4. **Root Cause**: Neo4j had 0 Chunk nodes - data was indexed Sept 19 before ADR-0050/0053 fixes
5. **Resolution**: Backfilled 4,373 chunks from Qdrant to Neo4j using ADR-0050 recovery script

## Problem Statement

### What Actually Happened

```
Sept 15: ADR-0050 identified chunks weren't being created
    ↓
Sept 15-19: ADR-0053 implemented WriteSynchronizationManager fix
    ↓
Sept 19 18:12: Data indexed (before fix was fully deployed)
    ↓
Sept 21: Discovery that Neo4j had 0 Chunk nodes while Qdrant had 4,373
    ↓
Resolution: Backfilled chunks from Qdrant → Neo4j, GraphRAG restored
```

### Evidence of the Real Issue

**Before Backfill (Sept 21)**:
```bash
Neo4j: MATCH (c:Chunk {project: 'claude-l9-template'}) RETURN count(c)
Result: 0 chunks

Qdrant: project-claude-l9-template collection
Result: 4,373 points
```

**After Backfill**:
```bash
Neo4j: 7,363 Chunk nodes
File->HAS_CHUNK->Chunk: 6,749 relationships
Chunk->PART_OF->File: 6,749 relationships (created for hybrid_retriever)
```

### Why This Matters

1. **Data timing issues can break GraphRAG** - Indexing during partial deployment
2. **Silent degradation** - System returns results but without graph enrichment
3. **Need for sync validation** - Must verify Neo4j-Qdrant consistency
4. **Backfill capability critical** - ADR-0050 recovery script saved the system

## Decision

While the label consistency was already correct, implement comprehensive validation and monitoring to prevent future silent failures from data synchronization issues.

## Implementation

### 1. Completed - Data Backfill

Ran ADR-0050 recovery script to backfill missing Chunk nodes:
```bash
python3 neural-tools/scripts/backfill_chunks_to_neo4j.py \
  --project claude-l9-template \
  --batch-size 500 \
  --create-files
```

Result: Successfully migrated 4,373 chunks from Qdrant to Neo4j
```

### 2. Validation Tests - Prevent Future Failures

```python
# tests/test_graphrag_validation.py
"""
Critical validation tests for GraphRAG data consistency
Must be run in CI/CD to prevent silent failures
"""

async def test_chunk_label_consistency():
    """Verify chunks are queryable by hybrid retriever"""

    # Create a test chunk using WriteSynchronizationManager
    sync_manager = WriteSynchronizationManager(neo4j, qdrant, "test-project")
    success, chunk_id, _ = await sync_manager.write_chunk(
        content="test content",
        metadata={"file_path": "test.py"},
        vector=[0.1] * 768
    )
    assert success, "Failed to create test chunk"

    # Verify chunk exists with correct label
    result = await neo4j.execute_cypher("""
        MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
        RETURN c.chunk_id as id
    """, {"chunk_id": chunk_id, "project": "test-project"})

    assert result.get('records'), "Chunk node not found with Chunk label"

    # Verify hybrid retriever can find it
    retriever = HybridRetriever(container)
    context = await retriever._fetch_graph_context([chunk_id])
    assert context[0], "Hybrid retriever failed to fetch graph context"

async def test_neo4j_qdrant_count_match():
    """Verify chunk counts match between Neo4j and Qdrant"""

    # Get Qdrant count
    qdrant_count = await qdrant.count_points("project-test")

    # Get Neo4j count
    result = await neo4j.execute_cypher("""
        MATCH (c:Chunk {project: $project})
        RETURN count(c) as count
    """, {"project": "test-project"})
    neo4j_count = result['records'][0]['count']

    assert abs(qdrant_count - neo4j_count) <= 1, \
        f"Count mismatch: Qdrant={qdrant_count}, Neo4j={neo4j_count}"

async def test_graph_context_enrichment():
    """Verify GraphRAG actually returns graph context"""

    # Perform hybrid search
    retriever = HybridRetriever(container)
    results = await retriever.find_similar_with_context(
        query="test query",
        limit=5,
        include_graph_context=True
    )

    # At least one result should have non-empty graph context
    has_context = any(r.get('graph_context') for r in results)
    assert has_context, "No graph context returned - GraphRAG is broken!"
```

### 3. CI/CD Integration

```yaml
# .github/workflows/graphrag-validation.yml
name: GraphRAG Validation

on:
  push:
    paths:
      - 'neural-tools/src/servers/services/hybrid_retriever.py'
      - 'neural-tools/src/servers/services/sync_manager.py'
      - 'neural-tools/src/servers/services/indexer_service.py'
  pull_request:
  schedule:
    - cron: '0 0 * * *'  # Daily validation

jobs:
  validate-graphrag:
    runs-on: ubuntu-latest

    services:
      neo4j:
        image: neo4j:5.22.0
        env:
          NEO4J_AUTH: neo4j/graphrag-password
        ports:
          - 47687:7687

      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 46333:6333

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r neural-tools/requirements.txt
          pip install pytest pytest-asyncio

      - name: Run GraphRAG validation tests
        run: |
          cd neural-tools
          pytest tests/test_graphrag_validation.py -v --tb=short
        env:
          NEO4J_URI: bolt://localhost:47687
          NEO4J_PASSWORD: graphrag-password
          QDRANT_HOST: localhost
          QDRANT_PORT: 46333

      - name: Check chunk consistency
        run: |
          python scripts/check_chunk_consistency.py
        continue-on-error: false  # Fail the build on any mismatch
```

### 4. Monitoring Script

```python
# scripts/check_chunk_consistency.py
#!/usr/bin/env python3
"""
Check GraphRAG chunk consistency between Neo4j and Qdrant
Run this in CI/CD and periodically in production
"""

import sys
import asyncio
from servers.services.service_container import ServiceContainer
from servers.services.project_context_manager import ProjectContextManager

async def check_consistency():
    """Check for GraphRAG data consistency issues"""

    context = ProjectContextManager()
    await context.set_project('.')
    container = ServiceContainer(context)
    container.initialize()

    issues = []

    # 1. Check label consistency
    result = await container.neo4j.execute_cypher("""
        MATCH (c:CodeChunk) RETURN count(c) as count
    """)
    if result['records'][0]['count'] > 0:
        issues.append("Found CodeChunk nodes - should be Chunk!")

    # 2. Check chunk counts
    neo4j_chunks = await container.neo4j.execute_cypher("""
        MATCH (c:Chunk {project: $project})
        RETURN count(c) as count
    """, {"project": container.project_name})

    qdrant_info = await container.qdrant.get_collection_info(
        f"project-{container.project_name}"
    )

    neo4j_count = neo4j_chunks['records'][0]['count']
    qdrant_count = qdrant_info.points_count

    if abs(neo4j_count - qdrant_count) > 10:  # Allow small drift
        issues.append(
            f"Chunk count mismatch: Neo4j={neo4j_count}, Qdrant={qdrant_count}"
        )

    # 3. Test graph context retrieval
    sample = await container.neo4j.execute_cypher("""
        MATCH (c:Chunk {project: $project})
        RETURN c.chunk_id as chunk_id
        LIMIT 1
    """, {"project": container.project_name})

    if sample['records']:
        chunk_id = sample['records'][0]['chunk_id']
        from servers.services.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(container)
        context = await retriever._fetch_graph_context([chunk_id])

        if not context[0]:
            issues.append("Graph context fetch failed - hybrid search broken!")

    # Report results
    if issues:
        print("❌ GraphRAG Validation Failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("✅ GraphRAG validation passed")
        print(f"  - Neo4j chunks: {neo4j_count}")
        print(f"  - Qdrant chunks: {qdrant_count}")
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(check_consistency())
    sys.exit(exit_code)
```

## Consequences

### Positive
- GraphRAG will actually provide graph context as designed
- CI/CD will catch any future label mismatches
- Daily validation ensures consistency doesn't drift
- Clear error messages when issues detected

### Negative
- Need to re-index or migrate existing data with wrong labels
- Additional CI/CD time for validation tests

### Risks Mitigated
- Silent degradation to semantic-only search
- Wasted graph infrastructure
- False confidence in "working" GraphRAG

## Validation

After implementation:
1. GraphRAG returns non-empty `graph_context`
2. Hybrid search shows relationships, imports, and multi-hop traversal
3. CI/CD tests pass consistently
4. Monitoring script shows matching counts

## References

- ADR-0050: Identified missing Chunk nodes in Neo4j
- ADR-0053: WriteSynchronizationManager implementation (created Chunk nodes)
- Issue: GraphRAG returning empty graph context despite healthy services
- Investigation: September 21, 2025 discovery of label mismatch

## Status

**Accepted** - Critical fix required for GraphRAG functionality