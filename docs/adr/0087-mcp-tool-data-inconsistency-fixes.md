# ADR-0087: MCP Tool Data Inconsistency Fixes

**Date: September 23, 2025**
**Status: Accepted**
**Authors: Claude (AI Assistant)**

## Context

After successfully indexing 3,436 files into Neo4j with complete graph relationships (2,293 IMPORTS, 592 DEFINED_IN, 22 CALLS), several MCP tools are failing to retrieve data despite the data being present and correctly structured in the database.

### Current State
- **Indexer**: Successfully processes files and creates graph nodes/relationships
- **Database**: Contains 3,439 File nodes, 557 Functions, 189 Modules, 35 Classes, 16 Chunks
- **Tools**: Multiple tools returning empty results or errors despite data availability

## Problem Analysis

### 1. Semantic Search Tool Issues

**Symptom**: Returns "no_results" despite having indexed content

**Root Cause**:
- The `semantic_search` function searches for `n.content`, `n.name`, or `n.description` fields
- File nodes don't have `content` or `description` fields
- Only Chunk nodes have `content` field (16 nodes vs 3,439 files)

**Query Issue** (neo4j_service.py:425-432):
```cypher
MATCH (n)
WHERE (n:File OR n:Class OR n:Method)
  AND (n.content CONTAINS $query      -- Files don't have content field
       OR n.name CONTAINS $query
       OR n.description CONTAINS $query) -- Files don't have description field
```

**Actual File Node Properties**:
```
["created_time", "language", "name", "path", "project", "content_hash",
 "file_type", "is_config", "size", "is_test", "indexed_time", "importance",
 "complexity_score", "is_docs", "updated_time"]
```

### 2. Dependency Analysis Tool Issues

**Symptom**: Returns empty dependencies despite 2,293 IMPORTS relationships existing

**Root Cause**: Path format mismatch
- Tool expects absolute paths: `/Users/mkr/local-coding/claude-l9-template/neural-tools/...`
- Database stores relative paths: `neural-tools/src/servers/services/service_container.py`

**Query Issue** (dependency_analysis.py:171):
```cypher
MATCH (target:File {path: $target_file, project: $project})
-- Exact match fails when paths don't match format
```

### 3. Project Understanding Tool Issues

**Symptom**: Returns error `'>=' not supported between instances of 'NoneType' and 'float'`

**Root Cause**: Missing or null values in aggregation/comparison operations

### 4. Vector Search Issues

**Symptom**: No embeddings being used for semantic search

**Root Cause**:
- Only 16 Chunk nodes have been created (vs 3,439 files)
- Vector indexes exist but are underutilized
- Semantic search falls back to text search instead of using embeddings

## Solution Design

### Fix 1: Update Semantic Search Query

```python
# neo4j_service.py - Update semantic_search method
async def semantic_search(self, query_text: str, limit: int = 10):
    """Perform semantic search with proper field matching"""

    # Search Chunks with content
    cypher = """
    MATCH (c:Chunk)
    WHERE c.content CONTAINS $query
    RETURN c, 'Chunk' as node_type, c.content as matched_content
    LIMIT $limit

    UNION

    MATCH (f:File)
    WHERE f.name CONTAINS $query
       OR f.path CONTAINS $query
    RETURN f, 'File' as node_type, f.path as matched_content
    LIMIT $limit

    UNION

    MATCH (func:Function)
    WHERE func.name CONTAINS $query
    RETURN func, 'Function' as node_type, func.name as matched_content
    LIMIT $limit
    """
```

### Fix 2: Normalize Path Handling

```python
# dependency_analysis.py - Add path normalization
async def _execute_dependency_analysis(neo4j_service, target_file: str, ...):
    # Normalize path to relative format
    if target_file.startswith('/'):
        # Convert absolute to relative
        base_path = '/Users/mkr/local-coding/claude-l9-template/'
        if target_file.startswith(base_path):
            target_file = target_file[len(base_path):]

    # Also try both exact and CONTAINS match
    analysis_query = """
    MATCH (target:File {project: $project})
    WHERE target.path = $target_file
       OR target.path CONTAINS $target_file
    """
```

### Fix 3: Add Null Handling

```python
# project_operations.py - Add null checks
def calculate_metrics(data):
    if data and isinstance(data, (int, float)):
        return data
    return 0  # Default value for null/missing data
```

### Fix 4: Enhance Chunk Creation

```python
# indexer_service.py - Create more chunks for better search
async def create_chunks(self, file_content: str, file_path: str):
    """Create semantic chunks from file content"""
    # Increase chunk creation for better coverage
    # Current: 16 chunks for 3,439 files (0.46% coverage)
    # Target: At least 1 chunk per significant file
```

## Implementation Plan

### Phase 1: Quick Fixes (Immediate)
1. Update semantic_search to use correct field names
2. Add path normalization to dependency_analysis
3. Add null checks to project_understanding

### Phase 2: Enhance Coverage (Short-term)
1. Increase chunk creation during indexing
2. Add content extraction for File nodes
3. Implement proper vector embeddings for all content

### Phase 3: Comprehensive Solution (Long-term)
1. Unified search interface combining text and vector search
2. Automatic path format detection and normalization
3. Robust error handling for all aggregation operations

## Metrics

### Current State
- Files with chunks: 16/3,439 (0.46%)
- Successful semantic searches: 0%
- Successful dependency analyses: 0%

### Target State
- Files with chunks: >80%
- Successful semantic searches: >90%
- Successful dependency analyses: >95%

## Trade-offs

### Pros
- Tools will actually return data that exists
- Better search coverage and accuracy
- More robust error handling

### Cons
- Increased indexing time for chunk creation
- More storage for content fields
- Additional CPU for path normalization

## Decision

Accept this ADR and implement fixes in three phases, starting with immediate query fixes that don't require re-indexing.

## Consequences

### Positive
- All MCP tools will function correctly
- Users can search and analyze their codebase effectively
- GraphRAG capabilities fully utilized

### Negative
- Need to re-index for full chunk coverage
- Slight performance impact from additional normalization

## References
- ADR-0085: Indexer MCP Integration
- ADR-0066: Elite GraphRAG with Neo4j HNSW
- ADR-0072: Unified Neo4j Storage