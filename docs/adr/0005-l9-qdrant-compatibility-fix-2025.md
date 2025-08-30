# ADR-0005: L9 Qdrant Compatibility Fix - Vector Configuration Standardization

**Date**: 2025-08-30  
**Status**: Accepted  
**Deciders**: L9 Engineering Team  
**Technical Story**: Project indexing failure due to mixed Qdrant API usage

## Context

Our L9 Neural Tools Docker architecture experienced complete project indexing failure with 0 documents indexed in Qdrant collections and 0 relationships in Kuzu graph database, despite healthy container status.

### Problem Analysis

**Symptoms Observed**:
- `'VectorParams' object has no attribute 'get'` errors
- `1 validation error for PointStruct sparse_vector Extra inputs are not permitted` 
- Pydantic validation failures preventing any document indexing
- 0 semantic search capability despite 16+ hours of container uptime

**Root Cause Identified**:
Mixed Qdrant API usage in `neural-mcp-server-enhanced.py` where critical functions use deprecated sparse vector syntax while others use modern syntax.

**Evidence**:
- `memory_store_enhanced()` (line 406): Uses deprecated `sparse_vector={}` parameter
- `memory_search_enhanced()` (line 523): Uses deprecated `NamedSparseVector` approach  
- `project_auto_index()` (line 1368): Uses CORRECT `vector={"dense": ..., "sparse": ...}` syntax

## Decision

**Standardize ALL Qdrant vector operations to use modern API syntax** following the working pattern in `project_auto_index()`.

### Technical Implementation

**Fix 1: memory_store_enhanced() lines 406-407**
```python
# Before (deprecated):
vector={"dense": dense_embedding},
sparse_vector={"sparse": sparse_vector},

# After (modern):
vector={
    "dense": dense_embedding,
    "sparse": sparse_vector
},
```

**Fix 2: memory_search_enhanced() line 523**
Replace `NamedSparseVector` usage with unified vector structure.

**Fix 3: Collection creation consistency**
Ensure all collection configurations align with modern vector parameter expectations.

## Consequences

### Positive

**Performance Optimizations Maintained**:
- **gRPC Protocol**: 3-4x faster queries via `prefer_grpc=True`
- **INT8 Quantization**: 4x memory reduction with 99th percentile quantization
- **Kuzu GraphRAG**: 3-10x faster than Neo4j for relationship queries
- **Nomic v2-MoE**: 30-40% lower inference costs
- **RRF Hybrid Search**: State-of-the-art semantic + keyword fusion
- **MMR Diversity**: Prevents redundant results
- **PRISM Scoring**: Intelligent importance-based result boosting

**Deep Understanding Features Restored**:
- Full semantic search capability
- Code relationship mapping via GraphRAG
- Multi-language AST analysis (13+ languages)
- Intelligent change detection with deterministic IDs

**System Reliability**:
- Eliminates Pydantic validation failures
- Enables full project indexing
- Restores 9 MCP tools functionality
- Maintains L9 Docker isolation architecture

### Negative

**Minimal Risk**:
- Requires container rebuild to ensure clean state
- Brief downtime during fix deployment
- Need to re-index full project after fix

### Neutral

- API standardization improves maintainability
- Follows established L9 architecture principles
- Aligns with working `project_auto_index` implementation

## Implementation Plan

1. **Apply Systematic Fix**: Standardize vector configurations in `neural-mcp-server-enhanced.py`
2. **Rebuild Container**: Ensure clean state with compatible Qdrant setup
3. **Full Re-indexing**: Use corrected MCP tools to index complete project
4. **Validation**: Verify semantic search and GraphRAG functionality

## Alternatives Considered

1. **Qdrant Version Downgrade**: Rejected - loses performance optimizations
2. **Separate Collections**: Rejected - increases complexity without benefit
3. **Client Library Changes**: Rejected - container standardization preferred

## Related Decisions

- ADR-0003: L9 Hybrid Search Architecture Mandate
- ADR-0004: L9 Dual-Path Memory Architecture Token Optimization
- L9 Docker Connectivity Architecture

## Performance Metrics

**Target Outcomes**:
- Restore 100% project indexing capability
- Maintain sub-100ms semantic search latency
- Enable full GraphRAG relationship queries
- Support 9 MCP tools with optimal performance

**Success Criteria**:
- 0 Qdrant validation errors
- >0 documents indexed in collections  
- >0 relationships in Kuzu graph
- Functional semantic code search

---

**Confidence Level**: 95%+ based on systematic L9 diagnostic assessment  
**Impact**: HIGH - Restores core L9 neural architecture functionality  
**Urgency**: CRITICAL - Required for any semantic project understanding