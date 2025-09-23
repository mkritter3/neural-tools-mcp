# ADR-0073: MCP Tool Migration to Unified Neo4j Architecture

**Date: September 22, 2025**
**Status: PROPOSED**
**Supersedes: N/A**
**Related: ADR-0066 (Neo4j Vector Consolidation), ADR-0072 (Elite HNSW Performance)**

## Context

Following successful implementation of ADR-0072 (Neo4j Vector Index Elite HNSW Performance), our MCP tools architecture has a critical inconsistency that prevents users from accessing the elite GraphRAG capabilities we built.

### Current Architecture Problem

We currently have **two MCP servers** with incompatible architectures:

#### 1. **Primary Server** (`neural_server_stdio.py`)
- **Status**: Active (configured in `.mcp.json`)
- **Architecture**: Legacy dual-storage (Neo4j + Qdrant)
- **Search Implementation**: `semantic_code_search` still uses **Qdrant** for vector search
- **Problem**: Users cannot access ADR-0072's elite Neo4j vector performance

```python
# Line 1484: Still using Qdrant!
search_results = await container.qdrant.search_vectors(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=limit
)
```

#### 2. **Unified Server** (`unified_neural_server_stdio.py`)
- **Status**: Implemented but not deployed
- **Architecture**: ADR-0066/0072 compliant (unified Neo4j only)
- **Search Implementation**: Uses `UnifiedIndexerService` with elite HNSW vector search
- **Problem**: Not accessible to users

### Critical Impact

**User Experience**: Users invoking `semantic_code_search` get:
- ❌ **O(n) Qdrant linear search** instead of **O(log n) Neo4j HNSW**
- ❌ **Dual-write complexity** instead of **unified storage**
- ❌ **Legacy architecture** instead of **elite GraphRAG**

This defeats the entire purpose of ADR-0066 and ADR-0072 implementation.

## Decision

Migrate the primary MCP server to use unified Neo4j architecture, ensuring all tools leverage ADR-0072's elite HNSW performance while maintaining backward compatibility.

### Migration Strategy

#### Phase 1: Core Search Tool Migration

**Goal**: Update `semantic_code_search` to use ADR-0072 Neo4j vector search

**Implementation**:
```python
async def semantic_code_search_impl(query: str, limit: int) -> List[types.TextContent]:
    try:
        # ADR-0072: Use unified Neo4j vector search instead of Qdrant
        project_name, container, _ = await get_project_context({})

        # Initialize unified indexer service
        from servers.services.unified_graphiti_service import UnifiedIndexerService
        unified_indexer = UnifiedIndexerService(project_name)

        # ADR-0072: Elite hybrid search with Neo4j vector indexes
        search_results = await unified_indexer.search_unified_knowledge(
            query=query.strip(),
            limit=limit
        )

        # Format results for MCP compatibility
        if search_results.get("status") == "success":
            formatted_results = []
            for result in search_results.get("results", []):
                formatted_results.append({
                    "score": result.get("similarity_score", 0.0),
                    "file_path": result.get("file_path", ""),
                    "snippet": result.get("content", "")[:200] + ("..." if len(result.get("content", "")) > 200 else ""),
                    "search_method": result.get("search_method", "neo4j_elite_hybrid"),
                    "metadata": result.get("metadata", {})
                })

            response = {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "architecture": "neo4j_elite_hnsw",
                "performance_note": search_results.get("performance_note", "")
            }
        else:
            # Fallback to existing Qdrant behavior for compatibility
            response = await _legacy_qdrant_search(query, limit)
            response["fallback_reason"] = "neo4j_vector_search_failed"

        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

    except Exception as e:
        # Graceful fallback to existing implementation
        logger.warning(f"Elite search failed, falling back to Qdrant: {e}")
        return await _legacy_qdrant_search(query, limit)
```

#### Phase 2: GraphRAG Tool Enhancement

**Goal**: Ensure `graphrag_hybrid_search` leverages ADR-0072 performance

**Current Status**: Already uses retriever pattern - verify it utilizes Neo4j vector indexes

#### Phase 3: Service Container Update

**Goal**: Update `get_project_context()` to include unified indexer services

**Implementation**:
```python
async def get_project_context(arguments: Dict[str, Any]):
    """Enhanced project context with unified architecture support"""
    # ... existing project detection logic ...

    # ADR-0072: Add unified indexer to container
    if not hasattr(container, 'unified_indexer'):
        from servers.services.unified_graphiti_service import UnifiedIndexerService
        container.unified_indexer = UnifiedIndexerService(project_name)
        await container.unified_indexer.initialize()

    return project_name, container, retriever
```

### Backward Compatibility Strategy

#### 1. **Graceful Fallback**
- Wrap ADR-0072 calls in try/catch blocks
- Fall back to existing Qdrant implementation on failure
- Log fallback events for monitoring

#### 2. **Response Format Compatibility**
- Maintain existing MCP tool response schemas
- Add new fields (`architecture`, `search_method`, `performance_note`) for visibility
- Preserve `score`, `file_path`, `snippet` fields exactly

#### 3. **Progressive Enhancement**
- Phase rollout allows validation at each step
- Rollback capability to previous implementation
- Feature flags for experimental vs. stable behavior

### Migration Validation

#### 1. **Performance Validation**
```bash
# Test ADR-0072 integration
python3 test_mcp_unified_integration.py

# Validate search performance
python3 test_neo4j_vector_performance.py

# MCP tool compatibility
python3 tests/test_mcp_neural_tools.py
```

#### 2. **Compatibility Testing**
- All existing MCP tools continue to work
- Response formats remain compatible
- Error handling maintains graceful degradation

#### 3. **Performance Monitoring**
- Track query latency improvements
- Monitor vector search utilization
- Measure fallback frequency

## Consequences

### ✅ Positive

1. **Elite Performance Delivered**
   - Users get O(log n) HNSW vector search performance
   - <100ms P95 query latency capability (ADR-0071)
   - Eliminates dual-write complexity (ADR-0066)

2. **Unified Architecture Achieved**
   - Single Neo4j storage eliminates Qdrant dependency
   - Consistent search experience across all tools
   - Simplified service architecture

3. **Backward Compatibility Maintained**
   - Existing MCP integrations continue working
   - Graceful fallback for edge cases
   - Progressive enhancement approach

4. **Developer Experience Improved**
   - Clear migration path for additional tools
   - Better error messages and debugging
   - Simplified architecture reduces complexity

### ⚠️ Considerations

1. **Migration Complexity**
   - Multiple MCP tools require updates
   - Service container changes needed
   - Testing across different usage patterns

2. **Fallback Maintenance**
   - Dual code paths increase maintenance burden
   - Need monitoring to detect fallback usage
   - Eventually deprecate Qdrant dependencies

3. **Service Dependencies**
   - Requires Neo4j and Nomic services to be available
   - More complex initialization sequence
   - Additional error handling for service failures

## Implementation Plan

### Phase 1: Core Migration (Week 1)
- [ ] Update `semantic_code_search_impl()` to use unified Neo4j search
- [ ] Add graceful fallback to existing Qdrant implementation
- [ ] Implement response format compatibility layer
- [ ] Create migration validation tests

### Phase 2: Enhanced GraphRAG (Week 1)
- [ ] Verify `graphrag_hybrid_search_impl()` uses Neo4j vector indexes
- [ ] Update retriever pattern to leverage ADR-0072 performance
- [ ] Add performance metrics to responses

### Phase 3: Service Integration (Week 1)
- [ ] Update `get_project_context()` to include unified services
- [ ] Modify service container initialization
- [ ] Add health checks for unified architecture

### Phase 4: Validation & Deployment (Week 1)
- [ ] Comprehensive MCP tool testing
- [ ] Performance validation against ADR-0071 targets
- [ ] Documentation updates
- [ ] Production deployment

### Phase 5: Cleanup (Week 2)
- [ ] Monitor fallback usage patterns
- [ ] Remove Qdrant dependencies when stable
- [ ] Deprecate legacy unified server file
- [ ] Update global MCP configuration

## Validation Commands

```bash
# Test unified search integration
python3 -c "
import asyncio
from neural_mcp.neural_server_stdio import semantic_code_search_impl
result = asyncio.run(semantic_code_search_impl('vector search performance', 5))
print(result[0].text)
"

# Verify Neo4j vector usage
echo 'SHOW INDEXES WHERE type = \"VECTOR\";' | docker exec -i claude-l9-template-neo4j-1 cypher-shell -u neo4j -p graphrag-password

# MCP tool compatibility
python3 tests/test_mcp_neural_tools.py

# Performance benchmarking
python3 test_neo4j_vector_performance.py
```

## Migration Metrics

### Performance Targets
| Metric | Current (Qdrant) | Target (Neo4j HNSW) |
|--------|------------------|---------------------|
| Query Latency | ~200-500ms | <100ms P95 |
| Search Algorithm | Linear O(n) | HNSW O(log n) |
| Storage Systems | Neo4j + Qdrant | Neo4j only |
| Architecture | Dual-write | Unified |

### Success Criteria
- [ ] `semantic_code_search` uses Neo4j vector indexes
- [ ] Query performance meets ADR-0071 targets (<100ms P95)
- [ ] Zero breaking changes to MCP tool interfaces
- [ ] Fallback usage <5% of total queries
- [ ] All existing tests pass

## Future Enhancements

1. **Advanced Vector Search Features**
   - Multi-vector search capabilities
   - Vector search with filters
   - Semantic similarity clustering

2. **Performance Optimization**
   - Vector index tuning parameters
   - Query result caching strategies
   - Batch vector operations

3. **Monitoring & Observability**
   - Vector search performance metrics
   - Architecture usage analytics
   - Migration success tracking

## Conclusion

ADR-0073 ensures that users can access the elite GraphRAG performance delivered by ADR-0066 and ADR-0072. By migrating MCP tools to use unified Neo4j architecture, we eliminate the architectural inconsistency and deliver the full benefits of our elite HNSW implementation.

**Result**: Users get O(log n) vector search performance through familiar MCP tools, completing the vision of unified elite GraphRAG architecture. 🎯

---

*This ADR completes the integration between elite backend performance (ADR-0072) and user-facing tools, ensuring the architectural benefits reach end users.*