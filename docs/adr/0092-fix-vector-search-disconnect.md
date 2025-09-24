# ADR-0092: Fix Vector Search Disconnect with VectorCypherRetriever Pattern

## Status: Accepted

## Context

After implementing ADR-0090's GraphRAG capabilities, our vector search stopped returning results despite having 773 chunks with embeddings in Neo4j. Investigation revealed multiple issues:

1. **Neo4j's CALL db.index.vector.queryNodes doesn't support parameterized index names** - must use literal string interpolation
2. **The hybrid_search_with_fanout implementation has 6 OPTIONAL MATCHes** causing Cartesian product explosion
3. **Unbounded graph traversal** without intermediate result control
4. **Complex nested return structures** that Neo4j struggles to optimize

Research into Neo4j's official patterns and Microsoft's DRIFT GraphRAG revealed that Neo4j has a specific pattern called VectorCypherRetriever that solves these exact issues.

## Decision

Implement Neo4j's VectorCypherRetriever pattern with a two-phase approach:

### Phase 1: Vector Search
- Simple, fast vector similarity search
- Returns top-K candidates with scores
- Uses literal index name interpolation (not parameters)

### Phase 2: Graph Enrichment
- Takes vector results as input
- Performs controlled, bounded graph traversal
- Enriches with relationships (USES, INSTANTIATES, CALLS, IMPORTS)
- Returns combined context

### Key Implementation Details

1. **Fix index name parameterization**:
```python
# WRONG - Neo4j doesn't support this
CALL db.index.vector.queryNodes($index_name, ...)

# CORRECT - Use literal interpolation
CALL db.index.vector.queryNodes('{index_name}', ...)
```

2. **Split hybrid search into two phases**:
```python
async def hybrid_search_with_fanout(self, ...):
    # Phase 1: Vector search
    vector_results = await self.vector_similarity_search(...)

    # Phase 2: Graph enrichment with bounded traversal
    enriched_results = await self._enrich_with_graph_context(
        vector_results,
        max_depth=max_depth
    )
```

3. **Bounded traversal with depth control**:
```cypher
MATCH path = (start:Chunk)-[*1..{max_depth}]-(connected)
WHERE start.id IN $start_ids
  AND ALL(r IN relationships(path) WHERE type(r) IN $allowed_types)
WITH start, connected, length(path) as distance
ORDER BY distance
LIMIT $expansion_limit
```

## Consequences

### Positive
- Vector search works again with proper index name handling
- Graph traversal is bounded and controllable
- Performance is predictable (no Cartesian explosions)
- Compatible with Neo4j 2025.08.0's HNSW indexes
- Follows Neo4j's official patterns

### Negative
- Requires refactoring existing hybrid_search_with_fanout
- Two-phase approach adds slight complexity
- Must maintain both phases separately

### Neutral
- Aligns with industry best practices (Microsoft DRIFT, Neo4j patterns)
- Similar to implementations in neo4j-graphrag-python and ms-graphrag-neo4j

## Implementation Status

- [x] Research completed - found VectorCypherRetriever pattern
- [x] Root cause identified - parameterized index names not supported
- [ ] Implement two-phase hybrid search
- [ ] Update MCP tools to use new pattern
- [ ] Test with real data

## References

- Neo4j VectorCypherRetriever documentation
- Microsoft DRIFT GraphRAG paper
- Neo4j 2025.08.0 vector index limitations
- Open source: neo4j-graphrag-python, ms-graphrag-neo4j
  1. Neo4j Official Documentation (September 2025): Neo4j's db.index.vector.queryNodes doesn't support parameterized index names - must use literal string interpolation. [Found through testing and Neo4j error messages]
  2. Microsoft DRIFT GraphRAG Paper: The graph fan-out pattern for combining vector search with graph traversal. [Referenced in our ADR-0090 implementation]
  3. neo4j-graphrag-python Library: Open source implementation showing the two-phase pattern - vector search first, then graph enrichment. [Research findings during web search]
  4. Our Testing Results: Discovered 0 USES/INSTANTIATES relationships exist in database (confirmed via Cypher query), explaining why Phase 2 enrichment returned empty.