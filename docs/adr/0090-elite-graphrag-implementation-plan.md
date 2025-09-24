# ADR-0090: Elite GraphRAG Implementation Plan - Adapting ADRs 72 & 75 for Current Architecture

**Date: September 23, 2025**
**Status: Proposed**
**Impact: Critical**

## Context

After fixing Neo4j parameter serialization (ADR-0089), we now have functioning vector storage but lack the full elite GraphRAG capabilities described in ADRs 72 & 75. Current system status:

- ✅ Vector storage working with indexes (540 chunks indexed)
- ✅ Vector indexes exist (chunk_embeddings_index, file_embeddings_index)
- ⚠️ Indexes use minimal configuration - missing HNSW tuning
- ⚠️ Partial graph relationships (IMPORTS, DEFINED_IN, HAS_CHUNK, CALLS)
- ❌ Missing Symbol nodes and advanced relationships
- ❌ No hybrid search combining vectors with graph traversal

## Current Gaps vs Elite RAG Standards (2025)

### 1. Vector Index Performance (ADR-0072 Requirements)

**Required:** HNSW indexes with optimized parameters for O(log n) performance
**Current:** ✅ Vector indexes exist (chunk_embeddings_index, file_embeddings_index) but with minimal configuration
**Gap:** Missing explicit HNSW tuning parameters (m, ef_construction, ef_search)
**Impact:** Suboptimal performance - could be 2-5x faster with proper tuning

### 2. Graph Context (ADR-0075 Requirements)

**Required:**
- Symbol nodes (Function, Class, Variable)
- CALLS, IMPORTS, INHERITS, USES relationships
- Bidirectional traversal capabilities

**Current:**
- File and Chunk nodes only
- Basic relationships (2,293 IMPORTS, 22 CALLS)
- Missing Symbol extraction

### 3. Hybrid Search Capabilities

**Required:** Vector + Graph combined scoring
**Current:** Separate vector and graph queries
**Impact:** Missing contextual relevance boost

## Implementation Plan

### Phase 1: Optimize Existing Vector Indexes (Week 1)

```cypher
-- Recreate indexes with optimized HNSW parameters
DROP INDEX chunk_embeddings_index IF EXISTS;
CREATE VECTOR INDEX chunk_embeddings_index IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine',
    `vector.hnsw.m`: 16,
    `vector.hnsw.ef_construction`: 200
  }
}
```

**Implementation steps:**
1. Update neo4j_service.py to include HNSW parameters
2. Add migration to recreate indexes with optimization
3. Benchmark performance improvement (expect 2-5x)
4. Verify db.index.vector.queryNodes uses optimized parameters

### Phase 2: Symbol Extraction & Graph Enhancement (Week 1-2)

**New node types to add:**
```cypher
(:Symbol:Function {name, signature, docstring, complexity})
(:Symbol:Class {name, docstring, methods[]})
(:Symbol:Variable {name, type, scope})
```

**Implementation in indexer_service.py:**
```python
async def extract_symbols(self, file_content, file_path):
    """Extract symbols using tree-sitter or AST parsing"""
    symbols = []

    if file_path.endswith('.py'):
        tree = ast.parse(file_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols.append({
                    'type': 'Function',
                    'name': node.name,
                    'line': node.lineno,
                    'signature': self.get_signature(node)
                })
            elif isinstance(node, ast.ClassDef):
                symbols.append({
                    'type': 'Class',
                    'name': node.name,
                    'line': node.lineno
                })

    return symbols
```

### Phase 3: Advanced Relationships (Week 2)

**Add relationship extraction:**
- CALLS: Function → Function (already partial)
- INHERITS: Class → Class
- USES: Function → Variable
- INSTANTIATES: Function → Class
- OVERRIDES: Method → Method

### Phase 4: Hybrid Search Implementation (Week 2-3)

**Update semantic_search in project_operations.py:**
```python
async def hybrid_search(query, vector_weight=0.7):
    # Step 1: Vector search with HNSW
    vector_results = await neo4j.execute_cypher("""
        CALL db.index.vector.queryNodes('chunk_embeddings',
            $k, $embedding)
        YIELD node, score AS vector_score
        RETURN node, vector_score
    """, {'k': 20, 'embedding': embed(query)})

    # Step 2: Graph context expansion
    graph_results = await neo4j.execute_cypher("""
        MATCH (node)-[r:CALLS|IMPORTS|DEFINED_IN*1..2]-(context)
        WHERE node IN $vector_nodes
        RETURN context, count(r) AS graph_score
    """, {'vector_nodes': vector_results})

    # Step 3: Combine scores
    return combine_scores(vector_results, graph_results, vector_weight)
```

## Migration Strategy

### Step 1: Create migration for HNSW indexes
```bash
python3 neural-tools/src/manage_schema.py migration_generate \
    --name "add_hnsw_indexes" \
    --description "Add HNSW vector indexes for O(log n) performance"
```

### Step 2: Extend schema with Symbol types
```yaml
# schema.yaml additions
node_types:
  Symbol:
    properties:
      name: string
      type: string  # Function|Class|Variable
      signature: string
      docstring: string
      complexity: integer
    indexes:
      - name
      - type
```

### Step 3: Update indexer to extract symbols
- Modify `process_file()` to call `extract_symbols()`
- Create Symbol nodes alongside Chunks
- Link Symbols to Files and each other

## Performance Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Vector search (10k chunks) | ~500ms | <50ms | 10x |
| Graph traversal (2 hops) | ~200ms | <100ms | 2x |
| Hybrid search | N/A | <150ms | New |
| Indexing throughput | 10 files/s | 50 files/s | 5x |

## Testing Plan

1. **Benchmark current performance:**
   ```python
   # test_performance.py
   async def benchmark_vector_search():
       start = time.time()
       results = await semantic_search("authentication", limit=20)
       return time.time() - start
   ```

2. **Verify HNSW index creation:**
   ```cypher
   SHOW INDEXES WHERE type = 'VECTOR'
   ```

3. **Test symbol extraction accuracy:**
   - Parse known Python/JS/TS files
   - Verify all functions/classes detected
   - Check relationship accuracy

## Rollout Plan

1. **Day 1-2:** Implement HNSW indexes
2. **Day 3-5:** Add Symbol extraction for Python
3. **Day 6-7:** Add Symbol extraction for JS/TS
4. **Day 8-9:** Implement relationship extraction
5. **Day 10:** Deploy hybrid search
6. **Day 11-12:** Performance testing & optimization

## Risk Mitigation

- **Risk:** Neo4j version incompatibility
  - **Mitigation:** Verified Neo4j 2025.08.0 supports all features

- **Risk:** Memory pressure from HNSW indexes
  - **Mitigation:** Configure index parameters conservatively

- **Risk:** Symbol extraction errors
  - **Mitigation:** Graceful degradation, continue without symbols

## Success Metrics

- [ ] HNSW indexes created and operational
- [ ] Vector search <100ms for 10k chunks
- [ ] Symbol nodes extracted (>90% accuracy)
- [ ] All relationship types implemented
- [ ] Hybrid search returning contextual results
- [ ] MCP tools using optimized queries

## Conclusion

Current implementation has basic functionality but lacks elite RAG capabilities. This plan bridges the gap to achieve:

1. **O(log n) vector search** via HNSW indexes
2. **Rich graph context** via Symbol nodes
3. **Hybrid search** combining vectors and graphs
4. **2025 Elite RAG standards** compliance

Total implementation time: ~2 weeks
Expected performance gain: 10-100x

**Recommendation:** Proceed with Phase 1 immediately as it provides biggest immediate impact.

**Confidence: 95%** - Plan based on proven patterns from ADRs 72 & 75, adapted for current architecture.