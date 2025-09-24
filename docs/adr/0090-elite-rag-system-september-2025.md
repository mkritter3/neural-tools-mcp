# ADR-0090: Elite RAG System Architecture for September 2025

**Status:** Proposed
**Date:** September 23, 2025
**Author:** Claude with Grok 4 collaboration
**Tags:** GraphRAG, HNSW, Vector Search, Code Embeddings, Hybrid Retrieval

## Context

Our current L9 Neural GraphRAG system has achieved baseline functionality with:
- Neo4j HNSW vector indexes (576 chunks with 768-dim embeddings)
- Basic GraphRAG with Function nodes (549) and CALLS relationships (22)
- Batch processing for chunk insertion (ADR-0089)
- MCP integration with semantic search

However, to achieve elite-tier RAG performance in September 2025, we need to implement state-of-the-art optimizations based on current research and industry best practices.

## Decision

Upgrade to an elite RAG system through incremental, non-breaking improvements in five key areas:

### 1. Vector Search Optimizations

#### HNSW Parameter Tuning
```cypher
-- Upgrade index configuration
CREATE VECTOR INDEX chunk_embeddings_index_v2 IF NOT EXISTS
FOR (n:Chunk) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 768,  -- Upgrade to 1024 when switching models
    `vector.similarity_function`: 'cosine',
    `vector.hnsw.m`: 24,  -- Increase from 16 for better accuracy
    `vector.hnsw.ef_construction`: 150,  -- Increase from 100
    `vector.quantization.enabled`: true,  -- Enable for 2x speed
    `vector.quantization.type`: 'int8'
  }
}
```

**Trade-offs:**
- +10% memory usage
- +5-10% recall improvement
- -15% indexing speed (mitigated by batch processing)

### 2. Graph Context Enhancements

#### Complete Dependency Analysis
```python
# Enhanced Tree-sitter extraction
class EnhancedTreeSitterExtractor:
    async def extract_imports(self, content: str, language: str):
        """Extract import statements for IMPORTS relationships"""
        tree = self.parser.parse(content.encode())
        imports = []

        # Query for import nodes based on language
        import_query = self.language_queries[language]['imports']
        captures = import_query.captures(tree.root_node)

        for node, _ in captures:
            imports.append({
                'module': node.text.decode(),
                'line': node.start_point[0],
                'type': 'import'
            })

        return imports

    async def extract_ast_boundaries(self, content: str, language: str):
        """Extract semantic boundaries for AST-aware chunking"""
        boundaries = []
        # Extract function/class boundaries
        for node_type in ['function_definition', 'class_definition']:
            nodes = self.query_nodes(content, node_type, language)
            for node in nodes:
                boundaries.append({
                    'start': node.start_byte,
                    'end': node.end_byte,
                    'type': node_type,
                    'name': self.get_node_name(node)
                })
        return boundaries
```

#### Community Summaries (LazyGraphRAG approach)
```cypher
-- Generate function communities using Leiden algorithm
CALL gds.leiden.write({
  nodeProjection: 'Function',
  relationshipProjection: {
    CALLS: {orientation: 'UNDIRECTED'}
  },
  writeProperty: 'community_id'
})

-- Create community summaries
MATCH (f:Function)
WITH f.community_id as community, collect(f) as functions
CREATE (c:Community {
  id: community,
  size: size(functions),
  summary: functions[0].name + ' and ' + (size(functions) - 1) + ' related functions',
  importance: avg([f IN functions | f.complexity_score])
})
```

### 3. Hybrid Retrieval Strategies

#### DRIFT-Inspired Fan-Out Search
```python
async def hybrid_search_with_fanout(
    self,
    query_embedding: List[float],
    max_depth: int = 2,
    vector_weight: float = 0.7
) -> List[Dict]:
    """
    Hybrid search combining vector similarity with graph traversal
    """
    cypher = """
    // 1. Vector search for initial nodes
    CALL db.index.vector.queryNodes('chunk_embeddings_index', $k, $query_embedding)
    YIELD node as chunk, score as vector_score

    // 2. Fan-out to related entities
    MATCH (chunk)-[:BELONGS_TO]->(f:File)
    OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
    OPTIONAL MATCH (f)<-[:IMPORTS]-(importing:File)
    OPTIONAL MATCH (f)-[:HAS_FUNCTION]->(func:Function)-[:CALLS*1..$depth]->(called:Function)

    // 3. Community context
    OPTIONAL MATCH (func)-[:IN_COMMUNITY]->(c:Community)

    // 4. Combine scores
    WITH chunk, f, vector_score,
         collect(DISTINCT imported) as imports,
         collect(DISTINCT importing) as importers,
         collect(DISTINCT called) as call_chain,
         c.summary as community_context

    RETURN chunk, f, vector_score,
           size(imports) + size(importers) as import_relevance,
           size(call_chain) as call_depth,
           community_context,
           $vector_weight * vector_score +
           (1 - $vector_weight) * (import_relevance + call_depth) / 10 as final_score
    ORDER BY final_score DESC
    LIMIT $limit
    """

    return await self.neo4j.execute_cypher(
        cypher,
        {
            'query_embedding': query_embedding,
            'k': 20,  # Get more candidates for reranking
            'depth': max_depth,
            'vector_weight': vector_weight,
            'limit': 10
        }
    )
```

#### AST-Aware Surgical Expansion
```python
def expand_to_ast_boundary(self, chunk: Dict, content: str) -> Dict:
    """
    Expand chunk to complete AST node (function/class)
    """
    start_line = chunk['start_line']
    end_line = chunk['end_line']

    # Find containing AST node
    for boundary in self.ast_boundaries:
        if boundary['start'] <= start_line <= boundary['end']:
            # Expand to full function/class
            return {
                **chunk,
                'expanded_content': content[boundary['start']:boundary['end']],
                'ast_type': boundary['type'],
                'ast_name': boundary['name']
            }

    return chunk
```

### 4. Embedding Model Upgrade Path

#### Phase 1: Upgrade to Nomic Embed Code 7B (Immediate)
- Drop-in replacement for Nomic Embed v2
- Same 768 dimensions, better code performance
- Open source, no API costs

#### Phase 2: Evaluate Voyage-code-3 (Q4 2025)
- 32K context (vs current 8K)
- Flexible dimensions (256-2048)
- Quantization support for speed
- Matryoshka embeddings for multi-resolution

```python
# Configuration for model switching
EMBEDDING_CONFIGS = {
    'nomic-embed-code-7b': {
        'dimensions': 768,
        'max_tokens': 8192,
        'batch_size': 32,
        'quantization': None
    },
    'voyage-code-3': {
        'dimensions': 1024,  # Start with 1024, test up to 2048
        'max_tokens': 32768,
        'batch_size': 16,
        'quantization': 'int8',  # 2x speed improvement
        'matryoshka_dims': [256, 512, 768, 1024]  # Multi-resolution
    }
}
```

### 5. Performance Optimizations

#### Intelligent Chunking Strategy
```python
class SemanticCodeChunker:
    def __init__(self, target_tokens: int = 512, overlap_ratio: float = 0.4):
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio

    def chunk_with_ast(self, content: str, ast_boundaries: List[Dict]) -> List[Dict]:
        """
        Create chunks that respect AST boundaries
        """
        chunks = []

        for boundary in ast_boundaries:
            node_content = content[boundary['start']:boundary['end']]
            node_tokens = self.count_tokens(node_content)

            if node_tokens <= self.target_tokens:
                # Small node: use as-is
                chunks.append({
                    'content': node_content,
                    'type': boundary['type'],
                    'name': boundary['name'],
                    'start_line': boundary['start_line'],
                    'end_line': boundary['end_line']
                })
            else:
                # Large node: split with overlap
                sub_chunks = self.split_with_overlap(
                    node_content,
                    self.target_tokens,
                    self.overlap_ratio
                )
                chunks.extend(sub_chunks)

        return chunks
```

#### Hybrid Reranking
```python
async def rerank_results(
    self,
    results: List[Dict],
    query: str
) -> List[Dict]:
    """
    Rerank using hybrid scoring
    """
    # Calculate PageRank for graph importance
    pagerank_scores = await self.calculate_pagerank()

    for result in results:
        vector_score = result['vector_score']

        # Graph-based features
        pagerank = pagerank_scores.get(result['node_id'], 0.0)
        import_count = result.get('import_relevance', 0)
        call_depth = result.get('call_depth', 0)
        community_importance = result.get('community_importance', 0.5)

        # Hybrid score with tunable weights
        result['final_score'] = (
            0.5 * vector_score +
            0.2 * pagerank +
            0.15 * (import_count / 10) +
            0.10 * (call_depth / 5) +
            0.05 * community_importance
        )

    return sorted(results, key=lambda x: x['final_score'], reverse=True)
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
1. ✅ Fix Tree-sitter initialization (actually instantiate it)
2. ✅ Implement import extraction for IMPORTS relationships
3. ✅ Fix language detection
4. ⬜ Upgrade HNSW parameters

### Phase 2: Graph Enhancement (Week 2)
1. ⬜ Implement community detection (Leiden algorithm)
2. ⬜ Add AST-aware chunking
3. ⬜ Create hybrid search with fan-out
4. ⬜ Add community summaries

### Phase 3: Model Upgrade (Week 3)
1. ⬜ Switch to Nomic Embed Code 7B
2. ⬜ Test quantization options
3. ⬜ Implement Matryoshka embeddings (optional)
4. ⬜ Benchmark performance

### Phase 4: Production Optimization (Week 4)
1. ⬜ Implement hybrid reranking
2. ⬜ Add context sufficiency metrics
3. ⬜ Optimize batch sizes
4. ⬜ Performance monitoring dashboard

## Monitoring & Success Metrics

### Key Performance Indicators
- **Recall@10**: Target >0.85 (current: ~0.60)
- **MRR (Mean Reciprocal Rank)**: Target >0.75
- **Query latency P95**: <500ms (current: ~500ms)
- **Index build time**: <100ms per file
- **Memory usage**: <4GB for 100K chunks

### Quality Metrics
- **Semantic similarity accuracy**: >90% on CodeSearchNet
- **Multi-hop reasoning success**: >80% on dependency queries
- **False positive rate**: <5%

## Risks and Mitigations

1. **Risk**: HNSW parameter changes require reindexing
   - **Mitigation**: Create new index alongside old, switch atomically

2. **Risk**: Model upgrade breaks existing embeddings
   - **Mitigation**: Dual embedding storage during transition

3. **Risk**: Increased latency from graph traversal
   - **Mitigation**: Redis caching + query optimization

4. **Risk**: Memory pressure from larger embeddings
   - **Mitigation**: Quantization + selective indexing

## Alternatives Considered

1. **TigerVector**: Better benchmarks but poor Neo4j integration
2. **GRAG Framework**: Too complex for incremental upgrade
3. **OpenAI Embeddings**: Expensive, closed-source
4. **Full GraphRAG**: 1000x cost of LazyGraphRAG approach

## References

- [LazyGraphRAG: Setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag)
- [DRIFT Search: Dynamic reasoning with community context](https://microsoft.github.io/graphrag/query/drift_search/)
- [Voyage-code-3: Next-gen code embeddings](https://blog.voyageai.com/2024/12/04/voyage-code-3/)
- [Nomic Embed Code: Open 7B code embedder](https://www.nomic.ai/blog/posts/nomic-embed-code)
- [HNSW Hyperparameter Guide](https://opensearch.org/blog/hnsw-hyperparameters/)

## Conclusion

This ADR outlines a practical path to elite-tier RAG performance through incremental improvements that preserve our existing infrastructure. By focusing on HNSW optimization, graph context enhancement, hybrid retrieval, and strategic model upgrades, we can achieve 90%+ recall on code queries while maintaining <500ms latency.

The LazyGraphRAG-inspired approach provides 99.9% cost reduction compared to full GraphRAG while maintaining quality, making this economically sustainable at scale.