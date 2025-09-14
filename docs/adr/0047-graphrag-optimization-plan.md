# ADR-0047: GraphRAG Optimization Implementation Plan

**Status:** Proposed
**Date:** September 13, 2025
**Author:** L9 Engineering Team

## Context

Our L9 Neural GraphRAG system currently processes ~2,400 vectors from 269 files with functional but unoptimized performance. Deep analysis using Grok 4 and multi-model consensus identified five key optimization opportunities aligned with 2025 elite-tier RAG standards.

## Decision

Implement a phased optimization approach balancing developer experience improvements with user-facing enhancements.

## Implementation Phases

### Phase 1: Quick Wins (Weeks 1-2)

#### 1.1 Vector Quantization
**Implementation:**
```python
# Update qdrant_service.py
async def create_collection_with_quantization(self, collection_name: str):
    await self.client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True
            )
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=100  # Already optimized
        )
    )
```

**Expected Impact:**
- 4-8x memory reduction (768*4 bytes → 768 bytes per vector)
- 3-5x faster search operations
- <2% accuracy loss with proper quantile tuning

#### 1.2 Simple Timestamp-Based Incremental Indexing
**Implementation:**
```python
# Add to indexer_service.py
class IncrementalIndexer:
    def __init__(self):
        self.file_timestamps = {}  # path -> last_modified

    async def needs_reindex(self, file_path: Path) -> bool:
        current_mtime = file_path.stat().st_mtime
        last_indexed = self.file_timestamps.get(str(file_path), 0)
        return current_mtime > last_indexed

    async def update_file(self, file_path: Path):
        if await self.needs_reindex(file_path):
            # Remove old chunks
            await self.qdrant.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=str(file_path))
                        )]
                    )
                )
            )
            # Index new content
            await self.index_file(file_path)
            self.file_timestamps[str(file_path)] = file_path.stat().st_mtime
```

**Expected Impact:**
- 50-100x faster updates for single file changes
- Eliminates full re-indexing overhead
- Simple, reliable implementation

### Phase 2: Quality Improvements (Weeks 3-4)

#### 2.1 AST-Aware Semantic Chunking
**Implementation:**
```python
# Add to indexer_service.py
from tree_sitter import Parser, Language

class ASTChunker:
    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(Language('build/languages.so', 'python'))

    def chunk_by_ast(self, code: str) -> List[CodeChunk]:
        tree = self.parser.parse(bytes(code, 'utf8'))
        chunks = []

        # Extract functions and classes as semantic units
        query = '''
        [
            (function_definition) @func
            (class_definition) @class
        ]
        '''
        captures = self.parser.query(query).captures(tree.root_node)

        for node, _ in captures:
            chunk_text = code[node.start_byte:node.end_byte]
            chunks.append(CodeChunk(
                content=chunk_text,
                type=node.type,
                start_line=node.start_point[0],
                end_line=node.end_point[0]
            ))

        return chunks
```

**Expected Impact:**
- 2x better context preservation
- More coherent retrieval results
- Reduced chunk fragmentation

#### 2.2 Query Result Caching
**Implementation:**
```python
# Add to hybrid_retriever.py
class CachedRetriever:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour

    async def search_with_cache(self, query: str, **kwargs):
        cache_key = f"search:{hashlib.md5(f'{query}{kwargs}'.encode()).hexdigest()}"

        # Check cache
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Perform search
        results = await self.hybrid_search(query, **kwargs)

        # Cache results
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(results)
        )

        return results
```

**Expected Impact:**
- 10-100x faster repeated queries
- Reduced load on vector/graph databases
- Better user experience

### Phase 3: Foundation Building (Weeks 5-6)

#### 3.1 Hierarchical Graph Structure
**Implementation:**
```python
# Update neo4j_service.py
async def create_hierarchical_relationships(self):
    queries = [
        # Project -> Module hierarchy
        '''
        MATCH (p:Project {name: $project})
        MATCH (m:Module {project: $project})
        WHERE m.path STARTS WITH p.root_path
        MERGE (p)-[:CONTAINS]->(m)
        ''',

        # Module -> Class hierarchy
        '''
        MATCH (m:Module {project: $project})
        MATCH (c:Class {project: $project, file_path: m.path})
        MERGE (m)-[:DEFINES]->(c)
        ''',

        # Class -> Method hierarchy
        '''
        MATCH (c:Class {project: $project})
        MATCH (f:Function {project: $project, class_name: c.name})
        MERGE (c)-[:HAS_METHOD]->(f)
        '''
    ]

    for query in queries:
        await self.execute_query(query, {"project": self.project_name})
```

**Expected Impact:**
- 3-5x faster graph traversal
- Better structural understanding
- Enables advanced graph algorithms

#### 3.2 Batch Embedding Processing
**Implementation:**
```python
# Update nomic_service.py
class BatchEmbeddingProcessor:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.pending_queue = []

    async def process_batch(self, texts: List[str]) -> List[np.ndarray]:
        # Batch texts for efficient processing
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Send batch to Nomic service
            response = await self.http_client.post(
                f"{self.base_url}/embed/batch",
                json={"texts": batch}
            )

            batch_embeddings = response.json()["embeddings"]
            embeddings.extend(batch_embeddings)

        return embeddings
```

**Expected Impact:**
- 5-10x faster embedding generation
- Reduced API calls to Nomic service
- Better resource utilization

### Phase 4: Advanced Features (Weeks 7-8)

#### 4.1 Content-Addressed Storage with Hashing
**Implementation:**
```python
# Enhanced incremental indexing
class ContentAddressedIndexer:
    def __init__(self):
        self.content_hashes = {}  # path -> sha256

    async def compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def needs_reindex(self, file_path: Path) -> bool:
        current_hash = await self.compute_file_hash(file_path)
        stored_hash = self.content_hashes.get(str(file_path))
        return current_hash != stored_hash
```

**Expected Impact:**
- 100% accurate change detection
- No false re-indexing from timestamp changes
- Foundation for Merkle tree implementation

#### 4.2 Multi-Tier Result Ranking
**Implementation:**
```python
# Add to hybrid_retriever.py
class MultiTierRanker:
    async def rank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        # Tier 1: Exact matches (score > 0.95)
        # Tier 2: High relevance (score > 0.80)
        # Tier 3: Moderate relevance (score > 0.60)

        for result in results:
            # Apply PRISM scoring
            prism_score = await self.calculate_prism_score(result)

            # Apply recency boost
            recency_boost = self.calculate_recency_boost(result)

            # Apply authority boost (from .canon.yaml)
            authority_boost = self.get_authority_weight(result)

            # Combine scores
            result.final_score = (
                result.base_score * 0.5 +
                prism_score * 0.2 +
                recency_boost * 0.1 +
                authority_boost * 0.2
            )

        return sorted(results, key=lambda r: r.final_score, reverse=True)
```

**Expected Impact:**
- 30-50% better result relevance
- More intuitive ranking
- Better handling of edge cases

## Performance Targets

| Metric | Current | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|--------|---------|---------|---------|---------|---------|
| Query Latency (P50) | ~300ms | 150ms | 120ms | 100ms | 80ms |
| Query Latency (P95) | ~800ms | 400ms | 300ms | 250ms | 200ms |
| Index Update Time | 30s full | 1s/file | 1s/file | 0.5s/file | 0.3s/file |
| Memory Usage | 10MB | 2.5MB | 2.5MB | 3MB | 3MB |
| Retrieval Recall@10 | 0.65 | 0.65 | 0.75 | 0.80 | 0.85 |

## Risk Mitigation

1. **Backwards Compatibility**: All changes maintain API compatibility
2. **Feature Flags**: Each optimization behind feature toggle
3. **Rollback Plan**: Git tags at each phase completion
4. **Testing**: Comprehensive benchmarks before/after each phase
5. **Monitoring**: Grafana dashboards for all metrics

## Validation Strategy

1. **Baseline Metrics**: Capture current performance metrics
2. **A/B Testing**: Run old and new implementations in parallel
3. **User Feedback**: Collect qualitative feedback on retrieval quality
4. **Load Testing**: Validate at 10x current scale
5. **Regression Testing**: Ensure no functionality lost

## Success Criteria

- ✅ Query latency <100ms P50
- ✅ Update time <1s per file
- ✅ Memory usage reduced by 75%
- ✅ Retrieval recall improved by 30%
- ✅ Zero production incidents during rollout

## Alternative Approaches Considered

1. **HyDE (Hypothetical Document Embeddings)**: Deferred due to added latency
2. **Full Merkle Tree**: Overkill for current scale, timestamp-based sufficient
3. **ColBERT/SPLADE**: Too complex for immediate needs
4. **Migration to Pinecone/Weaviate**: Unnecessary given Qdrant capabilities

## Implementation Timeline

- **Week 1-2**: Phase 1 (Quick Wins)
- **Week 3-4**: Phase 2 (Quality)
- **Week 5-6**: Phase 3 (Foundation)
- **Week 7-8**: Phase 4 (Advanced)
- **Week 9**: Testing and validation
- **Week 10**: Production rollout

## References

- Grok 4 Analysis (September 13, 2025)
- Gemini-2.5-pro Consensus Analysis
- Microsoft GraphRAG Paper (2024)
- Qdrant Quantization Benchmarks
- Tree-sitter AST Parsing Documentation

## Decision

Proceed with phased implementation starting with vector quantization and timestamp-based incremental indexing, followed by quality improvements and foundation building.

**Confidence:** 95%
**Assumptions:** Current scale remains <10k vectors, read-heavy workload continues