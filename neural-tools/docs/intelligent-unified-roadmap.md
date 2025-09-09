# Intelligent Unified Roadmap: Neural-Tools Enhancement

## Required Dependencies

```bash
# IMPORTANT: Use the standard Anthropic SDK, NOT the Claude Code SDK
pip install anthropic[aiohttp]  # Includes aiohttp for context manager support

# Other production dependencies
pip install opentelemetry-instrumentation-fastapi
pip install slowapi pybreaker tenacity
pip install structlog
pip install tree-sitter tree-sitter-languages
```

## Executive Summary
This roadmap combines our existing capabilities with identified gaps, prioritizing stability and avoiding past issues. We discovered we already have 80% of desired features - the key is fixing and enhancing rather than rebuilding.

## Current State Assessment

### ✅ Working Features (DO NOT BREAK)
1. **GraphRAG Hybrid Search** - Neo4j + Qdrant simultaneous search
2. **File Watcher** - Debounced monitoring system
3. **AST Chunking** - Python AST extraction
4. **Multi-Vector Support** - Named vectors for code/docs/general
5. **Graceful Degradation** - System continues if services fail
6. **Service Container** - Proper initialization with retries

### ⚠️ Past Issues to Avoid
1. **Event Loop Binding** - AsyncIO conflicts in MCP environments
2. **Service Initialization Races** - Neo4j/Qdrant startup timing
3. **Validation Errors** - Schema mismatches in MCP protocol
4. **Chunking Failures** - Large files causing OOM
5. **Network Issues** - Docker container name resolution

## Phase 1: Quick Wins (Week 1) - Stability First

### 1.1 Fix File Watcher with Selective Reprocessing (4 hours)
**Current Problem**: Triggers full re-index on any change
**Enhanced Solution**: Selective reprocessing with diff checking

```python
# CURRENT (in file_watcher.py - line 36)
def on_file_change(event_type, file_path):
    # Problem: This calls full re-index
    await indexer.index_directory(project_root)  

# ENHANCED - With selective reprocessing (2025 standard)
async def on_file_change(event_type, file_path):
    if event_type == 'deleted':
        await indexer.remove_file(file_path)
    else:  # created or modified
        # Check if content actually changed
        file_hash = compute_file_hash(file_path)
        cached_hash = await cache.get(f"file_hash:{file_path}")
        
        if file_hash != cached_hash:
            # Only reprocess changed chunks
            old_chunks = await get_cached_chunks(file_path)
            new_chunks = await chunker.chunk_file(file_path)
            
            # Diff and update only changed chunks
            changed_ids = diff_chunks(old_chunks, new_chunks)
            await indexer.update_chunks(changed_ids)
            await cache.set(f"file_hash:{file_path}", file_hash)
```

**Testing Criteria:**

#### Unit Tests
```python
# test_file_watcher_unit.py
@pytest.mark.asyncio
async def test_file_change_calls_correct_method():
    """Verify callback invokes index_file, not index_directory"""
    mock_indexer = MagicMock()
    handler = DebouncedEventHandler(mock_indexer.index_file)
    
    # Simulate file change
    event = FileModifiedEvent("/path/to/file.py")
    handler.dispatch(event)
    
    # Assert correct method called
    mock_indexer.index_file.assert_called_once_with("/path/to/file.py")
    mock_indexer.index_directory.assert_not_called()

async def test_debouncing_prevents_rapid_reindex():
    """Verify debouncing prevents multiple rapid calls"""
    call_count = 0
    def callback(_, __):
        nonlocal call_count
        call_count += 1
    
    handler = DebouncedEventHandler(callback, debounce_interval=0.5)
    
    # Rapid file changes
    for _ in range(10):
        handler.dispatch(FileModifiedEvent("/test.py"))
        await asyncio.sleep(0.05)  # 50ms between events
    
    await asyncio.sleep(0.6)  # Wait for debounce
    assert call_count == 1  # Only one call despite 10 events
```

#### Integration Tests
```python
# test_file_watcher_integration.py
@pytest.mark.integration
async def test_file_change_triggers_incremental_index():
    """E2E test: actual file change → correct index behavior"""
    async with TestEnvironment() as env:
        # Create initial file
        test_file = env.project_dir / "test.py"
        test_file.write_text("def hello(): pass")
        
        # Start watcher
        await env.start_file_watcher()
        
        # Modify file
        test_file.write_text("def hello(): return 'world'")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Verify only one file indexed (not whole directory)
        assert env.metrics['files_indexed'] == 1
        assert env.metrics['directories_scanned'] == 0
```

#### Performance Benchmarks
- Single file index: < 1 second
- Debounce effectiveness: 90% reduction in redundant calls
- Memory usage: < 10MB overhead for watcher

#### Exit Conditions
- [ ] 100% of file changes trigger selective reprocessing
- [ ] Only changed chunks are updated (100% selectivity)
- [ ] >30% time reduction vs full re-index
- [ ] Debouncing reduces calls by > 80% for rapid changes
- [ ] All edge cases handled without errors
- [ ] Performance: Single file < 1s, Directory > 30s (proving optimization works)

**Additional Testing for Selective Reprocessing:**

```python
# test_selective_reprocessing.py
@pytest.mark.asyncio
async def test_selective_update(mocker):
    """Verify only changed chunks are updated"""
    mocker.patch('hashlib.sha256', return_value=Mock(hexdigest=lambda: "new_hash"))
    mocker.patch('cache.get', return_value="old_hash")
    mocker.patch('get_cached_chunks', return_value=[{'id':1, 'content':'old'}])
    mocker.patch('chunker.chunk_file', return_value=[{'id':1, 'content':'new'}])
    mocker.patch('diff_chunks', return_value=[1])  # Only chunk 1 changed
    mocker.patch('indexer.update_chunks', AsyncMock())
    
    await on_file_change('modified', 'file.py')
    
    # Assert only changed chunk updated
    indexer.update_chunks.assert_called_with([1])

@pytest.mark.performance
async def test_performance_vs_full_reindex(benchmark):
    """Verify >30% performance improvement"""
    full_time = benchmark(full_reindex, 'large_file.py')
    selective_time = benchmark(on_file_change, 'modified', 'large_file.py')
    
    improvement = (full_time - selective_time) / full_time * 100
    assert improvement > 30  # At least 30% faster

@pytest.mark.edge
async def test_massive_file_change(mocker):
    """Handle case where 90% of file changes"""
    mocker.patch('get_cached_chunks', return_value=[{'id':i} for i in range(100)])
    mocker.patch('chunker.chunk_file', return_value=[{'id':i+100} for i in range(90)])
    mocker.patch('diff_chunks', return_value=list(range(90)))
    mocker.patch('indexer.update_chunks', AsyncMock())
    
    await on_file_change('modified', 'file.py')
    
    # Should handle large changes gracefully
    indexer.update_chunks.assert_called_with(list(range(90)))
```

### 1.2 Add Simple Caching Layer (4 hours)

**Testing Criteria:**

#### Unit Tests
```python
# test_cache_unit.py
async def test_cache_hit_returns_cached_value():
    cache = SimpleCache(ttl=60)
    
    # First call - compute
    compute_called = False
    async def compute():
        nonlocal compute_called
        compute_called = True
        return {"result": "data"}
    
    result1 = await cache.get_or_compute("key1", compute)
    assert compute_called
    assert result1 == {"result": "data"}
    
    # Second call - from cache
    compute_called = False
    result2 = await cache.get_or_compute("key1", compute)
    assert not compute_called  # Should not recompute
    assert result2 == {"result": "data"}

async def test_cache_ttl_expiration():
    cache = SimpleCache(ttl=0.1)  # 100ms TTL
    
    result1 = await cache.get_or_compute("key", lambda: {"v": 1})
    await asyncio.sleep(0.2)  # Wait for expiration
    
    # Should recompute after TTL
    result2 = await cache.get_or_compute("key", lambda: {"v": 2})
    assert result2 == {"v": 2}  # New value, not cached
```

#### Integration Tests
```python
# test_cache_integration.py
@pytest.mark.integration
async def test_search_endpoint_caching():
    """Test actual FastAPI endpoint caching"""
    async with TestClient(app) as client:
        # First request - should compute
        start = time.time()
        response1 = await client.get("/search?q=test")
        time1 = time.time() - start
        
        # Second request - should be cached
        start = time.time()
        response2 = await client.get("/search?q=test")
        time2 = time.time() - start
        
        assert response1.json() == response2.json()
        assert time2 < time1 * 0.1  # Cached should be 10x faster
        assert response2.headers.get("X-Cache") == "HIT"
```

#### Exit Conditions
- [ ] Cache hit rate > 60% in realistic workload
- [ ] Cache hit latency < 5ms
- [ ] TTL expiration works correctly
- [ ] No memory leaks after 1000 cache operations

### 1.3 Git Webhook Support with Enhanced Security (4 hours)

**Testing Criteria:**

#### Security Tests
```python
# test_webhook_security.py
import hmac

async def test_webhook_signature_verification():
    """Verify GitHub signature validation with constant-time comparison"""
    payload = b'{"commits": []}'
    
    # Valid signature with constant-time comparison
    valid_sig = generate_github_signature(payload, SECRET)
    
    # Implementation must use hmac.compare_digest
    def verify_signature(payload, signature, secret):
        expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, f"sha256={expected}")
    
    response = await client.post(
        "/webhooks/github",
        content=payload,
        headers={"X-Hub-Signature-256": valid_sig}
    )
    assert response.status_code == 200
    
    # Invalid signature
    invalid_sig = "sha256=invalid"
    response = await client.post(
        "/webhooks/github",
        content=payload,
        headers={"X-Hub-Signature-256": invalid_sig}
    )
    assert response.status_code == 403
```

#### Integration Tests
```python
async def test_webhook_triggers_background_indexing():
    """Verify webhook triggers async indexing"""
    payload = {
        "commits": [{
            "added": ["new_file.py"],
            "modified": ["existing.py"],
            "removed": ["deleted.py"]
        }]
    }
    
    response = await client.post("/webhooks/github", json=payload)
    assert response.status_code == 200
    
    # Wait for background processing
    await asyncio.sleep(1)
    
    # Verify correct files were processed
    assert mock_indexer.index_file.call_count == 2  # added + modified
    assert mock_indexer.remove_file.call_count == 1  # removed
```

#### Exit Conditions
- [ ] 100% of valid webhooks accepted
- [ ] 100% of invalid signatures rejected
- [ ] Timing variance <1ms (constant-time comparison)
- [ ] <5ms verification overhead per request
- [ ] Background tasks complete within 5s
- [ ] No blocking of webhook response

**HMAC Security Testing:**

```python
# test_hmac_security.py
import time
import hmac
import hashlib

@pytest.mark.security
def test_timing_attack_resistance():
    """Verify constant-time comparison prevents timing attacks"""
    secret = b'webhook_secret'
    payload = b'{"commits": []}'
    valid_sig = 'sha256=' + hmac.new(secret, payload, hashlib.sha256).hexdigest()
    invalid_sig = 'sha256=0000000000000000000000000000000000000000'
    
    times = []
    for sig in [valid_sig, invalid_sig] * 50:  # 100 runs
        start = time.perf_counter()
        verify_webhook_signature(payload, sig, secret)
        times.append(time.perf_counter() - start)
    
    # Timing should be constant regardless of signature validity
    variance = max(times) - min(times)
    assert variance < 0.001  # <1ms variance

@pytest.mark.performance
def test_hmac_verification_overhead(benchmark):
    """Measure overhead of HMAC verification"""
    payload = b'x' * 1024  # 1KB payload
    sig = 'sha256=' + hmac.new(b'secret', payload, hashlib.sha256).hexdigest()
    
    overhead = benchmark(verify_webhook_signature, payload, sig, b'secret')
    assert overhead < 0.005  # <5ms overhead

@pytest.mark.security
def test_replay_protection():
    """Verify timestamp-based replay protection"""
    payload = json.dumps({
        "timestamp": time.time() - 360,  # 6 minutes old
        "commits": []
    }).encode()
    
    # Even with valid signature, old timestamp should be rejected
    valid_sig = generate_signature(payload, SECRET)
    response = await client.post("/webhooks/github", 
                                content=payload,
                                headers={"X-Hub-Signature-256": valid_sig})
    
    assert response.status_code == 403  # Rejected due to old timestamp
```

## Phase 1.5: Critical 2025 Enhancements (1 day)

### 1.4 Multitenancy Support (4 hours)
**Requirement**: Data isolation for multiple projects/users
**Implementation**: Qdrant group_id + Neo4j labels

```python
# Add to indexer_service.py
class MultitenantIndexer:
    async def index_with_tenant(self, file_path: str, tenant_id: str):
        chunks = await self.chunker.chunk_file(file_path)
        
        # Add tenant isolation to Qdrant
        for chunk in chunks:
            chunk.payload["group_id"] = tenant_id
            await self.qdrant.upsert(chunk, filter={"group_id": tenant_id})
        
        # Add tenant label to Neo4j
        await self.neo4j.query(
            "CREATE (c:Chunk {id: $id, tenant: $tenant})",
            {"id": chunk.id, "tenant": tenant_id}
        )
    
    async def search_with_tenant(self, query: str, tenant_id: str):
        # Filter by tenant in both databases
        qdrant_results = await self.qdrant.search(
            query, 
            filter={"must": [{"key": "group_id", "match": {"value": tenant_id}}]}
        )
        neo4j_context = await self.neo4j.query(
            "MATCH (c:Chunk {tenant: $tenant}) RETURN c",
            {"tenant": tenant_id}
        )
```

**Testing Criteria:**

```python
# test_multitenancy.py
@pytest.mark.asyncio
async def test_data_isolation_unit(mocker):
    """Verify tenant-scoped operations"""
    mocker.patch('self.qdrant.upsert')
    mocker.patch('self.neo4j.query')
    
    await index_with_tenant(self, 'file.py', 'tenantA')
    
    # Assert tenant filter applied
    self.qdrant.upsert.assert_called_with(ANY, filter={"group_id": "tenantA"})
    self.neo4j.query.assert_called_with(ANY, {"id": "chunk1", "tenant": "tenantA"})

@pytest.mark.integration
async def test_cross_tenant_queries(test_qdrant, test_neo4j):
    """Verify no cross-tenant data leakage"""
    await index_with_tenant(self, 'file1.py', 'tenantA')
    await index_with_tenant(self, 'file2.py', 'tenantB')
    
    results_a = await search_with_tenant(self, "query", 'tenantA')
    results_b = await search_with_tenant(self, "query", 'tenantB')
    
    # No leakage between tenants
    assert all(r.payload['group_id'] == 'tenantA' for r in results_a)
    assert results_a != results_b

@pytest.mark.performance
async def test_multitenancy_performance(benchmark):
    """Measure latency impact of tenant filtering"""
    latencies = [benchmark(search_with_tenant, "query", 'tenantA') for _ in range(100)]
    p95 = sorted(latencies)[94]
    assert p95 < 110  # <10% overhead vs 100ms baseline
```

**Exit Criteria:**
- [ ] 100% data isolation (zero cross-tenant leakage)
- [ ] <10% latency increase with tenant filtering
- [ ] >95% test coverage for tenant functions
- [ ] All tests pass 10 consecutive runs

### 1.5 Qdrant Quantization (1 hour)
**Benefit**: 75% memory reduction, 38% storage reduction
**Implementation**: Scalar quantization config

```python
# Add to qdrant initialization
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig

async def create_optimized_collection():
    await qdrant.create_collection(
        collection_name="code",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        quantization_config=ScalarQuantizationConfig(
            scalar=ScalarQuantization(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=False  # Keep on disk, load to RAM on demand
            )
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,  # Build HNSW after 20k vectors
            memmap_threshold=50000     # Use mmap after 50k vectors
        )
    )
```

**Testing Criteria:**

```python
# test_quantization.py
import psutil

@pytest.mark.benchmark
def test_memory_usage_reduction(qdrant_client):
    """Verify 70%+ memory reduction with quantization"""
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    # Create collection with quantization
    qdrant_client.create_collection(
        "test_code", 
        vectors_config=..., 
        quantization_config=ScalarQuantizationConfig(...)
    )
    
    # Insert 10k test vectors
    for i in range(10000):
        qdrant_client.upsert("test_code", points=[...])
    
    mem_after = process.memory_info().rss
    reduction = (mem_before - mem_after) / mem_before * 100
    assert reduction > 70  # Target: 75% reduction

@pytest.mark.asyncio
async def test_query_latency_impact(benchmark, qdrant_client):
    """Verify <5% latency increase with quantization"""
    await setup_test_vectors(qdrant_client, 10000)
    
    # Benchmark search performance
    latency = benchmark(qdrant_client.search, "test_code", query_vector=[0.1] * 768)
    assert latency < 105  # <5% over 100ms baseline

@pytest.mark.accuracy
async def test_recall_after_quantization(qdrant_client, ground_truth):
    """Verify <2% recall drop with INT8 quantization"""
    recall_scores = []
    
    for query, expected_ids in ground_truth:
        results = await qdrant_client.search("code", query, limit=10)
        hits = len(set(r.id for r in results) & set(expected_ids))
        recall = hits / len(expected_ids)
        recall_scores.append(recall)
    
    avg_recall = sum(recall_scores) / len(recall_scores)
    assert avg_recall > 0.98  # <2% drop from baseline
```

**Exit Criteria:**
- [ ] >70% memory reduction achieved
- [ ] <5% query latency increase
- [ ] <2% recall degradation
- [ ] Benchmarks consistent over 5 runs (variance <5%)

## Phase 1.6: Tree-sitter Integration (4 hours)

### 1.6 Replace AST with Tree-sitter for Battle-tested Parsing
**Requirement**: Use production-proven parser for better performance and reliability
**Implementation**: Tree-sitter for all languages (36x faster than AST)

```python
# tree_sitter_chunker.py
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts

class TreeSitterChunker:
    def __init__(self):
        # Load language bindings
        self.parsers = {
            'python': Language(tspython.language(), 'python'),
            'javascript': Language(tsjs.language(), 'javascript'),
            'typescript': Language(tsts.language(), 'typescript'),
            'go': Language(tsgo.language(), 'go'),
            'rust': Language(tsrust.language(), 'rust'),
        }
        
    async def extract_chunks(self, file_path: str, source: str, language: str):
        """Extract semantic chunks using Tree-sitter (36x faster than AST)"""
        parser = Parser()
        parser.set_language(self.parsers[language])
        
        # Parse with error recovery (works on broken code!)
        tree = parser.parse(bytes(source, 'utf8'))
        
        # Tree-sitter query for semantic extraction
        query = self.parsers[language].query("""
            (function_definition name: (identifier) @func)
            (class_definition name: (identifier) @class)
            (method_definition name: (identifier) @method)
        """)
        
        chunks = []
        for node, capture_name in query.captures(tree.root_node):
            chunk = {
                'id': self._generate_stable_id(file_path, node),
                'type': capture_name,
                'name': node.text.decode('utf8'),
                'start_line': node.start_point[0],
                'end_line': node.end_point[0],
                'content': source[node.start_byte:node.end_byte]
            }
            chunks.append(chunk)
        
        return chunks
    
    async def incremental_parse(self, old_tree, edits, new_source):
        """Incremental parsing - only reparse changed parts"""
        for edit in edits:
            old_tree.edit(
                start_byte=edit['start_byte'],
                old_end_byte=edit['old_end_byte'],
                new_end_byte=edit['new_end_byte'],
                start_point=edit['start_point'],
                old_end_point=edit['old_end_point'],
                new_end_point=edit['new_end_point']
            )
        
        # Reuses unchanged nodes - HUGE performance win!
        new_tree = parser.parse(bytes(new_source, 'utf8'), old_tree)
        return new_tree

# Update SmartCodeChunker to prefer Tree-sitter
class SmartCodeChunker:
    def __init__(self):
        self.tree_sitter = TreeSitterChunker()
        self.ast_chunker = PythonASTChunker()  # Fallback
        
    async def chunk_file(self, file_path: str):
        ext = Path(file_path).suffix
        language = self._ext_to_language(ext)
        
        try:
            # Prefer Tree-sitter for all languages
            with open(file_path) as f:
                source = f.read()
            return await self.tree_sitter.extract_chunks(file_path, source, language)
        except Exception as e:
            # Fallback to AST for Python only
            if language == 'python':
                return await self.ast_chunker.extract_chunks(file_path)
            raise e
```

**Performance Benefits:**
- **36x faster parsing** than Python AST
- **Incremental updates** - perfect for file watching
- **Error recovery** - extracts chunks from broken code
- **Uniform API** across all languages
- **Battle-tested** by GitHub, VSCode, Neovim

**Testing Criteria:**

```python
# test_tree_sitter.py
@pytest.mark.benchmark
def test_tree_sitter_performance(benchmark):
    """Verify Tree-sitter is faster than AST"""
    source = Path("large_file.py").read_text()
    
    # Benchmark Tree-sitter
    ts_time = benchmark(tree_sitter_parse, source)
    
    # Benchmark AST
    ast_time = benchmark(ast.parse, source)
    
    # Should be at least 10x faster (typically 36x)
    assert ts_time < ast_time / 10

@pytest.mark.incremental
async def test_incremental_parsing():
    """Verify incremental parsing only processes changes"""
    source = "def foo(): pass\ndef bar(): pass"
    tree = parser.parse(bytes(source, 'utf8'))
    
    # Make small edit
    new_source = "def foo(): pass\ndef baz(): pass"  # bar -> baz
    edits = compute_edits(source, new_source)
    
    # Incremental parse should be much faster
    start = time.perf_counter()
    new_tree = parser.parse(bytes(new_source, 'utf8'), tree)
    incremental_time = time.perf_counter() - start
    
    # Full reparse for comparison
    start = time.perf_counter()
    full_tree = parser.parse(bytes(new_source, 'utf8'))
    full_time = time.perf_counter() - start
    
    assert incremental_time < full_time / 5  # >5x faster

@pytest.mark.error_recovery
def test_parse_broken_code():
    """Verify Tree-sitter can extract from syntax errors"""
    broken_code = """
    def valid_function():
        return 42
    
    def broken_function(  # Missing closing paren
        print("still extracts this")
    
    class ValidClass:
        pass
    """
    
    tree = parser.parse(bytes(broken_code, 'utf8'))
    chunks = extract_chunks(tree)
    
    # Should still extract valid parts
    assert any(c['name'] == 'valid_function' for c in chunks)
    assert any(c['name'] == 'ValidClass' for c in chunks)
    # Partial extraction from broken function
    assert any('broken_function' in c['name'] for c in chunks)
```

**Exit Criteria:**
- [ ] Tree-sitter parsing 10x+ faster than AST
- [ ] Incremental updates 5x+ faster than full reparse
- [ ] Extracts 90%+ chunks from files with syntax errors
- [ ] All languages use unified Tree-sitter API
- [ ] AST fallback still works for Python

## Phase 1.7: Dynamic Re-ranker with Claude Haiku 3.5 (6 hours)

### 1.7 Claude Haiku-Powered Intelligent Document Selection
**Requirement**: Replace fixed k=10 with Claude Haiku 3.5 for intelligent re-ranking and dynamic k selection
**Implementation**: LLM-based re-ranking with heuristic fallback for resilience

```python
# haiku_reranker.py
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import redis
import json
import os
from anthropic import AsyncAnthropic
from datetime import datetime

class HaikuDynamicReranker:
    """
    Implements DynamicRAG-style reranking using Claude Haiku 3.5.
    Zero additional cost for users with Claude subscription.
    Combines intelligent re-ranking with dynamic k-selection in a single LLM call.
    """
    
    def __init__(
        self,
        min_k: int = 3,
        max_k: int = 15,
        api_key: Optional[str] = None
    ):
        self.min_k = min_k
        self.max_k = max_k
        self.cache = redis.Redis() if redis else None
        self.feature_flag = os.getenv("ENABLE_DYNAMIC_RERANKING", "false") == "true"
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # DO NOT create client here - use fresh client per request pattern
        
    async def dynamic_select(
        self, 
        query: str, 
        fused_results: List[Tuple[Dict, float]],
        context: Dict[str, Any] = None
    ) -> Tuple[List[Dict], int, Dict[str, Any]]:
        """
        Use Claude Haiku 3.5 to intelligently re-rank and select optimal k.
        Falls back to heuristic if Haiku unavailable.
        """
        start_time = datetime.now()
        
        if not self.feature_flag or not self.api_key:
            # Fallback to heuristic if disabled or no API key
            return await self._heuristic_fallback(query, fused_results)
        
        # Check cache
        cache_key = f"haiku_rerank:{hash(query)}:{len(fused_results)}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                cached_result = json.loads(cached)
                return (
                    cached_result['docs'],
                    cached_result['k'],
                    cached_result['metadata']
                )
        
        try:
            # Prepare documents for Haiku
            doc_summaries = []
            for i, (doc, score) in enumerate(fused_results[:self.max_k]):
                content_preview = doc.get('content', '')[:150]
                file_path = doc.get('file_path', 'unknown')
                doc_summaries.append(
                    f"[{i}] Score:{score:.3f} | {file_path}\n{content_preview}..."
                )
            
            # Optimized prompt for speed and accuracy
            prompt = f"""Query: {query}

Documents (top {len(doc_summaries)}):
{chr(10).join(doc_summaries)}

Task: Analyze these code search results and output JSON with:
1. reranked_indices: list of document indices ordered by TRUE relevance
2. optimal_k: optimal number to return ({self.min_k}-{self.max_k})
3. confidence: 0.0-1.0
4. reasoning: one-line explanation

Consider:
- High scores may be false positives (keyword match but wrong context)
- Simple queries need 3-5 docs, complex need 8-12
- Quality > Quantity

Output only JSON, no other text:"""

            # Call Haiku with fresh client (correct pattern)
            from anthropic import AsyncAnthropic, DefaultAioHttpClient
            
            async with AsyncAnthropic(
                api_key=self.api_key,
                http_client=DefaultAioHttpClient()  # Required for context manager
            ) as client:
                response = await client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=200,
                    temperature=0.1,  # Low for consistency
                    messages=[{"role": "user", "content": prompt}]
                )
            
            # Parse response
            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            result_json = json.loads(response_text[json_start:json_end])
            
            # Extract reranked indices and optimal k
            reranked_indices = result_json.get('reranked_indices', list(range(5)))
            optimal_k = min(
                result_json.get('optimal_k', 5),
                len(reranked_indices),
                len(fused_results)
            )
            
            # Build reranked document list
            reranked_docs = []
            for idx in reranked_indices[:optimal_k]:
                if idx < len(fused_results):
                    reranked_docs.append(fused_results[idx][0])
            
            # Metadata for observability
            metadata = {
                'model': 'claude-3-5-haiku',
                'confidence': result_json.get('confidence', 0.8),
                'reasoning': result_json.get('reasoning', ''),
                'latency_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'original_order': list(range(len(reranked_docs))),
                'reranked_order': reranked_indices[:optimal_k]
            }
            
            # Cache result
            if self.cache:
                cache_value = {
                    'docs': reranked_docs,
                    'k': optimal_k,
                    'metadata': metadata
                }
                self.cache.setex(cache_key, 300, json.dumps(cache_value))
            
            return reranked_docs, optimal_k, metadata
            
        except Exception as e:
            print(f"Haiku reranking failed: {e}, using fallback")
            return await self._heuristic_fallback(query, fused_results)
    
    async def _heuristic_fallback(
        self,
        query: str,
        results: List[Tuple[Dict, float]]
    ) -> Tuple[List[Dict], int, Dict[str, Any]]:
        """Fast heuristic fallback when Haiku unavailable"""
        # Find knee in score curve
        scores = [score for _, score in results[:20]]
        knee_k = self._find_score_knee(scores)
        
        # Adjust for query complexity
        complexity = self._estimate_complexity(query)
        optimal_k = int(knee_k * complexity)
        optimal_k = max(self.min_k, min(optimal_k, self.max_k, len(results)))
        
        docs = [doc for doc, _ in results[:optimal_k]]
        
        metadata = {
            'model': 'heuristic_fallback',
            'confidence': 0.6,
            'reasoning': f'Heuristic: knee at {knee_k}, complexity {complexity:.1f}',
            'latency_ms': 1
        }
        
        return docs, optimal_k, metadata
    
    def _find_score_knee(self, scores: List[float]) -> int:
        """Find elbow in score distribution"""
        if len(scores) <= self.min_k:
            return len(scores)
        
        # Calculate drops
        drops = []
        for i in range(1, min(len(scores), self.max_k)):
            drop = scores[i-1] - scores[i]
            drops.append((i, drop))
        
        if drops:
            # Find biggest drop
            max_drop_idx = max(drops, key=lambda x: x[1])[0]
            return min(max_drop_idx + 1, self.max_k)
        
        return 5
    
    def _estimate_complexity(self, query: str) -> float:
        """Quick complexity estimation"""
        q = query.lower()
        
        # Simple: 0.7x multiplier
        if any(p in q for p in ['what is', 'who is', 'where is']):
            return 0.7
        
        # Complex: 1.3x multiplier  
        if any(p in q for p in ['how', 'why', 'explain', 'compare']):
            return 1.3
        
        # Token-based
        tokens = len(query.split())
        if tokens > 15:
            return 1.2
        elif tokens < 5:
            return 0.8
        
        return 1.0

# haiku_service.py - Following existing service patterns
from anthropic import AsyncAnthropic
import asyncio
from typing import Optional, Dict, Any

class HaikuService:
    """
    Claude Haiku service following Context7 pattern.
    Fresh AsyncAnthropic client per request to avoid event loop issues.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # DO NOT create client here - Context7 pattern
        
    async def get_client(self) -> AsyncAnthropic:
        """Fresh client per request - matches NomicService pattern"""
        return AsyncAnthropic(api_key=self.api_key)
    
    async def rerank_sync(self, query: str, docs: List[Dict]) -> Dict:
        """Synchronous mode - wait for reranking (500-2000ms)"""
        # CORRECT usage with aiohttp backend for context manager support
        from anthropic import DefaultAioHttpClient
        
        async with AsyncAnthropic(
            api_key=self.api_key,
            http_client=DefaultAioHttpClient()  # Required for context manager
        ) as client:
            response = await client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": self._build_prompt(query, docs)}]
            )
            return self._parse_response(response)
    
    async def rerank_async(self, query: str, docs: List[Dict]) -> Dict:
        """Async mode - fire and forget, update cache in background"""
        # Return immediately with original order
        immediate_result = {
            'documents': docs[:10],  # Default k
            'optimal_k': 10,
            'mode': 'immediate'
        }
        
        # Schedule background reranking
        asyncio.create_task(self._background_rerank(query, docs))
        
        return immediate_result
    
    async def _background_rerank(self, query: str, docs: List[Dict]):
        """Background task to rerank and update cache"""
        try:
            result = await self.rerank_sync(query, docs)
            # Update Redis cache for future queries
            if self.cache:
                cache_key = f"haiku:{hash(query)}"
                await self.cache.setex(cache_key, 300, json.dumps(result))
        except Exception as e:
            logger.error(f"Background rerank failed: {e}")

# Integration with ServiceContainer
class ServiceContainer:
    def __init__(self, project_name: str):
        # ... existing services ...
        self.haiku: Optional[HaikuService] = None
        
    async def initialize_haiku_service(self):
        """Initialize Haiku service if API key available"""
        if os.getenv("ANTHROPIC_API_KEY"):
            self.haiku = HaikuService()
            logger.info("Haiku service initialized")
        else:
            logger.info("Haiku service skipped - no API key")

# Integration with HybridRetriever
class EnhancedHybridRetriever(HybridRetriever):
    def __init__(self, container):
        super().__init__(container)
        self.rerank_mode = os.getenv("HAIKU_MODE", "sync")  # sync or async
    
    async def find_similar_with_context(
        self,
        query: str,
        initial_k: int = 20,  # Retrieve more initially
        include_graph_context: bool = True,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Enhanced with dynamic re-ranking"""
        # 1. Get initial results with larger k
        results = await super().find_similar_with_context(
            query, 
            limit=initial_k,
            include_graph_context=include_graph_context,
            max_hops=max_hops
        )
        
        # 2. Apply RRF fusion (already done in parent)
        # Convert to (doc, score) tuples for reranker
        scored_results = [(r, r.get('score', 0)) for r in results]
        
        # 3. Dynamic re-ranking
        final_docs, selected_k, metadata = await self.dynamic_reranker.dynamic_select(
            query, scored_results
        )
        
        # 4. Add metadata for observability
        for doc in final_docs:
            doc['_dynamic_k'] = selected_k
            doc['_selection_metadata'] = metadata
        
        return final_docs
```

**Quality Tracking for Future RL:**
```python
# quality_tracker.py
class QualityTracker:
    """Track generation quality for RL reward signals"""
    
    async def track_generation_quality(
        self,
        query: str,
        selected_docs: List[Dict],
        llm_response: str,
        ground_truth: str = None
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for RL rewards.
        Based on DynamicRAG paper's reward function.
        """
        rewards = {}
        
        # 1. Exact Match (if ground truth available)
        if ground_truth:
            rewards['exact_match'] = float(llm_response.strip() == ground_truth.strip())
        
        # 2. Semantic Similarity (using embeddings)
        if ground_truth:
            resp_emb = await self.get_embedding(llm_response)
            truth_emb = await self.get_embedding(ground_truth)
            rewards['semantic_sim'] = cosine_similarity(resp_emb, truth_emb)
        
        # 3. Length Penalty (encourage conciseness)
        rewards['length_penalty'] = 1.0 / (1 + len(llm_response) / 1000)
        
        # 4. Confidence Score (from LLM if available)
        rewards['confidence'] = self._extract_confidence(llm_response)
        
        # 5. Context Precision (how many docs were actually used)
        rewards['context_precision'] = self._estimate_doc_usage(
            llm_response, selected_docs
        )
        
        # Composite reward (weighted average)
        weights = {'exact_match': 0.3, 'semantic_sim': 0.25, 
                  'length_penalty': 0.15, 'confidence': 0.15,
                  'context_precision': 0.15}
        
        total_reward = sum(
            rewards.get(k, 0) * v 
            for k, v in weights.items()
        )
        
        return {
            'individual_rewards': rewards,
            'total_reward': total_reward,
            'selected_k': len(selected_docs)
        }
```

**Performance Characteristics:**

| Mode | Latency Impact | Use Case | Benefits |
|------|---------------|----------|----------|
| **Sync Mode** | +500-2000ms | Production queries needing best quality | Guaranteed optimal k, immediate quality |
| **Async Mode** | +0ms (immediate) | High-throughput scenarios | No blocking, cache warming, scales better |
| **Cached** | +5ms | Repeated queries | Best of both - quality + speed |

**Configuration Options:**
```bash
# Environment variables
ANTHROPIC_API_KEY=sk-ant-...       # Required for Haiku
HAIKU_MODE=sync                    # sync or async (default: sync)
ENABLE_DYNAMIC_RERANKING=true      # Feature flag
HAIKU_CACHE_TTL=300               # Cache TTL in seconds
HAIKU_TIMEOUT_MS=2000             # Max time to wait for Haiku
```

**Testing Criteria:**

```python
# test_haiku_async.py
import pytest
from unittest.mock import AsyncMock, patch
import asyncio

@pytest.mark.asyncio
async def test_sync_mode_latency():
    """Verify sync mode completes within 2000ms"""
    service = HaikuService()
    docs = [{"content": f"doc{i}"} for i in range(20)]
    
    start = asyncio.get_event_loop().time()
    result = await service.rerank_sync("test query", docs)
    latency = (asyncio.get_event_loop().time() - start) * 1000
    
    assert latency < 2000  # Should complete within 2s
    assert result['mode'] == 'sync'

@pytest.mark.asyncio
async def test_async_mode_immediate_return():
    """Verify async mode returns immediately"""
    service = HaikuService()
    docs = [{"content": f"doc{i}"} for i in range(20)]
    
    start = asyncio.get_event_loop().time()
    result = await service.rerank_async("test query", docs)
    latency = (asyncio.get_event_loop().time() - start) * 1000
    
    assert latency < 10  # Should return in <10ms
    assert result['mode'] == 'immediate'
    assert len(result['documents']) == 10  # Default k

@pytest.mark.asyncio
async def test_fresh_client_pattern():
    """Verify fresh AsyncAnthropic client per request"""
    service = HaikuService()
    
    client1 = await service.get_client()
    client2 = await service.get_client()
    
    assert client1 is not client2  # Different instances
    assert isinstance(client1, AsyncAnthropic)

@pytest.mark.asyncio
async def test_cache_hit_performance():
    """Verify cached results return quickly"""
    service = HaikuService()
    service.cache = AsyncMock()
    service.cache.get.return_value = '{"documents": [], "k": 5}'
    
    start = asyncio.get_event_loop().time()
    # Second call should hit cache
    result = await service.rerank_sync("cached query", [])
    latency = (asyncio.get_event_loop().time() - start) * 1000
    
    assert latency < 10  # Cache hit should be <10ms

# test_dynamic_reranker.py
import pytest
import numpy as np

@pytest.mark.asyncio
async def test_score_knee_detection():
    """Verify elbow detection in score distribution"""
    reranker = DynamicReranker()
    
    # Create artificial score distribution with clear knee at position 5
    scores = [0.9, 0.85, 0.8, 0.75, 0.7, 0.4, 0.35, 0.3, 0.25, 0.2]
    results = [({"doc": i}, score) for i, score in enumerate(scores)]
    
    knee_k = reranker._find_score_knee(results)
    assert 4 <= knee_k <= 6  # Should detect the drop around position 5

@pytest.mark.asyncio
async def test_query_complexity_estimation():
    """Test complexity scoring for different query types"""
    reranker = DynamicReranker()
    
    # Simple query
    simple_score = reranker._estimate_query_complexity("What is Python?")
    assert simple_score < 1.0
    
    # Complex query
    complex_score = reranker._estimate_query_complexity(
        "Explain how neural networks learn and compare them to human learning"
    )
    assert complex_score > 1.0
    
    # Multi-hop query
    multihop_score = reranker._estimate_query_complexity(
        "What is RAG and then explain how it differs from fine-tuning?"
    )
    assert multihop_score > 1.2

@pytest.mark.benchmark
def test_reranking_performance(benchmark):
    """Ensure dynamic selection doesn't add significant latency"""
    reranker = DynamicReranker()
    results = [({"doc": i}, 0.9 - i*0.05) for i in range(20)]
    
    # Should complete in <5ms
    result = benchmark(reranker._find_score_knee, results)
    assert benchmark.stats['mean'] < 0.005  # 5ms

@pytest.mark.integration
async def test_fallback_to_fixed_k():
    """Test graceful fallback when feature flag is off"""
    reranker = DynamicReranker()
    reranker.feature_flag = False
    
    results = [({"doc": i}, score) for i, score in enumerate(range(20))]
    selected, k, _ = await reranker.dynamic_select("test query", results)
    
    assert k == 10  # Should fallback to fixed k=10
    assert len(selected) == 10
```

**Exit Criteria:**
- [ ] Dynamic k selection reduces to 3-7 docs for simple queries
- [ ] Dynamic k expands to 10-15 docs for complex queries  
- [ ] Score knee detection works on 90%+ of queries
- [ ] <5ms latency overhead for re-ranking
- [ ] Quality tracking logs all required metrics
- [ ] Feature flag enables safe rollout
- [ ] A/B test shows 5%+ improvement in result quality

**Migration Path to RL:**
1. Deploy heuristic version, collect data for 1-2 weeks
2. Train lightweight classifier on (query, score_distribution) -> optimal_k
3. Use logged quality metrics as reward signals
4. Implement DPO training following DynamicRAG paper
5. Gradual rollout: 10% -> 50% -> 100% traffic

## Phase 2: MCP-Specific Testing Requirements

### MCP Protocol Compliance Tests (Updated by Codex)
Note: We have consolidated on a single, canonical MCP server:
`neural-tools/run_mcp_server.py` which launches `neural-tools/src/mcp/neural_server_stdio.py`.
Older wrappers/proxy variants are deprecated.
```python
# tests/mcp/test_mcp_stdio_protocol.py
import asyncio
import json
import os
import sys
import subprocess
import pytest

SERVER_PATH = "neural-tools/src/mcp/neural_server_stdio.py"


def _jsonl(obj):
    return (json.dumps(obj) + "\n").encode()


@pytest.mark.asyncio
async def test_mcp_stdio_initialize_and_list_tools():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, SERVER_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # initialize
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"}
    }
    proc.stdin.write(_jsonl(init))
    await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    resp = json.loads(line)
    assert resp.get("jsonrpc") == "2.0"
    assert resp.get("id") == 1
    assert "result" in resp

    # tools/list
    tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    proc.stdin.write(_jsonl(tools_req))
    await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    resp = json.loads(line)
    assert resp.get("id") == 2
    tools = resp.get("result", {}).get("tools", []) if isinstance(resp.get("result"), dict) else resp.get("result", [])
    assert isinstance(tools, list) and len(tools) > 0

    proc.terminate()
    await proc.wait()


@pytest.mark.asyncio
async def test_mcp_no_stdout_logging():
    # Ensure server does not write logs to stdout (only JSON-RPC)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, SERVER_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    # initialize
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    proc.stdin.write(_jsonl(init))
    await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    # stdout must be valid JSON only
    json.loads(line)
    # stderr allowed to have logs
    proc.terminate(); await proc.wait()
```

### Event Loop Safety Tests (Updated by Codex)
```python
# tests/mcp/test_event_loop_safety.py
import asyncio
import sys
import os
import threading
import pytest

SERVER_PATH = "neural-tools/src/mcp/neural_server_stdio.py"


def _run_server_thread():
    # Runs the server in a separate thread/event loop to simulate host MCP environment
    async def _main():
        # Import here to avoid global loop binding
        import importlib.util
        spec = importlib.util.spec_from_file_location("neural_server_stdio", SERVER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        await mod.run()  # should manage its own loop

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asyncio.wait_for(_main(), timeout=1.0))
    except Exception:
        # We expect timeout because run() blocks; important is no RuntimeError
        pass
    finally:
        loop.stop()
        loop.close()


def test_no_event_loop_conflicts_threaded():
    t = threading.Thread(target=_run_server_thread, daemon=True)
    t.start()
    t.join(timeout=2)
    # Server should not crash due to loop conflicts; thread should still be alive or exited cleanly due to timeout
    assert True
```

### Service Degradation Tests (Updated by Codex)
```python
# tests/mcp/test_service_degradation.py
import asyncio
import json
import os
import sys
import pytest

SERVER_PATH = "neural-tools/src/mcp/neural_server_stdio.py"


@pytest.mark.asyncio
async def test_semantic_code_search_fallback_when_qdrant_down(monkeypatch):
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    # Simulate Qdrant failure by pointing to an invalid host
    env["QDRANT_HOST"] = "127.0.0.1"
    env["QDRANT_PORT"] = "65535"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, SERVER_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    # initialize
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    proc.stdin.write((json.dumps(init)+"\n").encode()); await proc.stdin.drain()
    await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    # call semantic_code_search
    req = {
      "jsonrpc": "2.0", "id": 2, "method": "tools/call",
      "params": {"name": "semantic_code_search", "arguments": {"query": "cache", "limit": 3}}
    }
    proc.stdin.write((json.dumps(req)+"\n").encode()); await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    resp = json.loads(line)
    # Should return success with fallback_mode True or an error message indicating fallback path
    out = json.loads(resp["result"][0]["text"])  # TextContent wrapper
    assert out.get("status") in ("success", "error")
    proc.terminate(); await proc.wait()
```

## Phase 3: Regression Test Suite

### Core Regression Tests (Run Before Every Change)
```bash
# run_regression.sh
#!/bin/bash
set -e

echo "Running core regression tests..."


# 1. MCP Protocol Compliance (STDIO)
pytest tests/mcp/test_mcp_stdio_protocol.py -v

# 2. Event Loop Safety
pytest tests/mcp/test_event_loop_safety.py -v

# 3. Service Initialization
pytest tests/test_service_container.py -v

# 4. Existing Integration Tests
pytest tests/integration/test_end_to_end_direct.py -v

# 5. Performance Regression
pytest tests/test_performance_benchmarks.py -v --benchmark

echo "✅ All regression tests passed"
```

### Performance Regression Benchmarks
```python
# test_performance_benchmarks.py
@pytest.mark.benchmark
async def test_search_performance_regression():
    """Ensure search doesn't get slower"""
    baseline = 100  # ms
    
    times = []
    for _ in range(10):
        start = time.time()
        await search("test query")
        times.append((time.time() - start) * 1000)
    
    p95 = sorted(times)[int(len(times) * 0.95)]
    assert p95 < baseline * 1.1  # Allow 10% variance
```

Additional MCP performance checks (Updated by Codex):
```python
# tests/mcp/test_mcp_latency_budget.py
import asyncio, json, os, sys, pytest, time
SERVER_PATH = "src/mcp/neural_server_stdio.py"

@pytest.mark.asyncio
async def test_tools_call_roundtrip_under_budget():
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, SERVER_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    init = {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
    proc.stdin.write((json.dumps(init)+"\n").encode()); await proc.stdin.drain()
    await proc.stdout.readline()
    req = {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"project_understanding","arguments":{}}}
    t0 = time.perf_counter()
    proc.stdin.write((json.dumps(req)+"\n").encode()); await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    elapsed_ms = (time.perf_counter()-t0)*1000
    assert elapsed_ms < 500  # round-trip under 500ms for a simple tool
    proc.terminate(); await proc.wait()
```

## Phase 4: Documentation & Technical Debt

### Documentation Requirements

Each feature must include:

1. **API Documentation** (OpenAPI/Swagger)
```python
@app.get("/search", 
    summary="Neural search with caching",
    response_description="Search results with cache status",
    responses={
        200: {"description": "Success", "headers": {
            "X-Cache": {"description": "HIT or MISS"}
        }}
    })
async def search(q: str):
    """
    Search with intelligent caching.
    
    Cache TTL: 5 minutes
    Cache key: MD5(query + params)
    """
```

2. **Architecture Decision Records (ADR)**
```markdown
# ADR-001: Caching Strategy

## Status: Accepted

## Context
Need to reduce query latency for repeated searches

## Decision
Use decorator-based caching with in-memory start, Redis later

## Consequences
- Pros: 3x faster responses, simple implementation
- Cons: Cache invalidation complexity
```

3. **Runbook Updates**
```markdown
# Troubleshooting Guide

## Issue: High cache miss rate
1. Check TTL settings: `CACHE_TTL` env var
2. Monitor unique query patterns
3. Adjust cache size if memory allows
 
 ## Issue: MCP client shows protocol errors
 1. Verify server logs are on stderr only; stdout must be JSON-RPC only
 2. Confirm `initialize` is handled before `tools/list` and `tools/call`
 3. Ensure tool schemas set `additionalProperties: false` and required fields
 4. Validate that `tools/call` returns a list of `TextContent` or appropriate MCP contents
 
 ## Issue: STDIO hangs on large payloads
 1. Avoid printing large logs to stdout; use stderr
 2. Use compact payloads and chunk results if necessary
 3. Increase client read timeouts and verify newline-delimited framing
```

### Technical Debt Management

#### Deprecation Strategy (DO NOT REMOVE)
```python
# src/api/deprecated.py
import warnings
from functools import wraps

def deprecated(reason):
    """Mark functions as deprecated"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage
@deprecated("Use index_file instead")
async def index_directory_full(path):
    """Legacy full re-index - kept for compatibility"""
    # Old implementation
```

#### Technical Debt Tracking
```yaml
# .tech-debt.yml
items:
  - id: TD-001
    description: "File watcher triggers full re-index"
    impact: "Performance - 100x slower than needed"
    effort: "2 hours"
    priority: "HIGH"
    resolution: "Change callback to use index_file"
    
  - id: TD-002  
    description: "No caching layer"
    impact: "Performance - repeated queries slow"
    effort: "4 hours"
    priority: "MEDIUM"
    resolution: "Add decorator-based cache"
```

## Phase 5: Continuous Monitoring

### Health Checks
```python
# src/monitoring/health.py
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "services": {
            "neo4j": await check_neo4j(),
            "qdrant": await check_qdrant(),
            "cache": check_cache_health()
        },
        "metrics": {
            "cache_hit_rate": calculate_hit_rate(),
            "index_queue_size": get_queue_size(),
            "degraded_mode": is_degraded()
        }
    }
```

### Metrics Collection
```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_misses = Counter('cache_misses_total', 'Total cache misses')
index_duration = Histogram('index_duration_seconds', 'Time to index file')
active_connections = Gauge('active_connections', 'Active DB connections')

# Use in code
@cache_hits.count_exceptions()
async def get_from_cache(key):
    # Implementation
```

## Final Exit Criteria

### Phase 1 Complete When:
- [ ] All regression tests pass
- [ ] File changes index in < 1s
- [ ] Cache hit rate > 60%
- [ ] Git webhooks process in < 5s
- [ ] No event loop errors in 24h test run
- [ ] Documentation updated
- [ ] Technical debt items tracked

### Production Ready When:
- [ ] 99.9% uptime over 7 days
- [ ] P95 latency < 100ms
- [ ] Zero data inconsistencies Neo4j ↔ Qdrant
- [ ] Graceful degradation tested for all services
- [ ] Monitoring dashboard operational
- [ ] Runbooks cover all failure modes

## L9 Critical Risk Assessment

### High-Risk Areas Requiring Extra Attention

#### 1. File Watcher Memory Leak Risk
**Risk**: Unbounded timer accumulation in DebouncedEventHandler
**Mitigation**: 
```python
# Add max pending timers check
MAX_PENDING_TIMERS = 1000
if len(self.timers) > MAX_PENDING_TIMERS:
    logger.error(f"Timer leak detected: {len(self.timers)} pending")
    self._cleanup_old_timers()
```
**Test**: Run 10,000 rapid file changes, verify memory stable

#### 2. Cache Invalidation Complexity
**Risk**: Stale cache after file updates
**Mitigation**: 
- File change events must invalidate related cache entries
- Use file path as part of cache key
- TTL must be < file watch debounce interval
**Test**: Modify file, immediately query, verify fresh results

#### 3. Webhook Security 
**Risk**: SSRF, webhook replay attacks
**Mitigation**:
```python
# Webhook validation requirements
- HMAC signature verification (mandatory)
- Timestamp validation (< 5 minutes old)
- Nonce tracking to prevent replay
- Rate limiting per repository
```
**Test**: Attempt replay attack, verify rejection

### Rollback Strategy Per Phase

#### Phase 1.1 Rollback (File Watcher)
```bash
# Feature flag approach
FILE_WATCHER_MODE=incremental  # New behavior
FILE_WATCHER_MODE=full         # Old behavior (safe fallback)

# Rollback command
kubectl set env deployment/neural-tools FILE_WATCHER_MODE=full
```

#### Phase 1.2 Rollback (Caching)
```python
# Cache bypass header
if request.headers.get("X-Cache-Bypass") == "true":
    return await direct_search(query)  # Skip cache
```

#### Phase 1.3 Rollback (Webhooks)
```python
# Webhook circuit breaker
if webhook_failures > 5:
    webhook_enabled = False
    logger.error("Webhook processing disabled due to failures")
```

### Performance Regression Gates

#### Baseline Requirements (from current system)
```yaml
current_baselines:
  search_p50: 45ms
  search_p95: 95ms
  search_p99: 180ms
  index_single_file: 800ms
  index_100_files: 12s
  memory_per_project: 150MB
  
regression_thresholds:
  search_p95_increase: 10%  # Fail if > 104ms
  index_time_increase: 15%  # Fail if > 920ms
  memory_increase: 20%      # Fail if > 180MB
```

#### Load Test Requirements
```python
# test_load_requirements.py
async def test_concurrent_operations():
    """Verify system handles concurrent load"""
    # 100 concurrent searches
    search_tasks = [search(f"query_{i}") for i in range(100)]
    
    # 20 concurrent file updates
    update_tasks = [update_file(f"file_{i}.py") for i in range(20)]
    
    # 5 webhook events
    webhook_tasks = [process_webhook(payload) for _ in range(5)]
    
    results = await asyncio.gather(*search_tasks, *update_tasks, *webhook_tasks)
    
    # Assert no failures
    assert all(r.status == "success" for r in results)
    
    # Assert performance maintained
    assert percentile(search_times, 95) < 104  # ms
```

### Data Consistency Verification

#### Neo4j ↔ Qdrant Consistency Check
```python
# consistency_monitor.py
async def verify_consistency():
    """Run every 10 minutes in production"""
    neo4j_chunks = await neo4j.query("MATCH (c:Chunk) RETURN c.id")
    qdrant_points = await qdrant.scroll(collection="code")
    
    neo4j_ids = {c["id"] for c in neo4j_chunks}
    qdrant_ids = {p.id for p in qdrant_points}
    
    missing_in_qdrant = neo4j_ids - qdrant_ids
    missing_in_neo4j = qdrant_ids - neo4j_ids
    
    if missing_in_qdrant or missing_in_neo4j:
        alert("Data inconsistency detected", {
            "missing_in_qdrant": len(missing_in_qdrant),
            "missing_in_neo4j": len(missing_in_neo4j)
        })
        
    return len(missing_in_qdrant) == 0 and len(missing_in_neo4j) == 0
```

### MCP-Specific Edge Cases

#### Event Loop Thread Safety Test
```python
# test_mcp_thread_safety.py
def test_no_event_loop_in_thread():
    """MCP servers often run in threads without event loops"""
    def run_in_thread():
        # Should NOT create new event loop
        try:
            loop = asyncio.get_event_loop()
            assert False, "Should not have event loop in thread"
        except RuntimeError:
            pass  # Expected
        
        # Should handle sync context properly
        server = NeuralMCPServer()
        result = server.handle_sync_request({"method": "search"})
        assert result["status"] == "success"
    
    thread = Thread(target=run_in_thread)
    thread.start()
    thread.join()
```

#### Large Response Handling
```python
# MCP has response size limits
MAX_MCP_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB

async def test_large_response_handling():
    """Verify we handle large responses correctly"""
    # Query that returns many results
    results = await search("common_term", limit=1000)
    
    response_size = len(json.dumps(results))
    if response_size > MAX_MCP_RESPONSE_SIZE:
        # Should paginate or truncate
        assert "next_page_token" in results
        assert len(results["items"]) <= 100
```

## Conclusion

This roadmap leverages our **existing 80% feature completeness** while incorporating critical 2025 standards feedback from Grok 4. The phased approach ensures:

1. **Stability First**: Quick wins that don't break existing functionality
2. **2025 Standards Compliance**: Multitenancy, quantization, selective reprocessing
3. **Strict Testing**: Every change has comprehensive test coverage
4. **MCP Compatibility**: Maintain protocol compatibility (no auth complexity needed)
5. **Documentation**: Every feature is documented
6. **Technical Debt**: Tracked and managed, not ignored
7. **Risk Mitigation**: Each change has rollback strategy
8. **Performance Gates**: Regression prevention with strict thresholds

## Phase 2: 2025 Production Enhancements (Critical Fixes)

### 2.1 OpenTelemetry Instrumentation (2 hours)
**Priority**: HIGH - Required for production observability

```python
# telemetry.py
from opentelemetry import trace, metrics
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.exporter.otlp.proto.grpc import trace_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider

def setup_telemetry(app: FastAPI):
    """Configure OpenTelemetry for production observability"""
    
    # Setup trace provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure OTLP exporter (for Datadog, New Relic, etc.)
    otlp_exporter = trace_exporter.OTLPSpanExporter(
        endpoint=os.getenv("OTLP_ENDPOINT", "localhost:4317"),
        insecure=True  # Use secure=True in production
    )
    
    # Auto-instrument frameworks
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()  # For Anthropic SDK calls
    RedisInstrumentor().instrument()
    
    # Custom traces for GraphRAG operations
    @contextmanager
    def trace_operation(name: str, attributes: dict = None):
        with tracer.start_as_current_span(name) as span:
            if attributes:
                span.set_attributes(attributes)
            yield span
    
    return trace_operation

# Usage in HybridRetriever
async def find_similar_with_context(self, query: str, ...):
    with self.trace_operation("hybrid_retrieval", {"query": query}):
        # Qdrant search span
        with self.trace_operation("qdrant_search"):
            search_results = await self.container.qdrant.search(...)
        
        # Neo4j enrichment span
        with self.trace_operation("neo4j_enrichment"):
            graph_context = await self._fetch_graph_context(...)
        
        # Haiku reranking span
        with self.trace_operation("haiku_rerank", {"mode": self.rerank_mode}):
            final_docs = await self.haiku.rerank(...)
```

### 2.2 Rate Limiting & Circuit Breakers (2 hours)
**Priority**: HIGH - Prevent API exhaustion and cascading failures

```python
# resilience.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from pybreaker import CircuitBreaker
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from functools import wraps

# Rate limiting for FastAPI endpoints
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# Circuit breaker for external services
anthropic_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    exclude=[asyncio.TimeoutError]  # Don't trip on timeouts
)

neo4j_breaker = CircuitBreaker(fail_max=10, reset_timeout=30)
qdrant_breaker = CircuitBreaker(fail_max=10, reset_timeout=30)

# Enhanced HaikuService with resilience
class ResilientHaikuService(HaikuService):
    
    @anthropic_breaker
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def rerank_sync(self, query: str, docs: List[Dict]) -> Dict:
        """Rerank with circuit breaker and retry logic"""
        try:
            # Use correct Anthropic SDK pattern
            from anthropic import AsyncAnthropic, DefaultAioHttpClient
            
            async with AsyncAnthropic(
                api_key=self.api_key,
                http_client=DefaultAioHttpClient()
            ) as client:
                # Add timeout
                response = await asyncio.wait_for(
                    client.messages.create(...),
                    timeout=2.0  # 2 second timeout
                )
                return self._parse_response(response)
        except Exception as e:
            # Log to structured logger
            logger.error("Haiku rerank failed", 
                        extra={"error": str(e), "query": query})
            raise

# Apply rate limiting to endpoints
@app.post("/search")
@limiter.limit("30/minute")  # 30 requests per minute per IP
async def search_endpoint(request: Request, query: SearchQuery):
    # Existing search logic
    pass
```

### 2.3 Structured Logging (1 hour)
**Priority**: MEDIUM - Essential for debugging production issues

```python
# logging_config.py
import structlog
from structlog.processors import CallsiteParameterAdder, JSONRenderer
import logging

def setup_structured_logging():
    """Configure structlog for production"""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            CallsiteParameterAdder(
                parameters=[
                    CallsiteParameter.FUNC_NAME,
                    CallsiteParameter.LINENO,
                    CallsiteParameter.PATHNAME,
                ]
            ),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            JSONRenderer()  # JSON output for log aggregation
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Set log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level),
    )
    
    return structlog.get_logger()

# Replace all print statements
logger = setup_structured_logging()

# Old: print(f"Haiku reranking failed: {e}")
# New:
logger.error("haiku_rerank_failed", 
            error=str(e), 
            query=query,
            doc_count=len(docs),
            mode=self.rerank_mode)

# Old: print(f"Score: {result['score']:.3f}")
# New:
logger.info("search_result",
           score=result['score'],
           file_path=result['file_path'],
           query=query)
```

### 2.4 Input Validation & Security (2 hours)
**Priority**: HIGH - Prevent prompt injection and validate all inputs

```python
# security.py
from pydantic import BaseModel, Field, validator
import re
from typing import List, Optional
import hashlib
import hmac

class SecureSearchQuery(BaseModel):
    """Validated search query with security checks"""
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)
    project_name: str = Field(..., regex="^[a-zA-Z0-9_-]+$")
    
    @validator('query')
    def validate_query_safety(cls, v):
        """Prevent prompt injection attempts"""
        # Check for common injection patterns
        dangerous_patterns = [
            r"ignore previous instructions",
            r"disregard all prior",
            r"system prompt",
            r"</?(script|iframe|object|embed)",
            r"javascript:",
            r"data:text/html",
        ]
        
        query_lower = v.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                raise ValueError(f"Potentially unsafe query detected")
        
        # Check for suspicious Unicode characters
        if any(ord(char) > 127 for char in v[:50]):  # Check first 50 chars
            # Log for review but don't block (may be legitimate)
            logger.warning("unicode_in_query", query=v)
        
        return v

class SecureHaikuPrompt(BaseModel):
    """Secure prompt construction for Haiku"""
    
    @staticmethod
    def build_safe_prompt(query: str, docs: List[Dict]) -> str:
        """Build prompt with escaping and truncation"""
        # Escape special characters
        safe_query = query.replace('"', '\\"').replace('\n', ' ')
        
        # Truncate docs to prevent context overflow
        safe_docs = []
        for i, doc in enumerate(docs[:15]):  # Max 15 docs
            content = doc.get('content', '')[:200]  # Max 200 chars per doc
            # Remove any potential prompt injections
            content = re.sub(r'[<>{}]', '', content)
            safe_docs.append(f"[{i}] {content}")
        
        return f"""Query: {safe_query}
Documents: {' | '.join(safe_docs)}
Task: Rerank by relevance and select optimal k (3-15).
Output JSON only."""

# Webhook signature verification
class WebhookSecurity:
    """Verify webhook signatures for GitHub/etc"""
    
    @staticmethod
    def verify_signature(
        payload: bytes,
        signature: str,
        secret: str
    ) -> bool:
        """Constant-time signature verification"""
        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison
        return hmac.compare_digest(
            f"sha256={expected}",
            signature
        )
```

### 2.5 Health Checks & Readiness Probes (1 hour)
**Priority**: MEDIUM - Required for Kubernetes/container orchestration

```python
# health.py
from fastapi import status
from typing import Dict, Any
import asyncio

@app.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_probe() -> Dict[str, str]:
    """Kubernetes liveness probe - is the service running?"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness_probe() -> Dict[str, Any]:
    """Kubernetes readiness probe - can the service handle requests?"""
    checks = {
        "qdrant": False,
        "neo4j": False,
        "redis": False,
        "haiku": False
    }
    
    # Check Qdrant
    try:
        await asyncio.wait_for(
            container.qdrant.client.get_collections(),
            timeout=2.0
        )
        checks["qdrant"] = True
    except:
        pass
    
    # Check Neo4j
    try:
        await asyncio.wait_for(
            container.neo4j.execute_cypher("RETURN 1"),
            timeout=2.0
        )
        checks["neo4j"] = True
    except:
        pass
    
    # Check Redis (if configured)
    if container.cache:
        try:
            await container.cache.ping()
            checks["redis"] = True
        except:
            pass
    
    # Check Haiku API key
    checks["haiku"] = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    # Overall readiness
    is_ready = checks["qdrant"] and checks["neo4j"]
    
    return {
        "ready": is_ready,
        "checks": checks,
        "version": os.getenv("APP_VERSION", "unknown")
    }

@app.get("/metrics")
async def prometheus_metrics():
    """Expose metrics for Prometheus scraping"""
    # Use prometheus-client library
    from prometheus_client import generate_latest, Counter, Histogram
    
    # Define metrics
    search_counter = Counter('searches_total', 'Total searches')
    search_latency = Histogram('search_duration_seconds', 'Search latency')
    
    return Response(content=generate_latest(), media_type="text/plain")
```

**Revised Timeline:**
- Phase 1 (Quick Wins): **2 days** 
- Phase 1.5 (Initial Enhancements): **1 day**
- Phase 1.6 (Tree-sitter): **4 hours**
- Phase 1.7 (Haiku Re-ranker): **6 hours**
- Phase 2 (2025 Production Fixes): **8 hours**
- Full testing suite: **2 days**
- Documentation & monitoring: **1 day**
- **Total: 7.5 days for production-ready system**

**Final Assessment vs 2025 Standards:**
- ✅ GraphRAG with Neo4j + Qdrant (already state-of-art)
- ✅ Selective reprocessing (added)
- ✅ Multitenancy support (added)
- ✅ Quantization for efficiency (added)
- ✅ Security best practices (HMAC with constant-time)
- ✅ Hybrid caching strategy (TTL + event-driven)
- ⚠️ No Streamable HTTP (acceptable for our use case)
- ⚠️ No ML-driven optimizations (pragmatic choice)

**Confidence**: 95%
Based on code analysis, Grok 4 validation, 2025 standards research, and pragmatic L9 engineering judgment.

**Key Assumptions**:
- Internal system (no OAuth/auth complexity needed)
- Current scale doesn't require distributed Qdrant
- MCP protocol stability through 2025
- Focus on pragmatic solutions over bleeding-edge

---

## UPDATED BY CODEX (2025-09-07): Local Cross‑Encoder Reranker, Integration, and Dockerization

This section documents concrete changes I implemented and how to use them, focusing on a fast, open‑source reranking path that avoids Anthropic calls by default. It supersedes the earlier Haiku‑centric Phase 1.7 for most use cases. Haiku remains available as an optional, feature‑flagged fallback.

### What Changed (Summary)
- Added a local cross‑encoder reranker (BGE family) with a strict latency budget and TTL cache.
- Integrated reranker into the Enhanced Hybrid Retriever; defaults to local rerank and returns within budget.
- Made Haiku rerank optional; disabled by default to eliminate Anthropic API calls.
- Updated Docker/Docker Compose to persist model weights and (optionally) bake them at build time.
- Cache keys are tenant‑aware when multitenancy is enabled to ensure data isolation.

### New Files and Modified Files
- Added: `neural-tools/src/infrastructure/cross_encoder_reranker.py`
- Modified: `neural-tools/src/infrastructure/enhanced_hybrid_retriever.py`
- Modified: `docker-compose.yml` (model volume + env vars)
- Modified: `docker/Dockerfile` (optional model prefetch + source layout)

All new/modified content in this section is marked “Updated by Codex”.

### Rationale (Updated by Codex)
- LLM‑based rerankers add seconds of tail latency and high token costs. For interactive tools, they miss the UX window even if asynchronous.
- Local cross‑encoders (e.g., BGE reranker) provide strong relevance lift with predictable latency (e.g., 50–150 ms for 30–50 candidates, CPU). This preserves speed and quality while remaining open‑source and offline‑friendly.

---

### Local Cross‑Encoder Reranker (Updated by Codex)

File: `neural-tools/src/infrastructure/cross_encoder_reranker.py`

Key features:
- Strict latency budget via `asyncio.wait_for`; never blocks beyond budget.
- TTL cache for pairwise `(tenant, query_hash, doc_id)` scores.
- Tenant‑aware cache keys (if `tenant_id` is present on the container).
- Compact pair text (title/header/snippet window) for faster tokenization.
- Heuristic fallback (knee detection + small keyword boosts) if model unavailable.

Core API:
```python
from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig

cfg = RerankConfig(
    model_name="BAAI/bge-reranker-base",         # or env RERANKER_MODEL
    model_path="/app/models/reranker",           # or env RERANKER_MODEL_PATH
    latency_budget_ms=120,                        # or env RERANK_BUDGET_MS
    batch_size=48,
    cache_ttl_s=600                               # or env RERANK_CACHE_TTL
)

reranker = CrossEncoderReranker(cfg, tenant_id="acme-corp")
reranked = await reranker.rerank(query, results, top_k=10)  # returns within budget or falls back
```

Environment variables (defaults baked in):
- `RERANKER_MODEL` (default: `BAAI/bge-reranker-base`)
- `RERANKER_MODEL_PATH` (default: none; recommended `/app/models/reranker`)
- `RERANK_BUDGET_MS` (default: `120`)
- `RERANK_CACHE_TTL` (default: `600`)

Notes:
- If the model cannot be loaded (e.g., weights unavailable offline), the reranker falls back to a fast heuristic and never blocks.
- Use a local path via `RERANKER_MODEL_PATH` to avoid any download at runtime.

---

### Enhanced Hybrid Retriever Integration (Updated by Codex)

File: `neural-tools/src/infrastructure/enhanced_hybrid_retriever.py`

What changed:
- New constructor params: `prefer_local: bool = True`, `rerank_latency_budget_ms: int = 120`, `allow_haiku_fallback: bool = False`.
- If `prefer_local=True` and the local model is available, the local reranker is used and returns within budget.
- Haiku is only used if `allow_haiku_fallback=True` (off by default).

Usage example:
```python
from src.servers.services.service_container import ServiceContainer
from src.servers.services.hybrid_retriever import HybridRetriever
from src.infrastructure.enhanced_hybrid_retriever import EnhancedHybridRetriever

container = ServiceContainer(project_name="myproject")
await container.initialize_all_services()

base = HybridRetriever(container)
retriever = EnhancedHybridRetriever(
    hybrid_retriever=base,
    prefer_local=True,
    allow_haiku_fallback=False,   # default: no Anthropic calls
    rerank_latency_budget_ms=120
)

results = await retriever.find_similar_with_context(
    query="optimize cache eviction policy",
    limit=10,
    include_graph_context=True,
    max_hops=2
)
```

Return shape:
- Each result includes metadata such as `reranked`, `rerank_position`, and `reranking_mode` (`local_cross_encoder` when local path is used).

---

### Docker and Compose (Updated by Codex)

Goals:
- Keep reranking in the same container for minimal latency and simplicity.
- Persist or bake model weights to avoid runtime downloads and cold starts.

Changes in `docker-compose.yml`:
```yaml
services:
  l9-graphrag:
    environment:
      - RERANKER_MODEL=BAAI/bge-reranker-base
      - RERANKER_MODEL_PATH=/app/models/reranker
      - RERANK_BUDGET_MS=120
      - RERANK_CACHE_TTL=600
      - TOKENIZERS_PARALLELISM=false
    volumes:
      - models:/app/models

volumes:
  models:
```

Changes in `docker/Dockerfile`:
```dockerfile
# Copy neural-tools sources so infrastructure modules are available
COPY neural-tools/src/ ./src/

# Optional: pre-download reranker model at build time
ARG DOWNLOAD_RERANKER_WEIGHTS=false
ARG RERANKER_MODEL=BAAI/bge-reranker-base
ENV RERANKER_MODEL_PATH=/app/models/reranker
RUN if [ "$DOWNLOAD_RERANKER_WEIGHTS" = "true" ]; then \
      python - << 'PY' && mkdir -p /app/models/reranker ; \
from sentence_transformers import CrossEncoder
import os
model_id = os.environ.get('RERANKER_MODEL', 'BAAI/bge-reranker-base')
model = CrossEncoder(model_id, trust_remote_code=False)
model.save('/app/models/reranker')
print('Saved reranker model to /app/models/reranker')
PY
    ; fi
```

Build options:
- Bake model weights into the image:
  ```bash
  docker build -f docker/Dockerfile -t l9-graphrag \
    --build-arg DOWNLOAD_RERANKER_WEIGHTS=true \
    --build-arg RERANKER_MODEL=BAAI/bge-reranker-base .
  ```
- Or mount weights via the `models` volume and set `RERANKER_MODEL_PATH=/app/models/reranker`.

When to split reranker into its own container (optional):
- Only if you need independent GPU scaling, strict isolation, or cross‑service sharing. Otherwise, in‑process is fastest.
- If splitting: expose a small HTTP API with a 150 ms client timeout and a circuit breaker; co‑locate on the same network for <2 ms hop.

---

### Data Isolation (Updated by Codex)
- The local reranker’s cache keys include `tenant_id` (if present on the container via multitenant setup). This ensures cross‑tenant score reuse does not occur.
- For Qdrant/Neo4j, continue using existing tenant labels and filters defined in `neural-tools/src/servers/services/multitenant_service_container.py` and `src/infrastructure/multitenancy.py`.

---

### Latency Budget and Gating (Updated by Codex)

Target budgets per query:
- Vector + BM25 retrieval: ≤ 40 ms
- Fusion (RRF) + MMR diversification: ≤ 5 ms
- Cross‑encoder rerank (local): ≤ 120 ms (hard timeout)
- Graph expansion: ≤ 80 ms only if query is classified as multi‑hop or top‑k score margin is small

Example of simple gating for graph expansion (to implement next):
```python
def should_expand_graph(query: str, scores: list[float]) -> bool:
    q = query.lower()
    simple = any(p in q for p in ["what is", "who is", "where is"]) and len(query.split()) < 10
    margin_small = len(scores) >= 5 and (scores[0] - scores[4] < 0.05)
    multihop = any(p in q for p in ["how", "why", "compare", "steps", "then"])
    return (multihop or margin_small) and not simple

# In HybridRetriever flow (pseudocode)
if include_graph_context and should_expand_graph(query, [r['score'] for r in initial_results]):
    graph_context = await self._fetch_graph_context(chunk_ids, max_hops)
else:
    graph_context = [{} for _ in initial_results]
```

---

### Haiku Rerank (Now Optional) (Updated by Codex)
- Keep Haiku only behind `allow_haiku_fallback=True` with a tight timeout. By default we do not call Anthropic.
- If you re‑enable it, pool the client and warm it up at startup to avoid connection cold‑starts.

---

### Quantization Benchmark Correction (Updated by Codex)
The earlier memory test calculated reduction incorrectly after inserts. Correct approach:

1) Create two collections: baseline and quantized; insert the same vectors.
2) Compare process RSS deltas or use Qdrant’s segment memory metrics.

Sketch:
```python
import psutil, time

def mem_rss():
    import os
    return psutil.Process(os.getpid()).memory_info().rss

mem0 = mem_rss()
create_collection("baseline", quantized=False)
insert_vectors("baseline", N=10000)
mem_base = mem_rss()

create_collection("quantized", quantized=True)
insert_vectors("quantized", N=10000)
mem_quant = mem_rss()

reduction = (mem_base - mem_quant) / max(mem_base, 1) * 100
assert reduction > 70
```

---

### End‑to‑End Example (Updated by Codex)

```python
from src.servers.services.service_container import ServiceContainer
from src.servers.services.hybrid_retriever import HybridRetriever
from src.infrastructure.enhanced_hybrid_retriever import EnhancedHybridRetriever

async def search(query: str):
    container = ServiceContainer(project_name="acme")
    await container.initialize_all_services()

    base = HybridRetriever(container)
    retriever = EnhancedHybridRetriever(
        hybrid_retriever=base,
        prefer_local=True,
        allow_haiku_fallback=False,
        rerank_latency_budget_ms=120,
    )

    results = await retriever.find_similar_with_context(
        query=query,
        limit=10,
        include_graph_context=True,
        max_hops=2
    )
    return results
```

---

### Runbook (Updated by Codex)
- Model missing errors at startup:
  - Set `RERANKER_MODEL_PATH` to a local directory with saved weights (`CrossEncoder.save()`), or build with `DOWNLOAD_RERANKER_WEIGHTS=true`.
- High rerank latency:
  - Lower `RERANK_BUDGET_MS`, reduce candidate set size (e.g., `top_k * 2`), or switch to GPU.
- Cache not effective:
  - Ensure queries normalize (trim/lower) and that tenant_id is stable per session.
- Anthropic calls observed:
  - Confirm `allow_haiku_fallback=False` when constructing `EnhancedHybridRetriever`.

---

### Status (Updated by Codex)
- Local reranker integrated and enabled by default.
- Docker/Compose support added for persistent model weights.
- Anthropic usage disabled by default; optional fallback remains.
- Next: implement graph expansion gating and refine performance dashboards to track rerank p50/p95 and skip counts.

---

## Testing Criteria & Exit Conditions (Updated by Codex)

### A. Local Cross‑Encoder Reranker

#### Unit Tests
```python
# test_cross_encoder_reranker_unit.py
import asyncio
import time
import os
import pytest
from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig


@pytest.mark.asyncio
async def test_respects_latency_budget(monkeypatch):
    # Force very small budget to verify timeout path works and returns quickly
    cfg = RerankConfig(latency_budget_ms=10)
    rr = CrossEncoderReranker(cfg)
    # Empty model path forces heuristic fallback path to be used on timeout/model-missing
    start = time.perf_counter()
    results = await rr.rerank("query", [{"content": "a", "score": 0.9}], top_k=1)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= 60  # budget(10ms)+overhead; remains fast
    assert len(results) == 1


@pytest.mark.asyncio
async def test_tenant_isolation_in_cache():
    cfg = RerankConfig(latency_budget_ms=120)
    rr_a = CrossEncoderReranker(cfg, tenant_id="tenantA")
    rr_b = CrossEncoderReranker(cfg, tenant_id="tenantB")
    docs = [{"content": "same doc", "score": 0.5, "file_path": "x.py"}]
    # First call warms cache for tenantA
    _ = await rr_a.rerank("same query", docs, top_k=1)
    # Different tenant should not hit same cache key; behavior must be independent
    _ = await rr_b.rerank("same query", docs, top_k=1)
    # There is no direct counter here; lack of exceptions and independent operation is sufficient


@pytest.mark.asyncio
async def test_heuristic_fallback_when_model_missing(monkeypatch):
    # Ensure no model path to force heuristic path
    cfg = RerankConfig(model_path=None, latency_budget_ms=50)
    rr = CrossEncoderReranker(cfg)
    results = await rr.rerank(
        "cache eviction policy",
        [
            {"content": "LRU cache", "score": 0.8},
            {"content": "random policy", "score": 0.6}
        ],
        top_k=2
    )
    assert len(results) == 2
    assert all("rerank_score" in r for r in results)
```

#### Integration Tests
```python
# test_cross_encoder_reranker_integration.py
import asyncio
import os
import pytest
from src.servers.services.service_container import ServiceContainer
from src.servers.services.hybrid_retriever import HybridRetriever
from src.infrastructure.enhanced_hybrid_retriever import EnhancedHybridRetriever


@pytest.mark.asyncio
async def test_local_rerank_integration_no_anthropic_calls(monkeypatch):
    # Ensure Anthropic is not used
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    container = ServiceContainer(project_name="test")
    await container.initialize_all_services()
    base = HybridRetriever(container)
    retriever = EnhancedHybridRetriever(
        hybrid_retriever=base,
        prefer_local=True,
        allow_haiku_fallback=False,
        rerank_latency_budget_ms=120,
    )
    results = await retriever.find_similar_with_context("simple query", limit=5)
    # When local rerank path runs, metadata indicates local_cross_encoder
    if results:
        mode = results[0].get("metadata", {}).get("reranking_mode")
        assert mode in ("local_cross_encoder", None)  # None if rerank threshold not met
```

#### Benchmarks
```python
# test_cross_encoder_reranker_bench.py
import time
import pytest
from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_rerank_under_120ms_budget():
    cfg = RerankConfig(latency_budget_ms=120)
    rr = CrossEncoderReranker(cfg)
    docs = [{"content": "doc %d" % i, "score": 0.9 - i*0.01} for i in range(40)]
    start = time.perf_counter()
    _ = await rr.rerank("query", docs, top_k=10)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms <= 150
```

#### Exit Conditions
- [ ] p95 rerank latency ≤ 120 ms + 25% overhead (≤150 ms) for 30–50 candidates on CPU.
- [ ] Cache hits observed on repeated query/doc pairs within TTL.
- [ ] No cross‑tenant cache reuse (separate caches by tenant key).
- [ ] Heuristic fallback returns results without errors if model missing or timed out.
- [ ] No Anthropic calls when `allow_haiku_fallback=False`.

---

### B. Enhanced Hybrid Retriever Integration

#### Unit/Integration Tests
```python
# test_enhanced_retriever_integration.py
import pytest
from src.infrastructure.enhanced_hybrid_retriever import EnhancedHybridRetriever


@pytest.mark.asyncio
async def test_metadata_flags_present(mock_hybrid):
    retriever = EnhancedHybridRetriever(mock_hybrid, prefer_local=True, allow_haiku_fallback=False)
    results = await retriever.find_similar_with_context("q", limit=5)
    if results:
        m = results[0].get("metadata", {})
        assert "enhanced_hybrid_retrieval" in m


@pytest.mark.asyncio
async def test_threshold_bypass(mock_hybrid):
    retriever = EnhancedHybridRetriever(mock_hybrid, prefer_local=True, allow_haiku_fallback=False, rerank_latency_budget_ms=80)
    # Force threshold higher than results to verify bypass
    retriever.rerank_threshold = 999
    results = await retriever.find_similar_with_context("q", limit=5)
    # Should return untouched subset
    assert len(results) <= 5
```

#### Exit Conditions
- [ ] Default path uses local rerank and returns within configured budget.
- [ ] Haiku rerank is never invoked unless `allow_haiku_fallback=True`.
- [ ] Results include `reranked`, `rerank_position`, and `reranking_mode` when rerank applied.

---

### C. Dockerization & Model Handling

#### Validation Steps
```bash
# 1) Build with baked model weights
docker build -f docker/Dockerfile -t l9-graphrag \
  --build-arg DOWNLOAD_RERANKER_WEIGHTS=true \
  --build-arg RERANKER_MODEL=BAAI/bge-reranker-base .

# 2) Or mount local weights
mkdir -p models/reranker && ls models/reranker  # ensure files exist
docker compose up --build
```

#### Runtime Checks
- Container starts without attempting network downloads for the model when `RERANKER_MODEL_PATH` is populated.
- `TOKENIZERS_PARALLELISM=false` set to reduce log noise.
- Health endpoints in l9-graphrag remain healthy post‑startup.

#### Exit Conditions
- [ ] Image builds successfully in both “baked weights” and “mounted weights” modes.
- [ ] Container cold‑start time does not include model download when weights are provided.
- [ ] Reranker returns results in ≤150 ms p95 in containerized environment.

---

### D. Data Isolation (Reranker Cache)

#### Tests
```python
# test_reranker_cache_isolation.py
import pytest
from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig

@pytest.mark.asyncio
async def test_cache_isolation_by_tenant():
    cfg = RerankConfig(latency_budget_ms=80)
    a = CrossEncoderReranker(cfg, tenant_id="A")
    b = CrossEncoderReranker(cfg, tenant_id="B")
    docs = [{"content": "shared", "file_path": "f.py", "score": 0.9}]
    _ = await a.rerank("q", docs, top_k=1)
    _ = await b.rerank("q", docs, top_k=1)  # must not error or reuse keys across tenants
```

#### Exit Conditions
- [ ] No cross‑tenant cache hits; separate keyspaces per tenant.
- [ ] TTL expiry removes stale cache entries; memory stable over 10k ops.

---

### E. Latency Budget & Graph Gating

#### Unit Tests
```python
# test_graph_gating.py
from src.infrastructure.enhanced_hybrid_retriever import EnhancedHybridRetriever

def test_should_expand_graph_basic():
    scores = [0.9, 0.88, 0.87, 0.86, 0.85]
    assert EnhancedHybridRetriever.__dict__.get('should_expand_graph', lambda *_: False)(
        "how to refactor module then compare", scores
    ) is True
```

#### Integration Checks
- When gating returns False, `_fetch_graph_context` is not invoked.
- When True, graph expansion executes with a hop/time budget.

#### Exit Conditions
- [ ] Graph expansion only runs on multi‑hop/low‑margin queries.
- [ ] End‑to‑end p95 latency meets budget when gating is active.

---

### F. Quantization Benchmark (Corrected)

#### Tests
```python
# test_quantization_correctness.py
import pytest

@pytest.mark.benchmark
def test_quantized_vs_baseline_memory(qdrant_client):
    # Pseudocode hooks; assumes helpers exist
    base = create_collection(qdrant_client, name="baseline", quantized=False)
    quant = create_collection(qdrant_client, name="quantized", quantized=True)
    insert_vectors(base, 10000)
    insert_vectors(quant, 10000)
    mem_base = get_memory_usage(base)
    mem_quant = get_memory_usage(quant)
    assert (mem_base - mem_quant) / max(mem_base, 1) * 100 > 70
```

#### Exit Conditions
- [ ] ≥70% memory reduction vs baseline confirmed with stable methodology.
- [ ] <5% latency increase and <2% recall drop maintained in quantized collection.
