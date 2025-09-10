# ADR-0011: Semantic Search Enablement — Qdrant Alignment, Nomic Connectivity, and Code Structure Extraction

**Date**: 2025-09-09  
**Status**: Proposed  
**Decision Makers**: L9 Engineering Team  
**Consulted**: Previous ADRs 0005, 0008, 0010; repo audit 2025-09-09  
**Informed**: MCP stakeholders, Platform Ops

## Context

The stack is near-functional but semantic code search is not reliably end-to-end. Verified issues in the current repo:

- Qdrant client/server drift and API contract mismatch
  - Server pinned at `qdrant/qdrant:v1.12.4` (compose).
  - Host (MCP) may resolve `qdrant-client` to 1.15.x; indexer pins 1.10.0.
  - `CollectionManager.ensure_collection_exists(...)` calls `ensure_collection(collection_name, vector_size, distance=...)`, but `QdrantService.ensure_collection(...)` ignores `distance` and has a narrower signature, causing runtime errors and degraded mode.
- Embedding dimension propagation
  - System-wide `EMBED_DIM=768` is intended. `QdrantService.ensure_collection` defaults `vector_size=1536` unless overridden correctly; dimension must be enforced consistently to avoid upsert/search failures.
- Nomic reachability from host MCP
  - MCP STDIO process runs on host; `NomicEmbedClient` defaults to `172.18.0.5:8000` when `EMBEDDING_SERVICE_HOST` is unset. Compose exposes embeddings as `48000:8000`, so host should target `localhost:48000`.
- Code structure extraction
  - Indexer performs heuristic chunking and writes File/CodeChunk nodes; tree-sitter-based extraction for Classes/Functions is not wired, limiting graph quality and semantic recall.
- Observability
  - Indexer health endpoints and logging exist, but failure modes are quiet (e.g., collection creation mismatch) and lack explicit signal/metrics.

These issues block reliable semantic code search and hybrid retrieval despite otherwise healthy services (Neo4j writes, Redis phase-3 caching, embeddings container healthy).

## Decision

Adopt a focused enablement plan to make semantic code search reliably operational:

1. Version alignment and API contract fix for Qdrant
   - Either pin all Python clients to `qdrant-client==1.12.*` or upgrade server to `v1.15.*`; choose one path and enforce with a startup compatibility check.
   - Expand `QdrantService.ensure_collection(...)` to accept and honor `distance` and `vector_size` from `CollectionManager` calls (named-vector config remains).
2. Enforce embedding dimension propagation with runtime detection
   - Implement automatic dimension detection from Nomic embedder at startup; fallback to env `EMBED_DIM=768`.
   - Use detected dimensions consistently across all vector operations and collection creation.
   - Add dimension validation on upsert with clear error messages.
3. Correct Nomic connectivity for host MCP
   - Resolve MCP → embeddings endpoint via env (`EMBEDDING_SERVICE_HOST=localhost`, `EMBEDDING_SERVICE_PORT=48000` for local compose) and log the effective base URL.
   - Keep current retry/timeouts; add readiness probe on startup.
4. Implement code structure extraction (tree-sitter)
   - Enable per-language symbol extraction (classes/functions/methods) and persist to Neo4j using existing helpers.
   - Tag Qdrant payload with symbol metadata for better reranking/fusion.
   - Add batch processing with backpressure control to prevent parser performance bottlenecks.
5. Implement hybrid search ranking with configurable scoring
   - Add sophisticated result fusion combining vector similarity and graph relevance scores.
   - Support configurable scoring weights (e.g., `vector_weight=0.7`, `graph_weight=0.3`).
   - Enable context expansion through graph traversal for enriched results.
6. Improve observability and fail-fast with Context7 patterns
   - Add explicit logs and metrics for collection creation, dimension mismatches, and MCP connectivity checks.
   - Implement structured error handling with actionable remediation messages.
   - Add health checks with detailed component status and dependency validation.

Confidence: 95% — Based on concrete code analysis and validated misalignments.

## Technical Implementation Plan

Phase 1 — Unblock Vector Path (P1)
- Pin or upgrade:
  - Option A (fastest): Pin MCP/host `qdrant-client==1.12.*` (align to server 1.12.4).
  - Option B (preferred long-term): Upgrade server to 1.15.*, run backups and migration notes.
- Qdrant API contract:
  - Update `QdrantService.ensure_collection(collection_name, vector_size, distance=Distance.COSINE)`; use named vectors `{"dense": VectorParams(size=vector_size, distance=distance)}`.
  - Validate dimension on first upsert against `collection.config.params.vectors.size`; log error and refuse upsert if mismatch.
- Preflight checks (MCP startup):
  - Compare client/server versions; emit WARN if minor drift, ERROR if major; include server REST addr.
  - Describe/create `project_<name>_code` with `size=EMBED_DIM` and `distance=COSINE`.

Phase 2 — Fix MCP → Nomic Connectivity with Dimension Detection (P1)
- Set env for host dev: `EMBEDDING_SERVICE_HOST=localhost`, `EMBEDDING_SERVICE_PORT=48000` (compose port mapping).
- Add readiness probe on MCP init: HEAD/GET `/health`; log effective base URL and latency.
- Implement automatic dimension detection: `await detect_dim(base_url)` using single-embed test call.
- Propagate detected dimensions to all vector operations and collection creation.
- Keep existing retry/backoff; retain fallback local embeddings only for degraded testing.

Phase 3 — Structure Extraction with Batch Processing (P2)
- Use tree-sitter bindings already declared in indexer requirements to extract:
  - Python: classes, functions (name, span).
  - JS/TS: classes, functions, exported symbols.
  - (Optional) Go/Java in follow-ups.
- Implement batch processing with configurable batch sizes and backpressure control.
- Add parsing timeout per file to prevent blocking on malformed code.
- Persist to Neo4j via existing `create_class_node` and `create_function_node`; link to File and CodeChunk nodes.
- Augment Qdrant payload with `symbol_type`, `symbol_name`, `start_line`, `end_line`, `language`.
- Logging: per-file summary `symbols_extracted`, per-language counters; WARN on parser errors.

Phase 4 — Hybrid Search Implementation (P2)  
- Implement sophisticated result fusion combining vector similarity with graph relevance scores.
- Add configurable scoring weights via env vars: `HYBRID_VECTOR_WEIGHT=0.7`, `HYBRID_GRAPH_WEIGHT=0.3`.
- Enable context expansion through graph traversal (1-2 hops) for result enrichment.
- Support hybrid ranking algorithms: weighted sum, RRF (Reciprocal Rank Fusion), or custom.

Phase 5 — Enhanced Observability and Guardrails (P2)
- Logging: INFO logs for collection ensure, dimension checks, and Nomic base URL; ERROR on mismatches.
- Metrics: counters for `collections_created`, `dimension_mismatch_errors`, `embedding_timeouts`, `symbols_extracted_total`.
- Add structured error handling with actionable remediation suggestions.
- Implement dependency health checks with detailed status reporting.
- Indexer `/status`: include `degraded_services`, `chunks_created`, `symbols_extracted` counts, and `hybrid_search_config`.

## Migration Strategy (1536D → 768D Collections)

For existing deployments with 1536D collections:
1. **Snapshot Current State**: Export collection via Qdrant API snapshots
2. **Create New Collection**: With detected/configured 768D dimensions
3. **Reindex Content**: Process all documents through new embedding pipeline
4. **Verify & Cutover**: Validate search quality before switching traffic
5. **Rollback Plan**: Keep old collection for 48h as fallback

```python
async def migrate_collection(old_name: str, new_name: str, new_dim: int):
    # 1. Create snapshot
    await client.create_snapshot(collection_name=old_name)
    
    # 2. Create new collection with correct dimensions
    await client.create_collection(
        collection_name=new_name,
        vectors_config={"dense": VectorParams(size=new_dim, distance=Distance.COSINE)}
    )
    
    # 3. Reindex (handled by separate batch job)
    # 4. Atomic swap via alias or env var
```

## SLOs & Alerting

### Service Level Objectives
- **Search Latency**: p95 < 300ms, p99 < 500ms
- **Embedding Latency**: p95 < 900ms, p99 < 1500ms
- **Indexing Throughput**: > 10 files/second
- **Error Rates**: < 1% for all operations
- **Cache Hit Ratio**: > 80% after warmup

### Alert Thresholds
- **Critical**: Error rate > 5% for 5 minutes
- **Warning**: p95 latency > SLO for 10 minutes
- **Info**: Cache hit ratio < 60% for 30 minutes

### Correlation IDs
```python
import uuid
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar('request_id', default='')

async def with_correlation_id(func):
    async def wrapper(*args, **kwargs):
        req_id = str(uuid.uuid4())
        request_id.set(req_id)
        logger.info(f"Starting request", extra={"request_id": req_id})
        return await func(*args, **kwargs)
    return wrapper
```

## Robust Stability Improvements

### Circuit Breakers
```python
from circuit_breaker import CircuitBreaker

nomic_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=httpx.RequestError
)

@nomic_breaker
async def embed_with_circuit_breaker(texts: List[str]) -> List[List[float]]:
    return await embed(base_url, texts)
```

### Deterministic IDs with Collision Detection
```python
def generate_deterministic_id(file_path: str, chunk_index: int) -> str:
    """Generate deterministic 128-bit ID with collision checking."""
    content = f"{file_path}:chunk:{chunk_index}"
    full_hash = hashlib.sha256(content.encode()).hexdigest()
    doc_id = full_hash[:32]  # 128-bit hex string
    
    # Check for collisions
    existing = await client.retrieve(doc_id)
    if existing and existing.payload.get("file_path") != file_path:
        # Collision detected, add discriminator
        doc_id = hashlib.sha256(f"{content}:{uuid.uuid4()}".encode()).hexdigest()[:32]
        logger.warning(f"ID collision detected for {file_path}, using fallback ID")
    
    return doc_id
```

### Cache Invalidation Strategy
```python
class CacheInvalidator:
    async def invalidate_on_write(self, operation: str, keys: List[str]):
        """Write-through cache invalidation."""
        if operation in ["CREATE", "MERGE", "UPDATE", "DELETE"]:
            # Invalidate affected cache keys
            pipeline = redis_client.pipeline()
            for key in keys:
                pipeline.delete(key)
                pipeline.publish(f"cache:invalidated", key)
            await pipeline.execute()
            
    async def check_staleness(self, cached_result: Dict, doc_id: str) -> bool:
        """Check if cached result is stale."""
        cached_ts = cached_result.get("cached_at", 0)
        
        # Get last write timestamp for document
        last_write = await self.get_last_write_timestamp(doc_id)
        
        if last_write > cached_ts:
            logger.info(f"Stale cache detected for {doc_id}")
            return True
        return False
```

## Consequences

Positive Outcomes
- Reliable semantic search path: Embed → Upsert → Search works across codebase.
- Reduced production drift: explicit version alignment and checks.
- Higher-quality retrieval: symbol-aware payloads and richer Neo4j graph.
- Faster triage: clear logs/metrics for common failure modes.
- Production resilience: circuit breakers prevent cascade failures.
- Operational visibility: correlation IDs enable end-to-end tracing.

Risks & Mitigations
- Qdrant upgrade risk: schema/compat changes — snapshot storage, test on staging (Option B) or pin client (Option A).
- Parser performance: tree-sitter extraction cost — batch indexing, backpressure, and language gating.
- Logging volume: INFO expansions — sample and rotate; keep health server logs at `warning`.
- Dimension migration complexity: mitigated by automated migration script with rollback.

User Impact
- Search accuracy: improved top-k relevance via symbol metadata and correct dimensions.
- Fewer "silent failures": early errors are actionable with clear remedies.
- Better performance: SLOs ensure consistent response times.

Lifecycle
- Backward compatible for existing collections if sized at 768; otherwise use migration strategy.
- Future ADR may expand extraction to additional languages and hybrid ranking tuning.

## Rollout Plan

- Feature flags/env:
  - `EMBED_DIM` enforced; optional auto-detect from Nomic health.
  - `STRUCTURE_EXTRACTION_ENABLED=true` to gate tree-sitter phase.
- Deployment:
  - Dev: Pin client (Option A), validate E2E; Staging: consider server upgrade (Option B).
- Monitoring/Alerts:
  - Alert on `dimension_mismatch_errors > 0`, `embedding_timeouts spike`, or `collection_ensure_failed`.
- Rollback:
  - Revert client pin or container version; disable structure extraction via flag.

## Validation & Acceptance Criteria

- Connectivity & Dimension Detection:
  - MCP → Nomic: health check passes; automatic dimension detection succeeds; base URL logged.
  - MCP/Indexer → Qdrant: describe/create collection with detected dimensions and COSINE distance succeeds.
  - Dimension validation: runtime detection matches expected dimensions (768D default).
- Indexing/Upsert with Batch Processing:
  - 100 sample files indexed without dimension mismatch; upsert returns success.
  - Tree-sitter batch extraction completes within timeout limits; parsing errors logged appropriately.
  - Symbol extraction produces non-zero counts for classes/functions across supported languages.
- Hybrid Search Functionality:
  - Vector-only search returns results with similarity scores; no API errors.
  - Hybrid search combines vector and graph results with configurable scoring weights.
  - Context expansion through graph traversal (1-2 hops) enriches results appropriately.
  - Multiple ranking algorithms (weighted sum, RRF) produce different but valid result orderings.
- Search Quality:
  - Query returns results with expected payload fields including symbol metadata; latency within SLOs.
  - Hybrid results show measurable improvement in relevance over vector-only results.
- Enhanced Observability:
  - `/status` exposes queue depth, files processed, degraded services, hybrid search config.
  - Metrics exported for collections, extraction, dimension validation, and hybrid search performance.
  - Structured error messages provide actionable remediation steps.
  - Health checks validate all component dependencies with detailed status reporting.

## Alternatives Considered

- Keep client/server drift and rely on broad try/except with fallbacks
  - Rejected: masks root cause; degrades semantic search quality; increases MTTR.
- Switch vector DB (e.g., Milvus, Chroma) mid-stream
  - Rejected: Qdrant is already integrated and adequate; scope creep.
- Embed locally in indexer container
  - Rejected: increases container weight and complexity; current external embeddings service is healthy.

## Target Versions (2025-09)

- Qdrant Server: target v1.15.x (OSS) or pin clients to 1.12.x if staying on v1.12.4 server.
- Python Clients:
  - `qdrant-client`: 1.12.* (when server=1.12.4) or 1.15.* (when server=1.15.x).
  - `redis` (asyncio): 5.x; `arq`: 0.26.3.
  - `tree-sitter`: 0.25.1 (validate language grammars before upgrade).
  - `httpx`: ≥0.25 for timeouts/retries used in ADR.
- MCP SDK: latest stable (keep JSON-RPC STDIO compliance; stderr-only logging).
- Embeddings: runtime-detect dimension; default 768D unless model indicates otherwise.

## Compatibility Matrix

| Component | Server | Client | Notes |
| --- | --- | --- | --- |
| Qdrant | 1.12.4 | 1.12.* | Full compatibility; no named API drift. |
| Qdrant | 1.15.x | 1.15.* | Latest features; verify migrations/backups. |
| Redis (queue) | 7.x | redis-py 5.x + ARQ 0.26.3 | No-eviction; AOF on; Streams for DLQ. |
| Redis (cache) | 7.x | redis-py 5.x | LRU/allkeys; separate instance. |
| MCP | n/a | mcp SDK latest | STDIO JSON-RPC; log to stderr only. |
| tree-sitter | n/a | 0.25.1 | Requires compiled language libs or vendor grammars. |

## Reference Code Snippets

### Qdrant: Production-Ready Collection Management

```python
import qdrant_client
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue, UpdateStatus,
    CreateAlias, CollectionExistsException
)
import hashlib
import time
from typing import Dict, Any, List, Optional
from circuitbreaker import circuit
import logging

logger = logging.getLogger(__name__)

class QdrantService:
    def __init__(self):
        self.client = qdrant_client.QdrantClient(
            host="qdrant",
            port=6333,
            timeout=30,  # Production timeout
            grpc_port=6334,  # Use gRPC for better performance
            prefer_grpc=True
        )
        self.collection_name = "codebase_vectors"
        self.vector_dim = 768  # Nomic v2 dimension
        
    async def ensure_collection(self) -> bool:
        """Ensure collection exists with proper error handling"""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if not exists:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.vector_dim,
                            distance=Distance.COSINE,
                            on_disk=True  # Enable disk storage for large collections
                        )
                    },
                    on_disk_payload=True,  # Store payload on disk
                    optimizers_config={
                        "default_segment_number": 8,  # Optimize for parallel processing
                        "indexing_threshold": 20000,  # Delay indexing for batch inserts
                    }
                )
                logger.info(f"Created collection: {self.collection_name}")
                return True
            
            # Validate existing collection
            info = await self.client.get_collection(self.collection_name)
            if info.config.params.vectors.get("dense").size != self.vector_dim:
                raise ValueError(f"Collection dimension mismatch: expected {self.vector_dim}, got {info.config.params.vectors.get('dense').size}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    def generate_deterministic_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate deterministic ID for idempotent operations"""
        # Use file path + function name for code, URL for docs
        if "file_path" in metadata and "function_name" in metadata:
            id_source = f"{metadata['file_path']}:{metadata['function_name']}"
        elif "url" in metadata:
            id_source = metadata["url"]
        else:
            # Fallback to content hash
            id_source = content[:500]  # Use first 500 chars
            
        return hashlib.sha256(id_source.encode()).hexdigest()[:16]
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def upsert_vectors(self, vectors: List[List[float]], 
                            metadata: List[Dict[str, Any]], 
                            contents: List[str]) -> bool:
        """Upsert vectors with production error handling"""
        try:
            points = []
            for vector, meta, content in zip(vectors, metadata, contents):
                point_id = self.generate_deterministic_id(content, meta)
                
                # CRITICAL: Named vector format for upserts
                points.append(PointStruct(
                    id=point_id,
                    vector={"dense": vector},  # Named vector, not bare list
                    payload={
                        **meta,
                        "content": content[:1000],  # Truncate for storage
                        "indexed_at": int(time.time()),
                        "vector_model": "nomic-v2"
                    }
                ))
            
            # Batch upsert with retry logic
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                result = await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True  # Wait for operation to complete
                )
                
                if result.status != UpdateStatus.COMPLETED:
                    raise Exception(f"Upsert failed: {result}")
                    
            logger.info(f"Successfully upserted {len(points)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Vector upsert failed: {e}")
            raise
    
    @circuit(failure_threshold=3, recovery_timeout=30)
    async def hybrid_search(self, 
                           query_vector: List[float],
                           query_text: str,
                           filters: Optional[Dict[str, Any]] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Production hybrid search with named vectors"""
        try:
            # Build filter conditions
            filter_conditions = []
            if filters:
                for key, value in filters.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            
            # CRITICAL: Named vector format for search
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_vector),  # Tuple format for named vectors
                query_filter=Filter(must=filter_conditions) if filter_conditions else None,
                limit=limit,
                with_payload=True,
                score_threshold=0.5  # Minimum similarity threshold
            )
            
            # Enrich results with graph context
            enriched_results = []
            for result in results:
                enriched_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "explanation": self._generate_explanation(result, query_text)
                })
                
            return enriched_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    def _generate_explanation(self, result: Any, query: str) -> str:
        """Generate search result explanation"""
        score_pct = int(result.score * 100)
        return f"Matched with {score_pct}% similarity to query about {query[:50]}..."
```

### Nomic Embedding Service: Production Patterns

```python
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from circuitbreaker import circuit
import backoff
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingType(Enum):
    CODE = "code"
    TEXT = "text"
    HYBRID = "hybrid"

@dataclass
class EmbeddingRequest:
    content: str
    embedding_type: EmbeddingType
    metadata: Dict[str, Any]
    correlation_id: str

class NomicEmbeddingService:
    def __init__(self, host: str = "embeddings", port: int = 8000):
        self.base_url = f"http://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = {}  # Simple in-memory cache
        self.model = "nomic-v2"
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, content: str, embedding_type: EmbeddingType) -> str:
        """Generate cache key for embeddings"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{embedding_type.value}:{content_hash}"
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def generate_embedding(self, request: EmbeddingRequest) -> List[float]:
        """Generate embedding with production resilience"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(request.content, request.embedding_type)
            if cache_key in self.cache:
                logger.debug(f"Cache hit for {request.correlation_id}")
                return self.cache[cache_key]
            
            # Prepare request based on type
            if request.embedding_type == EmbeddingType.CODE:
                prepared_content = self._prepare_code_content(request.content)
            elif request.embedding_type == EmbeddingType.TEXT:
                prepared_content = self._prepare_text_content(request.content)
            else:  # HYBRID
                prepared_content = self._prepare_hybrid_content(request.content, request.metadata)
            
            # Make API call
            async with self.session.post(
                f"{self.base_url}/embed",
                json={
                    "text": prepared_content,
                    "model": self.model,
                    "correlation_id": request.correlation_id
                },
                headers={"X-Correlation-ID": request.correlation_id}
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                embedding = data["embedding"]
                
                # Validate embedding dimension
                if len(embedding) != 768:
                    raise ValueError(f"Invalid embedding dimension: {len(embedding)}")
                
                # Cache the result
                self.cache[cache_key] = embedding
                
                # Log metrics
                logger.info(f"Generated {request.embedding_type.value} embedding",
                          extra={
                              "correlation_id": request.correlation_id,
                              "content_length": len(request.content),
                              "cache_hit": False
                          })
                
                return embedding
                
        except aiohttp.ClientError as e:
            logger.error(f"Embedding service error: {e}",
                        extra={"correlation_id": request.correlation_id})
            raise
        except Exception as e:
            logger.error(f"Unexpected embedding error: {e}",
                        extra={"correlation_id": request.correlation_id})
            raise
    
    def _prepare_code_content(self, content: str) -> str:
        """Prepare code content for embedding"""
        # Remove comments and normalize whitespace
        lines = content.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(('#', '//', '/*')):
                cleaned.append(stripped)
        return ' '.join(cleaned)[:2000]  # Truncate to model limit
    
    def _prepare_text_content(self, content: str) -> str:
        """Prepare text content for embedding"""
        # Normalize whitespace and truncate
        normalized = ' '.join(content.split())
        return normalized[:2000]
    
    def _prepare_hybrid_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Prepare hybrid content with metadata context"""
        # Combine content with relevant metadata
        context_parts = []
        
        if "file_path" in metadata:
            context_parts.append(f"File: {metadata['file_path']}")
        if "function_name" in metadata:
            context_parts.append(f"Function: {metadata['function_name']}")
        if "description" in metadata:
            context_parts.append(f"Description: {metadata['description']}")
            
        context = ' '.join(context_parts)
        combined = f"{context} Content: {content}"
        
        return combined[:2000]
    
    async def batch_embed(self, requests: List[EmbeddingRequest], 
                          batch_size: int = 10) -> List[List[float]]:
        """Batch embedding generation with parallel processing"""
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i+batch_size]
            
            # Process batch in parallel
            tasks = [self.generate_embedding(req) for req in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle failures
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding failed for {batch[idx].correlation_id}: {result}")
                    # Use zero vector as fallback
                    results.append([0.0] * 768)
                else:
                    results.append(result)
                    
        return results
```

### Neo4j Service: Production Graph Operations

```python
from neo4j import AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable, TransientError
import asyncio
from typing import Dict, Any, List, Optional
from circuitbreaker import circuit
import backoff
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=50,
            connection_acquisition_timeout=30,
            max_transaction_retry_time=30,
            keep_alive=True
        )
        
    async def close(self):
        await self.driver.close()
    
    @asynccontextmanager
    async def get_session(self, database: str = "neo4j"):
        """Get a database session with automatic cleanup"""
        session = self.driver.session(database=database)
        try:
            yield session
        finally:
            await session.close()
    
    @backoff.on_exception(
        backoff.expo,
        (ServiceUnavailable, TransientError),
        max_tries=3,
        max_time=30
    )
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def execute_query(self, query: str, params: Dict[str, Any] = None,
                           correlation_id: str = None) -> List[Dict[str, Any]]:
        """Execute query with production resilience"""
        try:
            # Determine access mode based on query type
            write_keywords = ('CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP')
            is_write = any(keyword in query.upper() for keyword in write_keywords)
            
            async with self.get_session() as session:
                if is_write:
                    result = await session.execute_write(
                        self._execute_transaction,
                        query,
                        params or {},
                        correlation_id
                    )
                else:
                    result = await session.execute_read(
                        self._execute_transaction,
                        query,
                        params or {},
                        correlation_id
                    )
                
                logger.info(f"Query executed successfully",
                          extra={
                              "correlation_id": correlation_id,
                              "is_write": is_write,
                              "result_count": len(result)
                          })
                
                return result
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}",
                        extra={
                            "correlation_id": correlation_id,
                            "query": query[:100]
                        })
            raise
    
    @staticmethod
    async def _execute_transaction(tx, query: str, params: Dict[str, Any],
                                  correlation_id: str) -> List[Dict[str, Any]]:
        """Execute transaction with result processing"""
        result = await tx.run(query, params)
        records = []
        async for record in result:
            records.append(dict(record))
        return records
    
    async def get_code_context(self, file_path: str, function_name: str,
                              correlation_id: str) -> Dict[str, Any]:
        """Get comprehensive code context from graph"""
        query = """
        MATCH (f:File {path: $file_path})
        MATCH (fn:Function {name: $function_name})-[:DEFINED_IN]->(f)
        
        // Get function details
        OPTIONAL MATCH (fn)-[:CALLS]->(called:Function)
        OPTIONAL MATCH (caller:Function)-[:CALLS]->(fn)
        OPTIONAL MATCH (fn)-[:USES]->(dep:Dependency)
        OPTIONAL MATCH (fn)-[:HAS_PARAMETER]->(param:Parameter)
        OPTIONAL MATCH (fn)-[:RETURNS]->(ret:ReturnType)
        
        // Get module context
        OPTIONAL MATCH (f)-[:BELONGS_TO]->(m:Module)
        OPTIONAL MATCH (m)-[:DEPENDS_ON]->(dm:Module)
        
        RETURN {
            function: fn {.*},
            file: f {.path, .language, .size},
            module: m {.name, .type},
            calls: collect(DISTINCT called.name),
            calledBy: collect(DISTINCT caller.name),
            dependencies: collect(DISTINCT dep.name),
            parameters: collect(DISTINCT param {.name, .type}),
            returnType: ret.type,
            moduleDependencies: collect(DISTINCT dm.name)
        } as context
        """
        
        results = await self.execute_query(
            query,
            {"file_path": file_path, "function_name": function_name},
            correlation_id
        )
        
        return results[0]["context"] if results else {}
    
    async def find_similar_code(self, pattern: str, limit: int = 10,
                                correlation_id: str = None) -> List[Dict[str, Any]]:
        """Find similar code patterns in the graph"""
        query = """
        CALL db.index.fulltext.queryNodes('code_search', $pattern)
        YIELD node, score
        WHERE score > 0.5
        
        MATCH (node)-[:DEFINED_IN]->(f:File)
        OPTIONAL MATCH (f)-[:BELONGS_TO]->(m:Module)
        
        RETURN {
            node: node {.*},
            file: f.path,
            module: m.name,
            score: score
        } as result
        ORDER BY score DESC
        LIMIT $limit
        """
        
        results = await self.execute_query(
            query,
            {"pattern": pattern, "limit": limit},
            correlation_id
        )
        
        return [r["result"] for r in results]
    
    async def create_indexes(self, correlation_id: str = None):
        """Create necessary indexes for performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
            "CREATE INDEX IF NOT EXISTS FOR (m:Module) ON (m.name)",
            "CREATE FULLTEXT INDEX code_search IF NOT EXISTS FOR (n:Function|Class|Module) ON EACH [n.name, n.description, n.content]"
        ]
        
        for index_query in indexes:
            try:
                await self.execute_query(index_query, correlation_id=correlation_id)
                logger.info(f"Index created: {index_query[:50]}...")
            except Exception as e:
                logger.warning(f"Index creation skipped (may already exist): {e}")
```

### Observability & Monitoring: Production Patterns

```python
import prometheus_client as prom
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc import trace_exporter, metrics_exporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
import structlog
from typing import Any, Dict, Optional
import time

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
embedding_latency = prom.Histogram(
    'embedding_latency_seconds',
    'Embedding generation latency',
    ['model', 'type']
)

search_latency = prom.Histogram(
    'search_latency_seconds',
    'Search query latency',
    ['search_type', 'collection']
)

cache_hits = prom.Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses = prom.Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

dimension_mismatches = prom.Counter(
    'dimension_mismatch_errors_total',
    'Total dimension mismatch errors'
)

# OpenTelemetry setup
resource = Resource(attributes={
    "service.name": "neural-tools",
    "service.version": "1.0.0",
})

# Tracing
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Metrics
metrics.set_meter_provider(MeterProvider(resource=resource))
meter = metrics.get_meter(__name__)

# Custom metrics
embedding_counter = meter.create_counter(
    "embeddings_generated",
    description="Number of embeddings generated",
    unit="1"
)

search_counter = meter.create_counter(
    "searches_performed",
    description="Number of searches performed",
    unit="1"
)

class ObservabilityMiddleware:
    """Middleware for comprehensive observability"""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        
    async def track_operation(self, operation_name: str, 
                             correlation_id: str,
                             metadata: Dict[str, Any] = None):
        """Track operation with distributed tracing"""
        with tracer.start_as_current_span(operation_name) as span:
            span.set_attribute("correlation_id", correlation_id)
            
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"app.{key}", str(value))
            
            start_time = time.time()
            
            try:
                yield span
                
                # Record success metrics
                duration = time.time() - start_time
                self.logger.info(
                    f"Operation completed: {operation_name}",
                    correlation_id=correlation_id,
                    duration=duration,
                    **metadata or {}
                )
                
            except Exception as e:
                # Record failure metrics
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                self.logger.error(
                    f"Operation failed: {operation_name}",
                    correlation_id=correlation_id,
                    error=str(e),
                    **metadata or {}
                )
                raise
    
    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache access metrics"""
        if hit:
            cache_hits.labels(cache_type=cache_type).inc()
        else:
            cache_misses.labels(cache_type=cache_type).inc()
    
    def record_latency(self, metric: prom.Histogram, labels: Dict[str, str], duration: float):
        """Record latency metrics"""
        metric.labels(**labels).observe(duration)

# Health check implementation
class HealthChecker:
    """Comprehensive health checking"""
    
    def __init__(self, neo4j_service, qdrant_service, embedding_service):
        self.neo4j = neo4j_service
        self.qdrant = qdrant_service
        self.embeddings = embedding_service
        
    async def check_all(self) -> Dict[str, Any]:
        """Check all service dependencies"""
        results = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check Neo4j
        try:
            await self.neo4j.execute_query("RETURN 1 as health")
            results["checks"]["neo4j"] = {"status": "healthy"}
        except Exception as e:
            results["checks"]["neo4j"] = {"status": "unhealthy", "error": str(e)}
            results["status"] = "degraded"
        
        # Check Qdrant
        try:
            collections = await self.qdrant.client.get_collections()
            results["checks"]["qdrant"] = {
                "status": "healthy",
                "collections": len(collections.collections)
            }
        except Exception as e:
            results["checks"]["qdrant"] = {"status": "unhealthy", "error": str(e)}
            results["status"] = "degraded"
        
        # Check Embeddings
        try:
            test_embedding = await self.embeddings.generate_embedding(
                EmbeddingRequest(
                    content="health check",
                    embedding_type=EmbeddingType.TEXT,
                    metadata={},
                    correlation_id="health-check"
                )
            )
            results["checks"]["embeddings"] = {
                "status": "healthy",
                "dimension": len(test_embedding)
            }
        except Exception as e:
            results["checks"]["embeddings"] = {"status": "unhealthy", "error": str(e)}
            results["status"] = "degraded"
        
        return results
```

### MCP STDIO: Strict JSON-RPC + stderr Logging

```python
import sys, logging
from mcp.server import Server, NotificationOptions
import mcp.server.stdio
from mcp.server.models import InitializationOptions

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
server = Server("neural-tools")

async def run_stdio():
    async with mcp.server.stdio.stdio_server() as (r, w):
        await server.run(
            r, w,
            InitializationOptions(
                server_name="neural-tools",
                server_version="2025.09",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
```

### Redis Streams: DLQ Pattern (Async)

```python
import asyncio, json, redis.asyncio as redis

STREAM = "l9:dlq:embedding"
GROUP = "dlq_consumers"
CONSUMER = "worker-1"

async def setup(r):
    try:
        await r.xgroup_create(STREAM, GROUP, id="$", mkstream=True)
    except redis.ResponseError:
        pass  # group exists

async def produce(r, payload):
    await r.xadd(STREAM, {"payload": json.dumps(payload)})

async def consume(r):
    while True:
        msgs = await r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=10, block=5000)
        for stream, entries in msgs or []:
            for msg_id, fields in entries:
                try:
                    payload = json.loads(fields.get("payload", "{}"))
                    # process...
                    await r.xack(STREAM, GROUP, msg_id)
                except Exception:
                    # leave unacked for retry/inspection
                    pass
```

### Tree-sitter: Enhanced Batch Extraction with Timeout (Python)

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tree_sitter import Language, Parser
from typing import List, Tuple, Dict, Optional

LANG_SO = "/app/build/my-languages.so"  # built from tree-sitter-python, etc.
PY_LANG = Language(LANG_SO, "python")

async def extract_symbols_batch(files: List[Tuple[str, str]], 
                               batch_size: int = 10,
                               timeout_per_file: float = 5.0) -> Dict[str, List[dict]]:
    """Extract symbols from multiple files with batch processing and timeout control."""
    results = {}
    executor = ThreadPoolExecutor(max_workers=4)
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        tasks = []
        
        for file_path, source in batch:
            task = asyncio.get_event_loop().run_in_executor(
                executor, 
                lambda: extract_symbols_with_timeout(file_path, source, timeout_per_file)
            )
            tasks.append(task)
        
        # Process batch with timeout
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (file_path, _), result in zip(batch, batch_results):
            if isinstance(result, Exception):
                results[file_path] = {"error": str(result), "symbols": []}
            else:
                results[file_path] = {"symbols": result, "error": None}
    
    return results

def extract_symbols_with_timeout(file_path: str, source: str, timeout: float) -> List[dict]:
    """Extract symbols with timeout protection."""
    try:
        parser = Parser()
        parser.set_language(PY_LANG)
        tree = parser.parse(source.encode())
        root = tree.root_node
        symbols = []
        
        def walk(n):
            if n.type in ("class_definition", "function_definition"):
                # Extract name from first child that's an identifier
                name = None
                for child in n.children:
                    if child.type == "identifier":
                        name = source[child.start_byte:child.end_byte]
                        break
                
                symbols.append({
                    "type": n.type,
                    "name": name or "unnamed",
                    "start_line": n.start_point[0] + 1,
                    "end_line": n.end_point[0] + 1,
                    "start_byte": n.start_byte,
                    "end_byte": n.end_byte
                })
            for ch in n.children:
                walk(ch)
        
        walk(root)
        return symbols
        
    except Exception as e:
        raise RuntimeError(f"Parse error in {file_path}: {e}")
```

### Nomic Embeddings: HTTPX Client (Runtime Dim Detection)

```python
import httpx

async def embed(base_url: str, texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=60)) as client:
        r = await client.post(f"{base_url}/embed", json={"inputs": texts, "normalize": True})
        r.raise_for_status()
        data = r.json()
        return data.get("embeddings", [])

async def detect_dim(base_url: str) -> int:
    """Detect embedding dimensions from Nomic service at runtime."""
    emb = await embed(base_url, ["dim-detect"])
    return len(emb[0]) if emb else 0

async def validate_dimensions(base_url: str, expected_dim: Optional[int] = None) -> Dict[str, Any]:
    """Validate embedding dimensions and return detailed info."""
    try:
        actual_dim = await detect_dim(base_url)
        status = "success"
        message = f"Detected {actual_dim} dimensions"
        
        if expected_dim and actual_dim != expected_dim:
            status = "mismatch"
            message = f"Expected {expected_dim}D but got {actual_dim}D"
            
        return {
            "status": status,
            "actual_dimensions": actual_dim,
            "expected_dimensions": expected_dim,
            "message": message
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "actual_dimensions": None,
            "expected_dimensions": expected_dim
        }
```

### Hybrid Search: Configurable Scoring and Context Expansion

```python
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class HybridSearchConfig:
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    expansion_hops: int = 2
    ranking_algorithm: str = "weighted_sum"  # weighted_sum, rrf, custom
    context_limit: int = 10

class HybridSearchEngine:
    def __init__(self, neo4j_driver, qdrant_client, config: HybridSearchConfig = None):
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.config = config or HybridSearchConfig()
    
    async def hybrid_search(self, query: str, limit: int = 5, 
                           expand_context: bool = True) -> Dict[str, Any]:
        """
        Perform hybrid search combining vector similarity with graph context.
        Based on Context7 GraphRAG patterns with enhanced scoring.
        """
        # Phase 1: Vector similarity search
        vector_results = await self._vector_search(query, limit * 2)
        
        if not expand_context:
            return {"results": vector_results, "type": "vector_only"}
        
        # Phase 2: Graph context expansion
        doc_ids = [doc["doc_id"] for doc in vector_results]
        graph_context = await self._expand_graph_context(doc_ids, self.config.expansion_hops)
        
        # Phase 3: Hybrid ranking with configurable scoring
        final_results = await self._rank_hybrid_results(
            vector_results, graph_context, self.config
        )
        
        return {
            "results": final_results[:limit],
            "type": "hybrid",
            "vector_count": len(vector_results),
            "graph_context_count": len(graph_context),
            "config": self.config.__dict__
        }
    
    async def _expand_graph_context(self, doc_ids: List[str], hops: int) -> List[Dict]:
        """Expand results with graph context using configurable hop distance."""
        with self.neo4j_driver.session() as session:
            cypher_query = f"""
            MATCH (d:Document)
            WHERE d.id IN $doc_ids
            OPTIONAL MATCH (d)-[:RELATED_TO*1..{hops}]->(related:Document)
            WITH related, d, count(*) as relevance_score
            WHERE related IS NOT NULL AND related.id NOT IN $doc_ids
            RETURN DISTINCT 
                related.id as doc_id,
                related.title as title,
                related.content as content,
                relevance_score,
                collect(DISTINCT d.title) as source_docs
            ORDER BY relevance_score DESC
            LIMIT $limit
            """
            
            result = session.run(cypher_query, 
                               doc_ids=doc_ids, 
                               limit=self.config.context_limit)
            
            return [
                {
                    "doc_id": record["doc_id"],
                    "title": record["title"], 
                    "content": record["content"],
                    "graph_score": record["relevance_score"],
                    "source_docs": record["source_docs"],
                    "type": "graph_expansion"
                }
                for record in result
            ]
    
    async def _rank_hybrid_results(self, vector_results: List[Dict], 
                                  graph_context: List[Dict], 
                                  config: HybridSearchConfig) -> List[Dict]:
        """Apply configurable hybrid ranking algorithms."""
        all_results = []
        
        # Add vector results with scores
        for i, result in enumerate(vector_results):
            result["vector_rank"] = i + 1
            result["vector_score"] = result.get("score", 0.0)
            result["type"] = "vector"
            all_results.append(result)
        
        # Add graph context results  
        for i, result in enumerate(graph_context):
            result["graph_rank"] = i + 1
            result["vector_score"] = 0.0  # No direct vector match
            all_results.append(result)
        
        # Apply ranking algorithm
        if config.ranking_algorithm == "weighted_sum":
            return self._weighted_sum_ranking(all_results, config)
        elif config.ranking_algorithm == "rrf":
            return self._rrf_ranking(all_results, config)
        else:
            return all_results  # Custom ranking can be implemented
    
    def _weighted_sum_ranking(self, results: List[Dict], config: HybridSearchConfig) -> List[Dict]:
        """Weighted sum ranking combining vector and graph scores."""
        for result in results:
            vector_score = result.get("vector_score", 0.0)
            graph_score = result.get("graph_score", 0.0)
            
            # Normalize scores to 0-1 range
            normalized_vector = min(vector_score, 1.0) if vector_score > 0 else 0.0
            normalized_graph = min(graph_score / 10.0, 1.0) if graph_score > 0 else 0.0
            
            # Calculate hybrid score
            result["hybrid_score"] = (
                config.vector_weight * normalized_vector + 
                config.graph_weight * normalized_graph
            )
        
        return sorted(results, key=lambda x: x["hybrid_score"], reverse=True)
    
    def _rrf_ranking(self, results: List[Dict], config: HybridSearchConfig, k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion (RRF) ranking algorithm."""
        for result in results:
            vector_rank = result.get("vector_rank", float('inf'))
            graph_rank = result.get("graph_rank", float('inf'))
            
            # RRF formula: 1 / (k + rank)
            vector_rrf = 1 / (k + vector_rank) if vector_rank != float('inf') else 0
            graph_rrf = 1 / (k + graph_rank) if graph_rank != float('inf') else 0
            
            result["hybrid_score"] = (
                config.vector_weight * vector_rrf + 
                config.graph_weight * graph_rrf
            )
        
        return sorted(results, key=lambda x: x["hybrid_score"], reverse=True)
```

## Blue/Green Deployment Strategy

For production deployments, implement blue/green pattern for zero-downtime updates:

### Infrastructure Setup
```yaml
# docker-compose.blue.yml
services:
  qdrant-blue:
    image: qdrant/qdrant:v1.12.4
    ports:
      - "6333:6333"
    environment:
      - CLUSTER_NAME=blue
    labels:
      - "deployment=blue"
      
  embeddings-blue:
    build: ./embeddings
    ports:
      - "8000:8000"
    labels:
      - "deployment=blue"

# docker-compose.green.yml  
services:
  qdrant-green:
    image: qdrant/qdrant:v1.15.4
    ports:
      - "6334:6333"  # Different port initially
    environment:
      - CLUSTER_NAME=green
    labels:
      - "deployment=green"
      
  embeddings-green:
    build: ./embeddings
    ports:
      - "8001:8000"  # Different port initially
    labels:
      - "deployment=green"
```

### Deployment Script
```python
import asyncio
import docker
from typing import Dict, Any
import time

class BlueGreenDeployer:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.health_checker = HealthChecker()
        
    async def deploy(self, target: str = "green"):
        """Execute blue/green deployment"""
        current = "blue" if target == "green" else "green"
        
        # Step 1: Deploy target environment
        print(f"Deploying {target} environment...")
        self.docker_client.api.compose.up(
            project_name=f"neural-{target}",
            files=[f"docker-compose.{target}.yml"]
        )
        
        # Step 2: Wait for health
        await self.wait_for_health(target)
        
        # Step 3: Warm cache
        await self.warm_cache(target)
        
        # Step 4: Run smoke tests
        if not await self.smoke_test(target):
            raise Exception(f"Smoke tests failed for {target}")
        
        # Step 5: Switch traffic
        await self.switch_traffic(current, target)
        
        # Step 6: Monitor for errors
        await self.monitor_deployment(target, duration=300)
        
        # Step 7: Cleanup old environment
        print(f"Deployment successful. Cleaning up {current}...")
        await asyncio.sleep(3600)  # Keep old env for 1 hour
        self.docker_client.api.compose.down(
            project_name=f"neural-{current}"
        )
    
    async def wait_for_health(self, env: str, timeout: int = 300):
        """Wait for environment to be healthy"""
        start = time.time()
        while time.time() - start < timeout:
            health = await self.health_checker.check_all()
            if health["status"] == "healthy":
                print(f"{env} environment is healthy")
                return True
            await asyncio.sleep(5)
        raise TimeoutError(f"{env} environment failed to become healthy")
    
    async def warm_cache(self, env: str):
        """Pre-warm caches in new environment"""
        # Warm embedding cache
        common_queries = [
            "search functionality",
            "authentication",
            "database connection",
            "error handling"
        ]
        
        for query in common_queries:
            await self.embeddings.generate_embedding(
                EmbeddingRequest(
                    content=query,
                    embedding_type=EmbeddingType.TEXT,
                    metadata={},
                    correlation_id=f"warmup-{env}"
                )
            )
    
    async def smoke_test(self, env: str) -> bool:
        """Run critical smoke tests"""
        tests = [
            self.test_embedding_generation,
            self.test_vector_search,
            self.test_graph_query
        ]
        
        for test in tests:
            try:
                await test(env)
            except Exception as e:
                print(f"Smoke test failed: {e}")
                return False
        
        return True
    
    async def switch_traffic(self, old: str, new: str):
        """Atomic traffic switch using load balancer or DNS"""
        # Example using nginx configuration
        nginx_config = f"""
        upstream neural_backend {{
            server neural-{new}:8000;
        }}
        """
        
        # Write new config and reload nginx
        with open("/etc/nginx/conf.d/neural.conf", "w") as f:
            f.write(nginx_config)
        
        os.system("nginx -s reload")
        print(f"Traffic switched from {old} to {new}")
    
    async def monitor_deployment(self, env: str, duration: int):
        """Monitor deployment for issues"""
        start = time.time()
        error_threshold = 0.01  # 1% error rate
        
        while time.time() - start < duration:
            metrics = await self.get_metrics(env)
            error_rate = metrics.get("error_rate", 0)
            
            if error_rate > error_threshold:
                print(f"High error rate detected: {error_rate}")
                # Automatic rollback
                await self.rollback()
                raise Exception("Deployment rolled back due to high error rate")
            
            await asyncio.sleep(10)
        
        print(f"Deployment monitoring complete. {env} is stable.")
```

### Rollback Procedure
```python
async def rollback(self):
    """Emergency rollback to previous environment"""
    # Step 1: Switch traffic back
    await self.switch_traffic("green", "blue")
    
    # Step 2: Alert team
    await self.send_alert(
        "Deployment rollback executed",
        severity="high"
    )
    
    # Step 3: Capture diagnostics
    diagnostics = await self.capture_diagnostics()
    
    # Step 4: Log incident
    await self.log_incident({
        "type": "deployment_rollback",
        "timestamp": time.time(),
        "diagnostics": diagnostics
    })
```

## Migration Checklist

Before deploying to production:

- [ ] **Data Backup**: Full backup of Neo4j and Qdrant collections
- [ ] **Version Compatibility**: Verify client/server version alignment
- [ ] **Dimension Validation**: Confirm 768D embeddings across all services
- [ ] **Network Configuration**: Validate inter-service connectivity
- [ ] **Resource Allocation**: Ensure adequate CPU/memory for indexing
- [ ] **Monitoring Setup**: Prometheus/Grafana dashboards configured
- [ ] **Alerting Rules**: PagerDuty/Slack alerts for critical metrics
- [ ] **Rollback Plan**: Tested rollback procedure with <5min RTO
- [ ] **Load Testing**: Validated performance under expected load
- [ ] **Security Scan**: No exposed credentials or vulnerable dependencies

## Preflight Runbook

```bash
#!/bin/bash
# Pre-deployment validation script

echo "=== Neural Tools Preflight Check ==="

# 1. Check Docker services
docker-compose ps | grep -E "(qdrant|neo4j|embeddings|redis)" || exit 1

# 2. Validate environment variables
required_vars=("NEO4J_URI" "EMBEDDING_SERVICE_HOST" "REDIS_URL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set"
        exit 1
    fi
done

# 3. Test connectivity
curl -f http://localhost:48000/health || exit 1
curl -f http://localhost:6333/metrics || exit 1

# 4. Check disk space
df -h | grep -E "([8-9][0-9]|100)%" && echo "WARNING: Low disk space"

# 5. Validate collection dimensions
python3 -c "
import asyncio
from qdrant_client import AsyncQdrantClient

async def check():
    client = AsyncQdrantClient(host='localhost', port=6333)
    info = await client.get_collection('codebase_vectors')
    dim = info.config.params.vectors.get('dense').size
    assert dim == 768, f'Expected 768D, got {dim}D'
    print(f'✓ Collection dimension: {dim}D')

asyncio.run(check())
" || exit 1

echo "=== Preflight checks passed ✓ ==="
```

## Standards Alignment (2025-09-09)

- Qdrant: Latest OSS release observed v1.15.4 (GitHub releases). Named vectors remain supported; cosine/dot distances recommended for normalized embeddings. Align clients to 1.12.x (pin) or upgrade server to 1.15.x with backups/migration.
- MCP: Model Context Protocol continues to use JSON-RPC with STDIO transport as a supported mode (modelcontextprotocol.io). Continue emitting strict JSON-RPC responses and avoid stdout logging conflicts.
- Redis/ARQ: ARQ latest on PyPI is 0.26.3; Streams remain recommended for DLQ patterns (redis.io docs). Keep separate Redis instances for queues vs cache; avoid eviction on queues.
- tree-sitter: Python binding latest on PyPI is 0.25.1. Our indexer currently pins ~0.21.x; upgrade only after validating grammar compatibility and performance.
- Embeddings: Public docs for Nomic embed models do not guarantee a fixed dimension across variants; detect dimension programmatically on first call and propagate to Qdrant collection size. 768D remains common for v1/v1.5-class models.

References

- Qdrant releases: https://github.com/qdrant/qdrant/releases/latest  
- Redis Streams docs: https://redis.io/docs/latest/develop/data-types/streams/  
- ARQ (PyPI): https://pypi.org/pypi/arq/json  
- tree-sitter (PyPI): https://pypi.org/pypi/tree-sitter/json  
- Model Context Protocol site: https://modelcontextprotocol.io/
