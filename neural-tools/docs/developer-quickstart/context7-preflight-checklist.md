# Context7 Pre‑Flight Checklist (Sept 2025)

Purpose: Run these lookups in Context7 before implementing, testing, or diagnosing Neural Tools. Claude’s built‑in knowledge is Jan 2025; this checklist targets changes and best practices since then across Qdrant, Nomic Embed v2, Neo4j driver, MCP SDK, tree‑sitter, and observability.

Use pattern: For each item, prefer vendor docs, official SDK repos, release notes, and migration guides. Save links + snippets into your working memory for accurate references during coding.

## Environment Snapshot (collect first)
- Project paths: repo root, `neural-tools/src`, `.mcp.json`, `docker-compose.yml`.
- Runtime ports (host vs container): Neo4j (47687/7687), Qdrant (46333/6333), Redis (46379/6379), Indexer (48080/8080).
- Active images/tags: Neo4j image, Qdrant image, Redis, Nomic Embed v2 container.
- Python/runtime: interpreter version, installed `qdrant-client` version, Neo4j driver version, MCP SDK version, tree‑sitter libs.

## Qdrant (v1.10+)
Lookup
- Async client semantics: `AsyncQdrantClient` methods that are async (must await), including `get_collections`, `get_collection`, `create_collection`, `upsert`, `search`, `delete`.
- Named vectors + sparse vectors: current recommended schema for hybrid dense+sparse, payload structure, and search invocation patterns.
- Deletion API: canonical ways to delete by IDs and by filter; expected request shapes for async HTTP client.
- Collection info/metrics: response schema fields for vector size, distance, points count; changes since early 2025.
- Performance defaults: HNSW parameters, optimizer config, memmap thresholds, persistence and compaction recommendations for code snippet workloads.
- HTTP vs gRPC guidance in containers: default ports, stability, timeouts, and when to prefer HTTP.
- Client timeouts/retry patterns: recommended values for `httpx` transport when running inside IDE/agent.
Deliverables
- Confirmed API signatures and return shapes to align wrapper methods (search, ensure_collection, delete_points, get_collection_info).
- Dimensionality handling: explicit doc on vector size validation and error messages on mismatch.

## Nomic Embed v2 (MoE)
Lookup
- Endpoint schema: path (`/embed`), request body fields (`inputs`, `normalize`, batch size), response JSON shape (`embeddings`, `model`, `usage`).
- Default embedding dimension and model variants; any changes since Jan 2025 (e.g., 768/1024/1536).
- Container recommendations: official image/tag, resource (CPU/GPU/RAM), threading/concurrency, rate limits.
- Error handling: throttle/backoff guidance; network timeouts.
- Licensing/usage updates.
Deliverables
- Exact embedding dimension; confirm normalization behavior; recommended batch size.
- Runtime health endpoints and readiness probes (if available).

## Neo4j Python Driver (5.2x)
Lookup
- `AsyncGraphDatabase.driver` connection examples; transaction function patterns (`execute_read`, `execute_write`).
- Result summary fields used by code; any renames/deprecations since early 2025.
- TLS/auth changes in Dockerized setups; APOC plugin notes.
- Connection URI vs host/port precedence; best practice for multi‑env configs.
Deliverables
- Confirmed minimal connectivity test (`RETURN 1 AS test`) and error taxonomy (AuthError, ServiceUnavailable).

## MCP SDK (post–2025‑06‑18 protocol)
Lookup
- InitializationOptions and capabilities changes; streaming responses and notifications.
- Tool schema conventions; error payload guidance; reserved fields.
- STDIO transport best practices (stderr logging rules, newline‑delimited JSON‑RPC).
Deliverables
- Validate current server bootstrap pattern and any updated examples.

## tree‑sitter / tree‑sitter‑languages
Lookup
- Supported grammars list and prebuilt wheels; install guidance for CI/containers.
- Known issues for JS/TS/Python parsing; memory/performance tips for large repos.
- Fallback strategies when grammars missing; API changes.
Deliverables
- Verified language coverage and recommended chunking patterns for code (functions, classes, imports).

## Rerankers (Cross‑Encoder Models)
Lookup
- 2025‑Q3 recommended models (quality vs latency), token limits, throughput.
- Licensing considerations and offline/cache strategies.
Deliverables
- Model short‑list with expected latency and accuracy tradeoffs for 1–5 candidate reranking.

## Observability / Ops
Lookup
- Prometheus client metrics best practices; metric names that changed since early 2025.
- Qdrant/Neo4j exporters; common dashboards for indexing/search services.
- Health/readiness endpoints patterns for sidecars; uvicorn/FastAPI versions.
Deliverables
- Dashboard starter set: queue depth, error categories, service health, dim mismatch alerts.

## Docker / Compose
Lookup
- Recommended images/tags for Neo4j, Qdrant, Redis, Nomic; healthcheck examples.
- Network, user, and security options (no‑new‑privileges, read‑only FS) updates.
Deliverables
- Updated compose snippets if images or flags changed.

## Security / Compliance
Lookup
- Secrets handling for DB creds and API keys; updated guidelines.
- Any new supply‑chain advisories for the libraries in use.
Deliverables
- Checklist for env var management and dependency pinning.

## Concrete Context7 Queries (copy/paste)
- “Qdrant AsyncQdrantClient v1.10+ methods list with await requirements and return schemas for get_collections, get_collection, create_collection, upsert, search, delete; named vectors + sparse hybrid examples.”
- “Qdrant deletion by filter and by IDs using async HTTP client; request/response examples.”
- “Nomic Embed v2 MoE official container docs: /embed request/response schema, default embedding dimension in 2025‑Q3, recommended batch size, normalization flag.”
- “Neo4j Python driver 5.2x async patterns: execute_read/execute_write usage, result summary fields, typical errors; Docker TLS/auth notes.”
- “MCP server SDK 2025 updates since 2025‑06‑18: initialization/capabilities, streaming responses, error payload format; stdio logging rules.”
- “tree‑sitter‑languages 2025‑Q3 supported grammars and installation guidance; performance guidance for large repositories.”
- “Prometheus/Grafana templates for indexing pipelines: queue depth, error categories, service health.”
- “Top cross‑encoder rerankers 2025‑Q3 with latency/quality benchmarks and licenses.”

## How to Apply Results
- Update wrapper method signatures and return parsing based on verified SDK responses.
- Set/propagate `EMBED_DIM` based on confirmed Nomic dimension and enforce in Qdrant collection creation.
- Align Neo4j connection config precedence; prefer `NEO4J_URI` where available.
- Confirm MCP tool schemas and error formatting to match SDK best practices.

— End —
