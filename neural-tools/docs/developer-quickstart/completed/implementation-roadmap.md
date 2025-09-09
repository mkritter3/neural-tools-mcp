# Neural Tools Implementation Roadmap (Sept 2025)

This roadmap consolidates current state, critical gaps, and a concrete, testable plan to get Neural Tools (GraphRAG + MCP + Indexer) to a reliable, developer-friendly baseline. It also includes Context7 “pre‑flight” lookup items for areas where Claude’s local knowledge may be stale (post‑Jan 2025 changes).

## Executive Summary

- System boots and has clear architecture, but several integration mismatches prevent core features (semantic search, deletion, stable init) from working end‑to‑end.
- P0 fixes are small and surgical: method mismatches, env consistency, and vector dimension alignment. These unblock core indexing/search immediately.
- P1 focuses on consolidation and config centralization to reduce drift. P2 adds resilience, tests, and docs.

## Current State (Validated)

- Indexer is monolithic (1.5k+ LOC) with multiple responsibilities; duplicate indexer implementations exist.
- ServiceContainer now creates wrapper services (Neo4jService, QdrantService), and async init flow is consistent with a boolean return used by callers.
- Qdrant wrapper exposes `search_vectors(...)`, not `search(...)`. MCP server calls the wrong method name.
- Indexer calls `qdrant.delete_points(...)` but QdrantService doesn’t implement it.
- Env drift: Qdrant wrapper expects `QDRANT_HTTP_PORT`; compose sets only `QDRANT_PORT` for the indexer container. `.mcp.json` sets HTTP port correctly.
- Tree‑sitter gracefully falls back when grammars aren’t installed.

Important dimension mismatch risk:
- QdrantService `ensure_collection(...)` defaults to 1536 dims.
- Embedding client’s fallback produces 768‑dim vectors. If the external Nomic Embed v2 container returns a different dim (e.g., 768/1024/1536), vector size will mismatch and break upserts/searches.

## Roadmap Overview

### P0 — Unblock Core Functionality (today–tomorrow)

1) Fix MCP→Qdrant method mismatch
- File: `src/neural_mcp/neural_server_stdio.py`
- Replace calls to `container.qdrant.search(...)` with `container.qdrant.search_vectors(...)`.
- Align parameters: `collection_name`, `query_vector`, `limit`, and optional filter/threshold.
- Acceptance: `semantic_code_search` returns JSON with non‑empty results after initial index.

2) Implement Qdrant deletion in service wrapper
- File: `src/servers/services/qdrant_service.py`
- Add `async def delete_points(self, collection_name, points_selector=None, filter=None) -> Dict[str, Any]` that supports:
  - Deleting by explicit point IDs (`PointIdsList`)
  - Deleting by filter (payload match)
- Acceptance: Deleting a file triggers point removal in Qdrant (either by known IDs or by payload filter), and `remove_file_from_index(...)` completes without errors.

3) Standardize Qdrant HTTP port env
- File: `docker-compose.yml` (service `l9-indexer`)
- Add `QDRANT_HTTP_PORT=6333` to environment or adjust wrapper to fall back to `QDRANT_PORT` when `QDRANT_HTTP_PORT` absent.
- Acceptance: QdrantService `initialize()` succeeds under Docker without overriding envs via `.mcp.json`.

4) Align vector dimensions end‑to‑end
- Files: `qdrant_service.py`, `nomic_service.py`, `indexer_service.py`
- Strategy:
  - On first `NomicEmbedClient.get_embeddings()`, capture `len(embeddings[0])` and store as `EMBED_DIM`.
  - Ensure `ensure_collection(collection_name, vector_size=EMBED_DIM)` uses the same dim (override default 1536).
  - Optional: add env override `EMBED_DIM` for deterministic setups.
- Acceptance: Upserts do not error with dimension mismatch; `search_vectors(...)` returns results.

5) Minimal regression tests (local)
- Add/expand quick smoke tests (no external network):
  - Qdrant initialize (HTTP port), ensure collection, upsert sample point with `EMBED_DIM`, search returns >=1.
  - Neo4j minimal cypher `RETURN 1 as test` via `Neo4jService`.
  - MCP `semantic_code_search` happy path using fallback embeddings (if Nomic external not available).
- Acceptance: All smoke tests run green under local (host) and Docker.

### P1 — Consolidation & Config (this week)

6) Centralize configuration
- Create a config module (e.g., `servers/config/runtime.py`) that resolves envs once, with clear precedence:
  - `.env` → docker-compose env → `.mcp.json` env → defaults.
- Include canonical ports: Neo4j (47687 ext/7687 int), Qdrant (46333 ext/6333 int), Redis (46379/6379), Indexer (48080/8080).
- Acceptance: Services import from one config source; compose/.mcp.json values feed into it.

7) De‑duplicate service implementations
- Remove (or quarantine) legacy and graphRAG duplicates once primary indexer is stable:
  - `src/graphrag/services/indexer_service.py`
  - `src/services_old_duplicate/indexer_service.py`
- Acceptance: One canonical indexer path; older files archived under `.archive/`.

8) Extract indexer concerns into focused helpers (lightweight)
- Break out helpers without large architectural churn:
  - `file_queue.py` (queue + throttling)
  - `chunking.py` (chunking, keyword extraction)
  - `graph_ops.py` (Neo4j cypher routines)
  - `vector_ops.py` (Qdrant upsert/search utilities)
- Acceptance: Main indexer shrinks, unit tests possible per helper.

### P2 — Resilience, Quality, Docs (next 1–2 weeks)

9) Graceful degradation improvements
- Persist basic file tracking while services are down and reconcile later (write‑behind).
- Add backpressure when queue depth approaches threshold; surface metrics.

10) Error taxonomy and structured logging
- Introduce an exception hierarchy and standardized error payloads across services.
- Expand Prometheus metrics for failure categories; add dashboards.

11) Test coverage
- Unit tests for Qdrant/Neo4j wrappers (mock clients), chunking, queue behavior, and MCP tools (contract tests for input/output shapes).
- Optional: integration tests with docker‑compose (behind a make target).

12) Documentation refresh
- Unify docs under `docs/` with quickstart, troubleshooting, ADRs for key decisions (e.g., named vectors, embed dim).

## Detailed Change List (File‑Level)

- `src/neural_mcp/neural_server_stdio.py`
  - Fix Qdrant call: `container.qdrant.search_vectors(...)` instead of `.search(...)`.
  - Ensure correct result handling (`[{id, score, payload}]`).

- `src/servers/services/qdrant_service.py`
  - Implement `delete_points(...)` supporting both `PointIdsList` and filter.
  - Add `get_embed_dim()` or accept an explicit `vector_size` across ensure/collection flows.
  - Consider interpreting `QDRANT_PORT` as HTTP fallback when `QDRANT_HTTP_PORT` unset.

- `docker-compose.yml`
  - In `l9-indexer` env, set `QDRANT_HTTP_PORT=6333`.

- `src/servers/services/indexer_service.py`
  - Ensure it passes correct collection name and dims to Qdrant.
  - Confirm deletion flow calls the new `delete_points(...)` signature.

## Context7 Pre‑Flight Lookup Checklist (Sept 2025)

Use Context7 to fetch the latest guidance and API deltas before making changes. Claude’s local knowledge is Jan 2025; items below may have changed.

Semantic search stack
- Qdrant v1.10+ features and APIs:
  - Verify named vectors + sparse vector hybrid usage and current best practices (HTTP vs gRPC defaults, auth, client timeouts).
  - Confirm `AsyncQdrantClient` semantics, return types, and any deprecations to `get_collection`, `upsert`, `search`, and deletion APIs.
  - Recommended collection configuration (HNSW params, optimizers) for mixed code snippets; memory limits and persistence tuning.
- Nomic Embed v2 (MoE) container:
  - Endpoint path (`/embed`) payload format, response schema, and default vector dimension (768/1024/1536?); batching and normalization flags.
  - Recommended container tags, resource requirements (CPU/GPU/RAM), and concurrency guidelines.
  - Licensing and updated usage limits, if any.
- Rerankers (cross‑encoders):
  - Current recommended models and tradeoffs (e.g., BAAI/bge‑reranker‑base vs newer 2025 models); token limits, latency, and quality.

Graph/driver layer
- Neo4j Python driver (5.2x):
  - Any API changes to `AsyncGraphDatabase`, transaction functions, and result summary fields.
  - TLS/auth changes and containerized best practices.

MCP protocol/server SDK
- MCP SDK changes since 2025‑06‑18 protocol:
  - Initialization/capabilities structure, streaming responses, notifications, and any reserved field name changes.
  - Error payload schema and best practices for transport‑safe logging (stderr only).

Parsing/Chunking
- tree‑sitter‑languages:
  - Updated prebuilt wheels, supported grammars, and per‑language quirks for JS/TS/Python; diagnosing missing grammar cases.
  - Memory/performance guidance for large monorepos; language‑specific chunking patterns for code.

DevOps & Performance
- Container images (Neo4j/Qdrant/Redis/Nomic): updated recommended versions and health probes.
- Observability: Prometheus exporters, metric names that changed post‑Jan 2025, and canonical Grafana dashboards for indexing/search services.

## Operational Pre‑Flight (Local & Docker)

Before running indexing/search end‑to‑end, verify:
- Ports/envs
  - `NEO4J_URI` points to compose internal (`bolt://neo4j:7687`) in containers; `bolt://localhost:47687` on host/MCP.
  - Qdrant HTTP: `QDRANT_HTTP_PORT=6333` in containers; host MCP uses `46333`.
  - Indexer sidecar: `48080` → `8080` mapped; health returns 200.
- Embeddings
  - Confirm embedding dimension from Nomic endpoint or fallback; set/propagate `EMBED_DIM` accordingly.
  - Ensure Qdrant collection vector size matches `EMBED_DIM`.
- Permissions & paths
  - Indexer has read‑only mount of workspace and writable state/log mounts.
  - `.mcp.json` `PYTHONPATH` includes `neural-tools/src`.

## Validation & Test Plan

1) Unit‑style smoke tests (host)
- QdrantService: init, ensure collection (EMBED_DIM), upsert 1–2 points, search returns >=1.
- Neo4jService: `RETURN 1 as test` success; simple MERGE/SET.
- NomicEmbedClient: fallback embeddings size equals `EMBED_DIM`.

2) End‑to‑end (Docker)
- `docker-compose up -d neo4j qdrant redis`
- Start external Nomic container (or rely on fallback).
- `docker-compose up -d l9-indexer` → wait healthy.
- Confirm `/status` shows increasing `files_processed` after a small change.

3) MCP tools
- `semantic_code_search`: returns list with file paths/snippets.
- `indexer_status`: shows connected with realistic metrics.
- Optional: hybrid search returns results (if graph+vectors available).

## Risks & Mitigations

- Vector dim mismatch: enforce `EMBED_DIM` propagation; add a hard assert on first upsert vs collection.
- Port drift: centralize config; fail fast with clear error if Qdrant HTTP port looks wrong.
- Duplicate implementations: agree on canonical files before widespread refactors.
- Large repos & memory: ensure queue backpressure, chunk size control, and log/metric visibility.

## Appendix: Files To Touch

- `neural-tools/src/neural_mcp/neural_server_stdio.py` (Qdrant search call)
- `neural-tools/src/servers/services/qdrant_service.py` (delete_points, dim alignment)
- `neural-tools/src/servers/services/indexer_service.py` (use new deletion method; ensure dim/collection usage)
- `docker-compose.yml` (QDRANT_HTTP_PORT for indexer)
- Optional: `servers/config/runtime.py` (new) for centralized env resolution

— End —

