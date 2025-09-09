# Neural Tools Problems & Solutions Roadmap (Sept 2025)

## Executive Summary
- Overall system starts and responds via MCP, but several integration defects and config drifts block reliable indexing and search.
- The most disruptive issues are: uninitialized indexer queue, Qdrant async misuse, Neo4j connection mismatch, MCP→Qdrant API mismatch, missing Qdrant deletion in wrapper, and potential embedding dimension drift.
- Plan prioritizes P0 surgical fixes with explicit acceptance criteria, then P1 consolidation and P2 quality/observability.

Related: For up‑to‑date external knowledge, run the checks in `context7-preflight-checklist.md` before implementing fixes.

## Triage Overview
- P0 (Critical, unblock functionality):
  - Indexer queue not initialized (NoneType .get/.put errors)
  - Qdrant async/await misuse (collections attribute access on coroutine)
  - MCP→Qdrant API mismatch (`search` vs `search_vectors`)
  - Neo4j connection mismatch (URI vs host/port envs)
  - Qdrant deletion not implemented (indexer removal path)
  - Embedding dimension mismatch risk (collection vs vectors)
- P1 (High, consolidation):
  - Centralize configuration and ports
  - Remove duplicate service implementations
  - Extract focused helpers from monolithic indexer
  - Fallback indexing persistence and reconciliation
  - Tree-sitter coverage/validation improvements
- P2 (Medium, quality):
  - Error taxonomy + structured logging
  - Observability dashboards + alerts
  - Contract/integration tests and ADRs

## P0 Problems and Solutions

### Indexer Queue Not Initialized
- Symptom: `'NoneType' object has no attribute 'get'` / `'put'` in indexer.
- Impact: Queue processing loop fails; reindex requests fail; indexing stalls.
- Likely Root Cause: `self.pending_queue` set to `None` at init; only initialized in `run_indexer(...)`. The container entrypoint uses `start_monitoring()` path which does not initialize the queue.
- Evidence:
  - Declaration: `self.pending_queue: asyncio.Queue = None` (indexer_service.py ~139)
  - Usage without guard: `.get()` (~1157), `.put()` (~1205)
  - Queue init only in `run_indexer(...)`: `indexer.pending_queue = asyncio.Queue(...)` (~1504)
- Proposed Fix:
  - Initialize `self.pending_queue` in the indexer constructor or at the beginning of `start_monitoring()`.
  - Guard all queue usages (`_queue_change`, `process_queue`) with early init or explicit error message.
  - Ensure docker sidecar runner also sets queue before starting loops.
- Acceptance:
  - No NoneType errors when starting sidecar; reindex endpoint enqueues successfully; queue depth metrics > 0 after changes.
- Tests:
  - Unit: create indexer, call `_queue_change` before services init; assert no exceptions.
  - E2E: touch a file, see it enqueued and processed.

### Qdrant Async/Await Misuse
- Symptom: `"'coroutine' object has no attribute 'collections'"`.
- Impact: Collection operations fail intermittently; initialization stability suffers.
- Likely Root Cause: An async Qdrant client method (e.g., `get_collections()` / `get_collection()`) is called without `await`, then code accesses `.collections` from the coroutine.
- Evidence:
  - Wrapper paths appear awaited, but external callsites may still use async client directly without await.
- Proposed Fix:
  - Identify exact location from stack trace; ensure all AsyncQdrantClient calls are awaited.
  - Standardize: callers should use `QdrantService` methods only; avoid passing raw `AsyncQdrantClient` around.
- Acceptance:
  - No occurrences of the error across init/ensure/use flows for 15 minutes under load.
- Tests:
  - Mock AsyncQdrantClient; assert awaited calls; run smoke init multiple times.

### MCP→Qdrant API Mismatch
- Symptom: MCP `semantic_code_search` uses `container.qdrant.search(...)` which does not exist; wrapper exposes `search_vectors(...)`.
- Impact: Semantic search tool fails even when Qdrant is healthy.
- Proposed Fix:
  - Replace MCP call with `search_vectors(...)` and adapt return shape (id, score, payload).
- Acceptance:
  - MCP `semantic_code_search` returns non-empty results after initial indexing; no attribute errors in logs.
- Tests:
  - Contract test for tool: input `{query, limit}` → structured list with `file_path` + `snippet`.

### Neo4j Connection Mismatch
- Symptom: MCP reports Neo4j not connected; GraphRAG ops fail.
- Impact: Graph features unavailable; hybrid search degrades/fails.
- Root Cause:
  - `ServiceContainer.ensure_neo4j_client()` honors `NEO4J_URI`.
  - `Neo4jService.AsyncNeo4jClient` uses `NEO4J_HOST`/`NEO4J_PORT` (defaults not aligned with `.mcp.json`).
- Proposed Fix (choose one):
  - Prefer `NEO4J_URI` in `AsyncNeo4jClient` when present; otherwise fall back to host/port.
  - Or set `NEO4J_HOST=localhost` in MCP env to match `NEO4J_PORT=47687`.
- Acceptance:
  - `Neo4jService.initialize()` returns success; `RETURN 1 as test` succeeds; MCP graph queries no longer report unavailable.
- Tests:
  - Unit: init service using `NEO4J_URI` only; validate URI used.
  - E2E: simple MERGE/SET runs without error.

### Qdrant Deletion Not Implemented in Wrapper
- Symptom: Indexer removal flow calls `qdrant.delete_points(...)`, but wrapper has no such method.
- Impact: Deleted files remain in vector store; index integrity drifts.
- Proposed Fix:
  - Implement `delete_points(collection_name, points_selector=None, filter=None)` in `QdrantService`, supporting both ID list and filter payload.
- Acceptance:
  - Removing a file removes corresponding points (by IDs if known, or filter on payload).
- Tests:
  - Upsert N points → delete by IDs → count decreases; delete by filter → removed as expected.

### Embedding Dimension Mismatch Risk
- Symptom: Potential upsert/search failures due to vector size mismatch (fallback embeddings 768 vs default collection 1536).
- Impact: Upsert errors, empty or invalid search results.
- Proposed Fix:
-  - Detect embedding dimension on first call; ensure Qdrant collection uses the same dim (or read `EMBED_DIM` env if provided).
-  - Validate on upsert: hard assert if dims mismatch; log clear remediation.
- Acceptance:
  - Upserts succeed; `search_vectors` works; dimension mismatch logged early if misconfigured.
- Tests:
  - Simulate 768-d embeddings; ensure collection creation uses 768 and accepts writes.

## P1 Consolidation and Architecture

### Centralize Configuration
- Problem: Config drift across compose, MCP, service wrappers; port and naming inconsistencies.
- Solution:
  - Add a runtime config module with a single source of truth and clear precedence; normalize `QDRANT_HTTP_PORT` fallback to `QDRANT_PORT`.
- Acceptance: Services import from central config; changing one env updates behavior consistently.

### Remove Duplicate Service Implementations
- Problem: Three indexer variants and duplicate retrievers cause confusion and maintenance overhead.
- Solution: Choose canonical implementations; archive others under `.archive/`.
- Acceptance: One indexer path; references updated; no duplicate imports.

### Extract Focused Helpers from Indexer
- Problem: Monolithic indexer mixes queue, chunking, graph ops, vector ops.
- Solution: Light refactor to helpers (`file_queue.py`, `chunking.py`, `graph_ops.py`, `vector_ops.py`) keeping external API stable.
- Acceptance: Indexer core shrinks; unit tests possible per helper.

### Fallback Indexing Persistence
- Problem: In degraded mode, tracking is non-persistent.
- Solution: Persist seen-file state and reconcile when services recover.
- Acceptance: After outage, reconciliation indexes missed files without manual intervention.

### Tree-sitter Coverage/Validation
- Problem: Inconsistent grammar availability and error handling.
- Solution: Validate grammars on start; per-language fallbacks; clearer logs.
- Acceptance: Missing grammar surfaces as a clear, actionable warning; non-blocking.

## P2 Quality, Observability, Docs

### Error Taxonomy + Structured Logging
- Problem: Mixed patterns reduce debuggability.
- Solution: Introduce error classes and consistent payloads; unify logging format and levels.
- Acceptance: Aggregated error metrics by category; actionable logs.

### Observability
- Solution: Prometheus metrics for queues, errors, service health; Grafana dashboards; alerts on backlogs and service down.
- Acceptance: Dashboards reflect live state; alerts tested.

### Tests and ADRs
- Solution: Contract tests for MCP tools; unit tests for services/helpers; ADRs for key choices (embedding dims, neo4j URI precedence, qdrant HTTP vs gRPC).
- Acceptance: CI green; documented decisions in `docs/architecture/`.

## Context7 Pre‑Flight Lookups (Post‑Jan 2025)
- Qdrant v1.10+ APIs: named vectors, HTTP vs gRPC, deletion API patterns, collection stats responses.
- Nomic Embed v2 (MoE): endpoint schema, recommended dims, batching, containers/tags, resource guidance.
- Neo4j Python driver 5.2x: async session/transaction patterns, auth/TLS updates.
- MCP SDK since 2025‑06‑18: init/capabilities changes, streaming responses, error payload conventions.
- tree-sitter-languages: grammar set, wheel availability, memory/perf tips for large monorepos.
- Reranker landscape: updated 2025 recs for cross-encoders and latency/quality tradeoffs.

## Validation Plan (No Code)
- Reproduce queue errors by calling `_queue_change` before queue init; confirm NoneType exceptions.
- Capture stack for “coroutine has no attribute collections” to locate the exact call site.
- Verify MCP envs for Neo4j (`NEO4J_URI`, `NEO4J_HOST/PORT`) and Qdrant (`QDRANT_HTTP_PORT` vs `QDRANT_PORT`).
- Confirm embedding dim at runtime; compare to collection config.

## Rollout & Risk Management
- Sequence P0 patches by dependency: queue init → MCP→Qdrant call → Qdrant delete → Neo4j URI → dim enforcement.
- Deploy with verbose logging and metrics; watch for regression in sidecar status and MCP tools.
- Risks: dim drift, port drift, legacy code paths. Mitigate via asserts, central config, and retiring duplicates.

## Acceptance Metrics (Post‑P0)
- Indexer: 0 queue NoneType errors; steady increase in processed files; queue never > 80% threshold for > 5 min.
- MCP: `semantic_code_search` success rate > 95% across 20 trials; no attribute errors.
- Qdrant: initialize + ensure + upsert + search pass; deletion removes expected points.
- Neo4j: init success; basic graph write/read succeed.

## File Touch Map (For Implementation Phase)
- `neural-tools/src/servers/services/indexer_service.py` (queue init guards; deletion flow)
- `neural-tools/docker/scripts/indexer-entrypoint.py` (ensure queue init path)
- `neural-tools/src/neural_mcp/neural_server_stdio.py` (use `search_vectors`)
- `neural-tools/src/servers/services/qdrant_service.py` (implement `delete_points`, dim handling)
- `docker-compose.yml` (`QDRANT_HTTP_PORT` for indexer)
- Optional: `servers/config/runtime.py` (central config)

— End —
