# L9 Full Stack Manifest (Claude RAG + Indexer + MCP)

This document inventories the complete stack (containers, MCP server, configs, key code paths), clarifies what is canonical vs. legacy, and provides a reproducible runbook and test checklist.

## Overview
- Goal: Clean, reproducible, multi‑project RAG system for Claude.
- Integration pattern:
  - MCP (STDIO) launched by Claude Desktop via `.mcp.json`.
  - Lean Indexer Sidecar (no local ML) calls external Nomic v2 embeddings over HTTP.
  - Data stores: Neo4j (graph) + Qdrant (vectors).
  - Optional monitoring: Prometheus + Grafana.

## Components & Key Files

- MCP Server (STDIO)
  - Config: `.mcp.json`
  - Entrypoint: `neural-tools/run_mcp_server.py`
  - Canonical server: `neural-tools/src/neural_mcp/neural_server_stdio.py`
    - Tools: `neural_system_status`, `semantic_code_search`, `graphrag_hybrid_search`, `project_understanding`, `indexer_status`, `reindex_path`, `neural_tools_help`
    - Prescriptive validation and normalized args for robust usage.
  - Deprecated wrapper (do not extend): `neural-tools/src/servers/neural_server_stdio.py`

- Indexer Sidecar (lean; no ML deps)
  - Dockerfile: `docker/Dockerfile.indexer`
  - Entrypoint: `docker/scripts/indexer-entrypoint.py`
  - Indexer core: `neural-tools/src/servers/services/indexer_service.py`
  - Responsibilities: file watching, chunking, calling embeddings over HTTP, upserting to Qdrant, updating Neo4j.
  - Endpoints: `GET /health`, `GET /metrics`, `GET /status`, `POST /reindex-path`

- External Embeddings (Nomic v2)
  - Running container (example): `neural-flow-nomic-v2-production`
  - Accessed via HTTP: `EMBEDDING_SERVICE_HOST` + `EMBEDDING_SERVICE_PORT`
  - Client: `neural-tools/src/servers/services/nomic_service.py` (uses `httpx`)

- GraphRAG Main Service
  - Dockerfile: `docker/Dockerfile`
  - Compose service: `l9-graphrag`
  - Uses: Neo4j, Qdrant, and external embeddings via env

- Data Stores
  - Neo4j: Compose service `neo4j` with persistent volumes; Bolt at `7687`.
  - Qdrant: Compose service `qdrant` with persistent volumes; HTTP `6333`, gRPC `6334`.

- Monitoring (optional)
  - Prometheus: `prometheus` service
  - Grafana: `grafana` service

## Docker Compose Services
File: `docker-compose.yml`

- `l9-graphrag`: Main GraphRAG/API service. Ports: `3000` (app), `9090` (metrics).
- `l9-indexer`: Indexer sidecar. Port: `8080` (health/metrics/control).
- `neo4j`: Graph DB. Ports: `7474` (HTTP), `7687` (Bolt).
- `qdrant`: Vector DB. Ports: `6333` (HTTP), `6334` (gRPC).
- `redis`: Caching helper (optional).
- `embeddings`: Optional stub; actual production container may be different (see “External Embeddings”).
- `prometheus`, `grafana`: Optional monitoring.

Networking: default network is `l9-graphrag-network`.

## Environment Variables (Essentials)
- Embeddings
  - `EMBEDDING_SERVICE_HOST` (e.g., `neural-flow-nomic-v2-production` or `host.docker.internal`)
  - `EMBEDDING_SERVICE_PORT` (e.g., `8000`)
- Qdrant
  - `QDRANT_HOST` (e.g., `qdrant` or `localhost`)
  - `QDRANT_PORT` (default `6333`)
- Neo4j
  - `NEO4J_URI` (e.g., `bolt://neo4j:7687`)
  - `NEO4J_USERNAME` (e.g., `neo4j`)
  - `NEO4J_PASSWORD` (e.g., `graphrag-password`)
- Indexer
  - `PROJECT_NAME` (default: `default`)
  - `PROJECT_PATH` (default: `/workspace`)
  - `INITIAL_INDEX` (`true|false`)
  - `WATCHDOG_FORCE_POLLING=1` (recommended for Docker Desktop bind mounts)

## Volumes & Persistence
- Neo4j data/logs/import: named volumes.
- Qdrant storage: named volume.
- Indexer state/logs: named volumes mounted to `/app/state` and `/app/logs`.
  - IMPORTANT: Do not overlay tmpfs on these paths if persistence is required.

## MCP Configuration
- `.mcp.json`
  - Declares `neural-tools` server with `type: stdio`.
  - Command: absolute path to `neural-tools/run_mcp_server.py`.
- Claude Desktop reads `.mcp.json` and launches the STDIO server.

## Canonical vs. Legacy
- Canonical (use these):
  - MCP: `neural-tools/src/neural_mcp/neural_server_stdio.py` (via `run_mcp_server.py`)
  - Indexer: `docker/Dockerfile.indexer`, `docker/scripts/indexer-entrypoint.py`, `neural-tools/src/servers/services/indexer_service.py`
  - Service clients: `neural-tools/src/servers/services/*` (Qdrant, Neo4j, Hybrid Retriever, etc.)

- Legacy/Deprecated (avoid for new work):
  - MCP Proxy: `neural-tools/src/servers/mcp_proxy/*` (only needed for non‑STDIO/remote scenarios)
  - Legacy MCP wrapper: `neural-tools/src/servers/neural_server_stdio.py` (wrapper; do not add tools here)
  - Duplicated/old services: `neural-tools/src/services_old_duplicate/*`

## Reproducible Runbook (Fewest Steps)
1) Prerequisites
- Docker + Docker Compose
- Claude Desktop
- A running embeddings container (e.g., `neural-flow-nomic-v2-production`)

2) Networking: attach embeddings to stack network (if needed)
- `docker network connect l9-graphrag-network neural-flow-nomic-v2-production`

3) Configure `.env` (example)
```
PROJECT_NAME=default
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=graphrag-password
QDRANT_HOST=qdrant
QDRANT_PORT=6333
EMBEDDING_SERVICE_HOST=neural-flow-nomic-v2-production
EMBEDDING_SERVICE_PORT=8000
INITIAL_INDEX=true
```

4) Bring up the stack (omit `embeddings` service if using your external container)
- `docker compose up -d neo4j qdrant l9-graphrag l9-indexer`

5) MCP setup (Claude Desktop)
- Ensure `.mcp.json` points to `neural-tools/run_mcp_server.py`
- Restart Claude Desktop to reload tools

6) Smoke tests
- Health: `curl http://localhost:8080/health` (indexer)
- Metrics: `curl http://localhost:8080/metrics` (indexer)
- Status: `curl http://localhost:8080/status`
- MCP tools:
  - `neural_tools_help` → usage/constraints visible
  - `semantic_code_search` with a normal query → success (no degraded fallback)
  - `reindex_path` with a valid path → “enqueued”

## Multi‑Project Usage
- Pattern A: One sidecar per project (recommended)
  - For each project, run an indexer with unique `PROJECT_NAME` and a distinct workspace mount.
  - Qdrant collections: `project_<project_name>_code`
  - Neo4j labels/metadata: namespaced by project

- Pattern B: Multiple stacks (full isolation)
  - Duplicate compose (project‑specific names/network).

## Test Checklist (Comprehensive)
- Build & Start
  - Images build without ML deps in the sidecar
  - Compose up succeeds; Neo4j/Qdrant healthy
  - Indexer `/health` returns 200 within 60s
- Connectivity
  - Indexer → Embeddings: `curl http://<EMBEDDING_SERVICE_HOST>:<PORT>/health` returns 200 from inside indexer container (exec shell)
  - Indexer → Qdrant/Neo4j: simple ops succeed (collections list, `RETURN 1`)
- MCP Tools
  - `list_tools` includes all tools with tightened schemas
  - Negative args return prescriptive `validation_error` with `example` and `next_call`
  - `semantic_code_search` returns results; degraded mode handled if embeddings down
  - `reindex_path` returns “enqueued” and indexer logs show queue activity
- Persistence
  - Restart indexer container; state/logs remain present (no tmpfs overshadowing)
- Multi‑Project (optional)
  - Two indexers with different `PROJECT_NAME` run concurrently; collections are namespaced; MCP searches respect project context

## Ports (Defaults)
- MCP (STDIO): no network port (STDIO via Claude)
- Indexer Sidecar: `8080` (health/metrics/control)
- Neo4j: `7474` (HTTP), `7687` (Bolt)
- Qdrant: `6333` (HTTP), `6334` (gRPC)
- GraphRAG Service: `3000` (app), `9090` (metrics)
- Embeddings (Nomic v2): `8000` (example)

## Known Pitfalls & Notes
- Do not overlay `tmpfs` on `/app/state` or `/app/logs` if you need persistence; use named volumes only.
- On macOS/Windows, use `WATCHDOG_FORCE_POLLING=1` to avoid dropped file events on bind mounts.
- If MCP ever runs in a container, change MCP → sidecar URLs from `http://localhost:8080/...` to `http://l9-indexer:8080/...` (service DNS).
- If you keep the compose `embeddings` stub, ensure it matches your real image tag; otherwise, point `EMBEDDING_SERVICE_HOST` to your actual container and attach it to the network.

## Deprecation Actions
- Prefer `neural_mcp/neural_server_stdio.py` over any proxy/legacy servers.
- Treat `servers/mcp_proxy/*` and `services_old_duplicate/*` as legacy; do not extend.
- Keep `neural-tools/src/servers/neural_server_stdio.py` only as a wrapper; do not add tools there.

---
This manifest is the single source of truth for the full stack layout, connectivity, and operational patterns. Keep it updated alongside any stack changes to avoid drift or duplicate stacks.
