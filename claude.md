# L9 Engineering Contract

**Date: September 24, 2025 | MCP Protocol: 2025-06-18**
**Status: ADR-0096 COMPLETE - Robust Vector Search Working! 🎉**

**CRITICAL RULES:**
- Check Context7 for protocols before March 2025
- Never create parallel/duplicate stacks - integrate into existing architecture
- Edit existing files, don't create v2/v3/enhanced versions
- Commit to git regularly
- Verify everything - never assume
- Follow DRY, KISS, YAGNI, single-responsibility principles
- You MUST run pylance, ruff, and black checks after every Python code modify, enhancement or new code generation

**🚨 NEO4J DATA TYPES (ADR-0036):**
Only primitives: String, Long, Double, Boolean. NO Map{} or nested objects.
Flatten: `Map{statement->"import", line->11}` → `{import_statement:"import", import_line:11}`

**🚨 DOCKER CONFLICTS (ADR-0060) - CRITICAL FIX:**
Container naming uses timestamp+random to prevent 409 conflicts:
- Format: `indexer-{project}-{timestamp}-{random}`
- Redis distributed locking with 60s timeout
- Label-based discovery: `com.l9.project={name}`
- 7-day GC for stopped containers
- Falls back to local locks if Redis unavailable

**Prime Directive:** Truth > likeability. Correct me even if I prefer a different answer.

**Evidence Rule:** Verify time‑sensitive claims. Quote primary sources.

**Calibration:** End technical answers with `Confidence: NN%` + assumptions.

**E2E Thinking:** API→DB→Services→UI→Telemetry. 95% success gate. Small reversible steps.

## Mocking Policy
- Mocks for unit tests/local dev only
- NEVER mock in staging/prod - fail fast if deps unavailable
- Flag mock usage under `Mock Usage:` line

---

# L9 Neural GraphRAG MCP Architecture

**Version: L9 2025 Production | Protocol: MCP 2025-06-18**

## Core Architecture
1. **Single MCP Entrypoint** - All Claude interactions via one server
2. **Container-Host Bridge** - MCP on host→Docker services via ports
3. **Session Isolation** - Per-MCP-session resource pools
4. **Resilience** - Circuit breakers, rate limiting, health checks
5. **Direct Connections** - No proxy layers

## Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Neo4j | 47687 | GraphRAG + Vector Storage (HNSW) |
| ~~Qdrant~~ | ~~46333~~ | **NOT USED** - Neo4j handles vectors |
| Redis Cache | 46379 | Session cache |
| Redis Queue | 46380 | Task queue |
| Nomic | 48000 | Embeddings (Nomic Embed v2) |
| Indexer | 48100+ | Per-project (currently 48106)

## Configuration Priority (ADR-0037)
1. Environment variables (highest)
2. Config files (pyproject.toml, package.json)
3. Auto-detection (fallback)
4. Hard-coded defaults (last resort)

**Container→Host:** Use `host.docker.internal` not `localhost`

## Docker Setup & Management (ADR-0038)

### Active Docker Compose Files
- `docker-compose.yml` - Production stack (Neo4j, Redis, Nomic)
- `docker-compose.dev.yml` - Development overrides with debug logging
- `docker/README.md` - Complete Docker operations guide

### Image Management
- Production images: `service:production` tag
- Cleaned up 30+ old images (Sept 24, 2025)
- Removed obsolete Dockerfiles and monitoring configs
- Active containers: neo4j, redis-cache, redis-queue, nomic, indexers

## MCP Connection Pooling (L9 Conservative)
- Neo4j: 50 (min 5) - Complex queries + vector ops need headroom
- Redis: 25/15 - Lightweight ops

## Performance Requirements
- Concurrent Sessions: ≥15
- Success Rate: ≥95%
- Response Time: <500ms avg, <1000ms P95
- Throughput: ≥10 RPS

## Critical Fixes Implemented

### ✅ ADR-0029: Neo4j Project Isolation
- Every node has `{project: "name"}` property
- Composite constraints: `(project, path)`
- All queries filter by project

### ✅ ADR-0039: Collection Naming
- Single source of truth: `CollectionNamingManager`
- Format: `project-{name}` (no underscores)
- Applied to Neo4j node properties only (no Qdrant)

### ✅ ADR-0060: Container Conflicts (409 Errors)
- Unique names: `indexer-{project}-{timestamp}-{random}`
- Redis distributed locking (60s timeout)
- Label discovery not name-based
- Redis caching 30s TTL (P95 <5ms)
- **Test results: 100% pass, zero conflicts**

### ✅ ADR-0052: Automatic Indexer Init
- Zero-step indexing
- Auto project detection
- <30s to first index

### ✅ ADR-0031: Canonical Knowledge
- Pattern-based metadata extraction
- PRISM scoring for file importance
- Git metadata tracking
- 20+ metadata fields

### ✅ ADR-20/21: Custom GraphRAG Schemas
- Per-project schemas (React, Django, FastAPI)
- Migration system with rollback
- Schema validation tools

### ✅ ADR-0084: Neo4j Embedding Pipeline Optimization
- Task prefixes for Nomic Embed v2 (10x speedup)
- Connection pooling with persistent httpx client
- Redis cache integration (600-1000x on hits)
- Circuit breaker pattern for resilience

### ✅ ADR-0096: Robust Vector Search with Schema Contract (COMPLETE)
**BREAKTHROUGH - Elite Search Working!**
- ChunkSchema contract enforces consistency across all components
- RobustVectorSearch implements Neo4j's official VectorCypherRetriever pattern
- Fixed the brittleness cycle - no more "fix one thing, break another"
- Both fast_search and elite_search return real, usable results
- Graph context enrichment working (4-14 connections per result)

**Results:** 100% success rate, real file paths, actual content, graph connections

## Common Issues & Solutions

| Issue | Root Cause | Solution |
|-------|------------|----------|
| MCP fails to connect | Hardcoded Docker IPs | Use localhost + exposed ports |
| Auth failures | Password mismatch | Neo4j: `graphrag-password` |
| Indexer no data | Multiple disconnects | ADR-0085: Fix all 5 issues |
| Cross-project data | No isolation | ADR-0029: project property |
| GraphRAG empty | Indexing pipeline broken | ADR-0085: Complete fix |
| 409 Conflicts | Deterministic names | ADR-0060: unique names + Redis lock |
| Vector search empty | Neo4j not storing vectors | Fix Neo4j syntax + API calls |

## Development Workflow

**Dev Mode:** Work in `/Users/mkr/local-coding/claude-l9-template`
- Uses local `.mcp.json`
- Test changes immediately

**Deploy to Global:**
```bash
./scripts/deploy-to-global-mcp.sh
```

**Production Mode:** Use from any other directory
- Uses global `~/.claude/mcp_config.json`

**Rollback:**
```bash
rm -rf ~/.claude/mcp-servers/neural-tools
mv ~/.claude/mcp-servers/neural-tools-backup-* ~/.claude/mcp-servers/neural-tools
```

## Test Results (Sept 24, 2025)
- **Core Tools:** 100% pass (8/8)
- **Advanced:** 100% pass (4/4) - Elite search fixed!
- **Operations:** 100% pass (3/3)
- **Overall:** 100% pass (15/15) 🎉

## Key Learnings
1. Always verify env var propagation
2. Docker networking ≠ host networking
3. Code defaults must match deployment
4. Start conservative with pool sizing
5. Session isolation prevents contention
6. Import paths matter (relative vs absolute)
7. Redis auth required for distributed locks

## Protocol Standards
- MCP: 2025-06-18
- JSON-RPC: 2.0
- Neo4j Driver: 5.22.0
- Redis: RESP3
- Docker Compose: 3.8
- Nomic Embed: v2-MoE

**Confidence: 100%** - Complete architecture with ADR-0060 fix deployed globally.

## Important Reminders
- Do only what's asked; nothing more
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files
- NEVER proactively create documentation unless requested