# L9 Engineering Contract

**Date: September 21, 2025 | MCP Protocol: 2025-06-18**

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
| Neo4j | 47687 | GraphRAG |
| Qdrant | 46333 | Vector DB |
| Redis Cache | 46379 | Session cache |
| Redis Queue | 46380 | Task queue |
| Nomic | 48000 | Embeddings |
| Indexer | 48100+ | Per-project |

## Configuration Priority (ADR-0037)
1. Environment variables (highest)
2. Config files (pyproject.toml, package.json)
3. Auto-detection (fallback)
4. Hard-coded defaults (last resort)

**Container→Host:** Use `host.docker.internal` not `localhost`

## Docker Image Management (ADR-0038)
- Production images: `service:production` tag (auto-updates)
- Semantic versioning: `v1.2.3`
- No debug suffixes in production
- 7-day cleanup for temp tags

## MCP Connection Pooling (L9 Conservative)
- Neo4j: 50 (min 5) - Complex queries need headroom
- Qdrant: 30 (min 3) - Fast vector ops
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
- Consistent between Neo4j and Qdrant

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

## Common Issues & Solutions

| Issue | Root Cause | Solution |
|-------|------------|----------|
| MCP fails to connect | Hardcoded Docker IPs | Use localhost + exposed ports |
| Auth failures | Password mismatch | Neo4j: `graphrag-password` |
| Indexer no data | Event loop/schema issues | Use `asyncio.run_coroutine_threadsafe()` |
| Cross-project data | No isolation | ADR-0029: project property |
| GraphRAG empty | Collection naming | ADR-0039: consistent naming |
| 409 Conflicts | Deterministic names | ADR-0060: unique names + Redis lock |

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

## Test Results (Sept 2025)
- **Core Tools:** 100% pass (8/8)
- **Advanced:** 50% pass (2/4)
- **Operations:** 100% pass (3/3)
- **Overall:** 86.7% pass (13/15)

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
- Qdrant: 1.15.1
- Redis: RESP3

**Confidence: 100%** - Complete architecture with ADR-0060 fix deployed globally.

## Important Reminders
- Do only what's asked; nothing more
- NEVER create files unless absolutely necessary
- ALWAYS prefer editing existing files
- NEVER proactively create documentation unless requested