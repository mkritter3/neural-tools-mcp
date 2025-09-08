# Neural Indexer Sidecar Deployment Guide

## Quick Start

### 1. Build and Deploy with Docker Compose

```bash
# Build the complete stack
docker-compose up -d

# Check indexer status
curl http://localhost:8080/health
curl http://localhost:8080/status
```

### 2. Verify MCP Integration

```bash
# Test MCP neural tools (from Claude)
# Should now include indexer_status and reindex_path tools
```

### 3. Monitor with Prometheus Metrics

```bash
curl http://localhost:8080/metrics
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude MCP    │───▶│  Neural Indexer │───▶│   Vector/Graph  │
│     Client      │    │     Sidecar     │    │   Databases     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │                       ▼                       │
        │               ┌───────────────┐               │
        └──────────────▶│   l9-graphrag │◀──────────────┘
                        │   API Server  │
                        └───────────────┘
```

## Key Features Implemented

### ✅ Phase 1: Container Foundation
- Multi-stage Docker build
- Non-root user execution
- Graceful shutdown handling
- Health check endpoints

### ✅ Phase 2: Service Integration  
- Docker Compose integration
- Volume mounts for persistence
- Service dependencies and health checks
- MCP tool integration (`indexer_status`, `reindex_path`)

### ✅ Phase 3: State Management
- Persistent file hash tracking
- Atomic state saves with backup
- Corruption detection and recovery
- Periodic state checkpoints

### ✅ Phase 4: Monitoring & Observability
- Prometheus metrics endpoint
- Health and status API endpoints
- Structured JSON logging
- Service health tracking

### 🔄 Phase 5: Production Readiness (Basic)
- Environment-specific configuration support
- Security hardening (non-root, read-only filesystem)
- Resource limits and constraints

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_PATH` | `/workspace` | Path to project source code |
| `PROJECT_NAME` | `default` | Project identifier |
| `INITIAL_INDEX` | `true` | Perform initial indexing on startup |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j connection string |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `graphrag-password` | Neo4j password |
| `QDRANT_HOST` | `qdrant` | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `EMBEDDING_SERVICE_HOST` | `embeddings` | Embedding service hostname |
| `EMBEDDING_SERVICE_PORT` | `8000` | Embedding service port |
| `BATCH_SIZE` | `10` | Files per processing batch |
| `DEBOUNCE_INTERVAL` | `2.0` | File change debounce time (seconds) |
| `WATCHDOG_FORCE_POLLING` | `1` | Force polling for file changes (recommended for Docker) |

## API Endpoints

### Health Check
```bash
GET /health
# Returns: {"status": "ok", "timestamp": "...", "version": "1.0.0"}
```

### Status & Metrics
```bash
GET /status
# Returns detailed indexer status with queue depth, files processed, etc.
```

### Prometheus Metrics
```bash
GET /metrics
# Returns Prometheus-format metrics
```

### Reindex Path
```bash
POST /reindex-path?path=/workspace/some/file.py
# Triggers reindexing of specific file or directory
```

## MCP Tools

Two new tools are available in the neural-tools MCP server:

### `indexer_status`
Get indexer sidecar status and metrics

### `reindex_path` 
Trigger reindexing of a specific file or directory path

## Troubleshooting

### Container Won't Start
```bash
docker logs l9-neural-indexer
```

### Performance Issues
- Check queue depth: `curl localhost:8080/status`
- Adjust `BATCH_SIZE` and `DEBOUNCE_INTERVAL`
- Monitor Prometheus metrics

### State Recovery
State is automatically saved to `/app/state/indexer_state.json` with backup at `indexer_state_backup.json`

### Service Failures
The indexer operates in degraded mode if individual services (Neo4j, Qdrant, embeddings) are unavailable.

## Testing

Run the end-to-end test suite:

```bash
./scripts/test-neural-indexer-e2e.sh
```

This verifies:
- Container builds successfully
- All API endpoints respond correctly  
- Graceful shutdown works
- No critical errors in logs

## Production Considerations

1. **Resource Limits**: Set appropriate CPU/memory limits in docker-compose.yml
2. **Persistent Storage**: Ensure `indexer_state` and `indexer_logs` volumes are persistent
3. **Monitoring**: Integrate Prometheus metrics with your monitoring stack
4. **Backup**: The state files contain file hashes - consider backing up the state volume
5. **Security**: The container runs with non-root user and read-only filesystem

## Performance Tuning

For large codebases (>10k files):
- Increase `BATCH_SIZE` to 50-100
- Increase `DEBOUNCE_INTERVAL` to 5-10 seconds  
- Monitor queue depth and adjust resource limits accordingly
- Consider running multiple indexer instances for different projects