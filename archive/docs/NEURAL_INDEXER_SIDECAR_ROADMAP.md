# Neural Indexer Sidecar Container Implementation Roadmap

## ADR: L9-2025-001 - Neural Indexer Containerization Strategy

**Status:** ACCEPTED  
**Date:** 2025-09-08  
**Context:** Production neural search system requires continuous file monitoring and indexing  
**Decision:** Implement sidecar container architecture for neural indexer service  

---

## Executive Summary

## Change Notes (2025-09-08)

- MCP transport clarified: Document now explicitly states the MCP server uses STDIO via the MCP SDK and the architecture diagram label is updated to “MCP Server (STDIO)”.
- Dockerfile.indexer snippet updated: Uses existing `requirements/base.txt` and `requirements/prod.txt`, adds `PYTHONPATH`, and comments out a non-existent `docker/scripts/indexer-health.py` (health endpoint to be added in Phase 4).
- docker-compose indexer service: Sets `EMBEDDING_SERVICE_HOST=embeddings` to match the service name in `docker-compose.yml` and replaces the healthcheck with a neutral placeholder until a health endpoint is implemented.
- Service discovery example: Embeddings host changed from `neural-embeddings` to `embeddings` for consistency with Docker Compose.
- Entry point snippet fix: Adds missing `import os` for environment access in the signal-handling example.
- Current State clarification: Notes MCP as “(STDIO transport)” to reflect the running configuration.
- New Tech Stack Snapshot and runnable snippets: Adds `.mcp.json`, MCP bootstrap, indexer run commands, and environment variable examples for quick setup.

Notes
- The roadmap previously referenced `docker/scripts/indexer-health.py`, which does not exist yet. A minimal FastAPI/Prometheus health endpoint should be added in Phase 4 before enabling the compose healthcheck.
- These are documentation updates only and align the roadmap with the current codebase; no application code was changed.

Transform the existing neural indexer service from host-based manual execution to a production-ready containerized sidecar service with full E2E integration, monitoring, and operational excellence.

**Current State (validated against repo):**
- ✅ Neural indexing code functional (`indexer_service.py`)
- ✅ Database services containerized (Neo4j, Qdrant, Redis)
- ✅ MCP server provides search capabilities (STDIO transport)
- ❌ **No continuous file monitoring** 
- ❌ **No containerized indexer deployment**
- ❌ **No production automation**

**Target State:**
- 🎯 Containerized indexer service with file watching
- 🎯 Full docker-compose integration  
- 🎯 Production-ready monitoring and observability
- 🎯 Automated indexing of project codebases
- 🎯 Scalable multi-project support

---

## Architecture Overview

```mermaid
graph TB
    subgraph "L9 GraphRAG Network"
        A[l9-indexer] -->|writes| B[Neo4j Graph DB]
        A -->|writes| C[Qdrant Vector DB] 
        A -->|health| D[Prometheus]
        A -->|logs| E[Log Aggregator]
        
        F[Project Files] -->|watch| A
        G[l9-graphrag API] -->|reads| B
        G -->|reads| C
        
        H[MCP Server (STDIO)] -->|search| G
    end
    
    subgraph "Volumes"
        I[workspace:/workspace] -->|ro| A
        J[indexer_state:/app/state] -->|rw| A
        K[indexer_logs:/app/logs] -->|rw| A
    end
```

**Core Principles:**
- **Separation of Concerns**: Indexing ≠ Serving
- **Fault Tolerance**: Indexer crashes don't affect search
- **Scalability**: Independent resource allocation
- **Observability**: Full monitoring and logging
- **Production Ready**: Security, persistence, operations

---

## Phase 1: Container Foundation (Week 1)

### 1.1 Create Dockerfile.indexer

**Deliverable:** Multi-stage optimized container image

```dockerfile
# Stage 1: Dependencies
FROM python:3.11-slim as dependencies
WORKDIR /app
COPY requirements/base.txt requirements/prod.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r prod.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime
RUN groupadd -r indexer && useradd -r -g indexer indexer

WORKDIR /app
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY neural-tools/src/ ./src/

# Health check endpoint (optional; see Phase 4)
# COPY docker/scripts/indexer-health.py ./health.py
RUN chown -R indexer:indexer /app

USER indexer
EXPOSE 8080

ENV PYTHONPATH=/app:/app/src
CMD ["python", "-m", "src.servers.services.indexer_service", "/workspace", "--project-name", "${PROJECT_NAME:-default}"]
```

**Tasks:**
- [ ] Create optimized Dockerfile with multi-stage build
- [ ] Implement non-root user for security
- [ ] Add proper signal handling for graceful shutdown
- [ ] Create health check endpoint on port 8080
- [ ] Optimize image size and layer caching

**Acceptance Criteria:**
- Container builds successfully
- Health endpoint responds with indexing metrics (if enabled)
- Graceful shutdown on SIGTERM
- Image size < 500MB
- Non-root execution verified

**Risk Mitigation:**
- Test with minimal project first
- Validate file permissions
- Ensure Python path resolution

### 1.2 Container Entry Point & Signal Handling

**Deliverable:** Production-ready entry point script

```python
#!/usr/bin/env python3
"""
Production-ready indexer entry point with signal handling
"""
import signal
import asyncio
import logging
from pathlib import Path
import os

class IndexerRunner:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.indexer = None
        
    async def handle_shutdown(self, sig):
        logging.info(f"Received {sig}, starting graceful shutdown...")
        if self.indexer:
            await self.indexer.shutdown()
        self.shutdown_event.set()
        
    async def run(self):
        # Signal handlers
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda s, f: asyncio.create_task(self.handle_shutdown(s)))
            
        # Start indexer service
        from servers.services.indexer_service import run_indexer
        await run_indexer(
            project_path=os.environ.get('PROJECT_PATH', '/workspace'),
            project_name=os.environ.get('PROJECT_NAME', 'default'),
            initial_index=os.environ.get('INITIAL_INDEX', 'true').lower() == 'true'
        )
        
        await self.shutdown_event.wait()
```

**Tasks:**
- [ ] Implement graceful shutdown handling
- [ ] Add environment variable configuration
- [ ] Create structured logging configuration
- [ ] Add startup health validation
- [ ] Implement clean resource disposal

**Acceptance Criteria:**
- Handles SIGTERM/SIGINT gracefully
- Completes current indexing batch before shutdown
- Saves state before exit
- Proper log correlation IDs

---

## Phase 2: Service Integration (Week 2)

### 2.1 Docker Compose Service Definition

**Deliverable:** Production docker-compose.yml integration

```yaml
services:
  l9-indexer:
    build:
      context: .
      dockerfile: docker/Dockerfile.indexer
    container_name: l9-neural-indexer
    environment:
      # Service Configuration
      - PROJECT_PATH=/workspace
      - PROJECT_NAME=${PROJECT_NAME:-default}
      - INITIAL_INDEX=true
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      
      # Database Connections
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=graphrag-password
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      
      # Embedding Service
      - EMBEDDING_SERVICE_HOST=embeddings
      - EMBEDDING_SERVICE_PORT=8000
      
      # Performance Tuning
      - BATCH_SIZE=10
      - DEBOUNCE_INTERVAL=2.0
      - MAX_QUEUE_SIZE=1000
      
      # Monitoring
      - ENABLE_METRICS=true
      - METRICS_PORT=8080
      
    volumes:
      # Project Source (Read-Only)
      - ./:/workspace:ro
      # Persistent State
      - indexer_state:/app/state
      # Configuration
      - ./neural-tools/config:/app/config:ro
      # Logs
      - indexer_logs:/app/logs
      
    ports:
      - "8080:8080"  # Health/Metrics endpoint
      
    depends_on:
      neo4j:
        condition: service_healthy
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
        
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
      
    restart: unless-stopped
    
    # Resource Limits
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 512M
          cpus: "0.5"
          
    # Security
    user: "1000:1000"
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:size=100M,noexec,nosuid,nodev
      
    networks:
      - l9-graphrag-network
      
volumes:
  indexer_state:
    driver: local
  indexer_logs:
    driver: local
```

**Tasks:**
- [ ] Add indexer service to docker-compose.yml
- [ ] Configure all environment variables
- [ ] Set up volume mounts (source, state, config, logs)
- [ ] Configure service dependencies and health checks
- [ ] Add resource limits and security constraints
- [ ] Create network configuration

**Acceptance Criteria:**
- Service starts after dependencies are healthy
- All volume mounts accessible with correct permissions
- Health check passes consistently
- Resource limits enforced
- Security constraints applied

### 2.2 Volume & Mount Strategy

**Deliverable:** Secure and efficient volume architecture

**Source Code Mount:**
```yaml
volumes:
  - ./:/workspace:ro  # Read-only prevents indexer from modifying source
```

**State Persistence:**
```yaml
volumes:
  - indexer_state:/app/state  # File hashes, metrics, queue state
```

**Configuration:**
```yaml
volumes:
  - ./neural-tools/config:/app/config:ro  # Collection configs, settings
```

macOS/Windows note
- Docker Desktop file watching over bind mounts can drop events. For reliability on macOS/Windows, enable polling mode:
  - Add `WATCHDOG_FORCE_POLLING=1` to the indexer service environment.
  - Consider increasing debounce interval (e.g., `DEBOUNCE_INTERVAL=2.0`).

**Tasks:**
- [ ] Design state persistence schema
- [ ] Implement state backup/restore procedures
- [ ] Configure log rotation and retention
- [ ] Set up configuration hot-reloading
- [ ] Validate read-only source mounting

**Acceptance Criteria:**
- File hashes persist across container restarts
- Configuration changes detected without restart
- Logs rotate properly with retention policy
- Source code remains immutable from container

### 2.3 Service Discovery & Networking

**Deliverable:** Reliable inter-service communication

```python
# Service discovery configuration
SERVICE_DISCOVERY = {
    'neo4j': {
        'host': 'neo4j',  # Docker service name
        'port': 7687,
        'health_check': 'RETURN 1'
    },
    'qdrant': {
        'host': 'qdrant',
        'port': 6333, 
        'health_check': '/health'
    },
    'embeddings': {
        'host': 'embeddings',
        'port': 8000,
        'health_check': '/health'
    }
}
```

**Tasks:**
- [ ] Implement service discovery with retry logic
- [ ] Add connection pooling and circuit breakers
- [ ] Configure network policies and security
- [ ] Add service mesh readiness (future)
- [ ] Implement dependency health monitoring

---

## Phase 3: State Management & Persistence (Week 3)

### 3.1 Persistent State Architecture

**Deliverable:** Robust state management system

```python
# State persistence schema
INDEXER_STATE_SCHEMA = {
    'file_hashes': {
        '/path/to/file.py': 'sha256:abc123...',
        # Prevents re-indexing unchanged files
    },
    'processing_metrics': {
        'files_indexed': 1247,
        'chunks_created': 15890,
        'last_run': '2025-09-08T10:30:00Z',
        'average_processing_time': 1.2,
        'error_count': 3
    },
    'queue_state': {
        'pending_files': ['/path/to/pending.py'],
        'failed_files': ['/path/to/failed.py'],
        'retry_counts': {'/path/to/failed.py': 2}
    },
    'collection_status': {
        'project_default_code': {
            'document_count': 1247,
            'last_update': '2025-09-08T10:30:00Z',
            'health': 'healthy'
        }
    }
}
```

**Tasks:**
- [ ] Design comprehensive state schema
- [ ] Implement atomic state updates
- [ ] Add state validation and recovery
- [ ] Create backup/restore procedures
- [ ] Implement state migration handling

**Acceptance Criteria:**
- State survives container restarts
- Corrupted state auto-recovers
- State size remains bounded
- Migration between schema versions works

### 3.2 Graceful Shutdown & Recovery

**Deliverable:** Zero data loss shutdown procedure

```python
class GracefulShutdownHandler:
    async def shutdown_sequence(self):
        logger.info("Starting graceful shutdown...")
        
        # 1. Stop accepting new file events
        self.event_handler.stop()
        
        # 2. Complete current processing batch
        await self.complete_current_batch()
        
        # 3. Save all state to disk
        await self.save_state()
        
        # 4. Close database connections
        await self.close_connections()
        
        # 5. Signal completion
        logger.info("Graceful shutdown complete")
```

Minimal shutdown hook in indexer (recommended addition):
```python
# Inside IncrementalIndexer
class IncrementalIndexer(...):
    ...
    async def shutdown(self):
        try:
            # Stop receiving new events
            if hasattr(self, 'observer'):
                self.observer.stop()
            # Drain any in-flight work
            while not self.pending_queue.empty():
                await self.process_queue()
        finally:
            # Close service connections if applicable
            if self.container and getattr(self.container, 'close', None):
                await self.container.close()
```

**Tasks:**
- [ ] Implement batch completion before shutdown
- [ ] Add state persistence on shutdown
- [ ] Create connection cleanup procedures
- [ ] Add startup state validation and recovery
- [ ] Implement crash detection and recovery

**Acceptance Criteria:**
- No files lost during shutdown
- State consistency maintained
- Recovery from unexpected crashes
- Startup validation detects corruption

### 3.3 Backup & Disaster Recovery

**Deliverable:** Enterprise backup strategy

```yaml
# Backup configuration
backup:
  strategy: "incremental"
  retention: "30d"
  schedule: "0 2 * * *"  # Daily at 2 AM
  destinations:
    - type: "s3"
      bucket: "l9-indexer-backups"
    - type: "local"
      path: "/app/backups"
```

**Tasks:**
- [ ] Implement automated backup procedures
- [ ] Create point-in-time recovery
- [ ] Add backup validation and testing
- [ ] Design disaster recovery procedures
- [ ] Create operational runbooks

---

## Phase 4: Monitoring & Observability (Week 4)

### 4.1 Prometheus Metrics + Health Endpoint

**Deliverable:** Comprehensive metrics collection

```python
from fastapi import FastAPI, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

# Minimal metrics for the indexer sidecar
files_processed = Counter('indexer_files_processed_total', 'Total files processed')
chunks_created = Counter('indexer_chunks_created_total', 'Total chunks created')
processing_duration = Histogram('indexer_processing_duration_seconds', 'File processing time (s)')
queue_depth = Gauge('indexer_queue_depth', 'Current queue depth')
degraded_mode = Gauge('indexer_degraded_mode', 'Degraded mode (1=yes)')

app = FastAPI()

@app.get('/health')
async def health():
    # Return minimal, fast readiness info
    return {"status": "ok"}

@app.get('/metrics')
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Example integration points within indexer loop (pseudo):
# queue_depth.set(indexer.pending_queue.qsize())
# files_processed.inc()
# with processing_duration.time():
#     await indexer.index_file(path)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
```

**Tasks:**
- [ ] Add FastAPI/Prometheus server to sidecar
- [ ] Wire indexer metrics to counters/gauges
- [ ] Create Grafana dashboards
- [ ] Configure alerting rules
- [ ] Validate metrics scraping in Prometheus

**Acceptance Criteria:**
- All key metrics collected
- Dashboards show system health
- Alerts fire on failures
- Metrics help troubleshooting

### 4.2 Structured Logging & Correlation

**Deliverable:** Comprehensive logging system

```python
# Structured logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'json': {
            'format': '{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","correlation_id":"%(correlation_id)s","message":"%(message)s","file":"%(filename)s","line":"%(lineno)d"}'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'json'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/app/logs/indexer.log',
            'maxBytes': 100_000_000,  # 100MB
            'backupCount': 10,
            'formatter': 'json'
        }
    },
    'loggers': {
        'indexer': {'handlers': ['console', 'file'], 'level': 'INFO'}
    }
}
```

**Tasks:**
- [ ] Implement structured JSON logging
- [ ] Add correlation ID tracking
- [ ] Configure log aggregation
- [ ] Set up log-based alerting
- [ ] Create log analysis tools

### 4.3 Health Checks & Service Discovery

**Deliverable:** Robust health monitoring

```python
@app.get("/health")
async def health_check():
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {},
        'metrics': {}
    }
    
    # Check service dependencies
    for service_name, service in SERVICES.items():
        try:
            await service.health_check()
            health_status['services'][service_name] = 'healthy'
        except Exception as e:
            health_status['services'][service_name] = f'unhealthy: {e}'
            health_status['status'] = 'degraded'
    
    # Add key metrics
    health_status['metrics'] = {
        'files_in_queue': indexer.queue_depth,
        'last_processing_time': indexer.last_run,
        'error_count': indexer.error_count
    }
    
    return health_status
```

**Tasks:**
- [ ] Implement comprehensive health endpoints
- [ ] Add dependency health validation
- [ ] Create service readiness checks
- [ ] Configure container orchestration health
- [ ] Add custom health business logic

---

### 4.4 Indexer Control API + MCP Tools (Recommended)

**Deliverable:** Operational control via HTTP API and MCP tools

Indexer HTTP control API (embed in sidecar):
```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

def attach_api(indexer):
    @app.get('/status')
    async def status():
        return indexer.get_metrics()

    @app.post('/reindex-path')
    async def reindex_path(path: str):
        await indexer._queue_change(path, 'update')
        return {"enqueued": path}

# Run this FastAPI app alongside the indexer loop (port 8080)
```

MCP tools calling the sidecar API:
```python
# neural-tools/src/neural_mcp/neural_server_stdio.py
import httpx

@server.list_tools()
async def handle_list_tools():
    tools = [
        # ... existing tools ...,
        types.Tool(
            name="indexer_status",
            description="Get indexer metrics and health",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="reindex_path",
            description="Enqueue a path for reindexing",
            inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        )
    ]
    return tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    if name == 'indexer_status':
        async with httpx.AsyncClient() as client:
            r = await client.get('http://localhost:8080/status')
            return [types.TextContent(type='text', text=r.text)]
    if name == 'reindex_path':
        async with httpx.AsyncClient() as client:
            r = await client.post('http://localhost:8080/reindex-path', params={"path": arguments.get('path', '')})
            return [types.TextContent(type='text', text=r.text)]
    # ... existing handlers ...
```

**Tasks:**
- [ ] Add FastAPI control endpoints to sidecar
- [ ] Add MCP tools `indexer_status`, `reindex_path`
- [ ] Document operational flows (pause/resume optional)

**Acceptance Criteria:**
- MCP can report indexer status and trigger reindex of a path
- No tight coupling between MCP and indexer process

## Phase 5: Production Readiness (Week 5)

### 5.1 Multi-Environment Configuration

**Deliverable:** Environment-specific configurations

```yaml
# environments/development.yml
environment: development
indexer:
  batch_size: 5
  debounce_interval: 1.0
  log_level: DEBUG
  initial_index: true

databases:
  neo4j:
    uri: bolt://localhost:7687
  qdrant:
    host: localhost
    port: 6333

# environments/production.yml  
environment: production
indexer:
  batch_size: 20
  debounce_interval: 5.0
  log_level: INFO
  initial_index: false

databases:
  neo4j:
    uri: bolt://neo4j-prod:7687
  qdrant:
    host: qdrant-prod
    port: 6333
    
monitoring:
  prometheus:
    enabled: true
    scrape_interval: 15s
  alerting:
    enabled: true
    webhook: https://slack.com/webhook/alerts
```

**Tasks:**
- [ ] Create environment-specific configurations
- [ ] Implement configuration validation
- [ ] Add secrets management integration
- [ ] Create deployment automation
- [ ] Add configuration drift detection

### 5.2 Security Hardening

**Deliverable:** Production security posture

```dockerfile
# Security hardening
FROM python:3.11-slim

# Create non-root user
RUN groupadd -g 1000 indexer && useradd -u 1000 -g indexer indexer

# Install security updates
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Set security options
USER 1000:1000
WORKDIR /app

# Read-only file system
RUN mkdir -p /app/tmp /app/logs /app/state
VOLUME ["/app/tmp", "/app/logs", "/app/state"]

# Drop all capabilities
# Set in docker-compose:
# cap_drop: [ALL]
# security_opt: [no-new-privileges:true]
# read_only: true
```

**Tasks:**
- [ ] Implement container security best practices
- [ ] Add secrets management (HashiCorp Vault)
- [ ] Configure network policies
- [ ] Add security scanning
- [ ] Create security monitoring

### 5.3 Performance Optimization

**Deliverable:** Production-tuned performance

```python
# Performance configuration
PERFORMANCE_CONFIG = {
    'file_watching': {
        'use_native_watcher': True,  # inotify on Linux
        'batch_events': True,
        'debounce_ms': 2000,
        'max_events_per_second': 100
    },
    'indexing': {
        'batch_size': 20,
        'parallel_workers': 4,
        'chunk_size_target': 512,  # tokens
        'embedding_batch_size': 32
    },
    'database': {
        'connection_pool_size': 10,
        'query_timeout_ms': 30000,
        'retry_attempts': 3,
        'circuit_breaker_threshold': 5
    },
    'memory': {
        'max_cache_size_mb': 512,
        'gc_threshold': 0.8,
        'state_checkpoint_interval': 300  # seconds
    }
}
```

**Tasks:**
- [ ] Implement performance profiling
- [ ] Add connection pooling optimization
- [ ] Configure memory management
- [ ] Add performance regression testing
- [ ] Create capacity planning guidelines

### 5.4 Operational Excellence

**Deliverable:** Complete operational framework

```markdown
# Operational Runbooks

## Incident Response
1. Check indexer health endpoint
2. Verify service dependencies
3. Review recent error logs
4. Check resource utilization
5. Escalation procedures

## Deployment Procedures
1. Blue-green deployment strategy
2. Database migration procedures
3. Configuration validation
4. Smoke testing checklist
5. Rollback procedures

## Maintenance Procedures  
1. Log rotation and cleanup
2. State database maintenance
3. Performance tuning
4. Backup validation
5. Security patching
```

**Tasks:**
- [ ] Create operational runbooks
- [ ] Implement deployment automation
- [ ] Add monitoring dashboards
- [ ] Create troubleshooting guides
- [ ] Design capacity planning procedures

---

## Testing Strategy

### Unit Testing
- Container build verification
- Service integration tests
- State persistence validation
- Error handling verification
- Performance regression tests

### Integration Testing
- Full docker-compose stack
- Multi-project indexing
- Database failover scenarios
- Network partition handling
- Resource exhaustion testing

### End-to-End Testing
- Complete indexing workflow
- Search result validation
- Monitoring and alerting
- Backup and recovery
- Production deployment simulation

---

## Success Metrics

### Technical KPIs
- **Indexing Latency**: < 5 seconds per file
- **System Uptime**: > 99.9%
- **Error Rate**: < 0.1%
- **Resource Utilization**: CPU < 70%, Memory < 80%
- **Recovery Time**: < 2 minutes

### Business KPIs
- **Code Search Accuracy**: > 95% relevant results
- **Developer Adoption**: Usage metrics and feedback
- **System Reliability**: Incident frequency and resolution time
- **Operational Efficiency**: Manual intervention reduction

---

## Risk Assessment

### High Risk
- **File System Performance**: Large codebases may overwhelm watching
- **Resource Consumption**: Memory leaks or CPU spikes
- **Database Failures**: Neo4j or Qdrant unavailability

### Medium Risk  
- **Network Partitions**: Service communication failures
- **Configuration Drift**: Environment inconsistencies
- **Version Compatibility**: Dependency conflicts

### Low Risk
- **Container Registry**: Image availability
- **Log Storage**: Disk space management
- **Monitoring Gaps**: Metric collection failures

### Mitigation Strategies
- Comprehensive testing at each phase
- Circuit breakers and retry logic
- Monitoring and alerting at all levels
- Documentation and runbooks
- Regular disaster recovery testing

---

## Conclusion

This roadmap provides a comprehensive, production-ready approach to containerizing the neural indexer service. Each phase builds upon the previous, ensuring a robust, scalable, and maintainable solution.

The phased approach allows for:
- **Incremental Risk Management**: Small, testable changes
- **Continuous Validation**: Each phase has clear success criteria  
- **Operational Readiness**: Full monitoring and operational excellence
- **Future Scalability**: Architecture supports growth and enhancement

**Total Estimated Timeline: 5 weeks**  
**Resource Requirements: 1 senior engineer + DevOps support**  
**Success Probability: High (>90%) with proper execution**

---

## Tech Stack Snapshot (validated)

- Python: 3.11 (containers), local dev supports 3.12/3.13
- MCP Server: STDIO transport via `mcp.server` SDK
  - Entrypoint: `neural-tools/run_mcp_server.py`
  - Canonical server: `neural-tools/src/neural_mcp/neural_server_stdio.py`
- Indexer: Async watchdog-based incremental indexer
  - Module: `neural-tools/src/servers/services/indexer_service.py`
- Vector DB: Qdrant (HTTP 6333, gRPC 6334)
- Graph DB: Neo4j 5.x via official driver
- Embeddings: Nomic Embed HTTP service (`EMBEDDING_SERVICE_HOST`/`PORT`)
- Observability: OpenTelemetry scaffolding; Prometheus client optional

### Useful Code Snippets

- Claude MCP registration (`.mcp.json`):
  ```json
  {
    "mcpServers": {
      "neural-tools": {
        "type": "stdio",
        "command": "/Users/mkr/local-coding/claude-l9-template/neural-tools/run_mcp_server.py",
        "args": [],
        "env": {}
      }
    }
  }
  ```

- MCP STDIO server bootstrap:
  ```python
  # neural-tools/run_mcp_server.py
  def main():
      _add_src_to_path()
      from neural_mcp.neural_server_stdio import run
      asyncio.run(run())
  ```

- Indexer entrypoint (container CMD):
  ```bash
  python -m src.servers.services.indexer_service /workspace --project-name default --initial-index
  ```

- Run indexer locally (from repo root):
  ```bash
  python neural-tools/src/servers/services/indexer_service.py . --project-name default --initial-index
  ```

- Environment variables used by services:
  ```bash
  export NEO4J_URI=bolt://localhost:7687
  export NEO4J_USERNAME=neo4j
  export NEO4J_PASSWORD=graphrag-password
  export QDRANT_HOST=localhost
  export QDRANT_PORT=6333
  export EMBEDDING_SERVICE_HOST=localhost
  export EMBEDDING_SERVICE_PORT=8000
  ```

---

*This roadmap represents a comprehensive analysis of the neural indexer containerization requirements and provides a detailed implementation strategy for production deployment.*
