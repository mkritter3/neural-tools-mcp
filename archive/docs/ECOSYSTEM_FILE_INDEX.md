# Neural Ecosystem File Index

**Last Updated:** 2025-09-09  
**Purpose:** Complete index of all active files in the L9 Neural GraphRAG ecosystem

## 🏗️ Core Infrastructure

### Docker Configuration
- `docker-compose.yml` - Main development orchestration (7 services)
- `docker-compose.production.yml` - Production deployment configuration
- `docker/Dockerfile` - Main GraphRAG service container
- `docker/Dockerfile.indexer` - Neural indexer sidecar container
- `docker/scripts/indexer-entrypoint.py` - Indexer initialization script

### Environment Configuration
- `.env.mcp.local` - Local MCP server environment variables
- `.env.production.template` - Production environment template
- `deploy-production.sh` - Production deployment automation script

## 🧠 Neural Tools (MCP Server & Services)

### MCP Server Core
- `neural-tools/src/servers/mcp_server.py` - Main MCP server with neural search tools
- `neural-tools/src/servers/__init__.py` - Package initialization
- `neural-tools/pyproject.toml` - Python project configuration

### Service Layer
- `neural-tools/src/servers/services/indexer_service.py` - File monitoring & indexing orchestration
- `neural-tools/src/servers/services/neo4j_service.py` - Graph database operations
- `neural-tools/src/servers/services/qdrant_service.py` - Vector database operations
- `neural-tools/src/servers/services/tree_sitter_extractor.py` - Code structure extraction
- `neural-tools/src/servers/services/embedding_service.py` - Nomic embedding client
- `neural-tools/src/servers/services/service_container.py` - Dependency injection container

### Configuration
- `neural-tools/src/servers/config/runtime.py` - Runtime configuration management
- `neural-tools/config/requirements-indexer-lean.txt` - Indexer dependencies
- `neural-tools/config/requirements-mcp.txt` - MCP server dependencies

## 📊 GraphRAG Core

### Main Application
- `src/graphrag/__init__.py` - GraphRAG package initialization
- `src/graphrag/core/graph_builder.py` - Graph construction logic
- `src/graphrag/core/embedder.py` - Embedding generation
- `src/graphrag/core/indexer.py` - Document indexing
- `src/graphrag/core/retriever.py` - Graph retrieval logic
- `src/graphrag/core/query_engine.py` - Query processing

### Models & Services
- `src/graphrag/models/document.py` - Document data models
- `src/graphrag/models/graph.py` - Graph data models
- `src/graphrag/services/neo4j_client.py` - Neo4j integration
- `src/graphrag/services/qdrant_client.py` - Qdrant integration
- `src/graphrag/services/llm_client.py` - LLM integration

### Utilities
- `src/graphrag/utils/config.py` - Configuration utilities
- `src/graphrag/utils/logger.py` - Logging configuration
- `src/graphrag/utils/metrics.py` - Metrics collection

## 🔌 MCP Integration

### Server Configuration
- `.mcp.json` - MCP server registration and configuration
- `src/mcp/__init__.py` - MCP tools package
- `src/mcp/tools.py` - MCP tool implementations
- `src/mcp/handlers.py` - Request handlers

## 📚 Documentation

### Architecture Decision Records
- `docs/adr/0010-redis-resilience-architecture.md` - Redis dual-instance design
- `docs/adr/0011-semantic-search-enablement-qdrant-nomic-extraction.md` - Semantic search implementation
- `docs/adr/0012-extended-language-support-tree-sitter.md` - Multi-language support
- `docs/adr/0013-semantic-code-search-implementation.md` - Code search capabilities
- `docs/adr/0014-incremental-indexing-optimization.md` - Performance optimization
- `docs/adr/0015-mcp-server-enhanced-tools.md` - Enhanced MCP tools

### Guides & Documentation
- `CLAUDE.md` - Claude-specific instructions and context
- `README.md` - Main project documentation
- `GRAPHRAG_README.md` - GraphRAG specific documentation
- `GRAPHRAG_DOCKER_GUIDE.md` - Docker setup guide
- `NEURAL_INDEXER_SIDECAR_ROADMAP.md` - Indexer development roadmap
- `NEURAL_SEARCH_ROADMAP.md` - Neural search roadmap
- `FULL_STACK_MANIFEST.md` - Complete stack documentation

## 🧪 Testing Suite

### Integration Tests
- `test_phase3_integration.py` - Phase 3 integration tests
- `test_integration_phase3.py` - Alternative phase 3 tests
- `test_full_integration.py` - Complete system integration
- `test_live_services.py` - Live service validation

### Component Tests
- `test_tree_sitter.py` - Tree-sitter extraction tests
- `test_tree_sitter_integration.py` - Tree-sitter integration tests
- `test_indexer_minimal.py` - Minimal indexer tests
- `test_symbols_live.py` - Symbol extraction tests
- `test_cache_only.py` - Cache functionality tests

### Utilities
- `preflight_validation.py` - Pre-deployment validation
- `trigger_extraction_test.py` - Manual extraction trigger

## 🚀 Deployment & Operations

### Build System
- `Makefile` - Build automation commands
- `requirements.txt` - Python dependencies (main)
- `requirements-dev.txt` - Development dependencies

### Monitoring
- `monitoring/prometheus.yml` - Prometheus configuration
- `monitoring/grafana/dashboards/` - Grafana dashboards

## 📦 Supporting Services Configuration

### Database Schemas
- `migrations/neo4j/` - Neo4j schema migrations
- `migrations/qdrant/` - Qdrant collection configurations

### Service Definitions
- `systemd/neural-indexer.service` - Systemd service file
- `systemd/mcp-server.service` - MCP server service

## 🔧 Development Tools

### Git Configuration
- `.gitignore` - Git ignore patterns
- `.gitattributes` - Git attributes

### IDE Configuration
- `.vscode/settings.json` - VS Code settings
- `.vscode/launch.json` - Debug configurations

## 📊 Active Services in Docker Compose

1. **l9-graphrag** - Main GraphRAG service (ports: 43000, 49090)
2. **neo4j** - Graph database (ports: 47474, 47687)
3. **qdrant** - Vector database (ports: 46333, 46334)
4. **redis-queue** - Durable job queue (port: 46380)
5. **redis-cache** - LRU cache (port: 46379)
6. **embeddings** - Nomic embedding service (port: 48000)
7. **l9-indexer** - Neural indexer sidecar (port: 48080)
8. **prometheus** - Metrics collection (port: 49091)
9. **grafana** - Metrics visualization (port: 43001)

## 🌐 Network Architecture

- **Network Name:** `l9-graphrag-network`
- **Internal Communication:** Service names (neo4j, qdrant, redis-queue, redis-cache, embeddings)
- **External Access:** Mapped ports for development

## 📁 Persistent Volumes

- `neo4j_data` - Graph database storage
- `neo4j_logs` - Graph database logs
- `qdrant_data` - Vector database storage
- `redis_queue_data` - Durable queue persistence
- `redis_cache_data` - Cache persistence
- `indexer_state` - Indexer state management
- `indexer_logs` - Indexer logs
- `prometheus_data` - Metrics storage
- `grafana_data` - Dashboard persistence
- `models` - ML model cache
- `workspace` - Shared workspace

## 🔑 Key Integration Points

1. **MCP Server** → All services via service container
2. **Indexer** → Neo4j, Qdrant, Embeddings, Redis
3. **GraphRAG** → Neo4j, Qdrant
4. **Tree-sitter** → Indexer → Neo4j/Qdrant
5. **Redis Queue** → Job processing
6. **Redis Cache** → Performance optimization

---

**Note:** This index represents the complete active ecosystem as of the tree-sitter implementation (commit: ccdf289)