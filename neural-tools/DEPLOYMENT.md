# L9 Neural Tools - Deployment Guide

## Quick Start

1. **Environment Setup**
   ```bash
   cp .env.example .env
   # Edit .env and set a secure NEO4J_PASSWORD
   ```

2. **Start Services**
   ```bash
   docker-compose -f config/docker-compose.neural-tools.yml up -d
   ```

3. **Verify Deployment**
   ```bash
   python3 validate_infrastructure.py
   ```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    L9 Neural Tools                         │
├─────────────────────────────────────────────────────────────┤
│  neural-tools-server     │  15 MCP Tools + GraphRAG       │
│  neural-data-storage     │  Qdrant Vector Database        │
│  neural-embeddings       │  Nomic Embed v2-MoE Service    │
│  neo4j-graph            │  Neo4j GraphRAG Database       │
└─────────────────────────────────────────────────────────────┘
```

## Recent Infrastructure Fixes

### ✅ Critical Deployment Blockers Resolved

1. **Import Path Fixes**
   - Fixed Docker Compose health check: `neural_mcp_server_enhanced` → `neural_mcp_wrapper`
   - Updated uvicorn command: `neural-tools.nomic_embed_server` → `src.servers.nomic_embed_server`
   
2. **Dockerfile Modernization**
   - Updated COPY operations for organized src/ directory structure
   - Added proper PYTHONPATH configuration: `/app:/app/src:/app/src/servers:/app/common`
   - Fixed module imports in both l9-minimal and neural-embeddings containers

3. **Security Improvements**
   - Replaced hardcoded Neo4j passwords with environment variables
   - Created comprehensive .env.example template
   - Enabled secure credential management

4. **Build Optimization**
   - Added .dockerignore for faster builds and smaller contexts
   - Excluded development files, caches, and documentation from builds

## Container Services

### neural-tools-server
- **Image**: `l9-mcp-enhanced:minimal-fixed`
- **Purpose**: Main MCP server with 15 neural tools
- **Resources**: 4G memory, 4 CPUs
- **Health Check**: Validates all MCP tools accessible

### neural-data-storage  
- **Image**: `qdrant/qdrant:v1.10.0`
- **Purpose**: Vector database for semantic search
- **Features**: RRF hybrid search, INT8 quantization, MMR diversity
- **Resources**: 4G memory, 3 CPUs

### neural-embeddings
- **Image**: `neural-flow:nomic-v2-production` 
- **Purpose**: Nomic Embed v2-MoE embedding service
- **Features**: 305M active / 475M total parameters, 30-40% cost reduction
- **Resources**: 8G memory, 6 CPUs

### neo4j-graph
- **Image**: `neo4j:5.23.0-community`
- **Purpose**: GraphRAG code relationship storage
- **Features**: Persistent graph storage, Cypher queries
- **Resources**: 6G memory, 4 CPUs

## Environment Variables

### Required
- `NEO4J_PASSWORD`: Secure password for Neo4j database (replaces hardcoded default)

### Optional  
- `PROJECT_NAME`: Project identifier (default: "default")
- `PROJECT_DIR`: Source code directory (default: "../")

### Debug Ports (Optional)
- `QDRANT_DEBUG_PORT`: Qdrant REST API access (default: 6681)
- `NOMIC_DEBUG_PORT`: Nomic embeddings service access (default: 8081)
- `NEO4J_DEBUG_HTTP_PORT`: Neo4j browser access (default: 7475)
- `NEO4J_DEBUG_BOLT_PORT`: Neo4j bolt access (default: 7688)

## Validation Commands

```bash
# Validate Docker Compose syntax
docker-compose -f config/docker-compose.neural-tools.yml config

# Run infrastructure validation
python3 validate_infrastructure.py

# Check service health
docker-compose -f config/docker-compose.neural-tools.yml ps

# View logs
docker-compose -f config/docker-compose.neural-tools.yml logs neural-tools-server
```

## MCP Tools Available (15 Total)

### Memory & Search
- `memory_store_enhanced`: Store content with hybrid indexing
- `memory_search_enhanced`: RRF fusion search with MMR diversity

### Graph Operations  
- `graph_query`: Execute Cypher queries on Neo4j
- `neo4j_graph_query`: Advanced graph queries
- `neo4j_semantic_graph_search`: Semantic search across code graph
- `neo4j_code_dependencies`: Get code dependency graphs
- `neo4j_migration_status`: Check migration status
- `neo4j_index_code_graph`: Index code into graph

### Code Analysis
- `atomic_dependency_tracer`: Trace function/class dependencies
- `project_understanding`: Generate project insights
- `semantic_code_search`: Search by meaning, not text
- `vibe_preservation`: Learn and apply coding patterns

### System Management
- `schema_customization`: Customize Qdrant schemas
- `project_auto_index`: Smart file indexing
- `neural_system_status`: Comprehensive system status

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose -f config/docker-compose.neural-tools.yml logs [service-name]

# Verify environment
cat .env

# Validate configuration  
docker-compose -f config/docker-compose.neural-tools.yml config
```

### Import Errors
- Ensure containers are using correct PYTHONPATH
- Verify src/ directory structure is mounted correctly
- Check that neural_mcp_wrapper.py can import neural-mcp-server-enhanced.py

### Neo4j Connection Issues
- Verify NEO4J_PASSWORD is set in .env
- Check Neo4j container logs for authentication errors
- Confirm bolt://neo4j-graph:7687 is accessible from neural-tools-server

## Production Considerations

1. **Security**: Never commit .env files. Use secrets management in production.
2. **Persistence**: All data persists in .neural-tools/ directories
3. **Resources**: Adjust memory/CPU limits based on workload
4. **Monitoring**: Use health checks and service logs for monitoring
5. **Backup**: Regular backups of .neural-tools/ data directories

## Support

- Validate deployment: `python3 validate_infrastructure.py`
- Check service health: Docker Compose health checks
- View comprehensive logs: `docker-compose logs -f`