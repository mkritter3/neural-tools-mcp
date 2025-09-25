# Docker Setup - L9 Neural GraphRAG

## Quick Start

### Production Stack
```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps
docker-compose logs -f
```

### Development Mode
```bash
# Start with debug logging and hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Rebuild after code changes
docker-compose build --no-cache nomic
docker-compose build --no-cache indexer-dev
```

## Architecture

### Core Services

| Service | Port | Purpose |
|---------|------|---------|
| Neo4j | 47687 | GraphRAG database (graph + vectors) |
| Redis Cache | 46379 | Session state & embedding cache |
| Redis Queue | 46380 | Task queue & distributed locks |
| Nomic | 48000 | Embeddings (v2-MoE optimized) |
| Indexer | 48100+ | Per-project code indexing |

### Key Changes (Sept 2025)
- **No Qdrant**: Neo4j handles vector storage with HNSW indexes (ADR-0096)
- **Optimized Nomic**: Using v2-MoE with CPU optimizations
- **Dynamic Indexers**: Created per-project by MCP server (ports 48100+)

## Common Operations

### View Neo4j Browser
```bash
open http://localhost:47474
# Login: neo4j / graphrag-password
```

### Check Indexer Status
```bash
curl http://localhost:48100/status
curl http://localhost:48100/metrics
```

### Restart Services
```bash
# Restart specific service
docker-compose restart neo4j

# Full restart
docker-compose down && docker-compose up -d
```

### Clean Up
```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Clean old images
docker image prune -a
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs neo4j
docker-compose logs nomic

# Verify ports are free
lsof -i :47687  # Neo4j
lsof -i :48000  # Nomic
```

### Indexer Issues
```bash
# Check running indexers
docker ps | grep indexer

# View indexer logs
docker logs indexer-{project-name}

# Restart indexer
docker restart indexer-{project-name}
```

### Memory Issues
Adjust memory limits in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 4G  # Increase as needed
```

## Production Deployment

1. Use specific image versions (not `:latest`)
2. Set up volume backups for Neo4j data
3. Monitor with Prometheus/Grafana (optional)
4. Use Docker Swarm or K8s for scaling

## Environment Variables

Create `.env` file for overrides:
```env
# Project settings
PROJECT_NAME=myproject
PROJECT_PATH=/Users/me/myproject

# Service versions
NEO4J_VERSION=5.26.0
REDIS_VERSION=7-alpine

# Memory limits
NEO4J_HEAP_SIZE=4g
NEO4J_PAGECACHE_SIZE=2g
```