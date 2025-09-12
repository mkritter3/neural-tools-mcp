# GraphRAG Docker Container - Build & Usage Guide

## 🚀 Quick Start

Build and run the GraphRAG-enabled MCP server with Neo4j and Qdrant integration:

```bash
# Build the container
docker build -f neural-tools/Dockerfile.l9-minimal -t l9-mcp-graphrag:production neural-tools/

# Run with volume mount for data persistence
docker run -d --name l9-graphrag \
  -p 3000:3000 \
  -v "$(pwd)/neural-tools/config/.neural-tools:/app/.neural-tools" \
  l9-mcp-graphrag:production
```

## ✅ What's Included

### GraphRAG Implementation
- **Deterministic SHA256-based IDs**: Cross-references between Neo4j and Qdrant
- **Bidirectional linking**: Each vector point knows its graph node, and vice versa
- **Content deduplication**: SHA256 content hashing prevents duplicate indexing
- **Event debouncing**: Handles file system storms (e.g., git operations)

### MCP Tools (4 new GraphRAG tools)
- `graphrag_hybrid_search`: Semantic search with graph relationship enrichment
- `graphrag_impact_analysis`: Analyze code change impacts across dependencies
- `graphrag_find_dependencies`: Trace dependency chains through both vector and graph
- `graphrag_find_related`: Find contextually related code using hybrid patterns

### Services
- **MCP Server**: Exposes all tools via Model Context Protocol
- **Indexer Service**: Monitors `/workspace` and maintains synchronized indexes
- **File System Monitoring**: Uses `watchdog` for real-time file change detection

## 🔧 Container Details

### Image Info
- **Base**: `python:3.11-slim`
- **Size**: ~13.5GB (includes PyTorch + transformers)
- **Services**: Managed by supervisord

### Dependencies Added
- **Core**: `mcp`, `fastmcp`, `neo4j`, `qdrant-client`
- **ML**: `torch`, `transformers`, `sentence-transformers`
- **Monitoring**: `watchdog`, `structlog`, `prometheus-client`
- **All requirements**: See `neural-tools/config/requirements-l9-enhanced.txt`

## 🏗 Architecture

```
Container Structure:
├── /app/src/mcp/neural_server_stdio.py  # MCP server with GraphRAG tools (canonical)
├── /app/neural-tools-src/services/indexer_service.py  # File indexing with GraphRAG
├── /app/src/servers/services/hybrid_retriever.py  # GraphRAG query implementation
├── /app/config/supervisord.conf  # Service management
└── /app/.neural-tools/  # Neo4j + Qdrant data (mounted volume)
```

## 🧪 Testing

Verify the container works:

```bash
# Check services are running
docker exec l9-graphrag supervisorctl status

# Test GraphRAG imports
docker exec l9-graphrag python3 -c "
import sys
sys.path.insert(0, '/app/src/servers')
from services.hybrid_retriever import HybridRetriever
print('✅ GraphRAG ready!')
"

# Check logs
docker logs l9-graphrag
```

## 🔗 Integration

### With Neo4j & Qdrant
To connect to databases, ensure they're running and accessible:

```bash
# Example docker-compose.yml snippet:
services:
  l9-graphrag:
    image: l9-mcp-graphrag:production
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - neo4j
      - qdrant
```

### With Claude Code
The MCP server exposes GraphRAG tools automatically. Configure your MCP client to connect via stdio.

## 🔍 GraphRAG Features in Detail

### 1. Cross-Database References
Every code chunk gets a deterministic ID:
```python
# File: /src/auth/user.py, Lines: 10-20
# SHA256: 78cfbc8d16c6c84a3ea845f1850f8740...
# Neo4j ID: Full 64-char hex string
# Qdrant ID: First 16 hex chars as numeric (fits int64)
```

### 2. Hybrid Search Patterns
- **Semantic → Graph**: Find similar vectors, then traverse relationships
- **Graph → Semantic**: Start from nodes, find semantically similar content  
- **Combined**: Merge relevance scores from both sources

### 3. Event Debouncing
Handles rapid file changes intelligently:
- 3-second delay batches multiple events
- Processes each file only once per batch
- Prevents index thrashing during git operations

## 🎯 Use Cases

1. **Code Understanding**: Find related functions across semantic and structural relationships
2. **Impact Analysis**: Trace changes through both call graphs and semantic similarity
3. **Dependency Discovery**: Find implicit dependencies via usage patterns
4. **Context-Aware Search**: Get richer results combining vector similarity + graph structure

## 🚧 Requirements

- **Docker**: For containerization
- **Storage**: ~15GB for full image + data
- **Memory**: 4GB+ recommended for ML models
- **External DBs**: Neo4j + Qdrant for full functionality

## 📦 Distribution

The container is self-contained with all dependencies. Anyone can:

1. Clone this repo
2. Run the docker build command above  
3. Get a working GraphRAG system

**Build time**: ~15-20 minutes (depending on bandwidth for ML packages)

## 🔒 Security Notes

- Runs as root inside container (standard for Docker)
- No external network access required during build
- All dependencies pinned to specific versions
- No secrets or credentials in the image

---

**Built with**: Neo4j 5.28+ | Qdrant 1.10+ | PyTorch 2.8+ | Python 3.11
