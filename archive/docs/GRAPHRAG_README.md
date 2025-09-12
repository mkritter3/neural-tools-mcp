# GraphRAG Implementation - L9 Neural Tools

## 🎯 What is GraphRAG?

GraphRAG (Graph Retrieval Augmented Generation) combines semantic vector search with graph database relationships to provide richer, more contextual code analysis and retrieval.

## ✅ Implementation Complete

This project now includes a complete GraphRAG system that:

- **Links Neo4j ↔ Qdrant** with deterministic SHA256-based IDs
- **Provides 4 new MCP tools** for hybrid search and analysis
- **Handles file system monitoring** with intelligent debouncing
- **Runs in Docker** with all dependencies included

## 🚀 Quick Start

```bash
# Build the container
docker build -f neural-tools/Dockerfile.l9-minimal -t l9-mcp-graphrag:production neural-tools/

# Run it
docker run -d --name l9-graphrag l9-mcp-graphrag:production
```

## 📁 Key Files

### Core Implementation
- `neural-tools/src/services/hybrid_retriever.py` - GraphRAG query engine
- `neural-tools/src/services/indexer_service.py` - Enhanced indexer with GraphRAG
- `src/mcp/neural_server_stdio.py` - MCP server with GraphRAG tools (canonical)

### Configuration
- `neural-tools/config/requirements-l9-enhanced.txt` - All dependencies (includes `watchdog`)
- `neural-tools/Dockerfile.l9-minimal` - Production Docker build

### Testing
- `neural-tools/test_graphrag_simple.py` - Verification tests for core logic

## 🔗 MCP Tools Added

1. **`graphrag_hybrid_search`** - Semantic search enriched with graph relationships
2. **`graphrag_impact_analysis`** - Analyze code change impacts across dependencies  
3. **`graphrag_find_dependencies`** - Trace dependency chains through vector + graph
4. **`graphrag_find_related`** - Find contextually related code using hybrid patterns

## 🏗 Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   Neo4j Graph   │◄──►│  Qdrant Vectors  │
│  (Relationships)│    │ (Semantic Search)│
└─────────────────┘    └──────────────────┘
         ▲                        ▲
         │    SHA256-based IDs     │
         │                        │
    ┌────┴────────────────────────┴────┐
    │        GraphRAG Engine          │
    │    (HybridRetriever)            │
    └─────────────┬───────────────────┘
                  │
    ┌─────────────┴───────────────────┐
    │         MCP Server              │
    │  (4 GraphRAG Tools)             │
    └─────────────────────────────────┘
```

## 📚 Documentation

- **[GRAPHRAG_DOCKER_GUIDE.md](GRAPHRAG_DOCKER_GUIDE.md)** - Complete build & usage guide
- **[CLAUDE.md](CLAUDE.md)** - L9 engineering standards and verification protocols

## 🎉 Ready for Production

The implementation is complete, tested, and containerized. Anyone can:

1. Clone this repository
2. Run the Docker build command
3. Get a working GraphRAG system with all advanced features

**Features**: Deterministic cross-referencing, event debouncing, content deduplication, hybrid search patterns, and 4 production-ready MCP tools.

---

*Built with Neo4j 5.28+ | Qdrant 1.10+ | PyTorch 2.8+ | Python 3.11*
