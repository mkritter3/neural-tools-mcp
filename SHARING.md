# 🚀 L9 Neural GraphRAG - Sharing Guide

## For Recipients

To get started with this project:

```bash
# 1. Clone the repository
git clone <repository-url> l9-neural-graphrag
cd l9-neural-graphrag

# 2. Run the setup script
chmod +x setup.sh
./setup.sh

# That's it! The system will be running and configured.
```

## What You Get

✅ **Complete Neural GraphRAG System**
- Pattern-based metadata extraction (no LLM needed)
- Neural embeddings (Nomic 768-dim vectors)
- GraphRAG with Neo4j relationships
- Vector search with Qdrant
- Redis caching and queue system
- Auto-indexing of your codebase

✅ **No External Dependencies**
- Everything runs locally in Docker
- No API keys required
- No cloud services needed
- Fully self-contained

## System Requirements

- Docker Desktop installed and running
- Python 3.9+
- 8GB free RAM minimum
- 15GB free disk space

## Included Services

All services start automatically with `docker-compose up -d`:

| Service | Port | Purpose |
|---------|------|---------|
| Neo4j | 47687 | Graph database |
| Qdrant | 46333 | Vector search |
| Redis Cache | 46379 | Query caching |
| Redis Queue | 46380 | Task queue |
| Nomic Embeddings | 48000 | Text embeddings |
| Neural Indexer | 48080 | Auto-indexing |

## For Claude Users

The MCP server is pre-configured. Once setup completes, you'll have these tools available in Claude:

- `semantic_code_search()` - Search by meaning
- `graphrag_hybrid_search()` - Hybrid search with graph context
- `project_understanding()` - Get project overview
- `neural_system_status()` - Check system health

## Optional: Global Installation

Want neural-tools available in ALL your projects?

```bash
./scripts/install-global-mcp.sh
```

## Support

- Check `docs/adr/` for architecture decisions
- Run `docker-compose logs -f` for debugging
- Ensure Docker has at least 4GB memory allocated

---

This is a production-ready L9 2025 system following enterprise engineering standards.