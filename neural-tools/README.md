# Neural Tools MCP Server

A GraphRAG-powered MCP (Model Context Protocol) server that provides AI assistants with intelligent code search and analysis capabilities.

## 🎯 Quick Start

**Get running in 5 minutes:**
```bash
cd neural-tools
./build-and-run.sh --dev
```

See [QUICKSTART.md](QUICKSTART.md) for the complete setup guide.

## 🔥 Key Features

- **GraphRAG Search**: Combines Neo4j graph relationships with Qdrant semantic vectors
- **Smart Indexing**: Automatically chunks and indexes all code files  
- **MCP Tools**: 10+ tools for code search, dependency analysis, and impact assessment
- **Live Development**: Hot-reload code changes without rebuilding containers
- **Production Ready**: Systematic build process with health checks and monitoring

## 📚 Documentation

- [QUICKSTART.md](QUICKSTART.md) - Get started in 5 minutes
- [INDEXING_GUIDE.md](INDEXING_GUIDE.md) - How to index your codebase
- [SYSTEMATIC_BUILD_GUIDE.md](SYSTEMATIC_BUILD_GUIDE.md) - Development and deployment workflows
- [MCP_TOOLS.md](MCP_TOOLS.md) - Complete list of available tools

## 🏗️ Architecture

```
MCP Client (Claude, VS Code, etc.)
           ↓
    Neural MCP Server
           ↓
    ┌──────┴──────┐
    ↓             ↓
  Neo4j        Qdrant
  (Graph)      (Vector)
    ↓             ↓
    └──────┬──────┘
           ↓
      Nomic Embed
      (Embeddings)
```

## 🛠️ Development

### Add a New Tool
1. Edit `src/servers/neural_server_stdio.py`
2. Test with `./build-and-run.sh --dev`
3. Deploy with `./build-and-run.sh --rebuild`

### Project Structure
```
neural-tools/
├── src/              # Source code
├── config/           # Docker & environment configs
├── build-and-run.sh  # Main script
└── docs/             # Documentation
```

## 🚀 Commands

```bash
# Start development mode (live code updates)
./build-and-run.sh --dev

# Build and run production
./build-and-run.sh --rebuild

# View logs
docker-compose -f config/docker-compose.neural-tools.yml logs -f

# Stop everything
docker-compose -f config/docker-compose.neural-tools.yml down
```

## 📊 Services

| Service | Purpose | Port |
|---------|---------|------|
| neural-tools-server | MCP server with GraphRAG tools | - |
| neo4j-graph | Graph database for code relationships | 7475 |
| neural-data-storage | Qdrant vector database | 6681 |
| neural-embeddings | Nomic Embed v2 service | 8081 |

## 🔧 Requirements

- Docker & Docker Compose
- 16GB RAM (8GB minimum)
- 10GB disk space

## 📝 License

[Your License]

---

**Built for developers who want AI assistants to truly understand their code.** 🧠