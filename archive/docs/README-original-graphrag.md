# L9 GraphRAG

> **Production-ready Graph Retrieval Augmented Generation system with Neo4j and Qdrant**

[![Build Status](https://github.com/your-org/l9-graphrag/workflows/CI/badge.svg)](https://github.com/your-org/l9-graphrag/actions)
[![Coverage Status](https://codecov.io/gh/your-org/l9-graphrag/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/l9-graphrag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/l9-graphrag.git
cd l9-graphrag

# Install dependencies
pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run with Docker
docker compose up -d
```

## 🎯 What is GraphRAG?

GraphRAG (Graph Retrieval Augmented Generation) combines semantic vector search with graph database relationships to provide richer, more contextual code analysis and retrieval.

### Key Benefits

- **🔗 Hybrid Search**: Combines semantic similarity with structural relationships
- **📈 Better Context**: Graph traversal provides deeper code understanding  
- **🎯 Precise Results**: Cross-referenced data eliminates semantic ambiguity
- **⚡ Production Ready**: Optimized for enterprise-scale codebases

## ✨ Features

### Core Capabilities
- **Deterministic Cross-Referencing**: SHA256-based IDs link Neo4j graphs ↔ Qdrant vectors
- **Bidirectional Synchronization**: Real-time consistency between databases
- **Event Debouncing**: Intelligent batching of file system changes
- **Content Deduplication**: Hash-based duplicate detection

### MCP Tools
- `graphrag_hybrid_search` - Semantic search enriched with graph relationships
- `graphrag_impact_analysis` - Analyze code change impacts across dependencies  
- `graphrag_find_dependencies` - Trace dependency chains through vector + graph
- `graphrag_find_related` - Find contextually related code using hybrid patterns

### Infrastructure
- **Docker Support**: Production-ready containerization
- **Monitoring**: Prometheus metrics and structured logging
- **Scalability**: Async processing with configurable concurrency
- **Reliability**: Graceful degradation and error handling

## 📋 Requirements

- **Python**: 3.11 or higher
- **Neo4j**: 5.22.0 or higher
- **Qdrant**: 1.10.0 or higher
- **Docker**: 20.10+ (for containerized deployment)
- **Memory**: 4GB+ recommended for ML models
- **Storage**: 15GB+ for dependencies and data

## 🛠 Installation

### Development Installation

```bash
# Clone and install
git clone https://github.com/your-org/l9-graphrag.git
cd l9-graphrag
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Production Deployment

```bash
# Using Docker (recommended)
docker compose up -d

# Or build custom image
docker build -f docker/Dockerfile -t l9-graphrag:production .
docker run -d --name l9-graphrag -p 3000:3000 l9-graphrag:production
```

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

## 📖 Usage

### Basic Usage

```python
from graphrag import HybridRetriever

# Initialize retriever
retriever = HybridRetriever(
    neo4j_uri="bolt://localhost:7687",
    qdrant_host="localhost"
)

# Hybrid search
results = await retriever.find_similar_with_context(
    query="authentication middleware",
    limit=5
)
```

### MCP Server

```bash
# Start MCP server
python -m mcp.neural_server_stdio

# Use with Claude Code or other MCP clients
```

### Docker Compose

```yaml
version: '3.8'
services:
  l9-graphrag:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - QDRANT_HOST=qdrant
    depends_on:
      - neo4j
      - qdrant
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run performance tests
pytest tests/performance/ -v
```

## 📚 Documentation

- **[Architecture Guide](docs/architecture/)** - System design and components
- **[API Reference](docs/api/)** - Complete API documentation
- **[User Guide](docs/guides/)** - Usage examples and best practices
- **[Development](docs/development/)** - Contributing and development setup

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📊 Performance

- **Index Speed**: 1000+ files/minute
- **Query Latency**: <100ms p95
- **Memory Usage**: ~2GB for 100k code chunks
- **Concurrent Users**: 100+ (tested)

## 🔒 Security

- No credentials stored in code
- Environment-based configuration
- Secure database connections
- Input validation and sanitization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 Enterprise Support

For enterprise deployments, custom integrations, or professional support, please contact [enterprise@your-org.com](mailto:enterprise@your-org.com).

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and releases.

---

**Built with ❤️ by the L9 Engineering Team**

*Neo4j 5.28+ | Qdrant 1.10+ | PyTorch 2.8+ | Python 3.11*