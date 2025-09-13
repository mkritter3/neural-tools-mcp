# Installation Guide

## Quick Install (30 seconds)

```bash
git clone https://github.com/YOUR_USERNAME/l9-neural-graphrag.git
cd l9-neural-graphrag
./setup.sh
```

That's it! The system is now ready.

## Manual Installation Steps

If you prefer to understand each step:

### 1. Prerequisites

- **Docker Desktop** (8GB+ memory allocation)
- **Python 3.9+**
- **Claude Desktop** (free account)

### 2. Clone & Dependencies

```bash
git clone <repo-url>
cd l9-neural-graphrag
pip3 install neo4j==5.22.0 qdrant-client==1.15.1 nomic redis asyncio
```

### 3. Start Services

```bash
docker-compose up -d
```

Wait ~2 minutes for all services to be healthy.

### 4. Verify Installation

```bash
# Check all services running
docker-compose ps

# Test Neo4j connection
curl -u neo4j:graphrag-password http://localhost:47687

# Test Qdrant
curl http://localhost:46333/collections

# Test embeddings
curl http://localhost:48000/health
```

### 5. Connect to Claude

The `.mcp.json` file automatically configures Claude to use neural-tools.

Just restart Claude Desktop and the tools will be available.

## Troubleshooting

### Docker Issues

```bash
# Increase Docker memory to 8GB+ in Docker Desktop settings
# Restart Docker
docker-compose down -v
docker-compose up -d
```

### Port Conflicts

If any ports are in use:

```bash
# Find what's using the port
lsof -i :47687  # or other port

# Kill the process or change ports in docker-compose.yml
```

### Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r neural-tools/requirements.txt
```

## Verification Commands

```bash
# System health
docker-compose ps | grep -v Exit

# Service logs
docker-compose logs neo4j
docker-compose logs qdrant
docker-compose logs neural-indexer

# Test indexing
cd neural-tools
python -c "
from src.servers.services.service_container import ServiceContainer
container = ServiceContainer()
print('✅ Neural tools ready!')
"
```

## Next Steps

1. Open Claude Desktop
2. Navigate to any project directory
3. Use semantic search: `semantic_code_search("authentication logic")`
4. Check project status: `neural_system_status()`

The system automatically detects your current project and creates isolated collections.