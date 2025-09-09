# 🚀 Neural Tools MCP Server - Quick Start Guide

## What is This?
Neural Tools is a GraphRAG-powered MCP (Model Context Protocol) server that indexes and searches your codebase using:
- **Neo4j** for graph relationships between code entities
- **Qdrant** for semantic vector search
- **Nomic Embed v2** for generating embeddings
- **MCP Tools** for AI assistants to query and analyze your code

## 5-Minute Setup

### Prerequisites
- Docker & Docker Compose installed
- 16GB RAM recommended (minimum 8GB)
- 10GB free disk space

### Step 1: Clone & Navigate
```bash
git clone [your-repo]
cd neural-tools
```

### Step 2: Configure Your Project
```bash
# Create environment file
cp config/.env.example config/.env

# Edit config/.env and set:
PROJECT_NAME=my_project        # Unique name for your project
PROJECT_DIR=../                # Path to code you want to index
```

### Step 3: Start Everything
```bash
# For development (live code updates):
./build-and-run.sh --dev

# For production (stable):
./build-and-run.sh
```

### Step 4: Verify It's Running
```bash
# Check services are up
docker-compose -f config/docker-compose.neural-tools.yml ps

# Should show 4 services running:
# - neural-tools-server (MCP server)
# - neural-data-storage (Qdrant)
# - neural-embeddings (Nomic)
# - neo4j-graph (Neo4j)
```

## Index Your Codebase

### Via MCP Client (Claude, VS Code, etc.)
Once connected to the MCP server, use:
```json
{
  "tool": "project_indexer",
  "arguments": {
    "path": "/app/project",
    "recursive": true
  }
}
```

### Check Indexing Status
```json
{
  "tool": "neural_system_status",
  "arguments": {}
}
```

## Search Your Code

### Semantic Search
Find code by meaning, not just keywords:
```json
{
  "tool": "semantic_code_search",
  "arguments": {
    "query": "authentication logic for user login",
    "max_results": 10
  }
}
```

### GraphRAG Hybrid Search
Combines semantic + graph relationships:
```json
{
  "tool": "graphrag_hybrid_search",
  "arguments": {
    "query": "database connection handling",
    "include_dependencies": true
  }
}
```

### Find Dependencies
Trace what depends on what:
```json
{
  "tool": "graphrag_find_dependencies",
  "arguments": {
    "entity_name": "UserService"
  }
}
```

## Development Workflow

### Making Changes to MCP Tools

1. **Edit the code**:
```bash
vim src/mcp/neural_server_stdio.py
```

2. **If in dev mode** - changes apply immediately!

3. **If in production** - rebuild:
```bash
./build-and-run.sh --rebuild
```

### Adding a New MCP Tool

1. Add tool definition in `src/mcp/neural_server_stdio.py`:
```python
# In __init__ method
types.Tool(
    name="my_new_tool",
    description="What it does",
    inputSchema={...}
)

# Add implementation method
async def my_new_tool_impl(self, arguments: dict) -> list:
    # Your logic here
    return [types.TextContent(type="text", text="Result")]
```

2. Test in dev mode:
```bash
./build-and-run.sh --dev
# Test your tool via MCP client
```

3. Deploy to production:
```bash
./build-and-run.sh --rebuild
```

## Available MCP Tools

| Tool | Purpose |
|------|---------|
| `project_indexer` | Index your codebase into GraphRAG |
| `semantic_code_search` | Search by meaning |
| `graphrag_hybrid_search` | Semantic + graph search |
| `graphrag_find_dependencies` | Find what uses what |
| `graphrag_find_related` | Find related code |
| `graphrag_impact_analysis` | Analyze change impacts |
| `neo4j_graph_query` | Direct Cypher queries |
| `neural_system_status` | Check system health |

## Quick Commands

### View Logs
```bash
# All services
docker-compose -f config/docker-compose.neural-tools.yml logs -f

# Specific service
docker-compose -f config/docker-compose.neural-tools.yml logs -f neural-tools-server
```

### Stop Everything
```bash
docker-compose -f config/docker-compose.neural-tools.yml down
```

### Restart a Service
```bash
docker-compose -f config/docker-compose.neural-tools.yml restart neural-tools-server
```

### Access Web UIs
- **Neo4j Browser**: http://localhost:7475 (neo4j/neural-l9-2025)
- **Qdrant Dashboard**: http://localhost:6681/dashboard
- **Nomic API Docs**: http://localhost:8081/docs

## Troubleshooting

### "Container not found"
```bash
# Ensure services are running
./build-and-run.sh
```

### "Import error" in logs
```bash
# Rebuild with latest code
./build-and-run.sh --rebuild
```

### "Cannot connect to Neo4j/Qdrant"
```bash
# Check service health
docker-compose -f config/docker-compose.neural-tools.yml ps

# Restart problematic service
docker-compose -f config/docker-compose.neural-tools.yml restart [service-name]
```

### "Indexing not working"
```bash
# Check if services are ready
docker exec -it default-neural python3 -c "
from src.services.service_manager import ServiceManager
sm = ServiceManager()
print(sm.get_health_status())
"
```

## Architecture Overview

```
Your Project Directory
         |
         v
   [MCP Client]
         |
    MCP Protocol
         |
         v
 [neural-tools-server]
         |
    +----+----+
    |         |
    v         v
[Neo4j]   [Qdrant]
 Graph     Vector
  DB        DB
    |         |
    +----+----+
         |
         v
   [Nomic Embed]
   Embedding Service
```

## File Structure
```
neural-tools/
├── src/
│   ├── servers/
│   │   └── neural_server_stdio.py  # MCP server & tools (canonical)
│   └── services/
│       ├── indexer_service.py      # File indexing
│       ├── neo4j_service.py        # Graph operations
│       └── qdrant_service.py       # Vector operations
├── config/
│   ├── docker-compose.neural-tools.yml      # Main config
│   ├── docker-compose.neural-tools.dev.yml  # Dev override
│   └── .env                                  # Your settings
├── build-and-run.sh                # Start script
└── QUICKSTART.md                   # This file
```

## Next Steps

1. **Index your code**: Use `project_indexer` tool
2. **Try searches**: Test different search tools
3. **Explore the graph**: Open Neo4j browser
4. **Add custom tools**: Extend `src/mcp/neural_server_stdio.py`

## Common Workflows

### Daily Development
```bash
# Morning startup
./build-and-run.sh --dev

# Work on code...
# Changes apply immediately in dev mode

# Evening shutdown
docker-compose -f config/docker-compose.neural-tools.yml down
```

### Production Deployment
```bash
# Build and deploy
./build-and-run.sh --rebuild

# Monitor
docker-compose -f config/docker-compose.neural-tools.yml logs -f
```

### Adding New File Types to Index
Edit `src/services/indexer_service.py`:
```python
WATCH_PATTERNS = {
    '.py', '.js', '.ts',  # existing
    '.proto', '.graphql'   # add new
}
```

## Get Help

- **Logs**: Check `docker-compose logs` for errors
- **Status**: Use `neural_system_status` tool
- **Web UIs**: Neo4j and Qdrant dashboards
- **Test Script**: Run `./test-systematic-fix.sh`

---

**Ready to code?** Your GraphRAG-powered codebase search is just a `./build-and-run.sh` away! 🎉
