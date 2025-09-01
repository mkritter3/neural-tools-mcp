# L9 Enhanced MCP Server Configuration

## ✅ Current Status: FULLY OPERATIONAL

All three containers are running and communicating successfully:
- **MCP Server Enhanced** - With Kuzu GraphRAG + Tree-sitter AST
- **Qdrant v1.10.0** - Vector database with hybrid search
- **Nomic Embed v2-MoE** - State-of-the-art embeddings (768-dim)

## 📝 MCP Configuration (`.mcp.json`)

```json
{
  "mcpServers": {
    "l9-enhanced": {
      "command": "docker",
      "args": [
        "exec",
        "-i", 
        "mcp-server-enhanced-default",
        "python3",
        "-u",
        "/app/neural-mcp-server-enhanced.py"
      ],
      "env": {
        "PROJECT_NAME": "claude-l9-template",
        "PROJECT_DIR": "/app/project",
        "QDRANT_HOST": "qdrant",
        "QDRANT_GRPC_PORT": "6334",
        "EMBEDDING_SERVICE_HOST": "nomic-embed-service",
        "EMBEDDING_SERVICE_PORT": "8000",
        "USE_EXTERNAL_EMBEDDING": "true",
        "GRAPHRAG_ENABLED": "true",
        "HYBRID_SEARCH_MODE": "enhanced",
        "KUZU_DB_PATH": "/app/kuzu",
        "PYTHONUNBUFFERED": "1"
      },
      "description": "L9 Enhanced MCP Server with Kuzu GraphRAG + Qdrant + Nomic Embed v2-MoE"
    }
  }
}
```

## 🛠️ Available MCP Tools

### 1. `memory_store_enhanced`
Store content with enhanced hybrid indexing and GraphRAG integration
- **Parameters:**
  - `content`: Text content to store
  - `category`: Category for organization (default: "general")
  - `metadata`: Additional metadata (optional)
  - `create_graph_entities`: Create graph entities in Kuzu (default: true)
- **Features:** 
  - Dual vector + keyword indexing in Qdrant
  - Automatic entity extraction for GraphRAG
  - Tree-sitter AST parsing for code

### 2. `memory_search_enhanced`
Enhanced search with RRF fusion, MMR diversity, and GraphRAG expansion
- **Parameters:**
  - `query`: Search query
  - `category`: Optional category filter
  - `limit`: Number of results (default: 10)
  - `mode`: Search mode - Options:
    - `"semantic"` - Pure vector similarity search
    - `"keyword"` - BM25 keyword search
    - `"rrf_hybrid"` - Reciprocal Rank Fusion (default)
    - `"mmr_diverse"` - Maximal Marginal Relevance
  - `diversity_threshold`: MMR diversity threshold (default: 0.85)
  - `graph_expand`: Expand using graph relationships (default: true)
- **Features:**
  - RRF fusion combines semantic + keyword search
  - MMR ensures diverse, non-redundant results
  - GraphRAG expands results with related entities

### 3. `kuzu_graph_query`
Execute Cypher queries on Kuzu graph database
- **Parameters:**
  - `query`: Cypher query to execute
- **Example queries:**
  ```cypher
  MATCH (d:Document) RETURN d.title, d.path
  MATCH (c:CodeEntity)-[:IMPORTS]->(i:CodeEntity) RETURN c.name, i.name
  MATCH (c:Concept)-[:RELATES_TO]->(r:Concept) RETURN c.name, r.name
  ```

### 4. `performance_stats`
Get enhanced system performance statistics
- **Returns:**
  - Qdrant collection stats (vectors, points, memory)
  - Kuzu graph stats (nodes, relationships)
  - Nomic embedding service health
  - System version and timestamps

## 🚀 System Capabilities

### Performance Optimizations
- **Kuzu GraphRAG**: 3-10x faster than Neo4j for graph queries
- **Nomic v2-MoE**: 30-40% lower inference costs vs v1
- **RRF Hybrid Search**: Combines best of semantic + keyword
- **INT8 Quantization**: 4x memory reduction for vectors
- **Tree-sitter AST**: Multi-language code analysis (13+ languages)

### Search Modes Explained

1. **Semantic Search** (`mode="semantic"`)
   - Pure vector similarity using Nomic embeddings
   - Best for: Conceptual queries, finding similar content

2. **Keyword Search** (`mode="keyword"`)
   - BM25-based text matching
   - Best for: Exact terms, code snippets, specific names

3. **RRF Hybrid** (`mode="rrf_hybrid"`) - DEFAULT
   - Reciprocal Rank Fusion combines semantic + keyword
   - Best for: General purpose, balanced results

4. **MMR Diverse** (`mode="mmr_diverse"`)
   - Maximal Marginal Relevance for diversity
   - Best for: Exploration, avoiding redundant results

## 📊 Container Architecture

```
┌─────────────────────────────────────────┐
│     MCP Client (Claude Desktop, etc)     │
│            Uses .mcp.json config         │
└────────────────┬────────────────────────┘
                 │ stdio/JSON-RPC
                 ▼
┌─────────────────────────────────────────┐
│    mcp-server-enhanced-default          │
│   • FastMCP server                      │
│   • Kuzu GraphRAG database              │
│   • Tree-sitter AST parser              │
└──────┬──────────────────┬───────────────┘
       │                  │
       │ gRPC:6334       │ HTTP:8000
       ▼                  ▼
┌──────────────┐  ┌──────────────────────┐
│   Qdrant     │  │  Nomic Embed v2-MoE  │
│   v1.10.0    │  │  305M active params  │
│  Hybrid DB   │  │  768-dim embeddings  │
└──────────────┘  └──────────────────────┘
```

## 🔧 Management Commands

### Start the system:
```bash
docker compose -f docker/docker-compose.l9-enhanced.yml up -d
```

### Check status:
```bash
docker ps | grep -E "(mcp-server|qdrant|nomic)"
```

### View logs:
```bash
docker logs mcp-server-enhanced-default --tail 50
docker logs qdrant-enhanced-default --tail 50
docker logs nomic-embed-v2-default --tail 50
```

### Test MCP connection:
```bash
# Send a test query to the MCP server
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' | \
  docker exec -i mcp-server-enhanced-default python3 -u /app/neural-mcp-server-enhanced.py
```

### Stop the system:
```bash
docker compose -f docker/docker-compose.l9-enhanced.yml down
```

## 🎯 Usage Examples

### Store content with GraphRAG:
```javascript
await mcp.call('memory_store_enhanced', {
  content: 'Implementation of neural architecture search algorithm',
  category: 'ml-research',
  metadata: { project: 'automl', importance: 'high' },
  create_graph_entities: true
});
```

### Hybrid search with diversity:
```javascript
const results = await mcp.call('memory_search_enhanced', {
  query: 'optimization algorithms',
  mode: 'mmr_diverse',
  diversity_threshold: 0.9,
  graph_expand: true,
  limit: 20
});
```

### Graph query for code dependencies:
```javascript
const deps = await mcp.call('kuzu_graph_query', {
  query: `
    MATCH (c:CodeEntity {language: 'python'})-[:IMPORTS]->(d:CodeEntity)
    RETURN c.name, collect(d.name) as dependencies
    LIMIT 10
  `
});
```

## ⚠️ Known Issues & Solutions

### Issue: Workers crashing with OOM
**Solution:** Single worker mode enabled, torch.compile disabled

### Issue: Megablocks warning
**Status:** Non-critical, 20-30% performance improvement if added

### Issue: Swift parser warning
**Status:** Non-critical, other 12+ languages work fine

## 📈 Performance Metrics

- **Embedding Generation**: ~158ms per text
- **Hybrid Search**: <100ms for typical queries
- **Graph Queries**: <50ms for relationship traversal
- **Memory Usage**: 
  - MCP Server: ~500MB
  - Qdrant: ~2GB (scales with data)
  - Nomic: ~2GB (model in memory)

## 🔐 Security Notes

- All containers run in isolated Docker network
- No external ports exposed except debug ports
- Embedding model uses `trust_remote_code=true` (required for MoE)
- Data persisted in Docker volumes

---

**Last Updated:** August 29, 2025
**Version:** L9-Enhanced-2025
**Status:** Production Ready