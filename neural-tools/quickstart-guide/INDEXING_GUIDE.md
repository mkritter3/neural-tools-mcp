# GraphRAG Project Indexing Guide

## Overview
The Neural Tools MCP server includes a systematic indexing capability that processes all project files and stores them in Neo4j (graph database) and Qdrant (vector database) for GraphRAG-powered semantic search.

## How It Works

### File Processing
The indexer handles multiple file types intelligently:
- **Code files** (`.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, etc.): Chunks at logical boundaries (functions, classes)
- **Documentation** (`.md`, `.txt`): Chunks by paragraphs/sections
- **Configuration** (`.json`, `.yaml`, `.toml`): Processes as structured data
- **All files**: Creates embeddings using Nomic Embed v2-MoE

### Storage Architecture
1. **Neo4j Graph Database**: Stores file relationships, code entities (classes, methods, functions), and dependencies
2. **Qdrant Vector Database**: Stores semantic embeddings for similarity search
3. **Hybrid Search**: Combines graph traversal with semantic similarity for powerful code search

## Usage

### Via MCP Tools (Recommended)

Once the MCP server is running, use the `project_indexer` tool:

```json
{
  "tool": "project_indexer",
  "arguments": {
    "path": "/app/project",        // Path to index (default: entire project)
    "recursive": true,              // Index subdirectories (default: true)
    "clear_existing": false,        // Clear existing index first (default: false)
    "file_patterns": [".py", ".js"], // Optional: specific file types
    "force_reindex": false          // Force re-index unchanged files (default: false)
  }
}
```

### Docker Setup

The docker-compose configuration automatically mounts your project:

```yaml
# neural-tools/config/docker-compose.neural-tools.yml
services:
  neural-tools-server:
    volumes:
      - type: bind
        source: ${PROJECT_DIR:-../}  # Your project directory
        target: /app/project          # Mounted in container
        read_only: true
```

### Environment Variables

Set in your `.env` file:

```bash
# Project configuration
PROJECT_NAME=my_project        # Unique project identifier
PROJECT_DIR=/path/to/project   # Path to your project

# Service configuration (usually defaults are fine)
NEO4J_PASSWORD=neural-l9-2025
QDRANT_DEBUG_PORT=6681
NEO4J_DEBUG_HTTP_PORT=7475
```

## Systematic Workflow

1. **Start the services**:
   ```bash
   cd neural-tools/config
   docker-compose -f docker-compose.neural-tools.yml up -d
   ```

2. **Wait for services to initialize** (check logs):
   ```bash
   docker-compose logs -f neural-tools-server
   ```

3. **Trigger indexing** via MCP:
   - Use your MCP client to call the `project_indexer` tool
   - Or use the neural system status tool to verify services first

4. **Verify indexing**:
   - Check Neo4j: http://localhost:7475 (neo4j/neural-l9-2025)
   - Check Qdrant: http://localhost:6681/dashboard
   - Use `neo4j_graph_query` tool to query the graph
   - Use `semantic_code_search` tool to test search

## Features

### Intelligent Chunking
- **Code-aware**: Breaks at function/class boundaries
- **Size-limited**: Max 100 lines per chunk for precision
- **Overlap**: Maintains context between chunks

### Deduplication
- Uses file hashes to skip unchanged files
- Cooldown period prevents rapid re-indexing
- Force reindex option available when needed

### Graceful Degradation
- Continues if services are unavailable
- Falls back to keyword indexing without embeddings
- Tracks service health and failures

### File Filtering
Default patterns included:
```python
watch_patterns = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go',
    '.rs', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php',
    '.md', '.yaml', '.yml', '.json', '.toml', '.txt', '.sql'
}
```

Ignored patterns:
```python
ignore_patterns = {
    '__pycache__', '.git', 'node_modules', '.venv', 'venv',
    'dist', 'build', '.next', 'target', '.pytest_cache',
    '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.log',
    '.env', '.env.local', '.env.production', 'secrets',
    '*.key', '*.pem', '*.crt', '.neural-tools'
}
```

## Monitoring

### Indexing Statistics
The indexer returns comprehensive statistics:
```json
{
  "files_indexed": 150,
  "files_skipped": 10,
  "chunks_created": 1500,
  "total_documents_in_graph": 150,
  "total_chunks_in_graph": 1500,
  "total_entities_in_graph": 750,
  "neo4j_available": true,
  "qdrant_available": true,
  "nomic_available": true
}
```

### Service Health
Use `neural_system_status` tool to check:
- Neo4j connection and document count
- Qdrant connection and vector count
- Nomic embedding service availability

## Troubleshooting

### Path Not Found
- Ensure `PROJECT_DIR` is set correctly in `.env`
- Check docker volume mounting in logs
- Try absolute paths in the indexer tool

### Services Unavailable
- Check container status: `docker-compose ps`
- View logs: `docker-compose logs [service-name]`
- Ensure ports aren't in use: `lsof -i :7475 -i :6681`

### Indexing Errors
- Check file permissions
- Verify file encoding (UTF-8 expected)
- Look for syntax errors in code files
- Check service logs for detailed errors

### Performance Issues
- Adjust batch size in indexer settings
- Increase memory limits in docker-compose
- Consider indexing in stages for large projects

## Best Practices

1. **Initial Indexing**: Run with `clear_existing: true` for clean start
2. **Incremental Updates**: Use file watcher for continuous indexing
3. **Large Projects**: Index in stages by directory
4. **CI/CD Integration**: Trigger indexing after deployments
5. **Regular Maintenance**: Periodically clear and rebuild index

## Advanced Usage

### Custom File Patterns
```json
{
  "file_patterns": [".proto", ".graphql", ".sol"]
}
```

### Partial Indexing
Index specific directories:
```json
{
  "path": "/app/project/src/services",
  "recursive": true
}
```

### Force Re-indexing
When code structure changes significantly:
```json
{
  "force_reindex": true,
  "clear_existing": true
}
```

## Integration with GraphRAG Tools

Once indexed, use these MCP tools:
- `graphrag_hybrid_search`: Semantic + graph search
- `graphrag_impact_analysis`: Analyze change impacts
- `graphrag_find_dependencies`: Trace dependencies
- `graphrag_find_related`: Discover related code
- `semantic_code_search`: Pure semantic search
- `neo4j_graph_query`: Direct Cypher queries

## Architecture Benefits

This systematic approach provides:
- **Consistency**: Same indexing logic for all file types
- **Scalability**: Handles projects of any size
- **Reliability**: Graceful degradation and error recovery
- **Flexibility**: Configurable patterns and behaviors
- **Performance**: Intelligent chunking and caching
- **Integration**: Works seamlessly with GraphRAG tools

The indexer is production-ready and designed for continuous operation in development environments.