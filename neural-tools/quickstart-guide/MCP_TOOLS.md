# MCP Tools Reference

Complete reference for all MCP tools available in the Neural Tools server.

## 🔍 Search Tools

### `semantic_code_search`
Search code by meaning using vector similarity.

**Arguments:**
- `query` (string, required): Natural language search query
- `max_results` (integer): Maximum results to return (default: 10)
- `threshold` (float): Similarity threshold 0-1 (default: 0.7)
- `file_types` (array): Filter by file extensions

**Example:**
```json
{
  "tool": "semantic_code_search",
  "arguments": {
    "query": "function that handles user authentication",
    "max_results": 5,
    "file_types": [".py", ".js"]
  }
}
```

### `graphrag_hybrid_search`
Combines semantic search with graph relationships for more intelligent results.

**Arguments:**
- `query` (string, required): Search query
- `max_results` (integer): Maximum results (default: 10)
- `include_dependencies` (boolean): Include dependent code (default: false)
- `search_depth` (integer): Graph traversal depth (default: 2)

**Example:**
```json
{
  "tool": "graphrag_hybrid_search",
  "arguments": {
    "query": "database connection management",
    "include_dependencies": true,
    "search_depth": 3
  }
}
```

## 📊 Analysis Tools

### `graphrag_find_dependencies`
Find all code that depends on a specific entity.

**Arguments:**
- `entity_name` (string, required): Name of class/function/module
- `dependency_type` (string): Type filter (imports/calls/inherits)
- `max_depth` (integer): How deep to traverse (default: 3)

**Example:**
```json
{
  "tool": "graphrag_find_dependencies",
  "arguments": {
    "entity_name": "UserService",
    "dependency_type": "imports"
  }
}
```

### `graphrag_find_related`
Discover code related to a given entity through various relationships.

**Arguments:**
- `entity_name` (string, required): Starting entity
- `relationship_types` (array): Types to follow (default: all)
- `max_results` (integer): Limit results (default: 20)

**Example:**
```json
{
  "tool": "graphrag_find_related",
  "arguments": {
    "entity_name": "AuthenticationMiddleware",
    "relationship_types": ["CALLS", "IMPORTS", "INHERITS"]
  }
}
```

### `graphrag_impact_analysis`
Analyze the potential impact of changing a piece of code.

**Arguments:**
- `entity_name` (string, required): Entity to analyze
- `change_type` (string): Type of change (modify/delete/rename)
- `include_tests` (boolean): Include test files (default: true)

**Example:**
```json
{
  "tool": "graphrag_impact_analysis",
  "arguments": {
    "entity_name": "calculatePrice",
    "change_type": "modify",
    "include_tests": true
  }
}
```

## 🗂️ Indexing Tools

### `project_indexer`
Index project files into Neo4j and Qdrant for GraphRAG search.

**Arguments:**
- `path` (string): Directory to index (default: /app/project)
- `recursive` (boolean): Include subdirectories (default: true)
- `clear_existing` (boolean): Clear before indexing (default: false)
- `file_patterns` (array): File types to index
- `force_reindex` (boolean): Re-index unchanged files (default: false)

**Example:**
```json
{
  "tool": "project_indexer",
  "arguments": {
    "path": "/app/project/src",
    "recursive": true,
    "file_patterns": [".py", ".ts", ".tsx"],
    "clear_existing": false
  }
}
```

### `file_watcher`
Monitor and automatically index file changes.

**Arguments:**
- `action` (string, required): start/stop/status
- `path` (string): Directory to watch
- `auto_index` (boolean): Auto-index on change (default: true)

**Example:**
```json
{
  "tool": "file_watcher",
  "arguments": {
    "action": "start",
    "path": "/app/project",
    "auto_index": true
  }
}
```

## 🗄️ Database Tools

### `neo4j_graph_query`
Execute Cypher queries directly on the Neo4j graph.

**Arguments:**
- `query` (string, required): Cypher query
- `parameters` (object): Query parameters
- `limit` (integer): Result limit (default: 100)

**Example:**
```json
{
  "tool": "neo4j_graph_query",
  "arguments": {
    "query": "MATCH (f:Function)-[:CALLS]->(g:Function) WHERE f.name = $name RETURN g.name",
    "parameters": {"name": "processOrder"},
    "limit": 50
  }
}
```

### `qdrant_vector_search`
Direct vector similarity search in Qdrant.

**Arguments:**
- `query` (string, required): Search text
- `collection` (string): Collection name
- `limit` (integer): Result limit
- `with_payload` (boolean): Include metadata

**Example:**
```json
{
  "tool": "qdrant_vector_search",
  "arguments": {
    "query": "error handling logic",
    "collection": "code_chunks",
    "limit": 10,
    "with_payload": true
  }
}
```

## 🔧 System Tools

### `neural_system_status`
Check health and statistics of all services.

**Arguments:** None

**Example:**
```json
{
  "tool": "neural_system_status",
  "arguments": {}
}
```

**Returns:**
- Service availability (Neo4j, Qdrant, Nomic)
- Document/vector counts
- Memory usage
- Last index time

### `clear_all_data`
Clear all indexed data from Neo4j and Qdrant.

**Arguments:**
- `confirm` (boolean, required): Safety confirmation
- `keep_schema` (boolean): Preserve graph schema

**Example:**
```json
{
  "tool": "clear_all_data",
  "arguments": {
    "confirm": true,
    "keep_schema": true
  }
}
```

## 📈 Advanced Tools

### `code_complexity_analysis`
Analyze complexity metrics for code.

**Arguments:**
- `path` (string): File or directory path
- `metrics` (array): Which metrics to calculate

**Example:**
```json
{
  "tool": "code_complexity_analysis",
  "arguments": {
    "path": "/app/project/src/services",
    "metrics": ["cyclomatic", "cognitive", "lines"]
  }
}
```

### `find_similar_code`
Find code similar to a given snippet.

**Arguments:**
- `code_snippet` (string, required): Reference code
- `threshold` (float): Similarity threshold
- `language` (string): Language hint

**Example:**
```json
{
  "tool": "find_similar_code",
  "arguments": {
    "code_snippet": "def validate_email(email):\n    return '@' in email",
    "threshold": 0.8,
    "language": "python"
  }
}
```

## 🎯 Best Practices

### Searching
1. Start with `semantic_code_search` for general queries
2. Use `graphrag_hybrid_search` when relationships matter
3. Use `neo4j_graph_query` for specific graph patterns

### Indexing
1. Run `project_indexer` with `clear_existing: true` initially
2. Use `file_watcher` for continuous updates
3. Index incrementally for large projects

### Analysis
1. Use `graphrag_impact_analysis` before major refactoring
2. Combine `graphrag_find_dependencies` with search tools
3. Use `code_complexity_analysis` to identify refactoring targets

## 🔍 Query Examples

### Find all database queries
```json
{
  "tool": "semantic_code_search",
  "arguments": {
    "query": "SQL queries database SELECT INSERT UPDATE",
    "file_types": [".py", ".js", ".ts"]
  }
}
```

### Track authentication flow
```json
{
  "tool": "graphrag_find_dependencies",
  "arguments": {
    "entity_name": "authenticate",
    "dependency_type": "calls",
    "max_depth": 5
  }
}
```

### Identify test coverage
```json
{
  "tool": "neo4j_graph_query",
  "arguments": {
    "query": "MATCH (t:File)-[:TESTS]->(f:Function) WHERE f.name = $func RETURN t.path",
    "parameters": {"func": "processPayment"}
  }
}
```

## 🚨 Error Handling

All tools return structured responses:

**Success:**
```json
{
  "status": "success",
  "data": {...},
  "metadata": {
    "duration": 1.23,
    "result_count": 10
  }
}
```

**Error:**
```json
{
  "status": "error",
  "error": "Service unavailable",
  "details": "Neo4j connection failed",
  "suggestions": ["Check if Neo4j container is running"]
}
```

## 📊 Performance Tips

- Use `threshold` parameters to filter results
- Limit `max_results` for faster responses
- Use `file_types` filters when possible
- Cache frequent queries in your application
- Index incrementally rather than full re-indexes

---

**Need help?** Check system status with `neural_system_status` or view logs for detailed debugging.