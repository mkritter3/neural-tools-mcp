# ADR-0013: Semantic Code Search Using Extracted Symbols

**Status:** Proposed  
**Date:** 2025-09-09  
**Deciders:** Engineering Team  

## Context

We have tree-sitter symbol extraction working and storing symbols in Neo4j with vector embeddings in Qdrant. We need to leverage this infrastructure to provide semantic code search capabilities through our existing MCP server and neural indexer.

## Decision

Enhance the **existing** MCP server (`neural-tools/src/servers/mcp_server.py`) and Qdrant service to provide semantic symbol search using our already-indexed data.

## Integration Approach

### 1. Enhance Existing MCP Server
```python
# EDIT neural-tools/src/servers/mcp_server.py
# Add to existing tools array:
Tool(
    name="search_symbols",
    description="Search for code symbols semantically",
    parameters={
        "query": str,
        "symbol_types": Optional[List[str]],  # function, class, method, etc.
        "languages": Optional[List[str]],
        "limit": int
    }
)

# Add handler method:
async def handle_search_symbols(params):
    # Use existing qdrant_service
    # Query with symbol-specific namespace
    # Return ranked results with context
```

### 2. Modify Qdrant Service
```python
# EDIT neural-tools/src/servers/services/qdrant_service.py
# Enhance existing search method:
async def search_vectors(
    self,
    collection_name: str,
    query_vector: List[float],
    namespace: str,
    filter_conditions: Optional[Dict] = None,  # NEW: symbol type filters
    with_payload: bool = True,
    limit: int = 10
):
    # Add filtering by symbol_type, language, parent_class
    # Use existing Qdrant Filter API
    # Return enriched results
```

### 3. Enhance Indexer Service
```python
# EDIT neural-tools/src/servers/services/indexer_service.py
# Modify _index_semantic to include richer metadata:
async def _index_semantic(self, file_path: str, content: str):
    # Existing embedding logic...
    
    # Enhanced payload with symbol context:
    payload = {
        "file_path": file_path,
        "content": content_chunk,
        "symbol_name": symbol.get('name'),
        "symbol_type": symbol.get('type'),
        "symbol_context": self._get_symbol_context(symbol),  # NEW
        "language": symbol.get('language'),
        "line_range": [symbol['start_line'], symbol['end_line']]
    }
```

### 4. Add Symbol Context Enrichment
```python
# EDIT neural-tools/src/servers/services/neo4j_service.py
# Add method to existing Neo4jService:
async def get_symbol_relationships(self, symbol_id: str):
    """Get related symbols for context"""
    query = """
    MATCH (s:Symbol {id: $id})-[r]-(related:Symbol)
    RETURN related, type(r) as relationship
    LIMIT 10
    """
    # Return related symbols for search context
```

## Implementation Details

### Search Flow
1. User queries via MCP: "find authentication functions"
2. MCP server generates embedding via existing embedding service
3. Query Qdrant with filters (symbol_type='function', content contains 'auth')
4. Enrich results with Neo4j relationships
5. Return ranked results with code snippets

### Query Enhancement
```python
# Smart query interpretation in existing MCP handler:
- "class UserService" → filter: symbol_type='class', name contains 'UserService'
- "async functions" → filter: language='python', content contains 'async'
- "TODO comments" → search in content, not symbols
```

### Result Ranking
1. Vector similarity score (primary)
2. Symbol type relevance (boost exact type matches)
3. File path proximity (boost nearby files)
4. Recency (boost recently modified)

## Consequences

### Positive
- Leverages ALL existing infrastructure
- No new services or dependencies
- Works with current Docker setup
- Enhances existing MCP tools naturally

### Negative
- Increased Qdrant query complexity
- More metadata storage per vector
- Potential for slower queries with filters

### Neutral
- Uses existing embedding service
- Maintains current architecture
- Compatible with existing caching

## Testing Strategy

1. Extend existing MCP tests
2. Add search accuracy benchmarks
3. Test filter combinations
4. Verify result ranking quality

## Performance Optimizations

### Use Existing Infrastructure
- Redis cache for frequent queries (already configured)
- Qdrant's built-in filtering (no external processing)
- Neo4j indexes on symbol properties (already present)

### Query Optimization
```python
# In qdrant_service.py:
- Use Qdrant's native filtering
- Batch similar queries
- Cache embedding generations
```

## Rollback Plan

- Feature flag: `SYMBOL_SEARCH_ENABLED`
- Falls back to file content search
- Existing search unaffected

## Metrics

- Query latency (p50, p95, p99)
- Result relevance (click-through rate)
- Cache hit rate
- Symbol coverage percentage

## Example Queries

```python
# Via MCP:
await mcp.search_symbols({
    "query": "database connection handling",
    "symbol_types": ["function", "method"],
    "languages": ["python", "go"],
    "limit": 20
})

# Returns:
[
    {
        "symbol": "connect_to_database",
        "type": "function",
        "file": "src/db/connection.py",
        "line": 45,
        "score": 0.92,
        "snippet": "def connect_to_database(config: DBConfig)...",
        "related_symbols": ["DBConfig", "ConnectionPool"]
    }
]
```

## References

- Current MCP implementation: `neural-tools/src/servers/mcp_server.py`
- Qdrant filtering: https://qdrant.tech/documentation/concepts/filtering/
- Existing indexer: `neural-tools/src/servers/services/indexer_service.py`