# ADR-0015: MCP Server Enhanced Tool Suite

**Status:** Proposed  
**Date:** 2025-09-09  
**Deciders:** Engineering Team  

## Context

Our MCP server currently provides basic neural search capabilities. With tree-sitter extraction and semantic indexing operational, we can enhance the **existing** MCP server with powerful code intelligence tools that leverage our infrastructure.

## Decision

Enhance the **existing** MCP server (`neural-tools/src/servers/mcp_server.py`) with new tools that utilize our Neo4j graph, Qdrant vectors, and tree-sitter symbols - all without creating new services.

## Integration Approach

### 1. Enhance Existing MCP Server
```python
# EDIT neural-tools/src/servers/mcp_server.py
# Add to existing TOOLS array:

# Tool 1: Find Symbol Definition
Tool(
    name="find_definition",
    description="Jump to definition of a symbol",
    parameters={
        "symbol": str,
        "file_context": Optional[str],
        "language": Optional[str]
    }
)

# Tool 2: Find References
Tool(
    name="find_references", 
    description="Find all references to a symbol",
    parameters={
        "symbol": str,
        "scope": Optional[str],  # "project" | "file" | "directory"
        "include_tests": bool
    }
)

# Tool 3: Code Dependency Graph
Tool(
    name="get_dependencies",
    description="Get dependency graph for a module/class",
    parameters={
        "target": str,  # file, class, or function
        "depth": int,  # traversal depth
        "direction": str  # "imports" | "imported_by" | "both"
    }
)

# Tool 4: Intelligent Code Suggestions
Tool(
    name="suggest_code",
    description="Get similar code patterns from the codebase",
    parameters={
        "context": str,  # current code context
        "intent": str,  # what user wants to do
        "limit": int
    }
)

# Tool 5: Refactoring Assistant
Tool(
    name="analyze_refactoring",
    description="Analyze impact of proposed refactoring",
    parameters={
        "symbol": str,
        "change_type": str,  # "rename" | "move" | "extract" | "inline"
        "new_value": Optional[str]
    }
)
```

### 2. Implement Tool Handlers
```python
# EDIT existing handle_tool_call method:
async def handle_tool_call(self, tool_name: str, params: Dict):
    # Existing tools...
    
    if tool_name == "find_definition":
        return await self._find_definition(params)
    elif tool_name == "find_references":
        return await self._find_references(params)
    elif tool_name == "get_dependencies":
        return await self._get_dependencies(params)
    elif tool_name == "suggest_code":
        return await self._suggest_code(params)
    elif tool_name == "analyze_refactoring":
        return await self._analyze_refactoring(params)
```

### 3. Leverage Existing Services
```python
# ADD methods to existing MCP server class:

async def _find_definition(self, params: Dict):
    """Find symbol definition using Neo4j"""
    # Query Neo4j for symbol node
    query = """
    MATCH (s:Symbol {name: $name})
    WHERE s.type IN ['class', 'function', 'interface']
    RETURN s.file_path, s.start_line, s.end_line
    ORDER BY s.scope_level
    LIMIT 1
    """
    result = await self.neo4j.execute_query(query, name=params['symbol'])
    return self._format_location(result)

async def _find_references(self, params: Dict):
    """Find references using Qdrant semantic search"""
    # Generate embedding for symbol
    embedding = await self.embedding_service.embed(params['symbol'])
    
    # Search with context filter
    results = await self.qdrant.search_vectors(
        collection_name="neural_index",
        query_vector=embedding,
        filter_conditions={
            "must": [
                {"key": "content", "match": {"text": params['symbol']}}
            ]
        },
        limit=50
    )
    return self._format_references(results)

async def _get_dependencies(self, params: Dict):
    """Build dependency graph from Neo4j"""
    query = """
    MATCH path = (source:Symbol {name: $target})-[:IMPORTS*1..%d]-(dep:Symbol)
    RETURN path
    """ % params['depth']
    
    graph = await self.neo4j.execute_query(query, target=params['target'])
    return self._format_dependency_graph(graph)
```

### 4. Smart Code Suggestions
```python
async def _suggest_code(self, params: Dict):
    """Use embeddings to find similar patterns"""
    # Embed the context + intent
    query_text = f"{params['context']} {params['intent']}"
    embedding = await self.embedding_service.embed(query_text)
    
    # Search for similar code blocks
    results = await self.qdrant.search_vectors(
        collection_name="neural_index",
        query_vector=embedding,
        filter_conditions={
            "must": [
                {"key": "symbol_type", "match": {"any": ["function", "method"]}}
            ]
        },
        limit=params.get('limit', 5)
    )
    
    # Enrich with symbol info from Neo4j
    for result in results:
        symbol_info = await self._get_symbol_info(result['file_path'])
        result['symbols'] = symbol_info
    
    return results
```

### 5. Refactoring Impact Analysis
```python
async def _analyze_refactoring(self, params: Dict):
    """Analyze refactoring impact using graph traversal"""
    
    # Find all references
    references = await self._find_references({
        'symbol': params['symbol'],
        'scope': 'project'
    })
    
    # Analyze import chains
    impact_query = """
    MATCH (s:Symbol {name: $symbol})<-[:DEPENDS_ON]-(dependent)
    RETURN dependent.file_path, dependent.name, dependent.type
    """
    dependents = await self.neo4j.execute_query(
        impact_query, 
        symbol=params['symbol']
    )
    
    # Calculate risk score
    risk_score = self._calculate_refactor_risk(
        len(references),
        len(dependents),
        params['change_type']
    )
    
    return {
        'affected_files': len(set(r['file'] for r in references)),
        'references': len(references),
        'dependent_symbols': len(dependents),
        'risk_score': risk_score,
        'risk_level': self._risk_level(risk_score),
        'details': {
            'references': references[:10],
            'dependents': dependents[:10]
        }
    }
```

## Implementation Phases

### Phase 1: Navigation Tools (Week 1)
- find_definition
- find_references
- Basic Neo4j queries

### Phase 2: Analysis Tools (Week 2)
- get_dependencies
- Graph traversal algorithms
- Caching layer

### Phase 3: Intelligence Tools (Week 3)
- suggest_code
- analyze_refactoring
- ML-enhanced suggestions

## Consequences

### Positive
- Powerful IDE-like features via MCP
- No new infrastructure needed
- Leverages ALL existing services
- Natural extension of current capabilities

### Negative
- Increased query complexity
- Higher Neo4j/Qdrant load
- More complex caching requirements

### Neutral
- Uses existing Docker setup
- Compatible with current architecture
- Same deployment model

## Performance Considerations

### Caching Strategy
```python
# Use existing Redis for:
- Symbol definition cache (TTL: 1 hour)
- Dependency graph cache (TTL: 30 min)
- Reference cache (TTL: 15 min)
```

### Query Optimization
- Neo4j indexes on symbol names
- Qdrant filtering before vector search
- Batch similar requests

## Testing Strategy

1. Unit tests for each tool handler
2. Integration tests with real codebases
3. Performance benchmarks
4. Accuracy metrics for suggestions

## Metrics

- Tool usage frequency
- Response time per tool (p50, p95)
- Cache hit rates
- Suggestion acceptance rate
- Refactoring risk accuracy

## Example Usage

```python
# Find definition
result = await mcp.call_tool("find_definition", {
    "symbol": "IndexerService",
    "file_context": "neural-tools/src/servers/services/indexer_service.py"
})
# Returns: {"file": "...", "line": 45, "column": 6}

# Get dependencies
deps = await mcp.call_tool("get_dependencies", {
    "target": "TreeSitterExtractor",
    "depth": 2,
    "direction": "both"
})
# Returns: {"nodes": [...], "edges": [...], "visualization": "..."}

# Analyze refactoring
impact = await mcp.call_tool("analyze_refactoring", {
    "symbol": "process_file",
    "change_type": "rename",
    "new_value": "handle_file"
})
# Returns: {"risk_level": "medium", "affected_files": 12, ...}
```

## Rollback Plan

- Feature flags per tool
- Graceful degradation on service failure
- Existing tools unaffected

## References

- Current MCP server: `neural-tools/src/servers/mcp_server.py`
- MCP specification: https://modelcontextprotocol.io/
- Existing services: Neo4j, Qdrant, Redis (all operational)