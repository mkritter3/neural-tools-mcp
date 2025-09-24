# ADR-0093: Store USES and INSTANTIATES Relationships in Neo4j

## Status: Accepted

## Context

During testing of ADR-0092's elite_search MCP tool, we discovered that while the tree-sitter extractor successfully extracts USES and INSTANTIATES relationships (implemented in ADR-0090 Phase 3), these relationships are **never stored in Neo4j**.

Investigation revealed:
1. `tree_sitter_extractor.py` correctly extracts USES and INSTANTIATES relationships
2. The indexer collects these in `relationships_data` variable
3. BUT `_store_unified_neo4j()` function doesn't accept or process relationships
4. No Cypher queries exist to create these relationships in the graph

This explains why our GraphRAG queries return 0 USES/INSTANTIATES relationships despite having 6,294 chunks and 549 functions indexed.

## Decision

Implement relationship storage in the indexer by:

1. **Modify function signature** to accept relationships:
```python
async def _store_unified_neo4j(self, file_path: str, relative_path: Path,
                               chunks: List[Dict], embeddings: List[List[float]],
                               content: str, symbols_data: List[Dict] = None,
                               relationships_data: List[Dict] = None) -> bool:
```

2. **Add Cypher queries** to create relationships:
```cypher
// Create USES relationships (variable usage)
UNWIND $uses_relationships AS rel
MATCH (f:Function {name: rel.from_function, project: $project})
MATCH (v:Variable {name: rel.variable_name, project: $project})
MERGE (f)-[:USES {line: rel.line}]->(v)

// Create INSTANTIATES relationships (class instantiation)
UNWIND $instantiates_relationships AS rel
MATCH (f:Function {name: rel.from_function, project: $project})
MATCH (c:Class {name: rel.class_name, project: $project})
MERGE (f)-[:INSTANTIATES {line: rel.line}]->(c)
```

3. **Create Variable nodes** for USES relationships to point to
4. **Handle relationships alongside symbols** in the same transaction

## Implementation Details

### Phase 1: Update Storage Function
- Add `relationships_data` parameter
- Pass it through from the extraction phase
- Validate relationship data structure

### Phase 2: Create Supporting Nodes
- Variable nodes for USES relationships
- Ensure Function and Class nodes exist for relationships

### Phase 3: Create Relationships
- Process USES relationships
- Process INSTANTIATES relationships
- Add proper error handling for missing nodes

### Phase 4: Testing
- Verify relationships are created
- Test graph traversal with new relationships
- Validate elite_search improvements

## Consequences

### Positive
- Elite GraphRAG search will have full context
- USES relationships enable variable tracking
- INSTANTIATES relationships show class usage patterns
- Dependency analysis tool gets richer data
- Better code understanding and navigation

### Negative
- Reindexing required for existing code
- Slightly longer indexing time
- More complex graph structure
- Additional storage requirements

### Neutral
- Aligns with ADR-0090 Phase 3 design
- Completes the DRIFT-inspired GraphRAG implementation

## Migration Strategy

1. Deploy updated indexer with relationship storage
2. Trigger full reindex of existing projects
3. Verify relationships in Neo4j
4. Test elite_search with enriched graph

## Validation

After implementation, we should see:
- USES relationships connecting Functions to Variables
- INSTANTIATES relationships connecting Functions to Classes
- Elite search returning richer graph context
- Improved dependency analysis results

## References

- ADR-0090: Phase 3 specified USES/INSTANTIATES extraction
- ADR-0092: Discovered missing relationships during testing
- Tree-sitter extractor already implements extraction
- Microsoft DRIFT GraphRAG pattern requires these relationships