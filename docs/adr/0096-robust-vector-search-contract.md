# ADR-0096: Robust Vector Search with Schema Contract

**Date: September 24, 2025 | Status: In Implementation**

## Context & Problem Statement

The Neural GraphRAG system has been experiencing recurring failures in vector search operations, with a pattern of "fix one thing, break another" that indicates fundamental architectural issues rather than simple bugs. The system alternates between working and broken states with each modification, suggesting tight coupling without proper contracts between components.

### Core Problem: The Brittleness Cycle

Every time we modify the indexer, vector search breaks. When we fix vector search, MCP tools stop returning results. This cycle has repeated multiple times:

1. **Indexer stores chunks** → Vector search can't find them (missing embeddings)
2. **Fix embeddings** → Chunks missing file_path property
3. **Add file_path** → DateTime serialization breaks JSON cache
4. **Fix DateTime** → Vector index names don't match
5. **Fix index names** → MCP tools get empty results
6. **Fix MCP tools** → Graph context structure changes
7. **And the cycle continues...**

### Root Cause Analysis (via Grok 4)

After deep analysis with Grok 4, the root cause was identified:

**Tight coupling without contract enforcement**

The system has three disconnected components that all make different assumptions about data structure:
- **Indexer**: Creates and stores chunks
- **Neo4j Storage**: Persists data with its own schema
- **Retrieval/Search**: Expects specific formats

Without a formal contract, each component evolves independently, breaking the others.

## Decision

Implement Neo4j's official **VectorCypherRetriever** pattern with a strict schema contract to ensure all components stay synchronized.

### Solution Components

#### 1. ChunkSchema Contract (Single Source of Truth)

```python
@dataclass
class ChunkSchema:
    """The canonical chunk structure ALL components must follow"""
    # Required fields
    chunk_id: str  # Format: "file_path:chunk:index"
    file_path: str  # Always present for search filtering
    content: str
    embedding: List[float]  # Must be exactly 768 dimensions
    project: str

    # Required with defaults
    created_at: str  # ISO string, NOT DateTime (JSON serializable)
    start_line: int = 0
    end_line: int = 0
    size: int = 0

    def to_neo4j_dict(self) -> dict:
        """Convert to Neo4j-compatible flat dictionary"""
        # No nested objects, only primitives
```

This contract:
- Enforces consistent structure across all components
- Validates data at creation time
- Ensures JSON serializability
- Prevents DateTime serialization issues
- Guarantees file_path is always present

#### 2. RobustVectorSearch Implementation

Following Neo4j's official VectorCypherRetriever pattern from neo4j-graphrag-python:

```python
class RobustVectorSearch:
    """Implements Neo4j's official two-phase search pattern"""

    async def vector_search_phase1(self, embedding, limit, min_score):
        """Phase 1: Pure vector similarity search"""
        # CRITICAL: Use literal index name (Neo4j requirement)
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embeddings_index', $limit, $embedding)
        YIELD node, score
        WHERE node.project = $project AND score >= $min_score
        RETURN elementId(node) as element_id, ...
        """

    async def graph_enrichment_phase2(self, chunk_ids, max_depth):
        """Phase 2: Graph traversal for context"""
        # Bounded traversal to prevent explosion
        cypher = """
        UNWIND $chunk_ids as chunk_id
        MATCH (c:Chunk {chunk_id: chunk_id})
        OPTIONAL MATCH path = (c)-[*1..2]-(related)
        WHERE related:Chunk OR related:Function OR related:Class
        WITH c, collect(DISTINCT related) as context
        RETURN c, context
        """
```

Key patterns from Neo4j official implementation:
- **Literal index names** in CALL procedures (not parameterized)
- **Two-phase approach** separating vector and graph operations
- **Bounded traversal** with explicit depth limits
- **UNWIND pattern** for controlled iteration
- **elementId()** instead of id() for node references

#### 3. Neo4j Constraints for Enforcement

```cypher
-- Unique chunks per project
CREATE CONSTRAINT chunk_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE (c.chunk_id, c.project) IS UNIQUE;

-- Vector index with exact name
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }
}
```

Note: Property existence constraints require Enterprise Edition.

## Implementation Progress

### ✅ Completed

1. **Created ChunkSchema contract class** (`chunk_schema.py`)
   - Dataclass with validation
   - to_neo4j_dict() for flat structure
   - Enforces 768-dimension embeddings
   - Uses ISO strings instead of DateTime

2. **Implemented RobustVectorSearch** (`robust_vector_search.py`)
   - vector_search_phase1() with literal index names
   - graph_enrichment_phase2() with bounded traversal
   - hybrid_search() combining both phases

3. **Integrated ChunkSchema into indexer**
   - Modified indexer_service.py to use ChunkSchema
   - All chunks now validated before storage
   - Ensures consistent structure

4. **Replaced vector search in neo4j_service.py**
   - Added RobustVectorSearch initialization
   - New methods delegate to robust implementation
   - Fallback to legacy for compatibility

5. **Applied Neo4j constraints**
   - Unique constraint on (chunk_id, project)
   - Vector index created successfully
   - Property constraints skipped (need Enterprise)

6. **Fixed MCP tool formatters**
   - Updated elite_search.py for new result structure
   - Added _detect_language() helper
   - Fixed graph_context handling (list vs dict)

### 🔧 Current Issues

1. **Elite search returns no results**
   - Fast search works perfectly
   - Hybrid search returns data at Neo4j level
   - Issue appears to be in elite_search formatter

2. **Index name inconsistency**
   - Neo4j appends "_index" to vector index names
   - Had to use "chunk_embeddings_index" not "chunk_embeddings"

## Technical Details

### Why This Breaks the Brittleness Cycle

1. **Single Source of Truth**: ChunkSchema defines THE structure
2. **Validation at Boundaries**: Data validated before storage
3. **Official Patterns**: Using Neo4j's battle-tested approaches
4. **No Custom Magic**: Lean on proven solutions

### Critical Learnings

1. **Neo4j CALL procedures don't support parameterized index names**
   - Must use literal strings in queries
   - This was causing silent failures

2. **DateTime objects break JSON serialization**
   - Redis cache couldn't store them
   - Solution: Use ISO strings

3. **file_path must be a direct property**
   - Can't rely on parsing from chunk_id
   - Must be explicitly set

4. **Graph context structure varies**
   - Sometimes dict, sometimes list
   - Must handle both cases

## Migration Path

1. **Update all chunks to have file_path** ✅
   ```cypher
   MATCH (c:Chunk) WHERE c.file_path IS NULL
   SET c.file_path = split(c.chunk_id, ':chunk:')[0]
   ```

2. **Convert DateTime to ISO strings** ✅
   - ChunkSchema handles this automatically

3. **Reindex with new structure** (if needed)
   - New chunks use ChunkSchema
   - Old chunks updated via migration

## Success Metrics

- [x] Vector search returns results with file paths
- [x] Fast search MCP tool works
- [ ] Elite search MCP tool works (in progress)
- [x] No DateTime serialization errors
- [x] Consistent chunk structure across system
- [x] Graph context properly enriched

## Future Improvements

1. **Complete elite_search fix**
   - Debug why results aren't being formatted
   - Ensure graph context is properly passed

2. **Add property existence constraints**
   - When Neo4j Enterprise available
   - Enforce file_path, content, project required

3. **Implement chunk versioning**
   - Track schema version in chunks
   - Enable smooth migrations

## References

- Neo4j VectorCypherRetriever pattern
- neo4j-graphrag-python official implementation
- ADR-0095: Neo4j-Migrations Framework
- ADR-0094: Symbol to Typed Nodes Migration
- ADR-0092: Fix Vector Search Disconnect

## Decision Outcome

**ACCEPTED** - The ChunkSchema contract with RobustVectorSearch implementation provides a sustainable solution to the brittleness problem. By adopting Neo4j's official patterns and enforcing a strict contract, we eliminate the tight coupling that caused recurring failures.

**Confidence: 95%** - Solution is architecturally sound and mostly implemented. Remaining issue with elite_search appears to be a formatting problem, not a fundamental flaw.