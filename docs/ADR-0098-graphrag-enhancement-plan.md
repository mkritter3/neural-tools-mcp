# ADR-0098: GraphRAG Enhancement Plan - Full Document Retrieval & Graph Context

**Date: September 24, 2025 | Status: Proposed**

## Executive Summary

Enhance our GraphRAG implementation to provide **full document understanding** without truncation, using battle-tested open source solutions. No fundamental rewrites - only incremental improvements over 3 weeks.

## Current Limitations

1. **Truncation at ~200 chars** - Can't see complete code implementations
2. **Opaque graph context** - "6 related chunks" but can't access them
3. **Missing relationship types** - No visibility into USES/CALLS/IMPORTS
4. **Limited understanding** - Can find files but not trace execution flow

## Solution Architecture

### Core Pattern: Parent-Child Documents + Community Summaries

```
[Full Document] → [Parent Node in Neo4j]
                        ↓
              [Chunk 1] [Chunk 2] [Chunk 3]
                   ↓         ↓         ↓
              [Vectors] [Vectors] [Vectors]
                        ↓
                [Community Detection]
                        ↓
                [Community Summaries]
```

## Implementation Phases (3 Weeks)

### Phase 1: Fix Truncation (Week 1)

**Extend ChunkSchema** - Add parent document support:
```python
# neural-tools/src/shared/schemas/chunk.py
class ChunkSchema(BaseModel):
    # Keep existing fields for backward compatibility
    content: str  # Keep truncated for preview

    # Add new fields
    full_content: Optional[str] = None  # Full document content
    parent_doc_id: Optional[str] = None  # Link to parent
    chunk_position: int = 0
    total_chunks: int = 1
```

**Store Full Documents in Neo4j**:
```python
# indexing_pipeline.py modification
async def index_file_enhanced(file_path: str):
    full_content = await read_file(file_path)

    # Create parent document node
    doc_node = await create_node(
        labels=["Document"],
        properties={
            "doc_id": generate_doc_id(file_path),
            "full_content": full_content,  # No size limit in Neo4j!
            "file_path": file_path,
            "chunk_count": len(chunks)
        }
    )

    # Create chunks with parent reference
    chunks = split_into_chunks(full_content, chunk_size=1000)
    for i, chunk in enumerate(chunks):
        chunk_node = await create_node(
            labels=["Chunk"],
            properties={
                "content": chunk[:200],  # Keep for backward compat
                "parent_doc_id": doc_node["doc_id"],
                "chunk_position": i,
                "embedding": embed(chunk)
            }
        )

        # Create relationship
        await create_relationship(
            chunk_node, "FROM_DOCUMENT", doc_node
        )
```

### Phase 2: Expose Graph Relationships (Week 1-2)

**Enhanced Response Format**:
```python
{
    "content": "...",  # Keep existing truncated preview
    "full_content": "...",  # NEW: Complete document
    "graph_context": {
        "relationships": [
            {"type": "USES", "target": "BaseHook", "file": "hook_utils.py", "line": 42},
            {"type": "IMPORTS", "module": "validators", "symbols": ["validate_hook"]},
            {"type": "CALLS", "function": "validate_compliance", "params": ["hook_file"]}
        ],
        "code_flow": ["entry_point → processing → validation → output"],
        "dependencies": {
            "internal": ["hook_utils", "validators"],
            "external": ["pathlib", "asyncio"]
        },
        "community": "hook_validation_cluster",
        "community_summary": "Hook validation and compliance checking subsystem"
    }
}
```

**Cypher Query Enhancement**:
```cypher
// Get relationships with details
MATCH (chunk:Chunk {id: $chunk_id})-[r]->(related)
RETURN type(r) as relationship_type,
       related.name as target,
       related.file_path as file,
       r.line_number as line
```

### Phase 3: Add Community Detection (Week 2)

**Install Neo4j Graph Data Science**:
```bash
# Update docker-compose.yml
neo4j:
  image: neo4j:5.22.0
  environment:
    - NEO4J_PLUGINS=["graph-data-science"]
    - NEO4J_dbms_memory_heap_max__size=2G
```

**Generate Community Summaries**:
```python
async def generate_community_summaries():
    # Create graph projection
    await neo4j.run("""
        CALL gds.graph.project(
            'code-graph',
            ['Chunk', 'Document'],
            ['RELATES_TO', 'USES', 'CALLS', 'IMPORTS']
        )
    """)

    # Run Louvain community detection
    result = await neo4j.run("""
        CALL gds.louvain.write('code-graph', {
            writeProperty: 'community_id'
        })
        YIELD communityCount, modularity
    """)

    # Generate summaries per community
    communities = await neo4j.run("""
        MATCH (c:Chunk)
        WITH c.community_id as community, collect(c.content) as contents
        RETURN community, contents
    """)

    for community in communities:
        summary = await generate_llm_summary(community['contents'])
        await neo4j.run("""
            CREATE (comm:Community {
                id: $id,
                summary: $summary,
                node_count: $count
            })
        """, id=community['community'],
            summary=summary,
            count=len(community['contents']))
```

### Phase 4: LlamaIndex Integration (Week 2-3)

**Install Battle-Tested Libraries**:
```python
# requirements.txt additions
llama-index==0.10.58
llama-index-graph-stores-neo4j==0.2.11
llama-index-embeddings-nomic==0.2.0
neo4j-python-driver==5.22.0  # Already have
```

**LlamaIndex Wrapper**:
```python
# neural-tools/src/graphrag/llamaindex_wrapper.py
from llama_index.core import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.nomic import NomicEmbedding

class EnhancedGraphRAG:
    def __init__(self):
        # Wrap existing Neo4j
        self.graph_store = Neo4jPropertyGraphStore(
            url="bolt://localhost:47687",
            username="neo4j",
            password="graphrag-password",
            database="graphrag"
        )

        # Use existing Nomic
        self.embed_model = NomicEmbedding(
            api_key=os.getenv("NOMIC_API_KEY"),
            model="nomic-embed-v2",
            task_type="search_document"
        )

        # Create hybrid retriever
        self.index = PropertyGraphIndex.from_existing(
            graph_store=self.graph_store,
            embed_model=self.embed_model,
            include_embeddings=True
        )

    async def search(self, query: str) -> List[Document]:
        # 1. Vector similarity search on chunks
        vector_retriever = self.index.as_retriever(
            similarity_top_k=10,
            include_text=False  # We'll get full text from parent
        )

        # 2. Get initial chunks
        chunks = await vector_retriever.retrieve(query)

        # 3. Get parent documents
        parent_docs = []
        for chunk in chunks:
            parent_query = """
            MATCH (c:Chunk {id: $chunk_id})-[:FROM_DOCUMENT]->(d:Document)
            RETURN d.full_content as content, d.file_path as path
            """
            doc = await self.neo4j.run(parent_query, chunk_id=chunk.id)
            parent_docs.append(doc)

        # 4. Get community context
        community_query = """
        MATCH (c:Chunk {id: $chunk_id})-[:IN_COMMUNITY]->(comm:Community)
        RETURN comm.summary
        """

        # 5. Get relationships
        relationship_query = """
        MATCH (c:Chunk {id: $chunk_id})-[r]-(related)
        RETURN type(r), related.name, related.file_path
        """

        return self.merge_results(chunks, parent_docs, communities, relationships)
```

## Battle-Tested Open Source Stack

| Component | Library | Stars | Purpose |
|-----------|---------|-------|---------|
| GraphRAG Framework | LlamaIndex | 45k+ | Production GraphRAG patterns |
| Graph Database | Neo4j | 13k+ | Graph storage + vectors |
| Community Detection | Neo4j GDS | Built-in | Graph algorithms |
| Embeddings | Nomic Embed | Existing | Keep current setup |
| Graph Analysis | NetworkX | 18k+ | Fallback for custom algorithms |
| Alternative | LangChain | 94k+ | Optional alternative to LlamaIndex |

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory usage with full docs | High | Streaming responses, lazy loading |
| Slower indexing | Medium | Background jobs, incremental updates |
| Breaking changes | High | Feature flags, A/B testing |
| Community detection overhead | Medium | Cache summaries in Redis (30s TTL) |
| Schema incompatibility | Low | Additive changes only |

## Success Metrics

- ✅ **Zero truncation** - Full documents always available
- ✅ **Relationship visibility** - All USES/CALLS/IMPORTS exposed
- ✅ **Community understanding** - Summaries for code clusters
- ✅ **Performance maintained** - <500ms P95 latency
- ✅ **Backward compatibility** - Existing queries still work

## Immediate Actions

1. **Create feature branch**: `git checkout -b feat/adr-0098-graphrag-enhancement`
2. **Install LlamaIndex**: `pip install llama-index llama-index-graph-stores-neo4j`
3. **Extend ChunkSchema**: Add `full_content` and `parent_doc_id` fields
4. **Update indexer**: Store parent documents with full content
5. **Test with sample**: Index 10 files, verify full retrieval

## Configuration Changes

```python
# .env additions
ENABLE_FULL_DOCUMENT_RETRIEVAL=true
ENABLE_COMMUNITY_DETECTION=true
LLAMAINDEX_CACHE_DIR=/tmp/llamaindex
NEO4J_GDS_LICENSE=community  # or enterprise if available

# docker-compose.yml
neo4j:
  environment:
    - NEO4J_PLUGINS=["graph-data-science"]
    - NEO4J_dbms_memory_heap_initial__size=1G
    - NEO4J_dbms_memory_heap_max__size=2G
    - NEO4J_dbms_memory_pagecache_size=1G
```

## Validation Queries

```cypher
// Verify parent-child structure
MATCH (d:Document)<-[:FROM_DOCUMENT]-(c:Chunk)
RETURN d.file_path, count(c) as chunk_count

// Check community assignments
MATCH (c:Chunk)
WHERE c.community_id IS NOT NULL
RETURN c.community_id, count(*) as members

// Test relationship extraction
MATCH (c:Chunk)-[r:USES|CALLS|IMPORTS]->(target)
RETURN type(r), target.name, count(*) as usage_count
```

## Timeline

- **Day 1-3**: ChunkSchema extension, parent document storage
- **Day 4-7**: Relationship extraction and exposure
- **Day 8-10**: Community detection setup
- **Day 11-14**: LlamaIndex integration
- **Day 15-18**: Testing and optimization
- **Day 19-21**: Documentation and rollout

## Decision

We will proceed with this incremental enhancement plan, using LlamaIndex as our primary GraphRAG framework wrapping our existing Neo4j infrastructure. This avoids rewrites while delivering significant improvements in document understanding.

## Consequences

**Positive:**
- Full document context always available
- Rich graph relationships exposed
- Community-level understanding
- Battle-tested libraries reduce risk

**Negative:**
- Increased storage (full docs)
- Slightly higher query complexity
- Additional dependencies

**Mitigation:**
- Storage is cheap; Neo4j handles large strings well
- Query complexity hidden behind LlamaIndex abstractions
- Pin dependency versions for stability

## References

- [Microsoft GraphRAG Paper 2024](https://www.microsoft.com/en-us/research/project/graphrag/)
- [LlamaIndex GraphRAG v2 Docs](https://docs.llamaindex.ai/en/stable/examples/cookbooks/GraphRAG_v2/)
- [Neo4j GDS Documentation](https://neo4j.com/docs/graph-data-science/current/)
- [Parent Document Retriever Pattern](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)

## Status

**Proposed** - Ready for implementation upon approval

---

*Confidence: 95%* - Based on research, Grok-4 analysis, and proven patterns from production systems.