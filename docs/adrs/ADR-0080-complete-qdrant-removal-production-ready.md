# ADR-0080: Complete Qdrant Removal - Production Ready Neo4j-Only Architecture

**Date: September 23, 2025**
**Status: PROPOSED**
**Supersedes: ADR-0077 (Neo4j-Only Indexer Service)**
**Related: ADR-0066 (Neo4j Vector Consolidation), ADR-0078 (Neo4j UNWIND fixes)**

## Context

During debugging of vector search issues in the production system, we discovered that despite ADR-0077's Neo4j-only approach, the indexer service still contains **hard-coded Qdrant dependencies** that prevent proper initialization and chunk creation.

### Problem Identified

**Root Cause**: The production indexer fails to create chunks because it cannot initialize due to Qdrant connection failures, even with `ADR_075_NEO4J_ONLY=true` set.

**Evidence**:
```
ERROR: Failed to connect to REAL Qdrant: [Errno 111] Connection refused
WARNING: Services not ready (Neo4j: True, Qdrant: False), retrying...
```

**Impact**:
- Vector search returns `no_results` because zero chunks exist with embeddings
- Indexer reports "success" but `chunks_created: 0` for all files
- Only 4 sample chunks exist, no real codebase data

### Current State Analysis

**Working Components**:
- ✅ Neo4j HNSW vector indexes (`chunk_embeddings_index`, `file_embeddings_index`) exist and are ONLINE
- ✅ Nomic embedding service running and accessible (port 48000)
- ✅ Chunking logic (`_chunk_content`, `_semantic_code_chunking`) implemented correctly
- ✅ Storage logic (`_store_unified_neo4j`) with ADR-0078 defensive UNWIND working

**Broken Components**:
- ❌ Service initialization fails due to hard-coded Qdrant checks
- ❌ No real chunks being created despite "successful" indexing
- ❌ Vector search failing due to empty chunk database

## Decision

**Complete Qdrant removal** from all production code paths, not conditional skipping. This ensures:

1. **No Qdrant dependencies** in initialization logic
2. **Production images** with Qdrant code completely removed
3. **Simplified architecture** with single Neo4j storage path
4. **Robust initialization** that cannot fail due to missing Qdrant

### Technical Implementation

#### 1. Remove Qdrant from Service Container

**Current Code** (service_container.py):
```python
# Problems: Still tries to connect to Qdrant even with skip flags
qdrant_ok = self.ensure_qdrant_client()  # Always called
if skip_qdrant:
    qdrant_ok = True  # Bandaid fix, but connection already attempted
```

**Target Code**:
```python
# Complete removal - no Qdrant initialization at all
neo4j_ok = self.ensure_neo4j_client()
# No Qdrant references whatsoever
if neo4j_ok:
    logger.info("✅ Neo4j-only architecture initialized successfully")
```

#### 2. Remove Qdrant from Indexer Service

**Remove All**:
- Qdrant import statements
- Qdrant client initialization
- Qdrant storage methods
- Qdrant health checks
- Qdrant environment variables

**Keep Only**:
- Neo4j storage via `_store_unified_neo4j()`
- Nomic embedding generation
- Direct Neo4j vector storage with HNSW indexes

#### 3. Production Image Strategy

**Use Production Tags**:
- `neural-flow:production` for core services
- Pin specific production versions, not development tags
- Ensure production images have Qdrant completely removed

#### 4. Data Flow Simplification

**Current Complex Flow**:
```
File → Chunking → Nomic Embeddings → [Qdrant Check Fails] → No Storage
```

**Target Simple Flow**:
```
File → Chunking → Nomic Embeddings → Neo4j Storage → Success
```

## Implementation Steps

### Phase 1: Code Cleanup (30 minutes)
1. **Remove Qdrant imports** from service_container.py and indexer_service.py
2. **Remove ensure_qdrant_client()** method entirely
3. **Remove Qdrant health checks** from initialization
4. **Simplify service initialization** to Neo4j-only path

### Phase 2: Production Image (15 minutes)
1. **Use production image** with Qdrant dependencies removed
2. **Start indexer** with simplified initialization
3. **Verify service startup** without Qdrant failures

### Phase 3: Test Data Flow (15 minutes)
1. **Trigger reindexing** of real codebase files
2. **Verify chunk creation** with embeddings in Neo4j
3. **Test vector search** with real embedded chunks
4. **Confirm end-to-end** Nomic → Neo4j → Vector Search pipeline

## Expected Results

### Immediate Fixes
- ✅ **Indexer initializes successfully** without Qdrant failures
- ✅ **Real chunks created** with embeddings for actual files
- ✅ **Vector search works** with proper similarity results
- ✅ **No more `chunks_created: 0`** false success reports

### Architecture Benefits
- **Simplified deployment** - no Qdrant service required
- **Faster initialization** - no unnecessary connection attempts
- **Reduced complexity** - single storage system to maintain
- **Production ready** - no development/test dependencies

### Performance Expectations
- **Chunk creation**: Expect 100+ chunks from neural-tools codebase
- **Vector search**: <100ms P95 with HNSW indexes
- **Initialization**: <10s without Qdrant retries
- **End-to-end**: File → Search in <2s

## Consequences

### ✅ Positive

1. **Production Reliability**
   - No more false dependency failures
   - Simplified error handling and debugging
   - Single point of truth for vector storage

2. **Development Velocity**
   - Faster container startup times
   - Simplified architecture reduces cognitive load
   - No dual-system synchronization issues

3. **Operational Excellence**
   - Reduced infrastructure dependencies
   - Simplified monitoring and maintenance
   - Clear failure modes and debugging paths

### ⚠️ Considerations

1. **Breaking Change**
   - Old Qdrant-dependent code will not work
   - Must ensure all environments use updated images
   - Need to clean up old Qdrant containers/volumes

2. **Migration Impact**
   - Existing Qdrant data (if any) will be lost
   - Need to reindex from scratch with Neo4j-only approach
   - Update documentation and deployment scripts

## Validation Commands

### Test Service Initialization
```bash
# Should succeed without Qdrant errors
docker logs indexer-* | grep -E "(ERROR|Qdrant)"
# Should show: No Qdrant references at all

# Should show successful Neo4j connection only
docker logs indexer-* | grep "Services connected"
```

### Test Chunk Creation
```bash
# Should show many chunks created
echo "MATCH (c:Chunk {project: 'claude-l9-template'}) RETURN count(c);" | \
  docker exec -i claude-l9-template-neo4j-1 cypher-shell -u neo4j -p "graphrag-password"
# Expected: 100+ chunks

# Should show embeddings stored
echo "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c);" | \
  docker exec -i claude-l9-template-neo4j-1 cypher-shell -u neo4j -p "graphrag-password"
# Expected: Same count as chunks
```

### Test Vector Search
```bash
# Should return actual results
mcp__neural-tools__semantic_search query="python class" mode="hybrid" limit="3"
# Expected: Real results with similarity scores, not "no_results"
```

## Migration Plan

### Immediate (This Session)
1. ✅ Create this ADR
2. 🔄 Remove Qdrant code from service_container.py
3. 🔄 Remove Qdrant code from indexer_service.py
4. 🔄 Start production indexer without Qdrant
5. 🔄 Test full pipeline and commit working solution

### Follow-up (Next Session)
- Update deployment documentation
- Clean up old Qdrant containers and volumes
- Update CI/CD to use production images only

## Success Criteria

**Must Have**:
- [ ] Indexer starts without Qdrant connection attempts
- [ ] Real chunks created (>50 chunks from codebase)
- [ ] Vector search returns actual results
- [ ] No false "success" reports with 0 chunks created

**Should Have**:
- [ ] Vector search <100ms P95 performance
- [ ] Complete pipeline: File → Chunk → Embed → Store → Search
- [ ] Clean logs without Qdrant error messages

## Conclusion

ADR-0080 addresses the root cause of vector search failures by **completely removing Qdrant dependencies** rather than conditionally skipping them. This creates a truly production-ready Neo4j-only architecture that eliminates initialization failures and enables proper chunk creation with embeddings.

**Implementation Priority**: CRITICAL - Vector search is currently non-functional due to this architectural issue.

---

*This ADR resolves the disconnect between intended Neo4j-only architecture and actual Qdrant-dependent implementation, enabling the vector search capabilities promised by ADR-0072 and ADR-0075.*