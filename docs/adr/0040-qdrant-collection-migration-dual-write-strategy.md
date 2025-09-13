# ADR-0040: Qdrant Collection Migration with Dual-Write Strategy

**Status**: Accepted
**Date**: September 12, 2025
**Deciders**: L9 Engineering Team, Expert Analysis (Gemini 2.5 Pro)
**Context**: ADR-0039 Centralized Collection Naming, Open Source Qdrant Constraints

## Context

Following ADR-0039's centralized naming configuration, we need to migrate existing Qdrant collections from the legacy format (`project_claude-l9-template_code`) to the clean format (`project-claude-l9-template`) without the `_code` suffix.

**Critical Constraints**:
- **Live System**: Zero downtime required
- **Open Source Qdrant**: No cloud migration tools, using v1.12.4 locally
- **No Direct Rename**: Qdrant doesn't support collection renaming
- **Data Integrity**: Cannot lose writes during migration

**Initial Issue**: Simple copy-and-switch creates a race condition where writes during migration are lost.

## Decision

Implement a **four-phase dual-write migration strategy** to ensure zero data loss and zero downtime.

### Architecture Pattern

```python
# Phase 1: Dual-Write Implementation
async def write_to_collections(data):
    """Write to both old and new collections simultaneously"""
    old_name = collection_naming.get_legacy_name(project)  # project_name_code
    new_name = collection_naming.get_collection_name(project)  # project-name

    # Dual write with error handling
    results = []
    results.append(await write_to_collection(old_name, data))
    results.append(await write_to_collection(new_name, data))

    # Both must succeed for write to be considered successful
    return all(results)
```

## Implementation Plan

### Phase 1: Preparation & Dual Writing (Immediate)

**Goal**: Ensure all new data goes to both collections

1. **Update Write Paths**:
```python
# In indexer_service.py
async def index_chunk(self, chunk_data):
    # Get both collection names
    legacy_name = f"project_{self.project_name}_code"
    new_name = collection_naming.get_collection_name(self.project_name)

    # Create new collection if needed
    await self._ensure_collection(new_name)

    # Dual write with feature flag
    if ENABLE_DUAL_WRITE:
        await self.qdrant.upsert(legacy_name, points)
        await self.qdrant.upsert(new_name, points)
    else:
        await self.qdrant.upsert(legacy_name, points)
```

2. **Feature Flag Control**:
```python
# Environment variables
ENABLE_DUAL_WRITE = os.getenv('ENABLE_DUAL_WRITE', 'false') == 'true'
ENABLE_NEW_READS = os.getenv('ENABLE_NEW_READS', 'false') == 'true'
```

### Phase 2: Backfill Historical Data (Day 1-2)

**Goal**: Copy all existing data to new collections

```python
#!/usr/bin/env python3
"""
Idempotent backfill script for Qdrant collection migration
"""

async def backfill_collection(old_name: str, new_name: str):
    """
    Idempotent backfill using scroll API and batch upsert
    """
    # Create new collection with same config
    old_info = client.get_collection(old_name)
    client.create_collection(
        new_name,
        vectors_config=old_info.config.params.vectors
    )

    # Scroll through old collection
    offset = None
    total_migrated = 0

    while True:
        # Read batch from old collection
        points, next_offset = client.scroll(
            collection_name=old_name,
            limit=100,
            offset=offset
        )

        if not points:
            break

        # Convert and upsert to new collection
        point_structs = convert_to_point_structs(points)
        client.upsert(
            collection_name=new_name,
            points=point_structs  # Idempotent - overwrites if exists
        )

        total_migrated += len(points)
        offset = next_offset

        if offset is None:
            break

    return total_migrated
```

### Phase 3: Read Path Switchover (Day 3)

**Goal**: Start reading from new collections

1. **Update Read Paths**:
```python
# In semantic_code_search
async def search(query: str, limit: int):
    if ENABLE_NEW_READS:
        # Read from new collection
        collection_name = collection_naming.get_collection_name(project)
    else:
        # Backward compatibility during transition
        possible_names = collection_naming.get_possible_names_for_lookup(project)
        # Try each until one works

    return await qdrant.search(collection_name, query, limit)
```

2. **Validation Before Switch**:
```python
async def validate_migration(old_name: str, new_name: str):
    """Ensure data integrity before switching reads"""

    # Count verification
    old_count = client.count(old_name)
    new_count = client.count(new_name)
    assert old_count == new_count, f"Count mismatch: {old_count} vs {new_count}"

    # Sample search verification
    test_vector = [0.1] * 768  # Neutral vector
    old_results = client.search(old_name, test_vector, limit=10)
    new_results = client.search(new_name, test_vector, limit=10)

    # Verify same IDs returned
    old_ids = {r.id for r in old_results}
    new_ids = {r.id for r in new_results}
    assert old_ids == new_ids, "Search results mismatch"

    return True
```

### Phase 4: Cleanup (Day 30)

**Goal**: Remove legacy collections and code

1. **Disable Dual Writes**:
   - Set `ENABLE_DUAL_WRITE=false`
   - Deploy to stop writing to old collections

2. **Remove Legacy Support**:
   - Simplify `get_possible_names_for_lookup()` to return only new format
   - Remove legacy parsing logic

3. **Delete Old Collections**:
```python
# After 30-day safety period
for collection in legacy_collections:
    if confirm_safe_to_delete(collection):
        client.delete_collection(collection)
        logger.info(f"Deleted legacy collection: {collection}")
```

## Success Metrics

### Phase 1 (Immediate)
- ✅ Dual writes active for all new data
- ✅ No errors in dual-write operations
- ✅ New collections created alongside old

### Phase 2 (Day 2)
- ✅ 100% of historical data backfilled
- ✅ Point counts match between old and new
- ✅ Sample searches return identical results

### Phase 3 (Day 3)
- ✅ All reads switched to new collections
- ✅ Search functionality working correctly
- ✅ Zero user-visible impact

### Phase 4 (Day 30)
- ✅ Legacy collections deleted
- ✅ Code simplified (no dual-write logic)
- ✅ Single naming standard enforced

## Risk Mitigation

### Data Loss Prevention
- **Dual-write before backfill**: Ensures no writes lost during migration
- **Idempotent operations**: Script can be safely re-run
- **Validation checks**: Verify integrity before switching

### Rollback Strategy
- **Phase 1**: Disable dual-write flag
- **Phase 2**: Stop backfill script
- **Phase 3**: Switch reads back to old collections
- **Phase 4**: Keep old collections for 30 days

### Performance Impact
- **Dual writes**: ~2x write latency (acceptable short-term)
- **Backfill**: Run during low-traffic periods
- **Resource usage**: Monitor Qdrant memory/CPU

## Alternatives Considered

### Alternative 1: Simple Copy-and-Switch
- **Pros**: Simpler implementation
- **Cons**: Data loss risk during copy
- **Verdict**: Rejected - violates zero-data-loss requirement

### Alternative 2: Qdrant Aliases (If Supported)
- **Pros**: Atomic switchover
- **Cons**: Not available in our version (1.12.4)
- **Verdict**: Rejected - version constraint

### Alternative 3: Dual-Write Strategy (Chosen)
- **Pros**: Zero data loss, controlled rollout, observable
- **Cons**: More complex, temporary performance impact
- **Verdict**: Accepted - meets all requirements

## Compliance

### L9 2025 Standards ✅
- **Truth-First**: Acknowledges race condition risk
- **Evidence-Based**: Expert analysis validated approach
- **95% Gate**: Phased rollout with validation
- **Reversible**: Each phase can be rolled back

### Protocol Alignment ✅
- **MCP 2025-06-18**: Compatible with MCP architecture
- **ADR-0039**: Builds on centralized naming
- **ADR-0037**: Uses environment variables for control
- **ADR-0030**: Works with ephemeral containers

### Expert Validation ✅
- **Gemini 2.5 Pro**: Identified critical race condition
- **Key Innovation**: Dual-write prevents data loss
- **Best Practice**: Follows standard migration patterns

## Implementation Status

- [x] Phase 0: CollectionNamingManager implemented (ADR-0039)
- [ ] Phase 1: Implement dual-write capability
- [ ] Phase 2: Create backfill script
- [ ] Phase 3: Switch read paths
- [ ] Phase 4: Cleanup legacy collections

---

**Confidence**: 95% - Expert-validated dual-write strategy ensures zero data loss during live migration with comprehensive rollback capability.