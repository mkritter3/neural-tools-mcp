# ADR-0041: Collection Naming Single Source of Truth

**Status:** Accepted
**Date:** 2025-09-12
**Authors:** L9 Engineering Team
**Reviewers:** Gemini 2.5 Pro, Grok-4 (Expert Consensus Achieved)
**Context:** Collection Naming Inconsistency Bug Fix
**Version:** 2.0

---

## Context

### Problem Statement

The neural indexer system has a **critical collection naming inconsistency** that causes semantic search to fail. Different components use different naming conventions for Qdrant collections:

**Current State (BROKEN):**
- **Indexer creates**: `project_claude-l9-template_code` (underscore + hyphen mixed)
- **Search expects**: `project-claude-l9-template` (hyphen, no suffix)
- **User requirement**: "we wanted to get rid of the _code suffix everywhere"

This mismatch means indexed data is invisible to search operations.

### Evidence of the Problem

```
Actual Qdrant collections:
- project_claude-l9-template_code (created by indexer)
- project_eventfully-yours_code (created by indexer)

Search attempts:
- project-claude-l9-template (fails - collection not found)
- claude-l9-template (fails - collection not found)
```

### Root Cause Analysis

1. **Configuration Scatter**: Collection naming logic hardcoded in 4+ different files
2. **No Single Source of Truth**: Each component has its own naming convention
3. **ADR-0039 Incomplete**: Created CollectionNamingManager but indexer still uses legacy format
4. **Architecture Confusion**: ADR-0039 incorrectly referenced multi_project_indexer.py when we use single-project indexer per ADR-0030

---

## Decision

**We will complete the collection naming standardization by ensuring ALL components use the centralized CollectionNamingManager as the single source of truth.**

### Solution Architecture

```
┌─────────────────────────────────────────────────────┐
│         CollectionNamingManager (SSOT)              │
│   /neural-tools/src/servers/config/collection_naming.py │
│                                                      │
│  Template: "project-{name}" (configurable via env)  │
│  Sanitization: Consistent hyphen-based naming       │
│  Backward Compatibility: Legacy format support      │
└─────────────────────────────────────────────────────┘
                           ▲
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ Indexer │      │   MCP   │      │   API   │
    │ Service │      │ Server  │      │ Service │
    └─────────┘      └─────────┘      └─────────┘
```

### Implementation Strategy

1. **Fix Single-Project Indexer** (`indexer_service.py`):
   - Import CollectionNamingManager
   - Replace hardcoded `collection_prefix = f"project_{project_name}_"`
   - Use `collection_naming.get_collection_name(project_name)`

2. **Update Collection Manager** (`collection_config.py`):
   - Either delegate to CollectionNamingManager
   - Or merge functionality into CollectionNamingManager

3. **Ensure Consistency**:
   - ALL collection name generation goes through CollectionNamingManager
   - NO hardcoded collection naming anywhere else
   - Environment variable override capability preserved

---

## Detailed Design

### Phase 1: Fix Indexer Service

**Current Code** (indexer_service.py:178):
```python
# WRONG - Hardcoded legacy format
self.collection_prefix = f"project_{project_name}_"
```

**Fixed Code**:
```python
# CORRECT - Use centralized naming
from servers.config.collection_naming import collection_naming

class IncrementalIndexer:
    def __init__(self, project_path: str, project_name: str = "default", container: ServiceContainer = None):
        self.project_path = Path(project_path)
        self.project_name = project_name

        # ADR-0041: Use centralized collection naming
        self.collection_name = collection_naming.get_collection_name(project_name)
        # For backward compatibility during transition
        self.collection_prefix = f"{self.collection_name[:-5] if self.collection_name.endswith('_code') else self.collection_name}_"
```

### Phase 2: Update Collection References

**All methods using collection names must be updated**:

```python
async def _store_in_qdrant(self, file_path: str, chunks: List[Dict]):
    """Store embeddings in Qdrant"""
    # OLD: collection_name = f"{self.collection_prefix}code"
    # NEW: Use self.collection_name directly
    collection_name = self.collection_name

    await self._ensure_collection(collection_name)
    # ... rest of method
```

### Phase 3: Collection Manager Integration

**Option A - Delegation** (Recommended):
```python
# collection_config.py
from servers.config.collection_naming import collection_naming

class CollectionManager:
    def get_collection_name(self, collection_type: CollectionType) -> str:
        # Delegate to centralized naming
        base_name = collection_naming.get_collection_name(self.project_name)
        # Add type suffix if needed (but we're removing _code suffix)
        return base_name
```

**Option B - Consolidation**:
- Move all collection management logic into CollectionNamingManager
- Deprecate CollectionManager
- Single class for all collection operations

---

## Benefits

### Immediate Benefits

1. **Fixes Semantic Search**: Collections will be found correctly
2. **Eliminates Confusion**: One naming convention everywhere
3. **Simplifies Debugging**: Single place to check/change naming
4. **Satisfies User Requirement**: Removes `_code` suffix as requested

### Long-term Benefits

1. **Maintainability**: Changes to naming convention in one place
2. **Consistency**: Guaranteed same names across all components
3. **Flexibility**: Environment variable configuration
4. **Backward Compatibility**: Can still find legacy collections

---

## Trade-offs and Considerations

### Trade-offs

1. **Migration Complexity**: Existing collections need renaming or dual lookup
   - **Mitigation**: Backward compatibility in search, clean migration script

2. **Testing Required**: All components need retesting
   - **Mitigation**: Comprehensive test suite, staged rollout

3. **Documentation Updates**: Multiple ADRs reference old naming
   - **Mitigation**: Update documentation as part of implementation

### Risks

1. **Data Loss Risk**: Renaming collections could lose access to data
   - **Mitigation**: Backup before migration, dual lookup during transition

2. **Incomplete Migration**: Some component might be missed
   - **Mitigation**: Grep entire codebase for collection naming patterns

---

## Implementation Plan

### Day 1: Core Implementation
- [ ] Update indexer_service.py to use CollectionNamingManager
- [ ] Update collection_config.py to delegate or consolidate
- [ ] Test indexer creates correct collection names

### Day 2: Search Verification
- [ ] Verify MCP search finds new collections
- [ ] Test backward compatibility with old collections
- [ ] Run end-to-end indexing and search test

### Day 3: Migration & Cleanup
- [ ] Create migration script for existing collections
- [ ] Update all documentation
- [ ] Remove all hardcoded collection naming

### Validation Criteria

**Exit Conditions:**
- [ ] Indexer creates collections as `project-{name}` (no `_code` suffix)
- [ ] Search successfully finds indexed data
- [ ] No hardcoded collection names remain in codebase
- [ ] All tests pass with new naming convention
- [ ] Backward compatibility confirmed for existing collections

---

## Alternatives Considered

### 1. Keep `_code` Suffix
- **Pros**: No migration needed
- **Cons**: User explicitly requested removal
- **Rejected**: Doesn't meet requirements

### 2. Parallel Collections
- **Pros**: No data migration
- **Cons**: Double storage, complexity
- **Rejected**: Wasteful and confusing

### 3. Collection Aliases
- **Pros**: Transparent migration
- **Cons**: Qdrant doesn't support aliases
- **Rejected**: Not technically feasible

---

## Enhanced Zero-Downtime Migration Strategy (Expert Consensus)

Based on expert review from Gemini 2.5 Pro and Grok-4, we will implement a **dual-write strategy** for zero-downtime migration instead of simple copy-and-switch.

### Phase 1: Preparation (Code Deployment)

1. **Implement Enhanced CollectionNamingManager**:
```python
class CollectionNamingManager:
    """Single Source of Truth for Qdrant collection naming."""
    _PRIMARY_TEMPLATE = "project-{project_name}"
    _LEGACY_CODE_TEMPLATE = "project_{project_name}_code"

    @classmethod
    def get_collection_name(cls, project_name: str) -> str:
        """Returns the current, canonical collection name."""
        sanitized = cls._sanitize(project_name)
        return cls._PRIMARY_TEMPLATE.format(project_name=sanitized)

    @classmethod
    def get_legacy_collection_names(cls, project_name: str) -> list[str]:
        """Returns all known legacy collection names for migration."""
        return [
            cls._LEGACY_CODE_TEMPLATE.format(project_name=project_name),
            f"project_{project_name}_code",  # Mixed underscore/hyphen
        ]
```

2. **Add Dual-Write Capability to Indexer**:
```python
# In indexer_service.py
ENABLE_DUAL_WRITE = os.getenv("ENABLE_DUAL_WRITE", "false").lower() == "true"

async def _store_in_qdrant(self, file_path: str, chunks: List[Dict]):
    # Primary collection (new format)
    primary_collection = collection_naming.get_collection_name(self.project_name)
    await self._write_to_collection(primary_collection, chunks)

    # Dual-write to legacy during migration
    if ENABLE_DUAL_WRITE:
        legacy_names = collection_naming.get_legacy_collection_names(self.project_name)
        for legacy_name in legacy_names:
            try:
                await self._write_to_collection(legacy_name, chunks)
            except Exception as e:
                logger.warning(f"Dual-write to {legacy_name} failed: {e}")
```

3. **Deploy Code Changes** (no data migration yet)

### Phase 2: Backfill (Data Migration)

1. **Enable Dual-Writes**:
```bash
export ENABLE_DUAL_WRITE=true
# Restart indexer service
```

2. **Run Backfill Script**:
```python
async def backfill_collections():
    """Migrate data from legacy to new collection format."""
    for project in get_all_projects():
        old_name = f"project_{project}_code"
        new_name = f"project-{project}"

        # Create new collection if needed
        await ensure_collection(new_name)

        # Copy all points (idempotent)
        points = await get_all_points(old_name)
        await upsert_points(new_name, points)

        # Validate counts match
        assert get_count(old_name) == get_count(new_name)
```

### Phase 3: Switchover (Atomic Switch)

1. **Create Aliases for Backward Compatibility**:
```python
# Qdrant alias creation
await qdrant_client.update_collection_aliases(
    change_aliases_operations=[
        CreateAlias(
            create_alias=CreateAliasOperation(
                collection_name="project-claude-l9-template",  # New
                alias_name="project_claude-l9-template_code"    # Old name as alias
            )
        )
    ]
)
```

2. **Update Search to Use New Names**:
   - Search now uses primary collection name
   - Alias handles any legacy callers transparently

### Phase 4: Cleanup (After Confidence Period)

1. **Monitor for 1-2 weeks**
2. **Disable dual-writes**:
```bash
export ENABLE_DUAL_WRITE=false
```
3. **Delete old collections and aliases** (after 30-day deprecation)

### Rollback Strategy

If issues arise:
1. **Delete aliases** - immediate revert to old collections
2. **Re-enable dual-writes** if needed
3. Old collections remain intact until cleanup phase

---

## Migration Strategy (Original Simple Approach - Kept for Reference)

### Safe Migration Process

1. **Backup Current Collections**:
```python
# List all collections
collections = qdrant_client.get_collections()
for collection in collections.collections:
    if collection.name.endswith("_code"):
        # Back up the collection
        backup_collection(collection.name)
```

2. **Rename Collections**:
```python
# Rename to new format
old_name = "project_claude-l9-template_code"
new_name = "project-claude-l9-template"

# Qdrant doesn't have rename, so recreate
create_collection(new_name, vectors_config)
migrate_points(old_name, new_name)
delete_collection(old_name)
```

3. **Verify Migration**:
```python
# Test search works with new names
test_semantic_search(new_name)
test_hybrid_search(new_name)
```

---

## Expert Consensus Summary

### Gemini 2.5 Pro Assessment (Confidence: 9/10)

**Strong Endorsement**: "Your diagnosis of configuration scatter as the root cause is spot on. The centralized CollectionNamingManager is the correct architectural path forward."

**Key Recommendations**:
1. **Enhanced Interface**: Separate canonical names from legacy names in the API
2. **Dual-Write Strategy**: Essential for zero-downtime migration in production
3. **Phased Migration**: 4-phase approach (Prep → Backfill → Switchover → Cleanup)
4. **Alias Support**: Use Qdrant aliases for transparent backward compatibility

### Grok-4 Assessment (Confidence: 95%)

**Technical Validation**: "Solid proposal tackling a real pain point with naming inconsistencies. The focus on centralizing logic aligns well with our Python-centric stack."

**Key Points**:
1. **SSOT Pattern**: Strong fit for distributed components, reduces scatter and bugs
2. **Hybrid Dual-Write**: Best approach for production safety with phased cutover
3. **Temporary Compatibility**: 1-3 month deprecation timeline, not permanent
4. **Comprehensive Testing**: 90%+ coverage on affected modules with Qdrant fixtures

### Consensus Agreement

Both experts strongly agree on:
- ✅ **Centralized configuration** is the right pattern
- ✅ **Dual-write strategy** for zero-downtime migration
- ✅ **Temporary backward compatibility** (3-6 months max)
- ✅ **Comprehensive test coverage** including migration scripts
- ✅ **Full L9 2025 standards compliance**

---

## Expert Review Request (Original)

### Questions for Gemini 2.5 Pro & Grok 4

1. **Architecture**: Is centralized CollectionNamingManager the right pattern for configuration management in distributed systems?

2. **Migration Safety**: What's the safest approach to rename collections in production without data loss?

3. **Backward Compatibility**: Should we maintain permanent backward compatibility or plan deprecation?

4. **Testing Strategy**: What test coverage is needed for this critical naming change?

5. **L9 Standards**: Does this approach meet L9 2025 engineering standards for single source of truth?

---

## References

- **ADR-0030**: Multi-Container Indexer Orchestration (defines single-project indexer)
- **ADR-0039**: Centralized Collection Naming Configuration (created CollectionNamingManager)
- **Issue**: Semantic search returning empty results due to naming mismatch
- **User Requirement**: "we wanted to get rid of the _code suffix everywhere"

---

**Confidence: 95%**
**Assumptions**:
- Single-project indexer is the correct architecture per ADR-0030
- Qdrant doesn't have built-in collection renaming (requires recreate)
- User accepts brief downtime for migration