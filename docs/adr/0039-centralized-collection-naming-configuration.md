# ADR-0039: Centralized Collection Naming Configuration

**Status**: Superseded by ADR-0041
**Date**: September 12, 2025
**Deciders**: L9 Engineering Team, Expert Analysis (Gemini 2.5 Pro)
**Context**: ADR-0037 Container Configuration Priority, ADR-0038 Docker Lifecycle Management

> **Note**: This ADR has been superseded by ADR-0041 which includes expert consensus on dual-write migration strategy and corrects implementation references. See ADR-0041 for the authoritative implementation guide.

## Context

During ADR-0038 implementation, we discovered a critical configuration scatter anti-pattern affecting collection naming:

```python
# Current scattered approach (4+ different files):
semantic_code_search:    f"project-{project_name}"        # Dashes, no suffix
multi_project_indexer:   f"project_{project_name}_code"   # Underscores + suffix
core_tools:             f"project_{PROJECT_NAME}_code"    # Different variable
API parsing:            .replace('_code', '')             # Parsing both formats
```

**Impact**: Semantic search returning empty results due to collection name mismatch between indexer and search services.

## Decision

Implement a centralized `CollectionNamingManager` class following L9 2025 standards and MCP Protocol 2025-06-18.

### Architecture Pattern

```python
# /neural-tools/src/servers/config/collection_naming.py
import re
import os
from typing import List

class CollectionNamingManager:
    """
    Single source of truth for Qdrant collection naming conventions.
    L9 2025 Standard: September 12, 2025
    MCP Protocol: 2025-06-18
    """

    # L9 Standard: Clean naming without suffixes (ADR-0039)
    # Environment override for advanced users (ADR-0037 compliance)
    _template = os.environ.get("COLLECTION_NAME_TEMPLATE", "project-{name}")

    @classmethod
    def get_collection_name(cls, project_name: str, collection_type: str = "code") -> str:
        """
        Get standardized collection name following L9 2025 standards.

        Args:
            project_name: Name of the project
            collection_type: Type of collection (code, docs, assets) - future expansion

        Returns:
            Sanitized, standardized collection name
        """
        # ADR-0037: Environment variable priority
        override = os.getenv(f"COLLECTION_NAME_OVERRIDE_{project_name.upper()}")
        if override:
            return override

        # Sanitize and format
        sanitized = cls._sanitize(project_name)
        return cls._template.format(name=sanitized)

    @classmethod
    def get_possible_names_for_lookup(cls, project_name: str) -> List[str]:
        """
        Returns prioritized list of possible collection names for backward compatibility.
        Used during migration period to support both new and legacy formats.

        Returns:
            List of collection names to try, in priority order
        """
        canonical_name = cls.get_collection_name(project_name)

        # Legacy formats for migration compatibility
        sanitized = cls._sanitize(project_name)
        legacy_underscore = f"project_{sanitized}_code"
        legacy_raw = project_name

        # Return de-duplicated, prioritized list
        names = [canonical_name]
        if legacy_underscore not in names:
            names.append(legacy_underscore)
        if legacy_raw not in names:
            names.append(legacy_raw)

        return names

    @classmethod
    def parse_project_name(cls, collection_name: str) -> str:
        """
        Extract project name from collection name (migration-safe).

        Args:
            collection_name: Full collection name

        Returns:
            Extracted project name

        Raises:
            ValueError: If collection name format is invalid
        """
        # New standard: project-{name}
        if collection_name.startswith("project-"):
            return collection_name[8:]

        # Legacy: project_{name}_code
        elif collection_name.startswith("project_") and "_code" in collection_name:
            return collection_name[8:-5]

        # Legacy: raw project name
        elif "/" not in collection_name and "\\" not in collection_name:
            return collection_name

        raise ValueError(f"Invalid collection name format: {collection_name}")

    @classmethod
    def _sanitize(cls, name: str) -> str:
        """
        Sanitize project name to valid Qdrant collection name.
        Qdrant requires: ^[a-zA-Z0-9_-]{1,255}$

        - Lowercase for consistency
        - Replace spaces, dots with hyphens
        - Remove invalid characters
        - Trim hyphens
        """
        name = name.lower()
        name = re.sub(r'[\s_.]+', '-', name)
        name = re.sub(r'[^a-z0-9-]', '', name)
        name = name.strip('-')

        if not name:
            raise ValueError(f"Project name resulted in empty sanitized name")
        if len(name) > 255:
            name = name[:255].rstrip('-')

        return name

# Singleton instance for easy import
collection_naming = CollectionNamingManager()
```

## Implementation Plan

### Phase 1: Implement Manager (Immediate)

1. Create `collection_naming.py` with manager class
2. Add comprehensive unit tests for sanitization edge cases
3. Deploy to neural-tools service layer

### Phase 2: Update Write Paths (Day 1)

Update services that CREATE collections to use canonical names:

```python
# Example: collection_config.py delegating to centralized naming
from config.collection_naming import collection_naming

# Old hardcoded approach
# name = f"project_{self.project_name}_code"

# New centralized approach (ADR-0041 supersedes this)
base_name = collection_naming.get_collection_name(self.project_name)
```

Services to update:
- `indexer_service.py` (single-project indexer used by containers)
- `collection_config.py` (delegate to CollectionNamingManager)
- MCP server search functions (for backward compatibility)

### Phase 3: Update Read Paths (Day 1-2)

Update services that SEARCH collections to use backward-compatible lookup:

```python
# Example: semantic_code_search
from servers.config.collection_naming import collection_naming

async def semantic_code_search_impl(query: str, limit: int):
    # ... get embeddings ...

    # Try all possible collection names (new + legacy)
    possible_names = collection_naming.get_possible_names_for_lookup(project_name)

    for collection_name in possible_names:
        try:
            results = await container.qdrant.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return results
        except CollectionNotFoundError:
            continue  # Try next possible name

    # If all names fail, raise clear error
    raise CollectionNotFoundError(
        f"No collection found for project '{project_name}'. "
        f"Tried: {possible_names}"
    )
```

Services to update:
- `neural_server_stdio.py` (semantic_code_search)
- `core_tools.py`
- `api/main.py`

### Phase 4: Data Migration (Day 3-7)

Create migration script to rename existing collections:

```python
# scripts/migrate_collection_names.py
async def migrate_collections():
    """One-time migration to standardize collection names"""

    collections = await qdrant_client.get_collections()

    for collection in collections:
        old_name = collection.name

        # Parse project name from old format
        try:
            project_name = collection_naming.parse_project_name(old_name)
        except ValueError:
            logger.warning(f"Skipping unknown collection: {old_name}")
            continue

        # Get new canonical name
        new_name = collection_naming.get_collection_name(project_name)

        if old_name != new_name:
            logger.info(f"Migrating: {old_name} -> {new_name}")

            # Use Qdrant aliasing for zero-downtime rename
            await qdrant_client.update_collection_aliases(
                change_aliases_request=UpdateCollectionAliases(
                    actions=[
                        CreateAliasOperation(
                            collection_name=old_name,
                            alias_name=new_name
                        ),
                        DeleteAliasOperation(
                            alias_name=old_name
                        )
                    ]
                )
            )
```

### Phase 5: Deprecate Legacy Support (Day 30)

After migration verified in production:
1. Simplify `get_possible_names_for_lookup()` to return only canonical name
2. Remove legacy parsing logic
3. Update documentation

## Success Metrics

### Immediate (Day 1)
- ✅ Semantic search returns results
- ✅ New collections use standardized names
- ✅ Zero search failures during migration

### Migration Complete (Day 7)
- ✅ 100% collections using new naming standard
- ✅ Legacy lookup code removed
- ✅ Single source of truth established

### Long-term (Day 30)
- ✅ Zero naming-related bugs
- ✅ Easy to add new collection types
- ✅ Configuration changes require single file edit

## Alternatives Considered

### Alternative 1: Environment Variables Only
- **Pros**: Simple, no code changes
- **Cons**: Still scattered, hard to migrate data
- **Verdict**: Rejected - doesn't solve root problem

### Alternative 2: Database Config Table
- **Pros**: Runtime configurable
- **Cons**: Additional dependency, network calls
- **Verdict**: Rejected - overengineered for this use case

### Alternative 3: Centralized Manager (Chosen)
- **Pros**: Single source of truth, testable, migration-friendly
- **Cons**: Requires code changes across services
- **Verdict**: Accepted - best balance of simplicity and maintainability

## Compliance

### L9 2025 Standards ✅
- **Truth-First**: Single source eliminates conflicting truths
- **Evidence-Based**: Bug demonstrated need for centralization
- **95% Gate**: Backward compatibility ensures safe rollout
- **Reversible**: Can rollback via environment overrides

### Protocol Alignment ✅
- **MCP 2025-06-18**: Configuration centralization pattern
- **ADR-0037**: Environment variable override support
- **ADR-0030**: Compatible with ephemeral containers
- **ADR-0020**: Supports project-aware schemas

### Expert Validation ✅
- **Gemini 2.5 Pro**: Comprehensive design review completed
- **Key Recommendations**: Sanitization, aliasing, phased migration
- **Risk Mitigation**: Backward compatibility during transition

## Rollback Plan

If issues arise:
1. Set `COLLECTION_NAME_TEMPLATE` environment variable to old format
2. Deploy hotfix to affected services
3. Investigate and fix root cause
4. Resume migration with fixes

## Implementation Status

- [ ] **Phase 1**: Create CollectionNamingManager class
- [ ] **Phase 2**: Update write paths (indexers)
- [ ] **Phase 3**: Update read paths (search)
- [ ] **Phase 4**: Run migration script
- [ ] **Phase 5**: Remove legacy support

---

**Confidence**: 95% - Expert-validated design with comprehensive migration plan and L9 standards compliance.