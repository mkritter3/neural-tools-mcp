# ADR-0046: Neo4j Service Method Alignment for 2025 Standards

**Status:** Accepted
**Date:** September 13, 2025
**Author:** Claude (Opus 4.1)

## Context

During L9 validation testing on September 13, 2025, we discovered a critical method naming mismatch between the Neo4j Python Driver 5.x/6.x standards and our Neo4jService implementation:

1. **Neo4j Python Driver 5.x+ Standard (2025)**: Uses `execute_query()` as the primary high-level query execution method
2. **Our Implementation**: Neo4jService has `execute_cypher()` but not `execute_query()`
3. **Migration Manager Expectation**: Calls `neo4j.execute_query()` following the official driver pattern
4. **Test Failure**: `'Neo4jService' object has no attribute 'execute_query'`

### Neo4j Driver Evolution (as of September 2025)

According to the latest Neo4j Python Driver documentation (version 5.x/6.x):
- `execute_query()` is the recommended high-level API for query execution
- Provides built-in retry mechanisms and transaction management
- Supports routing control, database selection, and result transformation
- Available on both `Driver` and `AsyncDriver` classes

## Decision

We will align our Neo4jService with the Neo4j Python Driver 5.x+ standards by:

1. **Adding `execute_query()` method** to Neo4jService that wraps the existing functionality
2. **Maintaining backward compatibility** by keeping `execute_cypher()`
3. **Following the official API signature** for consistency with Neo4j documentation

## Implementation

### Solution 1: Alias Method (Quick Fix)
```python
class Neo4jService:
    async def execute_query(self, cypher_query: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute query using Neo4j 5.x+ standard naming. Delegates to execute_cypher."""
        return await self.execute_cypher(cypher_query, parameters)
```

### Solution 2: Full Alignment (Recommended)
```python
class Neo4jService:
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict] = None,
        database: Optional[str] = None,
        routing: str = "WRITE"
    ) -> Dict[str, Any]:
        """
        Execute query following Neo4j Python Driver 5.x+ standards.

        This method aligns with the official driver.execute_query() API
        while maintaining our service abstractions for caching and project isolation.
        """
        # Implementation with full feature support
        ...

    async def execute_cypher(self, cypher_query: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility.
        @deprecated Use execute_query() instead (Neo4j 5.x+ standard)
        """
        return await self.execute_query(cypher_query, parameters)
```

## Consequences

### Positive
- **Standards Compliance**: Aligns with Neo4j Python Driver 5.x+ (2025) standards
- **Future Proof**: Matches official Neo4j documentation and examples
- **Reduced Confusion**: Developers familiar with Neo4j will expect `execute_query()`
- **Better Integration**: MigrationManager and other tools can use standard methods
- **Test Compliance**: Fixes the failing migration tests

### Negative
- **API Surface Growth**: Adds another method to maintain
- **Potential Confusion**: Two methods doing similar things (mitigated by deprecation)
- **Migration Effort**: Existing code using `execute_cypher()` should eventually migrate

## Migration Path

1. **Phase 1** (Immediate): Add `execute_query()` as alias to fix tests
2. **Phase 2** (Next Sprint): Implement full Neo4j 5.x+ API alignment
3. **Phase 3** (Future): Deprecate `execute_cypher()` with warnings
4. **Phase 4** (Major Version): Remove `execute_cypher()` entirely

## Testing Requirements

- Verify MigrationManager works with `execute_query()`
- Ensure backward compatibility for `execute_cypher()`
- Test with Neo4j Python Driver 5.x+ features
- Validate caching and project isolation still work

## References

- Neo4j Python Driver 5.x/6.x Documentation (2025)
- [Context7 Library: /neo4j/neo4j-python-driver]
- L9 Validation Test Suite Results
- ADR-0029: Project Isolation via Neo4j Properties

## Decision Outcome

Implement **Solution 1** immediately to unblock tests, then proceed with **Solution 2** for full standards compliance.