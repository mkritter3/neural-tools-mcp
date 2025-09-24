# ADR-0094: Fix Symbol/Class Schema Mismatch for INSTANTIATES Relationships

**Date:** September 24, 2025
**Status:** Accepted
**Impact:** High - Blocking GraphRAG relationship creation

## Context

During implementation of ADR-0093 (USES and INSTANTIATES relationships), we discovered that INSTANTIATES relationships are not being created despite successful extraction by tree-sitter. Investigation revealed a fundamental schema mismatch in the codebase.

## Problem

### Current State
1. **Symbol Storage**: The indexer stores all code elements as generic `:Symbol` nodes with a `type` property (`class`, `function`, `method`)
2. **Relationship Expectations**: USES and INSTANTIATES relationships expect specific node types:
   - USES expects `:Function` and `:Variable` nodes
   - INSTANTIATES expects `:Function` and `:Class` nodes

### Evidence
```cypher
// What exists in database
(:Symbol {type: 'class', name: 'DatabaseConnection'})
(:Symbol {type: 'function', name: 'process_data'})

// What relationships expect
MATCH (f:Function) // Not :Symbol {type:'function'}
MATCH (c:Class)    // Not :Symbol {type:'class'}
```

### Impact
- INSTANTIATES relationships fail with "Cannot match Class nodes"
- USES relationships only work because Function nodes are created ad-hoc
- Schema inconsistency makes queries complex and error-prone

## Decision

Implement proper Neo4j schema migration using Neo4j-Migrations framework to:
1. Convert `:Symbol {type:'class'}` nodes to `:Class` nodes
2. Convert `:Symbol {type:'function'}` nodes to `:Function` nodes
3. Convert `:Symbol {type:'method'}` nodes to `:Method` nodes
4. Ensure backward compatibility with existing queries

## Solution

### Phase 1: Quick Fix (Immediate)
Update the INSTANTIATES relationship creation to handle both schemas:
```cypher
// Support both Symbol and Class nodes
MATCH (f:Function {name: $func_name, project: $project})
MATCH (c)
WHERE (c:Class OR (c:Symbol AND c.type = 'class'))
  AND c.name = $class_name
  AND c.project = $project
MERGE (f)-[:INSTANTIATES {line: $line}]->(c)
```

### Phase 2: Proper Migration (Recommended)
Use Neo4j-Migrations to evolve the schema properly:

1. **Install neo4j-migrations**
```bash
pip install neo4j-migrations
```

2. **Create migration scripts** in `neural-tools/migrations/`:
```cypher
-- V001__convert_symbols_to_typed_nodes.cypher
-- Convert Symbol nodes to properly typed nodes

// Migrate classes
MATCH (s:Symbol {type: 'class'})
SET s:Class
REMOVE s:Symbol;

// Migrate functions
MATCH (s:Symbol {type: 'function'})
SET s:Function
REMOVE s:Symbol;

// Migrate methods
MATCH (s:Symbol {type: 'method'})
SET s:Method
REMOVE s:Symbol;
```

3. **Update indexer** to create typed nodes directly:
```python
# Instead of creating Symbol nodes
if symbol["type"] == "class":
    cypher = "MERGE (c:Class {name: $name, project: $project, ...})"
elif symbol["type"] == "function":
    cypher = "MERGE (f:Function {name: $name, project: $project, ...})"
```

### Phase 3: Schema Validation
Add constraints to ensure data integrity:
```cypher
CREATE CONSTRAINT class_project_name IF NOT EXISTS
FOR (c:Class) REQUIRE (c.project, c.name, c.file_path) IS UNIQUE;

CREATE CONSTRAINT function_project_name IF NOT EXISTS
FOR (f:Function) REQUIRE (f.project, f.name, c.file_path) IS UNIQUE;
```

## Consequences

### Positive
- Clean, consistent schema aligned with domain model
- Simpler, more performant queries
- Better type safety in Cypher queries
- Alignment with Neo4j best practices
- Proper relationship creation for GraphRAG

### Negative
- Migration complexity for existing deployments
- Temporary dual-schema support needed
- Need to update all existing queries

### Neutral
- Following Neo4j-Migrations framework (industry standard)
- One-time migration cost

## Implementation Checklist

- [ ] Implement quick fix for immediate functionality
- [ ] Set up Neo4j-Migrations in the project
- [ ] Create migration scripts V001-V003
- [ ] Update indexer to create typed nodes
- [ ] Update all Cypher queries to use typed nodes
- [ ] Add schema constraints
- [ ] Test migration on staging environment
- [ ] Document migration process for production

## References

- Neo4j-Migrations: https://neo4j.com/labs/neo4j-migrations/
- Neo4j 2025 Migration Guide: https://neo4j.com/docs/upgrade-migration-guide/current/version-2025/
- ADR-0093: USES and INSTANTIATES Relationships
- Issue: INSTANTIATES relationships not created despite extraction

## Notes

This schema mismatch likely dates back to early GraphRAG implementation where a generic Symbol node seemed sufficient. As relationship complexity grows, typed nodes become essential for maintainability and performance.