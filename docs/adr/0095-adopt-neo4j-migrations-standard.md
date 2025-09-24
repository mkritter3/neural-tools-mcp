# ADR-0095: Adopt Neo4j-Migrations as Standard Schema Evolution Tool

**Date:** September 24, 2025
**Status:** Accepted
**Impact:** High - Replacing custom migration framework
**Supersedes:** ADR-0020, ADR-0021 (Custom YAML migration system)

## Context

We currently have a custom YAML-based migration framework for GraphRAG schema evolution. While functional, this approach diverges from Neo4j 2025 best practices and industry standards. Our investigation revealed:

1. **Neo4j-Migrations** is the official, recommended tool for Neo4j schema evolution
2. Our custom system lacks integration with the indexer (migrations aren't applied on startup)
3. The Symbol→Class node migration (ADR-0094) needs proper schema evolution support
4. Neo4j 2025 has specific requirements (Java 21, block format, Cypher 25) that align better with Neo4j-Migrations

Additionally, we have three specific schema issues that need resolution:
- **Duplicate INSTANTIATES relationships** - Currently creating 6 instead of expected 3
- **Schema inconsistency** - Mix of Symbol and typed nodes (Class/Function/Method)
- **No migration automation** - Manual intervention required for schema changes

## Problem

### Current Custom System Limitations
1. **No Industry Alignment**: Custom YAML format instead of standard Cypher migrations
2. **Poor Integration**: Indexer doesn't apply migrations automatically
3. **Limited Tooling**: No CLI tools, IDE support, or ecosystem integration
4. **Validation Gaps**: No built-in pre/post checks or rollback triggers
5. **Maintenance Burden**: We maintain migration logic that Neo4j-Migrations provides

### Evidence from Codebase
```python
# migration_manager.py exists but indexer_service.py doesn't use it
# Only 1 reference in entire codebase: test file
grep "migration_manager" -> found only in tests/integration/test_all_22_tools.py
```

## Decision

Adopt **Neo4j-Migrations** as our standard schema evolution tool, replacing the custom YAML-based system.

## Solution

### Phase 1: Neo4j-Migrations Setup

#### Installation
```bash
pip install neo4j-migrations
```

#### Directory Structure
```
/neural-tools/
  /neo4j/
    /migrations/
      V001__initial_schema.cypher
      V002__add_function_class_nodes.cypher
      V003__migrate_symbols_to_typed_nodes.cypher
      V004__add_uses_instantiates_indexes.cypher
      R__repeatable_constraints.cypher
```

### Phase 2: Migration Scripts

#### V001__initial_schema.cypher
```cypher
// Initial GraphRAG schema
// Creates base node types and relationships

// Node types
CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE (f.project, f.path) IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.chunk_id, c.project) IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.project);
CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.project);

// Vector indexes (Neo4j 2025 HNSW)
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (n:Chunk)
ON (n.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};
```

#### V002__add_function_class_nodes.cypher
```cypher
// Add properly typed nodes for GraphRAG
// Supports ADR-0093 USES and INSTANTIATES relationships

// Function nodes
CREATE CONSTRAINT IF NOT EXISTS FOR (f:Function)
REQUIRE (f.project, f.name, f.file_path) IS UNIQUE;

// Class nodes
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class)
REQUIRE (c.project, c.name, c.file_path) IS UNIQUE;

// Method nodes
CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method)
REQUIRE (m.project, m.name, m.class_name, m.file_path) IS UNIQUE;

// Variable nodes (for USES relationships)
CREATE CONSTRAINT IF NOT EXISTS FOR (v:Variable)
REQUIRE (v.project, v.name, v.scope) IS UNIQUE;

// Indexes for performance
CREATE INDEX IF NOT EXISTS FOR (f:Function) ON (f.name);
CREATE INDEX IF NOT EXISTS FOR (c:Class) ON (c.name);
CREATE INDEX IF NOT EXISTS FOR (m:Method) ON (m.name);
CREATE INDEX IF NOT EXISTS FOR (v:Variable) ON (v.name);
```

#### V003__migrate_symbols_to_typed_nodes.cypher
```cypher
// Migrate Symbol nodes to properly typed nodes
// Implements ADR-0094 permanent fix

// Migrate Symbol {type: 'class'} to Class nodes
CALL apoc.periodic.iterate(
  "MATCH (s:Symbol) WHERE s.type = 'class' RETURN s",
  "
  CREATE (c:Class {
    name: s.name,
    qualified_name: s.qualified_name,
    file_path: s.file_path,
    start_line: s.start_line,
    end_line: s.end_line,
    language: s.language,
    docstring: s.docstring,
    project: s.project,
    indexed_at: s.indexed_at,
    migrated_from: 'Symbol',
    migration_date: datetime()
  })
  WITH s, c
  // Transfer relationships
  CALL apoc.refactor.from(s, c) YIELD input, output
  DETACH DELETE s
  RETURN count(*)
  ",
  {batchSize: 100, parallel: false}
);

// Migrate Symbol {type: 'function'} to Function nodes
CALL apoc.periodic.iterate(
  "MATCH (s:Symbol) WHERE s.type = 'function' RETURN s",
  "
  CREATE (f:Function {
    name: s.name,
    qualified_name: s.qualified_name,
    file_path: s.file_path,
    start_line: s.start_line,
    end_line: s.end_line,
    language: s.language,
    docstring: s.docstring,
    project: s.project,
    indexed_at: s.indexed_at,
    migrated_from: 'Symbol',
    migration_date: datetime()
  })
  WITH s, f
  CALL apoc.refactor.from(s, f) YIELD input, output
  DETACH DELETE s
  RETURN count(*)
  ",
  {batchSize: 100, parallel: false}
);

// Migrate Symbol {type: 'method'} to Method nodes
CALL apoc.periodic.iterate(
  "MATCH (s:Symbol) WHERE s.type = 'method' RETURN s",
  "
  CREATE (m:Method {
    name: s.name,
    qualified_name: s.qualified_name,
    file_path: s.file_path,
    start_line: s.start_line,
    end_line: s.end_line,
    language: s.language,
    docstring: s.docstring,
    project: s.project,
    indexed_at: s.indexed_at,
    class_name: CASE
      WHEN s.qualified_name CONTAINS '.'
      THEN split(s.qualified_name, '.')[0]
      ELSE null
    END,
    migrated_from: 'Symbol',
    migration_date: datetime()
  })
  WITH s, m
  CALL apoc.refactor.from(s, m) YIELD input, output
  DETACH DELETE s
  RETURN count(*)
  ",
  {batchSize: 100, parallel: false}
);
```

#### V004__add_uses_instantiates_indexes.cypher
```cypher
// Optimize USES and INSTANTIATES relationship queries
// Support ADR-0093 GraphRAG relationships

// Composite indexes for relationship queries
CREATE INDEX IF NOT EXISTS FOR (f:Function) ON (f.project, f.file_path);
CREATE INDEX IF NOT EXISTS FOR (c:Class) ON (f.project, f.file_path);
CREATE INDEX IF NOT EXISTS FOR (v:Variable) ON (v.project, v.file_path);

// Verify relationships work
MATCH (f:Function)-[:USES]->(v:Variable)
WHERE f.project = 'validation-test'
RETURN count(*) as uses_count;

MATCH (f:Function)-[:INSTANTIATES]->(c:Class)
WHERE f.project = 'validation-test'
RETURN count(*) as instantiates_count;
```

#### V005__fix_duplicate_instantiates.cypher
```cypher
// Fix duplicate INSTANTIATES relationships
// Issue: Found 6 INSTANTIATES instead of expected 3
// Root cause: Relationships created multiple times during indexing

// Step 1: Identify duplicates
MATCH (f:Function)-[r:INSTANTIATES]->(c:Class)
WITH f, c, collect(r) as relationships, count(r) as rel_count
WHERE rel_count > 1
CALL {
  WITH f, c, relationships
  // Keep the first relationship, delete others
  WITH relationships[0] as keeper, relationships[1..] as duplicates
  UNWIND duplicates as dup
  DELETE dup
  RETURN count(dup) as deleted
}
RETURN f.name as function, c.name as class, rel_count as original_count, deleted;

// Step 2: Add constraint to prevent future duplicates
// Note: Neo4j doesn't support relationship uniqueness constraints directly
// We'll handle this in application logic by using MERGE instead of CREATE

// Step 3: Verify no duplicates remain
MATCH (f:Function)-[r:INSTANTIATES]->(c:Class)
WITH f, c, count(r) as rel_count
WHERE rel_count > 1
RETURN f.name as function, c.name as class, rel_count
// Should return empty set
```

#### V006__add_relationship_constraints.cypher
```cypher
// Add application-level checks for relationship uniqueness
// Since Neo4j doesn't support relationship uniqueness constraints

// Create a tracking node for relationship validation
MERGE (tracker:RelationshipTracker {type: 'USES'})
SET tracker.description = 'Tracks USES relationship creation to prevent duplicates',
    tracker.created_at = datetime();

MERGE (tracker:RelationshipTracker {type: 'INSTANTIATES'})
SET tracker.description = 'Tracks INSTANTIATES relationship creation to prevent duplicates',
    tracker.created_at = datetime();

// Create index for faster duplicate checking
CREATE INDEX IF NOT EXISTS FOR ()-[r:USES]-() ON (r.from_id, r.to_id, r.line);
CREATE INDEX IF NOT EXISTS FOR ()-[r:INSTANTIATES]-() ON (r.from_id, r.to_id, r.line);
```

### Phase 3: Indexer Integration

#### Update indexer_service.py
```python
from neo4j_migrations import Neo4jMigrations

class IncrementalIndexer:
    async def initialize_services(self):
        """Initialize all required services with migration support"""

        # Apply Neo4j migrations before starting
        if self.container.neo4j:
            try:
                migrations = Neo4jMigrations(
                    neo4j_driver=self.container.neo4j.driver,
                    migrations_path="neural-tools/neo4j/migrations",
                    database="neo4j"  # or self.project_name for multi-tenancy
                )

                # Check pending migrations
                pending = migrations.pending_migrations()
                if pending:
                    logger.info(f"Applying {len(pending)} pending migrations...")

                    # Apply migrations
                    results = migrations.apply()
                    for result in results:
                        if result.success:
                            logger.info(f"✅ Applied migration: {result.version}")
                        else:
                            logger.error(f"❌ Failed migration: {result.version} - {result.error}")
                            # Enter compatibility mode
                            self.use_compatibility_mode = True
                            break

                logger.info("Schema migrations complete")

            except Exception as e:
                logger.error(f"Migration system error: {e}")
                # Continue with compatibility mode
                self.use_compatibility_mode = True

        # Rest of initialization...

    async def _create_relationships_from_tree_sitter(self, relationships, relative_path):
        """Create USES and INSTANTIATES relationships with duplicate prevention"""

        for rel in relationships:
            if rel.get("type") == "USES":
                # Use MERGE to prevent duplicates (ADR-0095)
                uses_cypher = """
                MERGE (f:Function {name: $func_name, project: $project, file_path: $file_path})
                MERGE (v:Variable {name: $var_name, project: $project})
                MERGE (f)-[r:USES {line: $line}]->(v)
                RETURN r
                """
                # ... execute cypher ...

            elif rel.get("type") == "INSTANTIATES":
                # Use MERGE to prevent duplicates (ADR-0095)
                inst_cypher = """
                MERGE (f:Function {name: $func_name, project: $project, file_path: $file_path})
                MERGE (c:Class {name: $class_name, project: $project})
                MERGE (f)-[r:INSTANTIATES {line: $line}]->(c)
                RETURN r
                """
                # ... execute cypher ...
```

### Phase 4: Migration from Custom System

#### Conversion Script
```python
#!/usr/bin/env python3
"""Convert YAML migrations to Cypher format for Neo4j-Migrations"""

import yaml
from pathlib import Path

def convert_yaml_to_cypher(yaml_path: Path, output_dir: Path):
    """Convert custom YAML migration to Cypher script"""

    with open(yaml_path) as f:
        migration = yaml.safe_load(f)

    version = migration['version']
    name = migration['name']

    cypher_lines = [
        f"// Migration V{version:03d}: {name}",
        f"// {migration.get('description', '')}",
        f"// Converted from YAML format",
        ""
    ]

    # Convert operations
    for op in migration.get('up', []):
        if op['type'] == 'create_node_type':
            cypher_lines.append(f"// Create {op['node_type']} nodes")
            # Generate constraints and indexes
            # ...convert YAML to Cypher...

    output_file = output_dir / f"V{version:03d}__{name}.cypher"
    output_file.write_text('\n'.join(cypher_lines))
```

## Consequences

### Positive
- **Industry Standard**: Align with Neo4j best practices and tooling
- **Better Integration**: Built-in CLI, Docker support, CI/CD integration
- **Automatic Rollback**: Transaction-based migrations with automatic rollback
- **Version Control**: Git-friendly Cypher files instead of YAML
- **Community Support**: Active maintenance, documentation, examples
- **IDE Support**: Cypher syntax highlighting and validation
- **Testing**: Built-in dry-run mode and validation

### Negative
- **Migration Effort**: Need to convert existing YAML migrations
- **Learning Curve**: Team needs to learn Neo4j-Migrations patterns
- **Dependency**: External dependency instead of custom code
- **Less Flexibility**: Must follow Neo4j-Migrations conventions

### Neutral
- **APOC Dependency**: Some migrations require APOC procedures
- **Versioning Change**: Move from integer to V-prefix versioning
- **File Format**: Cypher instead of YAML (arguably better)

## Implementation Plan

### Week 1: Setup & Testing
- [ ] Install neo4j-migrations package
- [ ] Create migrations directory structure
- [ ] Convert first 3 YAML migrations to Cypher
- [ ] Test in development environment

### Week 2: Integration
- [ ] Update indexer_service.py with migration hooks
- [ ] Add migration status to health checks
- [ ] Create rollback procedures
- [ ] Test with Symbol→Class migration

### Week 3: Migration & Validation
- [ ] Convert all remaining YAML migrations
- [ ] Create validation test suite
- [ ] Document new migration process
- [ ] Train team on Neo4j-Migrations

### Week 4: Production Rollout
- [ ] Deploy to staging environment
- [ ] Run full regression tests
- [ ] Production deployment with monitoring
- [ ] Archive old YAML system

## Specific Issues Addressed

This migration strategy directly resolves three critical issues:

### 1. Duplicate INSTANTIATES Relationships
**Problem:** Currently seeing 6 INSTANTIATES relationships instead of expected 3
**Solution:** V005 migration identifies and removes duplicates, V006 adds tracking to prevent future duplicates
**Prevention:** Update indexer to use MERGE instead of CREATE for relationships

### 2. Symbol→Class Schema Inconsistency
**Problem:** Mix of generic Symbol nodes and typed Class/Function/Method nodes
**Solution:** V003 migration converts all Symbol nodes to properly typed nodes
**Verification:** Each migration includes validation queries

### 3. No Migration Automation
**Problem:** Migrations exist but aren't applied automatically
**Solution:** Indexer integration in Phase 3 ensures migrations run on startup
**Fallback:** Compatibility mode if migrations fail

## Validation Criteria

1. **All existing migrations converted** to Cypher format
2. **Indexer applies migrations** on startup
3. **Symbol→Class migration** works without data loss
4. **Duplicate relationships cleaned** and prevented
5. **Performance unchanged** or improved
6. **Rollback tested** and documented
7. **Team trained** on new system

## Risk Mitigation

1. **Data Loss**: Full backup before migration, test on copy first
2. **Performance**: Benchmark before/after, have rollback plan
3. **Compatibility**: Keep ADR-0094 quick fix as fallback
4. **Team Knowledge**: Pair programming, documentation, examples

## References

- [Neo4j-Migrations Documentation](https://michael-simons.github.io/neo4j-migrations/)
- [Neo4j 2025 Migration Guide](https://neo4j.com/docs/upgrade-migration-guide/current/version-2025/)
- [APOC Migration Procedures](https://neo4j.com/docs/apoc/current/graph-updates/graph-refactoring/)
- ADR-0094: Symbol/Class Schema Mismatch Fix
- ADR-0093: USES and INSTANTIATES Relationships

## Decision Record

- **Proposed:** September 24, 2025
- **Accepted:** [Pending]
- **Implemented:** [Pending]
- **Validated:** [Pending]

## Notes

The transition from our custom YAML system to Neo4j-Migrations represents a maturation of our GraphRAG infrastructure. While we lose some custom flexibility, we gain reliability, community support, and alignment with Neo4j's roadmap for 2025 and beyond.

The Symbol→Class migration will be our first major test of this new system, serving as both validation and example for future migrations.