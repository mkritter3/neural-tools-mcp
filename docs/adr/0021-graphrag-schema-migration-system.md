# ADR-0021: GraphRAG Schema Migration System

## Status
Proposed

## Context

ADR-0020 introduced per-project custom GraphRAG schemas with auto-detection and templates. However, it lacks a full migration system for:

1. **Schema Evolution**: Projects evolve, requiring schema changes over time
2. **Version Control**: No tracking of schema change history
3. **Rollback Capability**: No way to revert problematic schema changes
4. **Data Migration**: Schema changes may require transforming existing data
5. **Team Collaboration**: Multiple developers need to sync schema changes
6. **Production Safety**: Changes need testing before production deployment

Current limitations:
- Schema changes overwrite the previous version
- No migration history or audit trail
- No rollback mechanism
- No data transformation during schema changes
- No diff generation between schema versions
- No dry-run capability for testing migrations

## Decision

Implement a **Full Schema Migration System** similar to database migration tools (Supabase, Rails, Django) but adapted for dual-database GraphRAG architecture (Neo4j + Qdrant).

### Architecture Overview

```
Project Root/
├── .graphrag/
│   ├── schema.yaml                  # Current schema (generated)
│   ├── migrations/                  # Migration history
│   │   ├── 001_initial_schema.yaml
│   │   ├── 002_add_component_state.yaml
│   │   ├── 003_add_hook_dependencies.yaml
│   │   └── 004_add_context_providers.yaml
│   ├── migration_state.json         # Applied migrations tracking
│   └── rollback/                    # Rollback snapshots
│       ├── pre_002_snapshot.json
│       └── pre_003_snapshot.json
```

## Implementation Design

### Phase 1: Migration File Structure

```yaml
# .graphrag/migrations/001_initial_schema.yaml
version: 1
timestamp: 2025-09-11T10:00:00Z
author: system
description: Initial schema for React project
up:
  # Operations to apply this migration
  - operation: create_node_type
    params:
      name: Component
      properties:
        name: string
        type: enum[functional, class]
        file_path: string
      indexes: [name, file_path]
  
  - operation: create_relationship_type
    params:
      name: USES_HOOK
      from_types: [Component]
      to_types: [Hook]
      properties:
        line_number: int
  
  - operation: create_collection
    params:
      name: components
      vector_size: 768
      fields: [component_name, props, hooks_used]

down:
  # Operations to rollback this migration
  - operation: drop_collection
    params:
      name: components
  
  - operation: drop_relationship_type
    params:
      name: USES_HOOK
  
  - operation: drop_node_type
    params:
      name: Component

dependencies: []  # Migration dependencies
checksum: sha256:abc123...  # Integrity check
```

### Phase 2: Migration Operations

```python
class MigrationOperation(Enum):
    """Supported migration operations"""
    # Node operations
    CREATE_NODE_TYPE = "create_node_type"
    ALTER_NODE_TYPE = "alter_node_type"
    DROP_NODE_TYPE = "drop_node_type"
    ADD_NODE_PROPERTY = "add_node_property"
    REMOVE_NODE_PROPERTY = "remove_node_property"
    RENAME_NODE_TYPE = "rename_node_type"
    
    # Relationship operations
    CREATE_RELATIONSHIP_TYPE = "create_relationship_type"
    ALTER_RELATIONSHIP_TYPE = "alter_relationship_type"
    DROP_RELATIONSHIP_TYPE = "drop_relationship_type"
    ADD_RELATIONSHIP_PROPERTY = "add_relationship_property"
    REMOVE_RELATIONSHIP_PROPERTY = "remove_relationship_property"
    
    # Collection operations (Qdrant)
    CREATE_COLLECTION = "create_collection"
    ALTER_COLLECTION = "alter_collection"
    DROP_COLLECTION = "drop_collection"
    ADD_COLLECTION_FIELD = "add_collection_field"
    REMOVE_COLLECTION_FIELD = "remove_collection_field"
    REINDEX_COLLECTION = "reindex_collection"
    
    # Index operations (Neo4j)
    CREATE_INDEX = "create_index"
    DROP_INDEX = "drop_index"
    CREATE_CONSTRAINT = "create_constraint"
    DROP_CONSTRAINT = "drop_constraint"
    
    # Data operations
    TRANSFORM_DATA = "transform_data"
    BACKFILL_FIELD = "backfill_field"
    MIGRATE_RELATIONSHIPS = "migrate_relationships"
```

### Phase 3: Migration Manager

```python
class MigrationManager:
    """Manages schema migrations for GraphRAG projects"""
    
    def __init__(self, project_name: str, project_path: str):
        self.project_name = project_name
        self.project_path = Path(project_path)
        self.migrations_dir = self.project_path / ".graphrag" / "migrations"
        self.state_file = self.project_path / ".graphrag" / "migration_state.json"
        self.current_version = 0
        self.applied_migrations = []
        
    async def generate_migration(self, name: str) -> Migration:
        """Generate a new migration by comparing current state to schema.yaml"""
        current_schema = await self.get_current_schema()
        database_schema = await self.introspect_database()
        
        diff = self.calculate_diff(database_schema, current_schema)
        
        if not diff.has_changes():
            raise ValueError("No changes detected")
        
        migration = Migration(
            version=self.get_next_version(),
            name=name,
            timestamp=datetime.now(),
            operations=diff.to_operations()
        )
        
        # Generate rollback operations
        migration.down_operations = diff.to_rollback_operations()
        
        # Save migration file
        migration_file = self.migrations_dir / f"{migration.version:03d}_{name}.yaml"
        await self.save_migration(migration, migration_file)
        
        return migration
    
    async def migrate(self, target_version: Optional[int] = None) -> MigrationResult:
        """Apply migrations up to target version (or latest)"""
        migrations = await self.get_pending_migrations(target_version)
        
        if not migrations:
            return MigrationResult(status="no_changes", message="Already up to date")
        
        results = []
        for migration in migrations:
            try:
                # Create pre-migration snapshot for rollback
                snapshot = await self.create_snapshot(migration.version)
                
                # Execute migration operations
                await self.execute_migration(migration, direction="up")
                
                # Verify migration success
                await self.verify_migration(migration)
                
                # Update state
                self.mark_as_applied(migration)
                
                results.append({
                    "version": migration.version,
                    "name": migration.name,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Migration {migration.version} failed: {e}")
                
                # Attempt rollback
                if snapshot:
                    await self.restore_snapshot(snapshot)
                
                raise MigrationError(f"Migration failed at version {migration.version}: {e}")
        
        return MigrationResult(
            status="success",
            migrations_applied=results,
            current_version=self.current_version
        )
    
    async def rollback(self, target_version: int) -> MigrationResult:
        """Rollback to a specific version"""
        if target_version >= self.current_version:
            raise ValueError(f"Target version {target_version} must be less than current {self.current_version}")
        
        migrations_to_rollback = self.get_rollback_migrations(target_version)
        
        for migration in reversed(migrations_to_rollback):
            # Load rollback snapshot if available
            snapshot = await self.load_snapshot(migration.version)
            
            if snapshot:
                # Fast rollback using snapshot
                await self.restore_snapshot(snapshot)
            else:
                # Execute down operations
                await self.execute_migration(migration, direction="down")
            
            self.mark_as_rolled_back(migration)
        
        return MigrationResult(
            status="rolled_back",
            current_version=target_version
        )
    
    async def status(self) -> MigrationStatus:
        """Get current migration status"""
        return MigrationStatus(
            current_version=self.current_version,
            applied_migrations=self.applied_migrations,
            pending_migrations=await self.get_pending_migrations(),
            database_state=await self.introspect_database()
        )
```

### Phase 4: Data Migration Support

```python
class DataMigrator:
    """Handles data transformation during schema migrations"""
    
    async def transform_nodes(
        self,
        node_type: str,
        transformation: Callable,
        batch_size: int = 1000
    ):
        """Transform existing nodes during migration"""
        async with self.neo4j.session() as session:
            # Process in batches to avoid memory issues
            offset = 0
            while True:
                batch = await session.run(
                    f"MATCH (n:{node_type}) "
                    f"RETURN n SKIP {offset} LIMIT {batch_size}"
                )
                
                if not batch:
                    break
                
                transformed = []
                for record in batch:
                    node = record["n"]
                    new_props = transformation(node)
                    transformed.append((node.id, new_props))
                
                # Update nodes with new properties
                await self.batch_update_nodes(transformed)
                
                offset += batch_size
    
    async def migrate_vectors(
        self,
        old_collection: str,
        new_collection: str,
        field_mapping: Dict[str, str]
    ):
        """Migrate vectors between collections with field remapping"""
        # Scroll through old collection
        offset = None
        while True:
            records, offset = await self.qdrant.scroll(
                collection_name=old_collection,
                offset=offset,
                limit=100
            )
            
            if not records:
                break
            
            # Transform records
            new_records = []
            for record in records:
                new_payload = {}
                for old_field, new_field in field_mapping.items():
                    if old_field in record.payload:
                        new_payload[new_field] = record.payload[old_field]
                
                new_records.append(PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=new_payload
                ))
            
            # Insert into new collection
            await self.qdrant.upsert(
                collection_name=new_collection,
                points=new_records
            )
```

### Phase 5: Migration Tools (MCP Integration)

```python
# New MCP tools for migration management

@tool(name="migration_generate")
async def migration_generate_impl(arguments: dict) -> List[types.TextContent]:
    """Generate a new migration from schema changes"""
    name = arguments.get("name", f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    dry_run = arguments.get("dry_run", False)
    
    migration_manager = await get_migration_manager()
    migration = await migration_manager.generate_migration(name)
    
    if dry_run:
        # Show what would be changed without creating file
        return format_migration_preview(migration)
    
    return format_migration_created(migration)

@tool(name="migration_apply")
async def migration_apply_impl(arguments: dict) -> List[types.TextContent]:
    """Apply pending migrations"""
    target_version = arguments.get("target_version")  # None means latest
    dry_run = arguments.get("dry_run", False)
    
    migration_manager = await get_migration_manager()
    
    if dry_run:
        # Show what would be applied
        pending = await migration_manager.get_pending_migrations(target_version)
        return format_pending_migrations(pending)
    
    result = await migration_manager.migrate(target_version)
    return format_migration_result(result)

@tool(name="migration_rollback")
async def migration_rollback_impl(arguments: dict) -> List[types.TextContent]:
    """Rollback to a previous migration version"""
    target_version = arguments["target_version"]
    force = arguments.get("force", False)
    
    migration_manager = await get_migration_manager()
    
    if not force:
        # Confirm rollback with details of what will be lost
        impact = await migration_manager.analyze_rollback_impact(target_version)
        if impact.data_loss_risk:
            return format_rollback_warning(impact)
    
    result = await migration_manager.rollback(target_version)
    return format_rollback_result(result)

@tool(name="migration_status")
async def migration_status_impl(arguments: dict) -> List[types.TextContent]:
    """Get current migration status"""
    migration_manager = await get_migration_manager()
    status = await migration_manager.status()
    return format_migration_status(status)

@tool(name="schema_diff")
async def schema_diff_impl(arguments: dict) -> List[types.TextContent]:
    """Compare schema between environments or versions"""
    from_source = arguments.get("from", "database")  # database, file, version
    to_source = arguments.get("to", "schema.yaml")
    
    migration_manager = await get_migration_manager()
    diff = await migration_manager.calculate_diff(from_source, to_source)
    return format_schema_diff(diff)
```

## Migration Safety Features

### 1. Dry Run Mode
```bash
# Preview what migration would do without applying
migration_apply --dry-run

# Shows:
# - Operations to be executed
# - Estimated affected records
# - Rollback plan
```

### 2. Automatic Backups
```python
async def create_snapshot(self, version: int):
    """Create backup before migration"""
    snapshot = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "neo4j_dump": await self.dump_neo4j_schema(),
        "qdrant_collections": await self.dump_qdrant_metadata(),
        "sample_data": await self.sample_data(limit=100)
    }
    
    snapshot_file = self.rollback_dir / f"pre_{version:03d}_snapshot.json"
    await self.save_snapshot(snapshot, snapshot_file)
    return snapshot
```

### 3. Migration Verification
```python
async def verify_migration(self, migration: Migration):
    """Verify migration was successful"""
    for operation in migration.operations:
        if operation.type == "create_node_type":
            # Verify node type exists in Neo4j
            exists = await self.neo4j.node_type_exists(operation.params["name"])
            if not exists:
                raise MigrationError(f"Node type {operation.params['name']} not created")
        
        elif operation.type == "create_collection":
            # Verify collection exists in Qdrant
            exists = await self.qdrant.collection_exists(operation.params["name"])
            if not exists:
                raise MigrationError(f"Collection {operation.params['name']} not created")
```

### 4. Conflict Resolution
```python
async def check_conflicts(self, migration: Migration):
    """Check for conflicts with concurrent changes"""
    # Compare checksums
    current_checksum = await self.calculate_schema_checksum()
    if current_checksum != migration.base_checksum:
        conflicts = await self.identify_conflicts(migration)
        if conflicts.are_resolvable():
            return await self.auto_resolve_conflicts(conflicts)
        else:
            raise ConflictError("Manual conflict resolution required", conflicts)
```

## Migration Workflow

### Development Workflow
```bash
# 1. Make schema changes in schema.yaml
edit .graphrag/schema.yaml

# 2. Generate migration
mcp migration_generate --name "add_user_preferences"

# 3. Review generated migration
cat .graphrag/migrations/005_add_user_preferences.yaml

# 4. Test migration (dry run)
mcp migration_apply --dry-run

# 5. Apply migration
mcp migration_apply

# 6. If issues, rollback
mcp migration_rollback --target-version 4
```

### Team Collaboration
```bash
# Developer A creates migration
mcp migration_generate --name "add_auth_fields"
git add .graphrag/migrations/
git commit -m "Add auth fields migration"
git push

# Developer B pulls and applies
git pull
mcp migration_apply  # Automatically applies new migrations
```

### Production Deployment
```bash
# 1. Test migrations in staging
mcp migration_apply --environment staging --dry-run
mcp migration_apply --environment staging

# 2. Create backup
mcp migration_backup --environment production

# 3. Apply to production with monitoring
mcp migration_apply --environment production --monitor

# 4. If issues detected, automatic rollback
# (triggered by monitoring thresholds)
```

## Testing Strategy

```python
class MigrationTest:
    """Test framework for migrations"""
    
    async def test_migration_up_down(self, migration: Migration):
        """Test that migration can be applied and rolled back"""
        # Start with clean state
        await self.reset_test_database()
        
        # Apply migration
        await self.migration_manager.execute_migration(migration, "up")
        
        # Verify expected state
        await self.assert_schema_matches(migration.expected_schema)
        
        # Rollback
        await self.migration_manager.execute_migration(migration, "down")
        
        # Verify returned to original state
        await self.assert_schema_matches(self.original_schema)
    
    async def test_data_migration(self, migration: Migration):
        """Test that data is correctly transformed"""
        # Insert test data
        test_data = await self.create_test_data()
        
        # Apply migration
        await self.migration_manager.execute_migration(migration, "up")
        
        # Verify data transformation
        transformed_data = await self.fetch_transformed_data()
        await self.assert_data_transformed_correctly(test_data, transformed_data)
```

## Benefits

### For Developers
- **Safe schema evolution** with rollback capability
- **Team collaboration** through version-controlled migrations
- **Conflict detection** for concurrent schema changes
- **Testing support** with dry-run and staging environments

### For Operations
- **Production safety** with automatic backups and verification
- **Audit trail** of all schema changes
- **Monitoring integration** for automatic rollback
- **Zero-downtime migrations** with careful planning

### For Data Integrity
- **Data transformation** support during schema changes
- **Referential integrity** maintenance
- **Vector collection** migration with field mapping
- **Incremental rollback** to any previous version

## Consequences

### Positive
- Complete schema version control
- Safe production deployments
- Team collaboration support
- Data integrity during migrations
- Comprehensive audit trail

### Negative
- Additional complexity in schema management
- Storage overhead for migration history and snapshots
- Potential for migration conflicts in teams
- Learning curve for migration workflow

### Neutral
- Requires discipline in schema change process
- Migration files must be immutable once applied
- Order of migrations matters

## Implementation Timeline

### Phase 1: Core Migration Engine (Week 1)
- Migration file structure
- Basic up/down operations
- State tracking

### Phase 2: Data Migration (Week 2)
- Node transformation
- Vector migration
- Batch processing

### Phase 3: Safety Features (Week 3)
- Snapshots and rollback
- Verification system
- Conflict detection

### Phase 4: MCP Integration (Week 4)
- Migration tools
- CLI commands
- Status reporting

### Phase 5: Testing & Documentation (Week 5)
- Test framework
- Migration guides
- Team workflows

## Example Migrations

### Adding a New Property
```yaml
version: 2
name: add_component_state
up:
  - operation: add_node_property
    params:
      node_type: Component
      property: state
      type: json
      default: "{}"
  
  - operation: backfill_field
    params:
      node_type: Component
      field: state
      value: "{}"

down:
  - operation: remove_node_property
    params:
      node_type: Component
      property: state
```

### Renaming a Relationship
```yaml
version: 3
name: rename_uses_to_imports
up:
  - operation: create_relationship_type
    params:
      name: IMPORTS
      from_types: [Component]
      to_types: [Module]
  
  - operation: migrate_relationships
    params:
      from_type: USES
      to_type: IMPORTS
      transform: "MATCH (a)-[r:USES]->(b) CREATE (a)-[:IMPORTS]->(b) DELETE r"
  
  - operation: drop_relationship_type
    params:
      name: USES

down:
  - operation: create_relationship_type
    params:
      name: USES
      from_types: [Component]
      to_types: [Module]
  
  - operation: migrate_relationships
    params:
      from_type: IMPORTS
      to_type: USES
  
  - operation: drop_relationship_type
    params:
      name: IMPORTS
```

## References

- Django Migrations: https://docs.djangoproject.com/en/4.2/topics/migrations/
- Rails Active Record Migrations: https://guides.rubyonrails.org/active_record_migrations.html
- Supabase Migrations: https://supabase.com/docs/guides/database/migrations
- Flyway Database Migrations: https://flywaydb.org/
- Neo4j Migration Tools: https://neo4j.com/labs/neo4j-migrations/

## Decision Outcome

Implement a full schema migration system for GraphRAG with version control, rollback capability, data transformation support, and comprehensive safety features. Start with Phase 1 (core engine) and progressively add features.

**Target: 5 weeks for full implementation**