#!/usr/bin/env python3
"""
GraphRAG Schema Migration Manager
Implements ADR-0021 for full schema migration system
"""

import os
import json
import yaml
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


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


@dataclass
class MigrationStep:
    """Single operation in a migration"""
    operation: str
    params: Dict[str, Any]
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "params": self.params,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MigrationStep':
        return cls(
            operation=data["operation"],
            params=data.get("params", {}),
            description=data.get("description", "")
        )


@dataclass
class Migration:
    """A single migration with up and down operations"""
    version: int
    name: str
    timestamp: datetime
    author: str = "system"
    description: str = ""
    up_operations: List[MigrationStep] = field(default_factory=list)
    down_operations: List[MigrationStep] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "description": self.description,
            "up": [op.to_dict() for op in self.up_operations],
            "down": [op.to_dict() for op in self.down_operations],
            "dependencies": self.dependencies,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Migration':
        return cls(
            version=data["version"],
            name=data.get("name", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            author=data.get("author", "system"),
            description=data.get("description", ""),
            up_operations=[MigrationStep.from_dict(op) for op in data.get("up", [])],
            down_operations=[MigrationStep.from_dict(op) for op in data.get("down", [])],
            dependencies=data.get("dependencies", []),
            checksum=data.get("checksum", "")
        )
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for migration integrity"""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class MigrationState:
    """Current migration state for a project"""
    current_version: int = 0
    applied_migrations: List[Dict[str, Any]] = field(default_factory=list)
    last_applied: Optional[datetime] = None
    schema_checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_version": self.current_version,
            "applied_migrations": self.applied_migrations,
            "last_applied": self.last_applied.isoformat() if self.last_applied else None,
            "schema_checksum": self.schema_checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MigrationState':
        return cls(
            current_version=data.get("current_version", 0),
            applied_migrations=data.get("applied_migrations", []),
            last_applied=datetime.fromisoformat(data["last_applied"]) if data.get("last_applied") else None,
            schema_checksum=data.get("schema_checksum", "")
        )


@dataclass
class MigrationResult:
    """Result of a migration operation"""
    status: str  # success, failed, rolled_back, no_changes
    message: str = ""
    migrations_applied: List[Dict[str, Any]] = field(default_factory=list)
    current_version: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class MigrationDiff:
    """Differences between two schemas"""
    added_nodes: List[str] = field(default_factory=list)
    removed_nodes: List[str] = field(default_factory=list)
    modified_nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    added_relationships: List[str] = field(default_factory=list)
    removed_relationships: List[str] = field(default_factory=list)
    modified_relationships: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    added_collections: List[str] = field(default_factory=list)
    removed_collections: List[str] = field(default_factory=list)
    modified_collections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def has_changes(self) -> bool:
        """Check if there are any changes"""
        return bool(
            self.added_nodes or self.removed_nodes or self.modified_nodes or
            self.added_relationships or self.removed_relationships or self.modified_relationships or
            self.added_collections or self.removed_collections or self.modified_collections
        )
    
    def to_operations(self) -> List[MigrationStep]:
        """Convert diff to migration operations"""
        operations = []
        
        # Node operations
        for node in self.removed_nodes:
            operations.append(MigrationStep(
                operation=MigrationOperation.DROP_NODE_TYPE.value,
                params={"name": node}
            ))
        
        for node in self.added_nodes:
            operations.append(MigrationStep(
                operation=MigrationOperation.CREATE_NODE_TYPE.value,
                params={"name": node}
            ))
        
        for node, changes in self.modified_nodes.items():
            if "properties_added" in changes:
                for prop in changes["properties_added"]:
                    operations.append(MigrationStep(
                        operation=MigrationOperation.ADD_NODE_PROPERTY.value,
                        params={"node_type": node, "property": prop}
                    ))
            if "properties_removed" in changes:
                for prop in changes["properties_removed"]:
                    operations.append(MigrationStep(
                        operation=MigrationOperation.REMOVE_NODE_PROPERTY.value,
                        params={"node_type": node, "property": prop}
                    ))
        
        # Relationship operations
        for rel in self.removed_relationships:
            operations.append(MigrationStep(
                operation=MigrationOperation.DROP_RELATIONSHIP_TYPE.value,
                params={"name": rel}
            ))
        
        for rel in self.added_relationships:
            operations.append(MigrationStep(
                operation=MigrationOperation.CREATE_RELATIONSHIP_TYPE.value,
                params={"name": rel}
            ))
        
        # Collection operations
        for col in self.removed_collections:
            operations.append(MigrationStep(
                operation=MigrationOperation.DROP_COLLECTION.value,
                params={"name": col}
            ))
        
        for col in self.added_collections:
            operations.append(MigrationStep(
                operation=MigrationOperation.CREATE_COLLECTION.value,
                params={"name": col}
            ))
        
        return operations
    
    def to_rollback_operations(self) -> List[MigrationStep]:
        """Generate rollback operations (reverse of forward operations)"""
        operations = []
        
        # Reverse node operations
        for node in self.added_nodes:
            operations.append(MigrationStep(
                operation=MigrationOperation.DROP_NODE_TYPE.value,
                params={"name": node}
            ))
        
        for node in self.removed_nodes:
            operations.append(MigrationStep(
                operation=MigrationOperation.CREATE_NODE_TYPE.value,
                params={"name": node}
            ))
        
        # Reverse property operations
        for node, changes in self.modified_nodes.items():
            if "properties_removed" in changes:
                for prop in changes["properties_removed"]:
                    operations.append(MigrationStep(
                        operation=MigrationOperation.ADD_NODE_PROPERTY.value,
                        params={"node_type": node, "property": prop}
                    ))
            if "properties_added" in changes:
                for prop in changes["properties_added"]:
                    operations.append(MigrationStep(
                        operation=MigrationOperation.REMOVE_NODE_PROPERTY.value,
                        params={"node_type": node, "property": prop}
                    ))
        
        # Reverse relationship operations
        for rel in self.added_relationships:
            operations.append(MigrationStep(
                operation=MigrationOperation.DROP_RELATIONSHIP_TYPE.value,
                params={"name": rel}
            ))
        
        for rel in self.removed_relationships:
            operations.append(MigrationStep(
                operation=MigrationOperation.CREATE_RELATIONSHIP_TYPE.value,
                params={"name": rel}
            ))
        
        # Reverse collection operations
        for col in self.added_collections:
            operations.append(MigrationStep(
                operation=MigrationOperation.DROP_COLLECTION.value,
                params={"name": col}
            ))
        
        for col in self.removed_collections:
            operations.append(MigrationStep(
                operation=MigrationOperation.CREATE_COLLECTION.value,
                params={"name": col}
            ))
        
        return operations


class MigrationManager:
    """Manages schema migrations for GraphRAG projects"""
    
    def __init__(self, project_name: str, project_path: str = None,
                 neo4j_service=None, qdrant_service=None):
        self.project_name = project_name
        self.project_path = Path(project_path or os.getcwd())
        self.migrations_dir = self.project_path / ".graphrag" / "migrations"
        self.state_file = self.project_path / ".graphrag" / "migration_state.json"
        self.rollback_dir = self.project_path / ".graphrag" / "rollback"
        self.schema_file = self.project_path / ".graphrag" / "schema.yaml"

        # Ensure directories exist
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_dir.mkdir(parents=True, exist_ok=True)

        # Load current state
        self.state = self._load_state()

        # Service connections
        self.neo4j = neo4j_service
        self.qdrant = qdrant_service
    
    def _load_state(self) -> MigrationState:
        """Load migration state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                return MigrationState.from_dict(data)
        return MigrationState()
    
    def _save_state(self):
        """Save migration state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)
    
    def get_next_version(self) -> int:
        """Get the next migration version number"""
        existing_migrations = list(self.migrations_dir.glob("*.yaml"))
        if not existing_migrations:
            return 1
        
        versions = []
        for migration_file in existing_migrations:
            # Extract version from filename (e.g., "001_initial.yaml" -> 1)
            try:
                version = int(migration_file.stem.split('_')[0])
                versions.append(version)
            except (ValueError, IndexError):
                continue
        
        return max(versions) + 1 if versions else 1
    
    async def load_migration(self, version: int) -> Optional[Migration]:
        """Load a migration from file"""
        migration_files = list(self.migrations_dir.glob(f"{version:03d}_*.yaml"))
        if not migration_files:
            return None
        
        with open(migration_files[0], 'r') as f:
            data = yaml.safe_load(f)
            return Migration.from_dict(data)
    
    async def save_migration(self, migration: Migration, migration_file: Path):
        """Save a migration to file"""
        # Calculate checksum
        migration.checksum = migration.calculate_checksum()
        
        # Save to file
        with open(migration_file, 'w') as f:
            yaml.dump(migration.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved migration to {migration_file}")
    
    async def get_pending_migrations(self, target_version: Optional[int] = None) -> List[Migration]:
        """Get list of migrations that need to be applied"""
        pending = []
        
        # Get all migration files
        migration_files = sorted(self.migrations_dir.glob("*.yaml"))
        
        for migration_file in migration_files:
            # Extract version from filename
            try:
                version = int(migration_file.stem.split('_')[0])
            except (ValueError, IndexError):
                continue
            
            # Skip if already applied
            if version <= self.state.current_version:
                continue
            
            # Skip if beyond target version
            if target_version and version > target_version:
                continue
            
            # Load migration
            migration = await self.load_migration(version)
            if migration:
                pending.append(migration)
        
        return sorted(pending, key=lambda m: m.version)
    
    async def get_applied_migrations(self) -> List[Migration]:
        """Get list of applied migrations"""
        applied = []
        for version in range(1, self.state.current_version + 1):
            migration = await self.load_migration(version)
            if migration:
                applied.append(migration)
        return applied
    
    async def calculate_diff(self, from_schema: Dict, to_schema: Dict) -> MigrationDiff:
        """Calculate differences between two schemas"""
        diff = MigrationDiff()
        
        # Compare node types
        from_nodes = set(from_schema.get("node_types", {}).keys())
        to_nodes = set(to_schema.get("node_types", {}).keys())
        
        diff.added_nodes = list(to_nodes - from_nodes)
        diff.removed_nodes = list(from_nodes - to_nodes)
        
        # Check for modified nodes
        for node in from_nodes & to_nodes:
            from_props = set(from_schema["node_types"][node].get("properties", {}).keys())
            to_props = set(to_schema["node_types"][node].get("properties", {}).keys())
            
            if from_props != to_props:
                diff.modified_nodes[node] = {
                    "properties_added": list(to_props - from_props),
                    "properties_removed": list(from_props - to_props)
                }
        
        # Compare relationships
        from_rels = set(from_schema.get("relationship_types", {}).keys())
        to_rels = set(to_schema.get("relationship_types", {}).keys())
        
        diff.added_relationships = list(to_rels - from_rels)
        diff.removed_relationships = list(from_rels - to_rels)
        
        # Compare collections
        from_cols = set(from_schema.get("collections", {}).keys())
        to_cols = set(to_schema.get("collections", {}).keys())
        
        diff.added_collections = list(to_cols - from_cols)
        diff.removed_collections = list(from_cols - to_cols)
        
        return diff
    
    async def generate_migration(self, name: str, description: str = "") -> Migration:
        """Generate a new migration by comparing current state to schema.yaml"""
        # Load current schema from file
        if not self.schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_file}")
        
        with open(self.schema_file, 'r') as f:
            target_schema = yaml.safe_load(f)
        
        # Get current database schema
        current_schema = await self.introspect_database()
        
        # Calculate diff
        diff = await self.calculate_diff(current_schema, target_schema)
        
        if not diff.has_changes():
            raise ValueError("No changes detected between current database and schema.yaml")
        
        # Create migration
        migration = Migration(
            version=self.get_next_version(),
            name=name,
            timestamp=datetime.now(),
            description=description,
            up_operations=diff.to_operations(),
            down_operations=diff.to_rollback_operations()
        )
        
        # Save migration file
        migration_file = self.migrations_dir / f"{migration.version:03d}_{name}.yaml"
        await self.save_migration(migration, migration_file)
        
        return migration
    
    async def introspect_database(self) -> Dict[str, Any]:
        """Get current schema from database"""
        schema = {
            "node_types": {},
            "relationship_types": {},
            "collections": {}
        }
        
        if self.neo4j:
            # Get node types from Neo4j
            try:
                # Query for all node labels
                query = "CALL db.labels() YIELD label RETURN label"
                result = await self.neo4j.execute_query(query)
                
                for record in result:
                    label = record["label"]
                    # Get properties for this node type
                    prop_query = f"""
                    MATCH (n:{label})
                    WITH keys(n) AS props
                    UNWIND props AS prop
                    RETURN DISTINCT prop
                    LIMIT 100
                    """
                    prop_result = await self.neo4j.execute_query(prop_query)
                    properties = {prop["prop"]: "string" for prop in prop_result}
                    
                    schema["node_types"][label] = {"properties": properties}
            except Exception as e:
                logger.warning(f"Could not introspect Neo4j: {e}")
        
        if self.qdrant:
            # Get collections from Qdrant
            try:
                collections = await self.qdrant.get_collections()
                for collection in collections.collections:
                    schema["collections"][collection.name] = {
                        "vector_size": collection.config.params.vectors.size,
                        "distance": collection.config.params.vectors.distance
                    }
            except Exception as e:
                logger.warning(f"Could not introspect Qdrant: {e}")
        
        return schema
    
    async def create_snapshot(self, version: int) -> Dict[str, Any]:
        """Create a snapshot before migration"""
        snapshot = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name,
            "schema": await self.introspect_database(),
            "state": self.state.to_dict()
        }
        
        # Save snapshot
        snapshot_file = self.rollback_dir / f"pre_{version:03d}_snapshot.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        logger.info(f"Created snapshot: {snapshot_file}")
        return snapshot
    
    async def restore_snapshot(self, snapshot: Dict[str, Any]):
        """Restore from a snapshot"""
        # This would restore the database state
        # Implementation depends on Neo4j and Qdrant backup/restore capabilities
        logger.info(f"Restoring from snapshot version {snapshot['version']}")
        
        # Restore state
        self.state = MigrationState.from_dict(snapshot["state"])
        self._save_state()
    
    async def execute_operation(self, operation: MigrationStep):
        """Execute a single migration operation"""
        op_type = MigrationOperation(operation.operation)
        params = operation.params
        
        if op_type == MigrationOperation.CREATE_NODE_TYPE:
            if self.neo4j:
                # Create constraint for node type
                constraint = f"""
                CREATE CONSTRAINT {params['name']}_unique 
                IF NOT EXISTS 
                FOR (n:{params['name']}) 
                REQUIRE n.id IS UNIQUE
                """
                await self.neo4j.execute_query(constraint)
                logger.info(f"Created node type: {params['name']}")
        
        elif op_type == MigrationOperation.DROP_NODE_TYPE:
            if self.neo4j:
                # Drop all nodes of this type
                await self.neo4j.execute_query(f"MATCH (n:{params['name']}) DETACH DELETE n")
                # Drop constraint
                await self.neo4j.execute_query(
                    f"DROP CONSTRAINT {params['name']}_unique IF EXISTS"
                )
                logger.info(f"Dropped node type: {params['name']}")
        
        elif op_type == MigrationOperation.CREATE_COLLECTION:
            if self.qdrant:
                from qdrant_client.models import VectorParams, Distance
                await self.qdrant.create_collection(
                    collection_name=params["name"],
                    vectors_config=VectorParams(
                        size=params.get("vector_size", 768),
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {params['name']}")
        
        elif op_type == MigrationOperation.DROP_COLLECTION:
            if self.qdrant:
                await self.qdrant.delete_collection(params["name"])
                logger.info(f"Dropped collection: {params['name']}")
        
        # Add more operation implementations as needed
        else:
            logger.warning(f"Operation {op_type} not yet implemented")
    
    async def execute_migration(self, migration: Migration, direction: str = "up"):
        """Execute a migration in the specified direction"""
        operations = migration.up_operations if direction == "up" else migration.down_operations
        
        logger.info(f"Executing migration {migration.version} ({migration.name}) - {direction}")
        
        for operation in operations:
            try:
                await self.execute_operation(operation)
            except Exception as e:
                logger.error(f"Failed to execute operation: {operation.operation} - {e}")
                raise
        
        logger.info(f"Migration {migration.version} completed successfully")
    
    async def migrate(self, target_version: Optional[int] = None, dry_run: bool = False) -> MigrationResult:
        """Apply migrations up to target version"""
        pending = await self.get_pending_migrations(target_version)
        
        if not pending:
            return MigrationResult(
                status="no_changes",
                message="Already up to date",
                current_version=self.state.current_version
            )
        
        if dry_run:
            # Just return what would be done
            return MigrationResult(
                status="dry_run",
                message=f"Would apply {len(pending)} migrations",
                migrations_applied=[{"version": m.version, "name": m.name} for m in pending],
                current_version=self.state.current_version
            )
        
        results = []
        for migration in pending:
            try:
                # Create snapshot before migration
                await self.create_snapshot(migration.version)
                
                # Execute migration
                await self.execute_migration(migration, direction="up")
                
                # Update state
                self.state.current_version = migration.version
                self.state.applied_migrations.append({
                    "version": migration.version,
                    "name": migration.name,
                    "applied_at": datetime.now().isoformat()
                })
                self.state.last_applied = datetime.now()
                self._save_state()
                
                results.append({
                    "version": migration.version,
                    "name": migration.name,
                    "status": "success"
                })
                
            except Exception as e:
                logger.error(f"Migration {migration.version} failed: {e}")
                return MigrationResult(
                    status="failed",
                    message=f"Migration failed at version {migration.version}",
                    migrations_applied=results,
                    current_version=self.state.current_version,
                    errors=[str(e)]
                )
        
        return MigrationResult(
            status="success",
            message=f"Applied {len(results)} migrations",
            migrations_applied=results,
            current_version=self.state.current_version
        )
    
    async def rollback(self, target_version: int) -> MigrationResult:
        """Rollback to a specific version"""
        if target_version >= self.state.current_version:
            return MigrationResult(
                status="no_changes",
                message=f"Already at version {self.state.current_version}",
                current_version=self.state.current_version
            )
        
        # Get migrations to rollback (in reverse order)
        to_rollback = []
        for version in range(self.state.current_version, target_version, -1):
            migration = await self.load_migration(version)
            if migration:
                to_rollback.append(migration)
        
        results = []
        for migration in to_rollback:
            try:
                # Try to restore from snapshot first
                snapshot_file = self.rollback_dir / f"pre_{migration.version:03d}_snapshot.json"
                if snapshot_file.exists():
                    with open(snapshot_file, 'r') as f:
                        snapshot = json.load(f)
                    await self.restore_snapshot(snapshot)
                    logger.info(f"Restored from snapshot for version {migration.version}")
                else:
                    # Execute down operations
                    await self.execute_migration(migration, direction="down")
                
                # Update state
                self.state.current_version = migration.version - 1
                self.state.applied_migrations = [
                    m for m in self.state.applied_migrations 
                    if m["version"] < migration.version
                ]
                self._save_state()
                
                results.append({
                    "version": migration.version,
                    "name": migration.name,
                    "status": "rolled_back"
                })
                
            except Exception as e:
                logger.error(f"Rollback of {migration.version} failed: {e}")
                return MigrationResult(
                    status="failed",
                    message=f"Rollback failed at version {migration.version}",
                    migrations_applied=results,
                    current_version=self.state.current_version,
                    errors=[str(e)]
                )
        
        return MigrationResult(
            status="rolled_back",
            message=f"Rolled back {len(results)} migrations",
            migrations_applied=results,
            current_version=self.state.current_version
        )
    
    async def status(self) -> Dict[str, Any]:
        """Get current migration status"""
        pending = await self.get_pending_migrations()
        applied = await self.get_applied_migrations()
        
        return {
            "current_version": self.state.current_version,
            "last_applied": self.state.last_applied.isoformat() if self.state.last_applied else None,
            "applied_migrations": [
                {"version": m.version, "name": m.name, "timestamp": m.timestamp.isoformat()}
                for m in applied
            ],
            "pending_migrations": [
                {"version": m.version, "name": m.name, "timestamp": m.timestamp.isoformat()}
                for m in pending
            ],
            "total_applied": len(applied),
            "total_pending": len(pending)
        }