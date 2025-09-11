#!/usr/bin/env python3
"""
Per-Project GraphRAG Schema Manager
Implements ADR-0020 for custom project schemas
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ProjectType(Enum):
    """Common project types with built-in schemas"""
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    NEXTJS = "nextjs"
    DJANGO = "django"
    FASTAPI = "fastapi"
    FLASK = "flask"
    EXPRESS = "express"
    SPRINGBOOT = "springboot"
    RAILS = "rails"
    GENERIC = "generic"


@dataclass
class NodeType:
    """Definition of a graph node type"""
    name: str
    properties: Dict[str, str]
    indexes: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_cypher_constraint(self) -> str:
        """Generate Cypher constraint for this node type"""
        return f"CREATE CONSTRAINT {self.name}_unique IF NOT EXISTS FOR (n:{self.name}) REQUIRE n.id IS UNIQUE"
    
    def to_cypher_indexes(self) -> List[str]:
        """Generate Cypher indexes for this node type"""
        indexes = []
        for idx in self.indexes:
            indexes.append(
                f"CREATE INDEX {self.name}_{idx}_idx IF NOT EXISTS FOR (n:{self.name}) ON (n.{idx})"
            )
        return indexes


@dataclass
class RelationshipType:
    """Definition of a graph relationship type"""
    name: str
    from_types: List[str]
    to_types: List[str]
    properties: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    
    def validate(self, from_node: str, to_node: str) -> bool:
        """Check if relationship is valid between node types"""
        return from_node in self.from_types and to_node in self.to_types


@dataclass
class CollectionSchema:
    """Vector collection schema definition"""
    name: str
    vector_size: int
    distance_metric: str = "cosine"
    fields: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class GraphRAGSchema:
    """Complete GraphRAG schema for a project"""
    version: str = "1.0"
    project_type: ProjectType = ProjectType.GENERIC
    description: str = ""
    extends: List[str] = field(default_factory=list)
    node_types: Dict[str, NodeType] = field(default_factory=dict)
    relationship_types: Dict[str, RelationshipType] = field(default_factory=dict)
    collections: Dict[str, CollectionSchema] = field(default_factory=dict)
    extractors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class SchemaManager:
    """Manage per-project GraphRAG schemas"""
    
    # Built-in schema templates
    BUILTIN_SCHEMAS = {
        ProjectType.REACT: {
            "node_types": {
                "Component": {
                    "properties": {"name": "string", "type": "string", "props": "json", "file_path": "string"},
                    "indexes": ["name", "file_path"]
                },
                "Hook": {
                    "properties": {"name": "string", "custom": "boolean", "dependencies": "list"},
                    "indexes": ["name"]
                },
                "Context": {
                    "properties": {"name": "string", "provider": "string", "default_value": "json"},
                    "indexes": ["name"]
                },
                "Route": {
                    "properties": {"path": "string", "component": "string", "exact": "boolean"},
                    "indexes": ["path"]
                }
            },
            "relationship_types": {
                "USES_HOOK": {"from": ["Component", "Hook"], "to": ["Hook"]},
                "RENDERS": {"from": ["Component"], "to": ["Component"]},
                "PROVIDES_CONTEXT": {"from": ["Component"], "to": ["Context"]},
                "CONSUMES_CONTEXT": {"from": ["Component"], "to": ["Context"]},
                "ROUTES_TO": {"from": ["Route"], "to": ["Component"]}
            }
        },
        ProjectType.FASTAPI: {
            "node_types": {
                "Endpoint": {
                    "properties": {"path": "string", "method": "string", "response_model": "string"},
                    "indexes": ["path", "method"]
                },
                "Model": {
                    "properties": {"name": "string", "fields": "json", "validators": "list"},
                    "indexes": ["name"]
                },
                "Dependency": {
                    "properties": {"name": "string", "scope": "string", "injectable": "boolean"},
                    "indexes": ["name"]
                },
                "BackgroundTask": {
                    "properties": {"name": "string", "schedule": "string", "queue": "string"},
                    "indexes": ["name"]
                }
            },
            "relationship_types": {
                "VALIDATES": {"from": ["Endpoint"], "to": ["Model"]},
                "DEPENDS_ON": {"from": ["Endpoint", "BackgroundTask"], "to": ["Dependency"]},
                "RETURNS": {"from": ["Endpoint"], "to": ["Model"]},
                "TRIGGERS": {"from": ["Endpoint"], "to": ["BackgroundTask"]}
            }
        },
        ProjectType.DJANGO: {
            "node_types": {
                "Model": {
                    "properties": {"name": "string", "table": "string", "fields": "json", "abstract": "boolean"},
                    "indexes": ["name", "table"]
                },
                "View": {
                    "properties": {"name": "string", "url_pattern": "string", "template": "string"},
                    "indexes": ["name", "url_pattern"]
                },
                "Serializer": {
                    "properties": {"name": "string", "model": "string", "fields": "list"},
                    "indexes": ["name"]
                },
                "Signal": {
                    "properties": {"name": "string", "sender": "string", "receiver": "string"},
                    "indexes": ["name"]
                },
                "Migration": {
                    "properties": {"name": "string", "app": "string", "dependencies": "list"},
                    "indexes": ["name", "app"]
                }
            },
            "relationship_types": {
                "QUERIES": {"from": ["View", "Serializer"], "to": ["Model"]},
                "SERIALIZES": {"from": ["Serializer"], "to": ["Model"]},
                "FOREIGN_KEY": {"from": ["Model"], "to": ["Model"]},
                "LISTENS": {"from": ["Signal"], "to": ["Model"]},
                "MIGRATES": {"from": ["Migration"], "to": ["Model"]}
            }
        }
    }
    
    def __init__(self, project_name: str, project_path: str = None):
        self.project_name = project_name
        self.project_path = project_path or os.getcwd()
        self.schema_dir = Path(self.project_path) / ".graphrag"
        self.schema_file = self.schema_dir / "schema.yaml"
        self.current_schema: Optional[GraphRAGSchema] = None
        
    async def initialize(self):
        """Initialize or load project schema"""
        if self.schema_file.exists():
            self.current_schema = await self.load_schema()
            logger.info(f"Loaded existing schema for project {self.project_name}")
        else:
            # Auto-detect and create schema
            project_type = await self.detect_project_type()
            self.current_schema = await self.create_schema(project_type)
            logger.info(f"Created new {project_type.value} schema for project {self.project_name}")
    
    async def detect_project_type(self) -> ProjectType:
        """Auto-detect project type from files and dependencies"""
        project_path = Path(self.project_path)
        
        # Check for package.json (Node.js projects)
        package_json = project_path / "package.json"
        if package_json.exists():
            with open(package_json) as f:
                data = json.load(f)
                deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
                
                if "react" in deps:
                    if "next" in deps:
                        return ProjectType.NEXTJS
                    return ProjectType.REACT
                elif "vue" in deps:
                    return ProjectType.VUE
                elif "@angular/core" in deps:
                    return ProjectType.ANGULAR
                elif "express" in deps:
                    return ProjectType.EXPRESS
        
        # Check for Python projects
        requirements = project_path / "requirements.txt"
        pyproject = project_path / "pyproject.toml"
        
        if requirements.exists() or pyproject.exists():
            # Read requirements
            deps = set()
            if requirements.exists():
                with open(requirements) as f:
                    deps.update(line.split("==")[0].strip() for line in f)
            
            if "django" in deps or (project_path / "manage.py").exists():
                return ProjectType.DJANGO
            elif "fastapi" in deps:
                return ProjectType.FASTAPI
            elif "flask" in deps:
                return ProjectType.FLASK
        
        # Check for Java/Spring
        if (project_path / "pom.xml").exists():
            return ProjectType.SPRINGBOOT
        
        # Check for Ruby/Rails
        if (project_path / "Gemfile").exists():
            with open(project_path / "Gemfile") as f:
                if "rails" in f.read():
                    return ProjectType.RAILS
        
        return ProjectType.GENERIC
    
    async def create_schema(self, project_type: ProjectType = None) -> GraphRAGSchema:
        """Create new schema for project"""
        if not project_type:
            project_type = await self.detect_project_type()
        
        schema = GraphRAGSchema(
            project_type=project_type,
            description=f"GraphRAG schema for {self.project_name} ({project_type.value} project)"
        )
        
        # Apply built-in template if available
        if project_type in self.BUILTIN_SCHEMAS:
            template = self.BUILTIN_SCHEMAS[project_type]
            
            # Add node types
            for name, config in template.get("node_types", {}).items():
                schema.node_types[name] = NodeType(
                    name=name,
                    properties=config["properties"],
                    indexes=config.get("indexes", []),
                    description=config.get("description", "")
                )
            
            # Add relationship types
            for name, config in template.get("relationship_types", {}).items():
                schema.relationship_types[name] = RelationshipType(
                    name=name,
                    from_types=config["from"],
                    to_types=config["to"],
                    properties=config.get("properties", {}),
                    description=config.get("description", "")
                )
        
        # Add default collections
        schema.collections = {
            "code": CollectionSchema(
                name=f"project_{self.project_name}_code",
                vector_size=768,
                fields=["content", "file_path", "language", "node_type"]
            ),
            "docs": CollectionSchema(
                name=f"project_{self.project_name}_docs",
                vector_size=768,
                fields=["content", "file_path", "title", "section"]
            )
        }
        
        # Save schema
        await self.save_schema(schema)
        
        return schema
    
    async def load_schema(self) -> GraphRAGSchema:
        """Load schema from YAML file"""
        with open(self.schema_file) as f:
            data = yaml.safe_load(f)
        
        schema = GraphRAGSchema(
            version=data.get("version", "1.0"),
            project_type=ProjectType(data.get("project_type", "generic")),
            description=data.get("description", ""),
            extends=data.get("extends", [])
        )
        
        # Load node types
        for name, config in data.get("node_types", {}).items():
            schema.node_types[name] = NodeType(
                name=name,
                properties=config.get("properties", {}),
                indexes=config.get("indexes", []),
                constraints=config.get("constraints", []),
                description=config.get("description", "")
            )
        
        # Load relationship types
        for name, config in data.get("relationship_types", {}).items():
            schema.relationship_types[name] = RelationshipType(
                name=name,
                from_types=config.get("from", []),
                to_types=config.get("to", []),
                properties=config.get("properties", {}),
                description=config.get("description", "")
            )
        
        # Load collections
        for name, config in data.get("collections", {}).items():
            schema.collections[name] = CollectionSchema(
                name=config.get("name", name),
                vector_size=config.get("vector_size", 768),
                distance_metric=config.get("distance", "cosine"),
                fields=config.get("fields", []),
                description=config.get("description", "")
            )
        
        return schema
    
    async def save_schema(self, schema: GraphRAGSchema):
        """Save schema to YAML file"""
        # Ensure directory exists
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        data = {
            "version": schema.version,
            "project_type": schema.project_type.value,
            "description": schema.description,
            "extends": schema.extends,
            "created_at": schema.created_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "node_types": {},
            "relationship_types": {},
            "collections": {}
        }
        
        # Add node types
        for name, node in schema.node_types.items():
            data["node_types"][name] = {
                "properties": node.properties,
                "indexes": node.indexes,
                "constraints": node.constraints,
                "description": node.description
            }
        
        # Add relationship types
        for name, rel in schema.relationship_types.items():
            data["relationship_types"][name] = {
                "from": rel.from_types,
                "to": rel.to_types,
                "properties": rel.properties,
                "description": rel.description
            }
        
        # Add collections
        for name, col in schema.collections.items():
            data["collections"][name] = {
                "name": col.name,
                "vector_size": col.vector_size,
                "distance": col.distance_metric,
                "fields": col.fields,
                "description": col.description
            }
        
        # Write to file
        with open(self.schema_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved schema to {self.schema_file}")
    
    async def apply_to_neo4j(self, neo4j_service):
        """Apply schema constraints and indexes to Neo4j"""
        if not self.current_schema:
            await self.initialize()
        
        cypher_commands = []
        
        # Generate constraints and indexes for node types
        for node_type in self.current_schema.node_types.values():
            cypher_commands.append(node_type.to_cypher_constraint())
            cypher_commands.extend(node_type.to_cypher_indexes())
        
        # Execute commands
        for cmd in cypher_commands:
            try:
                await neo4j_service.execute_query(cmd)
                logger.info(f"Applied: {cmd}")
            except Exception as e:
                logger.warning(f"Failed to apply {cmd}: {e}")
    
    async def validate_node(self, node_type: str, properties: dict) -> bool:
        """Validate a node against schema"""
        if not self.current_schema:
            return True  # No schema, allow everything
        
        if node_type not in self.current_schema.node_types:
            logger.warning(f"Unknown node type: {node_type}")
            return False
        
        node_def = self.current_schema.node_types[node_type]
        
        # Check required properties
        for prop, prop_type in node_def.properties.items():
            if prop not in properties and "optional" not in prop_type:
                logger.warning(f"Missing required property {prop} for {node_type}")
                return False
        
        return True
    
    async def validate_relationship(self, rel_type: str, from_node: str, to_node: str) -> bool:
        """Validate a relationship against schema"""
        if not self.current_schema:
            return True  # No schema, allow everything
        
        if rel_type not in self.current_schema.relationship_types:
            logger.warning(f"Unknown relationship type: {rel_type}")
            return False
        
        rel_def = self.current_schema.relationship_types[rel_type]
        return rel_def.validate(from_node, to_node)
    
    def get_node_types(self) -> List[str]:
        """Get list of defined node types"""
        if not self.current_schema:
            return []
        return list(self.current_schema.node_types.keys())
    
    def get_relationship_types(self) -> List[str]:
        """Get list of defined relationship types"""
        if not self.current_schema:
            return []
        return list(self.current_schema.relationship_types.keys())
    
    def get_collection_names(self) -> List[str]:
        """Get list of collection names"""
        if not self.current_schema:
            return []
        return [col.name for col in self.current_schema.collections.values()]