#!/usr/bin/env python3
"""
JSON Schema definitions for MCP tools
Defines comprehensive validation schemas for all neural tools
"""

from typing import Dict, Any

# Common reusable schema components
COMMON_COMPONENTS = {
    "file_path": {
        "type": "string",
        "minLength": 1,
        "maxLength": 1000,
        "pattern": r"^[^<>:\"|?*]+$",  # Basic file path validation
        "description": "Valid file path"
    },
    "project_name": {
        "type": "string",
        "pattern": r"^[a-zA-Z0-9_-]+$",
        "minLength": 1,
        "maxLength": 50,
        "description": "Project identifier (alphanumeric, underscore, hyphen)"
    },
    "search_limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "default": 10,
        "description": "Maximum number of results to return"
    }
}

# GraphRAG and Neural Search Schemas
NEURAL_SEARCH_SCHEMA = {
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000,
            "description": "Natural language search query or code snippet"
        },
        "limit": COMMON_COMPONENTS["search_limit"],
        "include_graph_context": {
            "type": "boolean",
            "default": True,
            "description": "Whether to include graph relationships in results"
        },
        "max_hops": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "default": 2,
            "description": "Maximum graph traversal depth"
        }
    },
    "additionalProperties": False
}

SEMANTIC_CODE_SEARCH_SCHEMA = {
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
            "description": "Natural language search query for semantic code search"
        },
        "limit": COMMON_COMPONENTS["search_limit"]
    },
    "additionalProperties": False
}

# Neo4j Graph Query Schema
NEO4J_GRAPH_QUERY_SCHEMA = {
    "type": "object",
    "required": ["cypher_query"],
    "properties": {
        "cypher_query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 5000,
            "description": "Cypher query to execute against the graph database"
        },
        "parameters": {
            "type": "string",
            "default": "{}",
            "description": "JSON string of parameters for the Cypher query",
            "format": "json"
        }
    },
    "additionalProperties": False
}

# File Indexing Schemas
PROJECT_INDEXER_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            **COMMON_COMPONENTS["file_path"],
            "default": "/app/project",
            "description": "Path to index (relative to /app/project or absolute)"
        },
        "clear_existing": {
            "type": "boolean",
            "default": False,
            "description": "Clear existing index data before indexing"
        },
        "force_reindex": {
            "type": "boolean",
            "default": False,
            "description": "Force re-indexing even if files haven't changed"
        },
        "recursive": {
            "type": "boolean",
            "default": True,
            "description": "Index recursively through subdirectories"
        },
        "file_patterns": {
            "type": "array",
            "items": {
                "type": "string",
                "pattern": r"^\.[a-zA-Z0-9]+$"  # File extensions like .py, .js
            },
            "uniqueItems": True,
            "description": "File extensions to index (e.g., .py, .js, .md)"
        }
    },
    "additionalProperties": False
}

# GraphRAG Analysis Schemas
GRAPHRAG_IMPACT_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": ["file_path"],
    "properties": {
        "file_path": {
            **COMMON_COMPONENTS["file_path"],
            "description": "Path to the file being changed"
        },
        "change_type": {
            "type": "string",
            "enum": ["modify", "delete", "refactor"],
            "default": "modify",
            "description": "Type of change being made to the file"
        }
    },
    "additionalProperties": False
}

GRAPHRAG_FIND_DEPENDENCIES_SCHEMA = {
    "type": "object",
    "required": ["file_path"],
    "properties": {
        "file_path": {
            **COMMON_COMPONENTS["file_path"],
            "description": "File to analyze for dependencies"
        },
        "direction": {
            "type": "string",
            "enum": ["imports", "imported_by", "both"],
            "default": "both",
            "description": "Direction of dependency analysis"
        }
    },
    "additionalProperties": False
}

GRAPHRAG_FIND_RELATED_SCHEMA = {
    "type": "object",
    "required": ["file_path"],
    "properties": {
        "file_path": {
            **COMMON_COMPONENTS["file_path"],
            "description": "Starting file path for relationship traversal"
        },
        "max_depth": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "default": 3,
            "description": "Maximum traversal depth in the graph"
        },
        "relationship_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["IMPORTS", "CALLS", "PART_OF", "EXTENDS", "IMPLEMENTS", "REFERENCES"]
            },
            "default": ["IMPORTS", "CALLS", "PART_OF"],
            "uniqueItems": True,
            "description": "Types of relationships to follow in traversal"
        }
    },
    "additionalProperties": False
}

# Project Understanding Schema
PROJECT_UNDERSTANDING_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {
            "type": "string",
            "enum": ["full", "architecture", "dependencies", "core_logic"],
            "default": "full",
            "description": "Scope of analysis to perform"
        }
    },
    "additionalProperties": False
}

# Collection Management Schemas
COLLECTION_STATUS_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False
}

RECREATE_COLLECTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "confirm": {
            "type": "boolean",
            "default": False,
            "description": "Confirmation flag - must be true to proceed with destructive operation"
        }
    },
    "additionalProperties": False
}

VERIFY_INDEXING_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False
}

# Schema registry mapping tool names to their schemas
TOOL_SCHEMAS = {
    # GraphRAG and search tools
    "graphrag_hybrid_search": NEURAL_SEARCH_SCHEMA,
    "semantic_code_search": SEMANTIC_CODE_SEARCH_SCHEMA,
    "neo4j_graph_query": NEO4J_GRAPH_QUERY_SCHEMA,
    
    # Indexing tools
    "project_indexer": PROJECT_INDEXER_SCHEMA,
    
    # Analysis tools
    "graphrag_impact_analysis": GRAPHRAG_IMPACT_ANALYSIS_SCHEMA,
    "graphrag_find_dependencies": GRAPHRAG_FIND_DEPENDENCIES_SCHEMA,
    "graphrag_find_related": GRAPHRAG_FIND_RELATED_SCHEMA,
    "project_understanding": PROJECT_UNDERSTANDING_SCHEMA,
    
    # Collection management tools
    "collection_status": COLLECTION_STATUS_SCHEMA,
    "recreate_collections": RECREATE_COLLECTIONS_SCHEMA,
    "verify_indexing": VERIFY_INDEXING_SCHEMA,
}

# Error message templates for common validation failures
ERROR_TEMPLATES = {
    "required_parameter": "Parameter '{parameter}' is required for {tool_name}",
    "invalid_type": "Parameter '{parameter}' must be of type {expected_type}, got {actual_type}",
    "out_of_range": "Parameter '{parameter}' must be between {min_value} and {max_value}",
    "invalid_enum": "Parameter '{parameter}' must be one of: {allowed_values}",
    "too_long": "Parameter '{parameter}' exceeds maximum length of {max_length} characters",
    "too_short": "Parameter '{parameter}' must be at least {min_length} characters",
    "invalid_format": "Parameter '{parameter}' must be a valid {format_type}",
    "security_violation": "Parameter '{parameter}' contains potentially unsafe content",
    "invalid_pattern": "Parameter '{parameter}' does not match the required format",
    "empty_value": "Parameter '{parameter}' cannot be empty"
}

def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """
    Get the validation schema for a specific tool
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        JSON schema dictionary for the tool
        
    Raises:
        KeyError: If tool schema is not found
    """
    if tool_name not in TOOL_SCHEMAS:
        raise KeyError(f"No schema defined for tool: {tool_name}")
    
    return TOOL_SCHEMAS[tool_name]

def get_all_tool_schemas() -> Dict[str, Dict[str, Any]]:
    """Get all available tool schemas"""
    return TOOL_SCHEMAS.copy()

def validate_schema_exists(tool_name: str) -> bool:
    """Check if a schema exists for the given tool"""
    return tool_name in TOOL_SCHEMAS

def get_error_template(template_key: str) -> str:
    """Get an error message template"""
    return ERROR_TEMPLATES.get(template_key, "Validation error occurred")

# Example usage and schema testing utilities
def generate_example_parameters(tool_name: str) -> Dict[str, Any]:
    """
    Generate example parameters for a tool based on its schema
    Useful for documentation and testing
    """
    if tool_name not in TOOL_SCHEMAS:
        return {}
    
    schema = TOOL_SCHEMAS[tool_name]
    properties = schema.get('properties', {})
    required = schema.get('required', [])
    
    examples = {}
    
    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get('type', 'string')
        default_value = prop_schema.get('default')
        
        if default_value is not None:
            examples[prop_name] = default_value
        elif prop_name in required:
            # Generate required example values
            if prop_type == 'string':
                enum_values = prop_schema.get('enum')
                if enum_values:
                    examples[prop_name] = enum_values[0]
                elif prop_schema.get('format') == 'email':
                    examples[prop_name] = "user@example.com"
                elif 'path' in prop_name.lower():
                    examples[prop_name] = "/path/to/file"
                elif 'query' in prop_name.lower():
                    examples[prop_name] = "example search query"
                else:
                    examples[prop_name] = f"example_{prop_name}"
            elif prop_type == 'integer':
                minimum = prop_schema.get('minimum', 1)
                examples[prop_name] = minimum
            elif prop_type == 'number':
                minimum = prop_schema.get('minimum', 1.0)
                examples[prop_name] = minimum
            elif prop_type == 'boolean':
                examples[prop_name] = True
            elif prop_type == 'array':
                examples[prop_name] = []
            elif prop_type == 'object':
                examples[prop_name] = {}
    
    return examples