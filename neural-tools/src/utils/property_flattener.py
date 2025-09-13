#!/usr/bin/env python3
"""
Property Flattening Utility for Neo4j Compatibility (ADR-0036)

This module provides utilities to flatten complex objects into Neo4j-compatible 
primitive properties, ensuring all metadata and relationship data can be stored
without TypeError exceptions.

Author: L9 Engineering Team
Date: 2025-09-12
"""

from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


def flatten_complex_object(obj: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten complex nested objects into Neo4j-compatible primitive properties
    
    This function transforms nested dictionaries and complex objects into flat
    key-value pairs using only primitive types that Neo4j accepts:
    - String, Long (int), Double (float), Boolean
    - Arrays of the above primitive types
    
    Args:
        obj: Complex object to flatten (dict, list, or primitive)
        prefix: Property name prefix for nested keys
        
    Returns:
        Dictionary with only primitive types (String, Long, Double, Boolean, [primitives])
        
    Example:
        >>> complex_import = {
        ...     "statement": "from pathlib import Path",
        ...     "line": 11,
        ...     "metadata": {
        ...         "module": "pathlib", 
        ...         "items": ["Path"],
        ...         "is_stdlib": True
        ...     }
        ... }
        >>> flatten_complex_object(complex_import, "import")
        {
            "import_statement": "from pathlib import Path",    # String
            "import_line": 11,                                 # Long
            "import_metadata_module": "pathlib",               # String  
            "import_metadata_items": ["Path"],                 # [String]
            "import_metadata_is_stdlib": True                  # Boolean
        }
    """
    if not isinstance(obj, dict):
        # Handle non-dict inputs gracefully
        if isinstance(obj, (str, int, float, bool, type(None))):
            return {prefix: obj} if prefix else {"value": obj}
        elif isinstance(obj, list):
            return {prefix: _flatten_list(obj)} if prefix else {"value": _flatten_list(obj)}
        else:
            return {prefix: str(obj)} if prefix else {"value": str(obj)}
    
    flattened = {}
    
    for key, value in obj.items():
        # Create property name with prefix
        property_name = f"{prefix}_{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively flatten nested objects
            nested = flatten_complex_object(value, property_name)
            flattened.update(nested)
        elif isinstance(value, list):
            # Handle arrays - ensure all elements are primitives
            flattened[property_name] = _flatten_list(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            # Direct primitive assignment
            if value is not None:
                flattened[property_name] = value
        else:
            # Convert complex types to string representation
            flattened[property_name] = str(value)
    
    return flattened


def _flatten_list(lst: List[Any]) -> List[Union[str, int, float, bool]]:
    """
    Flatten a list to contain only primitive types
    
    Args:
        lst: List that may contain complex objects
        
    Returns:
        List containing only primitive types (str, int, float, bool)
    """
    flattened_list = []
    
    for item in lst:
        if isinstance(item, (str, int, float, bool)):
            flattened_list.append(item)
        elif isinstance(item, dict):
            # Convert nested dicts to string representation
            flattened_list.append(str(item))
        elif isinstance(item, list):
            # Convert nested lists to string representation
            flattened_list.append(str(item))
        else:
            # Convert other complex types to string
            flattened_list.append(str(item))
    
    return flattened_list


def extract_import_primitives(import_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract primitive properties from complex import objects
    
    Specifically handles import objects from code_parser.py with the structure:
    {'statement': 'from pathlib import Path', 'line': 11, 'module': 'pathlib'}
    
    Args:
        import_obj: Complex import object from code parser
        
    Returns:
        Dictionary with flattened primitive properties for Neo4j storage
        
    Example:
        >>> import_data = {'statement': 'from pathlib import Path', 'line': 11}
        >>> extract_import_primitives(import_data)
        {
            'import_statement': 'from pathlib import Path',
            'import_line': 11,
            'import_module': 'pathlib'  # Extracted from statement
        }
    """
    if not isinstance(import_obj, dict):
        return {"import_raw": str(import_obj)}
    
    primitives = {}
    
    # Extract statement (full import line)
    statement = import_obj.get('statement', '')
    if statement:
        primitives['import_statement'] = str(statement)
        
        # Try to extract module name from statement
        module_name = _extract_module_from_statement(statement)
        if module_name:
            primitives['import_module'] = module_name
    
    # Extract line number
    line = import_obj.get('line')
    if line is not None:
        primitives['import_line'] = int(line)
    
    # Extract explicit module if provided
    module = import_obj.get('module')
    if module:
        primitives['import_module'] = str(module)
    
    # Handle any additional fields as flattened properties
    for key, value in import_obj.items():
        if key not in ['statement', 'line', 'module']:
            if isinstance(value, (str, int, float, bool)):
                primitives[f'import_{key}'] = value
            else:
                primitives[f'import_{key}'] = str(value)
    
    return primitives


def _extract_module_from_statement(statement: str) -> str:
    """
    Extract the main module name from an import statement
    
    Args:
        statement: Import statement like 'from pathlib import Path' or 'import os'
        
    Returns:
        Module name like 'pathlib' or 'os'
    """
    statement = statement.strip()
    
    if statement.startswith('from '):
        # Handle 'from module import item' format
        parts = statement.split(' ')
        if len(parts) >= 2:
            module = parts[1]
            # Remove any dots for sub-modules (e.g., 'os.path' -> 'os')
            return module.split('.')[0]
    elif statement.startswith('import '):
        # Handle 'import module' format
        parts = statement.split(' ')
        if len(parts) >= 2:
            module = parts[1]
            # Handle 'import module as alias'
            if ' as ' in module:
                module = module.split(' as ')[0]
            # Remove any dots for sub-modules
            return module.split('.')[0]
    
    # Fallback: return the statement without 'import'/'from' keywords
    clean = statement.replace('from ', '').replace('import ', '')
    return clean.split(' ')[0].split('.')[0] if clean else 'unknown'


def preserve_adr_0031_metadata(file_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all ADR-0031 canonical metadata is stored as Neo4j primitives
    
    This function takes metadata from the canonical knowledge system (ADR-0031)
    and ensures all fields are properly typed as Neo4j-compatible primitives.
    
    Args:
        file_metadata: Raw metadata dictionary that may contain mixed types
        
    Returns:
        Dictionary with all metadata as primitive types with correct typing
    """
    canonical_metadata = {}
    
    # Canonical authority (Double, 0.0-1.0 range)
    if 'canonical_weight' in file_metadata:
        try:
            weight = float(file_metadata['canonical_weight'])
            canonical_metadata['canonical_weight'] = max(0.0, min(1.0, weight))
        except (ValueError, TypeError):
            canonical_metadata['canonical_weight'] = 0.5
    
    # PRISM scores (all doubles, 0.0-1.0 range)
    prism_fields = ['prism_complexity', 'prism_dependencies', 'prism_recency', 'prism_contextual']
    for field in prism_fields:
        if field in file_metadata:
            try:
                score = float(file_metadata[field])
                canonical_metadata[field] = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                canonical_metadata[field] = 0.0
    
    # Git metadata
    if 'git_last_modified' in file_metadata:
        canonical_metadata['git_last_modified'] = str(file_metadata['git_last_modified'])
    
    if 'git_change_frequency' in file_metadata:
        try:
            canonical_metadata['git_change_frequency'] = int(file_metadata['git_change_frequency'])
        except (ValueError, TypeError):
            canonical_metadata['git_change_frequency'] = 0
    
    if 'git_author_count' in file_metadata:
        try:
            canonical_metadata['git_author_count'] = int(file_metadata['git_author_count'])
        except (ValueError, TypeError):
            canonical_metadata['git_author_count'] = 1
    
    # Pattern extraction (primitives and arrays)
    if 'todo_count' in file_metadata:
        try:
            canonical_metadata['todo_count'] = int(file_metadata['todo_count'])
        except (ValueError, TypeError):
            canonical_metadata['todo_count'] = 0
    
    if 'fixme_count' in file_metadata:
        try:
            canonical_metadata['fixme_count'] = int(file_metadata['fixme_count'])
        except (ValueError, TypeError):
            canonical_metadata['fixme_count'] = 0
    
    if 'has_type_hints' in file_metadata:
        canonical_metadata['has_type_hints'] = bool(file_metadata['has_type_hints'])
    
    if 'has_async' in file_metadata:
        canonical_metadata['has_async'] = bool(file_metadata['has_async'])
    
    if 'component_type' in file_metadata:
        canonical_metadata['component_type'] = str(file_metadata['component_type'])
    
    if 'status' in file_metadata:
        canonical_metadata['status'] = str(file_metadata['status'])
    
    # Arrays of strings
    array_fields = ['security_patterns', 'dependencies', 'authority_markers']
    for field in array_fields:
        if field in file_metadata:
            raw_value = file_metadata[field]
            if isinstance(raw_value, list):
                canonical_metadata[field] = [str(item) for item in raw_value]
            elif isinstance(raw_value, str):
                # Handle comma-separated strings
                canonical_metadata[field] = [item.strip() for item in raw_value.split(',') if item.strip()]
            else:
                canonical_metadata[field] = [str(raw_value)] if raw_value else []
    
    return canonical_metadata


def validate_primitive_properties(properties: Dict[str, Any]) -> bool:
    """
    Validate that all properties are Neo4j-compatible primitives
    
    Args:
        properties: Dictionary to validate
        
    Returns:
        True if all properties are valid Neo4j primitives, False otherwise
    """
    for key, value in properties.items():
        if not _is_primitive_type(value):
            logger.warning(f"Non-primitive property detected: {key} = {type(value)} {value}")
            return False
    return True


def _is_primitive_type(value: Any) -> bool:
    """
    Check if a value is a Neo4j-compatible primitive type
    
    Neo4j accepts: String, Long, Double, Boolean, and arrays of these types
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    elif isinstance(value, list):
        return all(_is_primitive_type(item) for item in value)
    else:
        return False