#!/usr/bin/env python3
"""
JSON utilities for handling NumPy and other special types
Provides centralized serialization for the Neural MCP system
"""

import json
import numpy as np
from datetime import datetime, date
from pathlib import Path
from typing import Any

class NumpyJSONEncoder(json.JSONEncoder):
    """
    A JSONEncoder that handles NumPy types and other common non-serializable types.
    
    This encoder provides centralized handling for:
    - NumPy integers (int8, int16, int32, int64, etc.)
    - NumPy floats (float16, float32, float64, etc.)
    - NumPy arrays (converted to lists)
    - NumPy booleans
    - datetime and date objects
    - Path objects
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to JSON-compatible types.
        
        Args:
            obj: The object to serialize
            
        Returns:
            JSON-compatible representation of the object
        """
        # Handle NumPy integers
        if isinstance(obj, np.integer):
            return int(obj)
        
        # Handle NumPy floats
        if isinstance(obj, np.floating):
            return float(obj)
        
        # Handle NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle NumPy booleans
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle date objects
        if isinstance(obj, date):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle bytes (encode as base64 string)
        if isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode('utf-8')
        
        # Fall back to the default encoder
        return super().default(obj)


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    Safely serialize data to JSON string, handling NumPy and other special types.
    
    Args:
        data: The data to serialize
        **kwargs: Additional arguments to pass to json.dumps
        
    Returns:
        JSON string representation of the data
    """
    # Use our custom encoder by default
    if 'cls' not in kwargs:
        kwargs['cls'] = NumpyJSONEncoder
    
    return json.dumps(data, **kwargs)


def safe_json_response(data: Any, indent: int = 2) -> str:
    """
    Create a formatted JSON response suitable for API returns.
    
    Args:
        data: The data to serialize
        indent: Number of spaces for indentation (default: 2)
        
    Returns:
        Formatted JSON string
    """
    return safe_json_dumps(data, indent=indent)


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Test data with various NumPy types
    test_data = {
        "float32": np.float32(3.14159),
        "float64": np.float64(2.71828),
        "int32": np.int32(42),
        "int64": np.int64(9999999999),
        "bool": np.bool_(True),
        "array": np.array([1, 2, 3, 4, 5]),
        "matrix": np.array([[1, 2], [3, 4]]),
        "datetime": datetime.now(),
        "path": Path("/some/file/path.txt"),
        "regular_float": 1.23,
        "regular_string": "test"
    }
    
    # Test serialization
    print("Testing NumpyJSONEncoder:")
    print("-" * 50)
    
    try:
        # This would fail with standard json.dumps
        standard_result = json.dumps(test_data)
        print("Standard encoder: Success (unexpected!)")
    except TypeError as e:
        print(f"Standard encoder: Failed as expected - {e}")
    
    print()
    
    # This should work with our custom encoder
    custom_result = safe_json_dumps(test_data, indent=2)
    print("Custom encoder: Success!")
    print(custom_result[:200] + "...")
    
    print()
    print("✅ NumpyJSONEncoder is working correctly!")