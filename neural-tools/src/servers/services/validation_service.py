#!/usr/bin/env python3
"""
MCP-Compliant Parameter Validation Service
Implements comprehensive parameter validation with user-friendly error messages
Following MCP 2025-06-18 specification for error handling and parameter validation
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Error categories for consistent error reporting"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    CONNECTIVITY = "connectivity"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    SECURITY = "security"


@dataclass
class ValidationError:
    """Structured validation error with user-friendly details"""
    parameter: str
    issue: str
    message: str
    expected_type: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    provided_value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of parameter validation with detailed error information"""
    success: bool
    errors: List[ValidationError]
    error_type: ErrorType = ErrorType.VALIDATION
    error_message: str = ""
    suggested_fix: str = ""

    def to_mcp_error_response(self) -> Dict[str, Any]:
        """Convert to MCP-compliant error response format"""
        if self.success:
            return {"success": True}
        
        # Format validation errors for detailed feedback
        validation_details = []
        for error in self.errors:
            detail = {
                "parameter": error.parameter,
                "issue": error.issue,
                "message": error.message
            }
            if error.expected_type:
                detail["expectedType"] = error.expected_type
            if error.allowed_values:
                detail["allowedValues"] = error.allowed_values
            if error.provided_value is not None:
                detail["providedValue"] = error.provided_value
            validation_details.append(detail)

        return {
            "success": False,
            "errorType": self.error_type.value,
            "errorMessage": self.error_message,
            "validationErrors": validation_details,
            "suggestedFix": self.suggested_fix
        }


class ParameterValidator:
    """
    MCP-compliant parameter validator with comprehensive validation rules
    Supports JSON Schema-like validation with user-friendly error messages
    """
    
    def __init__(self):
        self.security_patterns = {
            'sql_injection': [
                r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b',
                r'[\'";]',
                r'--',
                r'/\*.*\*/',
                r'\bOR\s+\d+\s*=\s*\d+',
                r'\bAND\s+\d+\s*=\s*\d+'
            ],
            'xss': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe',
                r'<object',
                r'<embed'
            ],
            'path_traversal': [
                r'\.\./+',
                r'\.\.\\+',
                r'/etc/',
                r'\\windows\\',
                r'/proc/',
                r'/sys/'
            ]
        }

    def validate_parameters(self, parameters: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters against a JSON schema with comprehensive error reporting
        
        Args:
            parameters: Input parameters to validate
            schema: JSON schema defining validation rules
            
        Returns:
            ValidationResult with detailed error information
        """
        errors = []
        
        # Check required parameters
        required = schema.get('required', [])
        for param in required:
            if param not in parameters:
                errors.append(ValidationError(
                    parameter=param,
                    issue="required",
                    message=f"Parameter '{param}' is required",
                    expected_type=self._get_parameter_type(param, schema),
                    allowed_values=self._get_allowed_values(param, schema)
                ))

        # Validate each provided parameter
        properties = schema.get('properties', {})
        for param_name, param_value in parameters.items():
            if param_name in properties:
                param_schema = properties[param_name]
                param_errors = self._validate_single_parameter(param_name, param_value, param_schema)
                errors.extend(param_errors)
            else:
                # Unknown parameter - could be a warning or error based on additionalProperties
                if not schema.get('additionalProperties', True):
                    errors.append(ValidationError(
                        parameter=param_name,
                        issue="unknown",
                        message=f"Parameter '{param_name}' is not allowed",
                        provided_value=param_value
                    ))

        # Generate summary and suggestions
        if errors:
            error_message = self._generate_error_summary(errors)
            suggested_fix = self._generate_suggested_fix(errors, schema)
            
            return ValidationResult(
                success=False,
                errors=errors,
                error_type=ErrorType.VALIDATION,
                error_message=error_message,
                suggested_fix=suggested_fix
            )
        
        return ValidationResult(success=True, errors=[])

    def _validate_single_parameter(self, name: str, value: Any, schema: Dict[str, Any]) -> List[ValidationError]:
        """Validate a single parameter against its schema"""
        errors = []
        
        # Type validation
        expected_type = schema.get('type')
        if expected_type and not self._check_type(value, expected_type):
            errors.append(ValidationError(
                parameter=name,
                issue="type",
                message=f"Parameter '{name}' must be of type {expected_type}",
                expected_type=expected_type,
                provided_value=type(value).__name__
            ))
            return errors  # Skip further validation if type is wrong

        # String-specific validations
        if expected_type == 'string' and isinstance(value, str):
            errors.extend(self._validate_string(name, value, schema))
        
        # Number-specific validations
        elif expected_type in ['number', 'integer'] and isinstance(value, (int, float)):
            errors.extend(self._validate_number(name, value, schema))
        
        # Array-specific validations
        elif expected_type == 'array' and isinstance(value, list):
            errors.extend(self._validate_array(name, value, schema))
        
        # Object-specific validations (recursive)
        elif expected_type == 'object' and isinstance(value, dict):
            errors.extend(self._validate_object(name, value, schema))

        # Enum validation
        allowed_values = schema.get('enum')
        if allowed_values and value not in allowed_values:
            errors.append(ValidationError(
                parameter=name,
                issue="enum",
                message=f"Parameter '{name}' must be one of: {', '.join(map(str, allowed_values))}",
                allowed_values=allowed_values,
                provided_value=value
            ))

        # Security validation
        if isinstance(value, str):
            security_error = self._validate_security(name, value)
            if security_error:
                errors.append(security_error)

        return errors

    def _validate_string(self, name: str, value: str, schema: Dict[str, Any]) -> List[ValidationError]:
        """Validate string-specific constraints"""
        errors = []
        
        # Length constraints
        min_length = schema.get('minLength')
        if min_length is not None and len(value) < min_length:
            errors.append(ValidationError(
                parameter=name,
                issue="minLength",
                message=f"Parameter '{name}' must be at least {min_length} characters long",
                provided_value=len(value)
            ))
        
        max_length = schema.get('maxLength')
        if max_length is not None and len(value) > max_length:
            errors.append(ValidationError(
                parameter=name,
                issue="maxLength",
                message=f"Parameter '{name}' must be no more than {max_length} characters long",
                provided_value=len(value)
            ))
        
        # Empty string validation
        if not value.strip() and not schema.get('allowEmpty', False):
            errors.append(ValidationError(
                parameter=name,
                issue="empty",
                message=f"Parameter '{name}' cannot be empty",
                provided_value="(empty string)"
            ))
        
        # Format validation
        format_type = schema.get('format')
        if format_type:
            format_error = self._validate_format(name, value, format_type)
            if format_error:
                errors.append(format_error)
        
        # Pattern validation
        pattern = schema.get('pattern')
        if pattern and not re.match(pattern, value):
            errors.append(ValidationError(
                parameter=name,
                issue="pattern",
                message=f"Parameter '{name}' does not match the required pattern",
                provided_value=value
            ))
        
        return errors

    def _validate_number(self, name: str, value: Union[int, float], schema: Dict[str, Any]) -> List[ValidationError]:
        """Validate number-specific constraints"""
        errors = []
        
        # Range constraints
        minimum = schema.get('minimum')
        if minimum is not None and value < minimum:
            errors.append(ValidationError(
                parameter=name,
                issue="minimum",
                message=f"Parameter '{name}' must be at least {minimum}",
                provided_value=value
            ))
        
        maximum = schema.get('maximum')
        if maximum is not None and value > maximum:
            errors.append(ValidationError(
                parameter=name,
                issue="maximum",
                message=f"Parameter '{name}' must be no more than {maximum}",
                provided_value=value
            ))
        
        # Exclusive bounds
        exclusive_minimum = schema.get('exclusiveMinimum')
        if exclusive_minimum is not None and value <= exclusive_minimum:
            errors.append(ValidationError(
                parameter=name,
                issue="exclusiveMinimum",
                message=f"Parameter '{name}' must be greater than {exclusive_minimum}",
                provided_value=value
            ))
        
        exclusive_maximum = schema.get('exclusiveMaximum')
        if exclusive_maximum is not None and value >= exclusive_maximum:
            errors.append(ValidationError(
                parameter=name,
                issue="exclusiveMaximum",
                message=f"Parameter '{name}' must be less than {exclusive_maximum}",
                provided_value=value
            ))
        
        return errors

    def _validate_array(self, name: str, value: List[Any], schema: Dict[str, Any]) -> List[ValidationError]:
        """Validate array-specific constraints"""
        errors = []
        
        # Length constraints
        min_items = schema.get('minItems')
        if min_items is not None and len(value) < min_items:
            errors.append(ValidationError(
                parameter=name,
                issue="minItems",
                message=f"Parameter '{name}' must contain at least {min_items} items",
                provided_value=len(value)
            ))
        
        max_items = schema.get('maxItems')
        if max_items is not None and len(value) > max_items:
            errors.append(ValidationError(
                parameter=name,
                issue="maxItems",
                message=f"Parameter '{name}' must contain no more than {max_items} items",
                provided_value=len(value)
            ))
        
        # Item validation
        items_schema = schema.get('items')
        if items_schema:
            for i, item in enumerate(value):
                item_errors = self._validate_single_parameter(f"{name}[{i}]", item, items_schema)
                errors.extend(item_errors)
        
        # Unique items
        if schema.get('uniqueItems', False) and len(value) != len(set(map(str, value))):
            errors.append(ValidationError(
                parameter=name,
                issue="uniqueItems",
                message=f"Parameter '{name}' must contain unique items only",
                provided_value="(duplicate items found)"
            ))
        
        return errors

    def _validate_object(self, name: str, value: Dict[str, Any], schema: Dict[str, Any]) -> List[ValidationError]:
        """Validate object-specific constraints (recursive)"""
        errors = []
        
        # Recursive validation using the main validation method
        nested_result = self.validate_parameters(value, schema)
        if not nested_result.success:
            # Prefix parameter names with parent object name
            for error in nested_result.errors:
                error.parameter = f"{name}.{error.parameter}"
                errors.append(error)
        
        return errors

    def _validate_format(self, name: str, value: str, format_type: str) -> Optional[ValidationError]:
        """Validate string format constraints"""
        format_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'uri': r'^https?://[^\s/$.?#].[^\s]*$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'time': r'^\d{2}:\d{2}:\d{2}$',
            'datetime': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        }
        
        pattern = format_patterns.get(format_type)
        if pattern and not re.match(pattern, value, re.IGNORECASE):
            return ValidationError(
                parameter=name,
                issue="format",
                message=f"Parameter '{name}' must be a valid {format_type}",
                expected_type=format_type,
                provided_value=value
            )
        
        return None

    def _validate_security(self, name: str, value: str) -> Optional[ValidationError]:
        """Validate string for common security vulnerabilities"""
        for threat_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    return ValidationError(
                        parameter=name,
                        issue="security",
                        message=f"Parameter '{name}' contains potentially unsafe content ({threat_type})",
                        provided_value="(potentially unsafe content)"
                    )
        
        return None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type"""
        type_mapping = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid

    def _get_parameter_type(self, param_name: str, schema: Dict[str, Any]) -> Optional[str]:
        """Get the expected type for a parameter from schema"""
        properties = schema.get('properties', {})
        param_schema = properties.get(param_name, {})
        return param_schema.get('type')

    def _get_allowed_values(self, param_name: str, schema: Dict[str, Any]) -> Optional[List[Any]]:
        """Get allowed values (enum) for a parameter from schema"""
        properties = schema.get('properties', {})
        param_schema = properties.get(param_name, {})
        return param_schema.get('enum')

    def _generate_error_summary(self, errors: List[ValidationError]) -> str:
        """Generate a user-friendly error summary"""
        if len(errors) == 1:
            return errors[0].message
        
        error_types = {}
        for error in errors:
            error_types[error.issue] = error_types.get(error.issue, 0) + 1
        
        summary_parts = []
        if 'required' in error_types:
            count = error_types['required']
            summary_parts.append(f"{count} required parameter{'s' if count > 1 else ''} missing")
        
        if 'type' in error_types:
            count = error_types['type']
            summary_parts.append(f"{count} parameter{'s' if count > 1 else ''} with incorrect type")
        
        other_count = sum(count for issue, count in error_types.items() if issue not in ['required', 'type'])
        if other_count > 0:
            summary_parts.append(f"{other_count} other validation error{'s' if other_count > 1 else ''}")
        
        return f"Parameter validation failed: {', '.join(summary_parts)}"

    def _generate_suggested_fix(self, errors: List[ValidationError], schema: Dict[str, Any]) -> str:
        """Generate actionable suggestions for fixing validation errors"""
        suggestions = []
        
        # Group errors by type for better suggestions
        required_errors = [e for e in errors if e.issue == 'required']
        type_errors = [e for e in errors if e.issue == 'type']
        
        if required_errors:
            missing_params = [e.parameter for e in required_errors]
            example_params = {}
            properties = schema.get('properties', {})
            
            for param in missing_params:
                param_schema = properties.get(param, {})
                example_params[param] = self._generate_example_value(param_schema)
            
            example_json = json.dumps(example_params, indent=2)
            suggestions.append(f"Add missing required parameters: {example_json}")
        
        if type_errors:
            for error in type_errors:
                if error.expected_type:
                    suggestions.append(f"Change '{error.parameter}' to {error.expected_type} type")
        
        return "; ".join(suggestions) if suggestions else "Please check parameter format and try again"

    def _generate_example_value(self, schema: Dict[str, Any]) -> Any:
        """Generate an example value based on schema"""
        param_type = schema.get('type', 'string')
        
        if param_type == 'string':
            enum_values = schema.get('enum')
            if enum_values:
                return enum_values[0]
            format_type = schema.get('format')
            if format_type == 'email':
                return "user@example.com"
            elif format_type == 'uri':
                return "https://example.com"
            elif format_type == 'date':
                return "2024-01-01"
            return "example_value"
        elif param_type == 'integer':
            return schema.get('minimum', 1)
        elif param_type == 'number':
            return schema.get('minimum', 1.0)
        elif param_type == 'boolean':
            return True
        elif param_type == 'array':
            return []
        elif param_type == 'object':
            return {}
        
        return None


# Global validator instance
validator = ParameterValidator()


def validate_mcp_parameters(parameters: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """
    Convenience function for MCP parameter validation
    
    Args:
        parameters: Parameters to validate
        schema: JSON schema for validation
        
    Returns:
        ValidationResult with MCP-compliant error information
    """
    return validator.validate_parameters(parameters, schema)