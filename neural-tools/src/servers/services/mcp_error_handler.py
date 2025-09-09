#!/usr/bin/env python3
"""
MCP Error Handler and Validation Middleware
Provides MCP-compliant error handling and parameter validation middleware
Following MCP 2025-06-18 specification
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from functools import wraps

from .validation_service import ValidationResult, ErrorType, validate_mcp_parameters
from .schemas import get_tool_schema, validate_schema_exists

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP-related errors"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.SYSTEM, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}


class MCPValidationError(MCPError):
    """Validation-specific MCP error"""
    def __init__(self, validation_result: ValidationResult):
        super().__init__(validation_result.error_message, ErrorType.VALIDATION, validation_result.to_mcp_error_response())
        self.validation_result = validation_result


class MCPErrorHandler:
    """
    MCP-compliant error handler that formats errors according to the specification
    Provides both JSON-RPC protocol errors and tool execution errors
    """
    
    @staticmethod
    def create_protocol_error(error_code: int, message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a JSON-RPC protocol error response
        
        Args:
            error_code: JSON-RPC error code
            message: Error message
            request_id: Request ID if available
            
        Returns:
            JSON-RPC error response
        """
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": error_code,
                "message": message
            }
        }
        
        if request_id is not None:
            response["id"] = request_id
            
        return response
    
    @staticmethod
    def create_tool_error(error_message: str, error_details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a tool execution error response with isError flag
        
        Args:
            error_message: Main error message
            error_details: Additional error details
            
        Returns:
            MCP tool error response
        """
        content = [
            {
                "type": "text",
                "text": error_message
            }
        ]
        
        # Add structured error details if available
        if error_details:
            content.append({
                "type": "text",
                "text": f"Error details: {json.dumps(error_details, indent=2)}"
            })
        
        return {
            "content": content,
            "isError": True
        }
    
    @staticmethod
    def create_validation_error(validation_result: ValidationResult) -> Dict[str, Any]:
        """
        Create a user-friendly validation error response
        
        Args:
            validation_result: Validation result with detailed errors
            
        Returns:
            MCP validation error response
        """
        # Main error message
        main_message = validation_result.error_message
        
        # Detailed error breakdown
        error_details = []
        for error in validation_result.errors:
            detail = f"• {error.parameter}: {error.message}"
            if error.expected_type:
                detail += f" (expected: {error.expected_type})"
            if error.allowed_values:
                detail += f" (allowed: {', '.join(map(str, error.allowed_values))})"
            error_details.append(detail)
        
        # Suggestion for fixing
        suggestion = validation_result.suggested_fix
        
        # Combine into user-friendly message
        full_message = f"{main_message}\n\nIssues found:\n" + "\n".join(error_details)
        if suggestion:
            full_message += f"\n\nSuggested fix: {suggestion}"
        
        # Create structured error response
        error_response = validation_result.to_mcp_error_response()
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": full_message
                },
                {
                    "type": "text", 
                    "text": f"Structured validation details: {json.dumps(error_response, indent=2)}"
                }
            ],
            "isError": True
        }
    
    @staticmethod
    def handle_exception(e: Exception, tool_name: str = "unknown") -> Dict[str, Any]:
        """
        Handle any exception and convert to appropriate MCP error format
        
        Args:
            e: Exception to handle
            tool_name: Name of the tool where error occurred
            
        Returns:
            MCP error response
        """
        if isinstance(e, MCPValidationError):
            return MCPErrorHandler.create_validation_error(e.validation_result)
        elif isinstance(e, MCPError):
            return MCPErrorHandler.create_tool_error(e.message, e.details)
        else:
            # Generic exception handling
            error_message = f"Error in {tool_name}: {str(e)}"
            logger.error(f"Unhandled exception in {tool_name}: {e}", exc_info=True)
            
            return MCPErrorHandler.create_tool_error(
                error_message,
                {
                    "tool": tool_name,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e)
                }
            )


class MCPValidationMiddleware:
    """
    Middleware for automatic parameter validation in MCP tools
    Validates parameters against predefined schemas before tool execution
    """
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize validation middleware
        
        Args:
            strict_validation: Whether to enforce strict validation (fail on unknown parameters)
        """
        self.strict_validation = strict_validation
    
    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[ValidationResult]:
        """
        Validate parameters for a specific tool
        
        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate
            
        Returns:
            ValidationResult if validation fails, None if successful
        """
        # Check if schema exists for this tool
        if not validate_schema_exists(tool_name):
            if self.strict_validation:
                logger.warning(f"No validation schema found for tool: {tool_name}")
            return None
        
        try:
            # Get schema and validate
            schema = get_tool_schema(tool_name)
            result = validate_mcp_parameters(parameters, schema)
            
            if not result.success:
                logger.info(f"Parameter validation failed for {tool_name}: {result.error_message}")
                return result
                
        except Exception as e:
            logger.error(f"Error during parameter validation for {tool_name}: {e}")
            if self.strict_validation:
                # Create a validation error for schema issues
                from .validation_service import ValidationError, ValidationResult
                error = ValidationError(
                    parameter="schema",
                    issue="validation_error",
                    message=f"Schema validation error: {str(e)}"
                )
                return ValidationResult(
                    success=False,
                    errors=[error],
                    error_type=ErrorType.SYSTEM,
                    error_message="Parameter validation system error"
                )
        
        return None  # Validation successful
    
    def validation_decorator(self, tool_name: str):
        """
        Decorator for automatic parameter validation
        
        Args:
            tool_name: Name of the tool being decorated
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract parameters from arguments
                # Assume first argument after self is parameters dict
                if len(args) >= 2 and isinstance(args[1], dict):
                    parameters = args[1]
                elif 'parameters' in kwargs:
                    parameters = kwargs['parameters']
                else:
                    # No parameters to validate
                    return await func(*args, **kwargs)
                
                # Validate parameters
                validation_result = self.validate_tool_parameters(tool_name, parameters)
                if validation_result:
                    # Return validation error response
                    return MCPErrorHandler.create_validation_error(validation_result)
                
                # Proceed with function execution
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Handle any exceptions during execution
                    return MCPErrorHandler.handle_exception(e, tool_name)
            
            return wrapper
        return decorator


# Global middleware instance
validation_middleware = MCPValidationMiddleware(strict_validation=True)

# Convenience decorators for common tools
def validate_graphrag_search(func):
    """Decorator for GraphRAG search tools"""
    return validation_middleware.validation_decorator("graphrag_hybrid_search")(func)

def validate_semantic_search(func):
    """Decorator for semantic search tools"""
    return validation_middleware.validation_decorator("semantic_code_search")(func)

def validate_neo4j_query(func):
    """Decorator for Neo4j query tools"""
    return validation_middleware.validation_decorator("neo4j_graph_query")(func)

def validate_project_indexer(func):
    """Decorator for project indexer tools"""
    return validation_middleware.validation_decorator("project_indexer")(func)

def validate_impact_analysis(func):
    """Decorator for impact analysis tools"""
    return validation_middleware.validation_decorator("graphrag_impact_analysis")(func)

def validate_dependencies(func):
    """Decorator for dependency analysis tools"""
    return validation_middleware.validation_decorator("graphrag_find_dependencies")(func)

def validate_related_files(func):
    """Decorator for related files analysis"""
    return validation_middleware.validation_decorator("graphrag_find_related")(func)

def validate_project_understanding(func):
    """Decorator for project understanding tools"""
    return validation_middleware.validation_decorator("project_understanding")(func)


# Error code constants following JSON-RPC specification
class MCPErrorCodes:
    """Standard JSON-RPC error codes for MCP"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom MCP error codes (following JSON-RPC spec for application errors)
    TOOL_NOT_FOUND = -32001
    TOOL_EXECUTION_ERROR = -32002
    PARAMETER_VALIDATION_ERROR = -32003
    AUTHENTICATION_ERROR = -32004
    PERMISSION_DENIED = -32005
    SERVICE_UNAVAILABLE = -32006


# Utility functions for quick error responses
def create_missing_parameter_error(parameter_name: str, tool_name: str) -> Dict[str, Any]:
    """Quick helper for missing parameter errors"""
    message = f"Missing required parameter '{parameter_name}' for tool '{tool_name}'"
    return MCPErrorHandler.create_tool_error(message)

def create_invalid_parameter_error(parameter_name: str, expected_type: str, actual_type: str) -> Dict[str, Any]:
    """Quick helper for invalid parameter type errors"""
    message = f"Parameter '{parameter_name}' must be of type {expected_type}, got {actual_type}"
    return MCPErrorHandler.create_tool_error(message)

def create_service_unavailable_error(service_name: str) -> Dict[str, Any]:
    """Quick helper for service unavailable errors"""
    message = f"Service '{service_name}' is currently unavailable. Please try again later."
    return MCPErrorHandler.create_tool_error(message)

def create_authentication_error(message: str = "Authentication required") -> Dict[str, Any]:
    """Quick helper for authentication errors"""
    return MCPErrorHandler.create_tool_error(message)