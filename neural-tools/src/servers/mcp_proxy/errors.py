#!/usr/bin/env python3
"""
Error Handling for MCP HTTP Proxy
Maps HTTP errors to MCP-compliant error responses following 2025-06-18 protocol
"""

import logging
import json
from typing import Any, Coroutine, Dict, Optional
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class MCPProxyError(Exception):
    """Base exception for MCP proxy errors"""
    
    def __init__(self, code: int, message: str, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"MCP Error {code}: {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to MCP-compliant error response format"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "data": self.details
            },
            "timestamp": datetime.now().isoformat()
        }


class MCPErrorCodes:
    """MCP error codes following protocol specifications"""
    
    # Network and connectivity errors (1xxx)
    NETWORK_ERROR = 1000
    CONNECTION_ERROR = 1001  
    TIMEOUT_ERROR = 1002
    
    # Client/input errors (2xxx)
    INVALID_PARAMETERS = 2001
    MISSING_REQUIRED_PARAMETER = 2002
    PARAMETER_OUT_OF_RANGE = 2003
    RESOURCE_NOT_FOUND = 2004
    
    # Authentication/authorization errors (3xxx)
    AUTHENTICATION_ERROR = 3001
    AUTHORIZATION_ERROR = 3002
    RATE_LIMITED = 3003
    
    # Server/internal errors (4xxx)
    INTERNAL_TOOL_ERROR = 4001
    SERVICE_UNAVAILABLE = 4002
    UPSTREAM_ERROR = 4003
    CONFIGURATION_ERROR = 4004


async def handle_api_errors(api_call_coroutine: Coroutine[Any, Any, httpx.Response]) -> Dict[str, Any]:
    """
    Robust error handling wrapper for HTTP API calls to FastAPI backend
    
    Maps HTTP errors to MCP-compliant error responses according to:
    - Network errors -> 1xxx codes
    - Client errors (4xx) -> 2xxx codes  
    - Server errors (5xx) -> 4xxx codes
    
    Args:
        api_call_coroutine: The httpx.AsyncClient.request() coroutine
        
    Returns:
        Dict containing either success response or structured error
        
    Raises:
        MCPProxyError: For errors that should be propagated to MCP client
    """
    try:
        response = await api_call_coroutine
        response.raise_for_status()  # This raises HTTPStatusError for 4xx/5xx
        
        # Success case - return the JSON response
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from API: {e}")
            raise MCPProxyError(
                code=MCPErrorCodes.UPSTREAM_ERROR,
                message="Invalid response format from backend service",
                details={"response_text": response.text[:500]}
            )
    
    except httpx.TimeoutException as e:
        logger.error(f"HTTP timeout calling backend API: {e}")
        raise MCPProxyError(
            code=MCPErrorCodes.TIMEOUT_ERROR,
            message=f"Backend service request timed out: {e.request.url}",
            details={"timeout_seconds": str(e.request.timeout)}
        )
    
    except httpx.ConnectError as e:
        logger.error(f"Connection error to backend API: {e}")
        raise MCPProxyError(
            code=MCPErrorCodes.CONNECTION_ERROR,
            message=f"Could not connect to backend service: {e.request.url}",
            details={"connection_error": str(e)}
        )
    
    except httpx.RequestError as e:
        logger.error(f"Network error calling backend API: {e}")
        raise MCPProxyError(
            code=MCPErrorCodes.NETWORK_ERROR,
            message=f"Network error calling backend service: {str(e)}",
            details={"request_url": str(e.request.url)}
        )
    
    except httpx.HTTPStatusError as e:
        # Handle HTTP status errors from FastAPI backend
        status = e.response.status_code
        
        # Try to extract error details from response
        error_details = {}
        try:
            error_body = e.response.json()
            if isinstance(error_body, dict):
                error_details = error_body
        except (json.JSONDecodeError, ValueError):
            error_details = {"response_text": e.response.text[:500]}
        
        if 400 <= status < 500:
            # Client-side errors - usually bad input from the model
            logger.warning(f"Client error {status} from API: {error_details}")
            
            if status == 400:
                raise MCPProxyError(
                    code=MCPErrorCodes.INVALID_PARAMETERS,
                    message="Invalid parameters provided to tool",
                    details=error_details
                )
            elif status == 404:
                raise MCPProxyError(
                    code=MCPErrorCodes.RESOURCE_NOT_FOUND,
                    message="Requested resource not found",
                    details=error_details
                )
            elif status == 401:
                raise MCPProxyError(
                    code=MCPErrorCodes.AUTHENTICATION_ERROR,
                    message="Authentication failed with backend service",
                    details=error_details
                )
            elif status == 403:
                raise MCPProxyError(
                    code=MCPErrorCodes.AUTHORIZATION_ERROR,
                    message="Authorization failed for requested operation",
                    details=error_details
                )
            elif status == 422:
                # Unprocessable Entity (Pydantic validation errors)
                raise MCPProxyError(
                    code=MCPErrorCodes.INVALID_PARAMETERS,
                    message="Parameter validation failed",
                    details=error_details
                )
            elif status == 429:
                raise MCPProxyError(
                    code=MCPErrorCodes.RATE_LIMITED,
                    message="Rate limit exceeded for backend service",
                    details=error_details
                )
            else:
                raise MCPProxyError(
                    code=MCPErrorCodes.INVALID_PARAMETERS,
                    message=f"Client error {status}",
                    details=error_details
                )
        
        elif 500 <= status < 600:
            # Server-side errors - backend service issues
            logger.error(f"Server error {status} from API: {error_details}")
            
            if status == 503:
                raise MCPProxyError(
                    code=MCPErrorCodes.SERVICE_UNAVAILABLE,
                    message="Backend service is temporarily unavailable",
                    details=error_details
                )
            else:
                raise MCPProxyError(
                    code=MCPErrorCodes.INTERNAL_TOOL_ERROR,
                    message="Backend service encountered an internal error",
                    details={
                        "status_code": status,
                        "error_details": error_details
                    }
                )
        
        else:
            # Unexpected status code
            logger.error(f"Unexpected HTTP status {status} from API: {error_details}")
            raise MCPProxyError(
                code=MCPErrorCodes.UPSTREAM_ERROR,
                message=f"Unexpected response status: {status}",
                details=error_details
            )


def create_success_response(data: Any) -> Dict[str, Any]:
    """
    Create a standardized success response for MCP tools
    
    Args:
        data: The tool execution result data
        
    Returns:
        Formatted success response
    """
    return {
        "status": "success",
        "data": data,
        "timestamp": datetime.now().isoformat()
    }


def create_error_response(error: Exception) -> Dict[str, Any]:
    """
    Create a standardized error response for MCP tools
    
    Args:
        error: The exception that occurred
        
    Returns:
        Formatted error response
    """
    if isinstance(error, MCPProxyError):
        return error.to_dict()
    
    # For unexpected errors, create a generic internal error
    logger.error(f"Unexpected error in MCP proxy: {error}", exc_info=True)
    return MCPProxyError(
        code=MCPErrorCodes.INTERNAL_TOOL_ERROR,
        message="An unexpected error occurred",
        details={"error_type": type(error).__name__, "error_message": str(error)}
    ).to_dict()


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_error_handling():
        """Test error handling patterns"""
        print("Testing MCP Proxy Error Handling")
        
        # Test timeout error
        async def mock_timeout():
            raise httpx.TimeoutException("Request timed out")
        
        try:
            await handle_api_errors(mock_timeout())
        except MCPProxyError as e:
            print(f"Timeout error: {e.to_dict()}")
        
        # Test HTTP 400 error
        async def mock_400_error():
            response = httpx.Response(400)
            response._content = b'{"detail": "Invalid query parameter"}'
            raise httpx.HTTPStatusError("Bad request", request=None, response=response)
        
        try:
            await handle_api_errors(mock_400_error())
        except MCPProxyError as e:
            print(f"400 error: {e.to_dict()}")
        
        print("Error handling test complete")
    
    # Run the test
    asyncio.run(test_error_handling())