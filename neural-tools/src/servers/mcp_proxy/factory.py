#!/usr/bin/env python3
"""
Generic Tool Proxy Factory for MCP HTTP Proxy
Dynamically creates and registers MCP tools from JSON definitions
Eliminates boilerplate by auto-generating proxy functions
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional
from urllib.parse import urlparse
import httpx

# Import MCP types - adjust based on actual MCP SDK structure
try:
    import mcp.types as types
    from mcp.server import Server
except ImportError:
    # Mock for development - replace with actual MCP imports
    class types:
        class TextContent:
            def __init__(self, type: str, text: str):
                self.type = type
                self.text = text

from .errors import handle_api_errors, create_success_response, create_error_response, MCPProxyError

logger = logging.getLogger(__name__)


class ToolProxyFactory:
    """
    Dynamically creates and registers MCP tools from JSON definitions
    
    This factory reads tool definitions and creates HTTP proxy functions
    that handle parameter templating, request building, and error propagation.
    """

    def __init__(
        self,
        mcp_server: Server,
        http_client: httpx.AsyncClient,
        tool_definitions: List[Dict[str, Any]]
    ):
        self.mcp_server = mcp_server
        self.http_client = http_client
        self.tool_definitions = tool_definitions
        self._registered_tools = []

    @classmethod
    def from_json_file(cls, mcp_server: Server, http_client: httpx.AsyncClient, json_path: str) -> 'ToolProxyFactory':
        """
        Create factory from JSON configuration file
        
        Args:
            mcp_server: MCP server instance
            http_client: HTTP client for backend API calls
            json_path: Path to JSON tool definitions file
            
        Returns:
            Configured ToolProxyFactory instance
        """
        try:
            with open(json_path, 'r') as f:
                config = json.load(f)
            
            # Handle both formats: {"tools": [...]} or direct [...]
            if isinstance(config, dict) and "tools" in config:
                tool_definitions = config["tools"]
            elif isinstance(config, list):
                tool_definitions = config
            else:
                raise ValueError(f"Invalid JSON structure in {json_path}")
            
            logger.info(f"Loaded {len(tool_definitions)} tool definitions from {json_path}")
            return cls(mcp_server, http_client, tool_definitions)
            
        except FileNotFoundError:
            logger.error(f"Tool definitions file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in tool definitions file {json_path}: {e}")
            raise

    def register_tools(self) -> int:
        """
        Register all tools from definitions with the MCP server
        
        Returns:
            Number of tools successfully registered
        """
        registered_count = 0
        
        for tool_def in self.tool_definitions:
            try:
                proxy_func = self._create_proxy_function(tool_def)
                
                # Register with MCP server (adjust method name based on actual SDK)
                # This is a mock - replace with actual MCP server registration
                self._register_tool_with_server(tool_def, proxy_func)
                
                self._registered_tools.append(tool_def["name"])
                registered_count += 1
                logger.info(f"✅ Registered tool: {tool_def['name']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to register tool {tool_def.get('name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Tool registration complete: {registered_count}/{len(self.tool_definitions)} tools registered")
        return registered_count

    def _register_tool_with_server(self, tool_def: Dict[str, Any], proxy_func: Callable):
        """
        Register a single tool with the MCP server
        
        Args:
            tool_def: Tool definition from JSON
            proxy_func: Generated proxy function
        """
        # This is where you'd call the actual MCP server registration method
        # The exact API depends on your MCP SDK
        
        # Example (adjust based on actual SDK):
        # self.mcp_server.add_tool(
        #     name=tool_def["name"],
        #     description=tool_def["description"],  
        #     parameters=tool_def["parameters"],
        #     func=proxy_func
        # )
        
        # For now, just store the registration info
        logger.debug(f"Would register tool {tool_def['name']} with MCP server")

    def _create_proxy_function(self, tool_def: Dict[str, Any]) -> Callable[..., Coroutine]:
        """
        Create a specific async proxy function for a single tool definition
        
        Args:
            tool_def: Tool definition containing name, parameters, and implementation
            
        Returns:
            Async function that proxies calls to HTTP backend
        """
        tool_name = tool_def["name"]
        implementation = tool_def["implementation"]
        
        async def tool_proxy(**kwargs: Any) -> List[types.TextContent]:
            """
            Generic tool proxy function - handles HTTP calls to FastAPI backend
            
            This function:
            1. Extracts HTTP method and endpoint from tool definition
            2. Builds request arguments from kwargs using template resolution
            3. Makes HTTP request to FastAPI backend
            4. Handles errors and returns MCP-compliant responses
            """
            try:
                logger.debug(f"Executing tool {tool_name} with args: {kwargs}")
                
                # Extract implementation details
                method = implementation["method"].upper()
                
                # Use URL path only - base URL comes from client config
                url_path = self._extract_url_path(implementation["url"])
                
                # Resolve template variables in URL path
                url_path = self._resolve_template(url_path, kwargs)
                
                # Build request arguments (params, json, headers, etc.)
                request_args = self._build_request_args(implementation, kwargs)
                
                logger.debug(f"Making {method} request to {url_path} with args: {request_args}")
                
                # Make the HTTP request
                api_call = self.http_client.request(
                    method=method,
                    url=url_path,
                    **request_args
                )
                
                # Handle response and errors
                response_data = await handle_api_errors(api_call)
                
                # Return MCP-compliant response
                success_response = create_success_response(response_data)
                return [types.TextContent(
                    type="text",
                    text=json.dumps(success_response, indent=2)
                )]
                
            except MCPProxyError as e:
                # Known proxy errors - return structured error response
                error_response = e.to_dict()
                logger.warning(f"Tool {tool_name} failed with proxy error: {e.message}")
                return [types.TextContent(
                    type="text", 
                    text=json.dumps(error_response, indent=2)
                )]
                
            except Exception as e:
                # Unexpected errors - log and return generic error
                logger.error(f"Unexpected error in tool {tool_name}: {e}", exc_info=True)
                error_response = create_error_response(e)
                return [types.TextContent(
                    type="text",
                    text=json.dumps(error_response, indent=2)
                )]

        # Set function metadata for debugging
        tool_proxy.__name__ = f"{tool_name}_proxy"
        tool_proxy.__doc__ = f"HTTP proxy for {tool_name}: {tool_def.get('description', '')}"
        
        return tool_proxy

    def _extract_url_path(self, url: str) -> str:
        """
        Extract path component from URL, ignoring host
        
        This allows the base_url from config to be the single source of truth
        while tool definitions can specify full URLs for documentation.
        
        Args:
            url: Full URL or path from tool definition
            
        Returns:
            URL path component
        """
        parsed = urlparse(url)
        return parsed.path or "/"

    def _build_request_args(self, implementation: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build httpx request arguments from implementation config and function kwargs
        
        Handles template resolution ({{variable}}) and different parameter passing methods:
        - query_params -> params for GET requests
        - json_body/body_template -> json for POST requests
        - headers -> headers
        
        Args:
            implementation: Implementation block from tool definition
            kwargs: Arguments passed to the tool function
            
        Returns:
            Dictionary of arguments for httpx.AsyncClient.request()
        """
        request_args = {}

        # Handle query parameters (GET, DELETE, etc.)
        if "query_params" in implementation:
            params = {}
            for key, template in implementation["query_params"].items():
                value = self._resolve_template(template, kwargs)
                # Only include non-None, non-empty values
                if value is not None and value != "":
                    params[key] = value
            if params:
                request_args["params"] = params

        # Handle JSON body (POST, PUT, PATCH)
        if "json_body" in implementation:
            json_body = self._build_json_body(implementation["json_body"], kwargs)
            if json_body:
                request_args["json"] = json_body
        
        # Handle body_template (alternative JSON format)
        elif "body_template" in implementation:
            json_body = self._build_json_body(implementation["body_template"], kwargs)
            if json_body:
                request_args["json"] = json_body

        # Handle custom headers
        if "headers" in implementation:
            headers = {}
            for key, template in implementation["headers"].items():
                value = self._resolve_template(template, kwargs)
                if value is not None:
                    headers[key] = str(value)
            if headers:
                request_args["headers"] = headers

        # Handle files/form data (if needed in future)
        if "form_data" in implementation:
            form_data = {}
            for key, template in implementation["form_data"].items():
                value = self._resolve_template(template, kwargs)
                if value is not None:
                    form_data[key] = value
            if form_data:
                request_args["data"] = form_data

        return request_args

    def _build_json_body(self, body_template: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build JSON request body from template, resolving variables
        
        Args:
            body_template: Template with {{variable}} placeholders
            kwargs: Values for template resolution
            
        Returns:
            Resolved JSON body
        """
        json_body = {}
        
        for key, template in body_template.items():
            value = self._resolve_template(template, kwargs)
            # Include None values for JSON body (API might expect them)
            json_body[key] = value
            
        return json_body

    def _resolve_template(self, template: Any, values: Dict[str, Any]) -> Any:
        """
        Resolve a single template value, handling {{variable}} substitution
        
        Args:
            template: Template string with {{variable}} or direct value
            values: Dictionary of values for substitution
            
        Returns:
            Resolved value
        """
        if not isinstance(template, str):
            return template
            
        # Check if it's a template variable: {{variable_name}}
        match = re.match(r"^\{\{(\w+)\}\}$", template.strip())
        if match:
            var_name = match.group(1)
            return values.get(var_name)
        
        # Check if it contains template variables within a string
        def replace_vars(match):
            var_name = match.group(1)
            value = values.get(var_name)
            return str(value) if value is not None else ""
        
        # Replace all {{variable}} patterns in the string
        resolved = re.sub(r'\{\{(\w+)\}\}', replace_vars, template)
        return resolved if resolved != template else template

    def get_registered_tools(self) -> List[str]:
        """Get list of successfully registered tool names"""
        return self._registered_tools.copy()

    def get_tool_count(self) -> int:
        """Get count of registered tools"""
        return len(self._registered_tools)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import tempfile
    import os
    
    async def test_factory():
        """Test the proxy factory with mock data"""
        print("Testing ToolProxyFactory")
        
        # Create mock tool definition
        test_tools = [
            {
                "name": "test_search",
                "description": "Test search tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                },
                "implementation": {
                    "method": "GET",
                    "url": "http://localhost:8000/search",
                    "query_params": {
                        "query": "{{query}}",
                        "limit": "{{limit}}"
                    }
                }
            }
        ]
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"tools": test_tools}, f)
            temp_path = f.name
        
        try:
            # Mock MCP server and HTTP client
            mock_server = None  # Would be actual MCP server
            mock_client = httpx.AsyncClient(base_url="http://localhost:8000")
            
            # Create factory
            factory = ToolProxyFactory.from_json_file(mock_server, mock_client, temp_path)
            
            # Test proxy function creation
            tool_def = test_tools[0]
            proxy_func = factory._create_proxy_function(tool_def)
            
            print(f"✅ Created proxy function: {proxy_func.__name__}")
            print(f"   Description: {proxy_func.__doc__}")
            
            # Test template resolution
            test_kwargs = {"query": "test search", "limit": 5}
            request_args = factory._build_request_args(tool_def["implementation"], test_kwargs)
            print(f"✅ Built request args: {request_args}")
            
            await mock_client.aclose()
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
        
        print("Factory test complete")
    
    # Run the test
    asyncio.run(test_factory())