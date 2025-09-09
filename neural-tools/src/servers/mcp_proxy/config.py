#!/usr/bin/env python3
"""
Configuration for MCP HTTP Proxy Server
Manages settings for connecting to FastAPI backend with API key authentication
"""

import os
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Configuration settings for MCP HTTP proxy server"""
    
    # FastAPI backend connection
    FASTAPI_BASE_URL: str = Field(
        default="http://localhost:8000",
        description="Base URL for the FastAPI neural search server"
    )
    
    # API key for service-to-service authentication (disabled)
    FASTAPI_API_KEY: Optional[str] = Field(
        default="",
        description="API key for authenticating with FastAPI backend (currently disabled)"
    )
    
    # HTTP client settings
    HTTP_TIMEOUT: float = Field(
        default=30.0,
        description="Timeout in seconds for HTTP requests to FastAPI backend"
    )
    
    # Tool configuration
    MCP_TOOLS_CONFIG: str = Field(
        default="mcp_tools.json",
        description="Path to JSON file containing tool definitions"
    )
    
    # Logging settings
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    # Development settings
    DEBUG_MODE: bool = Field(
        default=False,
        description="Enable debug mode for additional logging"
    )

    @validator('FASTAPI_BASE_URL')
    def validate_base_url(cls, v):
        """Ensure base URL doesn't end with trailing slash"""
        return v.rstrip('/')
    
    @validator('MCP_TOOLS_CONFIG')
    def validate_tools_config_path(cls, v):
        """Resolve tools config path relative to this file"""
        if not os.path.isabs(v):
            # Make path relative to the parent directory (neural-tools/)
            config_dir = Path(__file__).parent.parent.parent.parent
            return str(config_dir / v)
        return v

    # Configuration loading
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra environment variables
    )


# Create global settings instance
settings = Settings()


def get_config_summary() -> dict:
    """Get a summary of current configuration for debugging"""
    return {
        "fastapi_base_url": settings.FASTAPI_BASE_URL,
        "api_key_configured": bool(settings.FASTAPI_API_KEY),
        "http_timeout": settings.HTTP_TIMEOUT,
        "tools_config_path": settings.MCP_TOOLS_CONFIG,
        "log_level": settings.LOG_LEVEL,
        "debug_mode": settings.DEBUG_MODE,
    }


if __name__ == "__main__":
    # Test configuration loading
    print("MCP Proxy Configuration:")
    print(f"FastAPI Base URL: {settings.FASTAPI_BASE_URL}")
    print(f"API Key configured: {'Yes' if settings.FASTAPI_API_KEY else 'No'}")
    print(f"Tools config: {settings.MCP_TOOLS_CONFIG}")
    print(f"HTTP Timeout: {settings.HTTP_TIMEOUT}s")
    print(f"Log Level: {settings.LOG_LEVEL}")
    
    print("\nFull configuration summary:")
    import json
    print(json.dumps(get_config_summary(), indent=2))