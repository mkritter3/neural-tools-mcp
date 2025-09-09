#!/usr/bin/env python3
"""
API Key Authentication for FastAPI Neural Search Server
Secures endpoints with X-API-Key header authentication
"""

import os
import logging
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# Load API key from environment
API_KEY = os.environ.get("FASTAPI_API_KEY", "neural-search-mcp-proxy-key-12345")
API_KEY_NAME = "X-API-Key"

# Create FastAPI security scheme
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Validate API key from request header
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not api_key:
        logger.warning("Missing API key in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


def verify_api_key_configured() -> bool:
    """
    Check if API key is properly configured
    
    Returns:
        True if API key is configured with a non-default value
    """
    return bool(API_KEY and API_KEY != "neural-search-mcp-proxy-key-12345")


def get_auth_summary() -> dict:
    """
    Get authentication configuration summary for debugging
    
    Returns:
        Dictionary with auth status info
    """
    return {
        "api_key_configured": bool(API_KEY),
        "api_key_is_default": API_KEY == "neural-search-mcp-proxy-key-12345",
        "header_name": API_KEY_NAME,
        "security_enabled": True
    }


if __name__ == "__main__":
    # Test authentication configuration
    print("FastAPI Authentication Configuration:")
    summary = get_auth_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    if not verify_api_key_configured():
        print("\n⚠️  WARNING: Using default API key - set FASTAPI_API_KEY environment variable")
    else:
        print("\n✅ API key properly configured")