"""
Semantic search tool - Compatibility wrapper for fast_search
ADR-0096: Redirects to fast_search for backward compatibility
"""

from neural_mcp.tools.fast_search import (
    TOOL_CONFIG as FAST_SEARCH_CONFIG,
    execute as fast_search_execute
)

# Use fast_search config but keep semantic_search name for compatibility
TOOL_CONFIG = {
    "name": "semantic_search",  # Keep old name for compatibility
    "description": FAST_SEARCH_CONFIG["description"],
    "inputSchema": FAST_SEARCH_CONFIG["inputSchema"]
}

# Delegate to fast_search
execute = fast_search_execute