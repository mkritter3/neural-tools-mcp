"""
Neural Tools Help - September 2025 Standards
Documentation and usage examples for all neural tools

ADR-0076: Modular tool architecture
"""

import json
import importlib
from pathlib import Path
from typing import List, Dict, Any

from mcp import types
from ..shared.performance_metrics import track_performance

import logging
logger = logging.getLogger(__name__)

# Tool metadata for automatic registration
TOOL_CONFIG = {
    "name": "neural_tools_help",
    "description": "Show usage examples and constraints for all neural tools.",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }
}

@track_performance
async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Main tool execution function - September 2025 Standards

    Args:
        arguments: Validated input parameters

    Returns:
        List of TextContent responses with help documentation
    """
    try:
        help_content = {
            "neural_tools_help": "September 2025 Modular Architecture",
            "architecture": "modular_september_2025",
            "adr_compliance": ["ADR-0075 (Connection Pooling)", "ADR-0076 (Modular Architecture)"],
            "available_tools": _discover_available_tools(),
            "usage_patterns": _get_usage_patterns(),
            "performance_optimizations": _get_performance_info(),
            "configuration": _get_configuration_info()
        }

        response = [types.TextContent(type="text", text=json.dumps(help_content, indent=2))]
        return response

    except Exception as e:
        logger.error(f"Help generation failed: {e}")
        return _make_error_response(f"Help generation failed: {e}")

def _discover_available_tools() -> dict:
    """Discover and document all available tools"""
    tools_dir = Path(__file__).parent
    tools = {}

    for tool_file in tools_dir.glob("*.py"):
        if tool_file.stem.startswith("_") or tool_file.stem == "neural_tools_help":
            continue

        try:
            module_name = f"neural_mcp.tools.{tool_file.stem}"
            module = importlib.import_module(module_name)

            if hasattr(module, 'TOOL_CONFIG'):
                config = module.TOOL_CONFIG
                tools[config["name"]] = {
                    "description": config["description"],
                    "input_schema": config["inputSchema"],
                    "module": tool_file.stem,
                    "status": "active"
                }
        except Exception as e:
            tools[tool_file.stem] = {
                "status": "error",
                "error": str(e)
            }

    return tools

def _get_usage_patterns() -> dict:
    """Get common usage patterns and examples"""
    return {
        "basic_usage": {
            "description": "All tools follow September 2025 standards",
            "pattern": "Each tool is a standalone module with standardized interface",
            "performance": "ADR-0075 connection pooling eliminates 1000ms+ initialization overhead"
        },
        "caching": {
            "description": "Automatic caching with TTL support",
            "cache_duration": "3600 seconds (1 hour) default",
            "cache_hit_benefits": "Sub-millisecond response times for repeated queries"
        },
        "error_handling": {
            "description": "Standardized error responses across all tools",
            "format": "JSON with status, message, tool, and architecture fields"
        },
        "chain_of_tools": {
            "description": "Tools can be composed in workflows",
            "example": "canon_understanding → dependency_analysis → semantic_search"
        }
    }

def _get_performance_info() -> dict:
    """Get performance optimization information"""
    return {
        "connection_pooling": {
            "adr": "ADR-0075",
            "benefit": "Eliminates 1000ms+ Neo4j initialization per query",
            "implementation": "Shared service instances across all tools"
        },
        "caching": {
            "strategy": "In-memory with TTL",
            "cleanup": "Automatic cleanup of expired entries",
            "hit_rate_target": ">80% for repeated queries"
        },
        "modular_architecture": {
            "adr": "ADR-0076",
            "benefits": [
                "Reduced maintenance overhead (75%)",
                "Isolated testing capabilities",
                "Faster development velocity",
                "Better code organization"
            ]
        }
    }

def _get_configuration_info() -> dict:
    """Get configuration and setup information"""
    return {
        "canon_yaml": {
            "description": "User-defined source of truth configuration",
            "location": ".canon.yaml in project root",
            "format": "September 2025 standards with community clustering",
            "required": False,
            "auto_generation": "Example provided when missing"
        },
        "neo4j": {
            "requirement": "Neo4j database with project isolation",
            "port": "47687 (configured in environment)",
            "authentication": "graphrag-password"
        },
        "environment": {
            "python_version": "3.8+",
            "required_packages": ["mcp", "neo4j", "yaml", "psutil"],
            "optional_packages": ["neo4j-graph-data-science"]
        }
    }

def _make_error_response(error: str) -> List[types.TextContent]:
    """Standard error response format"""
    return [types.TextContent(type="text", text=json.dumps({
        "status": "error",
        "message": error,
        "tool": TOOL_CONFIG["name"],
        "architecture": "modular_september_2025"
    }))]