#!/usr/bin/env python3
"""
L9 Production Module Wrapper - Neural MCP Server Enhanced
Dynamic import wrapper for neural-mcp-server-enhanced.py
Maintains L9 architecture compliance while solving Python import limitations
"""

import sys
import importlib.util
from pathlib import Path

def load_neural_server():
    """Dynamically load the neural-mcp-server-enhanced.py module
    
    Returns:
        module: The loaded module with all MCP tools accessible
    """
    current_dir = Path(__file__).parent
    module_path = current_dir / "neural-mcp-server-enhanced.py"
    
    if not module_path.exists():
        raise ImportError(f"Neural server module not found at {module_path}")
    
    # Dynamic import using importlib.util
    spec = importlib.util.spec_from_file_location(
        'neural_mcp_server_enhanced', 
        str(module_path)
    )
    
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules['neural_mcp_server_enhanced'] = module
    spec.loader.exec_module(module)
    
    return module

# Load the module at import time
_neural_server = load_neural_server()

# Export all functions and attributes from the main module
globals().update({
    name: getattr(_neural_server, name) 
    for name in dir(_neural_server) 
    if not name.startswith('_')
})

# Export the production_health_check function specifically
if hasattr(_neural_server, 'production_health_check'):
    production_health_check = _neural_server.production_health_check

# Ensure the module itself is accessible
neural_server_module = _neural_server

# Add wrapper-specific functions
def production_health_check():
    """Production health check for the wrapper module"""
    try:
        # Count available MCP tools
        available_tools = []
        tool_functions = [
            'memory_store_enhanced', 'memory_search_enhanced', 'graph_query',
            'schema_customization', 'atomic_dependency_tracer', 'project_understanding',
            'semantic_code_search', 'vibe_preservation', 'project_auto_index',
            'neural_system_status', 'neo4j_graph_query', 'neo4j_semantic_graph_search',
            'neo4j_code_dependencies', 'neo4j_migration_status', 'neo4j_index_code_graph'
        ]
        
        for tool_name in tool_functions:
            if hasattr(_neural_server, tool_name):
                available_tools.append(tool_name)
        
        total_tools = len(tool_functions)
        accessible_tools = len(available_tools)
        compliance_score = (accessible_tools / total_tools) * 100
        
        return {
            "status": "healthy" if compliance_score >= 100.0 else "degraded",
            "module_loaded": _neural_server is not None,
            "tools_accessible": accessible_tools,
            "total_tools": total_tools,
            "compliance_score": compliance_score,
            "available_tools": available_tools,
            "missing_tools": [t for t in tool_functions if not hasattr(_neural_server, t)]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "module_loaded": False,
            "tools_accessible": 0,
            "total_tools": 15,
            "compliance_score": 0.0
        }

# Export key functions explicitly for clarity
if hasattr(_neural_server, 'memory_store_enhanced'):
    memory_store_enhanced = _neural_server.memory_store_enhanced

if hasattr(_neural_server, 'memory_search_enhanced'):
    memory_search_enhanced = _neural_server.memory_search_enhanced

if hasattr(_neural_server, 'neural_system_status'):
    neural_system_status = _neural_server.neural_system_status

if hasattr(_neural_server, 'neo4j_graph_query'):
    neo4j_graph_query = _neural_server.neo4j_graph_query

if hasattr(_neural_server, 'initialize'):
    initialize = _neural_server.initialize

# All 15 MCP tools - dynamically exported
MCP_TOOLS = [
    "memory_store_enhanced",
    "memory_search_enhanced", 
    "graph_query",
    "schema_customization", 
    "atomic_dependency_tracer",
    "project_understanding",
    "semantic_code_search",
    "vibe_preservation",
    "project_auto_index", 
    "neural_system_status",
    "neo4j_graph_query",
    "neo4j_semantic_graph_search",
    "neo4j_code_dependencies",
    "neo4j_migration_status",
    "neo4j_index_code_graph"
]

# Verify all tools are accessible
def verify_tools_accessible():
    """Verify all 15 MCP tools are accessible through the wrapper"""
    accessible_tools = []
    missing_tools = []
    
    for tool_name in MCP_TOOLS:
        if hasattr(_neural_server, tool_name):
            accessible_tools.append(tool_name)
        else:
            missing_tools.append(tool_name)
    
    return {
        "accessible_tools": accessible_tools,
        "missing_tools": missing_tools,
        "total_accessible": len(accessible_tools),
        "compliance_score": len(accessible_tools) / len(MCP_TOOLS) * 100
    }

# Production readiness validation (function defined above)

if __name__ == "__main__":
    # Test the wrapper
    health = production_health_check()
    print("L9 Neural Server Wrapper - Production Health Check")
    print("=" * 50)
    print(f"Status: {health['status'].upper()}")
    print(f"Module Loaded: {health['module_loaded']}")
    print(f"Tools Accessible: {health['tools_accessible']}/{health['total_tools']}")
    print(f"Compliance Score: {health['compliance_score']:.1f}%")
    
    if health['missing_tools']:
        print(f"Missing Tools: {', '.join(health['missing_tools'])}")
    
    if health['status'] == 'healthy':
        print("✅ L9 Neural Server wrapper is production ready")
    else:
        print("❌ L9 Neural Server wrapper has issues")
        exit(1)