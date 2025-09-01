#!/usr/bin/env python3
"""
L9 Neural Tools - Shared Test Base Classes
Eliminates code duplication across test files
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Standardized logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseNeuralTester:
    """Base class for all neural tools testing"""
    
    def __init__(self, test_name: str = "BaseTest"):
        self.test_name = test_name
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
        # Standard test environment setup
        self._setup_environment()
        
    def _setup_environment(self):
        """Set up standard Docker environment variables"""
        os.environ['PROJECT_NAME'] = 'default'
        os.environ['QDRANT_URL'] = 'http://neural-data-storage:6333'
        os.environ['NEO4J_URI'] = 'bolt://neo4j-graph:7687'
        os.environ['NEO4J_USERNAME'] = 'neo4j'
        os.environ['NEO4J_PASSWORD'] = 'neural-l9-2025'
        os.environ['NOMIC_ENDPOINT'] = 'http://neural-embeddings:8000/embed'
        
    def log_test_start(self, test_description: str):
        """Standardized test start logging"""
        logger.info(f"🧪 Starting {self.test_name}: {test_description}")
        self.total_tests += 1
        
    def log_test_pass(self, test_description: str, details: Optional[str] = None):
        """Standardized test pass logging"""
        logger.info(f"✅ PASSED: {test_description}")
        if details:
            logger.info(f"   Details: {details}")
        self.passed_tests += 1
        
    def log_test_fail(self, test_description: str, error: str):
        """Standardized test failure logging"""
        logger.error(f"❌ FAILED: {test_description}")
        logger.error(f"   Error: {error}")
        
    def get_test_summary(self) -> Dict[str, Any]:
        """Get standardized test summary"""
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        return {
            "test_name": self.test_name,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.total_tests - self.passed_tests,
            "success_rate": f"{success_rate:.1f}%",
            "timestamp": datetime.now().isoformat()
        }
        
    def print_summary(self):
        """Print standardized test summary"""
        summary = self.get_test_summary()
        
        logger.info("=" * 60)
        logger.info(f"🎯 TEST SUMMARY - {summary['test_name']}")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Success Rate: {summary['success_rate']}")
        logger.info("=" * 60)

class MCPToolTester(BaseNeuralTester):
    """Base class for MCP tool testing with common patterns"""
    
    def __init__(self, test_name: str = "MCPToolTest"):
        super().__init__(test_name)
        
        # Standard MCP tool list for testing
        self.mcp_tools = [
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
        
    async def test_tool_exists(self, tool_name: str) -> bool:
        """Test if an MCP tool exists and is callable"""
        try:
            # Import the server module
            import neural_mcp_server_enhanced as server
            
            # Check if tool function exists
            tool_func = getattr(server, tool_name, None)
            if tool_func is None:
                self.log_test_fail(f"Tool existence: {tool_name}", f"Tool function {tool_name} not found")
                return False
                
            self.log_test_pass(f"Tool existence: {tool_name}")
            return True
            
        except Exception as e:
            self.log_test_fail(f"Tool existence: {tool_name}", str(e))
            return False
            
    async def test_tool_health(self, tool_name: str, test_args: Dict[str, Any] = None) -> bool:
        """Test basic health of an MCP tool"""
        try:
            # Import the server module
            import neural_mcp_server_enhanced as server
            
            # Get tool function
            tool_func = getattr(server, tool_name)
            
            # Use provided args or defaults
            args = test_args or self._get_default_args(tool_name)
            
            # Call tool
            result = await tool_func(**args)
            
            # Validate result structure
            if not isinstance(result, dict):
                self.log_test_fail(f"Tool health: {tool_name}", "Result is not a dictionary")
                return False
                
            if "status" not in result:
                self.log_test_fail(f"Tool health: {tool_name}", "Result missing 'status' field")
                return False
                
            self.log_test_pass(f"Tool health: {tool_name}", f"Status: {result.get('status')}")
            return True
            
        except Exception as e:
            self.log_test_fail(f"Tool health: {tool_name}", str(e))
            return False
            
    def _get_default_args(self, tool_name: str) -> Dict[str, Any]:
        """Get default test arguments for common MCP tools"""
        defaults = {
            "memory_store_enhanced": {"content": "test content", "category": "test"},
            "memory_search_enhanced": {"query": "test query"},
            "graph_query": {"query": "MATCH (n) RETURN count(n) LIMIT 1"},
            "semantic_code_search": {"query": "function definition"},
            "project_understanding": {"scope": "architecture"},
            "neural_system_status": {},
            "neo4j_migration_status": {},
        }
        
        return defaults.get(tool_name, {})

class ProductionTester(BaseNeuralTester):
    """Base class for production environment testing"""
    
    def __init__(self, test_name: str = "ProductionTest"):
        super().__init__(test_name)
        
    async def test_service_connectivity(self, service_name: str, endpoint: str) -> bool:
        """Test connectivity to a service endpoint"""
        try:
            import httpx
            
            self.log_test_start(f"Service connectivity: {service_name}")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{endpoint}/health", follow_redirects=True)
                
                if response.status_code == 200:
                    self.log_test_pass(f"Service connectivity: {service_name}")
                    return True
                else:
                    self.log_test_fail(f"Service connectivity: {service_name}", f"HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            self.log_test_fail(f"Service connectivity: {service_name}", str(e))
            return False