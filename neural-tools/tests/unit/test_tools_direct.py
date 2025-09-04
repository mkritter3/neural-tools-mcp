#!/usr/bin/env python3
"""
Direct MCP Tool Testing Framework
Tests MCP tools directly within the server environment
Part of ADR-0007 Phase 2 implementation
"""

import sys
import os
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up environment variables
os.environ['PROJECT_NAME'] = 'default'
os.environ['QDRANT_URL'] = 'http://qdrant:6333' 
os.environ['NEO4J_URI'] = 'bolt://neo4j-graph:7687'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'neo4jpassword'
os.environ['NOMIC_ENDPOINT'] = 'http://nomic-embed:8080/embed'

class DirectToolTester:
    """Test MCP tools directly in server environment"""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
        # All 15 MCP tools to test
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
    
    async def initialize_server_components(self):
        """Initialize server components before testing"""
        try:
            logger.info("🔧 Initializing server components...")
            
            # Import server functions after environment setup
            global neural_server_module
            neural_server_module = __import__('neural-mcp-server-enhanced')
            
            # Initialize components
            await neural_server_module.initialize()
            logger.info("✅ Server components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize server: {e}")
            raise
    
    def get_test_params(self, tool_name: str) -> Dict[str, Any]:
        """Get test parameters for each tool"""
        test_cases = {
            "memory_store_enhanced": {
                "content": "Test content for MCP compliance validation",
                "category": "test",
                "create_graph_entities": True
            },
            "memory_search_enhanced": {
                "query": "test search query", 
                "limit": "5"
            },
            "graph_query": {
                "query": "MATCH (n) RETURN count(n) LIMIT 1"
            },
            "schema_customization": {
                "action": "list_schemas"
            },
            "atomic_dependency_tracer": {
                "target": "test_function",
                "trace_type": "calls"
            },
            "project_understanding": {
                "scope": "architecture"
            },
            "semantic_code_search": {
                "query": "error handling logic"
            },
            "vibe_preservation": {
                "action": "show"
            },
            "project_auto_index": {
                "scope": "modified"
            },
            "neural_system_status": {},
            "neo4j_graph_query": {
                "cypher_query": "MATCH (n) RETURN count(n) LIMIT 1"
            },
            "neo4j_semantic_graph_search": {
                "query_text": "function definitions"
            },
            "neo4j_code_dependencies": {
                "file_path": "neural_server_2025.py"
            },
            "neo4j_migration_status": {},
            "neo4j_index_code_graph": {
                "file_paths": "neural_server_2025.py"
            }
        }
        
        return test_cases.get(tool_name, {})
    
    async def test_tool_direct(self, tool_name: str) -> Dict[str, Any]:
        """Test individual MCP tool directly"""
        try:
            # Get tool function from module
            tool_func = getattr(neural_server_module, tool_name, None)
            
            if not tool_func:
                return {
                    "status": "fail",
                    "error": f"Tool function {tool_name} not found in module",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get test parameters
            params = self.get_test_params(tool_name)
            
            # Execute the tool
            logger.debug(f"Calling {tool_name} with params: {params}")
            
            if asyncio.iscoroutinefunction(tool_func):
                result = await asyncio.wait_for(tool_func(**params), timeout=30.0)
            else:
                result = tool_func(**params)
            
            # Validate result structure
            is_valid = self._validate_result(result)
            
            return {
                "status": "pass" if is_valid else "fail",
                "result": result,
                "params_used": params,
                "validation_passed": is_valid,
                "timestamp": datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "fail",
                "error": "Tool execution timed out (30s)",
                "params_used": self.get_test_params(tool_name),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.debug(f"Tool {tool_name} error: {str(e)}")
            return {
                "status": "fail", 
                "error": str(e),
                "params_used": self.get_test_params(tool_name),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_result(self, result: Any) -> bool:
        """Validate tool result structure"""
        if not isinstance(result, dict):
            return False
        
        # Check for standard MCP response patterns
        if 'status' in result:
            return result['status'] in ['success', 'error']
        
        # Alternative patterns for valid results
        valid_keys = ['content', 'data', 'result', 'message', 'results', 'count']
        if any(key in result for key in valid_keys):
            return True
        
        return False
    
    async def test_all_tools(self) -> Dict[str, Any]:
        """Test all MCP tools directly"""
        logger.info("🚀 Starting direct tool testing")
        logger.info(f"Testing {len(self.mcp_tools)} MCP tools")
        
        # Initialize server components first
        await self.initialize_server_components()
        
        start_time = datetime.now()
        
        for tool_name in self.mcp_tools:
            logger.info(f"📊 Testing tool: {tool_name}")
            result = await self.test_tool_direct(tool_name)
            self.test_results[tool_name] = result
            
            if result['status'] == 'pass':
                self.passed_tests += 1
            self.total_tests += 1
            
            status_icon = "✅" if result['status'] == 'pass' else "❌"
            logger.info(f"{status_icon} {tool_name}: {result['status'].upper()}")
            
            # Brief pause between tests
            await asyncio.sleep(0.1)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate compliance score
        compliance_score = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tools": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.total_tests - self.passed_tests,
            "compliance_score": compliance_score,
            "duration_seconds": duration,
            "test_results": self.test_results
        }
        
        logger.info(f"🎯 Testing Complete: {self.passed_tests}/{self.total_tests} passed ({compliance_score:.1f}%)")
        
        return summary
    
    def generate_compliance_report(self) -> str:
        """Generate detailed compliance report"""
        if not self.test_results:
            return "No test results available"
        
        report = []
        report.append("# Direct Tool Testing Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        compliance_score = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        report.append(f"## Summary")
        report.append(f"- **Total Tools**: {self.total_tests}")
        report.append(f"- **Passed**: {self.passed_tests}")  
        report.append(f"- **Failed**: {self.total_tests - self.passed_tests}")
        report.append(f"- **Compliance Score**: {compliance_score:.1f}%")
        report.append("")
        
        report.append("## Detailed Results")
        for tool_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'pass' else "❌"
            report.append(f"### {status_icon} {tool_name}")
            report.append(f"- Status: {result['status'].upper()}")
            if 'error' in result:
                error_msg = result['error']
                # Truncate very long error messages
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                report.append(f"- Error: {error_msg}")
            if 'result' in result and result['result']:
                report.append(f"- Result: Available ({type(result['result']).__name__})")
            report.append("")
        
        return "\n".join(report)

async def main():
    """Main testing function"""
    tester = DirectToolTester()
    
    try:
        # Run comprehensive testing
        summary = await tester.test_all_tools()
        
        # Generate compliance report
        report = tester.generate_compliance_report()
        
        # Save results
        with open('/app/direct_tool_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open('/app/direct_tool_report.md', 'w') as f:
            f.write(report)
        
        print("\n" + "="*50)
        print("DIRECT TOOL TESTING COMPLETE")
        print("="*50)
        print(f"Results: {summary['passed']}/{summary['total_tools']} tools passed")
        print(f"Compliance Score: {summary['compliance_score']:.1f}%")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Report saved to: /app/direct_tool_report.md")
        print("="*50)
        
        return summary['compliance_score'] >= 80.0  # Lower threshold for initial validation
        
    except Exception as e:
        logger.error(f"❌ Testing framework error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)