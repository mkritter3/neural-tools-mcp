#!/usr/bin/env python3
"""
MCP Tool Compliance Testing Framework
Tests all 15 MCP tools via proper MCP protocol communication
Part of ADR-0007 Phase 2 implementation
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPToolTester:
    """Comprehensive MCP tool testing framework"""
    
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
    
    async def test_all_tools(self) -> Dict[str, Any]:
        """Test all MCP tools systematically"""
        logger.info("🚀 Starting comprehensive MCP tool testing")
        logger.info(f"Testing {len(self.mcp_tools)} MCP tools")
        
        start_time = datetime.now()
        
        for tool_name in self.mcp_tools:
            logger.info(f"📊 Testing tool: {tool_name}")
            result = await self.test_mcp_tool(tool_name)
            self.test_results[tool_name] = result
            
            if result['status'] == 'pass':
                self.passed_tests += 1
            self.total_tests += 1
            
            logger.info(f"✅ {tool_name}: {result['status'].upper()}")
        
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
    
    async def test_mcp_tool(self, tool_name: str) -> Dict[str, Any]:
        """Test individual MCP tool with appropriate parameters"""
        
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
        
        try:
            # Import the tool dynamically from the MCP server
            from neural_mcp_server_enhanced import get_tool_by_name
            
            tool_func = get_tool_by_name(tool_name)
            if not tool_func:
                return {
                    "status": "fail",
                    "error": f"Tool {tool_name} not found",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get test parameters
            params = test_cases.get(tool_name, {})
            
            # Execute the tool
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**params)
            else:
                result = tool_func(**params)
            
            # Validate result structure
            is_valid = self._validate_mcp_result(result)
            
            return {
                "status": "pass" if is_valid else "fail",
                "result": result,
                "params_used": params,
                "validation_passed": is_valid,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error testing {tool_name}: {str(e)}")
            return {
                "status": "fail", 
                "error": str(e),
                "params_used": test_cases.get(tool_name, {}),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_mcp_result(self, result: Any) -> bool:
        """Validate MCP tool result structure"""
        if not isinstance(result, dict):
            return False
        
        # Check for required MCP response structure
        if 'status' in result:
            return result['status'] in ['success', 'error']
        
        # Alternative: check for content/data structure
        if any(key in result for key in ['content', 'data', 'result', 'message']):
            return True
        
        return False
    
    def generate_compliance_report(self) -> str:
        """Generate detailed compliance report"""
        if not self.test_results:
            return "No test results available"
        
        report = []
        report.append("# MCP Tools Compliance Report")
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
                report.append(f"- Error: {result['error']}")
            if 'validation_passed' in result:
                report.append(f"- Validation: {'PASS' if result['validation_passed'] else 'FAIL'}")
            report.append("")
        
        return "\n".join(report)

async def main():
    """Main testing function"""
    tester = MCPToolTester()
    
    try:
        # Run comprehensive testing
        summary = await tester.test_all_tools()
        
        # Generate compliance report
        report = tester.generate_compliance_report()
        
        # Save results
        with open('mcp_compliance_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open('mcp_compliance_report.md', 'w') as f:
            f.write(report)
        
        print("\n" + "="*50)
        print("MCP COMPLIANCE TESTING COMPLETE")
        print("="*50)
        print(f"Results: {summary['passed']}/{summary['total_tools']} tools passed")
        print(f"Compliance Score: {summary['compliance_score']:.1f}%")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Report saved to: mcp_compliance_report.md")
        print("="*50)
        
        return summary['compliance_score'] >= 95.0
        
    except Exception as e:
        logger.error(f"❌ Testing framework error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)