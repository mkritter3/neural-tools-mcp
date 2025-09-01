#!/usr/bin/env python3
"""
MCP Protocol Testing Framework
Tests MCP tools via proper MCP protocol communication
Part of ADR-0007 Phase 2 implementation
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPProtocolTester:
    """Test MCP tools via proper MCP protocol"""
    
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
                "file_path": "neural-mcp-server-enhanced.py"
            },
            "neo4j_migration_status": {},
            "neo4j_index_code_graph": {
                "file_paths": "neural-mcp-server-enhanced.py"
            }
        }
        
        return test_cases.get(tool_name, {})
    
    async def test_tool_via_mcp(self, tool_name: str) -> Dict[str, Any]:
        """Test individual MCP tool via MCP protocol"""
        try:
            # Create MCP request message
            params = self.get_test_params(tool_name)
            
            mcp_request = {
                "jsonrpc": "2.0",
                "id": f"test_{tool_name}",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }
            
            # Set up environment
            env = os.environ.copy()
            env.update({
                'PROJECT_NAME': 'default',
                'QDRANT_URL': 'http://qdrant:6333',
                'NEO4J_URI': 'bolt://neo4j-graph:7687',
                'NEO4J_USERNAME': 'neo4j',
                'NEO4J_PASSWORD': 'neo4jpassword',
                'NOMIC_ENDPOINT': 'http://nomic-embed:8080/embed'
            })
            
            # Start MCP server process
            process = await asyncio.create_subprocess_exec(
                'python3', '/app/neural-mcp-server-enhanced.py',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd='/app'
            )
            
            # Send initialization request first (Zen MCP pattern)
            init_request = {
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "neural-tools-test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            # Send init request
            init_json = json.dumps(init_request) + '\n'
            process.stdin.write(init_json.encode())
            await process.stdin.drain()
            
            # Wait for init response
            await asyncio.sleep(1.0)
            
            # Send tool call request
            request_json = json.dumps(mcp_request) + '\n'
            process.stdin.write(request_json.encode())
            await process.stdin.drain()
            
            # Wait for response with timeout
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "status": "fail",
                    "error": "Timeout waiting for response",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Parse response
            response_text = stdout.decode()
            stderr_text = stderr.decode()
            
            if stderr_text:
                logger.warning(f"Tool {tool_name} stderr: {stderr_text[:200]}...")
            
            # Look for JSON responses in output
            responses = []
            for line in response_text.strip().split('\n'):
                if line.strip():
                    try:
                        resp = json.loads(line)
                        responses.append(resp)
                    except json.JSONDecodeError:
                        continue
            
            # Find and validate tool response (Zen MCP pattern)
            tool_response = None
            for resp in responses:
                if resp.get('id') == f"test_{tool_name}":
                    tool_response = resp
                    break
            
            if tool_response:
                # Validate JSON-RPC response format (Zen MCP pattern)
                validation_errors = []
                
                # Check required JSON-RPC fields
                if "jsonrpc" not in tool_response or tool_response["jsonrpc"] != "2.0":
                    validation_errors.append("Invalid or missing jsonrpc field")
                
                if "id" not in tool_response:
                    validation_errors.append("Missing id field")
                
                if "result" not in tool_response and "error" not in tool_response:
                    validation_errors.append("Response must have either result or error field")
                
                # If validation passed and we have a result
                if not validation_errors and 'result' in tool_response:
                    return {
                        "status": "pass",
                        "result": tool_response['result'],
                        "params_used": params,
                        "validation_passed": True,
                        "timestamp": datetime.now().isoformat()
                    }
                # If validation passed but we have an error
                elif not validation_errors and 'error' in tool_response:
                    error_msg = tool_response['error'].get('message', str(tool_response['error']))
                    return {
                        "status": "fail",
                        "error": f"Tool error: {error_msg}",
                        "params_used": params,
                        "validation_passed": True,
                        "timestamp": datetime.now().isoformat()
                    }
                # If validation failed
                else:
                    return {
                        "status": "fail",
                        "error": f"JSON-RPC validation failed: {', '.join(validation_errors)}",
                        "params_used": params,
                        "validation_passed": False,
                        "timestamp": datetime.now().isoformat()
                    }
            
            return {
                "status": "fail",
                "error": f"No valid response found. Raw output: {response_text[:200]}...",
                "params_used": params,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error testing {tool_name}: {str(e)}")
            return {
                "status": "fail", 
                "error": str(e),
                "params_used": self.get_test_params(tool_name),
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_all_tools(self) -> Dict[str, Any]:
        """Test all MCP tools via MCP protocol"""
        logger.info("🚀 Starting MCP protocol testing")
        logger.info(f"Testing {len(self.mcp_tools)} MCP tools via protocol")
        
        start_time = datetime.now()
        
        for tool_name in self.mcp_tools:
            logger.info(f"📊 Testing tool: {tool_name}")
            result = await self.test_tool_via_mcp(tool_name)
            self.test_results[tool_name] = result
            
            if result['status'] == 'pass':
                self.passed_tests += 1
            self.total_tests += 1
            
            status_icon = "✅" if result['status'] == 'pass' else "❌"
            logger.info(f"{status_icon} {tool_name}: {result['status'].upper()}")
        
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
        report.append("# MCP Protocol Testing Report")
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
            if 'result' in result:
                report.append(f"- Result: Available")
            report.append("")
        
        return "\n".join(report)

async def main():
    """Main testing function"""
    tester = MCPProtocolTester()
    
    try:
        # Run comprehensive testing
        summary = await tester.test_all_tools()
        
        # Generate compliance report
        report = tester.generate_compliance_report()
        
        # Save results
        with open('/app/mcp_protocol_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open('/app/mcp_protocol_report.md', 'w') as f:
            f.write(report)
        
        print("\n" + "="*50)
        print("MCP PROTOCOL TESTING COMPLETE")
        print("="*50)
        print(f"Results: {summary['passed']}/{summary['total_tools']} tools passed")
        print(f"Compliance Score: {summary['compliance_score']:.1f}%")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Report saved to: /app/mcp_protocol_report.md")
        print("="*50)
        
        return summary['compliance_score'] >= 95.0
        
    except Exception as e:
        logger.error(f"❌ Testing framework error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)