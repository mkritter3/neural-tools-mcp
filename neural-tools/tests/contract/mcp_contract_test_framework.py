"""
MCP Contract Test Framework
L9-grade contract testing for MCP tool API stability and backward compatibility
"""

import asyncio
import json
import inspect
import sys
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)

@dataclass
class ContractTestResult:
    """Result of a single contract test"""
    tool_name: str
    test_type: str
    status: str  # PASS, FAIL, SKIP
    execution_time_ms: float
    error_message: Optional[str] = None
    expected_signature: Optional[str] = None
    actual_signature: Optional[str] = None
    response_schema: Optional[Dict] = None

@dataclass
class MCPToolContract:
    """Contract definition for an MCP tool"""
    name: str
    parameters: Dict[str, Any]
    required_params: List[str]
    return_type: str
    description: str
    async_required: bool = True

class MCPContractTester:
    """L9 Contract Testing Framework for MCP Tools"""
    
    def __init__(self):
        self.results: List[ContractTestResult] = []
        self.known_contracts: Dict[str, MCPToolContract] = {}
        self.setup_known_contracts()
    
    def setup_known_contracts(self):
        """Define expected contracts for all MCP tools (P1 baseline)"""
        self.known_contracts = {
            "project_understanding": MCPToolContract(
                name="project_understanding",
                parameters={"scope": str, "max_tokens": str},
                required_params=[],
                return_type="Dict[str, Any]",
                description="Generate condensed project understanding"
            ),
            "semantic_code_search": MCPToolContract(
                name="semantic_code_search", 
                parameters={"query": str, "search_type": str, "limit": str, "filters": str},
                required_params=["query"],
                return_type="Dict[str, Any]",
                description="Search code by semantic meaning"
            ),
            "atomic_dependency_tracer": MCPToolContract(
                name="atomic_dependency_tracer",
                parameters={"target": str, "trace_type": str, "depth": str},
                required_params=["target"],
                return_type="Dict[str, Any]",
                description="Trace code dependencies"
            ),
            "vibe_preservation": MCPToolContract(
                name="vibe_preservation",
                parameters={"action": str, "code_sample": str, "context": str},
                required_params=["action"],
                return_type="Dict[str, Any]",
                description="Preserve code style and patterns"
            ),
            "project_auto_index": MCPToolContract(
                name="project_auto_index",
                parameters={"scope": str, "since_minutes": str, "force_reindex": str},
                required_params=[],
                return_type="Dict[str, Any]",
                description="Auto-index project files"
            ),
            "graph_query": MCPToolContract(
                name="graph_query",
                parameters={"query": str},
                required_params=["query"],
                return_type="Dict[str, Any]",
                description="Execute Cypher query on Neo4j"
            ),
            "memory_store_enhanced": MCPToolContract(
                name="memory_store_enhanced",
                parameters={"content": str, "category": str, "metadata": str},
                required_params=["content"],
                return_type="Dict[str, Any]",
                description="Store enhanced memory with embeddings"
            ),
            "memory_search_enhanced": MCPToolContract(
                name="memory_search_enhanced",
                parameters={"query": str, "category": str, "limit": str},
                required_params=["query"],
                return_type="Dict[str, Any]",
                description="Search enhanced memory"
            ),
            "schema_customization": MCPToolContract(
                name="schema_customization",
                parameters={"action": str, "collection_name": str, "schema_data": str},
                required_params=["action"],
                return_type="Dict[str, Any]",
                description="Customize collection schemas"
            )
        }
    
    async def test_tool_signature_compatibility(self, tool_name: str, tool_func: Callable) -> ContractTestResult:
        """Test that tool signature matches expected contract"""
        start_time = time.time()
        
        try:
            # Get actual signature
            sig = inspect.signature(tool_func)
            actual_params = {name: param.annotation for name, param in sig.parameters.items()}
            actual_signature = str(sig)
            
            # Get expected contract
            if tool_name not in self.known_contracts:
                return ContractTestResult(
                    tool_name=tool_name,
                    test_type="signature_compatibility",
                    status="SKIP",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error_message=f"No known contract for {tool_name}"
                )
            
            expected_contract = self.known_contracts[tool_name]
            expected_signature = f"({', '.join(f'{k}: {v.__name__ if hasattr(v, '__name__') else str(v)}' for k, v in expected_contract.parameters.items())})"
            
            # Validate required parameters exist
            missing_params = []
            for required in expected_contract.required_params:
                if required not in sig.parameters:
                    missing_params.append(required)
            
            if missing_params:
                return ContractTestResult(
                    tool_name=tool_name,
                    test_type="signature_compatibility",
                    status="FAIL",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error_message=f"Missing required parameters: {missing_params}",
                    expected_signature=expected_signature,
                    actual_signature=actual_signature
                )
            
            # Check if function is async as expected
            if expected_contract.async_required and not asyncio.iscoroutinefunction(tool_func):
                return ContractTestResult(
                    tool_name=tool_name,
                    test_type="signature_compatibility", 
                    status="FAIL",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error_message="Function should be async",
                    expected_signature=expected_signature,
                    actual_signature=actual_signature
                )
            
            return ContractTestResult(
                tool_name=tool_name,
                test_type="signature_compatibility",
                status="PASS",
                execution_time_ms=(time.time() - start_time) * 1000,
                expected_signature=expected_signature,
                actual_signature=actual_signature
            )
            
        except Exception as e:
            return ContractTestResult(
                tool_name=tool_name,
                test_type="signature_compatibility",
                status="FAIL",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Signature test failed: {str(e)}"
            )
    
    async def test_tool_execution(self, tool_name: str, tool_func: Callable) -> ContractTestResult:
        """Test that tool executes without errors using safe test inputs"""
        start_time = time.time()
        
        # Safe test inputs for each tool
        test_inputs = {
            "project_understanding": {"scope": "summary"},
            "semantic_code_search": {"query": "test function"},
            "atomic_dependency_tracer": {"target": "test_function"},
            "vibe_preservation": {"action": "analyze"},
            "project_auto_index": {"scope": "test"},
            "graph_query": {"query": "RETURN 1 as test LIMIT 1"},
            "memory_store_enhanced": {"content": "test memory"},
            "memory_search_enhanced": {"query": "test search"},
            "schema_customization": {"action": "list"}
        }
        
        if tool_name not in test_inputs:
            return ContractTestResult(
                tool_name=tool_name,
                test_type="execution_test",
                status="SKIP",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message="No test inputs defined"
            )
        
        try:
            # Execute tool with test inputs
            result = await tool_func(**test_inputs[tool_name])
            
            # Validate response structure
            if not isinstance(result, dict):
                return ContractTestResult(
                    tool_name=tool_name,
                    test_type="execution_test",
                    status="FAIL",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error_message=f"Expected dict response, got {type(result)}"
                )
            
            # Check for common response patterns
            response_schema = {
                "has_status": "status" in result,
                "has_data": any(key in result for key in ["results", "data", "content", "response"]),
                "has_error": "error" in result,
                "keys": list(result.keys())
            }
            
            return ContractTestResult(
                tool_name=tool_name,
                test_type="execution_test", 
                status="PASS",
                execution_time_ms=(time.time() - start_time) * 1000,
                response_schema=response_schema
            )
            
        except Exception as e:
            return ContractTestResult(
                tool_name=tool_name,
                test_type="execution_test",
                status="FAIL",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=f"Execution failed: {str(e)}"
            )
    
    async def discover_mcp_tools(self) -> Dict[str, Callable]:
        """Discover all MCP tools from the codebase"""
        tools = {}
        
        try:
            # Import MCP tool modules with proper path setup
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from servers.tools.core_tools import register_core_tools
            from servers.tools.memory_tools import register_memory_tools
            
            # Create mock MCP server to capture tool registrations
            class MockMCP:
                def __init__(self):
                    self.registered_tools = {}
                
                def tool(self):
                    def decorator(func):
                        self.registered_tools[func.__name__] = func
                        return func
                    return decorator
            
            # Register tools with mock MCP
            mock_mcp = MockMCP()
            register_core_tools(mock_mcp)
            register_memory_tools(mock_mcp)
            
            tools.update(mock_mcp.registered_tools)
            
        except Exception as e:
            logger.error(f"Failed to discover MCP tools: {e}")
        
        return tools
    
    async def run_full_contract_test_suite(self) -> Dict[str, Any]:
        """Run complete contract test suite"""
        print("🧪 Starting MCP Contract Test Suite...")
        
        # Discover all MCP tools
        tools = await self.discover_mcp_tools()
        print(f"📋 Discovered {len(tools)} MCP tools")
        
        # Run tests for each tool
        for tool_name, tool_func in tools.items():
            print(f"🔍 Testing {tool_name}...")
            
            # Test signature compatibility
            sig_result = await self.test_tool_signature_compatibility(tool_name, tool_func)
            self.results.append(sig_result)
            
            # Test execution (only if signature is valid)
            if sig_result.status == "PASS":
                exec_result = await self.test_tool_execution(tool_name, tool_func)
                self.results.append(exec_result)
        
        # Generate summary
        summary = self.generate_test_summary()
        return summary
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"]) 
        skipped = len([r for r in self.results if r.status == "SKIP"])
        
        avg_execution_time = sum(r.execution_time_ms for r in self.results) / total_tests if total_tests > 0 else 0
        
        failures = [r for r in self.results if r.status == "FAIL"]
        
        summary = {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
            "avg_execution_time_ms": avg_execution_time,
            "failures": [asdict(f) for f in failures],
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        return summary

if __name__ == "__main__":
    async def main():
        tester = MCPContractTester()
        results = await tester.run_full_contract_test_suite()
        
        print("\\n" + "="*60)
        print("🎯 MCP CONTRACT TEST RESULTS")
        print("="*60)
        print(f"✅ Passed: {results['passed']}")
        print(f"❌ Failed: {results['failed']}")  
        print(f"⏭️  Skipped: {results['skipped']}")
        print(f"📊 Success Rate: {results['success_rate']:.1f}%")
        print(f"⏱️  Avg Execution: {results['avg_execution_time_ms']:.1f}ms")
        
        if results['failures']:
            print("\\n💥 FAILURES:")
            for failure in results['failures']:
                print(f"  - {failure['tool_name']} ({failure['test_type']}): {failure['error_message']}")
        
        return results['success_rate'] >= 90.0  # L9 standard: 90% pass rate
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1)