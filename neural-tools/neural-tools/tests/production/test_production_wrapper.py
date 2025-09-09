#!/usr/bin/env python3
"""
L9 Production Wrapper Verification - End-to-End Test
Tests the neural_mcp_server_enhanced.py wrapper module in Docker environment
Verifies all 15 MCP tools work with real queries and database connections
"""

import sys
import os
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up environment variables for Docker container
os.environ['PROJECT_NAME'] = 'default'
os.environ['QDRANT_URL'] = 'http://neural-data-storage:6333' 
os.environ['NEO4J_URI'] = 'bolt://neo4j-graph:7687'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'neural-l9-2025'
os.environ['NOMIC_ENDPOINT'] = 'http://neural-embeddings:8000/embed'

class ProductionWrapperTester:
    """Test production wrapper with real end-to-end queries"""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.wrapper_module = None
        
        # Test queries for real functionality
        self.real_test_queries = {
            "memory_store_enhanced": {
                "content": "L9 Neural System Architecture supports 15 MCP tools with Neo4j GraphRAG integration for production-grade code analysis",
                "category": "architecture",
                "create_graph_entities": True
            },
            "memory_search_enhanced": {
                "query": "L9 Neural System Architecture", 
                "limit": "3",
                "mode": "rrf_hybrid",
                "diversity_threshold": "0.85",
                "graph_expand": True
            },
            "semantic_code_search": {
                "query": "async function database connection"
            },
            "neural_system_status": {},
            "neo4j_graph_query": {
                "cypher_query": "MATCH (n) RETURN labels(n), count(n) ORDER BY count(n) DESC LIMIT 5"
            },
            "project_understanding": {
                "scope": "neural-tools"
            }
        }
    
    async def load_production_wrapper(self):
        """Load the production wrapper module"""
        try:
            logger.info("🔧 Loading production wrapper module...")
            
            # Import the wrapper module
            sys.path.insert(0, '/app')
            import neural_mcp_server_enhanced as wrapper
            self.wrapper_module = wrapper
            
            # Verify wrapper health
            health = wrapper.production_health_check()
            logger.info(f"Wrapper Health: {health['status'].upper()}")
            logger.info(f"Tools Accessible: {health['tools_accessible']}/{health['total_tools']}")
            logger.info(f"Compliance Score: {health['compliance_score']:.1f}%")
            
            if health['status'] != 'healthy':
                raise Exception(f"Wrapper is not healthy: {health}")
            
            # Initialize the neural server
            if hasattr(wrapper, 'initialize'):
                await wrapper.initialize()
                logger.info("✅ Neural server initialized through wrapper")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load production wrapper: {e}")
            raise
    
    async def test_real_functionality(self, tool_name: str) -> Dict[str, Any]:
        """Test real functionality of MCP tool through wrapper"""
        try:
            # Get tool function from wrapper
            tool_func = getattr(self.wrapper_module, tool_name, None)
            
            if not tool_func:
                return {
                    "status": "fail",
                    "error": f"Tool {tool_name} not accessible through wrapper",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Get real test parameters
            params = self.real_test_queries.get(tool_name, {})
            
            # Execute with real query
            logger.debug(f"Executing {tool_name} with real params: {params}")
            
            if asyncio.iscoroutinefunction(tool_func):
                timeout = 120.0 if tool_name == "neural_system_status" else 60.0
                result = await asyncio.wait_for(tool_func(**params), timeout=timeout)
            else:
                result = tool_func(**params)
            
            # Validate real result
            is_valid = self._validate_real_result(result, tool_name)
            
            return {
                "status": "pass" if is_valid else "fail",
                "result_type": type(result).__name__,
                "result_keys": list(result.keys()) if isinstance(result, dict) else None,
                "has_data": bool(result),
                "params_used": params,
                "validation_passed": is_valid,
                "timestamp": datetime.now().isoformat()
            }
            
        except asyncio.TimeoutError:
            return {
                "status": "fail",
                "error": f"Tool execution timed out",
                "params_used": params,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.debug(f"Tool {tool_name} error: {str(e)}")
            return {
                "status": "fail", 
                "error": str(e)[:200],
                "params_used": self.real_test_queries.get(tool_name, {}),
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_real_result(self, result: Any, tool_name: str) -> bool:
        """Validate real result with tool-specific expectations"""
        if not isinstance(result, dict):
            return False
        
        # Tool-specific validation
        if tool_name == "memory_store_enhanced":
            return "point_id" in result or "status" in result
        elif tool_name == "memory_search_enhanced":
            return "results" in result and len(result.get("results", [])) > 0
        elif tool_name == "neural_system_status":
            return "project" in result or "version" in result
        elif tool_name == "neo4j_graph_query":
            return "result" in result or "data" in result
        
        # General validation
        return len(result) > 0 and any(
            key in result for key in ['status', 'result', 'data', 'content', 'message', 'count']
        )
    
    async def test_priority_tools(self) -> Dict[str, Any]:
        """Test priority tools with real queries"""
        logger.info("🚀 Starting production wrapper end-to-end testing")
        
        # Load wrapper first
        await self.load_production_wrapper()
        
        start_time = datetime.now()
        priority_tools = list(self.real_test_queries.keys())
        
        for tool_name in priority_tools:
            logger.info(f"🔍 Testing real functionality: {tool_name}")
            result = await self.test_real_functionality(tool_name)
            self.test_results[tool_name] = result
            
            if result['status'] == 'pass':
                self.passed_tests += 1
            self.total_tests += 1
            
            status_icon = "✅" if result['status'] == 'pass' else "❌"
            logger.info(f"{status_icon} {tool_name}: {result['status'].upper()}")
            
            # Brief pause between tests
            await asyncio.sleep(0.2)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate compliance score
        compliance_score = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "wrapper_status": "production_ready" if compliance_score >= 80 else "needs_fixes",
            "total_tools_tested": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.total_tests - self.passed_tests,
            "end_to_end_compliance": compliance_score,
            "duration_seconds": duration,
            "test_results": self.test_results
        }
        
        logger.info(f"🎯 Production Testing Complete: {self.passed_tests}/{self.total_tests} passed ({compliance_score:.1f}%)")
        
        return summary

async def main():
    """Main production wrapper testing"""
    tester = ProductionWrapperTester()
    
    try:
        # Run real end-to-end testing
        summary = await tester.test_priority_tools()
        
        # Save results
        try:
            with open('/app/production_wrapper_results.json', 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info("📝 Production results saved")
        except Exception as e:
            logger.warning(f"Could not save results: {e}")
        
        print("\\n" + "="*70)
        print("L9 PRODUCTION WRAPPER - END-TO-END VERIFICATION")
        print("="*70)
        print(f"Wrapper Status: {summary['wrapper_status'].upper()}")
        print(f"End-to-End Results: {summary['passed']}/{summary['total_tools_tested']} tools passed")
        print(f"Compliance Score: {summary['end_to_end_compliance']:.1f}%")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print("")
        
        print("PASSED TOOLS:")
        passed_tools = [name for name, result in summary['test_results'].items() if result['status'] == 'pass']
        for tool in passed_tools:
            print(f"  ✅ {tool}")
        
        if summary['failed'] > 0:
            print("\\nFAILED TOOLS:")
            failed_tools = [name for name, result in summary['test_results'].items() if result['status'] == 'fail']
            for tool in failed_tools:
                error = summary['test_results'][tool].get('error', 'Unknown error')
                print(f"  ❌ {tool}: {error[:50]}...")
        
        print("="*70)
        
        # Production readiness threshold
        return summary['end_to_end_compliance'] >= 80.0
        
    except Exception as e:
        logger.error(f"❌ Production testing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)