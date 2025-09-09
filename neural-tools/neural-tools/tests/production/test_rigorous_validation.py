#!/usr/bin/env python3
"""
L9 RIGOROUS PRODUCTION VALIDATION - Objective Testing Framework
NO SUBJECTIVE INTERPRETATION - Binary pass/fail with concrete evidence
Prevents false production readiness claims through measurable validation

MANDATORY TESTS (ALL must pass):
T1 - Data Persistence: Store → Verify in DB  
T2 - Data Retrieval: Search → Verify results match
T3 - Graph Operations: Neo4j query → Structured data returned
T4 - Service Integration: All 3 databases operational
T5 - Performance: All operations <5 seconds

EXIT CONDITIONS:
- ALL 5 PASS = Production Ready
- ANY FAIL = NOT Production Ready  
- NO EXCEPTIONS
"""

import sys
import os
import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Environment setup for Docker
os.environ['PROJECT_NAME'] = 'default'
os.environ['QDRANT_URL'] = 'http://neural-data-storage:6333'
os.environ['NEO4J_URI'] = 'bolt://neo4j-graph:7687'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'neural-l9-2025'
os.environ['NOMIC_ENDPOINT'] = 'http://neural-embeddings:8000/embed'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RigorousValidator:
    """Objective production validation with binary pass/fail results"""
    
    def __init__(self):
        self.test_results = {}
        self.neural_module = None
        self.test_data = {
            "content": "L9 RIGOROUS TEST DATA - Neural System Production Validation Framework with objective measurements",
            "category": "validation",
            "test_id": f"rigorous_test_{int(time.time())}"
        }
        
    async def initialize_system(self):
        """Initialize neural system for testing"""
        try:
            logger.info("🔧 Initializing neural system...")
            sys.path.insert(0, '/app')
            
            # Import the wrapper module
            import neural_mcp_server_enhanced as neural
            self.neural_module = neural
            
            # Initialize the server
            if hasattr(neural, 'initialize'):
                await neural.initialize()
                logger.info("✅ Neural system initialized")
                return True
            else:
                logger.error("❌ No initialize function found")
                return False
                
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def test_t1_data_persistence(self) -> Dict[str, Any]:
        """T1 - DATA PERSISTENCE: Store data + verify in database"""
        test_start = time.time()
        logger.info("🔍 T1 - Testing data persistence...")
        
        try:
            # Step 1: Store data via memory_store_enhanced
            if not hasattr(self.neural_module, 'memory_store_enhanced'):
                return {
                    "test": "T1_DATA_PERSISTENCE",
                    "status": "FAIL",
                    "error": "memory_store_enhanced function not found",
                    "duration": time.time() - test_start
                }
            
            store_result = await self.neural_module.memory_store_enhanced(
                content=self.test_data["content"],
                category=self.test_data["category"],
                create_graph_entities=True
            )
            
            # Step 2: Verify point_id returned
            if not isinstance(store_result, dict) or "point_id" not in store_result:
                return {
                    "test": "T1_DATA_PERSISTENCE", 
                    "status": "FAIL",
                    "error": "No point_id returned from memory_store_enhanced",
                    "store_result": str(store_result),
                    "duration": time.time() - test_start
                }
            
            point_id = store_result["point_id"]
            
            # Step 3: Verify data exists in Qdrant database
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Get the point from Qdrant to verify it exists
                    response = await client.get(
                        f'http://neural-data-storage:6333/collections/project_default_code/points/{point_id}'
                    )
                    
                    if response.status_code == 200:
                        point_data = response.json()
                        # Verify the content exists in the payload
                        if "result" in point_data and "payload" in point_data["result"]:
                            payload = point_data["result"]["payload"]
                            if "content" in payload and self.test_data["content"] in payload["content"]:
                                return {
                                    "test": "T1_DATA_PERSISTENCE",
                                    "status": "PASS",
                                    "point_id": point_id,
                                    "verification": "Data confirmed in Qdrant database",
                                    "duration": time.time() - test_start
                                }
            except Exception as db_error:
                logger.debug(f"Database verification error: {db_error}")
            
            return {
                "test": "T1_DATA_PERSISTENCE",
                "status": "FAIL", 
                "error": "Could not verify data in Qdrant database",
                "point_id": point_id,
                "duration": time.time() - test_start
            }
            
        except Exception as e:
            return {
                "test": "T1_DATA_PERSISTENCE",
                "status": "FAIL",
                "error": str(e),
                "duration": time.time() - test_start
            }
    
    async def test_t2_data_retrieval(self) -> Dict[str, Any]:
        """T2 - DATA RETRIEVAL: Search data + verify results match"""
        test_start = time.time()
        logger.info("🔍 T2 - Testing data retrieval...")
        
        try:
            if not hasattr(self.neural_module, 'memory_search_enhanced'):
                return {
                    "test": "T2_DATA_RETRIEVAL",
                    "status": "FAIL",
                    "error": "memory_search_enhanced function not found",
                    "duration": time.time() - test_start
                }
            
            # Search for the data we stored
            search_result = await self.neural_module.memory_search_enhanced(
                query="L9 RIGOROUS TEST DATA Neural System Production",
                limit="3",
                mode="rrf_hybrid",
                diversity_threshold="0.85",
                graph_expand=True
            )
            
            # Verify results structure
            if not isinstance(search_result, dict) or "results" not in search_result:
                return {
                    "test": "T2_DATA_RETRIEVAL",
                    "status": "FAIL", 
                    "error": "No results returned from memory_search_enhanced",
                    "search_result": str(search_result),
                    "duration": time.time() - test_start
                }
            
            results = search_result["results"]
            if not results or len(results) == 0:
                return {
                    "test": "T2_DATA_RETRIEVAL",
                    "status": "FAIL",
                    "error": "Empty results from search",
                    "duration": time.time() - test_start
                }
            
            # Verify our test data is in the results
            found_match = False
            for result in results:
                if isinstance(result, dict) and "content" in result:
                    if "L9 RIGOROUS TEST DATA" in result["content"]:
                        found_match = True
                        break
            
            if found_match:
                return {
                    "test": "T2_DATA_RETRIEVAL",
                    "status": "PASS",
                    "results_count": len(results),
                    "verification": "Test data found in search results",
                    "duration": time.time() - test_start
                }
            else:
                return {
                    "test": "T2_DATA_RETRIEVAL", 
                    "status": "FAIL",
                    "error": "Test data not found in search results",
                    "results_count": len(results),
                    "duration": time.time() - test_start
                }
            
        except Exception as e:
            return {
                "test": "T2_DATA_RETRIEVAL",
                "status": "FAIL",
                "error": str(e),
                "duration": time.time() - test_start
            }
    
    async def test_t3_graph_operations(self) -> Dict[str, Any]:
        """T3 - GRAPH OPERATIONS: Neo4j query + structured data returned"""
        test_start = time.time()
        logger.info("🔍 T3 - Testing graph operations...")
        
        try:
            if not hasattr(self.neural_module, 'neo4j_graph_query'):
                return {
                    "test": "T3_GRAPH_OPERATIONS",
                    "status": "FAIL",
                    "error": "neo4j_graph_query function not found",
                    "duration": time.time() - test_start
                }
            
            # Execute Neo4j query
            graph_result = await self.neural_module.neo4j_graph_query(
                cypher_query="MATCH (n) RETURN labels(n) as node_labels, count(n) as node_count ORDER BY node_count DESC LIMIT 3"
            )
            
            # Verify structured data returned
            if not isinstance(graph_result, dict):
                return {
                    "test": "T3_GRAPH_OPERATIONS",
                    "status": "FAIL",
                    "error": "No structured data returned from Neo4j query",
                    "graph_result": str(graph_result),
                    "duration": time.time() - test_start
                }
            
            # Check for expected data structure
            if "result" in graph_result or "data" in graph_result or "records" in graph_result:
                return {
                    "test": "T3_GRAPH_OPERATIONS", 
                    "status": "PASS",
                    "verification": "Structured data returned from Neo4j",
                    "result_keys": list(graph_result.keys()),
                    "duration": time.time() - test_start
                }
            else:
                return {
                    "test": "T3_GRAPH_OPERATIONS",
                    "status": "FAIL",
                    "error": "No expected data structure in Neo4j result",
                    "result_keys": list(graph_result.keys()),
                    "duration": time.time() - test_start
                }
            
        except Exception as e:
            return {
                "test": "T3_GRAPH_OPERATIONS",
                "status": "FAIL",
                "error": str(e),
                "duration": time.time() - test_start
            }
    
    async def test_t4_service_integration(self) -> Dict[str, Any]:
        """T4 - SERVICE INTEGRATION: All 3 databases operational"""
        test_start = time.time()
        logger.info("🔍 T4 - Testing service integration...")
        
        services_status = {}
        
        try:
            import httpx
            from neo4j import GraphDatabase
            
            # Test Qdrant
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get('http://neural-data-storage:6333/collections')
                    services_status['qdrant'] = response.status_code == 200
            except Exception as e:
                services_status['qdrant'] = False
                logger.debug(f"Qdrant test failed: {e}")
            
            # Test Neo4j
            try:
                driver = GraphDatabase.driver('bolt://neo4j-graph:7687', auth=('neo4j', 'neural-l9-2025'))
                with driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    services_status['neo4j'] = result.single()['test'] == 1
                driver.close()
            except Exception as e:
                services_status['neo4j'] = False
                logger.debug(f"Neo4j test failed: {e}")
            
            # Test Embeddings Service
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get('http://neural-embeddings:8000/health')
                    services_status['embeddings'] = response.status_code in [200, 404]  # 404 is acceptable if no health endpoint
            except Exception as e:
                # Try alternative endpoint
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.post(
                            'http://neural-embeddings:8000/embed',
                            json={'text': 'test'},
                            timeout=5.0
                        )
                        # Any response (even error) means service is up
                        services_status['embeddings'] = response.status_code in [200, 422]
                except Exception:
                    services_status['embeddings'] = False
                    logger.debug(f"Embeddings test failed: {e}")
            
            # Calculate results
            operational_count = sum(services_status.values())
            total_services = len(services_status)
            
            if operational_count == total_services:
                return {
                    "test": "T4_SERVICE_INTEGRATION",
                    "status": "PASS",
                    "services_status": services_status,
                    "operational_count": f"{operational_count}/{total_services}",
                    "verification": "All services operational",
                    "duration": time.time() - test_start
                }
            else:
                return {
                    "test": "T4_SERVICE_INTEGRATION",
                    "status": "FAIL",
                    "services_status": services_status,
                    "operational_count": f"{operational_count}/{total_services}",
                    "error": "Not all services operational",
                    "duration": time.time() - test_start
                }
            
        except Exception as e:
            return {
                "test": "T4_SERVICE_INTEGRATION",
                "status": "FAIL",
                "error": str(e),
                "duration": time.time() - test_start
            }
    
    async def test_t5_performance(self, test_results: List[Dict]) -> Dict[str, Any]:
        """T5 - PERFORMANCE: All operations completed in <5 seconds"""
        logger.info("🔍 T5 - Testing performance...")
        
        failed_tests = []
        max_duration = 0
        
        for test_result in test_results:
            duration = test_result.get('duration', 0)
            max_duration = max(max_duration, duration)
            
            if duration >= 5.0:  # 5 second threshold
                failed_tests.append({
                    "test": test_result.get('test', 'unknown'),
                    "duration": duration
                })
        
        if not failed_tests:
            return {
                "test": "T5_PERFORMANCE",
                "status": "PASS",
                "max_duration": round(max_duration, 2),
                "verification": "All operations completed in <5 seconds",
                "threshold": "5.0 seconds"
            }
        else:
            return {
                "test": "T5_PERFORMANCE", 
                "status": "FAIL",
                "max_duration": round(max_duration, 2),
                "failed_tests": failed_tests,
                "error": f"{len(failed_tests)} operations exceeded 5 second threshold",
                "threshold": "5.0 seconds"
            }
    
    async def execute_rigorous_validation(self) -> Dict[str, Any]:
        """Execute all mandatory tests with binary pass/fail results"""
        logger.info("🚀 EXECUTING RIGOROUS PRODUCTION VALIDATION")
        logger.info("=" * 70)
        logger.info("OBJECTIVE TESTING - NO SUBJECTIVE INTERPRETATION")
        logger.info("ALL TESTS MUST PASS FOR PRODUCTION READINESS")
        logger.info("=" * 70)
        
        validation_start = time.time()
        
        # Initialize system
        if not await self.initialize_system():
            return {
                "validation_status": "FAILED",
                "error": "System initialization failed",
                "timestamp": datetime.now().isoformat()
            }
        
        # Execute mandatory tests
        logger.info("Executing mandatory tests...")
        test_results = []
        
        # T1 - Data Persistence  
        t1_result = await self.test_t1_data_persistence()
        test_results.append(t1_result)
        logger.info(f"T1 RESULT: {t1_result['status']}")
        
        # T2 - Data Retrieval
        t2_result = await self.test_t2_data_retrieval()
        test_results.append(t2_result)
        logger.info(f"T2 RESULT: {t2_result['status']}")
        
        # T3 - Graph Operations
        t3_result = await self.test_t3_graph_operations()
        test_results.append(t3_result)
        logger.info(f"T3 RESULT: {t3_result['status']}")
        
        # T4 - Service Integration
        t4_result = await self.test_t4_service_integration()
        test_results.append(t4_result)
        logger.info(f"T4 RESULT: {t4_result['status']}")
        
        # T5 - Performance
        t5_result = await self.test_t5_performance(test_results)
        test_results.append(t5_result)
        logger.info(f"T5 RESULT: {t5_result['status']}")
        
        # Calculate final results
        passed_tests = [t for t in test_results if t.get('status') == 'PASS']
        failed_tests = [t for t in test_results if t.get('status') == 'FAIL']
        
        total_duration = time.time() - validation_start
        
        # Binary determination: ALL tests must pass
        production_ready = len(failed_tests) == 0
        
        final_result = {
            "timestamp": datetime.now().isoformat(),
            "validation_duration": round(total_duration, 2),
            "production_ready": production_ready,
            "production_status": "READY" if production_ready else "NOT_READY",
            "tests_passed": len(passed_tests),
            "tests_failed": len(failed_tests),
            "total_tests": len(test_results),
            "test_results": test_results,
            "failed_tests": [t.get('test', 'unknown') for t in failed_tests] if failed_tests else [],
            "exit_condition": "ALL_TESTS_PASSED" if production_ready else "TESTS_FAILED"
        }
        
        # Save results
        try:
            with open('/app/rigorous_validation_results.json', 'w') as f:
                json.dump(final_result, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save results: {e}")
        
        return final_result

async def main():
    """Execute rigorous validation with objective results"""
    validator = RigorousValidator()
    
    try:
        results = await validator.execute_rigorous_validation()
        
        # Display results
        print("\n" + "="*80)
        print("L9 RIGOROUS PRODUCTION VALIDATION - OBJECTIVE RESULTS") 
        print("="*80)
        print(f"PRODUCTION STATUS: {results['production_status']}")
        print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']}")
        print(f"Validation Duration: {results['validation_duration']}s")
        print("")
        
        print("DETAILED TEST RESULTS:")
        for test_result in results['test_results']:
            status_icon = "✅" if test_result.get('status') == 'PASS' else "❌"
            test_name = test_result.get('test', 'UNKNOWN')
            duration = test_result.get('duration', 0)
            print(f"  {status_icon} {test_name}: {test_result.get('status', 'UNKNOWN')} ({duration:.2f}s)")
            
            if test_result.get('status') == 'FAIL' and 'error' in test_result:
                print(f"      Error: {test_result['error']}")
        
        print("")
        if results['production_ready']:
            print("🎯 OBJECTIVE VERDICT: PRODUCTION READY")
            print("   All mandatory tests passed with measurable evidence")
        else:
            print("❌ OBJECTIVE VERDICT: NOT PRODUCTION READY") 
            print(f"   Failed tests: {', '.join(results['failed_tests'])}")
        
        print("="*80)
        
        return results['production_ready']
        
    except Exception as e:
        logger.error(f"❌ Validation framework error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)