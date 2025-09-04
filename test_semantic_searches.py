#!/usr/bin/env python3
"""
Test semantic code search functionality with specific queries requested by user.
Tests the following searches:
1. "neural server MCP protocol" - to understand MCP implementation  
2. "SeaGOAT integration" - to see recent changes
3. "Docker configuration" - to understand container setup
4. "test base classes" - to find testing infrastructure
"""

import json
import asyncio
import subprocess
import sys
from typing import Dict, Any, List

class SemanticSearchTester:
    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process = None
        self.request_id = 1

    async def start_server(self):
        """Start the MCP server process."""
        self.process = await asyncio.create_subprocess_exec(
            *self.server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print(f"✓ Started MCP server")

    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send JSON-RPC request to MCP server."""
        if not self.process:
            raise RuntimeError("Server not started")

        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params:
            request["params"] = params

        self.request_id += 1
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read response with timeout
        try:
            response_line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10.0)
            if not response_line:
                raise RuntimeError("No response from server")
            return json.loads(response_line.decode().strip())
        except asyncio.TimeoutError:
            raise RuntimeError("Response timeout")

    async def initialize_server(self) -> bool:
        """Initialize the MCP server."""
        try:
            response = await self.send_request("initialize", {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "roots": {"listChanged": False},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "semantic-search-tester",
                    "version": "1.0.0"
                }
            })
            
            if "result" in response:
                print("✓ Server initialized successfully")
                # Send initialized notification
                await self.send_request("initialized", {})
                return True
            else:
                print(f"✗ Server initialization failed: {response}")
                return False
        except Exception as e:
            print(f"✗ Initialization error: {e}")
            return False

    async def test_semantic_search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Test semantic code search with a specific query."""
        try:
            print(f"\n=== Semantic Search: '{query}' ===")
            
            params = {
                "name": "semantic_code_search",
                "arguments": {
                    "query": query,
                    "limit": limit
                }
            }
            
            response = await self.send_request("tools/call", params)
            
            if "result" in response and "error" not in response:
                result = response["result"]
                print("✓ Search completed successfully")
                
                # Extract and display results
                if "content" in result:
                    for content in result["content"]:
                        if content["type"] == "text":
                            try:
                                data = json.loads(content["text"])
                                status = data.get("status", "unknown")
                                results = data.get("results", [])
                                total = data.get("total_found", 0)
                                
                                print(f"  Status: {status}")
                                print(f"  Results found: {total}")
                                
                                if results:
                                    print(f"  Top {min(3, len(results))} matches:")
                                    for i, match in enumerate(results[:3]):
                                        file_path = match.get("file_path", "unknown")
                                        score = match.get("score", 0.0)
                                        snippet = match.get("snippet", "")[:100] + "..." if len(match.get("snippet", "")) > 100 else match.get("snippet", "")
                                        print(f"    {i+1}. File: {file_path}")
                                        print(f"       Score: {score:.3f}")
                                        print(f"       Snippet: {snippet}")
                                
                                return {
                                    "success": True,
                                    "query": query,
                                    "status": status,
                                    "results_count": total,
                                    "results": results[:3],  # Keep top 3 for analysis
                                    "quality": self._assess_result_quality(query, results)
                                }
                            except json.JSONDecodeError as e:
                                print(f"  Raw response: {content['text'][:200]}...")
                                return {
                                    "success": False,
                                    "query": query,
                                    "error": "JSON decode error",
                                    "raw_response": content["text"]
                                }
                else:
                    print(f"  Unexpected result format: {result}")
                    return {
                        "success": False,
                        "query": query,
                        "error": "Unexpected result format",
                        "response": result
                    }
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error")
                print(f"✗ Search failed: {error_msg}")
                return {
                    "success": False,
                    "query": query,
                    "error": error_msg,
                    "response": response
                }
                
        except Exception as e:
            print(f"✗ Search error: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e)
            }

    def _assess_result_quality(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality and relevance of search results."""
        if not results:
            return {
                "score": 0.0,
                "assessment": "No results returned",
                "relevant_files": 0,
                "confidence": "low"
            }
        
        # Quality metrics
        total_results = len(results)
        high_confidence_results = sum(1 for r in results if r.get("score", 0.0) > 0.7)
        relevant_files = len(set(r.get("file_path", "") for r in results))
        
        # Query-specific relevance checks
        query_lower = query.lower()
        relevant_matches = 0
        
        for result in results:
            snippet = result.get("snippet", "").lower()
            file_path = result.get("file_path", "").lower()
            
            # Check if results contain query terms
            query_terms = query_lower.split()
            matches = sum(1 for term in query_terms if term in snippet or term in file_path)
            if matches > 0:
                relevant_matches += 1
        
        relevance_ratio = relevant_matches / total_results if total_results > 0 else 0
        
        # Overall quality score
        quality_score = (relevance_ratio + (high_confidence_results / total_results)) / 2
        
        # Confidence assessment
        if quality_score > 0.8 and relevant_files > 2:
            confidence = "high"
        elif quality_score > 0.5 and relevant_files > 1:
            confidence = "medium"
        else:
            confidence = "low"
        
        assessment = f"Found {total_results} results across {relevant_files} files, {relevant_matches} relevant matches"
        
        return {
            "score": quality_score,
            "assessment": assessment,
            "relevant_files": relevant_files,
            "confidence": confidence,
            "high_confidence_results": high_confidence_results,
            "relevance_ratio": relevance_ratio
        }

    async def cleanup(self):
        """Cleanup server process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            print("\n✓ Server process terminated")

async def main():
    """Main test function."""
    print("🔍 Semantic Code Search Test Suite")
    print("=" * 50)
    
    # Server command for Docker
    server_command = [
        "docker", "exec", "-i", "default-neural", 
        "python3", "-u", "/app/neural-tools-src/servers/neural_server_2025.py"
    ]
    
    tester = SemanticSearchTester(server_command)
    
    # Test queries requested by user
    test_queries = [
        "neural server MCP protocol",
        "SeaGOAT integration", 
        "Docker configuration",
        "test base classes"
    ]
    
    try:
        # Start server and initialize
        await tester.start_server()
        
        if not await tester.initialize_server():
            print("❌ Server initialization failed - aborting tests")
            return
        
        # Run all test queries
        results = {}
        
        for query in test_queries:
            result = await tester.test_semantic_search(query, limit=10)
            results[query] = result
            # Small delay between queries
            await asyncio.sleep(1.0)
        
        # Generate comprehensive report
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE SEARCH RESULTS ANALYSIS")
        print("=" * 70)
        
        overall_success = 0
        total_queries = len(test_queries)
        
        for query, result in results.items():
            print(f"\n🔍 Query: '{query}'")
            print("-" * 50)
            
            if result["success"]:
                overall_success += 1
                quality = result["quality"]
                print(f"✅ Status: SUCCESS")
                print(f"   Results Count: {result['results_count']}")
                print(f"   Quality Score: {quality['score']:.2f}")
                print(f"   Confidence: {quality['confidence'].upper()}")
                print(f"   Assessment: {quality['assessment']}")
                
                if result.get("results"):
                    print(f"   Top Results:")
                    for i, match in enumerate(result["results"]):
                        print(f"     {i+1}. {match.get('file_path', 'unknown')} (score: {match.get('score', 0):.3f})")
            else:
                print(f"❌ Status: FAILED")
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n📈 OVERALL ASSESSMENT:")
        print(f"   Successful Queries: {overall_success}/{total_queries}")
        success_rate = (overall_success / total_queries) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 75:
            print("🎉 SEMANTIC SEARCH SYSTEM: WORKING WELL")
            print("   - Search functionality is operational")
            print("   - Results are being returned with quality scoring")
            print("   - SeaGOAT integration appears successful")
        elif success_rate >= 50:
            print("⚠️  SEMANTIC SEARCH SYSTEM: PARTIALLY WORKING")  
            print("   - Some queries successful, some failing")
            print("   - May need optimization or debugging")
        else:
            print("❌ SEMANTIC SEARCH SYSTEM: NEEDS ATTENTION")
            print("   - Most queries are failing")
            print("   - Check SeaGOAT connection and indexing")
        
        # Codebase understanding assessment
        print(f"\n🏗️  CODEBASE STRUCTURE UNDERSTANDING:")
        
        successful_results = [r for r in results.values() if r["success"]]
        if successful_results:
            all_files = set()
            all_patterns = []
            
            for result in successful_results:
                for match in result.get("results", []):
                    if match.get("file_path"):
                        all_files.add(match["file_path"])
                    if match.get("snippet"):
                        all_patterns.append(match["snippet"][:50] + "...")
            
            print(f"   - Discovered {len(all_files)} unique files across searches")
            print(f"   - Found diverse code patterns and configurations")
            
            # Categorize discovered files
            file_categories = {
                "servers": [f for f in all_files if "server" in f.lower()],
                "tests": [f for f in all_files if "test" in f.lower()],
                "configs": [f for f in all_files if any(ext in f.lower() for ext in ["docker", "json", "yml", "yaml"])],
                "docs": [f for f in all_files if any(ext in f.lower() for ext in ["md", "txt", "doc"])]
            }
            
            for category, files in file_categories.items():
                if files:
                    print(f"   - {category.title()}: {len(files)} files")
                    
        else:
            print("   - Unable to analyze codebase structure due to search failures")
    
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())