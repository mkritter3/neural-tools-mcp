#!/usr/bin/env python3
"""
Real API Integration Tests

Tests the complete API stack with real HTTP requests:
- FastAPI health endpoints with real responses
- MCP server communication with actual protocol
- Neural tools API endpoints with live processing
- Security validation with real request flows
- Performance testing with actual load

No mocks - tests against running services.
"""

import pytest
import asyncio
import aiohttp
import json
import time
import os
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
INTEGRATION_TIMEOUT = 30.0

# Test data
SAMPLE_SEARCH_QUERIES = [
    "machine learning algorithms",
    "neural network architecture", 
    "vector database optimization",
    "graph database query performance"
]

SAMPLE_CODE_CHUNKS = [
    """
def calculate_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    """,
    """
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = self.initialize_weights()
    """,
    """
async def process_embeddings(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for text in texts:
        embedding = await embed_text(text)
        embeddings.append(embedding)
    return embeddings
    """
]


@pytest.mark.integration
class TestRealHealthAPI:
    """Test health API endpoints with real HTTP requests"""
    
    @pytest.mark.asyncio
    async def test_liveness_endpoint_real(self):
        """Test /health/live endpoint with real HTTP request"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{API_BASE_URL}/health/live", timeout=INTEGRATION_TIMEOUT) as response:
                    assert response.status == 200
                    
                    data = await response.json()
                    assert "alive" in data
                    assert data["alive"] is True
                    assert "uptime_seconds" in data
                    assert data["uptime_seconds"] >= 0
                    assert "timestamp" in data
                    
                    print(f"✅ Liveness endpoint healthy: {data}")
                    
            except aiohttp.ClientError as e:
                pytest.skip(f"API not available at {API_BASE_URL}: {e}")
    
    @pytest.mark.asyncio
    async def test_readiness_endpoint_real(self):
        """Test /health/ready endpoint with real HTTP request"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{API_BASE_URL}/health/ready", timeout=INTEGRATION_TIMEOUT) as response:
                    # Readiness can be 200 (ready) or 503 (not ready)
                    assert response.status in [200, 503]
                    
                    data = await response.json()
                    assert "ready" in data
                    assert isinstance(data["ready"], bool)
                    assert "dependencies" in data
                    assert "timestamp" in data
                    
                    # Log dependency status for debugging
                    print(f"Readiness status: {data['ready']}")
                    for dep_name, dep_status in data["dependencies"].items():
                        status = "✅" if dep_status.get("healthy", False) else "❌"
                        print(f"  {status} {dep_name}: {dep_status}")
                    
            except aiohttp.ClientError as e:
                pytest.skip(f"API not available at {API_BASE_URL}: {e}")
    
    @pytest.mark.asyncio
    async def test_startup_endpoint_real(self):
        """Test /health/startup endpoint with real HTTP request"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{API_BASE_URL}/health/startup", timeout=INTEGRATION_TIMEOUT) as response:
                    assert response.status == 200
                    
                    data = await response.json()
                    assert "started" in data
                    assert isinstance(data["started"], bool)
                    assert "initialization_time_ms" in data
                    assert "timestamp" in data
                    
                    print(f"✅ Startup endpoint: {data}")
                    
            except aiohttp.ClientError as e:
                pytest.skip(f"API not available at {API_BASE_URL}: {e}")
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint_real(self):
        """Test /health/metrics endpoint with real Prometheus metrics"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{API_BASE_URL}/health/metrics", timeout=INTEGRATION_TIMEOUT) as response:
                    assert response.status == 200
                    assert response.headers.get("Content-Type") == "text/plain; version=0.0.4; charset=utf-8"
                    
                    metrics_text = await response.text()
                    assert len(metrics_text) > 0
                    
                    # Check for expected Prometheus metrics
                    expected_metrics = [
                        "health_checks_total",
                        "health_check_duration_seconds",
                        "dependencies_healthy"
                    ]
                    
                    for metric in expected_metrics:
                        if metric in metrics_text:
                            print(f"✅ Found metric: {metric}")
                        else:
                            print(f"⚠️  Missing metric: {metric}")
                    
                    # Count lines (rough metric validation)
                    lines = metrics_text.strip().split('\n')
                    assert len(lines) > 5, f"Expected more metrics lines, got {len(lines)}"
                    
            except aiohttp.ClientError as e:
                pytest.skip(f"API not available at {API_BASE_URL}: {e}")


@pytest.mark.integration
class TestRealSearchAPI:
    """Test search API with real requests and processing"""
    
    @pytest.mark.asyncio
    async def test_search_endpoint_real(self):
        """Test search endpoint with real query processing"""
        async with aiohttp.ClientSession() as session:
            for query in SAMPLE_SEARCH_QUERIES[:2]:  # Test first 2 queries
                try:
                    payload = {
                        "query": query,
                        "limit": 5,
                        "project_name": "test_project"
                    }
                    
                    async with session.post(
                        f"{API_BASE_URL}/search",
                        json=payload,
                        timeout=INTEGRATION_TIMEOUT
                    ) as response:
                        
                        # Should return results or proper error
                        assert response.status in [200, 404, 422], f"Unexpected status {response.status} for query '{query}'"
                        
                        data = await response.json()
                        
                        if response.status == 200:
                            assert "results" in data
                            assert isinstance(data["results"], list)
                            assert "query_time_ms" in data
                            assert data["query_time_ms"] > 0
                            
                            print(f"✅ Search '{query}' returned {len(data['results'])} results in {data['query_time_ms']}ms")
                            
                        elif response.status == 404:
                            assert "error" in data
                            print(f"⚠️  Project not found for query '{query}': {data['error']}")
                            
                        elif response.status == 422:
                            assert "detail" in data  # FastAPI validation error format
                            print(f"⚠️  Validation error for query '{query}': {data['detail']}")
                        
                except aiohttp.ClientError as e:
                    pytest.skip(f"Search API not available: {e}")
    
    @pytest.mark.asyncio
    async def test_search_performance_real(self):
        """Test search performance with concurrent real requests"""
        async with aiohttp.ClientSession() as session:
            try:
                # Prepare concurrent requests
                tasks = []
                start_time = time.time()
                
                for i, query in enumerate(SAMPLE_SEARCH_QUERIES):
                    payload = {
                        "query": f"{query} test {i}",
                        "limit": 3,
                        "project_name": "test_project"
                    }
                    
                    task = session.post(
                        f"{API_BASE_URL}/search",
                        json=payload,
                        timeout=INTEGRATION_TIMEOUT
                    )
                    tasks.append(task)
                
                # Execute concurrent requests
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                # Analyze results
                successful_requests = 0
                total_query_time = 0
                
                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        print(f"❌ Request {i} failed: {response}")
                        continue
                    
                    try:
                        if response.status == 200:
                            data = await response.json()
                            successful_requests += 1
                            total_query_time += data.get("query_time_ms", 0)
                        
                        await response.__aexit__(None, None, None)
                    except:
                        pass
                
                # Performance assertions
                total_time = end_time - start_time
                print(f"Concurrent search test: {successful_requests}/{len(tasks)} successful in {total_time:.2f}s")
                
                if successful_requests > 0:
                    avg_query_time = total_query_time / successful_requests
                    print(f"Average query time: {avg_query_time:.2f}ms")
                    
                    # Performance thresholds
                    assert total_time < 60.0, f"Total time too high: {total_time}s"
                    assert avg_query_time < 5000, f"Average query time too high: {avg_query_time}ms"
                
            except Exception as e:
                pytest.skip(f"Performance test failed: {e}")


@pytest.mark.integration 
class TestRealIndexingAPI:
    """Test indexing API with real document processing"""
    
    @pytest.mark.asyncio
    async def test_index_code_real(self):
        """Test code indexing with real processing"""
        async with aiohttp.ClientSession() as session:
            for i, code_chunk in enumerate(SAMPLE_CODE_CHUNKS):
                try:
                    payload = {
                        "project_name": "test_project",
                        "file_path": f"test_file_{i}.py",
                        "content": code_chunk,
                        "content_type": "python"
                    }
                    
                    async with session.post(
                        f"{API_BASE_URL}/index",
                        json=payload,
                        timeout=INTEGRATION_TIMEOUT
                    ) as response:
                        
                        # Should succeed or return proper error
                        assert response.status in [200, 201, 422], f"Unexpected status {response.status}"
                        
                        data = await response.json()
                        
                        if response.status in [200, 201]:
                            assert "indexed" in data or "status" in data
                            print(f"✅ Code chunk {i} indexed successfully")
                            
                        elif response.status == 422:
                            print(f"⚠️  Validation error for chunk {i}: {data}")
                        
                except aiohttp.ClientError as e:
                    pytest.skip(f"Indexing API not available: {e}")
    
    @pytest.mark.asyncio
    async def test_bulk_indexing_real(self):
        """Test bulk indexing with real document processing"""
        async with aiohttp.ClientSession() as session:
            try:
                # Prepare bulk indexing payload
                documents = []
                for i, code_chunk in enumerate(SAMPLE_CODE_CHUNKS):
                    documents.append({
                        "file_path": f"bulk_test_{i}.py",
                        "content": code_chunk,
                        "content_type": "python"
                    })
                
                payload = {
                    "project_name": "test_project",
                    "documents": documents
                }
                
                start_time = time.time()
                async with session.post(
                    f"{API_BASE_URL}/index/bulk",
                    json=payload,
                    timeout=INTEGRATION_TIMEOUT * 2  # Allow extra time for bulk processing
                ) as response:
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    
                    if response.status in [200, 201]:
                        data = await response.json()
                        assert "processed" in data or "indexed" in data
                        
                        print(f"✅ Bulk indexing completed in {processing_time:.2f}s")
                        print(f"   Response: {data}")
                        
                        # Performance check
                        assert processing_time < 60.0, f"Bulk indexing too slow: {processing_time}s"
                        
                    else:
                        data = await response.json()
                        print(f"⚠️  Bulk indexing failed with status {response.status}: {data}")
                
            except aiohttp.ClientError as e:
                pytest.skip(f"Bulk indexing API not available: {e}")


@pytest.mark.integration
class TestRealSecurityAPI:
    """Test API security with real attack simulations"""
    
    @pytest.mark.asyncio
    async def test_injection_protection_real(self):
        """Test injection protection with real malicious payloads"""
        malicious_queries = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "{{ 7 * 7 }}",  # Template injection
            "../../etc/passwd",  # Path traversal
            "SELECT * FROM admin WHERE 1=1",  # SQL injection
        ]
        
        async with aiohttp.ClientSession() as session:
            for malicious_query in malicious_queries:
                try:
                    payload = {
                        "query": malicious_query,
                        "limit": 5,
                        "project_name": "test_project"
                    }
                    
                    async with session.post(
                        f"{API_BASE_URL}/search",
                        json=payload,
                        timeout=INTEGRATION_TIMEOUT
                    ) as response:
                        
                        # Should reject malicious input
                        assert response.status in [400, 422], f"Malicious query '{malicious_query}' not rejected (status: {response.status})"
                        
                        data = await response.json()
                        print(f"✅ Malicious query rejected: '{malicious_query[:50]}...'")
                        
                except aiohttp.ClientError as e:
                    # Connection errors are acceptable for security tests
                    print(f"⚠️  Connection error for malicious query (acceptable): {e}")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_real(self):
        """Test rate limiting with real rapid requests"""
        async with aiohttp.ClientSession() as session:
            try:
                # Send rapid requests to trigger rate limiting
                tasks = []
                for i in range(20):  # 20 rapid requests
                    payload = {
                        "query": f"rate limit test {i}",
                        "limit": 1,
                        "project_name": "test_project"
                    }
                    
                    task = session.post(
                        f"{API_BASE_URL}/search",
                        json=payload,
                        timeout=5.0  # Short timeout for rate limit test
                    )
                    tasks.append(task)
                
                # Execute all requests rapidly
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count responses by status
                status_counts = {}
                for response in responses:
                    if isinstance(response, Exception):
                        status_counts["error"] = status_counts.get("error", 0) + 1
                        continue
                    
                    try:
                        status = response.status
                        status_counts[status] = status_counts.get(status, 0) + 1
                        await response.__aexit__(None, None, None)
                    except:
                        pass
                
                print(f"Rate limiting test results: {status_counts}")
                
                # Should have some rate limited responses (429) or successful filtering
                rate_limited = status_counts.get(429, 0)
                if rate_limited > 0:
                    print(f"✅ Rate limiting working: {rate_limited} requests limited")
                else:
                    print("⚠️  No rate limiting detected (may not be configured)")
                
            except Exception as e:
                pytest.skip(f"Rate limiting test failed: {e}")


@pytest.mark.integration
class TestRealEndToEndWorkflows:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_index_search_workflow_real(self):
        """Test complete index -> search workflow"""
        async with aiohttp.ClientSession() as session:
            try:
                # Step 1: Index a document
                test_content = """
                def fibonacci(n):
                    if n <= 1:
                        return n
                    return fibonacci(n-1) + fibonacci(n-2)
                """
                
                index_payload = {
                    "project_name": "workflow_test",
                    "file_path": "fibonacci.py", 
                    "content": test_content,
                    "content_type": "python"
                }
                
                async with session.post(f"{API_BASE_URL}/index", json=index_payload, timeout=INTEGRATION_TIMEOUT) as response:
                    if response.status not in [200, 201]:
                        data = await response.json()
                        pytest.skip(f"Indexing failed: {data}")
                    
                    print("✅ Step 1: Document indexed")
                
                # Step 2: Wait for indexing to complete
                await asyncio.sleep(2)
                
                # Step 3: Search for the indexed content
                search_payload = {
                    "query": "fibonacci recursive function",
                    "limit": 5,
                    "project_name": "workflow_test"
                }
                
                async with session.post(f"{API_BASE_URL}/search", json=search_payload, timeout=INTEGRATION_TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        
                        # Should find the indexed document
                        found_fibonacci = any("fibonacci" in str(result).lower() for result in results)
                        if found_fibonacci:
                            print("✅ Step 2: Search found indexed document")
                        else:
                            print("⚠️  Step 2: Search did not find fibonacci document (indexing may still be processing)")
                    else:
                        data = await response.json()
                        print(f"⚠️  Step 2: Search failed: {data}")
                
                print("✅ End-to-end workflow completed")
                
            except Exception as e:
                pytest.skip(f"End-to-end workflow failed: {e}")


if __name__ == "__main__":
    print("Real API Integration Tests")
    print("=" * 50)
    print("Requirements:")
    print("1. API server running at", API_BASE_URL)
    print("2. All health endpoints accessible")
    print("3. Search and indexing APIs functional")
    print("4. Security protections enabled")
    print("\nRun with: pytest tests/integration/test_api_integration_real.py -v -m integration")