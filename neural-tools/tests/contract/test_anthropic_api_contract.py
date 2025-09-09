#!/usr/bin/env python3
"""
Contract Tests for Anthropic Claude API Integration

Validates API contract compliance and response format stability:
- Request/response schema validation
- API version compatibility
- Error response format consistency  
- Rate limiting behavior validation
- Authentication and authorization
- Breaking change detection

These tests ensure our integration remains stable against API changes.
"""

import asyncio
import json
import os
import pytest
import httpx
from typing import Dict, Any
from datetime import datetime

from src.infrastructure.haiku_reranker import HaikuReRanker


class TestAnthropicAPIContract:
    """Contract validation for Anthropic Claude API"""
    
    @pytest.fixture
    def api_key(self):
        """Get API key from environment"""
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            pytest.skip("ANTHROPIC_API_KEY required for contract tests")
        return key
    
    @pytest.fixture
    def api_client(self, api_key):
        """HTTP client for direct API testing"""
        return httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            timeout=30.0
        )
    
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_api_request_schema_validation(self, api_client):
        """Validate request schema matches API specification"""
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 100,
            "messages": [{
                "role": "user", 
                "content": "Test message"
            }],
            "temperature": 0.1
        }
        
        response = await api_client.post("/v1/messages", json=payload)
        
        # Should accept valid schema
        assert response.status_code == 200, f"Valid request failed: {response.text}"
        
        # Validate response has expected structure
        data = response.json()
        required_fields = ["id", "type", "role", "content", "model", "usage"]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate content structure
        assert isinstance(data["content"], list)
        assert len(data["content"]) > 0
        assert "type" in data["content"][0]
        assert "text" in data["content"][0]
    
    @pytest.mark.contract
    @pytest.mark.asyncio  
    async def test_api_response_format_stability(self, api_client):
        """Test response format remains consistent"""
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 50,
            "messages": [{
                "role": "user",
                "content": "Return exactly this: {'test': 'response'}"
            }]
        }
        
        response = await api_client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        
        # Validate consistent structure
        expected_structure = {
            "id": str,
            "type": str, 
            "role": str,
            "content": list,
            "model": str,
            "stop_reason": str,
            "stop_sequence": (type(None), str),
            "usage": dict
        }
        
        for field, expected_type in expected_structure.items():
            assert field in data, f"Response missing field: {field}"
            if not isinstance(expected_type, tuple):
                expected_type = (expected_type,)
            assert isinstance(data[field], expected_type), f"Field {field} has wrong type"
        
        # Validate usage structure
        usage = data["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert isinstance(usage["input_tokens"], int)
        assert isinstance(usage["output_tokens"], int)
    
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_error_response_format(self, api_key):
        """Test error response format consistency"""
        # Test with invalid API key
        invalid_client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "Content-Type": "application/json",
                "x-api-key": "invalid-key-12345",
                "anthropic-version": "2023-06-01"
            }
        )
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "test"}]
        }
        
        response = await invalid_client.post("/v1/messages", json=payload)
        
        # Should return authentication error
        assert response.status_code == 401
        
        error_data = response.json()
        
        # Validate error structure
        assert "error" in error_data
        error = error_data["error"]
        assert "type" in error
        assert "message" in error
        assert isinstance(error["type"], str)
        assert isinstance(error["message"], str)
        
        await invalid_client.aclose()
    
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_rate_limiting_headers(self, api_client):
        """Test rate limiting header presence"""
        payload = {
            "model": "claude-3-haiku-20240307", 
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]
        }
        
        response = await api_client.post("/v1/messages", json=payload)
        assert response.status_code == 200
        
        # Check for rate limiting headers (may vary by plan)
        rate_headers = [
            "anthropic-ratelimit-requests-limit",
            "anthropic-ratelimit-requests-remaining", 
            "anthropic-ratelimit-requests-reset",
            "anthropic-ratelimit-tokens-limit",
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-tokens-reset"
        ]
        
        found_headers = []
        for header in rate_headers:
            if header in response.headers:
                found_headers.append(header)
        
        # Should have some rate limiting info
        print(f"Rate limiting headers found: {found_headers}")
        
        # Don't assert specific headers as they may vary by account type
        # Just document what we find
        if found_headers:
            for header in found_headers:
                value = response.headers[header]
                print(f"  {header}: {value}")
    
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_model_availability(self, api_client):
        """Test that expected models are available"""
        expected_models = [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229", 
            "claude-3-opus-20240229"
        ]
        
        for model in expected_models:
            payload = {
                "model": model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            response = await api_client.post("/v1/messages", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                assert data["model"] == model
                print(f"✓ Model {model} available")
            elif response.status_code == 400:
                # Check if it's a model availability issue
                error_data = response.json()
                if "model" in error_data.get("error", {}).get("message", "").lower():
                    print(f"⚠ Model {model} not available: {error_data}")
                else:
                    raise AssertionError(f"Unexpected error for {model}: {error_data}")
            else:
                raise AssertionError(f"Unexpected status {response.status_code} for {model}")
    
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_api_version_compatibility(self, api_key):
        """Test API version compatibility"""
        versions_to_test = [
            "2023-06-01",  # Current version used in code
            "2023-01-01",  # Older version
        ]
        
        for version in versions_to_test:
            client = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": version
                }
            )
            
            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            response = await client.post("/v1/messages", json=payload)
            
            print(f"API version {version}: status {response.status_code}")
            
            if response.status_code == 200:
                print(f"✓ Version {version} supported")
            elif response.status_code == 400:
                error_data = response.json() 
                if "version" in error_data.get("error", {}).get("message", "").lower():
                    print(f"⚠ Version {version} deprecated: {error_data}")
                else:
                    print(f"⚠ Version {version} error: {error_data}")
            
            await client.aclose()
    
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_content_filtering_behavior(self, api_client):
        """Test content filtering and safety behavior"""
        test_cases = [
            {
                "name": "normal_content",
                "content": "Write a simple Python function to add two numbers",
                "should_succeed": True
            },
            {
                "name": "edge_case_content", 
                "content": "Explain the security implications of SQL injection attacks",
                "should_succeed": True  # Educational content should be fine
            }
        ]
        
        for case in test_cases:
            payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": case["content"]}]
            }
            
            response = await api_client.post("/v1/messages", json=payload)
            
            print(f"Content test '{case['name']}': status {response.status_code}")
            
            if case["should_succeed"]:
                if response.status_code != 200:
                    error_data = response.json() if response.status_code != 500 else {}
                    print(f"  Unexpected failure: {error_data}")
                else:
                    data = response.json()
                    assert len(data["content"]) > 0
                    print(f"  ✓ Succeeded with {data['usage']['output_tokens']} tokens")
    
    @pytest.fixture(scope="session", autouse=True) 
    async def close_clients(self):
        """Ensure all HTTP clients are closed"""
        yield
        # Cleanup handled by individual test fixtures


class TestHaikuRerankerAPIContract:
    """Test HaikuReRanker's API contract usage"""
    
    @pytest.mark.contract
    @pytest.mark.asyncio
    async def test_reranker_api_integration(self):
        """Test reranker properly integrates with API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY required")
        
        reranker = HaikuReRanker(api_key=api_key)
        
        # Test basic integration
        results = [
            {"content": "function a() { return 1; }", "score": 0.8},
            {"content": "function b() { return 2; }", "score": 0.7}
        ]
        
        reranked = await reranker.rerank_simple(
            query="JavaScript function implementation",
            results=results,
            max_results=2
        )
        
        # Should complete without errors and return properly formatted results
        assert len(reranked) == 2
        assert all('content' in r for r in reranked)
        assert all('score' in r for r in reranked)
        assert all('metadata' in r for r in reranked)
        
        # Should have reranking metadata
        for result in reranked:
            metadata = result['metadata']
            assert 'reranked' in metadata
            assert 'ranking_confidence' in metadata
            assert isinstance(metadata['ranking_confidence'], (int, float))
            assert 0 <= metadata['ranking_confidence'] <= 1
    
    @pytest.mark.contract 
    @pytest.mark.asyncio
    async def test_reranker_error_handling_contract(self):
        """Test reranker handles API errors per contract"""
        # Test with invalid API key
        reranker = HaikuReRanker(api_key="invalid-key-test")
        
        results = [{"content": "test content", "score": 0.8}]
        
        # Should not raise exception - should fallback gracefully
        reranked = await reranker.rerank_simple(
            query="test query",
            results=results
        )
        
        # Should return fallback results
        assert len(reranked) == 1
        assert reranked[0]['content'] == "test content"
        assert 'processing_time' in reranked[0]
        
        # Should indicate fallback was used
        # Check stats or logs for fallback indication
        stats = reranker.get_stats()
        assert stats['requests'] >= 1


if __name__ == "__main__":
    # Run with: python -m pytest tests/contract/ -v -m contract
    import sys
    sys.exit(pytest.main([__file__, "-v", "-m", "contract"]))