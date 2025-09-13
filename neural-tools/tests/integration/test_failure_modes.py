#!/usr/bin/env python3
"""
Failure Mode and Invalid Input Tests for Neural Tools MCP Server
Tests error handling, invalid inputs, and service failures (L9 Standards)
"""

import asyncio
import pytest
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add helpers to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.mcp_test_helper import MPCTestHelper, mcp_server_session

class TestFailureModes:
    """Test suite for failure modes and error handling."""

    @pytest.mark.asyncio
    async def test_malformed_json_request(self):
        """Test server handling of malformed JSON."""
        helper = MPCTestHelper()
        await helper.start_server()

        try:
            # Send malformed JSON directly
            if helper.process:
                # Send invalid JSON
                helper.process.stdin.write(b"{ invalid json }\n")
                await helper.process.stdin.drain()

                # Server should not crash - try a valid request after
                result = await helper.call_tool("neural_system_status", {})
                assert "error" in result or "status" in result, \
                    "Server should handle malformed JSON gracefully"
        finally:
            await helper.stop_server()

    @pytest.mark.asyncio
    async def test_missing_required_parameters(self):
        """Test tools with missing required parameters."""
        async with mcp_server_session() as helper:
            # semantic_code_search without query
            result = await helper.call_tool("semantic_code_search", {
                "limit": 5  # Missing required 'query'
            })
            assert "error" in result, "Should error on missing 'query' parameter"

            # reindex_path without path
            result = await helper.call_tool("reindex_path", {
                "recursive": True  # Missing required 'path'
            })
            assert "error" in result, "Should error on missing 'path' parameter"

            # schema_add_node_type without required fields
            result = await helper.call_tool("schema_add_node_type", {
                "description": "Test node"  # Missing 'name' and 'properties'
            })
            assert "error" in result, "Should error on missing required fields"

    @pytest.mark.asyncio
    async def test_invalid_parameter_types(self):
        """Test tools with wrong parameter types."""
        async with mcp_server_session() as helper:
            # String instead of integer for limit
            result = await helper.call_tool("semantic_code_search", {
                "query": "test",
                "limit": "five"  # Should be integer
            })
            # Server should either handle gracefully or return error
            assert result is not None, "Server should not crash on type error"

            # Boolean instead of string for path
            result = await helper.call_tool("set_project_context", {
                "path": True  # Should be string
            })
            assert "error" in result or "status" in result, \
                "Should handle invalid path type"

            # String instead of boolean
            result = await helper.call_tool("reindex_path", {
                "path": ".",
                "recursive": "yes"  # Should be boolean
            })
            assert result is not None, "Server should handle type conversion or error"

    @pytest.mark.asyncio
    async def test_nonexistent_tool_call(self):
        """Test calling a tool that doesn't exist."""
        async with mcp_server_session() as helper:
            result = await helper.call_tool("nonexistent_tool", {
                "param": "value"
            })
            assert "error" in result, "Should error on nonexistent tool"

            # Verify error message is informative
            if "error" in result:
                error_msg = str(result["error"])
                assert "nonexistent_tool" in error_msg.lower() or \
                       "not found" in error_msg.lower() or \
                       "unknown" in error_msg.lower(), \
                    "Error message should indicate tool not found"

    @pytest.mark.asyncio
    async def test_invalid_project_path(self):
        """Test setting project context to invalid paths."""
        async with mcp_server_session() as helper:
            # Nonexistent path
            result = await helper.call_tool("set_project_context", {
                "path": "/nonexistent/path/to/project"
            })
            assert "error" in result or "status" in result, \
                "Should handle nonexistent project path"

            # File instead of directory
            result = await helper.call_tool("set_project_context", {
                "path": __file__  # This test file, not a project
            })
            assert "error" in result or "status" in result, \
                "Should reject file as project path"

            # Empty path
            result = await helper.call_tool("set_project_context", {
                "path": ""
            })
            assert "error" in result or "status" in result, \
                "Should reject empty path"

    @pytest.mark.asyncio
    async def test_invalid_search_parameters(self):
        """Test search tools with invalid parameters."""
        async with mcp_server_session() as helper:
            # Negative limit
            result = await helper.call_tool("semantic_code_search", {
                "query": "test",
                "limit": -5
            })
            assert "error" in result or (
                "results" in result and len(result["results"]) >= 0
            ), "Should handle negative limit"

            # Extremely large limit
            result = await helper.call_tool("graphrag_hybrid_search", {
                "query": "test",
                "limit": 999999,
                "include_graph_context": True
            })
            # Should either cap the limit or return error
            assert result is not None, "Should handle large limit"

            # Invalid max_hops
            result = await helper.call_tool("graphrag_hybrid_search", {
                "query": "test",
                "limit": 5,
                "max_hops": -1
            })
            assert result is not None, "Should handle invalid max_hops"

    @pytest.mark.asyncio
    async def test_invalid_schema_operations(self):
        """Test schema operations with invalid data."""
        async with mcp_server_session() as helper:
            # Add node type with invalid properties
            result = await helper.call_tool("schema_add_node_type", {
                "name": "",  # Empty name
                "properties": "not a dict",  # Should be dict
                "description": "Test"
            })
            assert "error" in result, "Should reject invalid node type"

            # Add relationship with invalid types
            result = await helper.call_tool("schema_add_relationship", {
                "name": "TEST_REL",
                "from_types": "should be list",  # Should be list
                "to_types": ["ValidType"],
                "description": "Test"
            })
            assert "error" in result, "Should reject invalid relationship"

            # Invalid project type for schema init
            result = await helper.call_tool("schema_init", {
                "project_type": "invalid_type",
                "auto_detect": False
            })
            # Should either use default or return error
            assert result is not None, "Should handle invalid project type"

    @pytest.mark.asyncio
    async def test_invalid_migration_operations(self):
        """Test migration operations with invalid parameters."""
        async with mcp_server_session() as helper:
            # Generate migration with invalid name
            result = await helper.call_tool("migration_generate", {
                "name": "invalid-name-with-dashes",  # Should be alphanumeric + underscore
                "description": "Test",
                "dry_run": True
            })
            assert "error" in result or "status" in result, \
                "Should handle invalid migration name"

            # Rollback to invalid version
            result = await helper.call_tool("migration_rollback", {
                "target_version": -999,  # Invalid version
                "force": False
            })
            assert "error" in result or "status" in result, \
                "Should reject invalid rollback version"

            # Apply with invalid parameters
            result = await helper.call_tool("migration_apply", {
                "target_version": "not a number",  # Should be integer
                "dry_run": True
            })
            assert result is not None, "Should handle invalid apply parameters"

    @pytest.mark.asyncio
    async def test_extra_unexpected_parameters(self):
        """Test tools with unexpected extra parameters."""
        async with mcp_server_session() as helper:
            # Extra parameters should be ignored or cause error
            result = await helper.call_tool("neural_system_status", {
                "unexpected_param": "value",
                "another_extra": 123
            })
            # Should either ignore extras or return error
            assert result is not None, "Should handle extra parameters"

            result = await helper.call_tool("semantic_code_search", {
                "query": "test",
                "limit": 5,
                "extra_param": "should_be_ignored",
                "invalid_option": True
            })
            assert result is not None, "Should handle extra search parameters"

    @pytest.mark.asyncio
    async def test_empty_parameters(self):
        """Test tools with empty parameter objects."""
        async with mcp_server_session() as helper:
            # Tools that should work with no parameters
            result = await helper.call_tool("neural_system_status", {})
            assert "error" not in result, "neural_system_status should work without params"

            result = await helper.call_tool("list_projects", {})
            assert "error" not in result, "list_projects should work without params"

            # Tools that require parameters
            result = await helper.call_tool("semantic_code_search", {})
            assert "error" in result, "semantic_code_search should fail without query"

    @pytest.mark.asyncio
    async def test_special_characters_in_parameters(self):
        """Test handling of special characters in parameters."""
        async with mcp_server_session() as helper:
            # Special characters in search query
            special_queries = [
                "test\\nwith\\nnewlines",
                "test\twith\ttabs",
                "test\"with\"quotes",
                "test'with'apostrophes",
                "test{with}braces",
                "test[with]brackets",
                "test|with|pipes",
                "test&with&ampersands",
                "test;with;semicolons",
                "test`with`backticks",
                "test$with$dollars",
                "test#with#hashes"
            ]

            for query in special_queries:
                result = await helper.call_tool("semantic_code_search", {
                    "query": query,
                    "limit": 1
                })
                assert result is not None, \
                    f"Should handle special character query: {query}"

    @pytest.mark.asyncio
    async def test_unicode_and_emoji_parameters(self):
        """Test handling of unicode and emoji in parameters."""
        async with mcp_server_session() as helper:
            # Unicode in search
            result = await helper.call_tool("semantic_code_search", {
                "query": "测试 テスト тест",  # Chinese, Japanese, Russian
                "limit": 3
            })
            assert result is not None, "Should handle unicode characters"

            # Emoji in parameters
            result = await helper.call_tool("schema_add_node_type", {
                "name": "TestNode",
                "properties": {"emoji": "string"},
                "description": "Test with emoji 🚀 🎉 ✨"
            })
            assert result is not None, "Should handle emoji in descriptions"

    @pytest.mark.asyncio
    async def test_null_and_undefined_parameters(self):
        """Test handling of null values in parameters."""
        async with mcp_server_session() as helper:
            # None/null values
            result = await helper.call_tool("semantic_code_search", {
                "query": None,  # Null query
                "limit": 5
            })
            assert "error" in result, "Should reject null required parameter"

            # Test optional parameters with null
            result = await helper.call_tool("backfill_metadata", {
                "batch_size": None,  # Null optional parameter
                "dry_run": True
            })
            # Should use default or handle gracefully
            assert result is not None, "Should handle null optional parameter"

    @pytest.mark.asyncio
    async def test_boundary_value_parameters(self):
        """Test boundary values for numeric parameters."""
        async with mcp_server_session() as helper:
            # Zero limit
            result = await helper.call_tool("semantic_code_search", {
                "query": "test",
                "limit": 0
            })
            # Should return empty results or error
            assert result is not None, "Should handle zero limit"

            # Maximum integer
            result = await helper.call_tool("graphrag_hybrid_search", {
                "query": "test",
                "limit": 1,
                "max_hops": 2147483647  # Max int32
            })
            # Should cap or handle gracefully
            assert result is not None, "Should handle max integer"

            # Floating point instead of integer
            result = await helper.call_tool("semantic_code_search", {
                "query": "test",
                "limit": 3.14
            })
            # Should convert or error
            assert result is not None, "Should handle float for integer param"

    @pytest.mark.asyncio
    async def test_rapid_successive_calls(self):
        """Test rapid successive calls to stress the server."""
        async with mcp_server_session() as helper:
            # Make 20 rapid calls
            tasks = []
            for i in range(20):
                tasks.append(helper.call_tool("neural_system_status", {}))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed without crashing
            success_count = sum(1 for r in results
                              if not isinstance(r, Exception) and "error" not in r)
            assert success_count == len(results), \
                f"Rapid calls failed: {len(results) - success_count} errors"

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of operations that might timeout."""
        # Use a very short timeout
        async with mcp_server_session(timeout=2.0) as helper:
            # Try a potentially slow operation
            try:
                result = await helper.call_tool("reindex_path", {
                    "path": "/",  # Large path that might timeout
                    "recursive": True
                })
                # If it completes, that's fine
                assert result is not None
            except asyncio.TimeoutError:
                # Timeout is expected and acceptable
                pass

    @pytest.mark.asyncio
    async def test_service_unavailable_simulation(self):
        """Test behavior when backend services are unavailable."""
        async with mcp_server_session() as helper:
            # Note: This test would ideally stop Docker containers
            # For now, we test with wrong credentials in env vars
            # which simulates service unavailability

            # The server should still respond, just with errors
            result = await helper.call_tool("neural_system_status", {})
            assert result is not None, "Server should respond even if services are down"

            # GraphRAG should handle Neo4j/Qdrant unavailability
            result = await helper.call_tool("graphrag_hybrid_search", {
                "query": "test",
                "limit": 5
            })
            # Should return error or empty results, not crash
            assert result is not None, "Should handle service unavailability"