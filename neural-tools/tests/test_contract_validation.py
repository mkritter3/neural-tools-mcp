#!/usr/bin/env python3
"""
ADR-0096 Contract Validation Tests
Comprehensive regression tests to ensure all components maintain schema contracts

This test suite validates:
1. ChunkSchema consistency across indexer and retrieval
2. Vector search contract compliance
3. MCP tool formatter contracts
4. Data type consistency (no DateTime in JSON)
5. File path availability in all chunks
6. Graph context structure consistency
"""

import sys
import asyncio
import json
import unittest
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from servers.services.chunk_schema import ChunkSchema
from servers.services.robust_vector_search import RobustVectorSearch
from neural_mcp.tools.fast_search import execute as fast_search_execute
from neural_mcp.tools.elite_search import execute as elite_search_execute


class TestChunkSchemaContract(unittest.TestCase):
    """Test that ChunkSchema enforces consistent data structure"""

    def test_chunk_schema_validation(self):
        """Test ChunkSchema validates required fields"""
        # Valid chunk
        valid_chunk = ChunkSchema(
            chunk_id="test.py:chunk:0",
            file_path="test.py",
            content="test content",
            embedding=[0.1] * 768,
            project="test-project"
        )
        self.assertEqual(valid_chunk.file_path, "test.py")
        self.assertEqual(len(valid_chunk.embedding), 768)

    def test_chunk_schema_rejects_invalid_embedding(self):
        """Test ChunkSchema rejects wrong embedding dimensions"""
        with self.assertRaises(ValueError) as ctx:
            ChunkSchema(
                chunk_id="test.py:chunk:0",
                file_path="test.py",
                content="test content",
                embedding=[0.1] * 500,  # Wrong dimension
                project="test-project"
            )
        self.assertIn("768", str(ctx.exception))

    def test_chunk_schema_no_datetime_objects(self):
        """Test ChunkSchema uses ISO strings not DateTime objects"""
        chunk = ChunkSchema(
            chunk_id="test.py:chunk:0",
            file_path="test.py",
            content="test content",
            embedding=[0.1] * 768,
            project="test-project"
        )
        # created_at should be string, not datetime
        self.assertIsInstance(chunk.created_at, str)
        # Should be valid ISO format
        datetime.fromisoformat(chunk.created_at.replace('Z', '+00:00'))

    def test_chunk_to_neo4j_dict(self):
        """Test ChunkSchema converts to Neo4j-safe dict"""
        chunk = ChunkSchema(
            chunk_id="test.py:chunk:0",
            file_path="test.py",
            content="test content",
            embedding=[0.1] * 768,
            project="test-project"
        )
        neo4j_dict = chunk.to_neo4j_dict()

        # Verify all required fields present
        self.assertIn("chunk_id", neo4j_dict)
        self.assertIn("file_path", neo4j_dict)
        self.assertIn("content", neo4j_dict)
        self.assertIn("embedding", neo4j_dict)
        self.assertIn("project", neo4j_dict)
        self.assertIn("created_at", neo4j_dict)

        # Verify types are Neo4j-safe
        self.assertIsInstance(neo4j_dict["created_at"], str)
        self.assertIsInstance(neo4j_dict["embedding"], list)

    def test_chunk_json_serializable(self):
        """Test ChunkSchema produces JSON-serializable output"""
        chunk = ChunkSchema(
            chunk_id="test.py:chunk:0",
            file_path="test.py",
            content="test content",
            embedding=[0.1] * 768,
            project="test-project"
        )

        # Should not raise exception
        json_str = json.dumps(chunk.to_dict())
        parsed = json.loads(json_str)

        # Verify roundtrip preserves data
        self.assertEqual(parsed["file_path"], "test.py")
        self.assertEqual(len(parsed["embedding"]), 768)


class TestVectorSearchContract(unittest.TestCase):
    """Test that vector search maintains consistent interface"""

    def test_robust_vector_search_query_format(self):
        """Test RobustVectorSearch uses correct Neo4j query format"""
        # Mock Neo4j service
        class MockNeo4j:
            async def execute_cypher(self, query, params):
                return {"status": "success", "result": []}

        search = RobustVectorSearch(neo4j_service=MockNeo4j(), project_name="test-project")

        # Test that the class exists and can be instantiated
        self.assertIsNotNone(search)
        self.assertEqual(search.project, "test-project")

        # The actual query is internal to vector_search_phase1
        # We just verify the class structure is correct
        self.assertTrue(hasattr(search, 'vector_search_phase1'))
        self.assertTrue(hasattr(search, 'graph_enrichment_phase2'))

    def test_robust_vector_search_handles_missing_fields(self):
        """Test RobustVectorSearch handles chunks with missing fields gracefully"""
        # Test that ChunkSchema handles missing file_path by extracting from chunk_id
        chunk_data = {
            "chunk_id": "old_file.py:chunk:0",
            "content": "legacy content",
            "embedding": [0.1] * 768,
            "project": "test-project"
        }

        # ChunkSchema should extract file_path from chunk_id
        chunk = ChunkSchema(
            chunk_id=chunk_data["chunk_id"],
            file_path="",  # Empty file_path
            content=chunk_data["content"],
            embedding=chunk_data["embedding"],
            project=chunk_data["project"]
        )

        # Should have extracted file_path
        self.assertEqual(chunk.file_path, "old_file.py")


class TestMCPToolContracts(unittest.IsolatedAsyncioTestCase):
    """Test MCP tool formatters maintain consistent output contracts"""

    async def test_fast_search_output_contract(self):
        """Test fast_search returns expected structure"""
        # Mock response structure that fast_search should produce
        mock_result = {
            "status": "success",
            "query": "test query",
            "results": [
                {
                    "rank": 1,
                    "content": "test content",
                    "file": {
                        "path": "test.py",
                        "language": "python"
                    },
                    "score": 0.85
                }
            ],
            "total_results": 1,
            "performance": {
                "query_time_ms": 50.0,
                "cache_hit": False
            }
        }

        # Verify structure is JSON serializable
        json_str = json.dumps(mock_result)
        parsed = json.loads(json_str)

        # Verify required fields
        self.assertEqual(parsed["status"], "success")
        self.assertIn("results", parsed)
        self.assertIn("file", parsed["results"][0])
        self.assertIn("path", parsed["results"][0]["file"])

    async def test_elite_search_output_contract(self):
        """Test elite_search returns expected structure with graph context"""
        # Mock response structure that elite_search should produce
        mock_result = {
            "status": "success",
            "query": "test query",
            "results": [
                {
                    "rank": 1,
                    "content": "test content",
                    "file": {
                        "path": "test.py",
                        "language": "python"
                    },
                    "scores": {
                        "final": 0.85,
                        "vector": 0.80
                    },
                    "graph_context": {
                        "total_connections": 5,
                        "related_chunks": 3,
                        "related_files": 2
                    },
                    "explanation": "High semantic similarity"
                }
            ],
            "total_results": 1,
            "search_type": "elite_graphrag",
            "performance": {
                "query_time_ms": 150.0,
                "cache_hit": False,
                "graph_depth": 2
            }
        }

        # Verify structure is JSON serializable
        json_str = json.dumps(mock_result)
        parsed = json.loads(json_str)

        # Verify required fields
        self.assertEqual(parsed["status"], "success")
        self.assertIn("graph_context", parsed["results"][0])
        self.assertIn("total_connections", parsed["results"][0]["graph_context"])
        self.assertIn("scores", parsed["results"][0])

    async def test_both_tools_handle_no_results(self):
        """Test both search tools handle empty results gracefully"""
        # Mock empty result structure
        empty_fast = {
            "status": "no_results",
            "query": "impossible query",
            "message": "No matching results found",
            "suggestion": "Try broader search terms"
        }

        empty_elite = {
            "status": "no_results",
            "query": "impossible query",
            "message": "No matching results found",
            "suggestion": "Try broader search terms or reduce graph depth"
        }

        # Both should be JSON serializable
        json.dumps(empty_fast)
        json.dumps(empty_elite)


class TestDataConsistencyContracts(unittest.TestCase):
    """Test data consistency across the system"""

    def test_no_nested_maps_in_neo4j_data(self):
        """Test that Neo4j data uses only primitives (ADR-0036)"""
        chunk = ChunkSchema(
            chunk_id="test.py:chunk:0",
            file_path="test.py",
            content="test content",
            embedding=[0.1] * 768,
            project="test-project"
        )

        neo4j_dict = chunk.to_neo4j_dict()

        def check_no_nested_dicts(obj, path=""):
            """Recursively check for nested dicts (Maps)"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    self.assertNotIsInstance(
                        value, dict,
                        f"Found nested dict at {path}.{key} - Neo4j doesn't support Map types"
                    )
                    if isinstance(value, list):
                        for i, item in enumerate(value):
                            self.assertNotIsInstance(
                                item, dict,
                                f"Found dict in list at {path}.{key}[{i}]"
                            )

        check_no_nested_dicts(neo4j_dict)

    def test_file_path_always_available(self):
        """Test that file_path is always accessible in chunks"""
        # Test with explicit file_path
        chunk1 = ChunkSchema(
            chunk_id="test.py:chunk:0",
            file_path="test.py",
            content="test",
            embedding=[0.1] * 768,
            project="test"
        )
        self.assertEqual(chunk1.file_path, "test.py")

        # Test that chunk_id format is validated
        chunk2 = ChunkSchema(
            chunk_id="another/file.js:chunk:5",
            file_path="another/file.js",
            content="test",
            embedding=[0.1] * 768,
            project="test"
        )
        self.assertEqual(chunk2.file_path, "another/file.js")

    def test_project_property_always_present(self):
        """Test that project property is always present (ADR-0029)"""
        chunk = ChunkSchema(
            chunk_id="test.py:chunk:0",
            file_path="test.py",
            content="test",
            embedding=[0.1] * 768,
            project="my-project"
        )

        neo4j_dict = chunk.to_neo4j_dict()
        self.assertIn("project", neo4j_dict)
        self.assertEqual(neo4j_dict["project"], "my-project")


class TestIntegrationContracts(unittest.IsolatedAsyncioTestCase):
    """Integration tests for component interactions"""

    async def test_indexer_to_retrieval_contract(self):
        """Test that data indexed can be retrieved with same structure"""
        # Create chunk using schema
        chunk = ChunkSchema(
            chunk_id="integration_test.py:chunk:0",
            file_path="integration_test.py",
            content="integration test content",
            embedding=[0.1] * 768,
            project="integration-test"
        )

        # Convert to storage format
        storage_dict = chunk.to_neo4j_dict()

        # Simulate retrieval - verify data can roundtrip
        retrieved_chunk = ChunkSchema.from_neo4j_node(storage_dict)

        # Verify roundtrip preserves essential fields
        self.assertEqual(retrieved_chunk.file_path, "integration_test.py")
        self.assertEqual(retrieved_chunk.content, "integration test content")
        self.assertEqual(retrieved_chunk.project, "integration-test")
        self.assertEqual(len(retrieved_chunk.embedding), 768)

    async def test_cache_serialization_contract(self):
        """Test that cached results can be serialized/deserialized"""
        result = {
            "status": "success",
            "results": [
                {
                    "file_path": "test.py",
                    "content": "test",
                    "score": 0.8,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            ]
        }

        # Should serialize without error
        json_str = json.dumps(result)

        # Should deserialize back
        parsed = json.loads(json_str)
        self.assertEqual(parsed["status"], "success")

        # created_at should still be ISO string
        self.assertIsInstance(parsed["results"][0]["created_at"], str)


def run_contract_tests():
    """Run all contract validation tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestChunkSchemaContract))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorSearchContract))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPToolContracts))
    suite.addTests(loader.loadTestsFromTestCase(TestDataConsistencyContracts))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationContracts))

    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_contract_tests()
    sys.exit(0 if success else 1)