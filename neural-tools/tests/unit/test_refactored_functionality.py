#!/usr/bin/env python3
"""
L9 Neural Tools - Comprehensive Refactoring Tests
Tests all refactored components to ensure functionality is preserved
Generated based on systematic analysis of edge cases and integration points
"""

import asyncio
import json
import logging
import os
import sys
import unittest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any, Optional
import tempfile
import uuid

# Add path for imports
sys.path.append('/app/project/neural-tools')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestNeuralConfig(unittest.TestCase):
    """Test the centralized configuration management"""
    
    def setUp(self):
        """Set up test environment"""
        # Save original environment
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        """Restore original environment"""
        os.environ.clear()
        os.environ.update(self.original_env)
        
    def test_default_configuration(self):
        """Test configuration with default values"""
        # Clear environment variables
        for key in ['PROJECT_NAME', 'QDRANT_HOST', 'NEO4J_HOST']:
            os.environ.pop(key, None)
            
        from common.config import NeuralConfig
        config = NeuralConfig()
        
        # Test defaults
        self.assertEqual(config.PROJECT_NAME, "default")
        self.assertEqual(config.QDRANT_HOST, "neural-data-storage")
        self.assertEqual(config.NEO4J_HOST, "neo4j-graph")
        self.assertEqual(config.COLLECTION_PREFIX, "project_default_")
        
    def test_environment_variable_override(self):
        """Test configuration with custom environment variables"""
        os.environ['PROJECT_NAME'] = 'test_project'
        os.environ['QDRANT_HOST'] = 'custom-qdrant'
        os.environ['QDRANT_HTTP_PORT'] = '9999'
        
        from common.config import NeuralConfig
        config = NeuralConfig()
        
        self.assertEqual(config.PROJECT_NAME, "test_project")
        self.assertEqual(config.QDRANT_HOST, "custom-qdrant") 
        self.assertEqual(config.QDRANT_HTTP_PORT, 9999)
        self.assertEqual(config.QDRANT_URL, "http://custom-qdrant:9999")
        self.assertEqual(config.COLLECTION_PREFIX, "project_test_project_")
        
    def test_invalid_port_handling(self):
        """Test handling of invalid port values"""
        os.environ['QDRANT_HTTP_PORT'] = 'invalid'
        
        with self.assertRaises(ValueError):
            from common.config import NeuralConfig
            NeuralConfig()
            
    def test_feature_detection(self):
        """Test feature availability detection"""
        from common.config import NeuralConfig
        config = NeuralConfig()
        
        # Test feature availability methods
        self.assertIsInstance(config.is_feature_enabled('neo4j'), bool)
        self.assertIsInstance(config.is_feature_enabled('prism'), bool) 
        self.assertIsInstance(config.is_feature_enabled('tree_sitter'), bool)
        
        # Test invalid feature
        self.assertFalse(config.is_feature_enabled('nonexistent'))
        
    def test_service_info(self):
        """Test service information generation"""
        from common.config import NeuralConfig
        config = NeuralConfig()
        
        service_info = config.get_service_info()
        
        self.assertIn('qdrant', service_info)
        self.assertIn('neo4j', service_info)
        self.assertIn('embeddings', service_info)
        
        for service in service_info.values():
            self.assertIn('url', service)
            self.assertIn('healthy', service)
            self.assertEqual(service['healthy'], False)  # Default unhealthy state

class TestUtilityFunctions(unittest.TestCase):
    """Test shared utility functions"""
    
    def test_deterministic_point_id_consistency(self):
        """Test that same inputs produce same IDs"""
        from common.utils import generate_deterministic_point_id
        
        file_path = "/test/file.py"
        content = "test content"
        chunk_index = 0
        
        # Generate ID multiple times with same inputs
        id1 = generate_deterministic_point_id(file_path, content, chunk_index)
        id2 = generate_deterministic_point_id(file_path, content, chunk_index)
        id3 = generate_deterministic_point_id(file_path, content, chunk_index)
        
        # All should be identical
        self.assertEqual(id1, id2)
        self.assertEqual(id2, id3)
        
    def test_deterministic_point_id_uniqueness(self):
        """Test that different inputs produce different IDs"""
        from common.utils import generate_deterministic_point_id
        
        # Different file paths
        id1 = generate_deterministic_point_id("/file1.py", "content", 0)
        id2 = generate_deterministic_point_id("/file2.py", "content", 0)
        self.assertNotEqual(id1, id2)
        
        # Different content
        id3 = generate_deterministic_point_id("/file.py", "content1", 0)
        id4 = generate_deterministic_point_id("/file.py", "content2", 0)
        self.assertNotEqual(id3, id4)
        
        # Different chunk index
        id5 = generate_deterministic_point_id("/file.py", "content", 0)
        id6 = generate_deterministic_point_id("/file.py", "content", 1)
        self.assertNotEqual(id5, id6)
        
    def test_content_hash(self):
        """Test content hash generation"""
        from common.utils import get_content_hash
        
        # Same content produces same hash
        hash1 = get_content_hash("test content")
        hash2 = get_content_hash("test content")
        self.assertEqual(hash1, hash2)
        
        # Different content produces different hash
        hash3 = get_content_hash("different content")
        self.assertNotEqual(hash1, hash3)
        
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        from common.utils import cosine_similarity
        
        # Identical vectors should have similarity 1.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        # Orthogonal vectors should have similarity 0.0
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec3, vec4)
        self.assertAlmostEqual(similarity, 0.0, places=5)
        
        # Opposite vectors should have similarity -1.0
        vec5 = [1.0, 0.0, 0.0]
        vec6 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec5, vec6)
        self.assertAlmostEqual(similarity, -1.0, places=5)
        
        # Different length vectors should return 0.0
        vec7 = [1.0, 0.0]
        vec8 = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec7, vec8)
        self.assertEqual(similarity, 0.0)
        
        # Zero vectors should return 0.0
        vec9 = [0.0, 0.0, 0.0]
        vec10 = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec9, vec10)
        self.assertEqual(similarity, 0.0)
        
    def test_rrf_combination_empty_inputs(self):
        """Test RRF combination with empty inputs"""
        from common.utils import combine_with_rrf_internal
        
        # Empty lists
        result = combine_with_rrf_internal([], [])
        self.assertEqual(result, [])
        
        # One empty, one with data
        vector_results = [{"id": "1", "score": 0.9}]
        result = combine_with_rrf_internal(vector_results, [])
        self.assertEqual(len(result), 1)
        self.assertIn('rrf_score', result[0])
        
    def test_rrf_combination_duplicate_ids(self):
        """Test RRF combination with duplicate IDs across result sets"""
        from common.utils import combine_with_rrf_internal
        
        vector_results = [{"id": "1", "score": 0.9, "content": "test1"}]
        text_results = [{"id": "1", "score": 0.8, "content": "test1"}]
        
        result = combine_with_rrf_internal(vector_results, text_results)
        
        # Should have only one result with combined RRF score
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")
        
        # RRF score should be sum of individual scores
        expected_rrf = 1.0 / (60 + 1) + 1.0 / (60 + 1)  # Both at rank 0
        self.assertAlmostEqual(result[0]["rrf_score"], expected_rrf, places=5)
        
    def test_rrf_combination_missing_ids(self):
        """Test RRF combination with missing ID fields"""
        from common.utils import combine_with_rrf_internal
        
        vector_results = [{"score": 0.9}, {"id": "", "score": 0.8}]
        text_results = [{"id": "1", "score": 0.7}]
        
        result = combine_with_rrf_internal(vector_results, text_results)
        
        # Should only include result with valid ID
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")
        
    def test_mmr_diversity_empty_results(self):
        """Test MMR diversity with empty results"""
        from common.utils import apply_mmr_diversity
        
        # Empty list
        result = apply_mmr_diversity([], 0.8, 5)
        self.assertEqual(result, [])
        
        # Single result
        single_result = [{"id": "1", "score": 0.9, "embedding": [1.0, 0.0]}]
        result = apply_mmr_diversity(single_result, 0.8, 5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")
        
    def test_mmr_diversity_no_embeddings(self):
        """Test MMR diversity with results missing embeddings"""
        from common.utils import apply_mmr_diversity
        
        results = [
            {"id": "1", "score": 0.9, "embedding": [1.0, 0.0]},
            {"id": "2", "score": 0.8},  # No embedding
            {"id": "3", "score": 0.7}   # No embedding
        ]
        
        result = apply_mmr_diversity(results, 0.8, 5)
        
        # Should only include first result (others have no embeddings)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "1")
        
    def test_mmr_diversity_threshold_edge_cases(self):
        """Test MMR diversity with extreme threshold values"""
        from common.utils import apply_mmr_diversity
        
        results = [
            {"id": "1", "score": 0.9, "embedding": [1.0, 0.0]},
            {"id": "2", "score": 0.8, "embedding": [0.9, 0.1]},  # Similar
            {"id": "3", "score": 0.7, "embedding": [0.0, 1.0]}   # Different
        ]
        
        # Threshold 0.0 - should accept all different results
        result = apply_mmr_diversity(results, 0.0, 5)
        self.assertGreaterEqual(len(result), 1)
        
        # Threshold 1.0 - should be very strict
        result = apply_mmr_diversity(results, 1.0, 5)
        self.assertEqual(len(result), 3)  # Should include all as none are identical
        
    def test_token_estimation(self):
        """Test token estimation function"""
        from common.utils import estimate_tokens
        
        # Empty string
        tokens = estimate_tokens("")
        self.assertEqual(tokens, 1)  # Minimum 1
        
        # Short string
        tokens = estimate_tokens("test")
        self.assertEqual(tokens, 1)  # 4 characters -> 1 token
        
        # Longer string
        tokens = estimate_tokens("this is a longer test string with multiple words")
        self.assertGreater(tokens, 1)
        
    def test_format_timestamp(self):
        """Test timestamp formatting"""
        from common.utils import format_timestamp
        
        timestamp = format_timestamp()
        
        # Should be ISO format string
        self.assertIsInstance(timestamp, str)
        self.assertIn('T', timestamp)  # ISO format has T separator

class TestBaseNeuralTester(unittest.TestCase):
    """Test the base testing infrastructure"""
    
    def test_base_tester_initialization(self):
        """Test BaseNeuralTester initialization"""
        from common.test_base import BaseNeuralTester
        
        tester = BaseNeuralTester("TestSuite")
        
        self.assertEqual(tester.test_name, "TestSuite")
        self.assertEqual(tester.total_tests, 0)
        self.assertEqual(tester.passed_tests, 0)
        
    def test_test_logging_and_tracking(self):
        """Test test result logging and tracking"""
        from common.test_base import BaseNeuralTester
        
        tester = BaseNeuralTester("TestSuite")
        
        # Log test start
        tester.log_test_start("Test 1")
        self.assertEqual(tester.total_tests, 1)
        
        # Log test pass
        tester.log_test_pass("Test 1", "Success details")
        self.assertEqual(tester.passed_tests, 1)
        
        # Log test fail
        tester.log_test_start("Test 2")
        tester.log_test_fail("Test 2", "Error message")
        self.assertEqual(tester.total_tests, 2)
        self.assertEqual(tester.passed_tests, 1)
        
    def test_summary_generation(self):
        """Test test summary generation"""
        from common.test_base import BaseNeuralTester
        
        tester = BaseNeuralTester("TestSuite")
        
        # Add some test results
        tester.log_test_start("Test 1")
        tester.log_test_pass("Test 1")
        
        tester.log_test_start("Test 2")
        tester.log_test_fail("Test 2", "Error")
        
        summary = tester.get_test_summary()
        
        self.assertEqual(summary["test_name"], "TestSuite")
        self.assertEqual(summary["total_tests"], 2)
        self.assertEqual(summary["passed_tests"], 1)
        self.assertEqual(summary["failed_tests"], 1)
        self.assertEqual(summary["success_rate"], "50.0%")
        
    def test_mcp_tool_tester_initialization(self):
        """Test MCPToolTester initialization"""
        from common.test_base import MCPToolTester
        
        tester = MCPToolTester("MCPTest")
        
        self.assertEqual(len(tester.mcp_tools), 15)  # All 15 MCP tools
        self.assertIn("memory_store_enhanced", tester.mcp_tools)
        self.assertIn("memory_search_enhanced", tester.mcp_tools)
        self.assertIn("neural_system_status", tester.mcp_tools)

class TestMemoryToolsIntegration(unittest.IsolatedAsyncioTestCase):
    """Test memory tools integration and functionality"""
    
    async def test_memory_store_enhanced_structure(self):
        """Test memory_store_enhanced function structure"""
        from tools.memory_tools import memory_store_enhanced
        
        # Test with mock clients
        mock_qdrant = Mock()
        mock_nomic = AsyncMock()
        mock_neo4j = Mock()
        
        # Mock successful embedding response
        mock_embed_response = Mock()
        mock_embed_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_nomic.get_embeddings.return_value = mock_embed_response
        
        # Mock successful upsert
        mock_upsert_result = Mock()
        mock_upsert_result.status = "completed"
        mock_qdrant.upsert.return_value = mock_upsert_result
        
        # Test function call
        result = await memory_store_enhanced(
            content="test content",
            category="test",
            qdrant_client=mock_qdrant,
            nomic_client=mock_nomic,
            neo4j_client=mock_neo4j
        )
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        
        # Verify embedding was called
        mock_nomic.get_embeddings.assert_called_once_with(["test content"])
        
    async def test_memory_store_embedding_failure(self):
        """Test memory_store_enhanced with embedding failure"""
        from tools.memory_tools import memory_store_enhanced
        
        mock_qdrant = Mock()
        mock_nomic = AsyncMock()
        mock_neo4j = Mock()
        
        # Mock embedding failure
        mock_nomic.get_embeddings.side_effect = Exception("Embedding service down")
        
        result = await memory_store_enhanced(
            content="test content",
            qdrant_client=mock_qdrant,
            nomic_client=mock_nomic,
            neo4j_client=mock_neo4j
        )
        
        # Should return error
        self.assertEqual(result["status"], "error")
        self.assertIn("Embedding generation failed", result["message"])
        
    async def test_memory_search_enhanced_modes(self):
        """Test memory_search_enhanced different search modes"""
        from tools.memory_tools import memory_search_enhanced
        
        mock_qdrant = Mock()
        mock_nomic = AsyncMock()
        mock_neo4j = Mock()
        
        # Mock search results
        mock_qdrant.search.return_value = []
        
        # Mock embedding response
        mock_embed_response = Mock()
        mock_embed_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_nomic.get_embeddings.return_value = mock_embed_response
        
        # Test different modes
        modes = ["semantic", "rrf_hybrid", "mmr_diverse"]
        
        for mode in modes:
            result = await memory_search_enhanced(
                query="test query",
                mode=mode,
                qdrant_client=mock_qdrant,
                nomic_client=mock_nomic,
                neo4j_client=mock_neo4j
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
            self.assertIn("results", result)
            self.assertEqual(result["mode"], mode)

class TestIntegrationWithMainServer(unittest.TestCase):
    """Test integration with main MCP server"""
    
    def test_refactored_modules_importable(self):
        """Test that all refactored modules can be imported"""
        try:
            from common.config import config
            from common.utils import generate_deterministic_point_id
            from common.test_base import BaseNeuralTester
            from tools.memory_tools import memory_store_enhanced
            
            # If we get here, all imports succeeded
            self.assertTrue(True)
            
        except ImportError as e:
            self.fail(f"Failed to import refactored modules: {e}")
            
    def test_config_singleton_behavior(self):
        """Test that config behaves as expected singleton"""
        from common.config import config as config1
        from common.config import config as config2
        
        # Should be same instance
        self.assertIs(config1, config2)
        
        # Should have consistent values
        self.assertEqual(config1.PROJECT_NAME, config2.PROJECT_NAME)

def run_docker_integration_tests():
    """Run integration tests inside Docker container"""
    import subprocess
    
    test_commands = [
        "python3 -c 'from common.config import config; print(\"✅ Config loads:\", config.PROJECT_NAME)'",
        "python3 -c 'from common.utils import generate_deterministic_point_id; print(\"✅ Utils work:\", generate_deterministic_point_id(\"/test\", \"content\", 0))'",
        "python3 -c 'from tools.memory_tools import memory_store_enhanced; print(\"✅ Memory tools importable\")'",
        "python3 -c 'import neural_mcp_server_enhanced as server; print(\"✅ Server health:\", server.production_health_check()[\"status\"])'"
    ]
    
    all_passed = True
    
    for i, cmd in enumerate(test_commands, 1):
        try:
            result = subprocess.run(
                f"docker exec default-neural {cmd}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Integration Test {i}: PASSED")
                logger.info(f"   Output: {result.stdout.strip()}")
            else:
                logger.error(f"❌ Integration Test {i}: FAILED")
                logger.error(f"   Error: {result.stderr}")
                all_passed = False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Integration Test {i}: TIMEOUT")
            all_passed = False
        except Exception as e:
            logger.error(f"❌ Integration Test {i}: ERROR - {e}")
            all_passed = False
            
    return all_passed

def main():
    """Main test execution"""
    logger.info("🧪 Starting comprehensive refactoring tests...")
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run Docker integration tests if available
    if os.environ.get('RUN_DOCKER_TESTS', 'false').lower() == 'true':
        logger.info("\n🐳 Running Docker integration tests...")
        docker_success = run_docker_integration_tests()
        
        if docker_success:
            logger.info("✅ All Docker integration tests passed")
        else:
            logger.error("❌ Some Docker integration tests failed")
            return 1
    
    logger.info("✅ Comprehensive refactoring tests completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())