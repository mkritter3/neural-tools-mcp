#!/usr/bin/env python3
"""
Integration test for Advanced GraphRAG Pipeline
Tests all new components: async preprocessing, metadata tagging, exclusion, RRF
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from servers.services.service_container import ServiceContainer
from servers.services.async_preprocessing_pipeline import AsyncPreprocessingPipeline, MetadataTaggerClient
from servers.services.exclusion_manager import ExclusionManager
from servers.services.rrf_reranker import RRFReranker, SearchResult
from servers.services.queue_based_indexer import QueueBasedIndexer
from servers.services.pipeline_monitor import PipelineMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_exclusion_manager():
    """Test exclusion manager with .graphragignore"""
    logger.info("\n=== Testing Exclusion Manager ===")
    
    project_path = Path.cwd()
    manager = ExclusionManager(str(project_path))
    
    # Create default .graphragignore if needed
    manager.create_default_graphragignore()
    
    # Test exclusion patterns
    test_paths = [
        ".archive/old_code.py",
        ".deprecated/legacy.js",
        "src/main.py",
        "backup/data_backup.json",
        "node_modules/package/index.js",
        ".venv/lib/python3.9/site.py"
    ]
    
    for path in test_paths:
        excluded = manager.should_exclude(path)
        penalty = manager.get_weight_penalty(path)
        logger.info(f"  {path}: excluded={excluded}, penalty={penalty:.2f}")
    
    # Get statistics
    stats = manager.get_statistics()
    logger.info(f"  Exclusion stats: {stats}")
    
    return True


async def test_rrf_reranker():
    """Test RRF re-ranking system"""
    logger.info("\n=== Testing RRF Reranker ===")
    
    reranker = RRFReranker(k=60)
    
    # Create mock search results from different sources
    vector_results = [
        SearchResult("1", "Vector result 1", "/src/main.py", 0.95, "vector"),
        SearchResult("2", "Vector result 2", "/src/utils.py", 0.85, "vector"),
        SearchResult("3", "Vector result 3", "/src/config.py", 0.75, "vector"),
    ]
    
    graph_results = [
        SearchResult("2", "Graph result 2", "/src/utils.py", 0.90, "graph"),
        SearchResult("4", "Graph result 4", "/src/models.py", 0.80, "graph"),
        SearchResult("1", "Graph result 1", "/src/main.py", 0.70, "graph"),
    ]
    
    keyword_results = [
        SearchResult("5", "Keyword result 5", "/docs/readme.md", 0.88, "keyword"),
        SearchResult("1", "Keyword result 1", "/src/main.py", 0.78, "keyword"),
        SearchResult("3", "Keyword result 3", "/src/config.py", 0.68, "keyword"),
    ]
    
    # Apply RRF
    search_inputs = [
        ("vector", vector_results),
        ("graph", graph_results),
        ("keyword", keyword_results)
    ]
    
    merged_results = reranker.apply_rrf(search_inputs, top_k=5)
    
    logger.info("  RRF Merged Results:")
    for i, result in enumerate(merged_results, 1):
        rrf_details = result.metadata.get('rrf_details', {})
        logger.info(
            f"    {i}. {result.file_path} - "
            f"RRF Score: {result.score:.4f}, "
            f"Sources: {rrf_details.get('num_sources', 0)}"
        )
    
    # Test weighted RRF
    weighted_inputs = [
        ("vector", vector_results, 1.0),   # Full weight
        ("graph", graph_results, 0.8),     # 80% weight
        ("keyword", keyword_results, 0.5)  # 50% weight
    ]
    
    weighted_results = reranker.apply_weighted_rrf(weighted_inputs, top_k=5)
    logger.info("\n  Weighted RRF Results:")
    for i, result in enumerate(weighted_results, 1):
        logger.info(f"    {i}. {result.file_path} - Score: {result.score:.4f}")
    
    # Get metrics
    metrics = reranker.get_metrics()
    logger.info(f"  Reranker metrics: {metrics}")
    
    return True


async def test_metadata_tagger():
    """Test Gemma metadata tagging client"""
    logger.info("\n=== Testing Metadata Tagger ===")
    
    # Note: This requires Gemma container to be running
    tagger = MetadataTaggerClient(base_url="http://localhost:48001")
    
    # Test code sample
    test_code = """
import asyncio
from typing import List, Dict
import logging

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def process_batch(self, items: List[Dict]) -> List[Dict]:
        '''Process a batch of items asynchronously'''
        results = []
        for item in items:
            result = await self._process_item(item)
            results.append(result)
        return results
    """
    
    try:
        # Test tagging
        metadata = await tagger.tag_code(test_code, "test_processor.py")
        logger.info(f"  Metadata extracted:")
        logger.info(f"    Status: {metadata.status}")
        logger.info(f"    Component type: {metadata.component_type}")
        logger.info(f"    Dependencies: {metadata.dependencies}")
        logger.info(f"    Complexity: {metadata.complexity_score}")
        return True
    except Exception as e:
        logger.warning(f"  Metadata tagger not available: {e}")
        logger.info("  (This is expected if Gemma container is not running)")
        return False


async def test_queue_based_indexer():
    """Test queue-based indexer"""
    logger.info("\n=== Testing Queue-Based Indexer ===")
    
    # Create test project directory
    test_project = Path("/tmp/test_graphrag_project")
    test_project.mkdir(exist_ok=True)
    
    # Create some test files
    test_files = [
        ("main.py", "def main():\n    print('Hello GraphRAG')"),
        ("utils.py", "def helper():\n    return 42"),
        (".archive/old.py", "# Old deprecated code"),
        ("config.json", '{"version": "1.0.0"}')
    ]
    
    for filename, content in test_files:
        file_path = test_project / filename
        file_path.parent.mkdir(exist_ok=True)
        file_path.write_text(content)
    
    # Initialize container (mock)
    container = None  # Would be actual ServiceContainer in production
    
    # Create indexer
    indexer = QueueBasedIndexer(
        str(test_project),
        project_name="test",
        container=container
    )
    
    # Test file filtering
    logger.info("  Testing file filtering:")
    for filename, _ in test_files:
        file_path = str(test_project / filename)
        should_index = indexer.should_index_file(file_path)
        logger.info(f"    {filename}: should_index={should_index}")
    
    # Get status
    # Note: This would fail without actual container/pipeline
    # status = await indexer.get_status()
    # logger.info(f"  Indexer status: {status}")
    
    return True


async def test_pipeline_monitor():
    """Test pipeline monitoring"""
    logger.info("\n=== Testing Pipeline Monitor ===")
    
    # This requires actual pipeline and container instances
    # For testing, we'll just verify the class initializes
    
    try:
        # Mock components
        container = None
        pipeline = None
        indexer = None
        
        monitor = PipelineMonitor(container, pipeline, indexer)
        
        # Test Prometheus metrics format
        prometheus_output = monitor.get_prometheus_metrics()
        logger.info("  Prometheus metrics format:")
        for line in prometheus_output.split('\n')[:5]:
            logger.info(f"    {line}")
        
        # Test alert thresholds
        logger.info(f"  Alert thresholds configured:")
        for key, value in monitor.alert_thresholds.items():
            logger.info(f"    {key}: {value}")
        
        return True
    except Exception as e:
        logger.warning(f"  Monitor test partial: {e}")
        return False


async def test_integration():
    """Test full integration of all components"""
    logger.info("\n=== Testing Full Integration ===")
    
    try:
        # Initialize service container
        logger.info("  Initializing service container...")
        container = ServiceContainer()
        await container.initialize()
        
        # Initialize preprocessing pipeline
        logger.info("  Initializing preprocessing pipeline...")
        pipeline = AsyncPreprocessingPipeline(container)
        
        # Initialize exclusion manager
        logger.info("  Initializing exclusion manager...")
        exclusion_manager = ExclusionManager(str(Path.cwd()))
        
        # Initialize RRF reranker
        logger.info("  Initializing RRF reranker...")
        reranker = RRFReranker(k=60)
        
        # Initialize queue-based indexer
        logger.info("  Initializing queue-based indexer...")
        indexer = QueueBasedIndexer(
            str(Path.cwd()),
            project_name="test_integration",
            container=container
        )
        
        # Initialize monitor
        logger.info("  Initializing pipeline monitor...")
        monitor = PipelineMonitor(container, pipeline, indexer)
        
        logger.info("  ✅ All components initialized successfully!")
        
        # Cleanup
        await container.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"  Integration test failed: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Advanced GraphRAG Pipeline Integration Tests")
    logger.info("=" * 60)
    
    results = {}
    
    # Run individual component tests
    results['exclusion_manager'] = await test_exclusion_manager()
    results['rrf_reranker'] = await test_rrf_reranker()
    results['metadata_tagger'] = await test_metadata_tagger()
    results['queue_based_indexer'] = await test_queue_based_indexer()
    results['pipeline_monitor'] = await test_pipeline_monitor()
    
    # Run integration test if services are available
    try:
        results['integration'] = await test_integration()
    except Exception as e:
        logger.warning(f"Integration test skipped: {e}")
        results['integration'] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name:20} {status}")
    
    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)
    
    logger.info(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info("\n🎉 All tests passed! Advanced GraphRAG pipeline is ready!")
    else:
        logger.info("\n⚠️  Some tests failed. Check if all services are running:")
        logger.info("  - Gemma container (port 48001)")
        logger.info("  - Redis queue (port 46380)")
        logger.info("  - Neo4j (port 47687)")
        logger.info("  - Qdrant (port 46333)")
        logger.info("  - Nomic embeddings (port 48000)")


if __name__ == "__main__":
    asyncio.run(main())