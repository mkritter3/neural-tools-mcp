#!/usr/bin/env python3
"""
Resilient Hybrid Retriever - Error-resilient wrapper for HybridRetriever
Provides comprehensive error handling, fallback mechanisms, and graceful degradation
Following roadmap Phase 1.5 specifications
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

from infrastructure.error_handling import (
    database_retry, error_handler, graceful_degradation
)

logger = logging.getLogger(__name__)

class ResilientHybridRetriever:
    """Error-resilient wrapper for hybrid retrieval operations"""
    
    def __init__(self, hybrid_retriever):
        """
        Initialize with HybridRetriever instance
        
        Args:
            hybrid_retriever: Instance of HybridRetriever to wrap
        """
        self.hybrid_retriever = hybrid_retriever
        self.fallback_cache = {}  # Simple in-memory cache for fallbacks
        self._register_fallbacks()
    
    def _register_fallbacks(self):
        """Register fallback functions for different operations"""
        graceful_degradation.register_fallback(
            "hybrid_search", self._fallback_hybrid_search
        )
        graceful_degradation.register_fallback(
            "vector_search", self._fallback_vector_search
        )
        graceful_degradation.register_fallback(
            "graph_context", self._fallback_graph_context
        )
    
    async def _fallback_hybrid_search(
        self, 
        query: str, 
        limit: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fallback hybrid search using cached results or simple text matching"""
        logger.warning("Using fallback hybrid search")
        
        # Check cache first
        cache_key = f"hybrid_{query}_{limit}"
        if cache_key in self.fallback_cache:
            cached_result = self.fallback_cache[cache_key]
            if (datetime.now() - cached_result['timestamp']).total_seconds() < 3600:  # 1 hour TTL
                logger.info("Returning cached fallback result")
                return cached_result['results']
        
        # Generate minimal fallback results
        fallback_results = [{
            "file_path": "system/fallback",
            "content": f"Search service temporarily unavailable for query: '{query}'",
            "chunk_index": 0,
            "score": 0.1,
            "context_info": {
                "graph_context": [],
                "related_files": [],
                "is_fallback": True
            }
        }]
        
        # Cache the fallback result
        self.fallback_cache[cache_key] = {
            'results': fallback_results,
            'timestamp': datetime.now()
        }
        
        return fallback_results
    
    async def _fallback_vector_search(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Fallback vector search"""
        logger.warning("Using fallback vector search")
        return [{
            "id": "fallback_vector",
            "score": 0.0,
            "payload": {
                "content": "Vector search service temporarily unavailable",
                "file_path": "system/fallback",
                "source": "fallback_vector"
            }
        }]
    
    async def _fallback_graph_context(
        self,
        file_path: str,
        max_hops: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback graph context"""
        logger.warning("Using fallback graph context")
        return {
            "related_files": [],
            "function_calls": [],
            "class_relationships": [],
            "import_dependencies": [],
            "is_fallback": True
        }
    
    @database_retry
    async def find_similar_with_context(
        self,
        query: str,
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find similar content with comprehensive error handling
        
        Args:
            query: Search query
            limit: Maximum results to return
            include_graph_context: Whether to include graph context
            max_hops: Maximum graph traversal hops
            
        Returns:
            List of search results with context
        """
        try:
            # Attempt primary hybrid search
            results = await graceful_degradation.execute_with_fallback(
                "hybrid_search",
                self.hybrid_retriever.find_similar_with_context,
                query=query,
                limit=limit,
                include_graph_context=include_graph_context,
                max_hops=max_hops
            )
            
            if results['success']:
                logger.debug(f"Hybrid search successful: {len(results['result'])} results")
                return results['result']
            else:
                logger.warning(f"Hybrid search failed: {results.get('error', 'Unknown error')}")
                # Return fallback results
                return await self._fallback_hybrid_search(query, limit)
                
        except Exception as e:
            await error_handler.handle_error(
                e,
                context={
                    "query": query,
                    "limit": limit,
                    "include_graph_context": include_graph_context
                },
                operation_name="find_similar_with_context"
            )
            
            # Return fallback results
            logger.error(f"Critical error in hybrid search: {e}")
            return await self._fallback_hybrid_search(query, limit)
    
    @database_retry
    async def search_by_file_type(
        self,
        query: str,
        file_types: List[str],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search by file type with error handling
        
        Args:
            query: Search query
            file_types: List of file extensions to search
            limit: Maximum results
            
        Returns:
            Filtered search results
        """
        try:
            if hasattr(self.hybrid_retriever, 'search_by_file_type'):
                return await self.hybrid_retriever.search_by_file_type(query, file_types, limit)
            else:
                # Fallback to regular search with file type filtering
                all_results = await self.find_similar_with_context(query, limit * 2)
                
                filtered_results = []
                for result in all_results:
                    file_path = result.get('file_path', '')
                    if any(file_path.endswith(ext) for ext in file_types):
                        filtered_results.append(result)
                    
                    if len(filtered_results) >= limit:
                        break
                
                return filtered_results
                
        except Exception as e:
            await error_handler.handle_error(
                e,
                context={"query": query, "file_types": file_types},
                operation_name="search_by_file_type"
            )
            return []
    
    @database_retry
    async def get_file_context(
        self,
        file_path: str,
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Get file context with error handling
        
        Args:
            file_path: Path to file
            max_hops: Maximum graph traversal hops
            
        Returns:
            File context information
        """
        try:
            if hasattr(self.hybrid_retriever, 'get_file_context'):
                return await self.hybrid_retriever.get_file_context(file_path, max_hops)
            else:
                # Use graph context fallback
                return await graceful_degradation.execute_with_fallback(
                    "graph_context",
                    self._get_basic_file_context,
                    file_path=file_path,
                    max_hops=max_hops
                )
                
        except Exception as e:
            await error_handler.handle_error(
                e,
                context={"file_path": file_path, "max_hops": max_hops},
                operation_name="get_file_context"
            )
            return await self._fallback_graph_context(file_path, max_hops)
    
    async def _get_basic_file_context(self, file_path: str, max_hops: int = 2) -> Dict[str, Any]:
        """Basic file context extraction"""
        try:
            # Try to get context from the hybrid retriever's components
            if hasattr(self.hybrid_retriever, 'neo4j_service') and self.hybrid_retriever.neo4j_service:
                return await self.hybrid_retriever.neo4j_service.get_file_relationships(
                    file_path, max_hops
                )
            else:
                return {"related_files": [], "relationships": []}
                
        except Exception as e:
            logger.warning(f"Basic file context extraction failed: {e}")
            return {"related_files": [], "relationships": [], "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for hybrid retrieval system
        
        Returns:
            Health status including component health
        """
        health_status = {
            "overall_health": "unknown",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check hybrid retriever health
            if hasattr(self.hybrid_retriever, 'health_check'):
                hybrid_health = await self.hybrid_retriever.health_check()
                health_status["components"]["hybrid_retriever"] = hybrid_health
            
            # Check individual service health if available
            if hasattr(self.hybrid_retriever, 'qdrant_service'):
                qdrant_health = await self.hybrid_retriever.qdrant_service.health_check()
                health_status["components"]["qdrant"] = qdrant_health
            
            if hasattr(self.hybrid_retriever, 'neo4j_service'):
                neo4j_health = await self.hybrid_retriever.neo4j_service.health_check()
                health_status["components"]["neo4j"] = neo4j_health
            
            if hasattr(self.hybrid_retriever, 'embedding_service'):
                embedding_health = await self.hybrid_retriever.embedding_service.health_check()
                health_status["components"]["embedding"] = embedding_health
            
            # Determine overall health
            healthy_components = sum(
                1 for comp in health_status["components"].values()
                if comp.get("healthy", False)
            )
            total_components = len(health_status["components"])
            
            if healthy_components == total_components:
                health_status["overall_health"] = "healthy"
            elif healthy_components >= total_components // 2:
                health_status["overall_health"] = "degraded"
            else:
                health_status["overall_health"] = "unhealthy"
            
            health_status["healthy_components"] = healthy_components
            health_status["total_components"] = total_components
            
        except Exception as e:
            await error_handler.handle_error(
                e,
                context={},
                operation_name="health_check"
            )
            health_status["overall_health"] = "error"
            health_status["error"] = str(e)
        
        return health_status
    
    async def clear_fallback_cache(self):
        """Clear the fallback cache"""
        self.fallback_cache.clear()
        logger.info("Cleared fallback cache")
    
    def get_fallback_cache_stats(self) -> Dict[str, Any]:
        """Get fallback cache statistics"""
        return {
            "cache_size": len(self.fallback_cache),
            "cache_keys": list(self.fallback_cache.keys()),
            "oldest_entry": min(
                (entry["timestamp"] for entry in self.fallback_cache.values()),
                default=None
            ),
            "newest_entry": max(
                (entry["timestamp"] for entry in self.fallback_cache.values()),
                default=None
            )
        }

# Factory function for creating resilient retriever
async def create_resilient_retriever(project_name: str = "default") -> ResilientHybridRetriever:
    """
    Factory function to create resilient hybrid retriever
    
    Args:
        project_name: Project name for service initialization
        
    Returns:
        Initialized ResilientHybridRetriever instance
    """
    try:
        # Import here to avoid circular dependencies
        from .hybrid_retriever import HybridRetriever
        
        # Create base hybrid retriever
        hybrid_retriever = HybridRetriever(project_name)
        await hybrid_retriever.initialize()
        
        # Wrap with resilience
        resilient_retriever = ResilientHybridRetriever(hybrid_retriever)
        
        logger.info(f"Created resilient hybrid retriever for project: {project_name}")
        return resilient_retriever
        
    except Exception as e:
        logger.error(f"Failed to create resilient retriever: {e}")
        # Create with minimal fallback-only functionality
        from types import SimpleNamespace
        
        fallback_retriever = SimpleNamespace()
        fallback_retriever.initialized = False
        
        resilient_retriever = ResilientHybridRetriever(fallback_retriever)
        logger.warning("Created fallback-only resilient retriever")
        
        return resilient_retriever

# Example usage and testing
async def demo_resilient_retrieval():
    """Demonstrate resilient retrieval capabilities"""
    print("🛡️ Testing resilient hybrid retrieval...")
    
    try:
        # Create resilient retriever
        retriever = await create_resilient_retriever("test_project")
        
        # Test health check
        health = await retriever.health_check()
        print(f"   💚 Health status: {health['overall_health']}")
        print(f"   📊 Components: {health.get('healthy_components', 0)}/{health.get('total_components', 0)}")
        
        # Test search with error handling
        results = await retriever.find_similar_with_context("test query", limit=3)
        print(f"   🔍 Search results: {len(results)} items")
        
        for i, result in enumerate(results[:2]):
            print(f"      {i+1}. {result.get('file_path', 'unknown')} (score: {result.get('score', 0):.3f})")
        
        # Test cache stats
        cache_stats = retriever.get_fallback_cache_stats()
        print(f"   📦 Cache stats: {cache_stats['cache_size']} entries")
        
        print("✅ Resilient retrieval test completed")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_resilient_retrieval())