#!/usr/bin/env python3
"""
Qdrant Vector Database Service - Extracted from monolithic neural-mcp-server-enhanced.py
Provides vector storage and retrieval operations with hybrid search capabilities
Enhanced with comprehensive error handling and graceful degradation
"""

import os
import logging
import asyncio
import json
import hashlib
import time
from typing import List, Dict, Any, Optional
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, CollectionStatus,
    SearchRequest, Filter, FieldCondition, MatchValue
)

from infrastructure.error_handling import (
    database_retry, VectorDatabaseException, error_handler, 
    graceful_degradation, ErrorCategory, ErrorSeverity
)

logger = logging.getLogger(__name__)

class QdrantService:
    """Service class for Qdrant vector database operations"""
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.collection_prefix = f"project_{project_name}_"
        
        # Qdrant configuration via central runtime config (with fallbacks)
        try:
            from servers.config import get_runtime_config
            cfg = get_runtime_config()
            self.host = cfg.database.qdrant_host
            self.http_port = int(cfg.database.qdrant_port)
        except Exception:
            # Fallback to environment if config package unavailable
            self.host = os.environ.get('QDRANT_HOST', 'localhost')
            self.http_port = int(os.environ.get('QDRANT_HTTP_PORT') or os.environ.get('QDRANT_PORT', 6333))
        self.grpc_port = int(os.environ.get('QDRANT_GRPC_PORT', 6334))
        
        self.client = None
        self.initialized = False
        
        # Cache configuration for intelligent caching
        self.cache_enabled = os.getenv('QDRANT_CACHE_ENABLED', 'true').lower() == 'true'
        self.default_ttl = int(os.getenv('QDRANT_CACHE_TTL', 1800))  # 30 minutes default
        self.search_cache_ttl = int(os.getenv('QDRANT_SEARCH_CACHE_TTL', 900))  # 15 minutes for search results
        self.collection_cache_ttl = int(os.getenv('QDRANT_COLLECTION_CACHE_TTL', 3600))  # 1 hour for collections
        
        # Service container reference for cache access
        self.service_container = None
        
        # Register fallback search function
        self._register_fallbacks()
    
    def _register_fallbacks(self):
        """Register fallback functions for graceful degradation"""
        graceful_degradation.register_fallback("qdrant_search", self._fallback_search)
        graceful_degradation.register_fallback("qdrant_hybrid", self._fallback_hybrid_search)
    
    async def _fallback_search(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Fallback search when Qdrant is unavailable"""
        logger.warning("Using fallback search - Qdrant unavailable")
        return [{
            "id": "fallback",
            "score": 0.0,
            "payload": {
                "content": "Service temporarily unavailable",
                "file_path": "system/fallback",
                "source": "fallback"
            }
        }]
    
    async def _fallback_hybrid_search(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Fallback hybrid search when Qdrant is unavailable"""
        logger.warning("Using fallback hybrid search - Qdrant unavailable")
        return await self._fallback_search(*args, **kwargs)
    
    def set_service_container(self, container):
        """Set service container for cache access"""
        self.service_container = container
    
    def _generate_cache_key(self, operation: str, **params) -> str:
        """Generate cache key for Qdrant operations with parameters"""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        return f"l9:prod:neural_tools:qdrant:{operation}:{self.project_name}:{params_hash}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from Redis cache with analytics tracking"""
        if not self.cache_enabled or not self.service_container:
            return None
            
        try:
            redis_cache = await self.service_container.get_redis_cache_client()
            cached_data = await redis_cache.get(cache_key)
            
            if cached_data:
                ttl_remaining = await redis_cache.ttl(cache_key)
                
                # Record cache hit for analytics
                if hasattr(self.service_container, 'get_dlq_service'):
                    dlq_service = await self.service_container.get_dlq_service()
                    if hasattr(dlq_service.redis_client, 'lpush'):  # Check if analytics method exists
                        try:
                            # Use CacheWarmer analytics if available
                            cache_warmer = getattr(self.service_container, '_cache_warmer', None)
                            if cache_warmer and hasattr(cache_warmer, 'analytics'):
                                await cache_warmer.analytics.record_cache_access(
                                    cache_key, hit=True, ttl_remaining=ttl_remaining
                                )
                        except Exception:
                            pass  # Don't fail operations due to analytics issues
                
                return json.loads(cached_data)
                
            # Record cache miss for analytics  
            if hasattr(self.service_container, 'get_dlq_service'):
                dlq_service = await self.service_container.get_dlq_service()
                if hasattr(dlq_service.redis_client, 'lpush'):
                    try:
                        cache_warmer = getattr(self.service_container, '_cache_warmer', None)
                        if cache_warmer and hasattr(cache_warmer, 'analytics'):
                            await cache_warmer.analytics.record_cache_access(
                                cache_key, hit=False, ttl_remaining=None
                            )
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.warning(f"Cache get failed for {cache_key}: {e}")
            
        return None
    
    async def _store_in_cache(self, cache_key: str, data: Any, ttl: int) -> bool:
        """Store data in Redis cache"""
        if not self.cache_enabled or not self.service_container:
            return False
            
        try:
            redis_cache = await self.service_container.get_redis_cache_client()
            await redis_cache.setex(cache_key, ttl, json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.warning(f"Cache store failed for {cache_key}: {e}")
            return False
        
    @database_retry
    async def initialize(self) -> Dict[str, Any]:
        """Initialize Qdrant client with connection verification"""
        try:
            # Test basic connectivity using HTTP first
            async with httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(),
                timeout=10.0
            ) as client:
                response = await client.get(f"http://{self.host}:{self.http_port}/collections")
                
                if response.status_code != 200:
                    raise VectorDatabaseException(
                        f"Qdrant HTTP connectivity failed: {response.status_code}",
                        context={
                            "host": self.host,
                            "port": self.http_port,
                            "status_code": response.status_code
                        }
                    )
            
            # Initialize async client for operations (HTTP fallback since GRPC port not exposed)
            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.http_port,  # Use HTTP port since GRPC not exposed in Docker
                prefer_grpc=False     # Use HTTP for Docker setup
            )
            
            # Test client operations  
            collections_response = await self.client.get_collections()
            
            # Handle different API response structures (v1.12 vs v1.15+)
            if hasattr(collections_response, 'collections'):
                collections = collections_response.collections
            elif hasattr(collections_response, 'result') and hasattr(collections_response.result, 'collections'):
                collections = collections_response.result.collections
            else:
                collections = []
            
            self.initialized = True
            
            # Mark service as healthy
            graceful_degradation.mark_service_healthy("qdrant")
            
            return {
                "success": True,
                "message": "Qdrant service initialized successfully",
                "collections_count": len(collections),
                "grpc_endpoint": f"{self.host}:{self.grpc_port}"
            }
            
        except Exception as e:
            await error_handler.handle_error(
                e, 
                context={"service": "qdrant", "host": self.host, "port": self.http_port},
                operation_name="qdrant_initialize"
            )
            graceful_degradation.mark_service_unhealthy("qdrant")
            return {
                "success": False,
                "message": f"Qdrant initialization failed: {str(e)}"
            }
    
    @database_retry
    async def ensure_collection(self, collection_name: str, vector_size: int = 1536, distance=None) -> bool:
        """Ensure collection exists with proper configuration
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (default 1536, but typically 768 for Nomic)
            distance: Distance metric (default Distance.COSINE)
        """
        if distance is None:
            distance = Distance.COSINE
            
        if not self.initialized or not self.client:
            raise VectorDatabaseException(
                "Qdrant service not initialized",
                context={"collection_name": collection_name, "vector_size": vector_size}
            )
            
        try:
            # Check if collection exists
            collections_response = await self.client.get_collections()
            # Handle different API response structures (v1.12 vs v1.15+)
            if hasattr(collections_response, 'collections'):
                existing_names = [col.name for col in collections_response.collections]
            elif hasattr(collections_response, 'result') and hasattr(collections_response.result, 'collections'):
                existing_names = [col.name for col in collections_response.result.collections]
            else:
                existing_names = []
            
            if collection_name not in existing_names:
                # Create collection with optimized settings (unnamed default vector)
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance  # Now using the passed distance parameter
                    ),
                    # Production-grade settings
                    optimizers_config={
                        "default_segment_number": 8,
                        "memmap_threshold": 25000
                    },
                    # Enable hybrid search
                    hnsw_config={
                        "m": 32,
                        "ef_construct": 200
                    }
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            
            return True
            
        except Exception as e:
            await error_handler.handle_error(
                e,
                context={"collection_name": collection_name, "vector_size": vector_size},
                operation_name="ensure_collection"
            )
            raise VectorDatabaseException(
                f"Collection setup failed for {collection_name}: {e}",
                context={"collection_name": collection_name},
                original_exception=e
            )
    
    async def upsert_points(
        self, 
        collection_name: str, 
        points: List[PointStruct]
    ) -> Dict[str, Any]:
        """Upsert points with proper error handling and dimension validation"""
        if not self.initialized or not self.client:
            raise RuntimeError("Qdrant service not initialized")
            
        try:
            # Validate embedding dimensions against collection configuration
            if points:
                # Get collection info to check expected dimensions
                collection_info = await self.client.get_collection(collection_name)
                
                # Handle named vectors configuration
                # Get expected dimension from collection config
                if hasattr(collection_info.config.params.vectors, 'size'):
                    # Single unnamed vector configuration
                    expected_dim = collection_info.config.params.vectors.size
                else:
                    # Named vectors configuration (not currently used)
                    expected_dim = 768  # Default for Nomic embeddings
                
                # Check first point's vector dimensions
                first_point = points[0]
                actual_dim = None
                
                if hasattr(first_point, 'vector') and first_point.vector:
                    # Handle both dict (named vectors) and list formats
                    if isinstance(first_point.vector, dict):
                        # Named vector format (not currently used)
                        if "dense" in first_point.vector:
                            actual_dim = len(first_point.vector["dense"])
                    elif isinstance(first_point.vector, list):
                        # Bare list format (legacy)
                        actual_dim = len(first_point.vector)
                    
                    if actual_dim and actual_dim != expected_dim:
                        error_msg = f"Embedding dimension mismatch for collection '{collection_name}': expected {expected_dim}, got {actual_dim}. Check EMBED_DIM environment variable."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    if actual_dim:
                        logger.debug(f"Dimension validation passed: {actual_dim}D vectors for collection {collection_name}")
            
            # Convert dicts to PointStruct objects if needed
            from qdrant_client.models import PointStruct
            
            point_structs = []
            for p in points:
                if isinstance(p, dict):
                    # Debug: Log what we're sending
                    logger.debug(f"Creating PointStruct from dict: id={p.get('id')}, vector_type={type(p.get('vector'))}, vector_len={len(p.get('vector')) if p.get('vector') else 0}")
                    point_structs.append(PointStruct(**p))
                else:
                    point_structs.append(p)
            
            operation_info = await self.client.upsert(
                collection_name=collection_name,
                points=point_structs,
                wait=True
            )
            
            return {
                "status": "success",
                "operation_id": operation_info.operation_id,
                "points_count": len(points)
            }
            
        except Exception as e:
            logger.error(f"Upsert failed for {collection_name}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @database_retry
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search with intelligent caching"""
        if not self.initialized or not self.client:
            raise VectorDatabaseException(
                "Qdrant service not initialized",
                context={"collection_name": collection_name, "limit": limit}
            )
        
        # Generate cache key for vector search
        cache_params = {
            "collection_name": collection_name,
            "query_vector_hash": hashlib.sha256(str(query_vector).encode()).hexdigest()[:16],
            "limit": limit,
            "score_threshold": score_threshold,
            "filter_hash": hashlib.sha256(str(filter_conditions).encode()).hexdigest()[:8] if filter_conditions else None
        }
        cache_key = self._generate_cache_key("search_vectors", **cache_params)
        
        # Try cache first
        cached_results = await self._get_from_cache(cache_key)
        if cached_results is not None:
            logger.debug(f"Vector search cache hit for collection {collection_name}")
            return cached_results
            
        try:
            search_results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": str(result.id),
                    "score": float(result.score),
                    "payload": result.payload or {}
                })
            
            # Cache the results
            await self._store_in_cache(cache_key, results, self.search_cache_ttl)
            logger.debug(f"Cached vector search results for collection {collection_name}")
                
            return results
            
        except Exception as e:
            await error_handler.handle_error(
                e,
                context={
                    "collection_name": collection_name,
                    "limit": limit,
                    "score_threshold": score_threshold
                },
                operation_name="search_vectors"
            )
            
            # Return empty results for graceful degradation
            logger.warning(f"Vector search failed for {collection_name}, returning empty results")
            return []
    
    async def scroll_collection(
        self,
        collection_name: str,
        limit: int = 100,
        filter_conditions: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """Scroll through collection points for text search"""
        if not self.initialized or not self.client:
            raise RuntimeError("Qdrant service not initialized")
            
        try:
            scroll_result = await self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                scroll_filter=filter_conditions,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
                results.append({
                    "id": str(point.id),
                    "payload": point.payload or {}
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Collection scroll failed for {collection_name}: {e}")
            return []
    
    async def rrf_hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        limit: int = 10,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Perform RRF hybrid search combining vector and text search with caching"""
        # Generate cache key for hybrid search
        cache_params = {
            "collection_name": collection_name,
            "query_vector_hash": hashlib.sha256(str(query_vector).encode()).hexdigest()[:16],
            "query_text": query_text,
            "limit": limit,
            "k": k
        }
        cache_key = self._generate_cache_key("rrf_hybrid_search", **cache_params)
        
        # Try cache first
        cached_results = await self._get_from_cache(cache_key)
        if cached_results is not None:
            logger.debug(f"Hybrid search cache hit for collection {collection_name}")
            return cached_results
        
        try:
            # Vector search (will use its own caching)
            vector_results = await self.search_vectors(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit * 2,
                score_threshold=0.3
            )
            
            # Text search using content filtering
            text_filter = Filter(
                should=[
                    FieldCondition(
                        key="content", 
                        match=MatchValue(value=word)
                    ) for word in query_text.lower().split()
                ]
            )
            
            text_results = await self.scroll_collection(
                collection_name=collection_name,
                limit=limit * 2,
                filter_conditions=text_filter
            )
            
            # Combine using RRF
            results = self._combine_with_rrf(vector_results, text_results, k=k)[:limit]
            
            # Cache the hybrid search results
            await self._store_in_cache(cache_key, results, self.search_cache_ttl)
            logger.debug(f"Cached hybrid search results for collection {collection_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"RRF hybrid search failed: {e}")
            return []
    
    def _combine_with_rrf(self, vector_results, text_results, k: int = 60) -> List[Dict]:
        """Combine search results using Reciprocal Rank Fusion"""
        score_dict = {}
        
        # Process vector results
        for i, result in enumerate(vector_results):
            point_id = result["id"]
            rrf_score = 1.0 / (k + i + 1)
            score_dict[point_id] = score_dict.get(point_id, 0) + rrf_score
        
        # Process text results
        for i, result in enumerate(text_results):
            point_id = result["id"]
            rrf_score = 1.0 / (k + i + 1)
            score_dict[point_id] = score_dict.get(point_id, 0) + rrf_score
        
        # Sort by combined RRF score
        sorted_ids = sorted(score_dict.keys(), key=lambda x: score_dict[x], reverse=True)
        
        # Create result list maintaining point data
        combined_results = []
        point_lookup = {r["id"]: r for r in vector_results + text_results}
        
        for point_id in sorted_ids:
            if point_id in point_lookup:
                result = point_lookup[point_id]
                result["rrf_score"] = score_dict[point_id]
                combined_results.append(result)
        
        return combined_results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant service health"""
        try:
            if not self.initialized:
                return {"healthy": False, "message": "Service not initialized"}
            
            collections_response = await self.client.get_collections()
            # Handle different API response structures (v1.12 vs v1.15+)
            if hasattr(collections_response, 'collections'):
                collections = collections_response.collections
            elif hasattr(collections_response, 'result') and hasattr(collections_response.result, 'collections'):
                collections = collections_response.result.collections
            else:
                collections = []
            
            return {
                "healthy": True,
                "collections_count": len(collections),
                "endpoint": f"{self.host}:{self.grpc_port}"
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def get_collections(self) -> List[str]:
        """Get list of all collection names with intelligent caching"""
        if not self.initialized or not self.client:
            return []
        
        # Generate cache key for collections list
        cache_key = self._generate_cache_key("get_collections")
        
        # Try cache first
        cached_collections = await self._get_from_cache(cache_key)
        if cached_collections is not None:
            logger.debug("Collections list cache hit")
            return cached_collections
            
        try:
            collections_response = await self.client.get_collections()
            # Handle different API response structures (v1.12 vs v1.15+)
            if hasattr(collections_response, 'collections'):
                collections = [col.name for col in collections_response.collections]
            elif hasattr(collections_response, 'result') and hasattr(collections_response.result, 'collections'):
                collections = [col.name for col in collections_response.result.collections]
            else:
                collections = []
            
            # Cache the collections list
            await self._store_in_cache(cache_key, collections, self.collection_cache_ttl)
            logger.debug("Cached collections list")
            
            return collections
            
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []
    
    async def get_collection(self, collection_name: str):
        """Get collection information (raw client method wrapper)"""
        if not self.initialized or not self.client:
            raise RuntimeError("Qdrant service not initialized")
            
        try:
            return await self.client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        if not self.initialized or not self.client:
            raise RuntimeError("Qdrant service not initialized")
            
        try:
            await self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    async def delete_points(
        self,
        collection_name: str,
        points_selector: Optional[List[str]] = None,
        filter_conditions: Optional[Filter] = None
    ) -> Dict[str, Any]:
        """Delete points by IDs or filter conditions
        
        Args:
            collection_name: Name of the collection
            points_selector: List of point IDs to delete
            filter_conditions: Qdrant Filter object for conditional deletion
            
        Returns:
            Dict with operation status and details
        """
        if not self.initialized or not self.client:
            return {"status": "error", "message": "Service not initialized"}

        if not points_selector and not filter_conditions:
            return {"status": "error", "message": "Either points_selector or filter_conditions must be provided"}

        try:
            # Import required models
            from qdrant_client.models import PointIdsList, FilterSelector, Filter as QFilter

            # Determine selector type (accept PointIdsList or list[str])
            selector = None
            if points_selector is not None:
                if isinstance(points_selector, PointIdsList):
                    selector = points_selector
                    logger.info(f"Deleting points by IDs (PointIdsList) from {collection_name}")
                elif isinstance(points_selector, list):
                    selector = PointIdsList(points=points_selector)
                    logger.info(f"Deleting {len(points_selector)} points by IDs from {collection_name}")
                else:
                    return {"status": "error", "message": "Unsupported points_selector type"}
            else:
                # Delete by filter conditions (accept FilterSelector, Filter, or dict)
                if isinstance(filter_conditions, FilterSelector):
                    selector = filter_conditions
                elif isinstance(filter_conditions, QFilter):
                    selector = FilterSelector(filter=filter_conditions)
                elif isinstance(filter_conditions, dict):
                    # Best-effort conversion from dict to Qdrant Filter
                    selector = FilterSelector(filter=QFilter(**filter_conditions))
                else:
                    return {"status": "error", "message": "Unsupported filter_conditions type"}

            # Perform deletion
            operation_info = await self.client.delete(
                collection_name=collection_name,
                points_selector=selector,
                wait=True
            )

            return {
                "status": "success",
                "operation_id": getattr(operation_info, 'operation_id', None),
                "selector_type": "ids" if points_selector is not None else "filter"
            }

        except Exception as e:
            logger.error(f"Delete points failed for {collection_name}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information and statistics with intelligent caching"""
        if not self.initialized or not self.client:
            return {"exists": False, "message": "Service not initialized"}
        
        # Generate cache key for collection info
        cache_key = self._generate_cache_key("get_collection_info", collection_name=collection_name)
        
        # Try cache first
        cached_info = await self._get_from_cache(cache_key)
        if cached_info is not None:
            logger.debug(f"Collection info cache hit for {collection_name}")
            return cached_info
            
        try:
            collection_info = await self.client.get_collection(collection_name)
            info = {
                "exists": True,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                }
            }
            
            # Cache collection info with shorter TTL since it changes frequently
            await self._store_in_cache(cache_key, info, self.default_ttl // 2)
            logger.debug(f"Cached collection info for {collection_name}")
            
            return info
            
        except Exception as e:
            logger.warning(f"Collection {collection_name} info failed: {e}")
            error_info = {"exists": False, "error": str(e)}
            # Cache errors briefly to avoid hammering failed collections
            await self._store_in_cache(cache_key, error_info, 300)  # 5 minutes
            return error_info
