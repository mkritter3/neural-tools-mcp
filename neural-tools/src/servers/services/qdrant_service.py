#!/usr/bin/env python3
"""
Qdrant Vector Database Service - Extracted from monolithic neural-mcp-server-enhanced.py
Provides vector storage and retrieval operations with hybrid search capabilities
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, Distance, CollectionStatus,
    SearchRequest, Filter, FieldCondition, MatchValue
)

logger = logging.getLogger(__name__)

class QdrantService:
    """Service class for Qdrant vector database operations"""
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.collection_prefix = f"project_{project_name}_"
        
        # Qdrant configuration from environment
        self.host = os.environ.get('QDRANT_HOST', 'default-neural-storage')
        self.http_port = int(os.environ.get('QDRANT_HTTP_PORT', 6333))
        self.grpc_port = int(os.environ.get('QDRANT_GRPC_PORT', 6334))
        
        self.client = None
        self.initialized = False
        
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
                    return {
                        "success": False, 
                        "message": f"Qdrant HTTP connectivity failed: {response.status_code}"
                    }
            
            # Initialize GRPC client for operations
            self.client = QdrantClient(
                host=self.host,
                port=self.grpc_port, 
                prefer_grpc=True
            )
            
            # Test client operations
            collections = self.client.get_collections()
            self.initialized = True
            
            return {
                "success": True,
                "message": "Qdrant service initialized successfully",
                "collections_count": len(collections.collections),
                "grpc_endpoint": f"{self.host}:{self.grpc_port}"
            }
            
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            return {
                "success": False,
                "message": f"Qdrant initialization failed: {str(e)}"
            }
    
    async def ensure_collection(self, collection_name: str, vector_size: int = 1536) -> bool:
        """Ensure collection exists with proper configuration"""
        if not self.initialized or not self.client:
            raise RuntimeError("Qdrant service not initialized")
            
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                # Create collection with optimized settings (named vectors)
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=vector_size,
                            distance=Distance.COSINE
                        )
                    },
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
            logger.error(f"Collection setup failed for {collection_name}: {e}")
            return False
    
    async def upsert_points(
        self, 
        collection_name: str, 
        points: List[PointStruct]
    ) -> Dict[str, Any]:
        """Upsert points with proper error handling"""
        if not self.initialized or not self.client:
            raise RuntimeError("Qdrant service not initialized")
            
        try:
            operation_info = self.client.upsert(
                collection_name=collection_name,
                points=points,
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
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        if not self.initialized or not self.client:
            raise RuntimeError("Qdrant service not initialized")
            
        try:
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=("dense", query_vector),
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
                
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed for {collection_name}: {e}")
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
            scroll_result = self.client.scroll(
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
        """Perform RRF hybrid search combining vector and text search"""
        try:
            # Vector search
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
            return self._combine_with_rrf(vector_results, text_results, k=k)[:limit]
            
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
            
            collections = self.client.get_collections()
            return {
                "healthy": True,
                "collections_count": len(collections.collections),
                "endpoint": f"{self.host}:{self.grpc_port}"
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information and statistics"""
        if not self.initialized or not self.client:
            return {"exists": False, "message": "Service not initialized"}
            
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                "exists": True,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance)
                }
            }
            
        except Exception as e:
            logger.warning(f"Collection {collection_name} info failed: {e}")
            return {"exists": False, "error": str(e)}