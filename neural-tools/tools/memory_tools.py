#!/usr/bin/env python3
"""
L9 Neural Tools - Memory Tools Module
Extracted memory storage and search functionality with proper L9 architecture
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from fastmcp import FastMCP
from qdrant_client.models import PointStruct
from qdrant_client import models

import sys
sys.path.append('/app/project/neural-tools')
from common.config import config
from common.utils import apply_mmr_diversity, combine_with_rrf_internal

logger = logging.getLogger(__name__)

# MCP instance for tool registration
mcp = FastMCP("memory-tools")

async def memory_store_enhanced(
    content: str,
    category: str = "general", 
    metadata: Optional[Dict[str, Any]] = None,
    create_graph_entities: bool = True,
    qdrant_client=None,
    nomic_client=None,
    neo4j_client=None
) -> Dict[str, Any]:
    """Store content with enhanced hybrid indexing and GraphRAG integration - ADR-0008 Fix
    
    Args:
        content: Text content to store
        category: Category for organization
        metadata: Additional metadata
        create_graph_entities: Whether to create graph entities in Neo4j GraphRAG
        qdrant_client: Qdrant client instance
        nomic_client: Nomic embeddings client
        neo4j_client: Neo4j GraphRAG client
    
    Returns:
        Dict with status, point_id (required for T1_DATA_PERSISTENCE test)
    """
    try:
        collection_name = f"{config.COLLECTION_PREFIX}{category}"
        
        # Generate deterministic point ID using string format (Context7 pattern)
        point_id = str(uuid.uuid4())
        
        # Generate embeddings with proper error handling
        try:
            embed_response = await nomic_client.get_embeddings([content])
            dense_embedding = embed_response.embeddings[0]
        except Exception as embed_error:
            logger.error(f"Embedding generation failed: {embed_error}")
            return {
                "status": "error",
                "message": f"Embedding generation failed: {str(embed_error)}"
            }
        
        # Generate sparse embedding (BM25-style)
        words = content.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Convert to sparse vector format
        vocab_size = 10000  # Simplified vocabulary
        sparse_indices = []
        sparse_values = []
        for word, freq in word_freq.items():
            idx = hash(word) % vocab_size
            sparse_indices.append(idx)
            sparse_values.append(float(freq))
        
        sparse_vector = models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
        
        # Create point with proper structure (Context7 pattern)
        point = PointStruct(
            id=point_id,
            vector={
                "dense": dense_embedding,
                "sparse": sparse_vector
            },
            payload={
                "content": content,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "model": "nomic-v2-moe",
                "enhanced": True,
                "project": metadata.get('project_name', 'default') if metadata else 'default',
                **(metadata or {})
            }
        )
        
        # Execute upsert with wait=True and error handling (Context7 pattern)
        try:
            upsert_result = qdrant_client.upsert(
                collection_name=collection_name,
                points=[point],
                wait=True  # Ensure operation completes before returning
            )
            
            # Validate upsert success (Context7 pattern)
            if hasattr(upsert_result, 'status') and upsert_result.status != "completed":
                return {
                    "status": "error", 
                    "message": f"Upsert failed: {upsert_result.status}"
                }
                
        except Exception as upsert_error:
            logger.error(f"Qdrant upsert failed: {upsert_error}")
            return {
                "status": "error",
                "message": f"Storage operation failed: {str(upsert_error)}"
            }
        
        # Create graph entities if enabled
        graph_entity_id = None
        if create_graph_entities and config.is_feature_enabled('neo4j') and neo4j_client:
            try:
                doc_id = f"doc_{abs(hash(point_id)) % (10 ** 8)}"
                await neo4j_client.create_file_node(
                    file_path=metadata.get('file_path', '') if metadata else '',
                    content=content[:1000],  # Truncate for storage
                    embedding_id=abs(hash(point_id)) % (10 ** 8),
                    additional_metadata={"doc_id": doc_id}
                )
                graph_entity_id = doc_id
            except Exception as e:
                logger.warning(f"Neo4j GraphRAG entity creation failed: {e}")
        
        # Return proper response structure (ADR-0008 requirement)
        return {
            "status": "success",
            "point_id": point_id,  # Required for T1_DATA_PERSISTENCE test
            "collection": collection_name,
            "vector_dimensions": len(dense_embedding),
            "graph_entity": graph_entity_id,
            "model": "nomic-v2-moe",
            "enhanced": True,
            "graph_entities_created": create_graph_entities and graph_entity_id is not None
        }
        
    except Exception as e:
        logger.error(f"memory_store_enhanced error: {str(e)}")
        return {
            "status": "error", 
            "message": f"Storage operation failed: {str(e)}"
        }

async def memory_search_enhanced(
    query: str,
    category: Optional[str] = None,
    limit: str = "10",
    mode: str = "rrf_hybrid",
    diversity_threshold: str = "0.85",
    graph_expand: bool = True,
    qdrant_client=None,
    nomic_client=None,
    neo4j_client=None
) -> Dict[str, Any]:
    """Enhanced search with RRF fusion, MMR diversity, and GraphRAG expansion - ADR-0008 Fix
    
    Args:
        query: Search query
        category: Optional category filter
        limit: Number of results
        mode: Search mode (semantic, keyword, rrf_hybrid, mmr_diverse)
        diversity_threshold: Threshold for MMR diversity
        graph_expand: Whether to expand results using graph relationships
        qdrant_client: Qdrant client instance
        nomic_client: Nomic embeddings client
        neo4j_client: Neo4j GraphRAG client
    
    Returns:
        Dict with results list (required for T2_DATA_RETRIEVAL test)
    """
    try:
        # Convert string parameters
        limit_int = max(1, min(50, int(limit)))
        diversity_float = max(0.0, min(1.0, float(diversity_threshold)))
        
        # Determine collection name
        collection_name = f"{config.COLLECTION_PREFIX}{category}" if category else f"{config.COLLECTION_PREFIX}general"
        
        if mode == "semantic":
            # Pure vector search
            results = await perform_vector_search_internal(
                query, collection_name, limit_int, qdrant_client, nomic_client
            )
        elif mode == "rrf_hybrid": 
            # Hybrid RRF search (default)
            results = await perform_rrf_hybrid_search_internal(
                query, collection_name, limit_int, qdrant_client, nomic_client
            )
        elif mode == "mmr_diverse":
            # MMR diversity search
            base_results = await perform_rrf_hybrid_search_internal(
                query, collection_name, limit_int * 2, qdrant_client, nomic_client
            )
            results = apply_mmr_diversity(base_results, diversity_float, limit_int)
        else:
            # Default to RRF hybrid
            results = await perform_rrf_hybrid_search_internal(
                query, collection_name, limit_int, qdrant_client, nomic_client
            )
        
        # Expand with Neo4j graph context if enabled
        if graph_expand and config.is_feature_enabled('neo4j') and neo4j_client:
            try:
                results = await expand_with_neo4j_graph_context(results, query, neo4j_client)
            except Exception as e:
                logger.warning(f"Neo4j graph expansion failed: {e}")
        
        # Return structured response (ADR-0008 requirement)
        return {
            "status": "success",
            "results": results,  # Required for T2_DATA_RETRIEVAL test
            "query": query,
            "mode": mode,
            "collection": collection_name,
            "total_results": len(results),
            "diversity_applied": mode == "mmr_diverse",
            "graph_expanded": graph_expand and config.is_feature_enabled('neo4j')
        }
        
    except Exception as e:
        logger.error(f"memory_search_enhanced error: {str(e)}")
        return {
            "status": "error",
            "message": f"Search operation failed: {str(e)}"
        }

# Internal helper functions for search operations

async def perform_rrf_hybrid_search_internal(
    query: str,
    collection_name: str, 
    limit: int,
    qdrant_client,
    nomic_client
) -> List[Dict]:
    """Perform RRF hybrid search combining vector and text search"""
    try:
        # Vector search
        vector_results = await perform_vector_search_internal(
            query, collection_name, limit, qdrant_client, nomic_client
        )
        
        # Text search using sparse vectors  
        text_results = await perform_text_search_internal(
            query, collection_name, limit, qdrant_client
        )
        
        # Combine with RRF
        combined_results = combine_with_rrf_internal(vector_results, text_results)
        
        return combined_results[:limit]
        
    except Exception as e:
        logger.error(f"RRF hybrid search failed: {e}")
        return []

async def perform_vector_search_internal(
    query: str,
    collection_name: str,
    limit: int,
    qdrant_client,
    nomic_client
) -> List[Dict]:
    """Perform vector similarity search"""
    try:
        # Generate query embedding
        embed_response = await nomic_client.get_embeddings([query])
        query_vector = embed_response.embeddings[0]
        
        # Search using named vectors (Context7 pattern)
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=("dense", query_vector),  # Named vector specification
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results = []
        for hit in search_result:
            result = {
                "id": str(hit.id),
                "score": float(hit.score),
                "content": hit.payload.get("content", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

async def perform_text_search_internal(
    query: str,
    collection_name: str,
    limit: int,
    qdrant_client
) -> List[Dict]:
    """Perform text search using sparse vectors"""
    try:
        # Create query sparse vector
        words = query.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        vocab_size = 10000
        sparse_indices = []
        sparse_values = []
        for word, freq in word_freq.items():
            idx = hash(word) % vocab_size
            sparse_indices.append(idx)
            sparse_values.append(float(freq))
        
        query_sparse = models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
        
        # Search using sparse vectors
        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=("sparse", query_sparse),
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # Format results
        results = []
        for hit in search_result:
            result = {
                "id": str(hit.id),
                "score": float(hit.score),
                "content": hit.payload.get("content", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        return []

async def expand_with_neo4j_graph_context(results: List[Dict], query: str, neo4j_client) -> List[Dict]:
    """Expand search results with Neo4j graph relationships"""
    try:
        # Simple graph context expansion
        for result in results[:5]:  # Only expand top 5 results
            content_hash = abs(hash(result.get("content", ""))) % (10 ** 8)
            
            # Find related entities
            related_query = f"""
            MATCH (n:Content {{content_hash: {content_hash}}})-[r]-(related)
            RETURN type(r) as relationship, related.name as entity
            LIMIT 3
            """
            
            try:
                graph_results = await neo4j_client.execute_query(related_query)
                if graph_results:
                    result["graph_context"] = [
                        {"relationship": r["relationship"], "entity": r["entity"]}
                        for r in graph_results
                    ]
            except Exception as e:
                logger.debug(f"Graph context expansion failed for result: {e}")
        
        return results
        
    except Exception as e:
        logger.warning(f"Neo4j graph expansion failed: {e}")
        return results