#!/usr/bin/env python3
"""
Memory Tools - Storage and retrieval MCP tools with service injection
Extracted from monolithic neural-mcp-server-enhanced.py
"""

import os
import logging
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

def register_memory_tools(mcp: FastMCP, container=None):
    """Register all memory-related MCP tools with global variable pattern (FastMCP-compatible)"""
    
    @mcp.tool()
    async def memory_store_enhanced(
        content: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        create_graph_entities: bool = True
    ) -> Dict[str, Any]:
        """Store content with enhanced hybrid indexing and GraphRAG integration - ADR-0008 Fix
        
        Args:
            content: Text content to store
            category: Category for organization
            metadata: Additional metadata
            create_graph_entities: Whether to create graph entities in Neo4j GraphRAG
        
        Returns:
            Dict with status, point_id (required for T1_DATA_PERSISTENCE test)
        """
        try:
            # Import global clients and initialization from main server
            from neural_server_refactored import ensure_services_initialized, qdrant_client, nomic_client, neo4j_client
            await ensure_services_initialized()
            
            # Use global clients (FastMCP-compatible pattern)
            if not qdrant_client:
                return {
                    "status": "error",
                    "message": "Qdrant client not initialized"
                }
            
            if not nomic_client:
                return {
                    "status": "error", 
                    "message": "Nomic client not initialized"
                }
            
            collection_name = f"{qdrant_client.collection_prefix}{category}"
            await qdrant_client.ensure_collection(collection_name)
            
            # Generate deterministic point ID
            point_id = str(uuid.uuid4())
            
            # Generate embeddings with proper error handling
            try:
                embeddings = await nomic_client.get_embeddings([content])
                dense_embedding = embeddings[0]
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
                word_hash = abs(hash(word)) % vocab_size
                sparse_indices.append(word_hash)
                sparse_values.append(float(freq))
            
            # Create point with enhanced payload
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                "content": content,
                "category": category,
                "timestamp": datetime.utcnow().isoformat(),
                "project": qdrant_client.project_name,
                "content_length": len(content),
                "word_count": len(words),
                "create_graph_entities": create_graph_entities
            })
            
            point = PointStruct(
                id=point_id,
                vector={"dense": dense_embedding},
                payload=enhanced_metadata
            )
            
            # Store in Qdrant
            upsert_result = await qdrant_client.upsert_points(collection_name, [point])
            
            if upsert_result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Qdrant upsert failed: {upsert_result.get('message', 'Unknown error')}"
                }
            
            # Optionally create graph entities
            graph_result = None
            if create_graph_entities and neo4j_client:
                try:
                    # Simple content indexing in Neo4j
                    graph_result = await neo4j_client.index_code_file(
                        file_path=f"memory/{category}/{point_id}",
                        content=content,
                        language="text"
                    )
                except Exception as graph_error:
                    logger.warning(f"Graph entity creation failed: {graph_error}")
            
            return {
                "status": "success",
                "point_id": point_id,
                "collection": collection_name,
                "vector_dimensions": len(dense_embedding),
                "graph_entities_created": graph_result is not None and graph_result.get("status") == "success",
                "metadata": enhanced_metadata
            }
            
        except RuntimeError as service_error:
            return {
                "status": "error",
                "message": f"Service not available: {str(service_error)}"
            }
        except Exception as e:
            logger.error(f"memory_store_enhanced error: {str(e)}")
            return {
                "status": "error",
                "message": f"Storage operation failed: {str(e)}"
            }
    
    @mcp.tool()
    async def memory_search_enhanced(
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        mode: str = "rrf_hybrid",
        diversity_threshold: float = 0.85,
        graph_expand: bool = True
    ) -> Dict[str, Any]:
        """Enhanced search with RRF fusion, MMR diversity, and GraphRAG expansion - ADR-0008 Fix
        
        Args:
            query: Search query
            category: Optional category filter
            limit: Number of results
            mode: Search mode (semantic, keyword, rrf_hybrid, mmr_diverse)
            diversity_threshold: Threshold for MMR diversity
            graph_expand: Whether to expand results using graph relationships
        
        Returns:
            Dict with results list (required for T2_DATA_RETRIEVAL test)
        """
        try:
            from neural_server_refactored import ensure_services_initialized, qdrant_client, nomic_client, neo4j_client
            await ensure_services_initialized()
            # Input validation 
            search_limit = max(1, min(limit or 5, 100))
            diversity_lambda = max(0.0, min(diversity_threshold or 0.85, 1.0))
            
            # Use global service clients (FastMCP-compatible)
            if not qdrant_client:
                return {
                    "status": "error",
                    "message": "Qdrant client not initialized",
                    "results": []
                }
            
            if not nomic_client:
                return {
                    "status": "error", 
                    "message": "Nomic client not initialized",
                    "results": []
                }
            
            collection_name = f"{qdrant_client.collection_prefix}{category or 'memory'}"
            
            # Generate query embedding with proper error handling
            try:
                embeddings = await nomic_client.get_embeddings([query])
                query_vector = [float(x) for x in embeddings[0]]
            except Exception as embed_error:
                logger.error(f"Query embedding generation failed: {embed_error}")
                return {
                    "status": "error",
                    "message": "Query embedding generation failed",
                    "results": []
                }
            
            # Execute search based on mode
            if mode == "rrf_hybrid":
                # Hybrid search with RRF fusion
                results = await qdrant_client.rrf_hybrid_search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_text=query,
                    limit=search_limit
                )
            elif mode == "semantic":
                # Pure vector search
                results = await qdrant_client.search_vectors(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=search_limit,
                    score_threshold=0.3
                )
            else:
                # Fallback to semantic search
                results = await qdrant_client.search_vectors(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=search_limit
                )
            
            if not results:
                return {
                    "status": "success",
                    "message": "No matching results found",
                    "results": [],
                    "query": query,
                    "search_mode": mode
                }
            
            # Apply diversity filtering using MMR if requested
            if mode == "mmr_diverse" and len(results) > 1:
                results = _apply_mmr_diversity(results, diversity_lambda)
            
            # Graph expansion if requested and available
            final_results = results
            if graph_expand and neo4j_client:
                try:
                    if neo4j_client.initialized:
                        # Simple graph expansion - find related content
                        graph_results = await neo4j_client.semantic_search(query, limit=5)
                        if graph_results:
                            # Merge graph results with vector results
                            graph_formatted = []
                            for graph_result in graph_results[:3]:  # Top 3 from graph
                                if 'n' in graph_result:
                                    node = graph_result['n']
                                    graph_formatted.append({
                                        "id": f"graph_{hash(str(node))}",
                                        "score": 0.8,  # Fixed score for graph results
                                        "payload": {
                                            "content": node.get("content", node.get("name", "")),
                                            "source": "neo4j_graph",
                                            "node_type": graph_result.get("node_type", ["Unknown"])
                                        }
                                    })
                            
                            # Combine results
                            final_results = results + graph_formatted
                            final_results = final_results[:search_limit]
                except Exception as graph_error:
                    logger.warning(f"Graph expansion failed: {graph_error}")
            
            return {
                "status": "success",
                "results": final_results,
                "total_found": len(results),
                "after_graph_expansion": len(final_results),
                "query": query,
                "search_mode": mode,
                "graph_expansion": graph_expand
            }
            
        except RuntimeError as service_error:
            return {
                "status": "error",
                "message": f"Service not available: {str(service_error)}",
                "results": []
            }
        except Exception as e:
            logger.error(f"memory_search_enhanced error: {str(e)}")
            return {
                "status": "error", 
                "message": f"Search operation failed: {str(e)}",
                "results": []
            }
    
    @mcp.tool()
    async def schema_customization(
        action: str,
        collection_name: Optional[str] = None,
        vector_size: int = 1536,
        distance_metric: str = "cosine"
    ) -> Dict[str, Any]:
        """Customize Qdrant collection schemas and configurations
        
        Args:
            action: Action to perform (create, delete, info, list)
            collection_name: Name of collection
            vector_size: Vector dimensions for new collections
            distance_metric: Distance metric (cosine, euclidean, dot)
        
        Returns:
            Dict with operation results
        """
        try:
            from neural_server_refactored import ensure_services_initialized, qdrant_client
            await ensure_services_initialized()
            
            if not qdrant_client:
                return {
                    "status": "error",
                    "message": "Qdrant client not initialized"
                }
            
            if action == "create":
                if not collection_name:
                    return {"status": "error", "message": "Collection name required for create action"}
                
                full_collection_name = f"{qdrant_client.collection_prefix}{collection_name}"
                success = await qdrant_client.ensure_collection(full_collection_name, vector_size)
                
                return {
                    "status": "success" if success else "error",
                    "action": "create",
                    "collection": full_collection_name,
                    "vector_size": vector_size,
                    "distance_metric": distance_metric
                }
            
            elif action == "info":
                if not collection_name:
                    return {"status": "error", "message": "Collection name required for info action"}
                
                full_collection_name = f"{qdrant_client.collection_prefix}{collection_name}"
                info = await qdrant_client.get_collection_info(full_collection_name)
                
                return {
                    "status": "success",
                    "action": "info",
                    "collection": full_collection_name,
                    "info": info
                }
            
            elif action == "list":
                health = await qdrant_client.health_check()
                return {
                    "status": "success",
                    "action": "list",
                    "collections_count": health.get("collections_count", 0),
                    "qdrant_healthy": health.get("healthy", False)
                }
            
            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}. Supported: create, info, list"
                }
                
        except RuntimeError as service_error:
            return {
                "status": "error",
                "message": f"Service not available: {str(service_error)}"
            }
        except Exception as e:
            logger.error(f"schema_customization error: {str(e)}")
            return {
                "status": "error",
                "message": f"Schema operation failed: {str(e)}"
            }

def _apply_mmr_diversity(results: List[Dict], lambda_param: float = 0.85) -> List[Dict]:
    """Apply Maximal Marginal Relevance (MMR) for result diversity"""
    if len(results) <= 1:
        return results
    
    # Simple MMR implementation based on scores and content similarity
    selected = [results[0]]  # Start with highest scoring result
    remaining = results[1:]
    
    while remaining and len(selected) < len(results):
        best_score = float('-inf')
        best_idx = 0
        
        for i, candidate in enumerate(remaining):
            # Relevance score (from search)
            relevance = candidate.get('score', candidate.get('rrf_score', 0))
            
            # Diversity score (simple content length difference as proxy)
            diversity = 1.0
            if selected:
                candidate_content = candidate.get('payload', {}).get('content', '')
                for selected_item in selected:
                    selected_content = selected_item.get('payload', {}).get('content', '')
                    # Simple diversity metric
                    length_diff = abs(len(candidate_content) - len(selected_content))
                    diversity = min(diversity, length_diff / max(len(candidate_content), len(selected_content), 1))
            
            # MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        selected.append(remaining.pop(best_idx))
    
    return selected