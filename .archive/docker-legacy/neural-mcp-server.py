#!/usr/bin/env python3
"""
L9 Neural MCP Server - Production Docker Version
Runs entirely in Docker with Qdrant hybrid search
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# MCP SDK
from mcp.server.fastmcp import FastMCP
import mcp.types as types

# Qdrant client for hybrid search
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from fastembed import TextEmbedding, SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("l9-neural")

# Global clients
qdrant_client = None
dense_model = None
sparse_model = None

# Constants
QDRANT_HOST = os.environ.get('QDRANT_HOST', 'qdrant')
QDRANT_GRPC_PORT = int(os.environ.get('QDRANT_GRPC_PORT', 6334))
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'default')
COLLECTION_PREFIX = f"project_{PROJECT_NAME}_"

async def initialize():
    """Initialize Qdrant and embedding models"""
    global qdrant_client, dense_model, sparse_model
    
    try:
        # Connect to Qdrant using gRPC (3-4x faster)
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_GRPC_PORT,
            prefer_grpc=True
        )
        
        # Initialize embedding models
        dense_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # Ensure default collection exists
        await ensure_collection(f"{COLLECTION_PREFIX}memory")
        
        logger.info(f"✅ MCP Server initialized - Project: {PROJECT_NAME}")
        logger.info(f"📍 Qdrant: {QDRANT_HOST}:{QDRANT_GRPC_PORT}")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

async def ensure_collection(collection_name: str):
    """Ensure collection exists with hybrid search config"""
    try:
        collections = qdrant_client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            # Create with both dense and sparse vectors
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to ensure collection: {e}")

@mcp.tool()
async def memory_store(
    content: str,
    category: str = "general",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Store content in neural memory with hybrid indexing
    
    Args:
        content: Text content to store
        category: Category for organization (general, code, docs, etc.)
        metadata: Additional metadata to store
    """
    try:
        collection_name = f"{COLLECTION_PREFIX}{category}"
        await ensure_collection(collection_name)
        
        # Generate both dense and sparse embeddings
        dense_embedding = list(dense_model.embed([content]))[0]
        sparse_embedding = list(sparse_model.embed([content]))[0]
        
        # Convert sparse embedding to Qdrant format
        sparse_vector = models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )
        
        # Store in Qdrant
        point_id = hash(content + str(datetime.now()))
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=abs(point_id) % (10 ** 8),
                    vector={"dense": dense_embedding.tolist()},
                    sparse_vector={"sparse": sparse_vector},
                    payload={
                        "content": content,
                        "category": category,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {})
                    }
                )
            ]
        )
        
        return {
            "status": "success",
            "id": abs(point_id) % (10 ** 8),
            "collection": collection_name
        }
        
    except Exception as e:
        logger.error(f"Store failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def memory_search(
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
    mode: str = "hybrid"
) -> List[Dict[str, Any]]:
    """Search neural memory using semantic, keyword, or hybrid search
    
    Args:
        query: Search query
        category: Optional category filter
        limit: Number of results
        mode: Search mode (semantic, keyword, hybrid)
    """
    try:
        collection_name = f"{COLLECTION_PREFIX}{category or 'memory'}"
        
        # Generate embeddings based on mode
        results = []
        
        if mode in ["semantic", "hybrid"]:
            # Dense vector search
            dense_embedding = list(dense_model.embed([query]))[0]
            semantic_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=("dense", dense_embedding.tolist()),
                limit=limit
            )
            results.extend([
                {
                    "content": hit.payload.get("content"),
                    "score": hit.score,
                    "type": "semantic",
                    **hit.payload
                }
                for hit in semantic_results
            ])
        
        if mode in ["keyword", "hybrid"]:
            # Sparse vector search (BM25-like)
            sparse_embedding = list(sparse_model.embed([query]))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
            
            keyword_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=models.NamedSparseVector(
                    name="sparse",
                    vector=sparse_vector
                ),
                limit=limit
            )
            
            # Merge results if hybrid
            if mode == "hybrid":
                # Deduplicate and combine scores
                seen_content = {r["content"] for r in results}
                for hit in keyword_results:
                    content = hit.payload.get("content")
                    if content not in seen_content:
                        results.append({
                            "content": content,
                            "score": hit.score,
                            "type": "keyword",
                            **hit.payload
                        })
            else:
                results = [
                    {
                        "content": hit.payload.get("content"),
                        "score": hit.score,
                        "type": "keyword",
                        **hit.payload
                    }
                    for hit in keyword_results
                ]
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

@mcp.tool()
async def code_index(
    path: str,
    extensions: List[str] = [".py", ".js", ".ts", ".jsx", ".tsx"],
    chunk_size: int = 500
) -> Dict[str, Any]:
    """Index code files for semantic search
    
    Args:
        path: Directory path to index (mounted at /app/project)
        extensions: File extensions to index
        chunk_size: Lines per chunk for indexing
    """
    try:
        indexed_count = 0
        project_path = Path("/app/project") / path
        
        if not project_path.exists():
            return {"status": "error", "message": f"Path not found: {path}"}
        
        collection_name = f"{COLLECTION_PREFIX}code"
        await ensure_collection(collection_name)
        
        # Walk through files
        for file_path in project_path.rglob("*"):
            if file_path.suffix in extensions:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    # Chunk the file
                    for i in range(0, len(lines), chunk_size):
                        chunk = '\n'.join(lines[i:i+chunk_size])
                        if chunk.strip():
                            await memory_store(
                                content=chunk,
                                category="code",
                                metadata={
                                    "file_path": str(file_path.relative_to(Path("/app/project"))),
                                    "line_start": i + 1,
                                    "line_end": min(i + chunk_size, len(lines)),
                                    "language": file_path.suffix[1:]
                                }
                            )
                            indexed_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")
        
        return {
            "status": "success",
            "indexed_chunks": indexed_count,
            "collection": collection_name
        }
        
    except Exception as e:
        logger.error(f"Code indexing failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def collection_stats() -> Dict[str, Any]:
    """Get statistics for all collections"""
    try:
        collections = qdrant_client.get_collections().collections
        stats = {}
        
        for collection in collections:
            if collection.name.startswith(COLLECTION_PREFIX):
                info = qdrant_client.get_collection(collection.name)
                stats[collection.name] = {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status
                }
        
        return {
            "project": PROJECT_NAME,
            "collections": stats,
            "total_points": sum(s["points_count"] for s in stats.values())
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def collection_clear(category: str) -> Dict[str, Any]:
    """Clear a specific collection
    
    Args:
        category: Collection category to clear
    """
    try:
        collection_name = f"{COLLECTION_PREFIX}{category}"
        qdrant_client.delete_collection(collection_name)
        await ensure_collection(collection_name)
        
        return {
            "status": "success",
            "message": f"Cleared collection: {collection_name}"
        }
        
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        return {"status": "error", "message": str(e)}

# Run the server
if __name__ == "__main__":
    asyncio.run(initialize())
    mcp.run(transport='stdio')