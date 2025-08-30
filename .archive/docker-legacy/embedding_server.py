#!/usr/bin/env python3
"""
L9 Neural Flow - Embedding Server
FastAPI server that exposes embedding models from neural-flow container
"""

import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Neural Flow embedding imports (should be available in neural-flow:l9-production)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
embedding_model = None

class EmbedRequest(BaseModel):
    inputs: List[str]
    normalize: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize embedding model on startup"""
    global embedding_model
    
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    logger.info(f"Loading embedding model: {model_name}")
    
    try:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            embedding_model = SentenceTransformer(model_name)
            logger.info(f"✅ Loaded SentenceTransformer: {model_name}")
        else:
            logger.error("❌ SentenceTransformers not available")
            raise RuntimeError("No embedding backend available")
            
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down embedding server")

app = FastAPI(
    title="L9 Neural Flow Embedding Server",
    description="Shared embedding service for L9 MCP system",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
        "backend": "sentence-transformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "unknown"
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for input texts"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.inputs:
        raise HTTPException(status_code=400, detail="No input texts provided")
    
    if len(request.inputs) > int(os.getenv("MAX_BATCH_SIZE", "100")):
        raise HTTPException(status_code=400, detail="Batch size too large")
    
    try:
        start_time = time.time()
        
        # Generate embeddings
        embeddings = embedding_model.encode(
            request.inputs,
            normalize_embeddings=request.normalize,
            convert_to_numpy=True
        )
        
        # Convert to lists for JSON serialization
        if NUMPY_AVAILABLE and isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
        
        inference_time = time.time() - start_time
        
        logger.info(f"Generated {len(embeddings_list)} embeddings in {inference_time:.3f}s")
        
        return EmbedResponse(
            embeddings=embeddings_list,
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            usage={
                "prompt_tokens": sum(len(text.split()) for text in request.inputs),
                "total_tokens": sum(len(text.split()) for text in request.inputs),
                "inference_time_ms": int(inference_time * 1000)
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def model_info():
    """Get model information"""
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    
    # Try to get embedding dimension
    try:
        test_embedding = embedding_model.encode(["test"])
        embedding_dim = len(test_embedding[0]) if len(test_embedding) > 0 else "unknown"
    except:
        embedding_dim = "unknown"
    
    return {
        "model_name": model_name,
        "embedding_dimension": embedding_dim,
        "backend": "sentence-transformers" if SENTENCE_TRANSFORMERS_AVAILABLE else "unknown",
        "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", "100")),
        "normalize_default": True
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    )