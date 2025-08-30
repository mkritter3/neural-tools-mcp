#!/usr/bin/env python3
"""
Shared Model Server
Serves Qodo-Embed-1.5B and BM25 models to all projects
Saves 90%+ memory by centralizing model loading
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib
import time

# FastAPI for HTTP server
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Model imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    HAS_MODELS = True
except ImportError:
    # Fallback for testing without heavy dependencies
    HAS_MODELS = False
    print("⚠️  Model dependencies not available, using mock models")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Request/Response models
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_type: str = "dense"  # "dense" or "sparse"

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_info: Dict[str, Any]
    processing_time: float

class SparseEmbeddingResponse(BaseModel):
    sparse_embeddings: List[Dict[str, Any]]  # List of {indices: [], values: []}
    model_info: Dict[str, Any]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    memory_usage: Dict[str, str]
    uptime: float
    version: str

class SharedModelServer:
    """Centralized model server for all projects"""
    
    def __init__(self):
        self.app = FastAPI(title="Neural Memory - Shared Model Server", version="1.0.0")
        self.start_time = time.time()
        
        # Model instances
        self.dense_model = None
        self.sparse_model = None
        self.model_cache = {}
        
        # Performance stats
        self.request_count = 0
        self.total_processing_time = 0.0
        
        # Setup routes
        self._setup_routes()
    
    async def initialize_models(self):
        """Initialize embedding models"""
        logger.info("🤖 Initializing shared models...")
        
        try:
            if HAS_MODELS:
                # Load Qodo-Embed model
                logger.info("📦 Loading Qodo-Embed-1.5B model...")
                self.dense_model = SentenceTransformer('Qodo/Qodo-Embed-1-1.5B')
                logger.info("✅ Qodo-Embed model loaded")
                
                # Initialize BM25/TF-IDF model (lightweight)
                logger.info("📦 Loading BM25/TF-IDF model...")
                self.sparse_model = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    max_df=0.8,
                    min_df=2
                )
                logger.info("✅ BM25/TF-IDF model initialized")
            else:
                # Mock models for testing
                self.dense_model = "mock_dense"
                self.sparse_model = "mock_sparse"
                logger.info("✅ Mock models initialized for testing")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy" if self.dense_model and self.sparse_model else "initializing",
                models_loaded={
                    "dense_model": self.dense_model is not None,
                    "sparse_model": self.sparse_model is not None
                },
                memory_usage={
                    "total_requests": str(self.request_count),
                    "avg_processing_time": f"{self.total_processing_time / max(1, self.request_count):.3f}s"
                },
                uptime=time.time() - self.start_time,
                version="1.0.0"
            )
        
        @self.app.post("/embed/dense", response_model=EmbeddingResponse)
        async def create_dense_embeddings(request: EmbeddingRequest):
            """Generate dense embeddings using Qodo-Embed"""
            if not self.dense_model:
                raise HTTPException(status_code=503, detail="Dense model not available")
            
            start_time = time.time()
            
            try:
                if HAS_MODELS:
                    # Real model processing
                    embeddings = self.dense_model.encode(request.texts, convert_to_numpy=True)
                    embeddings_list = embeddings.tolist()
                else:
                    # Mock processing
                    embeddings_list = []
                    for text in request.texts:
                        # Generate consistent mock embedding based on text hash
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        mock_embedding = [
                            int(text_hash[i:i+2], 16) / 255.0 
                            for i in range(0, min(len(text_hash), 32), 2)
                        ]
                        # Pad to 1536 dimensions (Qodo-Embed size)
                        while len(mock_embedding) < 1536:
                            mock_embedding.extend(mock_embedding[:min(1536-len(mock_embedding), len(mock_embedding))])
                        embeddings_list.append(mock_embedding[:1536])
                
                processing_time = time.time() - start_time
                self.request_count += 1
                self.total_processing_time += processing_time
                
                return EmbeddingResponse(
                    embeddings=embeddings_list,
                    model_info={
                        "model": "Qodo-Embed-1.5B" if HAS_MODELS else "Mock",
                        "dimensions": 1536,
                        "texts_processed": len(request.texts)
                    },
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"Dense embedding error: {e}")
                raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
        
        @self.app.post("/embed/sparse", response_model=SparseEmbeddingResponse)
        async def create_sparse_embeddings(request: EmbeddingRequest):
            """Generate sparse embeddings using BM25/TF-IDF"""
            if not self.sparse_model:
                raise HTTPException(status_code=503, detail="Sparse model not available")
            
            start_time = time.time()
            
            try:
                sparse_embeddings = []
                
                if HAS_MODELS:
                    # Fit or use existing TF-IDF model
                    if not hasattr(self.sparse_model, 'vocabulary_'):
                        # First time - fit on provided texts
                        self.sparse_model.fit(request.texts)
                    
                    # Transform texts to sparse vectors
                    tfidf_matrix = self.sparse_model.transform(request.texts)
                    
                    for i in range(len(request.texts)):
                        row = tfidf_matrix.getrow(i)
                        indices = row.indices.tolist()
                        values = row.data.tolist()
                        
                        sparse_embeddings.append({
                            "indices": indices,
                            "values": values
                        })
                else:
                    # Mock sparse embeddings
                    for text in request.texts:
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        # Generate consistent mock sparse vector
                        indices = [abs(hash(text + str(i))) % 1000 + i*100 for i in range(5)]
                        values = [0.5 + i*0.1 for i in range(5)]
                        
                        sparse_embeddings.append({
                            "indices": indices,
                            "values": values
                        })
                
                processing_time = time.time() - start_time
                self.request_count += 1
                self.total_processing_time += processing_time
                
                return SparseEmbeddingResponse(
                    sparse_embeddings=sparse_embeddings,
                    model_info={
                        "model": "TF-IDF/BM25" if HAS_MODELS else "Mock",
                        "vocabulary_size": getattr(self.sparse_model, 'vocabulary_', {}) if HAS_MODELS else 1000,
                        "texts_processed": len(request.texts)
                    },
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"Sparse embedding error: {e}")
                raise HTTPException(status_code=500, detail=f"Sparse embedding failed: {e}")
        
        @self.app.get("/models/info")
        async def get_model_info():
            """Get information about loaded models"""
            return {
                "dense_model": {
                    "name": "Qodo-Embed-1.5B" if HAS_MODELS else "Mock Dense",
                    "dimensions": 1536,
                    "loaded": self.dense_model is not None
                },
                "sparse_model": {
                    "name": "TF-IDF/BM25" if HAS_MODELS else "Mock Sparse", 
                    "vocabulary_size": len(getattr(self.sparse_model, 'vocabulary_', {})) if HAS_MODELS else 1000,
                    "loaded": self.sparse_model is not None
                },
                "performance": {
                    "total_requests": self.request_count,
                    "average_processing_time": self.total_processing_time / max(1, self.request_count),
                    "uptime": time.time() - self.start_time
                }
            }

async def main():
    """Main entry point for shared model server"""
    
    print("🚀 Neural Memory - Shared Model Server")
    print("=" * 50)
    
    # Initialize server
    server = SharedModelServer()
    
    # Initialize models
    await server.initialize_models()
    
    print(f"✅ Models initialized successfully")
    print(f"🌐 Starting server on http://localhost:8090")
    print(f"📊 Health check: http://localhost:8090/health")
    print(f"📖 API docs: http://localhost:8090/docs")
    print("")
    
    # Start server
    config = uvicorn.Config(
        server.app,
        host="0.0.0.0",
        port=8090,
        log_level="info",
        access_log=True
    )
    
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()

if __name__ == "__main__":
    asyncio.run(main())