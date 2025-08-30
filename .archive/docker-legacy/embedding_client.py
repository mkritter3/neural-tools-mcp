#!/usr/bin/env python3
"""
L9 Neural Flow - Shared Embedding Service Client
Connects to containerized embedding service for distributed model inference
"""

import os
import json
import httpx
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]

class SharedEmbeddingClient:
    """Client for shared embedding service container"""
    
    def __init__(self, 
                 host: str = None, 
                 port: int = None,
                 timeout: int = 30):
        self.host = host or os.getenv("EMBEDDING_SERVICE_HOST", "embedding-service")
        self.port = port or int(os.getenv("EMBEDDING_SERVICE_PORT", "8000"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        
        logger.info(f"Initialized embedding client: {self.base_url}")
    
    async def health_check(self) -> bool:
        """Check if embedding service is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_embeddings(self, texts: List[str], normalize: bool = True) -> EmbeddingResponse:
        """Get embeddings for list of texts"""
        try:
            payload = {
                "inputs": texts,
                "normalize": normalize
            }
            
            response = await self.client.post(
                f"{self.base_url}/embed",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            return EmbeddingResponse(
                embeddings=data["embeddings"],
                model=data.get("model", "BAAI/bge-base-en-v1.5"),
                usage=data.get("usage", {"prompt_tokens": len(texts)})
            )
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error getting embeddings: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise
    
    async def get_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """Get embedding for single text"""
        response = await self.get_embeddings([text], normalize=normalize)
        return response.embeddings[0]
    
    async def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encode texts in batches for large datasets"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.get_embeddings(batch)
            all_embeddings.extend(response.embeddings)
            
            # Small delay between batches to prevent overwhelming service
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return all_embeddings
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

class EmbeddingManager:
    """High-level embedding manager with fallbacks"""
    
    def __init__(self):
        self.shared_client = SharedEmbeddingClient()
        self.fallback_available = False
        
        # Initialize fallback (local FastEmbed) if available
        try:
            from fastembed import TextEmbedding
            self.fallback_model = TextEmbedding("BAAI/bge-base-en-v1.5")
            self.fallback_available = True
            logger.info("Fallback embedding model initialized")
        except ImportError:
            logger.warning("FastEmbed not available - no fallback embeddings")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with automatic fallback"""
        # Try shared service first
        if await self.shared_client.health_check():
            try:
                response = await self.shared_client.get_embeddings(texts)
                return response.embeddings
            except Exception as e:
                logger.warning(f"Shared service failed, trying fallback: {e}")
        
        # Fallback to local model
        if self.fallback_available:
            try:
                embeddings = list(self.fallback_model.embed(texts))
                return [embedding.tolist() for embedding in embeddings]
            except Exception as e:
                logger.error(f"Fallback embedding failed: {e}")
                raise
        
        raise RuntimeError("No embedding service available")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get single embedding with automatic fallback"""
        embeddings = await self.get_embeddings([text])
        return embeddings[0]
    
    async def close(self):
        """Clean up resources"""
        await self.shared_client.close()

# Singleton instance
_embedding_manager = None

async def get_embedding_manager() -> EmbeddingManager:
    """Get global embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Convenience function for getting embeddings"""
    manager = await get_embedding_manager()
    return await manager.get_embeddings(texts)

async def embed_text(text: str) -> List[float]:
    """Convenience function for single embedding"""
    manager = await get_embedding_manager()
    return await manager.get_embedding(text)