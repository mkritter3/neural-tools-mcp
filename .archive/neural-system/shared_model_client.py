#!/usr/bin/env python3
"""
Shared Model Client
Client library for accessing the centralized model server
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import json

logger = logging.getLogger(__name__)

class SharedModelClient:
    """Client for accessing shared model server"""
    
    def __init__(self, server_host: str = "localhost", server_port: int = 8090):
        self.base_url = f"http://{server_host}:{server_port}"
        self.session = None
        self._model_info = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Check server health
            try:
                await self.health_check()
                logger.info(f"✅ Connected to shared model server at {self.base_url}")
            except Exception as e:
                logger.warning(f"⚠️  Could not connect to model server: {e}")
                # Don't fail initialization - allow fallback to local models
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health and get status"""
        if not self.session:
            await self.initialize()
        
        async with self.session.get(f"{self.base_url}/health") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Health check failed: {response.status}")
    
    async def get_dense_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Get dense embeddings from shared server
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tuple of (embeddings_list, model_info)
        """
        if not self.session:
            await self.initialize()
        
        payload = {
            "texts": texts,
            "model_type": "dense"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/embed/dense", 
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["embeddings"], result["model_info"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Dense embedding failed: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error getting dense embeddings: {e}")
            raise Exception(f"Failed to connect to model server: {e}")
    
    async def get_sparse_embeddings(self, texts: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Get sparse embeddings from shared server
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tuple of (sparse_embeddings_list, model_info)
        """
        if not self.session:
            await self.initialize()
        
        payload = {
            "texts": texts,
            "model_type": "sparse"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/embed/sparse",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["sparse_embeddings"], result["model_info"]
                else:
                    error_text = await response.text()
                    raise Exception(f"Sparse embedding failed: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error getting sparse embeddings: {e}")
            raise Exception(f"Failed to connect to model server: {e}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        if not self.session:
            await self.initialize()
        
        if self._model_info:
            return self._model_info
        
        try:
            async with self.session.get(f"{self.base_url}/models/info") as response:
                if response.status == 200:
                    self._model_info = await response.json()
                    return self._model_info
                else:
                    error_text = await response.text()
                    raise Exception(f"Model info failed: {response.status} - {error_text}")
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error getting model info: {e}")
            return {
                "dense_model": {"name": "Unavailable", "loaded": False},
                "sparse_model": {"name": "Unavailable", "loaded": False},
                "error": str(e)
            }
    
    async def batch_embeddings(self, texts: List[str], batch_size: int = 32) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """
        Get both dense and sparse embeddings in batches for efficiency
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches for processing
            
        Returns:
            Tuple of (dense_embeddings, sparse_embeddings)
        """
        dense_embeddings = []
        sparse_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Get both types concurrently
                dense_task = self.get_dense_embeddings(batch)
                sparse_task = self.get_sparse_embeddings(batch)
                
                batch_dense, dense_info = await dense_task
                batch_sparse, sparse_info = await sparse_task
                
                dense_embeddings.extend(batch_dense)
                sparse_embeddings.extend(batch_sparse)
                
                logger.debug(f"📦 Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Batch embedding failed for batch starting at {i}: {e}")
                # Add empty embeddings for failed batch to maintain alignment
                for _ in batch:
                    dense_embeddings.append([0.0] * 1536)  # Empty 1536-dim vector
                    sparse_embeddings.append({"indices": [], "values": []})
        
        return dense_embeddings, sparse_embeddings

class FallbackEmbedder:
    """Fallback local embedder when shared server is unavailable"""
    
    def __init__(self):
        logger.info("🔄 Initializing fallback embedder (local mode)")
        
    async def get_dense_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Generate mock dense embeddings locally"""
        import hashlib
        
        embeddings = []
        for text in texts:
            # Generate consistent hash-based embedding
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = []
            
            for i in range(0, min(len(text_hash), 32), 2):
                val = int(text_hash[i:i+2], 16) / 255.0
                embedding.append(val)
            
            # Pad to 1536 dimensions
            while len(embedding) < 1536:
                embedding.extend(embedding[:min(1536-len(embedding), len(embedding))])
            
            embeddings.append(embedding[:1536])
        
        return embeddings, {
            "model": "Local Fallback",
            "dimensions": 1536,
            "texts_processed": len(texts)
        }
    
    async def get_sparse_embeddings(self, texts: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate mock sparse embeddings locally"""
        import hashlib
        
        sparse_embeddings = []
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Generate consistent sparse vector
            indices = [abs(hash(text + str(i))) % 1000 + i*100 for i in range(5)]
            values = [0.3 + i*0.1 for i in range(5)]
            
            sparse_embeddings.append({
                "indices": indices,
                "values": values
            })
        
        return sparse_embeddings, {
            "model": "Local Fallback BM25",
            "texts_processed": len(texts)
        }

# Convenience function for easy access
async def get_embedder(server_host: str = "localhost", server_port: int = 8090, 
                      use_fallback_on_error: bool = True) -> SharedModelClient:
    """
    Get an embedder instance, with fallback to local mode
    
    Args:
        server_host: Model server host
        server_port: Model server port  
        use_fallback_on_error: Whether to use local fallback on server errors
        
    Returns:
        Embedder instance (shared client or fallback)
    """
    try:
        client = SharedModelClient(server_host, server_port)
        await client.initialize()
        
        # Test connection
        await client.health_check()
        return client
        
    except Exception as e:
        if use_fallback_on_error:
            logger.warning(f"⚠️  Falling back to local embedder due to: {e}")
            return FallbackEmbedder()
        else:
            raise

# Test function
async def test_client():
    """Test the shared model client"""
    print("🧪 Testing Shared Model Client")
    print("=" * 40)
    
    try:
        async with SharedModelClient() as client:
            # Health check
            health = await client.health_check()
            print(f"✅ Server health: {health['status']}")
            
            # Model info
            info = await client.get_model_info()
            print(f"📖 Dense model: {info['dense_model']['name']}")
            print(f"📖 Sparse model: {info['sparse_model']['name']}")
            
            # Test embeddings
            test_texts = [
                "This is a test sentence for embedding",
                "Another test text to verify the system works"
            ]
            
            dense_embeddings, dense_info = await client.get_dense_embeddings(test_texts)
            print(f"✅ Dense embeddings: {len(dense_embeddings)} x {len(dense_embeddings[0])}")
            
            sparse_embeddings, sparse_info = await client.get_sparse_embeddings(test_texts)
            print(f"✅ Sparse embeddings: {len(sparse_embeddings)} vectors")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Testing fallback embedder...")
        
        fallback = FallbackEmbedder()
        dense_embeddings, dense_info = await fallback.get_dense_embeddings(test_texts)
        print(f"✅ Fallback dense embeddings: {len(dense_embeddings)} x {len(dense_embeddings[0])}")

if __name__ == "__main__":
    asyncio.run(test_client())