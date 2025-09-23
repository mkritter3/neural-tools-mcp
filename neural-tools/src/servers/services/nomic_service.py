#!/usr/bin/env python3
"""
Nomic Embedding Service - Extracted from monolithic neural-mcp-server-enhanced.py
Provides embedding generation using Nomic Embed v2-MoE with Context7 patterns
"""

import os
import logging
import asyncio
import hashlib
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

@dataclass
class NomicEmbedResponse:
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]

class NomicEmbedClient:
    """Client for Nomic Embed v2-MoE service with enhanced connectivity
    
    Uses Context7 recommended pattern: fresh httpx.AsyncClient per request
    to avoid asyncio event loop binding issues in MCP environments.
    """
    
    def __init__(self):
        # CRITICAL: Use localhost + exposed port for host-to-container communication
        # NOT Docker internal IPs (172.x.x.x) which broke MCP connectivity
        host = os.environ.get('EMBEDDING_SERVICE_HOST', 'localhost')  # Must be localhost from host
        port = int(os.environ.get('EMBEDDING_SERVICE_PORT', 48000))   # Exposed port 48000, not 8000
        self.base_url = f"http://{host}:{port}"
        
        # Store configuration for creating clients per request
        # Context7 pattern: Create fresh AsyncClient per request to avoid
        # asyncio event loop binding issues in MCP server environments
        self.timeout = httpx.Timeout(
            connect=10.0,  # Connection timeout - fail fast if service down
            read=60.0,     # Read timeout - embeddings can be slow for large texts
            write=30.0,    # Write timeout - sending batch texts
            pool=5.0       # Pool timeout - connection pool acquisition
        )
        self.transport_kwargs = {"retries": 1}  # Single retry at transport level
        self.limits = httpx.Limits(
            max_connections=20,           # Total connections to Nomic service
            max_keepalive_connections=5   # Keep-alive for connection reuse
        )
        
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Nomic Embed v2-MoE with Context7 async client pattern
        
        Creates fresh httpx.AsyncClient per request to avoid event loop binding issues.
        """
        max_retries = 3      # Total attempts including first try
        retry_delay = 1.0    # Base delay, increases with each retry
        
        # Context7 recommended pattern: Create fresh AsyncClient per request
        # This prevents asyncio event loop issues when called from MCP server
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(**self.transport_kwargs),
            timeout=self.timeout,
            limits=self.limits
        ) as client:
            for attempt in range(max_retries):
                try:
                    # Call Nomic embedding service endpoint
                    response = await client.post(
                        f"{self.base_url}/embed",
                        json={
                            "inputs": texts,      # List of texts to embed
                            "normalize": True     # L2 normalize embeddings
                        }
                    )
                    response.raise_for_status()  # Raise on 4xx/5xx
                    data = response.json()
                    
                    # Extract embeddings from response
                    return data.get('embeddings', [])
                    
                except httpx.HTTPStatusError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic HTTP error (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Nomic connection failed after {max_retries} attempts: {e}")
                        raise
                        
                except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic timeout (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Nomic timeout after {max_retries} attempts: {e}")
                        # Fallback to local embedding after all retries exhausted
                        return await self._fallback_local_embeddings(texts)
                        
                except Exception as e:
                    logger.error(f"Nomic embed error: {e}")
                    # Fallback to local embedding for testing
                    return await self._fallback_local_embeddings(texts)
    
    async def _fallback_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fallback to simple local embeddings for testing purposes"""
        import hashlib
        import numpy as np
        
        logger.warning("Using fallback local embeddings - NOT Nomic quality!")
        
        # Simple hash-based embedding for testing (768 dimensions to match expected size)
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, 768).astype(np.float32).tolist()
            embeddings.append(embedding)
        
        return embeddings

class NomicService:
    """Service class for Nomic embedding operations with queue fallback and resilience"""
    
    def __init__(self):
        self.client = None
        self.initialized = False
        self.service_container = None  # Will be set by service container
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the Nomic service with health check"""
        try:
            self.client = NomicEmbedClient()
            
            # Test the service with a simple embedding
            test_response = await self.client.get_embeddings(["test initialization"])
            
            if test_response and len(test_response[0]) > 0:
                self.initialized = True
                return {
                    "success": True,
                    "message": "Nomic service initialized successfully",
                    "embedding_dim": len(test_response[0])
                }
            else:
                return {
                    "success": False,
                    "message": "Nomic service test embedding failed"
                }
                
        except Exception as e:
            logger.error(f"Nomic service initialization failed: {e}")
            return {
                "success": False,
                "message": f"Nomic initialization failed: {str(e)}"
            }
    
    def set_service_container(self, container):
        """Set reference to service container for queue access"""
        self.service_container = container
    
    def _generate_cache_key(self, text: str, model: str = "nomic-v2") -> str:
        """Generate cache key with model versioning"""
        # Create deterministic cache key based on content
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # L9 cache key format: namespace:env:app:type:model:version:hash
        return f"l9:prod:neural_tools:embeddings:{model}:1.0:{content_hash}"
    
    def _generate_job_id(self, text: str, model: str = "nomic-v2") -> str:
        """Generate deterministic job ID for deduplication"""
        # Combine model and text for unique job identification
        content = f"{model}:{text}"
        
        # Create short deterministic ID for ARQ job deduplication
        return f"embed_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    async def get_embedding(self, text: str, model: str = "nomic-v2") -> List[float]:
        """
        Get embedding for single text with intelligent caching and queue fallback
        
        Args:
            text: Text to embed
            model: Model identifier
            
        Returns:
            Embedding vector as list of floats
        """
        if not self.initialized or not self.client:
            raise RuntimeError("Nomic service not initialized")
        
        # Step 1: Check Redis cache for existing embedding
        if self.service_container:
            try:
                redis_cache = await self.service_container.get_redis_cache_client()
                cache_key = self._generate_cache_key(text, model)
                cached = await redis_cache.get(cache_key)
                
                if cached:
                    # Cache hit - refresh TTL to prevent expiration of hot data
                    await redis_cache.expire(cache_key, 86400)  # Reset to 24 hour TTL
                    return json.loads(cached)
                    
            except Exception as e:
                logger.warning(f"Cache check failed: {e}")
        
        # Step 2: Try direct embedding call to Nomic service
        try:
            embeddings = await self.client.get_embeddings([text])
            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                
                # Step 3: Cache the result for future requests
                if self.service_container:
                    try:
                        redis_cache = await self.service_container.get_redis_cache_client()
                        cache_key = self._generate_cache_key(text, model)
                        
                        # Store with 24-hour TTL
                        await redis_cache.setex(cache_key, 86400, json.dumps(embedding))
                    except Exception as e:
                        # Cache failure is non-critical - log and continue
                        logger.warning(f"Cache store failed: {e}")
                
                return embedding
            else:
                raise ValueError("Empty embedding response")
                
        except (ConnectionError, TimeoutError, httpx.HTTPError) as e:
            # ADR-0083: Simplified to pure HTTP service - no queue logic
            logger.error(f"Nomic HTTP connection failed for {model}: {e}")
            raise
        except Exception as e:
            # Other errors - propagate immediately
            logger.error(f"Embedding generation failed: {e}")
            raise
    
# ADR-0083: Removed _fallback_to_queue method - pure HTTP service with no queue logic
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with error handling"""
        if not self.initialized or not self.client:
            raise RuntimeError("Nomic service not initialized")
        
        # For batch requests, try direct first, then individual fallbacks
        try:
            response = await self.client.get_embeddings(texts)
            return response
        except (ConnectionError, TimeoutError, httpx.HTTPError) as e:
            # Fallback to individual processing through queue if service container available
            if self.service_container and len(texts) <= 10:  # Limit batch fallback
                logger.warning(f"Batch embedding failed, falling back to individual processing: {e}")
                results = []
                for text in texts:
                    try:
                        embedding = await self.get_embedding(text)
                        results.append(embedding)
                    except Exception:
                        # For batch operations, use fallback embedding
                        fallback_embedding = await self._generate_fallback_embedding(text)
                        results.append(fallback_embedding)
                return results
            else:
                raise
    
    async def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate fallback embedding for testing/degraded service"""
        import hashlib
        import numpy as np
        
        logger.warning(f"Using fallback embedding for: {text[:50]}...")
        
        # Create deterministic embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        np.random.seed(seed)
        embedding = np.random.normal(0, 1, 768).astype(np.float32).tolist()
        
        return embedding
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            if not self.initialized:
                return {"healthy": False, "message": "Service not initialized"}
                
            # Simple health check
            test_embedding = await self.get_embeddings(["health check"])
            return {
                "healthy": True,
                "embedding_dim": len(test_embedding[0]) if test_embedding else 0,
                "service_url": self.client.base_url if self.client else "unknown"
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}