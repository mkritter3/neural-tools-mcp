#!/usr/bin/env python3
"""
Nomic Embedding Service - Extracted from monolithic neural-mcp-server-enhanced.py
Provides embedding generation using Nomic Embed v2-MoE with Context7 patterns
"""

import os
import logging
import asyncio
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
        host = os.environ.get('EMBEDDING_SERVICE_HOST', 'neural-embeddings')
        port = int(os.environ.get('EMBEDDING_SERVICE_PORT', 8000))
        self.base_url = f"http://{host}:{port}"
        
        # Store configuration for creating clients per request
        # Context7 pattern: avoid storing AsyncClient in __init__
        self.timeout = httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=60.0,     # Read timeout
            write=30.0,    # Write timeout
            pool=5.0       # Pool timeout
        )
        self.transport_kwargs = {"retries": 1}
        self.limits = httpx.Limits(max_connections=20, max_keepalive_connections=5)
        
    async def get_embeddings(self, texts: List[str]) -> NomicEmbedResponse:
        """Get embeddings using Nomic Embed v2-MoE with Context7 async client pattern
        
        Creates fresh httpx.AsyncClient per request to avoid event loop binding issues.
        """
        max_retries = 3
        retry_delay = 1.0
        
        # Context7 recommended pattern: fresh AsyncClient per request
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(**self.transport_kwargs),
            timeout=self.timeout,
            limits=self.limits
        ) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/embed",
                        json={"inputs": texts, "normalize": True}
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    return NomicEmbedResponse(
                        embeddings=data.get('embeddings', []),
                        model=data.get('model', 'nomic-embed-text-v2-moe'),
                        usage=data.get('usage', {'prompt_tokens': 0})
                    )
                    
                except httpx.HTTPStatusError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic HTTP error (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Nomic connection failed after {max_retries} attempts: {e}")
                        raise
                        
                except httpx.TimeoutException as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic timeout (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Nomic timeout after {max_retries} attempts: {e}")
                        raise
                        
                except Exception as e:
                    logger.error(f"Nomic embed error: {e}")
                    raise

class NomicService:
    """Service class for Nomic embedding operations with proper initialization"""
    
    def __init__(self):
        self.client = None
        self.initialized = False
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the Nomic service with health check"""
        try:
            self.client = NomicEmbedClient()
            
            # Test the service with a simple embedding
            test_response = await self.client.get_embeddings(["test initialization"])
            
            if test_response.embeddings and len(test_response.embeddings[0]) > 0:
                self.initialized = True
                return {
                    "success": True,
                    "message": "Nomic service initialized successfully",
                    "embedding_dim": len(test_response.embeddings[0])
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
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts with error handling"""
        if not self.initialized or not self.client:
            raise RuntimeError("Nomic service not initialized")
            
        response = await self.client.get_embeddings(texts)
        return response.embeddings
    
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