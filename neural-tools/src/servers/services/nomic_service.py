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

    ADR-0084 Phase 2: Implements connection pooling for 20% latency reduction
    Uses singleton client with proper connection reuse and keep-alive.
    """

    def __init__(self):
        # Support both URL and host+port configuration
        # NOMIC_EMBEDDINGS_URL: Direct URL (container-to-container)
        # EMBEDDING_SERVICE_HOST/PORT: Host+port (host-to-container)
        nomic_url = os.environ.get('NOMIC_EMBEDDINGS_URL')

        if nomic_url:
            # Container-to-container communication using service name
            self.base_url = nomic_url
        else:
            # Host-to-container communication using localhost + exposed port
            host = os.environ.get('EMBEDDING_SERVICE_HOST', 'localhost')
            port = int(os.environ.get('EMBEDDING_SERVICE_PORT', 48000))
            self.base_url = f"http://{host}:{port}"

        # ADR-0084 Phase 2: Enhanced connection pooling configuration
        self.timeout = httpx.Timeout(
            connect=5.0,   # Faster connection timeout with pooling
            read=60.0,     # Read timeout - embeddings can be slow for large texts
            write=30.0,    # Write timeout - sending batch texts
            pool=5.0       # Pool timeout - connection pool acquisition
        )

        # ADR-0084 Phase 2: Optimized transport with connection reuse
        self.transport = httpx.AsyncHTTPTransport(
            retries=3,
            limits=httpx.Limits(
                max_connections=100,           # Increased for parallel processing
                max_keepalive_connections=20,  # More keep-alive connections
                keepalive_expiry=30.0         # Keep connections alive for 30s
            )
        )

        # Create persistent client for connection reuse
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            transport=self.transport,
            timeout=self.timeout
        )
        
    async def get_embeddings(self, texts: List[str], task_type: str = "search_document") -> List[List[float]]:
        """Get embeddings using Nomic Embed v2-MoE with connection pooling

        ADR-0084 Phase 2: Uses persistent client with connection pooling
        Implements batch processing for 10x throughput improvement
        """
        max_retries = 3      # Total attempts including first try
        retry_delay = 1.0    # Base delay, increases with each retry

        # ADR-0084 Phase 2: Batch processing configuration
        max_batch_size = 64  # Optimal batch size for Nomic v2

        # ADR-0084: CRITICAL - Add task prefixes (80% performance gain)
        prefixed_texts = [f"{task_type}: {text}" for text in texts]

        # Process in batches if input is large
        all_embeddings = []

        for i in range(0, len(prefixed_texts), max_batch_size):
            batch = prefixed_texts[i:i + max_batch_size]

            for attempt in range(max_retries):
                try:
                    # Use persistent client for connection reuse
                    response = await self.client.post(
                        "/embed",
                        json={
                            "inputs": batch,
                            "normalize": True  # L2 normalize embeddings
                        }
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Extract embeddings from response
                    embeddings = data.get('embeddings', [])

                    # ADR-0084: Validate embeddings dimension
                    if embeddings and len(embeddings[0]) != 768:
                        logger.error(f"Invalid embedding dimension: {len(embeddings[0])}, expected 768")
                        raise ValueError(f"Invalid embedding dimension from Nomic")

                    all_embeddings.extend(embeddings)
                    break  # Success, move to next batch

                except httpx.HTTPStatusError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic HTTP error (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Nomic connection failed after {max_retries} attempts: {e}")
                        raise RuntimeError(f"Nomic service unavailable after {max_retries} attempts") from e

                except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic timeout (attempt {attempt + 1}/{max_retries}): {e}")
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.error(f"Nomic timeout after {max_retries} attempts: {e}")
                        raise RuntimeError(f"Nomic service timeout after {max_retries} attempts") from e

                except Exception as e:
                    logger.error(f"Nomic embed error: {e}")
                    raise RuntimeError(f"Nomic embedding failed: {e}") from e

        return all_embeddings

    async def close(self):
        """Clean up client connections"""
        if hasattr(self, 'client'):
            await self.client.aclose()


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

            # ADR-0084 Phase 2: Initialize service container for Redis cache
            if not self.service_container:
                try:
                    from servers.services.service_container import ServiceContainer
                    self.service_container = ServiceContainer(project_name="default")
                    logger.info("✅ Service container initialized for Redis cache")
                except Exception as e:
                    logger.warning(f"Could not initialize service container for cache: {e}")
                    # Continue without cache - not critical

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
    
    def _generate_cache_key(self, text: str, model: str = "nomic-v2", task_type: str = "search_document") -> str:
        """Generate cache key with model versioning and task type"""
        # ADR-0084: Include task_type in cache key since different prefixes produce different embeddings
        content_with_task = f"{task_type}:{text}"
        content_hash = hashlib.sha256(content_with_task.encode()).hexdigest()

        # L9 cache key format: namespace:env:app:type:model:task:version:hash
        return f"l9:prod:neural_tools:embeddings:{model}:{task_type}:1.0:{content_hash}"
    
    def _generate_job_id(self, text: str, model: str = "nomic-v2") -> str:
        """Generate deterministic job ID for deduplication"""
        # Combine model and text for unique job identification
        content = f"{model}:{text}"
        
        # Create short deterministic ID for ARQ job deduplication
        return f"embed_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    async def get_embedding(self, text: str, model: str = "nomic-v2", task_type: str = "search_document") -> List[float]:
        """
        Get embedding for single text with intelligent caching
        ADR-0084: Add task prefixes for optimal performance

        Args:
            text: Text to embed
            model: Model identifier
            task_type: Task prefix for Nomic ('search_document' or 'search_query')

        Returns:
            Embedding vector as list of floats
        """
        if not self.initialized or not self.client:
            raise RuntimeError("Nomic service not initialized")

        # Step 1: Check Redis cache for existing embedding
        if self.service_container:
            try:
                redis_cache = await self.service_container.get_redis_cache_client()
                cache_key = self._generate_cache_key(text, model, task_type)
                cached = await redis_cache.get(cache_key)

                if cached:
                    # Cache hit - refresh TTL to prevent expiration of hot data
                    await redis_cache.expire(cache_key, 86400)  # Reset to 24 hour TTL
                    return json.loads(cached)

            except Exception as e:
                logger.warning(f"Cache check failed: {e}")

        # Step 2: Try direct embedding call to Nomic service with task prefix
        try:
            # ADR-0084: Pass task_type for proper prefixing
            embeddings = await self.client.get_embeddings([text], task_type=task_type)
            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                
                # Step 3: Cache the result for future requests
                if self.service_container:
                    try:
                        redis_cache = await self.service_container.get_redis_cache_client()
                        cache_key = self._generate_cache_key(text, model, task_type)

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
    
    async def get_embeddings(self, texts: List[str], task_type: str = "search_document") -> List[List[float]]:
        """Get embeddings for multiple texts with error handling
        ADR-0084: Add task prefixes for optimal performance"""
        if not self.initialized or not self.client:
            raise RuntimeError("Nomic service not initialized")

        # ADR-0084: Direct call with task prefixes
        try:
            response = await self.client.get_embeddings(texts, task_type=task_type)
            return response
        except Exception as e:
            # ADR-0084: Fail fast - no fallback embeddings
            logger.error(f"Batch embedding failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e
    
    
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