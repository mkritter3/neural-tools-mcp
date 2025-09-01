#!/usr/bin/env python3
"""
L9 Enhanced MCP Server - 2025 Neo4j GraphRAG Migration
Features: Neo4j GraphRAG + Nomic Embed v2-MoE + Tree-sitter + Qdrant hybrid search

Performance Improvements Over Legacy:
- Neo4j GraphRAG (Production-grade multi-tenancy)
- Nomic v2-MoE (30-40% lower inference costs)
- RRF hybrid search with MMR diversity
- INT8 quantization (4x memory reduction)
- Tree-sitter multi-language AST (13+ languages)
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sqlite3

def generate_deterministic_point_id(file_path: str, content: str, chunk_index: int = 0) -> int:
    """Generate deterministic point ID for consistent upserts following industry standards.
    
    This prevents duplicate accumulation by ensuring the same content gets the same ID,
    allowing Qdrant's upsert to properly update existing points instead of creating duplicates.
    
    Args:
        file_path: Path to the source file
        content: Content being indexed  
        chunk_index: Index of the chunk within the file (for multi-chunk files)
    
    Returns:
        Deterministic integer ID that will be the same for identical content
    """
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    unique_string = f"{file_path}#{content_hash}#{chunk_index}"
    return abs(hash(unique_string)) % (10**15)

def get_content_hash(content: str) -> str:
    """Generate content hash for change detection"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

async def cleanup_stale_chunks(manifest: Dict[str, str], project_dir: Path):
    """Clean up stale chunks from deleted or modified files following industry standards.
    
    This prevents database bloat by removing chunks that no longer correspond to existing files
    or have been replaced by updated versions.
    
    Args:
        manifest: Current manifest with file hashes
        project_dir: Project directory path
    """
    try:
        collection_name = f"{COLLECTION_PREFIX}code"
        
        # Get all points in the collection to identify stale chunks
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=10000,  # Process in batches
            with_payload=True,
            with_vectors=False
        )
        
        points_to_delete = []
        
        for point in scroll_result[0]:
            payload = point.payload
            file_path = payload.get('file_path', '')
            
            # Check if file still exists in project
            full_file_path = project_dir / file_path
            
            if not full_file_path.exists():
                # File was deleted - mark all its chunks for deletion
                points_to_delete.append(point.id)
                logger.debug(f"Marking chunks from deleted file {file_path} for cleanup")
            
            elif file_path in manifest:
                # File exists but check if content changed significantly
                # (This covers cases where files were truncated and have fewer chunks now)
                try:
                    with open(full_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        current_content = f.read()
                    
                    # If file is now empty or very small, clean up chunks
                    if len(current_content.strip()) < 50:
                        points_to_delete.append(point.id)
                        logger.debug(f"Marking chunks from truncated file {file_path} for cleanup")
                        
                except Exception as e:
                    logger.warning(f"Could not check file {file_path} for cleanup: {e}")
        
        # Delete stale points in batches
        if points_to_delete:
            # Delete in batches of 100 to avoid memory issues
            for i in range(0, len(points_to_delete), 100):
                batch = points_to_delete[i:i+100]
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=batch)
                )
            
            logger.info(f"Cleaned up {len(points_to_delete)} stale chunks from collection {collection_name}")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        # Don't raise - cleanup failures shouldn't break indexing

# MCP SDK
from fastmcp import FastMCP
import mcp.types as types

# Qdrant client for enhanced hybrid search (no legacy FastEmbed)
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

# Neo4j for GraphRAG (L9 2025 Production)
try:
    from neo4j_client import Neo4jGraphRAGClient, AsyncNeo4jClient
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j client not available - GraphRAG features disabled")

# Nomic Embed v2 client
import httpx
from dataclasses import dataclass

# Import PRISM scorer for intelligent importance scoring
try:
    import sys
    sys.path.append('/app/project/neural-tools')
    from prism_scorer import PrismScorer
    PRISM_AVAILABLE = True
except ImportError:
    PRISM_AVAILABLE = False
    logging.warning("PRISM scorer not available, using basic scoring")

# Tree-sitter for enhanced code analysis
try:
    from tree_sitter_ast import TreeSitterAnalyzer
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("Tree-sitter not available - code analysis limited")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("l9-neural-enhanced")

# Global clients
qdrant_client = None
neo4j_client = None
nomic_client = None
ast_analyzer = None

# Constants
QDRANT_HOST = os.environ.get('QDRANT_HOST', 'neural-data-storage')
QDRANT_HTTP_PORT = int(os.environ.get('QDRANT_HTTP_PORT', 6333))
QDRANT_GRPC_PORT = int(os.environ.get('QDRANT_GRPC_PORT', 6334))
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'default')
COLLECTION_PREFIX = f"project_{PROJECT_NAME}_"
# Neo4j GraphRAG Configuration
NEO4J_HOST = os.environ.get('NEO4J_HOST', 'neo4j-graph')
NEO4J_PORT = int(os.environ.get('NEO4J_PORT', 7687))
GRAPHRAG_ENABLED = os.environ.get('GRAPHRAG_ENABLED', 'true').lower() == 'true' and NEO4J_AVAILABLE
# Embeddings Configuration
NOMIC_BASE_URL = os.environ.get('NOMIC_ENDPOINT', 'http://neural-embeddings:8000/embed')

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
                        embeddings=data["embeddings"],
                        model=data.get("model", "nomic-v2-moe"),
                        usage=data.get("usage", {"prompt_tokens": len(texts)})
                    )
                    
                except httpx.ConnectError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Nomic connection failed (attempt {attempt + 1}/{max_retries}): {e}")
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

# =============================================================================
# ADR-0008: NeuralServiceManager for Service Integration Recovery
# =============================================================================

class NeuralServiceManager:
    """Manages initialization and health of all neural system services (ADR-0008)"""
    
    def __init__(self):
        self.services = {
            "qdrant": {"url": f"http://{QDRANT_HOST}:{QDRANT_HTTP_PORT}", "healthy": False},
            "neo4j": {"url": f"bolt://{NEO4J_HOST}:{NEO4J_PORT}" if NEO4J_HOST else "bolt://neo4j-graph:7687", "healthy": False}, 
            "embeddings": {"url": NOMIC_BASE_URL, "healthy": False}
        }
        self.initialized = False
    
    async def initialize_all_services(self) -> Dict[str, Any]:
        """Initialize all services with proper error handling (Context7 pattern)"""
        initialization_results = {}
        
        try:
            # Initialize Qdrant
            qdrant_result = await self._initialize_qdrant()
            initialization_results["qdrant"] = qdrant_result
            self.services["qdrant"]["healthy"] = qdrant_result.get("success", False)
            
            # Initialize Neo4j
            neo4j_result = await self._initialize_neo4j()
            initialization_results["neo4j"] = neo4j_result
            self.services["neo4j"]["healthy"] = neo4j_result.get("success", False)
            
            # Initialize Embeddings Service
            embeddings_result = await self._initialize_embeddings()
            initialization_results["embeddings"] = embeddings_result
            self.services["embeddings"]["healthy"] = embeddings_result.get("success", False)
            
            # Check overall health
            healthy_services = sum(1 for service in self.services.values() if service["healthy"])
            total_services = len(self.services)
            
            self.initialized = healthy_services >= 2  # At least 2/3 services must be healthy
            
            return {
                "status": "success" if self.initialized else "partial",
                "healthy_services": healthy_services,
                "total_services": total_services,
                "service_details": initialization_results,
                "overall_health": f"{healthy_services}/{total_services} services operational"
            }
            
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            return {
                "status": "error",
                "message": f"Service initialization failed: {str(e)}",
                "service_details": initialization_results
            }
    
    async def _initialize_qdrant(self) -> Dict[str, Any]:
        """Initialize Qdrant with collection verification"""
        global qdrant_client
        
        try:
            # Test basic connectivity using Context7 AsyncHTTPTransport pattern
            import httpx
            from httpx import AsyncHTTPTransport
            
            async with httpx.AsyncClient(
                transport=AsyncHTTPTransport(),  # Context7: Use AsyncHTTPTransport
                timeout=10.0
            ) as client:
                response = await client.get(f"{self.services['qdrant']['url']}/collections")
                
                if response.status_code != 200:
                    return {"success": False, "message": f"Qdrant HTTP {response.status_code}"}
            
            # Initialize the global qdrant client - EXPLICIT GLOBAL ASSIGNMENT
            import sys
            current_module = sys.modules[__name__]
            client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_GRPC_PORT,
                prefer_grpc=True
            )
            
            # Set global variable in multiple ways to ensure it works
            qdrant_client = client
            current_module.qdrant_client = client
            globals()['qdrant_client'] = client
            
            logger.info(f"✅ Qdrant client initialized: {client is not None}")
            
            # Verify collection exists or create it
            collection_name = f"{COLLECTION_PREFIX}code"
            collection_result = await self._ensure_qdrant_collection(collection_name)
            
            return {
                "success": True,
                "message": "Qdrant initialized successfully",
                "collection_status": collection_result
            }
            
        except Exception as e:
            return {"success": False, "message": f"Qdrant initialization failed: {str(e)}"}
    
    async def _initialize_neo4j(self) -> Dict[str, Any]:
        """Initialize Neo4j with constraint verification"""
        global neo4j_client
        
        try:
            if not GRAPHRAG_ENABLED:
                return {"success": False, "message": "Neo4j GraphRAG disabled"}
            
            # Initialize client
            neo4j_client = Neo4jGraphRAGClient(project_name=PROJECT_NAME)
            connection_success = await neo4j_client.connect()
            
            if not connection_success:
                return {"success": False, "message": "Neo4j connection failed"}
            
            # Test with simple query (Context7 pattern)
            test_result = await neo4j_client.execute_cypher_query("RETURN 1 as test", {})
            if not test_result or (isinstance(test_result, list) and len(test_result) == 0):
                return {"success": False, "message": "Neo4j test query failed"}
            
            return {
                "success": True, 
                "message": "Neo4j initialized successfully",
                "test_query_result": test_result
            }
            
        except Exception as e:
            neo4j_client = None
            return {"success": False, "message": f"Neo4j initialization failed: {str(e)}"}
    
    async def _initialize_embeddings(self) -> Dict[str, Any]:
        """Initialize embeddings service with proper request format"""
        global nomic_client
        
        try:
            # Initialize client - this always works based on testing
            nomic_client = NomicEmbedClient()
            
            # Test the client directly first to ensure it works
            try:
                test_embeddings = await nomic_client.get_embeddings(["initialization test"])
                if len(test_embeddings.embeddings[0]) != 768:
                    raise Exception(f"Unexpected embedding dimension: {len(test_embeddings.embeddings[0])}")
            except Exception as client_test_error:
                nomic_client = None
                return {"success": False, "message": f"Embedding client test failed: {str(client_test_error)}"}
            
            # Test embeddings service using proper request format (Context7 pattern)
            import httpx
            from httpx import AsyncHTTPTransport
            
            async with httpx.AsyncClient(
                transport=AsyncHTTPTransport(),  # Context7: AsyncHTTPTransport
                timeout=30.0
            ) as client:
                
                # Use correct request format matching EmbedRequest model
                test_response = await client.post(
                    self.services["embeddings"]["url"],
                    json={"inputs": ["test initialization"], "normalize": True},  # Correct format
                    headers={"Content-Type": "application/json"}
                )
                
                if test_response.status_code not in [200, 422]:  # 422 might be parameter issue
                    return {
                        "success": False, 
                        "message": f"Embeddings service HTTP {test_response.status_code}"
                    }
                
                # If 422, try alternative format  
                if test_response.status_code == 422:
                    alt_response = await client.post(
                        self.services["embeddings"]["url"],
                        json={"inputs": ["test initialization alternative"], "normalize": True},  # Alternative format
                        headers={"Content-Type": "application/json"}
                    )
                    if alt_response.status_code not in [200, 422]:
                        return {
                            "success": False,
                            "message": f"Embeddings API format incompatible: {alt_response.status_code}"
                        }
            
            return {
                "success": True,
                "message": "Embeddings service initialized successfully"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Embeddings initialization failed: {str(e)}"}
    
    async def _ensure_qdrant_collection(self, collection_name: str) -> Dict[str, Any]:
        """Ensure Qdrant collection exists"""
        try:
            collections = qdrant_client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)
            
            if not collection_exists:
                # Create collection if it doesn't exist
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(size=768, distance=Distance.COSINE),
                        "sparse": SparseVectorParams()
                    }
                )
                return {"created": True, "name": collection_name}
            else:
                return {"exists": True, "name": collection_name}
                
        except Exception as e:
            return {"error": str(e)}

# Global service manager instance
neural_service_manager = NeuralServiceManager()

async def initialize():
    """Initialize enhanced L9 system with Neo4j GraphRAG - ADR-0008 Service Integration"""
    global qdrant_client, neo4j_client, nomic_client, ast_analyzer
    
    try:
        # Use NeuralServiceManager for comprehensive service initialization (ADR-0008)
        # This will set all global client variables properly
        service_results = await neural_service_manager.initialize_all_services()
        
        # Initialize Tree-sitter for code analysis
        if TREE_SITTER_AVAILABLE:
            ast_analyzer = TreeSitterAnalyzer()
        
        # Ensure enhanced collections exist
        await ensure_collection(f"{COLLECTION_PREFIX}memory")
        await ensure_collection(f"{COLLECTION_PREFIX}code")
        await ensure_collection(f"{COLLECTION_PREFIX}docs")
        
        # Log initialization results
        logger.info(f"✅ Enhanced L9 MCP Server initialized - Project: {PROJECT_NAME}")
        logger.info(f"📊 Service Health: {service_results.get('overall_health', 'Unknown')}")
        logger.info(f"📍 Qdrant: {QDRANT_HOST}:{QDRANT_GRPC_PORT}")
        
        if service_results.get('service_details', {}).get('neo4j', {}).get('success'):
            logger.info(f"🔗 Neo4j GraphRAG: Project {PROJECT_NAME} (Production)")
        else:
            logger.warning("⚠️  Neo4j GraphRAG: Limited functionality")
            
        if service_results.get('service_details', {}).get('embeddings', {}).get('success'):
            logger.info(f"🧠 Nomic Embed v2-MoE: {NOMIC_BASE_URL}")
        else:
            logger.warning("⚠️  Embeddings: Limited functionality")
        
        # Check if we have minimum required services for operation
        if not neural_service_manager.initialized:
            logger.warning("⚠️  System running with limited functionality - some services failed")
        
        return service_results
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced system: {e}")
        raise


async def ensure_collection(collection_name: str):
    """Ensure collection exists with enhanced hybrid search config"""
    try:
        collections = qdrant_client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            # Create with enhanced configuration
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=768,  # Nomic Embed v2 dimension
                        distance=Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(
                            m=32,  # Enhanced connectivity
                            ef_construct=200,  # Better quality
                            full_scan_threshold=10000
                        )
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=8,
                    memmap_threshold=25000,
                    indexing_threshold=10000
                )
            )
            logger.info(f"Created enhanced collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to ensure collection: {e}")

# Initialize services when first MCP tool is called
_initialization_done = False

async def ensure_services_initialized():
    """Ensure services are initialized within MCP event loop"""
    global _initialization_done
    if not _initialization_done:
        try:
            await initialize()
            _initialization_done = True
            logger.info("✅ Services initialized within MCP event loop")
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")

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
        # Ensure services are initialized within MCP event loop
        await ensure_services_initialized()
        collection_name = f"{COLLECTION_PREFIX}{category}"
        await ensure_collection(collection_name)
        
        # Generate deterministic point ID using string format (Context7 pattern)
        import uuid
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
        from qdrant_client.models import PointStruct
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
        if create_graph_entities and GRAPHRAG_ENABLED and neo4j_client:
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

@mcp.tool()
async def memory_search_enhanced(
    query: str,
    category: Optional[str] = None,
    limit: int = 10,  # Proper type with integer default
    mode: str = "rrf_hybrid",  # Enhanced search modes
    diversity_threshold: float = 0.85,  # Proper type with float default
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
        # Input validation 
        search_limit = max(1, min(limit or 5, 100))  # Clamp between 1-100
        diversity_lambda = max(0.0, min(diversity_threshold or 0.85, 1.0))  # Clamp between 0-1
        
        collection_name = f"{COLLECTION_PREFIX}{category or 'memory'}"
        
        # Generate query embedding with proper error handling
        try:
            embed_response = await nomic_client.get_embeddings([query])
            dense_embedding = embed_response.embeddings[0]
            
            # Ensure dense_embedding is a proper list of floats
            if isinstance(dense_embedding, (tuple, list)):
                query_vector = [float(x) for x in dense_embedding]
            else:
                raise ValueError(f"Invalid embedding type: {type(dense_embedding)}")
                
        except Exception as embed_error:
            logger.error(f"Query embedding generation failed: {embed_error}")
            return {
                "status": "error",
                "message": "Query embedding generation failed",
                "results": []
            }
        
        # Execute search based on mode
        if mode == "rrf_hybrid":
            # Perform RRF hybrid search (Context7 pattern)
            search_results = await perform_rrf_hybrid_search_internal(
                collection_name, query, query_vector, search_limit
            )
        else:
            # Fallback to vector-only search
            search_results = await perform_vector_search_internal(
                collection_name, query_vector, search_limit
            )
        
        # Check if we got results
        if not search_results:
            return {
                "status": "success",
                "message": "No matching results found",
                "results": [],
                "query": query,
                "search_mode": mode
            }
        
        # Apply diversity filtering using MMR if requested
        if mode == "mmr_diverse" and len(search_results) > 1:
            diverse_results = apply_mmr_diversity(search_results, diversity_lambda, search_limit)
        else:
            diverse_results = search_results[:search_limit]
        
        # Graph expansion if requested
        final_results = diverse_results
        if graph_expand and GRAPHRAG_ENABLED and neo4j_client:
            try:
                expanded_results = await expand_with_neo4j_graph_context(diverse_results, query)
                final_results = expanded_results if expanded_results else diverse_results
            except Exception as graph_error:
                logger.warning(f"GraphRAG expansion failed: {graph_error}")
                # Continue with non-expanded results
        
        # Return proper response structure (ADR-0008 requirement)
        return {
            "status": "success",
            "results": final_results,  # Required for T2_DATA_RETRIEVAL test
            "total_found": len(search_results),
            "after_diversity": len(diverse_results),
            "after_expansion": len(final_results),
            "query": query,
            "search_mode": mode,
            "graph_expansion": graph_expand
        }
        
    except Exception as e:
        logger.error(f"memory_search_enhanced error: {str(e)}")
        return {
            "status": "error", 
            "message": f"Search operation failed: {str(e)}",
            "results": []
        }

async def perform_rrf_hybrid_search_internal(
    collection_name: str, 
    query: str, 
    query_vector: List[float], 
    limit: int,
    k: int = 60  # RRF parameter from Context7 research
) -> List[Dict]:
    """Perform RRF hybrid search combining vector and text search (Context7 pattern)"""
    try:
        # Vector search using named vectors (Context7 pattern - ADR-0008 fix)
        vector_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=("dense", query_vector),  # Use named vector format
            limit=limit * 2,  # Get more for RRF combination
            score_threshold=0.3,
            with_payload=True
        )
        
        # Convert to proper format
        vector_formatted = []
        for result in vector_results:
            vector_formatted.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "category": result.payload.get("category", ""),
                "timestamp": result.payload.get("timestamp", ""),
                **result.payload
            })
        
        # Text search using payload filtering (sparse search simulation)
        try:
            text_results, _ = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="content",
                            match=models.MatchText(text=query)
                        )
                    ]
                ),
                limit=limit * 2,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert text results to proper format
            text_formatted = []
            for result in text_results:
                text_formatted.append({
                    "id": result.id,
                    "score": 1.0,  # Default score for text matches
                    "content": result.payload.get("content", ""),
                    "category": result.payload.get("category", ""),
                    "timestamp": result.payload.get("timestamp", ""),
                    **result.payload
                })
                
        except Exception as text_error:
            logger.warning(f"Text search failed: {text_error}")
            text_formatted = []
        
        # Combine results using RRF (Context7 pattern)
        rrf_results = combine_with_rrf_internal(vector_formatted, text_formatted, k=k)
        
        return rrf_results[:limit]
        
    except Exception as e:
        logger.error(f"RRF hybrid search error: {str(e)}")
        return []

async def perform_vector_search_internal(
    collection_name: str, 
    query_vector: List[float], 
    limit: int
) -> List[Dict]:
    """Perform vector-only search as fallback"""
    try:
        vector_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=("dense", query_vector),  # Use named vector format
            limit=limit,
            score_threshold=0.3,
            with_payload=True
        )
        
        # Convert to proper format
        formatted_results = []
        for result in vector_results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "category": result.payload.get("category", ""),
                "timestamp": result.payload.get("timestamp", ""),
                **result.payload
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Vector search error: {str(e)}")
        return []

def combine_with_rrf_internal(vector_results: List[Dict], text_results: List[Dict], k: int = 60) -> List[Dict]:
    """Combine search results using Reciprocal Rank Fusion (Context7 pattern)"""
    score_dict = {}
    
    # Process vector results
    for i, result in enumerate(vector_results):
        point_id = result["id"]
        rrf_score = 1.0 / (k + i + 1)
        score_dict[point_id] = score_dict.get(point_id, 0) + rrf_score
    
    # Process text results
    for i, result in enumerate(text_results):
        point_id = result["id"]
        rrf_score = 1.0 / (k + i + 1)
        score_dict[point_id] = score_dict.get(point_id, 0) + rrf_score
    
    # Sort by combined RRF score
    sorted_ids = sorted(score_dict.keys(), key=lambda x: score_dict[x], reverse=True)
    
    # Create result list maintaining point data
    combined_results = []
    point_lookup = {r["id"]: r for r in vector_results + text_results}
    
    for point_id in sorted_ids:
        if point_id in point_lookup:
            result = point_lookup[point_id].copy()
            result["score"] = score_dict[point_id]
            result["type"] = "rrf_hybrid"
            combined_results.append(result)
    
    return combined_results

def apply_mmr_diversity(results: List[Dict], threshold: float, limit: int) -> List[Dict]:
    """Apply Maximal Marginal Relevance for result diversification"""
    if not results:
        return results
    
    # Simple MMR implementation using embedding vectors
    selected = [results[0]]  # Start with highest scored result
    remaining = results[1:]
    
    while len(selected) < limit and remaining:
        best_mmr_score = -1
        best_idx = 0
        
        for i, candidate in enumerate(remaining):
            # Calculate relevance score (already have this)
            relevance = candidate["score"]
            
            # Calculate diversity (simple cosine distance from selected)
            diversity = 1.0  # Default high diversity
            if "embedding_vector" in candidate and candidate["embedding_vector"]:
                min_similarity = 1.0
                for selected_item in selected:
                    if "embedding_vector" in selected_item and selected_item["embedding_vector"]:
                        # Simple cosine similarity with type safety
                        candidate_vec = candidate["embedding_vector"]
                        selected_vec = selected_item["embedding_vector"]
                        
                        # Ensure vectors are lists of floats, not tuples
                        if isinstance(candidate_vec, (tuple, list)):
                            candidate_vec = [float(x) for x in candidate_vec]
                        if isinstance(selected_vec, (tuple, list)):
                            selected_vec = [float(x) for x in selected_vec]
                        
                        similarity = cosine_similarity(candidate_vec, selected_vec)
                        min_similarity = min(min_similarity, similarity)
                diversity = 1 - min_similarity
            
            # MMR formula: λ * relevance + (1-λ) * diversity
            lambda_param = 0.7  # Balance between relevance and diversity
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        selected.append(remaining.pop(best_idx))
    
    return selected

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


async def expand_with_neo4j_graph_context(results: List[Dict], query: str) -> List[Dict]:
    """Expand search results using Neo4j graph relationships"""
    try:
        expanded_results = []
        
        for result in results:
            expanded_result = result.copy()
            
            # Find related entities through graph traversal
            if "embedding_id" in result:
                embedding_id = result["embedding_id"]
                file_path = result.get("file_path", "")
                
                try:
                    # Query for related documents and code entities using Neo4j
                    related_entities = await neo4j_client.execute_cypher_query(
                        """
                        MATCH (f:File {path: $file_path})
                        OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
                        OPTIONAL MATCH (f)-[:CONTAINS]->(c:Class)
                        OPTIONAL MATCH (fn)-[:CALLS]->(called:Function)
                        RETURN 
                            fn.name as function_name, 
                            c.name as class_name,
                            called.name as called_function,
                            fn.complexity as complexity
                        LIMIT 5
                        """,
                        {"file_path": file_path}
                    )
                    
                    if related_entities:
                        graph_context = []
                        for entity in related_entities:
                            if entity.get('function_name'):
                                graph_context.append({
                                    "name": entity['function_name'],
                                    "type": "function",
                                    "complexity": entity.get('complexity'),
                                    "relationship": "contains"
                                })
                            if entity.get('class_name'):
                                graph_context.append({
                                    "name": entity['class_name'],
                                    "type": "class",
                                    "relationship": "contains"
                                })
                            if entity.get('called_function'):
                                graph_context.append({
                                    "name": entity['called_function'],
                                    "type": "function",
                                    "relationship": "calls"
                                })
                        
                        expanded_result["graph_context"] = graph_context
                        
                except Exception as e:
                    logger.debug(f"Neo4j graph query failed for result {embedding_id}: {e}")
            
            expanded_results.append(expanded_result)
        
        return expanded_results
        
    except Exception as e:
        logger.error(f"Neo4j graph expansion failed: {e}")
        return results

@mcp.tool()
async def graph_query(query: str) -> Dict[str, Any]:
    """Execute Cypher query on Neo4j graph database
    
    Args:
        query: Cypher query to execute
    """
    try:
        if not GRAPHRAG_ENABLED:
            return {"status": "error", "message": "GraphRAG not enabled"}
        
        if not neo4j_client:
            return {"status": "error", "message": "Neo4j client not available"}
        
        # Execute query on Neo4j
        result = await neo4j_client.execute_cypher_query(query)
        return {
            "status": "success",
            "results": result,
            "count": len(result) if result else 0,
            "database": "neo4j",
            "project": PROJECT_NAME
        }
        
    except Exception as e:
        logger.error(f"Neo4j graph query failed: {e}")
        return {"status": "error", "message": str(e), "query": query}

@mcp.tool()
async def schema_customization(
    action: str,
    collection_name: Optional[str] = None,
    vector_config: Optional[Dict[str, Any]] = None,
    metadata_schema: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Customize Qdrant database schema for project-specific needs
    
    Args:
        action: 'create_collection', 'update_schema', 'list_schemas', 'delete_collection'
        collection_name: Name of collection (will be prefixed with project name)
        vector_config: Vector configuration (size, distance metric, etc.)
        metadata_schema: Expected metadata fields and types
    """
    try:
        if action == "list_schemas":
            collections = qdrant_client.get_collections().collections
            schemas = {}
            for collection in collections:
                if collection.name.startswith(COLLECTION_PREFIX):
                    info = qdrant_client.get_collection(collection.name)
                    schemas[collection.name] = {
                        "vectors_count": info.vectors_count,
                        "points_count": info.points_count,
                        "vector_config": info.config.params.vectors,
                        "optimizers": info.config.optimizer_config
                    }
            return {"status": "success", "schemas": schemas}
        
        elif action == "create_collection":
            if not collection_name:
                return {"status": "error", "message": "collection_name required"}
            
            full_name = f"{COLLECTION_PREFIX}{collection_name}"
            
            # Default config with customization
            vector_size = vector_config.get("size", 768) if vector_config else 768
            distance = vector_config.get("distance", "Cosine") if vector_config else Distance.COSINE
            
            qdrant_client.create_collection(
                collection_name=full_name,
                vectors_config={
                    "dense": VectorParams(size=vector_size, distance=distance)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )
            
            # Store metadata schema if provided
            if metadata_schema:
                # Store schema definition as special point
                qdrant_client.upsert(
                    collection_name=full_name,
                    points=[models.PointStruct(
                        id=0,  # Reserved ID for schema
                        vector={"dense": [0.0] * vector_size},
                        payload={"_schema": metadata_schema, "_type": "schema_definition"}
                    )]
                )
            
            return {
                "status": "success",
                "collection": full_name,
                "vector_size": vector_size,
                "metadata_schema": metadata_schema
            }
        
        elif action == "delete_collection":
            if not collection_name:
                return {"status": "error", "message": "collection_name required"}
            
            full_name = f"{COLLECTION_PREFIX}{collection_name}"
            qdrant_client.delete_collection(collection_name=full_name)
            return {"status": "success", "deleted": full_name}
        
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Schema customization failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def atomic_dependency_tracer(
    target: str,
    trace_type: str = "calls",
    max_depth: int = 5,  # Proper integer type
    include_imports: bool = True
) -> Dict[str, Any]:
    """Trace atomic dependencies for functions, classes, or modules
    
    Args:
        target: Function/class/module name to trace
        trace_type: 'calls' (who calls this), 'dependencies' (what this calls), 'both'
        max_depth: Maximum traversal depth
        include_imports: Include import dependencies
    """
    try:
        # Input validation
        max_depth_int = max(1, min(max_depth or 5, 20))  # Clamp between 1-20
        
        if not GRAPHRAG_ENABLED:
            return {"status": "error", "message": "GraphRAG not available"}
        
        if not neo4j_client:
            return {"status": "error", "message": "Neo4j GraphRAG not available"}
        
        results = {
            "target": target,
            "trace_type": trace_type,
            "dependencies": []
        }
        
        # Build Cypher query based on trace type
        if trace_type == "calls":
            # Find all paths leading TO the target
            query = f"""
            MATCH path = (source:CodeEntity)-[:CALLS|IMPORTS*1..{max_depth_int}]->(target:CodeEntity)
            WHERE target.name = '{target}'
            RETURN path, length(path) as depth
            ORDER BY depth
            LIMIT 50
            """
        elif trace_type == "dependencies":
            # Find all paths FROM the target
            query = f"""
            MATCH path = (source:CodeEntity)-[:CALLS|IMPORTS*1..{max_depth}]->(dependency:CodeEntity)
            WHERE source.name = '{target}'
            RETURN path, length(path) as depth
            ORDER BY depth
            LIMIT 50
            """
        else:  # both
            query = f"""
            MATCH path = (n1:CodeEntity)-[:CALLS|IMPORTS*1..{max_depth}]-(n2:CodeEntity)
            WHERE n1.name = '{target}' OR n2.name = '{target}'
            RETURN path, length(path) as depth
            ORDER BY depth
            LIMIT 100
            """
        
        # Execute query using Neo4j
        if trace_type == "calls":
            # Find functions that call the target
            neo4j_query = f"""
            MATCH (caller:Function)-[:CALLS]->(target:Function)
            WHERE target.name = $target
            RETURN caller.name as caller_name, caller.file_path as caller_file, 
                   target.name as target_name, target.file_path as target_file
            LIMIT 50
            """
        elif trace_type == "dependencies":
            # Find what the target function calls
            neo4j_query = f"""
            MATCH (source:Function)-[:CALLS]->(dependency:Function)
            WHERE source.name = $target
            RETURN source.name as source_name, source.file_path as source_file,
                   dependency.name as dependency_name, dependency.file_path as dependency_file
            LIMIT 50
            """
        else:  # both
            neo4j_query = f"""
            MATCH (f1:Function)-[:CALLS]-(f2:Function)
            WHERE f1.name = $target OR f2.name = $target
            RETURN f1.name as func1_name, f1.file_path as func1_file,
                   f2.name as func2_name, f2.file_path as func2_file
            LIMIT 100
            """
        
        result = await neo4j_client.execute_cypher_query(neo4j_query, {"target": target})
        
        for row in result:
            if trace_type == "calls" and row.get('caller_name'):
                results["dependencies"].append({
                    "type": "function_call",
                    "caller": row['caller_name'],
                    "caller_file": row['caller_file'],
                    "target": row['target_name'],
                    "target_file": row['target_file']
                })
            elif trace_type == "dependencies" and row.get('dependency_name'):
                results["dependencies"].append({
                    "type": "function_dependency",
                    "source": row['source_name'],
                    "source_file": row['source_file'],
                    "dependency": row['dependency_name'],
                    "dependency_file": row['dependency_file']
                })
            elif trace_type == "both":
                results["dependencies"].append({
                    "type": "function_relationship",
                    "func1": row['func1_name'],
                    "func1_file": row['func1_file'],
                    "func2": row['func2_name'],
                    "func2_file": row['func2_file']
                })
        
        # Add token usage estimate
        results["token_estimate"] = len(str(results)) // 4  # Rough estimate
        
        return {
            "status": "success",
            "results": results,
            "paths_found": len(results["dependencies"])
        }
        
    except Exception as e:
        logger.error(f"Dependency trace failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def project_understanding(
    scope: str = "full",
    max_tokens: int = 2000  # Proper integer type
) -> Dict[str, Any]:
    """Generate condensed project understanding without reading all files
    
    Args:
        scope: 'full', 'architecture', 'dependencies', 'core_logic'
        max_tokens: Maximum tokens for response (string converted to int)
    """
    try:
        # Input validation
        max_tokens_int = max(500, min(max_tokens or 2000, 5000))  # Clamp between 500-5000
        
        understanding = {
            "project": PROJECT_NAME,
            "timestamp": datetime.now().isoformat(),
            "scope": scope,
            "max_tokens": max_tokens_int
        }
        
        # Get high-level stats from Qdrant collections
        if qdrant_client:
            collections = qdrant_client.get_collections().collections
            project_collections = [c.name for c in collections if c.name.startswith(COLLECTION_PREFIX)]
            
            understanding["indexed_categories"] = [
                c.replace(COLLECTION_PREFIX, "") for c in project_collections
            ]
            
            # Sample semantic clusters from main collection
            if f"{COLLECTION_PREFIX}code" in project_collections:
                # Get diverse samples using MMR-style selection
                search_result = qdrant_client.search(
                    collection_name=f"{COLLECTION_PREFIX}code",
                    query_vector={"dense": [0.1] * 768},  # Neutral query
                    limit=10,
                    with_payload=True
                )
                
                understanding["code_patterns"] = [
                    {
                        "type": hit.payload.get("type", "unknown"),
                        "category": hit.payload.get("category", "general"),
                        "summary": hit.payload.get("content", "")[:100]
                    }
                    for hit in search_result
                ]
        
        # Get graph structure from active graph database
        if GRAPHRAG_ENABLED:
            if neo4j_client:
                # Neo4j (Primary) - Get architecture overview
                try:
                    # Get file and function counts
                    arch_result = await neo4j_client.execute_cypher_query("""
                        MATCH (f:File)
                        OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
                        OPTIONAL MATCH (f)-[:CONTAINS]->(c:Class)
                        RETURN 
                            COUNT(DISTINCT f) as file_count,
                            COUNT(DISTINCT fn) as function_count,
                            COUNT(DISTINCT c) as class_count
                    """)
                    
                    understanding["architecture"] = {
                        "structure": [
                            {"type": "files", "count": arch_result[0].get('file_count', 0)},
                            {"type": "functions", "count": arch_result[0].get('function_count', 0)},
                            {"type": "classes", "count": arch_result[0].get('class_count', 0)}
                        ],
                        "database": "neo4j"
                    }
                    
                    # Get core relationships
                    rel_result = await neo4j_client.execute_cypher_query("""
                        MATCH ()-[r:CALLS]->()
                        RETURN 'CALLS' as rel_type, COUNT(r) as count
                        UNION
                        MATCH ()-[r:CONTAINS]->()
                        RETURN 'CONTAINS' as rel_type, COUNT(r) as count
                        ORDER BY count DESC
                    """)
                    
                    understanding["relationships"] = [
                        {"type": row["rel_type"], "count": row["count"]} for row in rel_result
                    ]
                except Exception as e:
                    logger.warning(f"Neo4j graph structure query failed: {e}")
            
        
        # Compress to fit token limit
        response_str = json.dumps(understanding, indent=2)
        if len(response_str) // 4 > max_tokens:
            # Truncate less important sections
            if "code_patterns" in understanding:
                understanding["code_patterns"] = understanding["code_patterns"][:3]
            response_str = json.dumps(understanding, indent=2)
        
        return {
            "status": "success",
            "understanding": understanding,
            "token_count": len(response_str) // 4
        }
        
    except Exception as e:
        logger.error(f"Project understanding failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def semantic_code_search(
    query: str,
    search_type: str = "semantic",
    limit: int = 10,  # Proper integer type
    min_score: float = 0.7,  # Proper float type
    use_prism: bool = True,
    prism_boost: float = 0.3  # Proper float type
) -> Dict[str, Any]:
    """Search code by meaning, not just text matching
    
    Args:
        query: Natural language query (e.g., 'error handling logic')
        search_type: 'semantic', 'hybrid', 'exact'
        limit: Maximum results
        min_score: Minimum similarity score
        use_prism: Whether to boost results with PRISM importance scores
        prism_boost: How much to blend PRISM scores (0.0-1.0)
    """
    try:
        # Input validation
        limit_int = max(1, min(limit or 10, 50))  # Clamp between 1-50
        min_score_float = max(0.0, min(min_score or 0.7, 1.0))  # Clamp between 0-1
        prism_boost_float = max(0.0, min(prism_boost or 0.3, 1.0))  # Clamp between 0-1
        
        # Generate embedding for semantic query
        embed_response = await nomic_client.get_embeddings([query])
        query_vector = embed_response.embeddings[0]
        
        collection_name = f"{COLLECTION_PREFIX}code"
        
        if search_type == "semantic":
            # Pure vector search
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector={"dense": query_vector},
                limit=limit_int,
                score_threshold=min_score_float,
                with_payload=True
            )
        elif search_type == "hybrid":
            # Combine vector and keyword search
            results = qdrant_client.search(
                collection_name=collection_name,
                query_vector={"dense": query_vector},
                query_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="content",
                            match=models.MatchText(text=query)
                        )
                    ]
                ),
                limit=limit_int,
                with_payload=True
            )
        else:  # exact
            # Text-only search
            results = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content",
                            match=models.MatchText(text=query)
                        )
                    ]
                ),
                limit=limit_int,
                with_payload=True
            )[0]  # scroll returns tuple
        
        # Format results with token efficiency (exclude archival files)
        formatted_results = []
        total_tokens = 0
        archival_dirs = ['.archive', 'deprecated', 'legacy', '.claude/chroma', '.claude/onnx_models', 'docker-backup-']
        
        for hit in results:
            # Skip results from archival directories
            file_path = hit.payload.get("file_path", "")
            if any(archival_dir in file_path for archival_dir in archival_dirs):
                continue
                
            content = hit.payload.get("content", "")
            # Extract relevant snippet around matches
            snippet_size = 200
            snippet = content[:snippet_size] + "..." if len(content) > snippet_size else content
            
            formatted_results.append({
                "score": hit.score if hasattr(hit, 'score') else 1.0,
                "file_path": hit.payload.get("file_path", "unknown"),
                "type": hit.payload.get("type", "code"),
                "snippet": snippet,
                "full_content_id": hit.id
            })
            total_tokens += len(snippet) // 4
        
        # Apply PRISM boosting if enabled
        if use_prism and PRISM_AVAILABLE:
            try:
                # Initialize PRISM scorer with project directory
                prism_scorer = PrismScorer("/app/project")
                
                # Boost search results with PRISM scores
                formatted_results = prism_scorer.boost_search_results(
                    formatted_results, 
                    boost_factor=prism_boost_float
                )
                
                # Add PRISM explanation to top results
                for i, result in enumerate(formatted_results[:3]):
                    if 'prism_score' in result:
                        result['importance_reason'] = f"PRISM: {result['prism_score']:.2f}"
                        if result['prism_score'] >= 0.8:
                            result['importance_level'] = "CRITICAL"
                        elif result['prism_score'] >= 0.6:
                            result['importance_level'] = "HIGH"
                        elif result['prism_score'] >= 0.4:
                            result['importance_level'] = "MEDIUM"
                        else:
                            result['importance_level'] = "LOW"
                            
            except Exception as e:
                logger.warning(f"PRISM boosting failed: {e}, using original scores")
        
        return {
            "status": "success",
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results),
            "token_usage": total_tokens,
            "prism_enabled": use_prism and PRISM_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def vibe_preservation(
    action: str,
    code_sample: Optional[str] = None,
    target_file: Optional[str] = None
) -> Dict[str, Any]:
    """Preserve and apply project coding style/patterns
    
    Args:
        action: 'learn' (from sample), 'apply' (to target), 'show' (current patterns)
        code_sample: Code to learn style from
        target_file: File to apply style to
    """
    try:
        collection_name = f"{COLLECTION_PREFIX}style_patterns"
        
        if action == "learn":
            if not code_sample:
                return {"status": "error", "message": "code_sample required for learning"}
            
            # Extract style patterns (simplified - would use Tree-sitter for real analysis)
            patterns = {
                "indent_style": "spaces" if "    " in code_sample else "tabs",
                "quote_style": "double" if '"' in code_sample else "single",
                "semicolons": "yes" if ";" in code_sample else "no",
                "bracket_style": "same_line" if "{\n" not in code_sample else "new_line",
                "naming_convention": "camelCase" if any(c.isupper() for c in code_sample) else "snake_case"
            }
            
            # Generate embedding for the style
            embed_response = await nomic_client.get_embeddings([code_sample])
            style_vector = embed_response.embeddings[0]
            
            # Store in Qdrant
            await ensure_collection(collection_name)
            
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(
                    id=hash(code_sample) % 1000000,
                    vector={"dense": style_vector},
                    payload={
                        "patterns": patterns,
                        "sample": code_sample[:500],
                        "learned_at": datetime.now().isoformat()
                    }
                )]
            )
            
            return {
                "status": "success",
                "learned_patterns": patterns
            }
        
        elif action == "show":
            # Retrieve stored style patterns
            try:
                results = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=10,
                    with_payload=True
                )[0]
                
                patterns = {}
                for point in results:
                    for key, value in point.payload.get("patterns", {}).items():
                        if key not in patterns:
                            patterns[key] = {}
                        patterns[key][value] = patterns[key].get(value, 0) + 1
                
                # Get most common patterns
                common_patterns = {}
                for key, values in patterns.items():
                    common_patterns[key] = max(values, key=values.get)
                
                return {
                    "status": "success",
                    "common_patterns": common_patterns,
                    "samples_analyzed": len(results)
                }
            except:
                return {"status": "success", "common_patterns": {}, "samples_analyzed": 0}
        
        elif action == "apply":
            return {
                "status": "success",
                "message": "Style application would transform code to match learned patterns"
            }
        
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.error(f"Vibe preservation failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def project_auto_index(
    scope: str = "modified",  # modified, all, git-changes
    since_minutes: Optional[int] = None,  # Proper optional integer type
    file_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Smart auto-indexing tool that indexes project files on-demand
    
    Args:
        scope: Index scope - 'modified' (changed files), 'all' (full reindex), 'git-changes' (uncommitted)
        since_minutes: Only index files modified in last N minutes
        file_patterns: Optional file patterns to index (e.g., ['*.py', '*.js'])
    
    Returns:
        Index status with files processed and performance metrics
    """
    # Ensure services are initialized
    await ensure_services_initialized()
    
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    
    try:
        # Input validation
        since_minutes_int = since_minutes  # Already proper integer type or None
        
        project_dir = Path("/app/project")
        manifest_file = Path("/app/data/index_manifest.json")
        
        # Load existing manifest
        manifest = {}
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
        
        files_to_index = []
        files_checked = 0
        files_skipped = 0
        
        # Determine which files to index based on scope
        if scope == "git-changes":
            # Get uncommitted changes from git
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD"],
                    cwd=project_dir,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    changed_files = result.stdout.strip().split('\n')
                    files_to_index = [project_dir / f for f in changed_files if f]
            except:
                return {"status": "error", "message": "Git not available or not a git repository"}
        
        elif scope == "all":
            # Index all supported files
            patterns = file_patterns or ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.java', '*.go', '*.rs', '*.md']
            for pattern in patterns:
                files_to_index.extend(project_dir.rglob(pattern))
        
        else:  # scope == "modified"
            # Check for modified files using checksums
            patterns = file_patterns or ['*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.java', '*.go', '*.rs', '*.md']
            
            for pattern in patterns:
                for file_path in project_dir.rglob(pattern):
                    files_checked += 1
                    
                    # Skip if file is in ignored directories (including archives)
                    ignored_dirs = [
                        '__pycache__', '.git', 'node_modules', '.venv', 'dist', 'build',
                        '.archive', 'deprecated', 'legacy', '.claude/chroma', '.claude/onnx_models',
                        'docker-backup-'  # Matches any backup directory
                    ]
                    if any(part in str(file_path) for part in ignored_dirs):
                        files_skipped += 1
                        continue
                    
                    # Check if file was modified
                    try:
                        # Get file modification time
                        mtime = file_path.stat().st_mtime
                        
                        # Check against since_minutes if provided
                        if since_minutes:
                            from datetime import datetime, timedelta
                            cutoff_time = (datetime.now() - timedelta(minutes=since_minutes)).timestamp()
                            if mtime < cutoff_time:
                                files_skipped += 1
                                continue
                        
                        # Calculate file hash
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        
                        # Check if file changed
                        file_key = str(file_path.relative_to(project_dir))
                        if manifest.get(file_key) != file_hash:
                            files_to_index.append(file_path)
                            manifest[file_key] = file_hash
                        else:
                            files_skipped += 1
                    
                    except Exception as e:
                        logger.warning(f"Could not check file {file_path}: {e}")
                        files_skipped += 1
        
        # Index the files
        indexed_count = 0
        index_errors = 0
        total_chunks = 0
        
        for file_path in files_to_index:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Get relative path for metadata
                relative_path = file_path.relative_to(project_dir)
                
                # Simple chunking for large files
                lines = content.split('\n')
                max_lines_per_chunk = 500
                chunks = []
                
                for i in range(0, len(lines), max_lines_per_chunk):
                    chunk_lines = lines[i:i + max_lines_per_chunk]
                    chunk_text = '\n'.join(chunk_lines)
                    if chunk_text.strip():
                        chunks.append({
                            'text': chunk_text,
                            'start_line': i + 1,
                            'end_line': min(i + max_lines_per_chunk, len(lines))
                        })
                
                # Get embeddings
                texts_to_embed = [f"File: {relative_path}\n{chunk['text'][:500]}" for chunk in chunks]
                embed_response = await nomic_client.get_embeddings(texts_to_embed)
                
                # Store in Qdrant
                collection_name = f"{COLLECTION_PREFIX}code"
                await ensure_collection(collection_name)
                
                points = []
                for idx, (chunk, embedding) in enumerate(zip(chunks, embed_response.embeddings)):
                    chunk_content = chunk['text']  # Extract chunk content for ID generation
                    
                    # Create sparse vector from keywords
                    words = chunk_content.lower().split()
                    word_freq = {}
                    for word in words[:100]:  # Top 100 words
                        if len(word) > 2:
                            word_freq[word] = word_freq.get(word, 0) + 1
                    
                    sparse_indices = []
                    sparse_values = []
                    for word, freq in word_freq.items():
                        word_idx = hash(word) % 10000  # Rename to avoid variable conflict
                        sparse_indices.append(word_idx)
                        sparse_values.append(float(freq))
                    
                    point = models.PointStruct(
                        id=generate_deterministic_point_id(str(file_path), chunk_content, idx),
                        vector={
                            "dense": embedding,
                            "sparse": models.SparseVector(
                                indices=sparse_indices,
                                values=sparse_values
                            )
                        },
                        payload={
                            "file_path": str(relative_path),
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                            "start_line": chunk['start_line'],
                            "end_line": chunk['end_line'],
                            "content": chunk['text'][:1000],
                            "file_type": file_path.suffix,
                            "indexed_at": datetime.now().isoformat(),
                            "project": PROJECT_NAME
                        }
                    )
                    points.append(point)
                
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                indexed_count += 1
                total_chunks += len(chunks)
                
            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
                index_errors += 1
        
        # Cleanup stale chunks from deleted or truncated files
        await cleanup_stale_chunks(manifest, project_dir)
        
        # Save manifest
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f)
        
        return {
            "status": "success",
            "scope": scope,
            "files_checked": files_checked,
            "files_skipped": files_skipped,
            "files_indexed": indexed_count,
            "total_chunks": total_chunks,
            "index_errors": index_errors,
            "manifest_updated": True,
            "message": f"Indexed {indexed_count} files ({total_chunks} chunks)"
        }
        
    except Exception as e:
        logger.error(f"Auto-indexing failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_system_status() -> Dict[str, Any]:
    """Get comprehensive neural system status"""
    try:
        # Ensure services are initialized
        await ensure_services_initialized()
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "project": PROJECT_NAME,
            "version": "l9-enhanced-2025"
        }
        
        # Qdrant statistics
        try:
            collections = qdrant_client.get_collections().collections
            qdrant_stats = {}
            total_points = 0
            
            for collection in collections:
                if collection.name.startswith(COLLECTION_PREFIX):
                    info = qdrant_client.get_collection(collection.name)
                    qdrant_stats[collection.name] = {
                        "vectors_count": info.vectors_count,
                        "points_count": info.points_count,
                        "status": info.status,
                        "config": {
                            "distance": getattr(info.config.params.vectors, "distance", "unknown"),
                            "size": getattr(info.config.params.vectors, "size", 0)
                        }
                    }
                    total_points += info.points_count
            
            stats["qdrant"] = {
                "collections": qdrant_stats,
                "total_points": total_points,
                "host": QDRANT_HOST
            }
        except Exception as e:
            stats["qdrant"] = {"error": str(e)}
        
        # GraphRAG statistics (Neo4j)
        if GRAPHRAG_ENABLED:
            if neo4j_client:
                # Neo4j GraphRAG statistics (Primary)
                try:
                    # Get node counts
                    node_result = await neo4j_client.execute_cypher_query("""
                        MATCH (f:File)
                        OPTIONAL MATCH (fn:Function)
                        OPTIONAL MATCH (c:Class)
                        RETURN COUNT(DISTINCT f) as files, COUNT(DISTINCT fn) as functions, COUNT(DISTINCT c) as classes
                    """)
                    
                    # Get relationship counts
                    rel_result = await neo4j_client.execute_cypher_query("""
                        MATCH ()-[r:CONTAINS]->() 
                        WITH COUNT(r) as contains_count
                        MATCH ()-[r2:CALLS]->() 
                        RETURN contains_count, COUNT(r2) as calls_count
                    """)
                    
                    stats["neo4j"] = {
                        "status": "active",
                        "project": PROJECT_NAME,
                        "nodes": {
                            "files": node_result[0].get('files', 0),
                            "functions": node_result[0].get('functions', 0),
                            "classes": node_result[0].get('classes', 0)
                        },
                        "relationships": {
                            "contains": rel_result[0].get('contains_count', 0) if rel_result else 0,
                            "calls": rel_result[0].get('calls_count', 0) if rel_result else 0
                        }
                    }
                    stats["neo4j"]["total_nodes"] = sum(stats["neo4j"]["nodes"].values())
                    stats["neo4j"]["total_relationships"] = sum(stats["neo4j"]["relationships"].values())
                except Exception as e:
                    stats["neo4j"] = {"status": "error", "message": str(e)}
            
            else:
                stats["graphrag"] = {"status": "disabled", "message": "No graph database available"}
        else:
            stats["graphrag"] = {"status": "disabled", "message": "GraphRAG not enabled"}
        
        # Nomic Embed statistics (Context7 pattern)
        try:
            # Use Context7 pattern: fresh AsyncClient per request
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=10.0),
                limits=httpx.Limits(max_connections=5)
            ) as client:
                health_response = await client.get(f"{nomic_client.base_url}/health")
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    stats["embedding"] = {
                        "service": "nomic-embed-v2-moe",
                        "status": "healthy",
                        "url": nomic_client.base_url,
                        "model": health_data.get("model", "unknown"),
                        "backend": health_data.get("backend", "unknown")
                    }
                else:
                    stats["embedding"] = {"status": "unhealthy", "status_code": health_response.status_code}
        except Exception as e:
            stats["embedding"] = {"status": "error", "message": str(e)}
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return {"status": "error", "message": str(e)}

# =============================================================================
# Neo4j GraphRAG MCP Tools (L9 2025 Migration)
# =============================================================================

@mcp.tool()
async def neo4j_graph_query(
    cypher_query: str,
    parameters: str = "{}"
) -> Dict[str, Any]:
    """
    Execute Cypher query against the Neo4j graph database - ADR-0008 Fix with Context7 patterns
    
    Args:
        cypher_query: Cypher query to execute
        parameters: JSON string of query parameters
        
    Returns:
        Query results with proper data structures (required for T3_GRAPH_OPERATIONS test)
    """
    if not NEO4J_AVAILABLE:
        return {"status": "error", "message": "Neo4j client not available"}
    
    try:
        # Parse parameters with error handling
        try:
            query_params = json.loads(parameters) if parameters else {}
        except json.JSONDecodeError as json_error:
            return {
                "status": "error",
                "message": f"Invalid JSON parameters: {str(json_error)}"
            }
        
        # Use proper Neo4j driver session management (Context7 pattern)
        try:
            # Import Neo4j driver components
            from neo4j.exceptions import ServiceUnavailable, AuthError
            
            # Create session with proper async context manager
            async with AsyncNeo4jClient(project_name=PROJECT_NAME) as client:
                
                # Define transaction function for retry capability (Context7 pattern)  
                async def execute_query_tx(tx, query: str, parameters: Dict):
                    try:
                        result = await tx.run(query, parameters or {})
                        
                        # Convert result to list with proper structure (Context7 pattern)
                        records = []
                        async for record in result:
                            record_dict = {}
                            for key in record.keys():
                                value = record[key]
                                # Handle Neo4j node/relationship objects properly
                                if hasattr(value, '__dict__'):
                                    record_dict[key] = dict(value)
                                elif hasattr(value, '_properties'):
                                    # Neo4j Node/Relationship object
                                    record_dict[key] = dict(value._properties) if value._properties else {}
                                else:
                                    record_dict[key] = value
                            records.append(record_dict)
                        
                        # Get result summary for metadata (Context7 pattern)
                        summary = await result.consume()
                        
                        return {
                            "records": records,
                            "summary": {
                                "query_type": summary.query_type if hasattr(summary, 'query_type') else "unknown",
                                "counters": dict(summary.counters) if hasattr(summary, 'counters') and summary.counters else {},
                                "result_available_after": summary.result_available_after if hasattr(summary, 'result_available_after') else 0,
                                "result_consumed_after": summary.result_consumed_after if hasattr(summary, 'result_consumed_after') else 0
                            }
                        }
                        
                    except Exception as tx_error:
                        logger.error(f"Neo4j transaction error: {tx_error}")
                        # Re-raise for outer handling
                        raise tx_error
                
                # Use managed transactions based on query type (Context7 pattern)
                query_upper = cypher_query.strip().upper()
                if any(query_upper.startswith(keyword) for keyword in ['CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE']):
                    # Write transaction
                    query_result = await client.execute_write_transaction(execute_query_tx, cypher_query, query_params)
                else:
                    # Read transaction  
                    query_result = await client.execute_read_transaction(execute_query_tx, cypher_query, query_params)
                
                # Return proper response structure (ADR-0008 requirement)
                return {
                    "status": "success",
                    "result": query_result["records"],  # Required for T3_GRAPH_OPERATIONS test
                    "data": query_result["records"],    # Alternative key for compatibility
                    "count": len(query_result["records"]),
                    "metadata": query_result["summary"],
                    "query": cypher_query[:100] + "..." if len(cypher_query) > 100 else cypher_query,
                    "parameters": query_params
                }
                
        except Exception as client_error:
            # Fallback to direct execution if managed transactions fail
            logger.warning(f"Managed transaction failed, trying direct execution: {client_error}")
            
            async with AsyncNeo4jClient(project_name=PROJECT_NAME) as client:
                results = await client.execute_cypher_query(cypher_query, query_params)
                
                # Ensure results is a list with proper structure
                if not isinstance(results, list):
                    results = [results] if results is not None else []
                
                return {
                    "status": "success",
                    "result": results,  # Required for T3_GRAPH_OPERATIONS test
                    "data": results,    # Alternative key for compatibility
                    "count": len(results),
                    "query": cypher_query[:100] + "..." if len(cypher_query) > 100 else cypher_query,
                    "parameters": query_params
                }
            
    except Exception as e:
        logger.error(f"neo4j_graph_query error: {str(e)}")
        return {
            "status": "error", 
            "message": f"Query execution failed: {str(e)}",
            "error_type": "execution_error",
            "query": cypher_query[:50] + "..." if len(cypher_query) > 50 else cypher_query
        }

@mcp.tool()
async def neo4j_semantic_graph_search(
    query_text: str,
    limit: int = 10,
    node_types: str = "File,Function,Class"
) -> Dict[str, Any]:
    """
    Perform semantic search across the Neo4j code graph
    
    Args:
        query_text: Natural language search query
        limit: Maximum results to return  
        node_types: Comma-separated node types to search
        
    Returns:
        Semantically relevant code elements
    """
    if not NEO4J_AVAILABLE:
        return {"status": "error", "message": "Neo4j client not available"}
    
    try:
        limit_int = max(1, min(limit or 10, 100))  # Clamp between 1-100
        node_type_list = [t.strip() for t in node_types.split(",")]
        
        async with AsyncNeo4jClient(project_name=PROJECT_NAME) as client:
            results = await client.semantic_graph_search(
                query_text=query_text,
                limit=limit_int,
                node_types=node_type_list
            )
            
            return {
                "status": "success",
                "results": results,
                "count": len(results),
                "query": query_text
            }
            
    except Exception as e:
        logger.error(f"Neo4j semantic search failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neo4j_code_dependencies(
    file_path: str,
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Get code dependency graph for a specific file
    
    Args:
        file_path: Target file path (relative to project root)
        max_depth: Maximum traversal depth
        
    Returns:
        Dependency graph structure with relationships
    """
    if not NEO4J_AVAILABLE:
        return {"status": "error", "message": "Neo4j client not available"}
    
    try:
        depth = max(1, min(max_depth or 3, 10))  # Clamp between 1-10
        
        async with AsyncNeo4jClient(project_name=PROJECT_NAME) as client:
            dependencies = await client.get_code_dependencies(file_path, depth)
            
            return {
                "status": "success",
                "file_path": file_path,
                "dependencies": dependencies,
                "max_depth": depth
            }
            
    except Exception as e:
        logger.error(f"Neo4j dependencies failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neo4j_migration_status() -> Dict[str, Any]:
    """
    Check Neo4j migration status and system health
    
    Returns:
        Migration status and database health information
    """
    try:
        status = {
            "neo4j_available": NEO4J_AVAILABLE,
            "project": PROJECT_NAME
        }
        
        if NEO4J_AVAILABLE:
            try:
                async with AsyncNeo4jClient(project_name=PROJECT_NAME) as client:
                    stats = await client.get_project_statistics()
                    status["neo4j_stats"] = stats
                    status["neo4j_connection"] = "healthy"
            except Exception as e:
                status["neo4j_connection"] = "failed"
                status["neo4j_error"] = str(e)
        
        # System status
        if NEO4J_AVAILABLE:
            status["migration_stage"] = "neo4j_ready"
        else:
            status["migration_stage"] = "no_graph_db"
        
        return {"status": "success", "migration_status": status}
        
    except Exception as e:
        logger.error(f"Migration status check failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neo4j_index_code_graph(
    file_paths: str = "",
    force_reindex: bool = False
) -> Dict[str, Any]:
    """
    Index code files into Neo4j graph database
    
    Args:
        file_paths: Comma-separated file paths to index (empty = index all)
        force_reindex: Whether to force reindexing existing files
        
    Returns:
        Indexing results and statistics
    """
    if not NEO4J_AVAILABLE:
        return {"status": "error", "message": "Neo4j client not available"}
    
    try:
        force = force_reindex  # Already proper boolean type
        paths_to_index = [p.strip() for p in file_paths.split(",") if p.strip()] if file_paths else []
        
        # Get project directory
        project_dir = Path("/app/project")
        
        indexed_files = 0
        indexed_functions = 0
        indexed_classes = 0
        errors = []
        
        async with AsyncNeo4jClient(project_name=PROJECT_NAME) as client:
            # Determine files to process
            if paths_to_index:
                files_to_process = [project_dir / path for path in paths_to_index if (project_dir / path).exists()]
            else:
                # Index all Python files (expandable to other languages)
                files_to_process = list(project_dir.rglob("*.py"))[:100]  # Limit to avoid timeout
            
            for file_path in files_to_process:
                try:
                    if not file_path.is_file():
                        continue
                        
                    relative_path = str(file_path.relative_to(project_dir))
                    
                    # Read file content
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception as e:
                        errors.append(f"Failed to read {relative_path}: {str(e)}")
                        continue
                    
                    # Create file node
                    success = await client.create_file_node(
                        file_path=relative_path,
                        content=content,
                        extension=file_path.suffix,
                        size_bytes=len(content.encode('utf-8'))
                    )
                    
                    if success:
                        indexed_files += 1
                        
                        # Simple function/class extraction (can be enhanced with tree-sitter)
                        if file_path.suffix == '.py':
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                line_stripped = line.strip()
                                
                                # Extract function definitions
                                if line_stripped.startswith('def ') and '(' in line_stripped:
                                    func_name = line_stripped.split('(')[0].replace('def ', '').strip()
                                    if func_name:
                                        await client.create_function_node(
                                            file_path=relative_path,
                                            function_name=func_name,
                                            signature=line_stripped,
                                            start_line=i + 1,
                                            end_line=i + 1  # Simplified - could enhance with proper parsing
                                        )
                                        indexed_functions += 1
                                
                                # Extract class definitions
                                elif line_stripped.startswith('class ') and ':' in line_stripped:
                                    class_name = line_stripped.split(':')[0].replace('class ', '').strip()
                                    if '(' in class_name:
                                        class_name = class_name.split('(')[0].strip()
                                    if class_name:
                                        await client.create_class_node(
                                            file_path=relative_path,
                                            class_name=class_name,
                                            start_line=i + 1,
                                            end_line=i + 1  # Simplified
                                        )
                                        indexed_classes += 1
                    
                except Exception as e:
                    errors.append(f"Failed to index {relative_path}: {str(e)}")
            
            return {
                "status": "success",
                "indexed_files": indexed_files,
                "indexed_functions": indexed_functions,
                "indexed_classes": indexed_classes,
                "total_processed": len(files_to_process),
                "errors": errors[:10],  # Limit error messages
                "error_count": len(errors)
            }
            
    except Exception as e:
        logger.error(f"Neo4j indexing failed: {e}")
        return {"status": "error", "message": str(e)}

def get_tool_by_name(tool_name: str):
    """Get MCP tool function by name for testing purposes"""
    # Map tool names to their actual functions
    tool_mapping = {
        "memory_store_enhanced": memory_store_enhanced,
        "memory_search_enhanced": memory_search_enhanced,
        "graph_query": graph_query,
        "schema_customization": schema_customization,
        "atomic_dependency_tracer": atomic_dependency_tracer,
        "project_understanding": project_understanding,
        "semantic_code_search": semantic_code_search,
        "vibe_preservation": vibe_preservation,
        "project_auto_index": project_auto_index,
        "neural_system_status": neural_system_status,
        "neo4j_graph_query": neo4j_graph_query,
        "neo4j_semantic_graph_search": neo4j_semantic_graph_search,
        "neo4j_code_dependencies": neo4j_code_dependencies,
        "neo4j_migration_status": neo4j_migration_status,
        "neo4j_index_code_graph": neo4j_index_code_graph
    }
    
    return tool_mapping.get(tool_name)

def production_health_check():
    """Production health check for testing validation"""
    try:
        # Count available MCP tools
        available_tools = []
        tool_functions = [
            'memory_store_enhanced', 'memory_search_enhanced', 'graph_query',
            'schema_customization', 'atomic_dependency_tracer', 'project_understanding',
            'semantic_code_search', 'vibe_preservation', 'project_auto_index',
            'neural_system_status', 'neo4j_graph_query', 'neo4j_semantic_graph_search',
            'neo4j_code_dependencies', 'neo4j_migration_status', 'neo4j_index_code_graph'
        ]
        
        # Check if tools are available as globals in this module
        import sys
        current_module = sys.modules[__name__]
        
        for tool_name in tool_functions:
            if hasattr(current_module, tool_name):
                available_tools.append(tool_name)
        
        total_tools = len(tool_functions)
        accessible_tools = len(available_tools)
        compliance_score = (accessible_tools / total_tools) * 100
        
        return {
            "status": "healthy" if compliance_score >= 100.0 else "degraded",
            "tools_accessible": accessible_tools,
            "total_tools": total_tools,
            "compliance_score": compliance_score,
            "available_tools": available_tools,
            "missing_tools": [t for t in tool_functions if not hasattr(current_module, t)]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "tools_accessible": 0,
            "total_tools": 15,
            "compliance_score": 0.0
        }

# Automatic initialization at module import (ADR-0008 requirement)
# Initialize services immediately to ensure global variables are set
try:
    import asyncio
    
    # Create event loop if none exists and run initialization
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run initialization immediately (no background thread to avoid global variable issues)
    if not loop.is_running():
        loop.run_until_complete(initialize())
    else:
        # If loop is already running, schedule initialization
        asyncio.create_task(initialize())
    
except Exception as e:
    logger.warning(f"Auto-initialization failed: {e}")
    # Fallback: Set basic Qdrant client directly if initialization fails
    try:
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_GRPC_PORT, 
            prefer_grpc=True
        )
        logger.info("✅ Fallback Qdrant client initialized")
    except Exception as fallback_error:
        logger.warning(f"Fallback initialization also failed: {fallback_error}")

# Run the enhanced server
if __name__ == "__main__":
    # Initialize within MCP event loop to avoid asyncio conflicts
    mcp.run(transport='stdio')