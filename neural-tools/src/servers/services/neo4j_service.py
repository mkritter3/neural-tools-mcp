#!/usr/bin/env python3
"""
Neo4j Graph Database Service - Extracted from monolithic neural-mcp-server-enhanced.py
Provides GraphRAG operations and code relationship analysis
"""

import os
import logging
import json
import hashlib
import time
from typing import Dict, List, Any, Optional

# ADR-0096: Import robust vector search implementation
try:
    from robust_vector_search import RobustVectorSearch
    ROBUST_SEARCH_AVAILABLE = True
except ImportError:
    from pathlib import Path
    import sys
    # Try to add the services directory to path
    services_dir = Path(__file__).parent
    sys.path.insert(0, str(services_dir))
    try:
        from robust_vector_search import RobustVectorSearch
        ROBUST_SEARCH_AVAILABLE = True
    except ImportError:
        ROBUST_SEARCH_AVAILABLE = False
        RobustVectorSearch = None

# Neo4j imports with availability check
NEO4J_AVAILABLE = False
try:
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError

    NEO4J_AVAILABLE = True
except ImportError:
    logging.warning("Neo4j driver not available - GraphRAG features disabled")

logger = logging.getLogger(__name__)


class AsyncNeo4jClient:
    """Async Neo4j client with proper session management"""

    def __init__(self, project_name: str = "default"):
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available")

        self.project_name = project_name

        # Neo4j configuration from environment - prefer NEO4J_URI if available
        neo4j_uri = os.environ.get("NEO4J_URI")
        neo4j_username = os.environ.get("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.environ.get("NEO4J_PASSWORD", "graphrag-password")

        if neo4j_uri:
            # Use NEO4J_URI when provided (MCP config case)
            self.uri = neo4j_uri
            logger.info(f"Using NEO4J_URI: {self.uri}")
        else:
            # Fall back to NEO4J_HOST/NEO4J_PORT construction
            neo4j_host = os.environ.get("NEO4J_HOST", "localhost")
            neo4j_port = int(os.environ.get("NEO4J_PORT", 47687))
            self.uri = f"bolt://{neo4j_host}:{neo4j_port}"
            logger.info(f"Constructed URI from NEO4J_HOST/PORT: {self.uri}")
        self.driver = AsyncGraphDatabase.driver(
            self.uri, auth=(neo4j_username, neo4j_password)
        )

        # ADR-0096: Initialize robust vector search
        self.robust_search = None  # Will be initialized after connection

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the driver"""
        if self.driver:
            await self.driver.close()

    async def execute_query(
        self, cypher_query: str, parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute Cypher query with proper error handling and session management"""
        try:
            async with self.driver.session() as session:

                # Define transaction function for retry capability
                async def execute_query_tx(tx, query: str, parameters: Dict):
                    try:
                        result = await tx.run(query, parameters or {})

                        # Convert result to list with proper structure
                        records = []
                        async for record in result:
                            record_dict = {}
                            for key in record.keys():
                                value = record[key]
                                # Handle Neo4j node/relationship objects
                                if hasattr(value, "__dict__"):
                                    record_dict[key] = dict(value)
                                else:
                                    record_dict[key] = value
                            records.append(record_dict)

                        # Get result summary for metadata
                        summary = await result.consume()

                        return {
                            "records": records,
                            "summary": {
                                "query_type": summary.query_type,
                                "counters": (
                                    summary.counters.__dict__
                                    if hasattr(summary.counters, "__dict__")
                                    else {}
                                ),
                                "result_available_after": summary.result_available_after,
                                "result_consumed_after": summary.result_consumed_after,
                            },
                        }

                    except Exception as tx_error:
                        raise tx_error

                # Execute with managed transaction for retry capability
                # Check if query contains write operations (anywhere in the query)
                query_upper = cypher_query.strip().upper()
                write_keywords = ("CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DROP")
                is_write_query = any(
                    keyword in query_upper for keyword in write_keywords
                )

                if is_write_query:
                    # Write transaction
                    query_result = await session.execute_write(
                        execute_query_tx, cypher_query, parameters or {}
                    )
                else:
                    # Read transaction
                    query_result = await session.execute_read(
                        execute_query_tx, cypher_query, parameters or {}
                    )

                return {
                    "status": "success",
                    "result": query_result["records"],
                    "metadata": query_result["summary"],
                    "query": cypher_query,
                    "parameters": parameters,
                }

        except AuthError as auth_error:
            return {
                "status": "error",
                "message": f"Neo4j authentication failed: {str(auth_error)}",
                "error_type": "authentication_error",
            }
        except ServiceUnavailable as service_error:
            return {
                "status": "error",
                "message": f"Neo4j service unavailable: {str(service_error)}",
                "error_type": "service_unavailable",
            }
        except Exception as e:
            logger.error(f"neo4j_graph_query error: {str(e)}")
            return {
                "status": "error",
                "message": f"Query execution failed: {str(e)}",
                "error_type": "execution_error",
            }


class Neo4jService:
    """Service class for Neo4j GraphRAG operations with intelligent caching"""

    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.client = None
        self.initialized = False
        self.available = NEO4J_AVAILABLE
        self.service_container = (
            None  # Will be set by service container for cache access
        )

        # ADR-0096: Robust vector search instance
        self.robust_search = None

        # Cache configuration
        self.enable_caching = (
            os.getenv("NEO4J_ENABLE_CACHING", "true").lower() == "true"
        )
        self.cache_ttl_default = int(
            os.getenv("NEO4J_CACHE_TTL", 3600)
        )  # 1 hour default
        self.cache_ttl_semantic = int(
            os.getenv("NEO4J_SEMANTIC_CACHE_TTL", 1800)
        )  # 30 min for semantic queries

    def set_service_container(self, container):
        """Set reference to service container for cache access"""
        self.service_container = container

    def _generate_cache_key(
        self, query: str, parameters: Optional[Dict] = None, key_type: str = "query"
    ) -> str:
        """Generate cache key for Neo4j queries with parameters"""
        # Create deterministic hash from query and parameters
        query_content = f"{query}:{json.dumps(parameters or {}, sort_keys=True)}"
        query_hash = hashlib.sha256(query_content.encode()).hexdigest()[:16]
        return f"l9:prod:neural_tools:neo4j:{key_type}:{self.project_name}:{query_hash}"

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get query result from cache if available"""
        if not self.enable_caching or not self.service_container:
            return None

        try:
            redis_cache = await self.service_container.get_redis_cache_client()
            cached_result = await redis_cache.get(cache_key)

            if cached_result:
                result = json.loads(cached_result)
                # Refresh TTL on cache hit for frequently accessed queries
                await redis_cache.expire(cache_key, self.cache_ttl_default)
                logger.debug(f"Neo4j cache hit for key: {cache_key[:50]}...")
                return result

        except Exception as e:
            logger.warning(f"Failed to get from Neo4j cache: {e}")

        return None

    async def _store_in_cache(
        self, cache_key: str, result: Dict[str, Any], ttl: Optional[int] = None
    ):
        """Store query result in cache"""
        if not self.enable_caching or not self.service_container:
            return

        try:
            redis_cache = await self.service_container.get_redis_cache_client()
            cache_ttl = ttl or self.cache_ttl_default

            await redis_cache.setex(cache_key, cache_ttl, json.dumps(result))
            logger.debug(f"Neo4j result cached with TTL {cache_ttl}s")

        except Exception as e:
            logger.warning(f"Failed to store in Neo4j cache: {e}")

    async def initialize(self) -> Dict[str, Any]:
        """Initialize Neo4j service with connectivity check"""
        if not self.available:
            return {
                "success": False,
                "message": "Neo4j driver not available - GraphRAG features disabled",
            }

        try:
            self.client = AsyncNeo4jClient(self.project_name)

            # Test connectivity with simple query
            result = await self.client.execute_query("RETURN 1 as test")

            if result["status"] == "success" and result["result"]:
                # Ensure basic constraints exist
                await self._ensure_constraints()

                # ADR-0096: Initialize robust vector search if available
                if ROBUST_SEARCH_AVAILABLE and RobustVectorSearch:
                    self.robust_search = RobustVectorSearch(self, self.project_name)
                else:
                    logger.warning("RobustVectorSearch not available, using legacy search")

                self.initialized = True

                return {
                    "success": True,
                    "message": "Neo4j service initialized successfully",
                    "uri": self.client.uri,
                }
            else:
                return {
                    "success": False,
                    "message": f"Neo4j connection test failed: {result.get('message', 'Unknown error')}",
                }

        except Exception as e:
            logger.error(f"Neo4j service initialization failed: {e}")
            return {
                "success": False,
                "message": f"Neo4j initialization failed: {str(e)}",
            }

    async def _ensure_constraints(self):
        """Ensure graph constraints with project-based isolation (ADR-0029) and vector indexes (ADR-0066)"""
        try:
            # ADR-0029: Composite constraints for multi-project isolation
            # These constraints ensure uniqueness WITHIN each project
            constraints = [
                # Composite constraints including project property
                "CREATE CONSTRAINT file_project_path_unique IF NOT EXISTS FOR (f:File) REQUIRE (f.project, f.path) IS UNIQUE",
                "CREATE CONSTRAINT class_project_name_unique IF NOT EXISTS FOR (c:Class) REQUIRE (c.project, c.name, c.file_path) IS UNIQUE",
                "CREATE CONSTRAINT method_project_signature_unique IF NOT EXISTS FOR (m:Method) REQUIRE (m.project, m.name, m.class_name, m.file_path) IS UNIQUE",
                # CRITICAL: Index on project property for performance
                "CREATE INDEX project_property_index IF NOT EXISTS FOR (n:File) ON (n.project)",
                "CREATE INDEX project_property_chunk_index IF NOT EXISTS FOR (n:Chunk) ON (n.project)",
                # Additional indexes for query performance
                "CREATE INDEX file_language_index IF NOT EXISTS FOR (f:File) ON (f.language)",
                "CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)",
                "CREATE INDEX method_name_index IF NOT EXISTS FOR (m:Method) ON (m.name)",
            ]

            for constraint in constraints:
                await self.client.execute_query(constraint)

            # ADR-0066: Create vector indexes for elite GraphRAG performance
            await self._ensure_vector_indexes()

            logger.info(
                "Neo4j constraints and vector indexes created for elite GraphRAG (ADR-0029, ADR-0066)"
            )

        except Exception as e:
            logger.warning(f"Could not create constraints: {e}")

    async def _ensure_vector_indexes(self):
        """ADR-0066: Create vector indexes for HNSW-optimized semantic search"""
        try:
            # Check if vector indexes already exist
            check_query = """
            SHOW INDEXES
            YIELD name, type
            WHERE type = 'VECTOR'
            RETURN name
            """

            existing_result = await self.client.execute_query(check_query)
            existing_indexes = []
            if existing_result.get("status") == "success":
                existing_indexes = [
                    record["name"] for record in existing_result["result"]
                ]

            # Vector indexes for elite GraphRAG performance - Neo4j 5.23+ compliant
            # Using minimal required configuration for maximum compatibility
            vector_indexes = [
                {
                    "name": "chunk_embeddings_index",
                    "query": """
                    CREATE VECTOR INDEX chunk_embeddings_index IF NOT EXISTS
                    FOR (c:Chunk) ON c.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine',
                            `vector.hnsw.m`: 24,
                            `vector.hnsw.ef_construction`: 150,
                            `vector.quantization.enabled`: true
                        }
                    }
                    """,
                    "description": "Elite chunk search: ADR-0090 HNSW optimized with M=24, ef=150, int8 quantization for 2x speed",
                },
                {
                    "name": "file_embeddings_index",
                    "query": """
                    CREATE VECTOR INDEX file_embeddings_index IF NOT EXISTS
                    FOR (f:File) ON f.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine',
                            `vector.hnsw.m`: 24,
                            `vector.hnsw.ef_construction`: 150,
                            `vector.quantization.enabled`: true
                        }
                    }
                    """,
                    "description": "File search: ADR-0090 HNSW optimized with M=24, ef=150, int8 quantization for 2x speed",
                },
            ]

            for index_config in vector_indexes:
                index_name = index_config["name"]
                if index_name not in existing_indexes:
                    logger.info(f"Creating vector index: {index_name}")
                    result = await self.client.execute_query(index_config["query"])
                    if result.get("status") == "success":
                        logger.info(
                            f"✅ Vector index {index_name} created successfully"
                        )
                    else:
                        logger.error(
                            f"❌ Failed to create vector index {index_name}: {result.get('message')}"
                        )
                else:
                    logger.info(f"✅ Vector index {index_name} already exists")

            logger.info(
                "Vector indexes ensured for elite GraphRAG performance (ADR-0066)"
            )

        except Exception as e:
            logger.error(f"Failed to create vector indexes: {e}")
            # Don't fail initialization if vector indexes fail - they can be created later

    async def execute_query(
        self, cypher_query: str, parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute query using Neo4j 5.x+ standard naming (ADR-0046). Delegates to execute_cypher."""
        return await self.execute_cypher(cypher_query, parameters)

    async def execute_cypher(
        self, cypher_query: str, parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute Cypher query with service validation, project isolation (ADR-0029) and intelligent caching"""
        if not self.initialized or not self.client:
            return {"status": "error", "message": "Neo4j service not initialized"}

        # ADR-0029: Automatically inject project name for isolation
        # Fix: Properly merge parameters instead of overwriting
        if parameters is None:
            parameters = {}
        # Only add project if not already present (don't overwrite user-provided value)
        if "project" not in parameters:
            parameters["project"] = self.project_name

        # Check cache for read queries (SELECT-like operations)
        is_read_query = self._is_read_only_query(cypher_query)
        cache_key = None

        if is_read_query and self.enable_caching:
            cache_key = self._generate_cache_key(cypher_query, parameters, "cypher")
            cached_result = await self._get_from_cache(cache_key)

            if cached_result:
                # Add cache metadata to result
                cached_result["cache_hit"] = True
                cached_result["cached_at"] = cached_result.get(
                    "cached_at", int(time.time())
                )
                return cached_result

        # Execute query
        result = await self.client.execute_query(cypher_query, parameters)

        # Cache successful read queries
        if is_read_query and self.enable_caching and result.get("status") == "success":
            # Add caching metadata
            result["cache_hit"] = False
            result["cached_at"] = int(time.time())
            await self._store_in_cache(cache_key, result, self.cache_ttl_default)

        return result

    def _is_read_only_query(self, cypher_query: str) -> bool:
        """Check if Cypher query is read-only (cacheable)"""
        # Convert to lowercase for checking
        query_lower = cypher_query.strip().lower()

        # Write operations that should not be cached
        write_keywords = ["create", "merge", "set", "delete", "remove", "drop"]

        # Check if query contains any write keywords (anywhere in the query)
        for keyword in write_keywords:
            if keyword in query_lower:
                return False

        # It's a read query if it doesn't contain write operations
        return True

    async def semantic_search(
        self, query_text: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform semantic search across graph entities with intelligent caching"""
        if not self.initialized:
            return []

        # Check cache for semantic search results (shorter TTL due to content sensitivity)
        search_params = {"query": query_text.lower(), "limit": limit}
        cache_key = self._generate_cache_key(
            f"semantic_search:{query_text}", search_params, "semantic"
        )

        if self.enable_caching:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.debug(f"Semantic search cache hit for: {query_text[:50]}...")
                return cached_result.get("search_results", [])

        # ADR-0087: Fixed semantic search to use correct field names
        # Search Chunks with content, Files by path/name, Functions by name
        cypher = """
        MATCH (c:Chunk)
        WHERE c.content CONTAINS $query
           AND c.project = $project
        RETURN c, 'Chunk' as node_type, c.content as matched_content
        LIMIT $limit

        UNION

        MATCH (f:File)
        WHERE (f.name CONTAINS $query OR f.path CONTAINS $query)
           AND f.project = $project
        RETURN f, 'File' as node_type, f.path as matched_content
        LIMIT $limit

        UNION

        MATCH (func:Function)
        WHERE func.name CONTAINS $query
           AND func.project = $project
        RETURN func, 'Function' as node_type, func.name as matched_content
        LIMIT $limit

        UNION

        MATCH (m:Method)
        WHERE m.name CONTAINS $query
           AND m.project = $project
        RETURN m, 'Method' as node_type, m.name as matched_content
        LIMIT $limit
        """

        result = await self.execute_cypher(cypher, search_params)

        if result["status"] == "success":
            search_results = result["result"]

            # Cache semantic search results with shorter TTL
            if self.enable_caching:
                cache_result = {
                    "search_results": search_results,
                    "query": query_text,
                    "limit": limit,
                    "cached_at": int(time.time()),
                }
                await self._store_in_cache(
                    cache_key, cache_result, self.cache_ttl_semantic
                )

            return search_results
        else:
            logger.error(f"Semantic search failed: {result.get('message')}")
            return []

    async def vector_similarity_search_legacy(
        self,
        query_embedding: List[float],
        node_type: str = "Chunk",
        limit: int = 10,
        min_score: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """ADR-0066: Elite GraphRAG vector similarity search using Neo4j HNSW indexes"""
        if not self.initialized:
            return []

        # Validate inputs
        if not query_embedding or len(query_embedding) != 768:
            logger.error(
                f"Invalid embedding dimensions: expected 768, got {len(query_embedding)}"
            )
            return []

        valid_node_types = ["Chunk", "File"]
        if node_type not in valid_node_types:
            logger.error(
                f"Invalid node type: {node_type}. Must be one of {valid_node_types}"
            )
            return []

        # Cache key for vector search
        embedding_hash = hashlib.sha256(str(query_embedding).encode()).hexdigest()[:16]
        cache_key = self._generate_cache_key(
            f"vector_search:{node_type}:{embedding_hash}",
            {"limit": limit, "min_score": min_score},
            "vector",
        )

        if self.enable_caching:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.debug(f"Vector search cache hit for {node_type}")
                return cached_result.get("search_results", [])

        # Use appropriate vector index based on node type (Neo4j may modify index names)
        index_name = (
            "chunk_embeddings_index"
            if node_type == "Chunk"
            else "file_embeddings_index"
        )

        # ADR-0092 FIX: Use literal index name in CALL statement (Neo4j doesn't support parameterized index names)
        # Confirmed by Grok 4 analysis and Neo4j 2025.08.0 documentation
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', $top_k, $query_vector)
        YIELD node, score
        WHERE score >= $min_score
          AND node.project = $project
        RETURN node, score, labels(node) as node_type
        ORDER BY score DESC
        LIMIT $limit
        """

        search_params = {
            "top_k": limit * 3,  # Get more candidates for filtering
            "query_vector": query_embedding,
            "min_score": min_score,
            "limit": limit,
            "project": self.project_name,
        }

        result = await self.execute_cypher(cypher, search_params)

        # ADR-0092: Debug logging for vector search
        logger.debug(
            f"🔍 Vector search: index={index_name}, project={search_params.get('project')}"
        )
        logger.debug(
            f"🔍 Query params: {len(search_params.get('query_vector', []))} dim vector, min_score={search_params.get('min_score')}"
        )
        if result.get("status") != "success":
            logger.error(
                f"🔍 Vector search error: {result.get('message', 'Unknown error')}"
            )

        if result["status"] == "success":
            search_results = result["result"]

            # Enhance results with metadata
            enhanced_results = []
            for record in search_results:
                node = record["node"]
                score = record["score"]
                enhanced_result = {
                    "node": node,
                    "similarity_score": score,
                    "node_type": record["node_type"],
                    "search_type": "vector_similarity",
                    "index_used": index_name,
                }
                enhanced_results.append(enhanced_result)

            # Cache vector search results
            if self.enable_caching:
                cache_result = {
                    "search_results": enhanced_results,
                    "node_type": node_type,
                    "limit": limit,
                    "min_score": min_score,
                    "cached_at": int(time.time()),
                }
                await self._store_in_cache(
                    cache_key, cache_result, self.cache_ttl_semantic
                )

            logger.info(
                f"✅ Elite vector search completed: {len(enhanced_results)} results for {node_type} (index: {index_name})"
            )
            return enhanced_results
        else:
            logger.error(f"Vector similarity search failed: {result.get('message')}")
            return []

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        limit: int = 10,
        vector_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """ADR-0090 Phase 4: Elite GraphRAG hybrid search with graph fan-out and community context"""
        if not self.initialized:
            return []

        # Validate inputs
        if vector_weight < 0 or vector_weight > 1:
            logger.error(
                f"Invalid vector_weight: {vector_weight}. Must be between 0 and 1"
            )
            return []

        text_weight = 1.0 - vector_weight

        # Get vector similarity results
        vector_results = await self.vector_similarity_search(
            query_embedding, "Chunk", limit * 2
        )

        # Get text search results
        text_results = await self.semantic_search(query_text, limit * 2)

        # Combine and score results
        combined_results = {}

        # Add vector results with vector weight
        for result in vector_results:
            node = result["node"]
            node_id = node.get("chunk_id") or node.get("path")
            if node_id:
                combined_results[node_id] = {
                    "node": node,
                    "vector_score": result["similarity_score"] * vector_weight,
                    "text_score": 0.0,
                    "combined_score": result["similarity_score"] * vector_weight,
                    "node_type": result["node_type"],
                    "search_type": "hybrid",
                }

        # Add text results with text weight
        for result in text_results:
            node = result["n"]
            node_id = node.get("chunk_id") or node.get("path")
            if node_id:
                text_score = 1.0 * text_weight  # Simple text relevance score

                if node_id in combined_results:
                    # Update existing result
                    combined_results[node_id]["text_score"] = text_score
                    combined_results[node_id]["combined_score"] += text_score
                else:
                    # New text-only result
                    combined_results[node_id] = {
                        "node": node,
                        "vector_score": 0.0,
                        "text_score": text_score,
                        "combined_score": text_score,
                        "node_type": result["node_type"],
                        "search_type": "hybrid",
                    }

        # Sort by combined score and return top results
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["combined_score"], reverse=True
        )[:limit]

        logger.info(
            f"✅ Hybrid search completed: {len(sorted_results)} results (vector_weight: {vector_weight})"
        )
        return sorted_results

    async def vector_similarity_search(
        self,
        query_embedding: List[float],
        node_type: str = "Chunk",
        limit: int = 10,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """ADR-0096: Use RobustVectorSearch for all vector operations"""
        if not self.initialized or not self.robust_search:
            logger.warning("RobustVectorSearch not initialized, falling back to legacy")
            return await self.vector_similarity_search_legacy(
                query_embedding, node_type, limit, min_score
            )

        # Use the robust vector search implementation
        results = await self.robust_search.vector_search_phase1(
            query_embedding, limit, min_score
        )

        # Format for compatibility with existing code
        formatted_results = []
        for r in results:
            formatted_results.append({
                "node": {
                    "id": r.get("element_id"),
                    "chunk_id": r.get("chunk_id"),
                    "file_path": r.get("file_path"),
                    "content": r.get("content"),
                    "start_line": r.get("start_line"),
                    "end_line": r.get("end_line"),
                },
                "similarity_score": r.get("score", 0),
                "score": r.get("score", 0),
                "node_type": ["Chunk"],
                "search_type": "vector_similarity",
            })

        return formatted_results

    async def hybrid_search_with_fanout_legacy(
        self,
        query_text: str,
        query_embedding: List[float],
        max_depth: int = 2,
        limit: int = 10,
        vector_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """ADR-0092: Two-phase VectorCypherRetriever pattern for Neo4j-compatible GraphRAG"""
        if not self.initialized:
            return []

        # Phase 1: Simple vector search (using fixed implementation)
        vector_results = await self.vector_similarity_search(
            query_embedding,
            node_type="Chunk",
            limit=limit * 2,  # Get more candidates for graph enrichment
            min_score=0.5,  # Lower threshold for initial candidates
        )

        if not vector_results:
            logger.warning("No vector results found in Phase 1")
            return []

        # Extract chunk IDs and scores for Phase 2
        chunk_data = []
        for vr in vector_results:
            if vr.get("node") and vr["node"].get("id"):
                chunk_data.append({"id": vr["node"]["id"], "score": vr.get("score", 0)})

        if not chunk_data:
            logger.warning("No valid chunk IDs found for Phase 2")
            return []

        # Phase 2: Controlled graph enrichment with bounded traversal
        # ADR-0092: Gracefully handle missing USES/INSTANTIATES relationships
        enrich_cypher = """
        // Start with our vector search results
        UNWIND $chunks as chunk_info
        MATCH (chunk:Chunk {id: chunk_info.id, project: $project})

        // Get the file this chunk belongs to (use HAS_CHUNK which exists)
        OPTIONAL MATCH (chunk)<-[:HAS_CHUNK]-(f:File)

        // Bounded traversal for relationships (separate queries for control)
        WITH chunk, f, chunk_info.score as vector_score

        // Get direct imports (1-hop) - these exist in our graph
        OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File)
        WITH chunk, f, vector_score, collect(DISTINCT imported) as imports

        // Get functions defined in this file
        OPTIONAL MATCH (f)<-[:DEFINED_IN]-(func:Function)
        WITH chunk, f, vector_score, imports, collect(DISTINCT func) as functions

        // Get what functions this file calls (using CALLS which exists)
        OPTIONAL MATCH (f)<-[:DEFINED_IN]-(func:Function)-[:CALLS]->(called:Function)
        WITH chunk, f, vector_score, imports, functions,
             collect(DISTINCT called) as call_chain

        // Try to get USES relationships (may not exist yet)
        OPTIONAL MATCH (f)<-[:DEFINED_IN]-(func:Function)-[:USES]->(var:Variable)
        WITH chunk, f, vector_score, imports, functions, call_chain,
             collect(DISTINCT var) as used_vars

        // Try to get INSTANTIATES relationships (may not exist yet)
        OPTIONAL MATCH (f)<-[:DEFINED_IN]-(func:Function)-[:INSTANTIATES]->(cls:Class)
        WITH chunk, f, vector_score, imports, functions, call_chain, used_vars,
             collect(DISTINCT cls) as instantiated_classes

        // Calculate graph-boosted score (prioritize existing relationships)
        WITH chunk, f, vector_score,
             size(imports) as import_count,
             size(functions) as function_count,
             size(call_chain) as call_depth,
             size(used_vars) as variable_count,
             size(instantiated_classes) as class_count,
             imports, functions[0..5] as sample_functions,
             call_chain[0..5] as sample_calls,
             used_vars[0..5] as sample_vars

        // Combine scores with configurable weighting
        // ADR-0092: Adjust weights to prioritize existing relationships
        WITH chunk, f, vector_score,
             import_count, function_count, call_depth, variable_count, class_count,
             imports, sample_functions, sample_calls, sample_vars,
             ($vector_weight * vector_score +
              (1 - $vector_weight) * (
                  CASE WHEN import_count > 0
                       THEN 0.4 * import_count / 10.0
                       ELSE 0 END +
                  CASE WHEN function_count > 0
                       THEN 0.3 * function_count / 10.0
                       ELSE 0 END +
                  CASE WHEN call_depth > 0
                       THEN 0.3 * call_depth / 10.0
                       ELSE 0 END +
                  CASE WHEN variable_count > 0
                       THEN 0.1 * variable_count / 10.0
                       ELSE 0 END +
                  CASE WHEN class_count > 0
                       THEN 0.1 * class_count / 10.0
                       ELSE 0 END
              )) as final_score

        RETURN chunk, f, vector_score, final_score,
               import_count, function_count, call_depth, variable_count, class_count,
               imports, sample_functions, sample_calls, sample_vars
        ORDER BY final_score DESC
        LIMIT $limit
        """

        result = await self.execute_cypher(
            enrich_cypher,
            {
                "chunks": chunk_data,
                "project": self.project_name,
                "vector_weight": vector_weight,
                "limit": limit,
            },
        )

        # ADR-0092: Log enrichment results
        logger.info(
            f"🔍 Phase 2 enrichment: status={result.get('status')}, records={len(result.get('result', []))}"
        )

        if result["status"] == "success":
            enhanced_results = []
            for record in result["result"]:
                enhanced_result = {
                    "chunk": record.get("chunk"),
                    "file": record.get("f"),
                    "vector_score": record.get("vector_score", 0),
                    "final_score": record.get("final_score", 0),
                    "graph_context": {
                        "import_count": record.get("import_count", 0),
                        "function_count": record.get("function_count", 0),
                        "call_depth": record.get("call_depth", 0),
                        "variable_count": record.get("variable_count", 0),
                        "class_count": record.get("class_count", 0),
                        "imports": record.get("imports", []),
                        "functions": record.get("sample_functions", []),
                        "calls": record.get("sample_calls", []),
                        "variables": record.get("sample_vars", []),
                    },
                }
                enhanced_results.append(enhanced_result)

            logger.info(
                f"✅ VectorCypherRetriever complete: {len(enhanced_results)} enriched results"
            )
            return enhanced_results
        else:
            # Fall back to pure vector results if enrichment fails
            logger.warning(
                f"Graph enrichment failed: {result.get('message')}, returning vector results only"
            )
            return [
                {
                    "chunk": vr.get("node"),
                    "vector_score": vr.get("score", 0),
                    "final_score": vr.get("score", 0),
                    "graph_context": {},
                }
                for vr in vector_results[:limit]
            ]

    async def hybrid_search_with_fanout(
        self,
        query_text: str,
        query_embedding: List[float],
        max_depth: int = 2,
        limit: int = 10,
        vector_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """ADR-0096: Use RobustVectorSearch for hybrid search"""
        if not self.initialized or not self.robust_search:
            logger.warning("RobustVectorSearch not initialized, falling back to legacy")
            return await self.hybrid_search_with_fanout_legacy(
                query_text, query_embedding, max_depth, limit, vector_weight
            )

        # Use the robust hybrid search implementation
        results = await self.robust_search.hybrid_search(
            query_embedding, limit, min_score=0.5, enrich_context=(max_depth > 0)
        )

        # Format for compatibility with existing code
        formatted_results = []
        for r in results:
            formatted_results.append({
                "chunk": {
                    "chunk_id": r.get("chunk_id"),
                    "file_path": r.get("file_path"),
                    "content": r.get("content"),
                    "start_line": r.get("start_line"),
                    "end_line": r.get("end_line"),
                },
                "vector_score": r.get("score", 0),
                "final_score": r.get("score", 0),  # Could be enhanced with graph context
                "graph_context": r.get("graph_context", {}),
            })

        return formatted_results

    async def get_code_dependencies(
        self, file_path: str, max_depth: int = 3
    ) -> Dict[str, Any]:
        """Get code dependencies for a file with traversal depth"""
        if not self.initialized:
            return {"dependencies": [], "dependents": []}

        # Get dependencies (what this file depends on)
        deps_cypher = """
        MATCH (f:File {path: $file_path})-[:IMPORTS|CALLS*1..$max_depth]->(dep)
        WHERE dep:File OR dep:Class OR dep:Method
        RETURN DISTINCT dep, labels(dep) as type
        """

        # Get dependents (what depends on this file)
        dependents_cypher = """
        MATCH (dep)-[:IMPORTS|CALLS*1..$max_depth]->(f:File {path: $file_path})
        WHERE dep:File OR dep:Class OR dep:Method  
        RETURN DISTINCT dep, labels(dep) as type
        """

        deps_result = await self.execute_cypher(
            deps_cypher, {"file_path": file_path, "max_depth": max_depth}
        )

        dependents_result = await self.execute_cypher(
            dependents_cypher, {"file_path": file_path, "max_depth": max_depth}
        )

        return {
            "dependencies": (
                deps_result.get("result", [])
                if deps_result["status"] == "success"
                else []
            ),
            "dependents": (
                dependents_result.get("result", [])
                if dependents_result["status"] == "success"
                else []
            ),
            "file_path": file_path,
            "max_depth": max_depth,
        }

    async def index_code_file(
        self, file_path: str, content: str, language: str = "python"
    ) -> Dict[str, Any]:
        """Index a code file with basic structure extraction"""
        if not self.initialized:
            return {"status": "error", "message": "Service not initialized"}

        try:
            # Create or update file node
            cypher = """
            MERGE (f:File {path: $file_path})
            SET f.content = $content,
                f.language = $language,
                f.size = $size,
                f.updated = datetime()
            RETURN f
            """

            result = await self.execute_cypher(
                cypher,
                {
                    "file_path": file_path,
                    "content": content[:5000],  # Limit content size
                    "language": language,
                    "size": len(content),
                },
            )

            if result["status"] == "success":
                return {"status": "success", "file_path": file_path, "indexed": True}
            else:
                return result

        except Exception as e:
            logger.error(f"Code file indexing failed for {file_path}: {e}")
            return {"status": "error", "message": f"Indexing failed: {str(e)}"}

    async def health_check(self) -> Dict[str, Any]:
        """Check Neo4j service health"""
        try:
            if not self.available:
                return {"healthy": False, "message": "Neo4j driver not available"}

            if not self.initialized:
                return {"healthy": False, "message": "Service not initialized"}

            # Simple connectivity test
            result = await self.execute_cypher("RETURN 1 as health_check")
            return {
                "healthy": result["status"] == "success",
                "message": (
                    "Neo4j GraphRAG operational"
                    if result["status"] == "success"
                    else result.get("message")
                ),
                "uri": self.client.uri if self.client else "unknown",
            }

        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_migration_status(self) -> Dict[str, Any]:
        """Get Neo4j migration and indexing status"""
        if not self.initialized:
            return {"status": "not_initialized", "nodes": 0, "relationships": 0}

        try:
            # Count nodes and relationships
            stats_cypher = """
            CALL db.stats.retrieve('GRAPH') 
            YIELD data
            RETURN data.nodes as node_count, data.relationships as rel_count
            """

            # Try alternative if APOC not available
            fallback_cypher = """
            MATCH (n) 
            RETURN count(n) as node_count, 
                   size(()-[]->()) as rel_count
            """

            result = await self.execute_cypher(stats_cypher)
            if result["status"] != "success":
                result = await self.execute_cypher(fallback_cypher)

            if result["status"] == "success" and result["result"]:
                stats = result["result"][0]
                return {
                    "status": "active",
                    "nodes": stats.get("node_count", 0),
                    "relationships": stats.get("rel_count", 0),
                    "migration_complete": True,
                }
            else:
                return {
                    "status": "error",
                    "message": result.get("message", "Stats query failed"),
                    "nodes": 0,
                    "relationships": 0,
                }

        except Exception as e:
            logger.error(f"Migration status check failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "nodes": 0,
                "relationships": 0,
            }
