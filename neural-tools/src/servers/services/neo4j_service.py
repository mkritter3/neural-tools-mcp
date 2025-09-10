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
import asyncio

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
        neo4j_uri = os.environ.get('NEO4J_URI')
        neo4j_username = os.environ.get('NEO4J_USERNAME', 'neo4j')
        neo4j_password = os.environ.get('NEO4J_PASSWORD', 'neural-l9-2025')
        
        if neo4j_uri:
            # Use NEO4J_URI when provided (MCP config case)
            self.uri = neo4j_uri
            logger.info(f"Using NEO4J_URI: {self.uri}")
        else:
            # Fall back to NEO4J_HOST/NEO4J_PORT construction
            neo4j_host = os.environ.get('NEO4J_HOST', 'default-neo4j-graph')
            neo4j_port = int(os.environ.get('NEO4J_PORT', 7687))
            self.uri = f"bolt://{neo4j_host}:{neo4j_port}"
            logger.info(f"Constructed URI from NEO4J_HOST/PORT: {self.uri}")
        self.driver = AsyncGraphDatabase.driver(
            self.uri, 
            auth=(neo4j_username, neo4j_password)
        )
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def close(self):
        """Close the driver"""
        if self.driver:
            await self.driver.close()
    
    async def execute_query(self, cypher_query: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
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
                                if hasattr(value, '__dict__'):
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
                                "counters": summary.counters.__dict__ if hasattr(summary.counters, '__dict__') else {},
                                "result_available_after": summary.result_available_after,
                                "result_consumed_after": summary.result_consumed_after
                            }
                        }
                        
                    except Exception as tx_error:
                        raise tx_error
                
                # Execute with managed transaction for retry capability
                # Check if query contains write operations (anywhere in the query)
                query_upper = cypher_query.strip().upper()
                write_keywords = ('CREATE', 'MERGE', 'SET', 'DELETE', 'REMOVE', 'DROP')
                is_write_query = any(keyword in query_upper for keyword in write_keywords)
                
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
                    "parameters": parameters
                }
                
        except AuthError as auth_error:
            return {
                "status": "error",
                "message": f"Neo4j authentication failed: {str(auth_error)}",
                "error_type": "authentication_error"
            }
        except ServiceUnavailable as service_error:
            return {
                "status": "error", 
                "message": f"Neo4j service unavailable: {str(service_error)}",
                "error_type": "service_unavailable"
            }
        except Exception as e:
            logger.error(f"neo4j_graph_query error: {str(e)}")
            return {
                "status": "error",
                "message": f"Query execution failed: {str(e)}",
                "error_type": "execution_error"
            }

class Neo4jService:
    """Service class for Neo4j GraphRAG operations with intelligent caching"""
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.client = None
        self.initialized = False
        self.available = NEO4J_AVAILABLE
        self.service_container = None  # Will be set by service container for cache access
        
        # Cache configuration
        self.enable_caching = os.getenv('NEO4J_ENABLE_CACHING', 'true').lower() == 'true'
        self.cache_ttl_default = int(os.getenv('NEO4J_CACHE_TTL', 3600))  # 1 hour default
        self.cache_ttl_semantic = int(os.getenv('NEO4J_SEMANTIC_CACHE_TTL', 1800))  # 30 min for semantic queries
    
    def set_service_container(self, container):
        """Set reference to service container for cache access"""
        self.service_container = container
    
    def _generate_cache_key(self, query: str, parameters: Optional[Dict] = None, key_type: str = "query") -> str:
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
    
    async def _store_in_cache(self, cache_key: str, result: Dict[str, Any], ttl: Optional[int] = None):
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
                "message": "Neo4j driver not available - GraphRAG features disabled"
            }
            
        try:
            self.client = AsyncNeo4jClient(self.project_name)
            
            # Test connectivity with simple query
            result = await self.client.execute_query("RETURN 1 as test")
            
            if result["status"] == "success" and result["result"]:
                # Ensure basic constraints exist
                await self._ensure_constraints()
                self.initialized = True
                
                return {
                    "success": True,
                    "message": "Neo4j service initialized successfully",
                    "uri": self.client.uri
                }
            else:
                return {
                    "success": False,
                    "message": f"Neo4j connection test failed: {result.get('message', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"Neo4j service initialization failed: {e}")
            return {
                "success": False,
                "message": f"Neo4j initialization failed: {str(e)}"
            }
    
    async def _ensure_constraints(self):
        """Ensure basic graph constraints exist for code relationships"""
        try:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE (c.name, c.file_path) IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE (m.name, m.class_name, m.file_path) IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.language)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Class) ON (c.name)",
                "CREATE INDEX IF NOT EXISTS FOR (m:Method) ON (m.name)"
            ]
            
            for constraint in constraints:
                await self.client.execute_query(constraint)
                
        except Exception as e:
            logger.warning(f"Could not create constraints: {e}")
    
    async def execute_cypher(self, cypher_query: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute Cypher query with service validation and intelligent caching"""
        if not self.initialized or not self.client:
            return {
                "status": "error",
                "message": "Neo4j service not initialized"
            }
        
        # Check cache for read queries (SELECT-like operations)
        is_read_query = self._is_read_only_query(cypher_query)
        cache_key = None
        
        if is_read_query and self.enable_caching:
            cache_key = self._generate_cache_key(cypher_query, parameters, "cypher")
            cached_result = await self._get_from_cache(cache_key)
            
            if cached_result:
                # Add cache metadata to result
                cached_result["cache_hit"] = True
                cached_result["cached_at"] = cached_result.get("cached_at", int(time.time()))
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
        write_keywords = ['create', 'merge', 'set', 'delete', 'remove', 'drop']
        
        # Check if query contains any write keywords (anywhere in the query)
        for keyword in write_keywords:
            if keyword in query_lower:
                return False
        
        # It's a read query if it doesn't contain write operations
        return True
    
    async def semantic_search(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search across graph entities with intelligent caching"""
        if not self.initialized:
            return []
        
        # Check cache for semantic search results (shorter TTL due to content sensitivity)
        search_params = {"query": query_text.lower(), "limit": limit}
        cache_key = self._generate_cache_key(f"semantic_search:{query_text}", search_params, "semantic")
        
        if self.enable_caching:
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.debug(f"Semantic search cache hit for: {query_text[:50]}...")
                return cached_result.get("search_results", [])
        
        # Search across files, classes, and methods for relevant text
        cypher = """
        MATCH (n) 
        WHERE (n:File OR n:Class OR n:Method)
          AND (n.content CONTAINS $query 
               OR n.name CONTAINS $query 
               OR n.description CONTAINS $query)
        RETURN n, labels(n) as node_type
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
                    "cached_at": int(time.time())
                }
                await self._store_in_cache(cache_key, cache_result, self.cache_ttl_semantic)
            
            return search_results
        else:
            logger.error(f"Semantic search failed: {result.get('message')}")
            return []
    
    async def get_code_dependencies(self, file_path: str, max_depth: int = 3) -> Dict[str, Any]:
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
        
        deps_result = await self.execute_cypher(deps_cypher, {
            "file_path": file_path,
            "max_depth": max_depth
        })
        
        dependents_result = await self.execute_cypher(dependents_cypher, {
            "file_path": file_path,
            "max_depth": max_depth
        })
        
        return {
            "dependencies": deps_result.get("result", []) if deps_result["status"] == "success" else [],
            "dependents": dependents_result.get("result", []) if dependents_result["status"] == "success" else [],
            "file_path": file_path,
            "max_depth": max_depth
        }
    
    async def index_code_file(self, file_path: str, content: str, language: str = "python") -> Dict[str, Any]:
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
            
            result = await self.execute_cypher(cypher, {
                "file_path": file_path,
                "content": content[:5000],  # Limit content size
                "language": language,
                "size": len(content)
            })
            
            if result["status"] == "success":
                return {
                    "status": "success",
                    "file_path": file_path,
                    "indexed": True
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Code file indexing failed for {file_path}: {e}")
            return {
                "status": "error",
                "message": f"Indexing failed: {str(e)}"
            }
    
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
                "message": "Neo4j GraphRAG operational" if result["status"] == "success" else result.get("message"),
                "uri": self.client.uri if self.client else "unknown"
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
                    "migration_complete": True
                }
            else:
                return {
                    "status": "error",
                    "message": result.get("message", "Stats query failed"),
                    "nodes": 0,
                    "relationships": 0
                }
                
        except Exception as e:
            logger.error(f"Migration status check failed: {e}")
            return {
                "status": "error", 
                "message": str(e),
                "nodes": 0,
                "relationships": 0
            }