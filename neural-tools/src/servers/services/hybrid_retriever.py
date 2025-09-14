#!/usr/bin/env python3
"""
GraphRAG Hybrid Retriever for L9 Neural Tools
Combines Qdrant semantic search with Neo4j graph traversal
Enables powerful hybrid queries across both databases
Enhanced with caching layer for improved performance
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Implements GraphRAG patterns for hybrid search
    Combines vector similarity (Qdrant) with graph relationships (Neo4j)
    """
    
    def __init__(self, container):
        """
        Initialize with service container

        Args:
            container: ServiceContainer with Neo4j, Qdrant, Nomic services
        """
        self.container = container
        # Use centralized collection naming - no _code suffix per ADR-0041
        from servers.config.collection_naming import collection_naming
        self.collection_name = collection_naming.get_collection_name(container.project_name)
        
    async def unified_search(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        ADR-0047: Unified search interface with automatic service fallback

        Single entry point for all search operations. Automatically handles:
        - BM25S + dense hybrid search when available
        - Falls back to RRF combination if BM25S unavailable
        - Falls back to pure vector search if text search fails
        - Falls back to pure graph search if vector search fails
        - Gracefully handles service failures

        Args:
            query: Search query (natural language or code)
            limit: Number of results to return
            search_type: "auto" (default), "vector", "graph", "hybrid"

        Returns:
            Unified search results regardless of backend availability
        """
        results = []
        search_methods_tried = []

        # Step 1: Try to get embeddings for vector search
        query_vector = None
        try:
            embeddings = await self.container.nomic.get_embeddings([query])
            if embeddings:
                query_vector = embeddings[0]
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")

        # Step 2: Try hybrid search (best option)
        if query_vector and search_type in ["auto", "hybrid"]:
            try:
                search_methods_tried.append("hybrid")

                # Use enhanced RRF search with BM25S support
                hybrid_results = await self.container.qdrant.rrf_hybrid_search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_text=query,
                    limit=limit,
                    k=60  # Standard RRF parameter
                )

                if hybrid_results:
                    results = hybrid_results
                    logger.info(f"Unified search succeeded with hybrid method")

            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")

        # Step 3: Fallback to pure vector search
        if not results and query_vector and search_type in ["auto", "vector"]:
            try:
                search_methods_tried.append("vector")

                vector_results = await self.container.qdrant.search_vectors(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=0.3
                )

                if vector_results:
                    results = vector_results
                    logger.info(f"Unified search succeeded with vector-only method")

            except Exception as e:
                logger.warning(f"Vector search failed: {e}")

        # Step 4: Fallback to graph search
        if not results and search_type in ["auto", "graph"]:
            try:
                search_methods_tried.append("graph")

                # Use Neo4j full-text search
                graph_results = await self._graph_text_search(query, limit)

                if graph_results:
                    results = graph_results
                    logger.info(f"Unified search succeeded with graph-only method")

            except Exception as e:
                logger.warning(f"Graph search failed: {e}")

        # Step 5: Last resort - keyword matching in cached data
        if not results:
            logger.warning(f"All search methods failed. Tried: {search_methods_tried}")
            results = await self._emergency_keyword_search(query, limit)

        # Add metadata about search method used
        for result in results:
            result['search_methods'] = search_methods_tried
            result['search_type'] = search_methods_tried[0] if search_methods_tried else "emergency"

        return results

    async def _graph_text_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform text search directly in Neo4j"""
        try:
            # Simple text matching query
            cypher = """
            MATCH (c:CodeChunk)
            WHERE c.project = $project
            AND (c.content CONTAINS $query OR c.file_path CONTAINS $query)
            RETURN c.chunk_id as chunk_id, c.file_path as file_path,
                   c.content as content, c.start_line as start_line,
                   c.end_line as end_line
            LIMIT $limit
            """

            result = await self.container.neo4j.execute_query(
                cypher,
                project=self.container.project_name,
                query=query,
                limit=limit
            )

            results = []
            for record in result.get('records', []):
                results.append({
                    'chunk_id': record.get('chunk_id'),
                    'file_path': record.get('file_path'),
                    'content': record.get('content'),
                    'start_line': record.get('start_line'),
                    'end_line': record.get('end_line'),
                    'score': 0.5  # Default score for text matches
                })

            return results

        except Exception as e:
            logger.error(f"Graph text search failed: {e}")
            return []

    async def _emergency_keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Emergency fallback using any available data source"""
        # This could search in local cache, file system, or return empty
        logger.warning("Using emergency keyword search - no backend services available")
        return []

    async def find_similar_with_context(
        self,
        query: str,
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        ADR-0047: Enhanced with unified search backend

        Find semantically similar code with graph context.
        Now uses unified search for better resilience.

        Args:
            query: Natural language query or code snippet
            limit: Number of similar chunks to find
            include_graph_context: Whether to fetch graph relationships
            max_hops: Maximum graph traversal depth

        Returns:
            List of results with code chunks and their relationships
        """
        try:
            # ADR-0047: Use unified search for better resilience
            search_results = await self.unified_search(
                query=query,
                limit=limit,
                search_type="auto"  # Let it use the best available method
            )

            if not search_results:
                return []
            
            # Extract chunk IDs and initial results
            results = []
            chunk_ids = []
            
            for hit in search_results:
                if isinstance(hit, dict):
                    _payload = hit.get('payload', {})
                    _score = hit.get('score', 0.0)
                else:
                    _payload = getattr(hit, 'payload', {}) or {}
                    _score = float(getattr(hit, 'score', 0.0))
                result = {
                    'score': float(_score),
                    'chunk_id': _payload.get('chunk_hash'),
                    'neo4j_chunk_id': _payload.get('neo4j_chunk_id') or _payload.get('chunk_hash'),  # Use chunk_hash as fallback
                    'file_path': _payload.get('file_path'),
                    'content': _payload.get('content'),
                    'start_line': _payload.get('start_line'),
                    'end_line': _payload.get('end_line'),
                    'file_type': _payload.get('file_type')
                }
                results.append(result)

                # Use chunk_hash as fallback if neo4j_chunk_id not present
                chunk_id_for_neo4j = _payload.get('neo4j_chunk_id') or _payload.get('chunk_hash')
                if chunk_id_for_neo4j:
                    chunk_ids.append(chunk_id_for_neo4j)
                    logger.debug(f"Added chunk_id for Neo4j lookup: {chunk_id_for_neo4j}")
            
            # Step 3: Enrich with graph context from Neo4j
            if include_graph_context and chunk_ids and self.container.neo4j:
                logger.info(f"Fetching graph context for {len(chunk_ids)} chunks")
                graph_context = await self._fetch_graph_context(chunk_ids, max_hops)
                logger.info(f"Got {len(graph_context)} graph contexts")

                # Merge graph context with results and boost scores (ADR 0017)
                for i, (result, context) in enumerate(zip(results, graph_context)):
                    result['graph_context'] = context
                    if context:
                        logger.debug(f"Added graph context to result {i}: {list(context.keys())}")
                    # Boost score based on graph importance
                    result['original_score'] = result['score']
                    result['score'] = self._adjust_score_by_graph_importance(
                        result['score'], 
                        context
                    )
            
            # Re-sort by adjusted scores
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _fetch_graph_context(
        self, 
        chunk_ids: List[str], 
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Fetch graph relationships for given chunk IDs
        
        Args:
            chunk_ids: List of Neo4j chunk IDs
            max_hops: Maximum traversal depth
            
        Returns:
            List of graph contexts for each chunk
        """
        contexts = []
        
        for chunk_id in chunk_ids:
            try:
                # Enhanced query for ADR 0017: GraphRAG True Hybrid Search
                # ADR-0029: All queries must filter by project for isolation
                cypher = """
                MATCH (c:CodeChunk {id: $chunk_id, project: $project})
                OPTIONAL MATCH (c)-[:PART_OF]->(f:File {project: $project})
                
                // Get functions in this file/chunk
                OPTIONAL MATCH (func:Function {project: $project})-[:DEFINED_IN]->(f)
                WHERE func.start_line >= c.start_line AND func.end_line <= c.end_line
                
                // Get classes in this file/chunk  
                OPTIONAL MATCH (cls:Class {project: $project})-[:DEFINED_IN]->(f)
                WHERE cls.start_line >= c.start_line AND cls.end_line <= c.end_line
                
                // Get functions this chunk's functions call
                OPTIONAL MATCH (func)-[:CALLS]->(called:Function {project: $project})
                
                // Get functions that call this chunk's functions
                OPTIONAL MATCH (caller:Function {project: $project})-[:CALLS]->(func)
                
                // Get file-level imports
                OPTIONAL MATCH (f)-[:IMPORTS]->(imported:File {project: $project})
                OPTIONAL MATCH (importer:File {project: $project})-[:IMPORTS]->(f)
                
                // Get related chunks in same file
                OPTIONAL MATCH (related:CodeChunk {project: $project})-[:PART_OF]->(f)
                WHERE related.id <> c.id
                
                RETURN 
                    f.path AS file_path,
                    f.type AS file_type,
                    collect(DISTINCT func.name) AS functions,
                    collect(DISTINCT func.signature) AS function_signatures,
                    collect(DISTINCT cls.name) AS classes,
                    collect(DISTINCT called.name) AS calls,
                    collect(DISTINCT caller.name) AS called_by,
                    collect(DISTINCT imported.path) AS imports,
                    collect(DISTINCT importer.path) AS imported_by,
                    collect(DISTINCT {
                        id: related.id,
                        start_line: related.start_line,
                        end_line: related.end_line
                    })[0..5] AS related_chunks
                LIMIT 1
                """
                
                # ADR-0029: Explicitly pass both chunk_id and project parameters
                # Fix: neo4j_service was overwriting params dict, need to pass project explicitly
                logger.debug(f"Executing graph context query for chunk: {chunk_id}, project: {self.container.project_name}")
                result = await self.container.neo4j.execute_cypher(cypher, {
                    'chunk_id': chunk_id,
                    'project': self.container.project_name  # Explicitly pass project to prevent overwrite
                })
                logger.debug(f"Graph context query result type: {type(result)}, value: {result if result != 0 else 'zero'}")

                # Enhanced error handling - check for error status or numeric error code
                if result is None or result == 0:
                    logger.error(f"Neo4j returned invalid result ({result}) for chunk {chunk_id}")
                    logger.debug(f"Query was: {cypher[:200]}...")  # Log first 200 chars of query
                    logger.debug(f"Project: {self.container.project_name}, Chunk ID: {chunk_id}")
                    contexts.append({})
                    continue

                # Check for explicit error status
                if isinstance(result, dict) and result.get('status') == 'error':
                    logger.error(f"Neo4j query failed: {result.get('message', 'Unknown error')}")
                    contexts.append({})
                    continue

                # Process successful result
                context = {}
                if result and isinstance(result, dict) and result.get('status') == 'success':
                    records = result.get('result', [])
                    if records:
                        context = records[0]  # Get first record
                        logger.debug(f"Got context from success result: {list(context.keys()) if context else 'empty'}")
                elif result and isinstance(result, list) and len(result) > 0:
                    context = result[0]  # Direct list result
                    logger.debug(f"Got context from list result: {list(context.keys()) if context else 'empty'}")
                else:
                    logger.warning(f"Unexpected result format from Neo4j: {type(result)}, value: {result}")
                    contexts.append({})
                    continue

                # If we have a context, clean it up and enhance it
                if context:
                    # Clean up empty collections and build enhanced context (ADR 0017)
                    enhanced_context = {
                        'file_path': context.get('file_path'),
                        'functions': [f for f in context.get('functions', []) if f],
                        'function_signatures': [s for s in context.get('function_signatures', []) if s],
                        'classes': [c for c in context.get('classes', []) if c],
                        'calls': [c for c in context.get('calls', []) if c],
                        'called_by': [c for c in context.get('called_by', []) if c],
                        'imports': [i for i in context.get('imports', []) if i],
                        'imported_by': [i for i in context.get('imported_by', []) if i],
                        'related_chunks': [
                            c for c in context.get('related_chunks', [])
                            if c and c.get('id')
                        ]
                    }
                    contexts.append(enhanced_context)
                else:
                    contexts.append({})
                    
            except Exception as e:
                logger.error(f"Failed to fetch graph context for chunk {chunk_id}: {e}")
                contexts.append({})
        
        return contexts
    
    def _adjust_score_by_graph_importance(self, base_score: float, context: Dict) -> float:
        """
        Boost score based on graph relationships (ADR 0017)
        
        Args:
            base_score: Original similarity score from vector search
            context: Graph context with relationships
            
        Returns:
            Adjusted score incorporating graph importance
        """
        boost = 0.0
        
        # Boost if file is heavily imported (high coupling)
        imported_by_count = len(context.get('imported_by', []))
        if imported_by_count > 10:
            boost += 0.15
        elif imported_by_count > 5:
            boost += 0.08
        elif imported_by_count > 2:
            boost += 0.04
        
        # Boost if contains many functions (functional complexity)
        function_count = len(context.get('functions', []))
        if function_count > 5:
            boost += 0.08
        elif function_count > 2:
            boost += 0.04
        
        # Boost if highly connected (many function calls)
        calls_count = len(context.get('calls', [])) + len(context.get('called_by', []))
        if calls_count > 15:
            boost += 0.12
        elif calls_count > 8:
            boost += 0.06
        elif calls_count > 3:
            boost += 0.03
        
        # Boost if contains classes (OOP importance)
        class_count = len(context.get('classes', []))
        if class_count > 0:
            boost += 0.05 * min(class_count, 3)  # Cap at 0.15
        
        # Apply boost with diminishing returns
        adjusted_score = base_score * (1 + boost)
        
        # Cap at 1.0 but preserve relative ordering
        return min(adjusted_score, 0.99)
    
    async def find_related_by_graph(
        self,
        file_path: str,
        relationship_types: List[str] = ['IMPORTS', 'CALLS', 'PART_OF'],
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Find related code by graph traversal
        
        Args:
            file_path: Starting file path
            relationship_types: Types of relationships to follow
            max_depth: Maximum traversal depth
            
        Returns:
            Graph of related files and their relationships
        """
        try:
            rel_pattern = '|'.join(relationship_types)
            
            # ADR-0029: Add project filter for multi-project isolation
            cypher = f"""
            MATCH (start:File {{path: $file_path, project: $project}})
            CALL apoc.path.subgraphAll(start, {{
                relationshipFilter: '{rel_pattern}',
                maxLevel: $max_depth
            }})
            YIELD nodes, relationships
            RETURN nodes, relationships
            """
            
            # Note: This requires APOC plugin in Neo4j
            # Fallback query without APOC:
            # ADR-0029: Add project filter for multi-project isolation
            fallback_cypher = """
            MATCH path = (start:File {path: $file_path, project: $project})-[*1..3]-(related)
            WHERE ALL(r IN relationships(path) WHERE type(r) IN $rel_types)
                  AND ALL(n IN nodes(path) WHERE n.project = $project)
            RETURN collect(DISTINCT nodes(path)) AS nodes,
                   collect(DISTINCT relationships(path)) AS relationships
            """
            
            try:
                # ADR-0029: project parameter will be auto-injected by neo4j_service
                result = await self.container.neo4j.execute_cypher(cypher, {
                    'file_path': file_path,
                    'max_depth': max_depth
                })
            except:
                # Fallback if APOC not available
                # ADR-0029: project parameter will be auto-injected by neo4j_service
                result = await self.container.neo4j.execute_cypher(fallback_cypher, {
                    'file_path': file_path,
                    'rel_types': relationship_types
                })
            
            if result:
                return {
                    'nodes': result[0].get('nodes', []),
                    'relationships': result[0].get('relationships', [])
                }
            
            return {'nodes': [], 'relationships': []}
            
        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return {'nodes': [], 'relationships': []}
    
    async def find_dependencies(
        self,
        file_path: str,
        direction: str = 'both'  # 'imports', 'imported_by', or 'both'
    ) -> Dict[str, List[str]]:
        """
        Find file dependencies through import relationships
        
        Args:
            file_path: File to analyze
            direction: Direction of dependencies to find
            
        Returns:
            Dictionary with 'imports' and/or 'imported_by' lists
        """
        try:
            result = {}
            
            if direction in ['imports', 'both']:
                # ADR-0029: Add project filter for multi-project isolation
                cypher_imports = """
                MATCH (f:File {path: $file_path, project: $project})-[:IMPORTS]->(m:Module)
                OPTIONAL MATCH (m)<-[:PROVIDES]-(provider:File {project: $project})
                RETURN collect(DISTINCT COALESCE(provider.path, m.name)) AS imports
                """
                # ADR-0029: project parameter will be auto-injected by neo4j_service
                imports_result = await self.container.neo4j.execute_cypher(
                    cypher_imports, {'file_path': file_path}
                )
                if imports_result:
                    result['imports'] = imports_result[0].get('imports', [])
            
            if direction in ['imported_by', 'both']:
                # ADR-0029: Add project filter for multi-project isolation
                cypher_imported = """
                MATCH (f:File {path: $file_path, project: $project})<-[:IMPORTS]-(importer:File {project: $project})
                RETURN collect(DISTINCT importer.path) AS imported_by
                """
                # ADR-0029: project parameter will be auto-injected by neo4j_service
                imported_result = await self.container.neo4j.execute_cypher(
                    cypher_imported, {'file_path': file_path}
                )
                if imported_result:
                    result['imported_by'] = imported_result[0].get('imported_by', [])
            
            return result
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {}
    
    async def find_similar_functions(
        self,
        function_signature: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find functions similar to a given signature
        Combines semantic similarity with structural analysis
        
        Args:
            function_signature: Function signature or description
            limit: Number of results
            
        Returns:
            List of similar functions with context
        """
        # First, find similar chunks
        results = await self.find_similar_with_context(
            f"function: {function_signature}",
            limit=limit * 2  # Get more to filter
        )
        
        # Filter for function chunks
        function_results = []
        for result in results:
            if result.get('graph_context'):
                # Check if this chunk contains a function
                related_chunks = result['graph_context'].get('related_chunks', [])
                for chunk in related_chunks:
                    if chunk.get('type') == 'function':
                        function_results.append({
                            **result,
                            'function_info': chunk
                        })
                        break
        
        return function_results[:limit]
    
    async def analyze_impact(
        self,
        file_path: str,
        change_type: str = 'modify'  # 'modify', 'delete', 'refactor'
    ) -> Dict[str, Any]:
        """
        Analyze the impact of changing a file
        Uses graph traversal to find affected components
        
        Args:
            file_path: File being changed
            change_type: Type of change
            
        Returns:
            Impact analysis with affected files and risk assessment
        """
        try:
            # Find direct dependencies
            deps = await self.find_dependencies(file_path, direction='imported_by')
            directly_affected = deps.get('imported_by', [])
            
            # Find transitive dependencies (2-3 hops)
            # ADR-0029: Add project filter for multi-project isolation
            cypher = """
            MATCH (f:File {path: $file_path, project: $project})
            OPTIONAL MATCH (f)<-[:IMPORTS*1..3]-(affected:File {project: $project})
            WITH collect(DISTINCT affected.path) AS transitive
            OPTIONAL MATCH (f)-[:CONTAINS]->(chunk:CodeChunk {project: $project})
            WHERE chunk.type IN ['class', 'function']
            RETURN transitive, count(DISTINCT chunk) AS api_surface
            """
            
            # ADR-0029: project parameter will be auto-injected by neo4j_service
            result = await self.container.neo4j.execute_cypher(
                cypher, {'file_path': file_path}
            )
            
            if result:
                transitive_affected = result[0].get('transitive', [])
                api_surface = result[0].get('api_surface', 0)
            else:
                transitive_affected = []
                api_surface = 0
            
            # Calculate risk level
            total_affected = len(directly_affected) + len(transitive_affected)
            if total_affected == 0:
                risk_level = 'low'
            elif total_affected < 5:
                risk_level = 'medium'
            elif total_affected < 20:
                risk_level = 'high'
            else:
                risk_level = 'critical'
            
            return {
                'file_path': file_path,
                'change_type': change_type,
                'directly_affected': directly_affected,
                'transitive_affected': transitive_affected,
                'total_affected': total_affected,
                'api_surface': api_surface,
                'risk_level': risk_level,
                'recommendations': self._get_impact_recommendations(
                    risk_level, change_type, total_affected
                )
            }
            
        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
            return {
                'error': str(e),
                'risk_level': 'unknown'
            }
    
    def _get_impact_recommendations(
        self,
        risk_level: str,
        change_type: str,
        affected_count: int
    ) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []
        
        if risk_level in ['high', 'critical']:
            recommendations.append(f"⚠️ High impact: {affected_count} files affected")
            recommendations.append("Consider creating a feature branch")
            recommendations.append("Add comprehensive tests before making changes")
            
            if change_type == 'delete':
                recommendations.append("Search for dynamic imports that might not be detected")
            elif change_type == 'refactor':
                recommendations.append("Consider doing the refactor in smaller increments")
        
        if risk_level == 'critical':
            recommendations.append("🚨 Critical impact - review with team before proceeding")
            recommendations.append("Consider creating a migration guide")
        
        if affected_count > 0:
            recommendations.append(f"Run tests for all {affected_count} affected files")
        
        return recommendations

class CachedHybridRetriever:
    """
    Cached wrapper around HybridRetriever for improved performance
    Implements transparent caching with configurable TTL
    """
    
    def __init__(self, hybrid_retriever: HybridRetriever, cache_ttl: int = 300):
        """
        Initialize cached retriever
        
        Args:
            hybrid_retriever: Instance of HybridRetriever to wrap
            cache_ttl: Cache TTL in seconds (default 5 minutes)
        """
        self.retriever = hybrid_retriever
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (result, expiry_time)
        
        # Cache metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'total_queries': 0,
            'cache_size': 0,
            'avg_response_time_ms': 0
        }
        
        logger.info(f"Initialized cached hybrid retriever with {cache_ttl}s TTL")
    
    def _generate_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for method call with parameters"""
        key_data = {
            'method': method,
            'project': self.retriever.container.project_name,
            **kwargs
        }
        
        # Sort and serialize for consistent keys
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache:
            return False
        
        _, expiry_time = self._cache[key]
        return time.time() < expiry_time
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if valid"""
        if self._is_cache_valid(key):
            result, _ = self._cache[key]
            self.metrics['hits'] += 1
            return result
        
        # Remove expired entry
        if key in self._cache:
            del self._cache[key]
        
        self.metrics['misses'] += 1
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache with TTL"""
        expiry_time = time.time() + self.cache_ttl
        self._cache[key] = (value, expiry_time)
        self.metrics['cache_size'] = len(self._cache)
    
    async def find_similar_with_context(
        self,
        query: str,
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Cached version of find_similar_with_context
        """
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            method='find_similar_with_context',
            query=query.strip().lower(),
            limit=limit,
            include_graph_context=include_graph_context,
            max_hops=max_hops
        )
        
        # Try cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            response_time = (time.time() - start_time) * 1000
            logger.debug(f"Cache HIT for query '{query[:50]}...' ({response_time:.1f}ms)")
            return cached_result
        
        # Cache miss - compute result
        logger.debug(f"Cache MISS for query '{query[:50]}...'")
        
        result = await self.retriever.find_similar_with_context(
            query=query,
            limit=limit,
            include_graph_context=include_graph_context,
            max_hops=max_hops
        )
        
        # Cache the result
        self._set_cache(cache_key, result)
        
        response_time = (time.time() - start_time) * 1000
        logger.debug(f"Query computed and cached ({response_time:.1f}ms)")
        
        # Update average response time
        total_time = self.metrics['avg_response_time_ms'] * (self.metrics['total_queries'] - 1)
        self.metrics['avg_response_time_ms'] = (total_time + response_time) / self.metrics['total_queries']
        
        return result
    
    async def find_dependencies(
        self,
        file_path: str,
        direction: str = 'both'
    ) -> Dict[str, List[str]]:
        """Cached version of find_dependencies"""
        cache_key = self._generate_cache_key(
            method='find_dependencies',
            file_path=file_path,
            direction=direction
        )
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await self.retriever.find_dependencies(file_path, direction)
        self._set_cache(cache_key, result)
        
        return result
    
    async def analyze_impact(
        self,
        file_path: str,
        change_type: str = 'modify'
    ) -> Dict[str, Any]:
        """Cached version of analyze_impact"""
        cache_key = self._generate_cache_key(
            method='analyze_impact',
            file_path=file_path,
            change_type=change_type
        )
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = await self.retriever.analyze_impact(file_path, change_type)
        self._set_cache(cache_key, result)
        
        return result
    
    def invalidate_file_cache(self, file_path: str) -> int:
        """
        Invalidate cache entries related to a specific file
        Returns number of entries invalidated
        """
        keys_to_remove = []
        
        for key, (cached_data, _) in self._cache.items():
            # Check if cached data contains references to the file
            if isinstance(cached_data, (list, dict)):
                data_str = json.dumps(cached_data, default=str).lower()
                if file_path.lower() in data_str:
                    keys_to_remove.append(key)
            elif isinstance(cached_data, str) and file_path.lower() in cached_data.lower():
                keys_to_remove.append(key)
        
        # Remove identified keys
        for key in keys_to_remove:
            del self._cache[key]
        
        self.metrics['cache_size'] = len(self._cache)
        
        if keys_to_remove:
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries for {file_path}")
        
        return len(keys_to_remove)
    
    def clear_cache(self) -> int:
        """Clear all cache entries"""
        cache_size = len(self._cache)
        self._cache.clear()
        self.metrics['cache_size'] = 0
        
        if cache_size > 0:
            logger.info(f"Cleared {cache_size} cache entries")
        
        return cache_size
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, expiry_time) in self._cache.items():
            if current_time >= expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        self.metrics['cache_size'] = len(self._cache)
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_rate = 0
        if self.metrics['total_queries'] > 0:
            hit_rate = (self.metrics['hits'] / self.metrics['total_queries']) * 100
        
        return {
            'hit_rate_percent': hit_rate,
            'cache_ttl_seconds': self.cache_ttl,
            **self.metrics
        }
    
    def reset_metrics(self) -> None:
        """Reset cache metrics"""
        for key in ['hits', 'misses', 'total_queries', 'avg_response_time_ms']:
            self.metrics[key] = 0

# Example usage
async def example_usage():
    """Demonstrate hybrid retrieval capabilities"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from servers.services.service_container import ServiceContainer
    
    # Initialize services
    container = ServiceContainer("myproject")
    await container.initialize_all_services()
    
    # Create hybrid retriever
    retriever = HybridRetriever(container)
    
    # Example 1: Find similar code with context
    results = await retriever.find_similar_with_context(
        "function to calculate user authentication token",
        limit=5
    )
    
    for result in results:
        print(f"Score: {result['score']:.3f}")
        print(f"File: {result['file_path']}:{result['start_line']}-{result['end_line']}")
        if result.get('graph_context'):
            print(f"Imports: {result['graph_context'].get('imports', [])}")
            print(f"Used by: {result['graph_context'].get('imported_by', [])}")
        print("---")
    
    # Example 2: Analyze change impact
    impact = await retriever.analyze_impact(
        "src/auth/user_service.py",
        change_type="refactor"
    )
    
    print(f"Risk Level: {impact['risk_level']}")
    print(f"Affected Files: {impact['total_affected']}")
    for rec in impact.get('recommendations', []):
        print(f"  - {rec}")

if __name__ == "__main__":
    asyncio.run(example_usage())
