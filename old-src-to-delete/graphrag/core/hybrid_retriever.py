#!/usr/bin/env python3
"""
GraphRAG Hybrid Retriever for L9 Neural Tools
Combines Qdrant semantic search with Neo4j graph traversal
Enables powerful hybrid queries across both databases
"""

import asyncio
import logging
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
        self.collection_prefix = f"project_{container.project_name}_"
        
    async def find_similar_with_context(
        self, 
        query: str,
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar code with graph context
        
        This is the core GraphRAG query pattern:
        1. Semantic search in Qdrant
        2. Enrich with graph relationships from Neo4j
        3. Return combined context
        
        Args:
            query: Natural language query or code snippet
            limit: Number of similar chunks to find
            include_graph_context: Whether to fetch graph relationships
            max_hops: Maximum graph traversal depth
            
        Returns:
            List of results with code chunks and their relationships
        """
        try:
            # Step 1: Get query embedding
            embeddings = await self.container.nomic.get_embeddings([query])
            if not embeddings:
                logger.warning("Failed to generate query embedding")
                return []
            
            query_vector = embeddings[0]
            
            # Step 2: Semantic search in Qdrant
            collection_name = f"{self.collection_prefix}code"
            search_results = await self.container.qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True
            )
            
            if not search_results:
                return []
            
            # Extract chunk IDs and initial results
            results = []
            chunk_ids = []
            
            for hit in search_results:
                result = {
                    'score': hit.score,
                    'chunk_id': hit.payload.get('chunk_hash'),
                    'neo4j_chunk_id': hit.payload.get('neo4j_chunk_id'),
                    'file_path': hit.payload.get('file_path'),
                    'content': hit.payload.get('content'),
                    'start_line': hit.payload.get('start_line'),
                    'end_line': hit.payload.get('end_line'),
                    'file_type': hit.payload.get('file_type')
                }
                results.append(result)
                
                if hit.payload.get('neo4j_chunk_id'):
                    chunk_ids.append(hit.payload['neo4j_chunk_id'])
            
            # Step 3: Enrich with graph context from Neo4j
            if include_graph_context and chunk_ids and self.container.neo4j:
                graph_context = await self._fetch_graph_context(chunk_ids, max_hops)
                
                # Merge graph context with results
                for result, context in zip(results, graph_context):
                    result['graph_context'] = context
            
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
                # Query for chunk relationships
                cypher = """
                MATCH (c:CodeChunk {id: $chunk_id})
                OPTIONAL MATCH (c)-[:PART_OF]->(f:File)
                OPTIONAL MATCH (f)-[:IMPORTS]->(m:Module)
                OPTIONAL MATCH (f)<-[:IMPORTS]-(importer:File)
                OPTIONAL MATCH (related:CodeChunk)-[:PART_OF]->(f)
                WHERE related.id <> c.id AND related.type IN ['function', 'class']
                RETURN 
                    f.path AS file_path,
                    f.type AS file_type,
                    f.content_hash AS file_hash,
                    collect(DISTINCT m.name) AS imports,
                    collect(DISTINCT importer.path) AS imported_by,
                    collect(DISTINCT {
                        id: related.id,
                        type: related.type,
                        start_line: related.start_line,
                        end_line: related.end_line
                    }) AS related_chunks
                LIMIT 1
                """
                
                result = await self.container.neo4j.execute_cypher(cypher, {
                    'chunk_id': chunk_id
                })
                
                if result:
                    context = result[0]
                    # Clean up empty collections
                    context['imports'] = [i for i in context.get('imports', []) if i]
                    context['imported_by'] = [i for i in context.get('imported_by', []) if i]
                    context['related_chunks'] = [
                        c for c in context.get('related_chunks', []) 
                        if c and c.get('id')
                    ]
                    contexts.append(context)
                else:
                    contexts.append({})
                    
            except Exception as e:
                logger.error(f"Failed to fetch graph context for chunk {chunk_id}: {e}")
                contexts.append({})
        
        return contexts
    
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
            
            cypher = f"""
            MATCH (start:File {{path: $file_path}})
            CALL apoc.path.subgraphAll(start, {{
                relationshipFilter: '{rel_pattern}',
                maxLevel: $max_depth
            }})
            YIELD nodes, relationships
            RETURN nodes, relationships
            """
            
            # Note: This requires APOC plugin in Neo4j
            # Fallback query without APOC:
            fallback_cypher = """
            MATCH path = (start:File {path: $file_path})-[*1..3]-(related)
            WHERE ALL(r IN relationships(path) WHERE type(r) IN $rel_types)
            RETURN collect(DISTINCT nodes(path)) AS nodes,
                   collect(DISTINCT relationships(path)) AS relationships
            """
            
            try:
                result = await self.container.neo4j.execute_cypher(cypher, {
                    'file_path': file_path,
                    'max_depth': max_depth
                })
            except:
                # Fallback if APOC not available
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
                cypher_imports = """
                MATCH (f:File {path: $file_path})-[:IMPORTS]->(m:Module)
                OPTIONAL MATCH (m)<-[:PROVIDES]-(provider:File)
                RETURN collect(DISTINCT COALESCE(provider.path, m.name)) AS imports
                """
                imports_result = await self.container.neo4j.execute_cypher(
                    cypher_imports, {'file_path': file_path}
                )
                if imports_result:
                    result['imports'] = imports_result[0].get('imports', [])
            
            if direction in ['imported_by', 'both']:
                cypher_imported = """
                MATCH (f:File {path: $file_path})<-[:IMPORTS]-(importer:File)
                RETURN collect(DISTINCT importer.path) AS imported_by
                """
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
            cypher = """
            MATCH (f:File {path: $file_path})
            OPTIONAL MATCH (f)<-[:IMPORTS*1..3]-(affected:File)
            WITH collect(DISTINCT affected.path) AS transitive
            OPTIONAL MATCH (f)-[:CONTAINS]->(chunk:CodeChunk)
            WHERE chunk.type IN ['class', 'function']
            RETURN transitive, count(DISTINCT chunk) AS api_surface
            """
            
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