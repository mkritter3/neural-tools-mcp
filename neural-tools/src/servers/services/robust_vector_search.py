"""
Robust Vector Search - Following Neo4j Official Patterns
ADR-0096: Implements the VectorCypherRetriever pattern
"""

from typing import List, Dict, Any, Optional
import logging
from chunk_schema import ChunkSchema

logger = logging.getLogger(__name__)


class RobustVectorSearch:
    """
    Implements Neo4j's official VectorCypherRetriever pattern
    This is the battle-tested approach used by neo4j-graphrag-python
    """

    def __init__(self, neo4j_service, project_name: str):
        self.neo4j = neo4j_service
        self.project = project_name

    async def vector_search_phase1(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Pure vector similarity search
        Uses literal index name (not parameterized) as per Neo4j requirements
        """
        if not query_embedding or len(query_embedding) != 768:
            logger.error(f"Invalid embedding: dimension {len(query_embedding) if query_embedding else 0}")
            return []

        # CRITICAL: Use literal index name, not parameter
        # Neo4j CALL procedures don't support parameterized index names
        # Note: Neo4j may append '_index' to the name we specify
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embeddings_index', $limit, $embedding)
        YIELD node, score
        WHERE node.project = $project AND score >= $min_score
        RETURN
            elementId(node) as element_id,
            node.chunk_id as chunk_id,
            node.file_path as file_path,
            node.content as content,
            node.start_line as start_line,
            node.end_line as end_line,
            score
        ORDER BY score DESC
        """

        params = {
            'embedding': query_embedding,
            'limit': limit * 2,  # Get extra for filtering
            'project': self.project,
            'min_score': min_score
        }

        result = await self.neo4j.execute_cypher(cypher, params)

        if result['status'] != 'success':
            logger.error(f"Vector search failed: {result.get('message')}")
            return []

        return result.get('result', [])

    async def graph_enrichment_phase2(
        self,
        chunk_ids: List[str],
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Graph traversal for context enrichment
        Uses UNWIND and bounded paths as per official pattern
        """
        if not chunk_ids:
            return []

        # Official pattern: UNWIND for controlled iteration
        # Bounded paths {1,2} prevent graph explosion
        cypher = """
        UNWIND $chunk_ids as chunk_id
        MATCH (c:Chunk {chunk_id: chunk_id, project: $project})

        // Controlled traversal with bounded depth
        WITH c
        OPTIONAL MATCH path = (c)-[*1..2]-(related)
        WHERE related:Chunk OR related:Function OR related:Class

        // Collect distinct related nodes
        WITH c, collect(DISTINCT {
            id: elementId(related),
            type: labels(related)[0],
            name: coalesce(related.name, related.chunk_id),
            file_path: related.file_path
        }) as context

        RETURN
            c.chunk_id as chunk_id,
            c.content as content,
            c.file_path as file_path,
            context
        """

        params = {
            'chunk_ids': chunk_ids[:10],  # Limit to prevent explosion
            'project': self.project
        }

        result = await self.neo4j.execute_cypher(cypher, params)

        if result['status'] != 'success':
            logger.error(f"Graph enrichment failed: {result.get('message')}")
            return []

        return result.get('result', [])

    async def hybrid_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.5,
        enrich_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Complete two-phase search following VectorCypherRetriever pattern
        """
        # Phase 1: Vector search
        vector_results = await self.vector_search_phase1(
            query_embedding, limit, min_score
        )

        if not vector_results:
            logger.info("No vector results found")
            return []

        # Extract chunk IDs for Phase 2
        chunk_ids = [r['chunk_id'] for r in vector_results if r.get('chunk_id')]

        # Phase 2: Optional graph enrichment
        if enrich_context and chunk_ids:
            enriched = await self.graph_enrichment_phase2(chunk_ids)

            # Merge enrichment with vector results
            enrichment_map = {e['chunk_id']: e['context'] for e in enriched}

            for result in vector_results:
                chunk_id = result.get('chunk_id')
                if chunk_id in enrichment_map:
                    result['graph_context'] = enrichment_map[chunk_id]

        return vector_results


def create_robust_search(neo4j_service, project_name: str) -> RobustVectorSearch:
    """Factory function to create robust search instance"""
    return RobustVectorSearch(neo4j_service, project_name)