"""
ChunkSchema - The Single Source of Truth Contract
This enforces consistency between indexer, storage, and retrieval
ADR-0096: Schema Contract Pattern
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class ChunkSchema:
    """
    The canonical chunk structure that ALL components must follow.
    This is the contract between indexer, Neo4j storage, and retrieval.
    """
    # Required fields
    chunk_id: str  # Format: "file_path:chunk:index"
    file_path: str  # Always present for search filtering
    content: str  # The actual text content
    embedding: List[float]  # Must be exactly 768 dimensions
    project: str  # Project isolation

    # Required with defaults
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())  # ISO string, NOT DateTime
    start_line: int = 0
    end_line: int = 0
    size: int = 0

    def __post_init__(self):
        """Validate the contract"""
        # Enforce embedding dimensions
        if self.embedding and len(self.embedding) != 768:
            raise ValueError(f"Embedding must be 768 dimensions, got {len(self.embedding)}")

        # Ensure ISO string for created_at
        if isinstance(self.created_at, datetime):
            self.created_at = self.created_at.isoformat()

        # Validate chunk_id format
        if not self.chunk_id or ':chunk:' not in self.chunk_id:
            raise ValueError(f"Invalid chunk_id format: {self.chunk_id}")

        # Extract file_path from chunk_id if missing
        if not self.file_path:
            self.file_path = self.chunk_id.split(':chunk:')[0]

    def to_neo4j_dict(self) -> dict:
        """Convert to Neo4j-compatible dictionary (no nested objects)"""
        return {
            'chunk_id': self.chunk_id,
            'file_path': self.file_path,
            'content': self.content,
            'embedding': self.embedding,
            'project': self.project,
            'created_at': self.created_at,  # ISO string
            'start_line': self.start_line,
            'end_line': self.end_line,
            'size': self.size or len(self.content)
        }

    @classmethod
    def from_neo4j_node(cls, node: dict) -> 'ChunkSchema':
        """Create from Neo4j node result"""
        return cls(
            chunk_id=node.get('chunk_id'),
            file_path=node.get('file_path'),
            content=node.get('content'),
            embedding=node.get('embedding', []),
            project=node.get('project'),
            created_at=node.get('created_at', datetime.utcnow().isoformat()),
            start_line=node.get('start_line', 0),
            end_line=node.get('end_line', 0),
            size=node.get('size', 0)
        )


# Neo4j Constraints to enforce at database level
NEO4J_CHUNK_CONSTRAINTS = [
    "CREATE CONSTRAINT chunk_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.chunk_id, c.project) IS UNIQUE",
    "CREATE CONSTRAINT chunk_has_file_path IF NOT EXISTS FOR (c:Chunk) REQUIRE c.file_path IS NOT NULL",
    "CREATE CONSTRAINT chunk_has_content IF NOT EXISTS FOR (c:Chunk) REQUIRE c.content IS NOT NULL",
    "CREATE CONSTRAINT chunk_has_project IF NOT EXISTS FOR (c:Chunk) REQUIRE c.project IS NOT NULL"
]

# Vector index definition
NEO4J_VECTOR_INDEX = """
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }
}
"""