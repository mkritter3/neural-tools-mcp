"""
ChunkSchema - The Single Source of Truth Contract
This enforces consistency between indexer, storage, and retrieval
ADR-0096: Schema Contract Pattern
ADR-0098: Incremental backfill support with versioning
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

# Current schema version
CHUNK_SCHEMA_VERSION = "1.1"


@dataclass
class ChunkSchema:
    """
    The canonical chunk structure that ALL components must follow.
    This is the contract between indexer, Neo4j storage, and retrieval.

    Version History:
    - 1.0: Original schema (basic fields)
    - 1.1: Added file metadata and git information
    """
    # Required fields (v1.0)
    chunk_id: str  # Format: "file_path:chunk:index"
    file_path: str  # Always present for search filtering
    content: str  # The actual text content
    embedding: List[float]  # Must be exactly 768 dimensions
    project: str  # Project isolation

    # Required with defaults (v1.0)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())  # ISO string, NOT DateTime
    start_line: int = 0
    end_line: int = 0
    size: int = 0

    # Optional fields (v1.1) - Added for incremental backfill
    file_modified_at: Optional[str] = None  # File modification timestamp (ISO)
    file_created_at: Optional[str] = None   # File creation timestamp (ISO)
    git_commit_sha: Optional[str] = None    # Git commit when indexed
    git_author: Optional[str] = None        # Last git author
    file_type: Optional[str] = None         # File extension/type
    language: Optional[str] = None          # Programming language
    is_canonical: Optional[bool] = None     # From PRISM scoring

    # Backfill metadata
    metadata_version: str = "1.0"           # Schema version of this chunk
    backfill_status: Optional[str] = None   # pending/processing/complete
    backfill_timestamp: Optional[str] = None  # When backfilled

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
        base_dict = {
            'chunk_id': self.chunk_id,
            'file_path': self.file_path,
            'content': self.content,
            'embedding': self.embedding,
            'project': self.project,
            'created_at': self.created_at,  # ISO string
            'start_line': self.start_line,
            'end_line': self.end_line,
            'size': self.size or len(self.content),
            'metadata_version': self.metadata_version
        }

        # Add v1.1 fields if present (for backfilled chunks)
        if self.file_modified_at is not None:
            base_dict['file_modified_at'] = self.file_modified_at
        if self.file_created_at is not None:
            base_dict['file_created_at'] = self.file_created_at
        if self.git_commit_sha is not None:
            base_dict['git_commit_sha'] = self.git_commit_sha
        if self.git_author is not None:
            base_dict['git_author'] = self.git_author
        if self.file_type is not None:
            base_dict['file_type'] = self.file_type
        if self.language is not None:
            base_dict['language'] = self.language
        if self.is_canonical is not None:
            base_dict['is_canonical'] = self.is_canonical
        if self.backfill_status is not None:
            base_dict['backfill_status'] = self.backfill_status
        if self.backfill_timestamp is not None:
            base_dict['backfill_timestamp'] = self.backfill_timestamp

        return base_dict

    def to_dict(self) -> dict:
        """Convert to regular dictionary for JSON serialization"""
        return {
            'chunk_id': self.chunk_id,
            'file_path': self.file_path,
            'content': self.content,
            'embedding': self.embedding,
            'project': self.project,
            'created_at': self.created_at,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'size': self.size
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