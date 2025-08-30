#!/usr/bin/env python3
"""
L9 Qdrant Hybrid Search - Simplified with Native Qdrant Support
Replaces complex triple-index with single Qdrant collection
"""

import os
import ast
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Qdrant imports
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from qdrant_client.models import PointStruct, SparseVector
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HybridSearchResult:
    """Unified search result from Qdrant hybrid search"""
    content: str
    file_path: str
    line_range: Tuple[int, int]
    score: float
    search_type: str  # "hybrid", "semantic", "keyword", "pattern"
    ast_patterns: List[str]
    metadata: Dict[str, Any]

class ASTPatternExtractor:
    """Extract AST patterns from Python code"""
    
    def extract_patterns(self, code: str) -> List[str]:
        """Extract structural patterns from code"""
        patterns = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Function definitions
                if isinstance(node, ast.FunctionDef):
                    patterns.append(f"func:{node.name}")
                    # Extract decorators
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            patterns.append(f"decorator:@{decorator.id}")
                            
                # Class definitions
                elif isinstance(node, ast.ClassDef):
                    patterns.append(f"class:{node.name}")
                    # Extract base classes
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            patterns.append(f"inherits:{base.id}")
                            
                # Import statements
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        patterns.append(f"import:{alias.name}")
                        
                # Async functions
                elif isinstance(node, ast.AsyncFunctionDef):
                    patterns.append(f"async:{node.name}")
                    
        except SyntaxError:
            # Not valid Python, skip AST parsing
            pass
            
        return patterns

class L9QdrantHybridSearch:
    """
    Simplified L9 Hybrid Search using Qdrant's Native Capabilities
    - No more triple-index complexity
    - Server-side RRF fusion
    - AST patterns stored in metadata
    """
    
    def __init__(self,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6433,  # L9 template port (avoid conflict with enterprise v3)
                 collection_name: str = "l9_code_search"):
        """Initialize Qdrant-based hybrid search"""
        
        # Qdrant client with gRPC for performance
        # For gRPC, use port 6434 (L9 template gRPC port)
        grpc_port = 6434 if qdrant_port == 6433 else qdrant_port + 1
        self.client = QdrantClient(
            host=qdrant_host,
            port=grpc_port,  # Use gRPC port for 3-4x performance
            prefer_grpc=True  # 3-4x faster
        )
        
        self.collection_name = collection_name
        
        # Models
        self.dense_model = None
        self.sparse_model = None
        self.ast_extractor = ASTPatternExtractor()
        
        logger.info("🚀 L9 Qdrant Hybrid Search initialized")
        
    async def initialize(self):
        """Initialize models and create collection"""
        
        # Load models
        logger.info("📥 Loading embedding models...")
        self.dense_model = SentenceTransformer('Qodo/Qodo-Embed-1-1.5B')
        self.sparse_model = SparseTextEmbedding('Qdrant/bm25')
        
        # Setup Qdrant collection
        await self._setup_collection()
        
        logger.info("✅ Hybrid search ready with Qdrant")
        
    async def _setup_collection(self):
        """Create Qdrant collection with hybrid support"""
        
        # Check if exists
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        
        if exists:
            logger.info(f"📂 Using existing collection: {self.collection_name}")
            return
            
        # Create hybrid collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "semantic": VectorParams(
                    size=1536,  # Qodo-Embed
                    distance=Distance.COSINE,
                    on_disk=False
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    on_disk=False
                )
            },
            # Performance optimizations
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            )
        )
        
        # Create indices for metadata filtering
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="file_path",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="ast_patterns",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        
        logger.info(f"✅ Created hybrid collection: {self.collection_name}")
        
    async def index_code(self,
                        doc_id: str,
                        content: str,
                        file_path: str,
                        line_range: Tuple[int, int],
                        metadata: Optional[Dict] = None) -> bool:
        """Index code with semantic, keyword, and AST patterns"""
        
        try:
            # Extract AST patterns
            ast_patterns = self.ast_extractor.extract_patterns(content)
            
            # Generate embeddings
            dense_embedding = self.dense_model.encode(content, convert_to_numpy=True)
            sparse_result = list(self.sparse_model.embed([content]))[0]
            
            # Prepare payload with AST patterns
            payload = {
                "content": content,
                "file_path": file_path,
                "line_start": line_range[0],
                "line_end": line_range[1],
                "ast_patterns": ast_patterns,  # Store AST patterns in metadata!
                "indexed_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Create point
            point = PointStruct(
                id=doc_id,
                vector={
                    "semantic": dense_embedding.tolist(),
                    "bm25": SparseVector(
                        indices=sparse_result.indices.tolist(),
                        values=sparse_result.values.tolist()
                    )
                },
                payload=payload
            )
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"✅ Indexed {file_path}:{line_range} with {len(ast_patterns)} AST patterns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            return False
            
    async def search(self,
                    query: str,
                    limit: int = 10,
                    file_filter: Optional[str] = None,
                    pattern_filter: Optional[List[str]] = None) -> List[HybridSearchResult]:
        """
        Perform hybrid search with optional AST pattern filtering
        Qdrant handles BM25 + Semantic fusion natively!
        """
        
        # Generate query embeddings
        dense_embedding = self.dense_model.encode(query, convert_to_numpy=True)
        sparse_result = list(self.sparse_model.embed([query]))[0]
        
        # Build filters
        filter_conditions = []
        
        if file_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchValue(value=file_filter)
                )
            )
            
        if pattern_filter:
            # Filter by AST patterns
            filter_conditions.append(
                models.FieldCondition(
                    key="ast_patterns",
                    match=models.MatchAny(any=pattern_filter)
                )
            )
            
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        # Qdrant native hybrid search with prefetch
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # BM25 search
                models.Prefetch(
                    query=SparseVector(
                        indices=sparse_result.indices.tolist(),
                        values=sparse_result.values.tolist()
                    ),
                    using="bm25",
                    limit=20,
                    filter=query_filter
                ),
                # Semantic search
                models.Prefetch(
                    query=dense_embedding.tolist(),
                    using="semantic",
                    limit=20,
                    filter=query_filter
                )
            ],
            # Qdrant's native RRF fusion!
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=limit
        )
        
        # Convert to HybridSearchResult
        search_results = []
        for point in results.points:
            payload = point.payload
            search_results.append(HybridSearchResult(
                content=payload.get("content", ""),
                file_path=payload.get("file_path", ""),
                line_range=(payload.get("line_start", 0), payload.get("line_end", 0)),
                score=point.score if hasattr(point, 'score') else 1.0,
                search_type="hybrid",
                ast_patterns=payload.get("ast_patterns", []),
                metadata=payload
            ))
            
        logger.info(f"🔍 Hybrid search returned {len(search_results)} results")
        return search_results
        
    async def search_by_pattern(self,
                               patterns: List[str],
                               limit: int = 10) -> List[HybridSearchResult]:
        """Search specifically by AST patterns"""
        
        # Filter by AST patterns
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="ast_patterns",
                    match=models.MatchAny(any=patterns)
                )
            ]
        )
        
        # Use scroll to get all matching documents
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        # Convert to results
        search_results = []
        for point in results[0]:  # First element is the points list
            payload = point.payload
            search_results.append(HybridSearchResult(
                content=payload.get("content", ""),
                file_path=payload.get("file_path", ""),
                line_range=(payload.get("line_start", 0), payload.get("line_end", 0)),
                score=1.0,  # No score in scroll
                search_type="pattern",
                ast_patterns=payload.get("ast_patterns", []),
                metadata=payload
            ))
            
        logger.info(f"🎯 Pattern search returned {len(search_results)} results")
        return search_results
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        
        info = self.client.get_collection(self.collection_name)
        
        return {
            "collection": self.collection_name,
            "total_documents": info.points_count,
            "vectors_count": info.vectors_count,
            "status": "healthy" if info.status == "green" else "degraded",
            "config": {
                "dense_vector_size": 1536,
                "sparse_vectors": "BM25 with IDF",
                "ast_patterns": "Stored in metadata"
            }
        }


async def main():
    """Test Qdrant hybrid search"""
    
    # Initialize
    search = L9QdrantHybridSearch()
    await search.initialize()
    
    # Test code samples
    test_code = [
        {
            "id": "code_001",
            "content": """
async def authenticate_user(username: str, password: str) -> bool:
    user = await db.find_user(username)
    if user and verify_password(password, user.password_hash):
        return True
    return False
""",
            "file": "auth.py",
            "lines": (10, 15)
        },
        {
            "id": "code_002",
            "content": """
class UserController:
    def __init__(self, db_connection):
        self.db = db_connection
        
    async def create_user(self, user_data: dict):
        return await self.db.insert('users', user_data)
""",
            "file": "controllers/user.py",
            "lines": (1, 7)
        }
    ]
    
    # Index code
    print("\n📝 Indexing code samples...")
    for code in test_code:
        success = await search.index_code(
            doc_id=code["id"],
            content=code["content"],
            file_path=code["file"],
            line_range=code["lines"]
        )
        print(f"  {'✅' if success else '❌'} {code['file']}")
        
    # Test hybrid search
    print("\n🔍 Testing hybrid search...")
    results = await search.search(
        query="authenticate user with password",
        limit=5
    )
    
    for result in results:
        print(f"  Score: {result.score:.3f} | {result.file_path}:{result.line_range}")
        print(f"    AST: {', '.join(result.ast_patterns[:3])}")
        
    # Test pattern search
    print("\n🎯 Testing AST pattern search...")
    pattern_results = await search.search_by_pattern(
        patterns=["async:authenticate_user", "class:UserController"],
        limit=5
    )
    
    for result in pattern_results:
        print(f"  {result.file_path} | Patterns: {', '.join(result.ast_patterns[:3])}")
        
    # Show stats
    stats = await search.get_stats()
    print(f"\n📊 Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())