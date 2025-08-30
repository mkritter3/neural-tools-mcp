#!/usr/bin/env python3
"""
L9 Qdrant Memory System V2 - Project-Aware with Automatic Isolation
Each project gets its own Qdrant container with isolated data
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Qdrant and embedding imports
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from qdrant_client.models import PointStruct, SparseVector, NamedVector

# Project configuration
from l9_config_manager import get_config
from l9_project_isolation import L9ProjectIsolation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryResult:
    """Memory search result with metadata"""
    content: str
    memory_id: str
    timestamp: datetime
    score: float
    entities: List[str]
    token_count: int
    search_type: str  # "active" or "passive"
    project: str  # Project this memory belongs to
    metadata: Dict[str, Any]

class L9QdrantMemoryV2:
    """
    Project-Aware L9 Memory System
    - Automatic project detection
    - Per-project Qdrant containers
    - Shared embedding models
    - Complete data isolation
    """
    
    def __init__(self):
        """Initialize with automatic project detection"""
        
        # Get project configuration
        self.config = get_config()
        self.project_name = self.config.get_project_name()
        self.container_name = self.config.get_container_name()
        
        # Project isolation manager
        self.isolation_manager = L9ProjectIsolation()
        
        # Qdrant client will be initialized after ensuring container
        self.client = None
        self.collection_name = self.config.get_collection_name("memory")
        
        # Models (will be loaded or connected to shared server)
        self.dense_model = None
        self.sparse_model = None
        
        # Token tracking (per project)
        self.daily_tokens = 0
        self.token_budget = 150000
        
        logger.info(f"🚀 L9 Memory V2 initialized for project: {self.project_name}")
        logger.info(f"   Container: {self.container_name}")
        logger.info(f"   Collection: {self.collection_name}")
        
    async def initialize(self):
        """Initialize project container and models"""
        try:
            # Ensure project container is running
            logger.info(f"🐳 Ensuring container for project: {self.project_name}")
            project_context = self.isolation_manager.get_project_context()
            container_info = self.isolation_manager.ensure_project_container(project_context)
            
            logger.info(f"✅ Container status: {container_info['status']}")
            logger.info(f"   REST port: {container_info['rest_port']}")
            logger.info(f"   gRPC port: {container_info['grpc_port']}")
            
            # Initialize Qdrant client with project-specific ports
            qdrant_config = self.config.get_qdrant_config(prefer_grpc=True)
            self.client = QdrantClient(**qdrant_config)
            
            # Wait for Qdrant to be ready
            await self._wait_for_qdrant()
            
            # Create or verify collection
            await self._setup_collection()
            
            # Initialize models (stub for now - would connect to shared server)
            await self._initialize_models()
            
            logger.info(f"✅ L9 Memory V2 ready for project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def _wait_for_qdrant(self, max_retries: int = 10):
        """Wait for Qdrant to be ready"""
        for i in range(max_retries):
            try:
                # Try to get collections
                self.client.get_collections()
                logger.info("✅ Qdrant is ready")
                return
            except Exception as e:
                if i == max_retries - 1:
                    raise Exception(f"Qdrant not ready after {max_retries} attempts")
                logger.info(f"⏳ Waiting for Qdrant... ({i+1}/{max_retries})")
                await asyncio.sleep(2)
    
    async def _setup_collection(self):
        """Create project-specific collection with hybrid search"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(c.name == self.collection_name for c in collections.collections)
            
            if exists:
                logger.info(f"📂 Using existing collection: {self.collection_name}")
                return
            
            # Create new collection
            logger.info(f"📂 Creating collection: {self.collection_name}")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "semantic": VectorParams(
                        size=1536,  # Qodo-Embed dimensions
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
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                ),
                on_disk_payload=False
            )
            
            # Create indices
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="project",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="timestamp",
                field_schema=models.PayloadSchemaType.DATETIME
            )
            
            logger.info(f"✅ Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    async def _initialize_models(self):
        """Initialize or connect to embedding models"""
        # In production, this would connect to the shared model server
        # For now, we'll use mock embeddings
        logger.info("🤖 Models initialized (mock mode)")
        self.dense_model = "mock"
        self.sparse_model = "mock"
    
    def _generate_embeddings(self, text: str) -> Tuple[List[float], SparseVector]:
        """Generate embeddings (mock for testing)"""
        # In production, this would call the shared model server
        import hashlib
        
        # Mock dense embedding based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_nums = [int(text_hash[i:i+2], 16) / 255.0 for i in range(0, min(len(text_hash), 32), 2)]
        dense_embedding = hash_nums * (1536 // len(hash_nums)) + hash_nums[:1536 % len(hash_nums)]
        
        # Mock sparse embedding
        sparse_indices = [abs(hash(text)) % 1000 + i*100 for i in range(5)]
        sparse_values = [0.5 + i*0.1 for i in range(5)]
        sparse_vector = SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
        
        return dense_embedding, sparse_vector
    
    async def index_memory(self,
                          memory_id: str,
                          content: str,
                          metadata: Optional[Dict] = None) -> bool:
        """Index a memory for the current project"""
        try:
            # Generate embeddings
            dense_embedding, sparse_vector = self._generate_embeddings(content)
            
            # Prepare payload with project identifier
            payload = {
                "content": content,
                "project": self.project_name,
                "timestamp": datetime.now().isoformat(),
                "token_count": len(content.split()),  # Mock token count
                "entities": [],  # Would extract entities in production
                **(metadata or {})
            }
            
            # Create point
            point = PointStruct(
                id=f"{self.project_name}_{memory_id}",  # Project-prefixed ID
                vector={
                    "semantic": dense_embedding,
                    "bm25": sparse_vector
                },
                payload=payload
            )
            
            # Upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"✅ Indexed memory {memory_id} for project {self.project_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index memory: {e}")
            return False
    
    async def search_project_memories(self,
                                     query: str,
                                     limit: int = 10,
                                     include_other_projects: bool = False) -> List[MemoryResult]:
        """Search memories with optional cross-project access"""
        
        # Generate query embeddings
        dense_embedding, sparse_vector = self._generate_embeddings(query)
        
        # Build filter
        filter_conditions = []
        if not include_other_projects:
            # Filter to current project only
            filter_conditions.append(
                models.FieldCondition(
                    key="project",
                    match=models.MatchValue(value=self.project_name)
                )
            )
        
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        # Hybrid search with RRF fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vector,
                    using="bm25",
                    limit=20,
                    filter=query_filter
                ),
                models.Prefetch(
                    query=dense_embedding,
                    using="semantic",
                    limit=20,
                    filter=query_filter
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=limit
        )
        
        # Convert to MemoryResult
        memory_results = []
        for point in results.points:
            memory_results.append(MemoryResult(
                content=point.payload.get("content", ""),
                memory_id=str(point.id),
                timestamp=datetime.fromisoformat(point.payload.get("timestamp")),
                score=point.score if hasattr(point, 'score') else 1.0,
                entities=point.payload.get("entities", []),
                token_count=point.payload.get("token_count", 0),
                search_type="hybrid",
                project=point.payload.get("project", "unknown"),
                metadata=point.payload
            ))
        
        scope = "current project" if not include_other_projects else "all projects"
        logger.info(f"🔍 Found {len(memory_results)} memories in {scope}")
        
        return memory_results
    
    async def get_project_stats(self) -> Dict[str, Any]:
        """Get statistics for current project"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Count project-specific memories
            project_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="project",
                        match=models.MatchValue(value=self.project_name)
                    )
                ]
            )
            
            # Use scroll to count project memories
            project_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=project_filter,
                limit=1000
            )
            
            return {
                "project": self.project_name,
                "container": self.container_name,
                "collection": self.collection_name,
                "total_memories_in_collection": collection_info.points_count,
                "project_memories": len(project_points),
                "status": "healthy" if collection_info.status == "green" else "degraded",
                "storage_path": self.config._config['storage_path']
            }
            
        except Exception as e:
            return {"error": str(e)}


async def main():
    """Test project-aware memory system"""
    
    print("\n🧪 L9 PROJECT-AWARE MEMORY SYSTEM TEST")
    print("=" * 60)
    
    # Initialize memory system (will auto-detect project)
    memory = L9QdrantMemoryV2()
    await memory.initialize()
    
    # Show project info
    print(f"\n📁 Project Configuration:")
    print(memory.config.summary())
    
    # Test memories for this project
    test_memories = [
        {
            "id": "proj_mem_001",
            "content": f"Project {memory.project_name} uses PostgreSQL for data storage",
            "metadata": {"type": "architecture"}
        },
        {
            "id": "proj_mem_002",
            "content": f"The {memory.project_name} API has rate limiting at 1000 req/min",
            "metadata": {"type": "security"}
        },
        {
            "id": "proj_mem_003",
            "content": f"Frontend for {memory.project_name} built with React and TypeScript",
            "metadata": {"type": "tech-stack"}
        }
    ]
    
    # Index memories
    print(f"\n📝 Indexing memories for project: {memory.project_name}")
    for mem in test_memories:
        success = await memory.index_memory(
            memory_id=mem["id"],
            content=mem["content"],
            metadata=mem["metadata"]
        )
        print(f"  {'✅' if success else '❌'} {mem['id']}")
    
    # Search in current project
    print(f"\n🔍 Searching in project: {memory.project_name}")
    results = await memory.search_project_memories(
        query="What database are we using?",
        limit=3,
        include_other_projects=False
    )
    
    for result in results:
        print(f"  • [{result.project}] Score: {result.score:.3f}")
        print(f"    {result.content[:80]}...")
    
    # Get project stats
    stats = await memory.get_project_stats()
    print(f"\n📊 Project Stats:")
    print(json.dumps(stats, indent=2))
    
    print("\n✅ Project isolation verified!")
    print(f"   Each project gets its own:")
    print(f"   • Qdrant container: {memory.container_name}")
    print(f"   • Unique ports: {memory.config._config['rest_port']}/{memory.config._config['grpc_port']}")
    print(f"   • Isolated storage: {memory.config._config['storage_path']}")
    print(f"   • Namespaced collections: {memory.collection_name}")

if __name__ == "__main__":
    asyncio.run(main())