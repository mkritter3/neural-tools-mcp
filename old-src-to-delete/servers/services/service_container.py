"""
Minimal ServiceContainer implementation for multi-project GraphRAG
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ServiceContainer:
    """Simplified service container for multi-project GraphRAG support"""
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.config_base = Path("/app/config/.neural-tools")
        self.neo4j_client = None
        self.qdrant_client = None
        self.initialized = False
        
        # Add attributes expected by MCP server
        self.neo4j = None
        self.qdrant = None
        self.nomic = None  # Embedding client
        
    def ensure_neo4j_client(self):
        """Initialize Neo4j client for this project"""
        try:
            # Lazy import to avoid dependency issues
            from neo4j import GraphDatabase
            
            # Project-specific Neo4j database path
            neo4j_path = self.config_base / "neo4j" / self.project_name
            neo4j_path.mkdir(parents=True, exist_ok=True)
            
            # Use embedded/file-based Neo4j for simplicity in testing
            # In production, this would connect to a real Neo4j instance
            logger.info(f"Neo4j path for {self.project_name}: {neo4j_path}")
            
            # Mock client for now - would be real connection
            self.neo4j_client = MockNeo4jClient(neo4j_path)
            return True
            
        except ImportError:
            logger.warning("Neo4j driver not available, using mock")
            self.neo4j_client = MockNeo4jClient(None)
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j for {self.project_name}: {e}")
            return False
    
    def ensure_qdrant_client(self):
        """Initialize Qdrant client for this project"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # Project-specific Qdrant database path  
            qdrant_path = self.config_base / "qdrant" / self.project_name
            qdrant_path.mkdir(parents=True, exist_ok=True)
            
            # Use file-based Qdrant for multi-project isolation
            self.qdrant_client = QdrantClient(path=str(qdrant_path))
            
            # Ensure default collection exists
            collection_name = f"{self.project_name}_code"
            try:
                collections = self.qdrant_client.get_collections().collections
                if not any(c.name == collection_name for c in collections):
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                    logger.info(f"Created Qdrant collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not ensure Qdrant collection: {e}")
                
            return True
            
        except ImportError:
            logger.warning("Qdrant client not available, using mock")
            self.qdrant_client = MockQdrantClient()
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant for {self.project_name}: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize all services for this project"""
        if self.initialized:
            return True
            
        logger.info(f"Initializing ServiceContainer for project: {self.project_name}")
        
        neo4j_ok = self.ensure_neo4j_client()
        qdrant_ok = self.ensure_qdrant_client()
        
        # Set the attributes expected by MCP server
        self.neo4j = self.neo4j_client if neo4j_ok else None
        self.qdrant = self.qdrant_client if qdrant_ok else None
        self.nomic = MockNomicClient()  # Mock embedding client
        
        self.initialized = True
        return neo4j_ok and qdrant_ok
    
    async def initialize_all_services(self) -> bool:
        """Async version of initialize for MCP server compatibility"""
        return self.initialize()
    
    def get_neo4j_client(self):
        """Get Neo4j client, initializing if needed"""
        if not self.neo4j_client:
            self.ensure_neo4j_client()
        return self.neo4j_client
    
    def get_qdrant_client(self):
        """Get Qdrant client, initializing if needed"""
        if not self.qdrant_client:
            self.ensure_qdrant_client()
        return self.qdrant_client


class MockNeo4jClient:
    """Mock Neo4j client for testing"""
    def __init__(self, path):
        self.path = path
        
    def run(self, query: str, **kwargs):
        """Mock query execution"""
        return MockResult()
    
    def close(self):
        pass


class MockQdrantClient:
    """Mock Qdrant client for testing"""
    def __init__(self):
        pass
        
    def search(self, collection_name: str, **kwargs):
        """Mock search"""
        return []
    
    def upsert(self, collection_name: str, **kwargs):
        """Mock upsert"""
        return True


class MockResult:
    """Mock Neo4j result"""
    def data(self):
        return []


class MockNomicClient:
    """Mock Nomic embedding client for testing"""
    def __init__(self):
        self.model_name = "nomic-embed-text-v1"
    
    async def embed_text(self, text: str):
        """Mock text embedding"""
        # Return a mock 768-dimensional vector
        import hashlib
        import numpy as np
        hash_obj = hashlib.md5(text.encode())
        np.random.seed(int(hash_obj.hexdigest()[:8], 16))
        return np.random.randn(768).tolist()
    
    def embed_texts(self, texts: list):
        """Mock batch text embedding"""
        return [self.embed_text(text) for text in texts]