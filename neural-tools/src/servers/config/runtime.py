"""
Runtime Configuration Module
Centralized configuration management with environment variable precedence
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    # Neo4j Configuration with URI precedence
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    
    # Qdrant Configuration with port normalization
    qdrant_host: str
    qdrant_port: int
    qdrant_timeout: float


@dataclass
class EmbeddingConfig:
    """Embedding service configuration"""
    embed_dimension: int
    embed_model: str
    nomic_api_url: str
    max_batch_size: int


@dataclass
class IndexingConfig:
    """Indexing and processing configuration"""
    project_name: str
    max_queue_size: int
    max_workers: int
    file_watch_enabled: bool


@dataclass
class RuntimeConfig:
    """Complete runtime configuration"""
    database: DatabaseConfig
    embedding: EmbeddingConfig
    indexing: IndexingConfig
    
    @classmethod
    def from_environment(cls) -> 'RuntimeConfig':
        """Create configuration from environment variables with precedence rules"""
        
        # Neo4j: URI precedence over HOST/PORT
        # Debug: Check what environment variables are actually set
        logger.debug(f"Environment check - NEO4J_URI: {os.environ.get('NEO4J_URI', 'NOT_SET')}")
        logger.debug(f"All NEO4J env vars: {[(k,v) for k,v in os.environ.items() if 'NEO4J' in k]}")
        
        neo4j_uri = os.environ.get('NEO4J_URI')
        if not neo4j_uri:
            neo4j_host = os.environ.get('NEO4J_HOST', 'default-neo4j-graph')
            neo4j_port = int(os.environ.get('NEO4J_PORT', 7687))
            neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"
            logger.warning(f"NEO4J_URI not found, constructed from defaults: {neo4j_uri}")
        else:
            logger.info(f"Using NEO4J_URI from environment: {neo4j_uri}")
        
        # Qdrant: Normalize QDRANT_HTTP_PORT to QDRANT_PORT
        # Debug: Check Qdrant environment variables
        logger.debug(f"Environment check - QDRANT_PORT: {os.environ.get('QDRANT_PORT', 'NOT_SET')}")
        logger.debug(f"Environment check - QDRANT_HOST: {os.environ.get('QDRANT_HOST', 'NOT_SET')}")
        
        qdrant_port = int(
            os.environ.get('QDRANT_PORT', 
                os.environ.get('QDRANT_HTTP_PORT', '6333')
            )
        )
        logger.info(f"Using Qdrant port: {qdrant_port}")
        
        # Embedding: Multiple fallback paths for dimension detection
        embed_dimension = int(
            os.environ.get('EMBED_DIM',
                os.environ.get('EMBEDDING_DIMENSION', '768')
            )
        )
        
        database_config = DatabaseConfig(
            neo4j_uri=neo4j_uri,
            neo4j_username=os.environ.get('NEO4J_USERNAME', 'neo4j'),
            neo4j_password=os.environ.get('NEO4J_PASSWORD', 'graphrag-password'),
            qdrant_host=os.environ.get('QDRANT_HOST', 'localhost'),
            qdrant_port=qdrant_port,
            qdrant_timeout=float(os.environ.get('QDRANT_TIMEOUT', '5.0'))
        )
        
        embedding_config = EmbeddingConfig(
            embed_dimension=embed_dimension,
            embed_model=os.environ.get('EMBED_MODEL', 'nomic-embed-text-v1.5'),
            nomic_api_url=os.environ.get('NOMIC_API_URL', 'http://localhost:8080'),
            max_batch_size=int(os.environ.get('EMBED_BATCH_SIZE', '100'))
        )
        
        indexing_config = IndexingConfig(
            project_name=os.environ.get('PROJECT_NAME', 'default'),
            max_queue_size=int(os.environ.get('MAX_QUEUE_SIZE', '1000')),
            max_workers=int(os.environ.get('MAX_WORKERS', '4')),
            file_watch_enabled=os.environ.get('FILE_WATCH_ENABLED', 'true').lower() == 'true'
        )
        
        config = cls(
            database=database_config,
            embedding=embedding_config,
            indexing=indexing_config
        )
        
        logger.info(f"Runtime configuration initialized: "
                   f"Neo4j={config.database.neo4j_uri}, "
                   f"Qdrant={config.database.qdrant_host}:{config.database.qdrant_port}, "
                   f"EmbedDim={config.embedding.embed_dimension}, "
                   f"Project={config.indexing.project_name}")
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary for debugging"""
        return {
            "database": {
                "neo4j_uri": self.database.neo4j_uri,
                "neo4j_username": self.database.neo4j_username,
                "qdrant_host": self.database.qdrant_host,
                "qdrant_port": self.database.qdrant_port,
                "qdrant_timeout": self.database.qdrant_timeout
            },
            "embedding": {
                "embed_dimension": self.embedding.embed_dimension,
                "embed_model": self.embedding.embed_model,
                "nomic_api_url": self.embedding.nomic_api_url,
                "max_batch_size": self.embedding.max_batch_size
            },
            "indexing": {
                "project_name": self.indexing.project_name,
                "max_queue_size": self.indexing.max_queue_size,
                "max_workers": self.indexing.max_workers,
                "file_watch_enabled": self.indexing.file_watch_enabled
            }
        }


# Global configuration instance
_runtime_config: Optional[RuntimeConfig] = None


def get_runtime_config() -> RuntimeConfig:
    """Get global runtime configuration instance"""
    global _runtime_config
    
    if _runtime_config is None:
        _runtime_config = RuntimeConfig.from_environment()
    
    return _runtime_config


def reload_config() -> RuntimeConfig:
    """Force reload configuration from environment"""
    global _runtime_config
    _runtime_config = RuntimeConfig.from_environment()
    return _runtime_config