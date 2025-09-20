#!/usr/bin/env python3
"""
Battle-Tested Collection Configuration Management
Implements centralized collection naming and validation patterns from Neo4j GraphRAG
Updated per ADR-0041 to delegate to CollectionNamingManager
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

# CRITICAL: DO NOT REMOVE - Required for imports to work (see ADR-0056)
# This compensates for cross-package imports between services/ and config/
services_dir = Path(__file__).parent
sys.path.insert(0, str(services_dir.parent))  # Allows: from config.collection_naming import ...

# ADR-0041: Import centralized collection naming
from config.collection_naming import collection_naming

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Collection types for different data categories"""
    CODE = "code"
    DOCS = "docs"
    GENERAL = "general"


@dataclass
class CollectionConfig:
    """Collection configuration following Neo4j GraphRAG patterns"""
    name: str
    vector_dimension: int
    id_property_external: str  # Qdrant ID field
    id_property_neo4j: str     # Neo4j ID field
    embedding_model: str
    distance_metric: str = "cosine"
    
    def to_qdrant_config(self) -> Dict[str, Any]:
        """Convert to Qdrant collection configuration"""
        return {
            "collection_name": self.name,
            "vector_size": self.vector_dimension,
            "distance": self.distance_metric
        }
    
    def to_neo4j_config(self) -> Dict[str, Any]:
        """Convert to Neo4j retriever configuration"""
        return {
            "collection_name": self.name,
            "id_property_external": self.id_property_external,
            "id_property_neo4j": self.id_property_neo4j
        }


class CollectionManager:
    """
    Centralized collection management following battle-tested patterns
    Based on Neo4j GraphRAG QdrantNeo4jRetriever and Microsoft GraphRAG vector_store_configs
    """
    
    def __init__(self, project_name: Optional[str] = None, embedding_dimension: Optional[int] = None):
        self.project_name = project_name or os.environ.get('PROJECT_NAME', 'default')
        self.embedding_dimension = (
            embedding_dimension or 
            int(os.environ.get('EMBED_DIM', 
                int(os.environ.get('EMBEDDING_DIMENSION', '768'))
            ))
        )
        self.embedding_model = os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text-v1.5')
        
        # Initialize collection configurations
        self._configs = self._initialize_collection_configs()
        
        logger.info(f"Initialized CollectionManager for project: {self.project_name}, embedding_dim: {self.embedding_dimension}")
    
    def _initialize_collection_configs(self) -> Dict[CollectionType, CollectionConfig]:
        """Initialize collection configurations following GraphRAG patterns
        ADR-0041: Now delegates to CollectionNamingManager for names
        """
        # ADR-0041: Use centralized naming (no _code suffix)
        base_name = collection_naming.get_collection_name(self.project_name)

        configs = {
            CollectionType.CODE: CollectionConfig(
                name=base_name,  # ADR-0041: No _code suffix per user requirement
                vector_dimension=self.embedding_dimension,
                id_property_external="neo4j_id",  # Following Neo4j GraphRAG pattern
                id_property_neo4j="id",
                embedding_model=self.embedding_model
            ),
            CollectionType.DOCS: CollectionConfig(
                name=f"{base_name}-docs",  # Future: separate docs collection
                vector_dimension=self.embedding_dimension,
                id_property_external="neo4j_id",
                id_property_neo4j="id",
                embedding_model=self.embedding_model
            ),
            CollectionType.GENERAL: CollectionConfig(
                name=f"{base_name}-general",  # Future: general collection
                vector_dimension=self.embedding_dimension,
                id_property_external="neo4j_id",
                id_property_neo4j="id",
                embedding_model=self.embedding_model
            )
        }

        return configs
    
    def get_collection_config(self, collection_type: CollectionType) -> CollectionConfig:
        """Get configuration for specific collection type"""
        return self._configs[collection_type]
    
    def get_collection_name(self, collection_type: CollectionType) -> str:
        """Get collection name for specific type"""
        return self._configs[collection_type].name
    
    def get_all_collection_names(self) -> Dict[str, str]:
        """Get all collection names mapped by type"""
        return {
            collection_type.value: config.name 
            for collection_type, config in self._configs.items()
        }
    
    def get_qdrant_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get Qdrant configurations for all collections"""
        return {
            collection_type.value: config.to_qdrant_config()
            for collection_type, config in self._configs.items()
        }
    
    def get_neo4j_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get Neo4j configurations for all collections"""
        return {
            collection_type.value: config.to_neo4j_config()
            for collection_type, config in self._configs.items()
        }
    
    async def validate_collection_exists(self, collection_type: CollectionType, qdrant_client) -> bool:
        """
        Validate that collection exists in Qdrant
        Following Neo4j GraphRAG validation patterns
        """
        try:
            collection_name = self.get_collection_name(collection_type)
            # Get collections using proper Qdrant client (awaiting async method)
            collection_names = await qdrant_client.get_collections()
            
            exists = collection_name in collection_names
            if not exists:
                logger.warning(f"Collection {collection_name} does not exist in Qdrant")
            
            return exists
        except Exception as e:
            logger.error(f"Error validating collection {collection_type.value}: {e}")
            return False
    
    async def ensure_collection_exists(self, collection_type: CollectionType, qdrant_client) -> bool:
        """
        Ensure collection exists, create if missing
        Following Neo4j GraphRAG auto-creation patterns
        """
        try:
            if await self.validate_collection_exists(collection_type, qdrant_client):
                return True
            
            # Create missing collection using QdrantService wrapper
            config = self.get_collection_config(collection_type)
            
            # Use QdrantService wrapper's ensure_collection method
            logger.info(f"Attempting to create collection: {config.name} with vector_size: {config.vector_dimension}")
            
            try:
                from qdrant_client.models import Distance
                
                # Use QdrantService wrapper's ensure_collection method
                success = await qdrant_client.ensure_collection(
                    collection_name=config.name,
                    vector_size=config.vector_dimension,
                    distance=Distance.COSINE
                )
                
                if success:
                    logger.info(f"Successfully ensured collection: {config.name}")
                else:
                    logger.error(f"Failed to ensure collection: {config.name}")
                
                if not success:
                    logger.error(f"Qdrant ensure_collection returned False for {config.name}")
                    return False
                else:
                    logger.info(f"Successfully created collection: {config.name}")
                    return True
                    
            except Exception as e:
                logger.error(f"Qdrant ensure_collection exception for {config.name}: {type(e).__name__}: {e}", exc_info=True)
                return False
            
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_type.value}: {e}")
            return False
    
    async def recreate_collection(self, collection_type: CollectionType, qdrant_client) -> bool:
        """
        Recreate collection (delete and create new)
        Following Microsoft GraphRAG migration patterns
        """
        try:
            config = self.get_collection_config(collection_type)
            
            # Delete if exists
            try:
                await qdrant_client.delete_collection(config.name)
                logger.info(f"Deleted existing collection: {config.name}")
            except Exception as e:
                logger.debug(f"Collection {config.name} didn't exist or couldn't be deleted: {e}")
            
            # Create new
            from qdrant_client import models
            qdrant_client.create_collection(
                collection_name=config.name,
                vectors_config=models.VectorParams(
                    size=config.vector_dimension,
                    distance=models.Distance.COSINE
                )
            )
            
            logger.info(f"Recreated collection: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error recreating collection {collection_type.value}: {e}")
            return False
    
    async def get_collection_stats(self, qdrant_client) -> Dict[str, Any]:
        """
        Get comprehensive statistics for all collections
        Following Neo4j GraphRAG monitoring patterns with enhanced validation
        """
        stats = {}
        
        try:
            # Always assume it's a QdrantService wrapper and use its methods
            collection_names = await qdrant_client.get_collections()
            existing_names = {name: name for name in collection_names}  # Simple mapping for compatibility
            
            for collection_type, config in self._configs.items():
                collection_name = config.name
                
                if collection_name in existing_names:
                    try:
                        # Get collection info using appropriate method
                        if hasattr(qdrant_client, 'get_collection_info'):
                            # This is a QdrantService wrapper - use get_collection_info
                            collection_info = await qdrant_client.get_collection_info(collection_name)
                            # Convert to format expected by the rest of the code
                            if collection_info.get("exists", False):
                                # Create a mock object with the expected structure
                                class MockCollectionInfo:
                                    def __init__(self, info_dict):
                                        self.config = type('', (), {
                                            'params': type('', (), {
                                                'vectors': type('', (), {
                                                    'size': info_dict.get('config', {}).get('vector_size'),
                                                    'distance': type('', (), {'name': info_dict.get('config', {}).get('distance', 'COSINE')})()
                                                })()
                                            })()
                                        })()
                                        self.vectors_count = info_dict.get('vectors_count', 0)
                                        self.indexed_vectors_count = info_dict.get('indexed_vectors_count', 0)
                                        self.points_count = info_dict.get('points_count', 0)
                                        self.status = info_dict.get('status', 'unknown')
                                collection_info = MockCollectionInfo(collection_info)
                            else:
                                continue  # Skip if collection doesn't exist
                        else:
                            # This is a raw QdrantClient - get_collection is NOT async
                            collection_info = qdrant_client.get_collection(collection_name)
                        
                        # Validate vector dimensions match configuration
                        expected_dim = config.vector_dimension
                        actual_config = collection_info.config.params.vectors
                        dimension_match = actual_config.size == expected_dim if actual_config else False
                        
                        stats[collection_type.value] = {
                            "name": collection_name,
                            "exists": True,
                            "vectors_count": collection_info.vectors_count,
                            "indexed_vectors_count": collection_info.indexed_vectors_count,
                            "points_count": collection_info.points_count,
                            "status": collection_info.status,
                            "validation": {
                                "dimension_match": dimension_match,
                                "expected_dimension": expected_dim,
                                "actual_dimension": actual_config.size if actual_config else None,
                                "distance_metric": actual_config.distance.name if actual_config else None
                            }
                        }
                    except Exception as e:
                        stats[collection_type.value] = {
                            "name": collection_name,
                            "exists": True,
                            "error": str(e),
                            "validation": {"status": "error"}
                        }
                else:
                    stats[collection_type.value] = {
                        "name": collection_name,
                        "exists": False,
                        "validation": {"status": "missing"}
                    }
                    
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    async def validate_vector_store_integrity(self, qdrant_client, neo4j_client = None) -> Dict[str, Any]:
        """
        Comprehensive vector store validation following Microsoft GraphRAG patterns
        Validates collection configuration, vector dimensions, and Neo4j-Qdrant consistency
        """
        validation_results = {
            "timestamp": self._get_timestamp(),
            "project_name": self.project_name,
            "collections": {},
            "cross_validation": {},
            "recommendations": []
        }
        
        try:
            # Step 1: Validate each collection configuration
            for collection_type, config in self._configs.items():
                collection_name = config.name
                collection_validation = await self._validate_single_collection(
                    collection_type, config, qdrant_client
                )
                validation_results["collections"][collection_type.value] = collection_validation
                
                # Collect recommendations
                if not collection_validation["exists"]:
                    validation_results["recommendations"].append(
                        f"Create missing collection: {collection_name}"
                    )
                elif not collection_validation["configuration_valid"]:
                    validation_results["recommendations"].append(
                        f"Fix configuration for collection: {collection_name}"
                    )
            
            # Step 2: Cross-validate Neo4j and Qdrant consistency if Neo4j available
            if neo4j_client:
                cross_validation = await self._validate_neo4j_qdrant_consistency(
                    qdrant_client, neo4j_client
                )
                validation_results["cross_validation"] = cross_validation
                
                if cross_validation.get("inconsistencies"):
                    validation_results["recommendations"].append(
                        "Fix Neo4j-Qdrant ID mapping inconsistencies"
                    )
            
            # Step 3: Overall health assessment
            all_valid = all(
                coll.get("exists", False) and coll.get("configuration_valid", False)
                for coll in validation_results["collections"].values()
            )
            
            validation_results["overall_status"] = "healthy" if all_valid else "needs_attention"
            
        except Exception as e:
            logger.error(f"Vector store validation error: {e}")
            validation_results["error"] = str(e)
            validation_results["overall_status"] = "error"
        
        return validation_results
    
    async def _validate_single_collection(self, collection_type: CollectionType, config: CollectionConfig, qdrant_client) -> Dict[str, Any]:
        """Validate a single collection against its configuration"""
        result = {
            "name": config.name,
            "type": collection_type.value,
            "exists": False,
            "configuration_valid": False,
            "issues": []
        }
        
        try:
            # Check if collection exists using async wrapper method
            collection_info_dict = await qdrant_client.get_collection_info(config.name)
            if collection_info_dict.get("exists", False):
                result["exists"] = True
                # Get raw collection info for detailed validation
                collection_info = await qdrant_client.get_collection(config.name)
            else:
                return result
            
            # Validate vector configuration
            actual_config = collection_info.config.params.vectors
            if actual_config:
                # Check dimension
                if actual_config.size != config.vector_dimension:
                    result["issues"].append(
                        f"Vector dimension mismatch: expected {config.vector_dimension}, got {actual_config.size}"
                    )
                
                # Check distance metric
                expected_distance = config.distance_metric.upper()
                actual_distance = actual_config.distance.name.upper()
                if actual_distance != expected_distance:
                    result["issues"].append(
                        f"Distance metric mismatch: expected {expected_distance}, got {actual_distance}"
                    )
                
                result["configuration_valid"] = len(result["issues"]) == 0
            else:
                result["issues"].append("Unable to retrieve vector configuration")
                
        except Exception as e:
            logger.debug(f"Collection {config.name} validation error: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _validate_neo4j_qdrant_consistency(self, qdrant_client, neo4j_client) -> Dict[str, Any]:
        """
        Validate consistency between Neo4j CodeChunk nodes and Qdrant vectors
        Following Neo4j GraphRAG QdrantNeo4jRetriever patterns
        """
        consistency_results = {
            "neo4j_chunks": 0,
            "neo4j_with_collection_name": 0,
            "qdrant_vectors": 0,
            "inconsistencies": [],
            "recommendations": []
        }
        
        try:
            code_collection_name = self.get_collection_name(CollectionType.CODE)
            
            # Query Neo4j for CodeChunk statistics
            neo4j_queries = {
                "total_chunks": "MATCH (c:CodeChunk) RETURN count(c) as count",
                "chunks_with_collection": f"""
                    MATCH (c:CodeChunk) 
                    WHERE c.collection_name = '{code_collection_name}'
                    RETURN count(c) as count
                """,
                "chunks_without_collection": "MATCH (c:CodeChunk) WHERE c.collection_name IS NULL RETURN count(c) as count"
            }
            
            for query_name, query in neo4j_queries.items():
                try:
                    result = neo4j_client.execute_cypher(query)
                    count = result[0]['count'] if result else 0
                    
                    if query_name == "total_chunks":
                        consistency_results["neo4j_chunks"] = count
                    elif query_name == "chunks_with_collection":
                        consistency_results["neo4j_with_collection_name"] = count
                        
                except Exception as e:
                    logger.error(f"Neo4j query {query_name} failed: {e}")
            
            # Query Qdrant for vector count
            try:
                # Use QdrantService wrapper method instead of raw client
                collection_info = await qdrant_client.get_collection_info(code_collection_name)
                consistency_results["qdrant_vectors"] = collection_info.get("vectors_count", 0)
            except Exception as e:
                logger.error(f"Qdrant collection info failed: {e}")
            
            # Analyze inconsistencies
            neo4j_total = consistency_results["neo4j_chunks"]
            neo4j_with_collection = consistency_results["neo4j_with_collection_name"]
            qdrant_total = consistency_results["qdrant_vectors"]
            
            if neo4j_total != neo4j_with_collection:
                consistency_results["inconsistencies"].append(
                    f"Neo4j has {neo4j_total - neo4j_with_collection} CodeChunk nodes without collection_name"
                )
                consistency_results["recommendations"].append(
                    "Run re-indexing to fix missing collection_name values"
                )
            
            if abs(neo4j_with_collection - qdrant_total) > 5:  # Allow small variance
                consistency_results["inconsistencies"].append(
                    f"Count mismatch: Neo4j has {neo4j_with_collection} chunks with collection_name, Qdrant has {qdrant_total} vectors"
                )
                consistency_results["recommendations"].append(
                    "Check for orphaned vectors or missing embeddings"
                )
            
        except Exception as e:
            logger.error(f"Neo4j-Qdrant consistency validation error: {e}")
            consistency_results["error"] = str(e)
        
        return consistency_results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.utcnow().isoformat()


# Global instance following singleton pattern from Microsoft GraphRAG
_collection_manager: Optional[CollectionManager] = None


def get_collection_manager(project_name: Optional[str] = None, embedding_dimension: Optional[int] = None) -> CollectionManager:
    """Get global collection manager instance"""
    global _collection_manager
    
    if _collection_manager is None:
        _collection_manager = CollectionManager(project_name, embedding_dimension)
    elif project_name and _collection_manager.project_name != project_name:
        # Reinitialize if project name changed
        _collection_manager = CollectionManager(project_name, embedding_dimension)
    elif embedding_dimension and _collection_manager.embedding_dimension != embedding_dimension:
        # Reinitialize if embedding dimension changed
        _collection_manager = CollectionManager(project_name, embedding_dimension)
    
    return _collection_manager


# Convenience functions for backward compatibility
def get_code_collection_name(project_name: Optional[str] = None) -> str:
    """Get code collection name"""
    manager = get_collection_manager(project_name)
    return manager.get_collection_name(CollectionType.CODE)


def get_docs_collection_name(project_name: Optional[str] = None) -> str:
    """Get docs collection name"""
    manager = get_collection_manager(project_name)
    return manager.get_collection_name(CollectionType.DOCS)


def get_general_collection_name(project_name: Optional[str] = None) -> str:
    """Get general collection name"""
    manager = get_collection_manager(project_name)
    return manager.get_collection_name(CollectionType.GENERAL)