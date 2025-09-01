#!/usr/bin/env python3
"""
L9 Neural Tools - Centralized Configuration Management
Consolidates all environment variables and feature flags in one location
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralConfig:
    """Centralized configuration for all neural tools"""
    
    def __init__(self):
        # Project configuration
        self.PROJECT_NAME = os.environ.get("PROJECT_NAME", "default")
        self.PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/app/project"))
        self.COLLECTION_PREFIX = f"project_{self.PROJECT_NAME}_"
        
        # Service endpoints
        self.QDRANT_HOST = os.environ.get("QDRANT_HOST", "neural-data-storage")
        self.QDRANT_HTTP_PORT = int(os.environ.get("QDRANT_HTTP_PORT", "6333"))
        self.QDRANT_URL = f"http://{self.QDRANT_HOST}:{self.QDRANT_HTTP_PORT}"
        
        self.NEO4J_HOST = os.environ.get("NEO4J_HOST", "neo4j-graph")
        self.NEO4J_PORT = int(os.environ.get("NEO4J_PORT", "7687"))
        self.NEO4J_URI = f"bolt://{self.NEO4J_HOST}:{self.NEO4J_PORT}"
        self.NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
        self.NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neural-l9-2025")
        
        self.NOMIC_HOST = os.environ.get("NOMIC_HOST", "neural-embeddings")
        self.NOMIC_PORT = int(os.environ.get("NOMIC_PORT", "8000"))
        self.NOMIC_BASE_URL = f"http://{self.NOMIC_HOST}:{self.NOMIC_PORT}"
        self.NOMIC_ENDPOINT = f"{self.NOMIC_BASE_URL}/embed"
        
        # Feature flags - determined at initialization
        self.features = self._initialize_features()
        
    def _initialize_features(self) -> Dict[str, bool]:
        """Initialize feature availability flags"""
        features = {
            "neo4j": False,
            "prism": False, 
            "tree_sitter": False
        }
        
        # Check Neo4j availability
        try:
            from neo4j_client import Neo4jGraphRAGClient, AsyncNeo4jClient
            features["neo4j"] = True
            logger.info("✅ Neo4j GraphRAG available")
        except ImportError:
            logger.warning("❌ Neo4j client not available - GraphRAG features disabled")
            
        # Check PRISM availability
        try:
            import sys
            sys.path.append('/app/project/neural-tools')
            from prism_scorer import PrismScorer
            features["prism"] = True
            logger.info("✅ PRISM scorer available")
        except ImportError:
            logger.warning("❌ PRISM scorer not available - using basic scoring")
            
        # Check Tree-sitter availability
        try:
            from tree_sitter_ast import TreeSitterAnalyzer
            features["tree_sitter"] = True
            logger.info("✅ Tree-sitter available")
        except ImportError:
            logger.warning("❌ Tree-sitter not available - code analysis limited")
            
        return features
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is available"""
        return self.features.get(feature, False)
        
    def get_service_info(self) -> Dict[str, Any]:
        """Get service endpoint information"""
        return {
            "qdrant": {"url": self.QDRANT_URL, "healthy": False},
            "neo4j": {"url": self.NEO4J_URI, "healthy": False},
            "embeddings": {"url": self.NOMIC_BASE_URL, "healthy": False}
        }

# Global configuration instance
config = NeuralConfig()