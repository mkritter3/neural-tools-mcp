#!/usr/bin/env python3
"""
Local Nomic Embedding Service - Open Source Implementation
Uses nomic-ai/nomic-embed-text-v2-moe directly from HuggingFace
Proven patterns from Context7 and 2024 documentation
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available - install with: pip install sentence-transformers torch")

logger = logging.getLogger(__name__)

@dataclass
class NomicEmbedResponse:
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, int]

class LocalNomicService:
    """
    Local Nomic Embedding Service using open-source nomic-embed-text-v2-moe
    
    Uses proven 2024 patterns:
    - SentenceTransformer with trust_remote_code=True
    - Proper prompt_name="passage" for documents 
    - Task prefixes: search_document:, search_query:, clustering:, classification:
    - Matryoshka truncation support (768/256/128 dims)
    - Multilingual support (~100 languages)
    """
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v2-moe"):
        self.model_name = model_name
        self.model = None
        self.initialized = False
        
        # Configuration from environment
        self.embedding_dim = int(os.environ.get('NOMIC_EMBEDDING_DIM', 768))  # 768, 256, or 128
        self.device = 'cuda' if os.environ.get('CUDA_AVAILABLE') == 'true' else 'cpu'
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the local Nomic model using proven 2024 patterns"""
        try:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                return {
                    "success": False,
                    "message": "sentence-transformers library not available. Install with: pip install sentence-transformers torch"
                }
            
            logger.info(f"Loading Nomic model: {self.model_name}")
            
            # Build model kwargs with proven patterns
            model_kwargs = {
                "trust_remote_code": True,  # REQUIRED for Nomic models
                "device": self.device
            }
            
            # Add Matryoshka truncation if requested
            if self.embedding_dim < 768:
                model_kwargs["truncate_dim"] = self.embedding_dim
                logger.info(f"Using Matryoshka truncation to {self.embedding_dim} dimensions")
            
            # Load model with proven sentence-transformers pattern
            self.model = SentenceTransformer(
                self.model_name,
                **model_kwargs
            )
            
            self.initialized = True
            
            # Test with a simple embedding to verify it works
            test_result = self.model.encode(["search_document: test"], prompt_name="passage")
            actual_dim = len(test_result[0]) if hasattr(test_result[0], '__len__') else test_result.shape[-1]
            
            return {
                "success": True,
                "message": f"Local Nomic model loaded successfully",
                "model": self.model_name,
                "device": self.device,
                "embedding_dim": actual_dim,
                "method": "sentence_transformers_2024"
            }
            
        except Exception as e:
            logger.error(f"Local Nomic initialization failed: {e}")
            return {
                "success": False,
                "message": f"Local Nomic initialization failed: {str(e)}"
            }
    
    async def get_embeddings(self, texts: List[str], task_type: str = "search_document") -> List[List[float]]:
        """Generate embeddings using proven 2024 patterns"""
        if not self.initialized:
            raise RuntimeError("Local Nomic service not initialized")
        
        if not texts:
            return []
        
        try:
            # Add task prefix using proven patterns - REQUIRED for Nomic models
            prefixed_texts = []
            for text in texts:
                if not any(text.startswith(prefix) for prefix in [
                    'search_document:', 'search_query:', 'clustering:', 'classification:'
                ]):
                    prefixed_texts.append(f"{task_type}: {text}")
                else:
                    prefixed_texts.append(text)
            
            # Use proven 2024 pattern with prompt_name for documents
            embeddings = self.model.encode(
                prefixed_texts,
                prompt_name="passage",  # Proven pattern for document embeddings
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            # Convert to standard Python list format
            return embeddings.tolist()
                
        except Exception as e:
            logger.error(f"Local Nomic embedding failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health with actual embedding test"""
        if not self.initialized:
            return {"healthy": False, "message": "Service not initialized"}
        
        try:
            # Test actual embedding generation
            test_embedding = await self.get_embeddings(["health check test"])
            return {
                "healthy": True,
                "embedding_dim": len(test_embedding[0]) if test_embedding else 0,
                "model": self.model_name,
                "device": self.device,
                "method": "sentence_transformers_2024",
                "test_embedding_shape": f"({len(test_embedding)}, {len(test_embedding[0]) if test_embedding else 0})"
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

# Compatibility wrapper to match existing NomicService interface
class NomicService(LocalNomicService):
    """Drop-in replacement for the API-based NomicService"""
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Match the existing interface expectation"""
        return await super().get_embeddings(texts, task_type="search_document")