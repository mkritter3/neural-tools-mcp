#!/usr/bin/env python3
"""
L9 Neural Tools - Shared Utilities
Common utility functions used across multiple components
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_deterministic_point_id(file_path: str, content: str, chunk_index: int = 0) -> int:
    """Generate deterministic point ID for consistent upserts following industry standards.
    
    This prevents duplicate accumulation by ensuring the same content gets the same ID,
    allowing Qdrant's upsert to properly update existing points instead of creating duplicates.
    
    Args:
        file_path: Path to the source file
        content: Content being indexed  
        chunk_index: Index of the chunk within the file (for multi-chunk files)
    
    Returns:
        Deterministic integer ID that will be the same for identical content
    """
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    unique_string = f"{file_path}#{content_hash}#{chunk_index}"
    return abs(hash(unique_string)) % (10**15)

def get_content_hash(content: str) -> str:
    """Generate content hash for change detection"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    import math
    
    if len(vec1) != len(vec2):
        return 0.0
        
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
        
    return dot_product / (magnitude1 * magnitude2)

def combine_with_rrf_internal(vector_results: List[Dict], text_results: List[Dict], k: int = 60) -> List[Dict]:
    """Combine results using Reciprocal Rank Fusion (RRF) algorithm
    
    RRF Score = 1/(k + rank) for each result, then sum across searches.
    This gives balanced weighting between vector and text search results.
    
    Args:
        vector_results: Results from vector similarity search
        text_results: Results from text/keyword search  
        k: RRF parameter (default 60 is standard)
    
    Returns:
        Combined results sorted by RRF score
    """
    rrf_scores = {}
    
    # Process vector results
    for i, result in enumerate(vector_results):
        point_id = str(result.get('id', ''))
        if point_id:
            rrf_scores[point_id] = rrf_scores.get(point_id, 0) + 1.0 / (k + i + 1)
    
    # Process text results  
    for i, result in enumerate(text_results):
        point_id = str(result.get('id', ''))
        if point_id:
            rrf_scores[point_id] = rrf_scores.get(point_id, 0) + 1.0 / (k + i + 1)
    
    # Combine and sort results
    combined_results = []
    all_results = {str(r.get('id', '')): r for r in vector_results + text_results}
    
    for point_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        if point_id in all_results:
            result = all_results[point_id].copy()
            result['rrf_score'] = rrf_score
            combined_results.append(result)
    
    return combined_results

def apply_mmr_diversity(results: List[Dict], threshold: float, limit: int) -> List[Dict]:
    """Apply Maximal Marginal Relevance for result diversification
    
    Args:
        results: Search results with embeddings
        threshold: Minimum cosine similarity threshold for diversity (0.0-1.0)
        limit: Maximum number of results to return
        
    Returns:
        Diversified results list
    """
    if not results or len(results) <= 1:
        return results[:limit]
    
    selected = [results[0]]  # Always include highest scored result
    remaining = results[1:]
    
    while len(selected) < limit and remaining:
        best_candidate = None
        best_score = -1
        best_idx = -1
        
        for i, candidate in enumerate(remaining):
            # Get embedding for similarity calculation
            candidate_embedding = candidate.get('embedding')
            if not candidate_embedding:
                continue
                
            # Calculate minimum similarity to already selected results
            min_similarity = 1.0
            for selected_result in selected:
                selected_embedding = selected_result.get('embedding')
                if selected_embedding:
                    similarity = cosine_similarity(candidate_embedding, selected_embedding)
                    min_similarity = min(min_similarity, similarity)
            
            # MMR score balances relevance vs diversity
            relevance_score = candidate.get('score', 0)
            diversity_bonus = 1.0 - min_similarity
            
            mmr_score = 0.7 * relevance_score + 0.3 * diversity_bonus
            
            # Only consider if sufficiently different
            if min_similarity < threshold and mmr_score > best_score:
                best_candidate = candidate
                best_score = mmr_score
                best_idx = i
        
        if best_candidate:
            selected.append(best_candidate)
            remaining.pop(best_idx)
        else:
            break
    
    return selected

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for text"""
    # Simple approximation: ~4 characters per token for English text
    return max(1, len(text) // 4)

def format_timestamp() -> str:
    """Get formatted timestamp for logging"""
    return datetime.now().isoformat()