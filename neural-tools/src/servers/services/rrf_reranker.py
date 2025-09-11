"""
Reciprocal Rank Fusion (RRF) Re-ranking System
L9 2025 Architecture - ADR-0022 Implementation

Implements hybrid search with RRF algorithm to merge results from
multiple retrieval methods (vector search, graph traversal, keyword search).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result structure"""
    id: str
    content: str
    file_path: str
    score: float
    source: str  # 'vector', 'graph', 'keyword'
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class RRFReranker:
    """
    Reciprocal Rank Fusion reranker for GraphRAG hybrid search
    Following ADR-0022 specifications with k=60
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF reranker
        
        Args:
            k: RRF parameter (default 60 based on research)
               Higher k gives more weight to top results
        """
        self.k = k
        self.metrics = {
            'total_rerankings': 0,
            'avg_input_sources': 0,
            'avg_output_size': 0,
            'cache_hits': 0
        }
        
        # Simple cache for repeated queries
        self.cache = {}
        self.cache_max_size = 100
    
    def apply_rrf(
        self,
        search_results: List[Tuple[str, List[SearchResult]]],
        top_k: int = 50
    ) -> List[SearchResult]:
        """
        Apply Reciprocal Rank Fusion to merge multiple ranked lists
        
        Args:
            search_results: List of (source_name, ranked_results) tuples
            top_k: Number of top results to return
            
        Returns:
            Merged and re-ranked list of search results
        """
        # Generate cache key
        cache_key = self._generate_cache_key(search_results)
        if cache_key in self.cache:
            self.metrics['cache_hits'] += 1
            logger.debug("RRF cache hit")
            return self.cache[cache_key][:top_k]
        
        # Track unique items and their RRF scores
        item_scores = defaultdict(float)
        item_data = {}
        source_contributions = defaultdict(lambda: defaultdict(float))
        
        # Calculate RRF scores
        for source_name, results in search_results:
            for rank, result in enumerate(results, 1):
                # RRF formula: 1 / (rank + k)
                rrf_score = 1.0 / (rank + self.k)
                
                # Use file_path as unique identifier
                item_id = result.file_path
                
                # Accumulate RRF score
                item_scores[item_id] += rrf_score
                source_contributions[item_id][source_name] = rrf_score
                
                # Store the result data (keep the best version)
                if item_id not in item_data or result.score > item_data[item_id].score:
                    item_data[item_id] = result
        
        # Create merged results with RRF scores
        merged_results = []
        for item_id, total_rrf_score in item_scores.items():
            result = item_data[item_id]
            
            # Create new result with merged information
            merged_result = SearchResult(
                id=result.id,
                content=result.content,
                file_path=result.file_path,
                score=total_rrf_score,  # Use RRF score
                source='hybrid',  # Mark as hybrid result
                metadata=result.metadata or {},
                timestamp=result.timestamp
            )
            
            # Add source contribution information
            merged_result.metadata['rrf_details'] = {
                'original_score': result.score,
                'rrf_score': total_rrf_score,
                'source_contributions': dict(source_contributions[item_id]),
                'num_sources': len(source_contributions[item_id])
            }
            
            merged_results.append(merged_result)
        
        # Sort by RRF score (descending)
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update metrics
        self.metrics['total_rerankings'] += 1
        self.metrics['avg_input_sources'] = (
            (self.metrics['avg_input_sources'] * (self.metrics['total_rerankings'] - 1) + 
             len(search_results)) / self.metrics['total_rerankings']
        )
        self.metrics['avg_output_size'] = (
            (self.metrics['avg_output_size'] * (self.metrics['total_rerankings'] - 1) + 
             min(len(merged_results), top_k)) / self.metrics['total_rerankings']
        )
        
        # Cache the result
        self._update_cache(cache_key, merged_results)
        
        logger.info(
            f"RRF merged {len(search_results)} sources with "
            f"{sum(len(r) for _, r in search_results)} total results "
            f"into {min(len(merged_results), top_k)} final results"
        )
        
        return merged_results[:top_k]
    
    def apply_weighted_rrf(
        self,
        search_results: List[Tuple[str, List[SearchResult], float]],
        top_k: int = 50
    ) -> List[SearchResult]:
        """
        Apply weighted RRF where each source has a different weight
        
        Args:
            search_results: List of (source_name, ranked_results, weight) tuples
            top_k: Number of top results to return
            
        Returns:
            Merged and re-ranked list with weighted RRF scores
        """
        weighted_results = []
        
        for source_name, results, weight in search_results:
            # Apply weight to each result's contribution
            weighted_source_results = []
            for result in results:
                weighted_result = SearchResult(
                    id=result.id,
                    content=result.content,
                    file_path=result.file_path,
                    score=result.score * weight,  # Apply weight
                    source=result.source,
                    metadata=result.metadata,
                    timestamp=result.timestamp
                )
                weighted_source_results.append(weighted_result)
            
            weighted_results.append((f"{source_name}_w{weight}", weighted_source_results))
        
        # Now apply standard RRF
        return self.apply_rrf(weighted_results, top_k)
    
    def apply_with_metadata_boost(
        self,
        search_results: List[Tuple[str, List[SearchResult]]],
        top_k: int = 50,
        boost_config: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """
        Apply RRF with additional metadata-based boosting
        
        Args:
            search_results: List of (source_name, ranked_results) tuples
            top_k: Number of top results to return
            boost_config: Configuration for metadata boosts
            
        Returns:
            Re-ranked results with metadata boosts applied
        """
        # Default boost configuration
        if boost_config is None:
            boost_config = {
                'recency_boost': True,
                'recency_max_boost': 1.3,  # 30% max boost
                'recency_days_threshold': 30,
                'status_penalties': {
                    'archived': 0.2,  # 80% penalty
                    'deprecated': 0.3,  # 70% penalty
                    'experimental': 0.8,  # 20% penalty
                },
                'component_boosts': {
                    'service': 1.1,  # 10% boost for services
                    'model': 1.05,   # 5% boost for models
                    'config': 0.95,  # 5% penalty for configs
                }
            }
        
        # First apply standard RRF
        merged_results = self.apply_rrf(search_results, top_k * 2)  # Get more for re-ranking
        
        # Apply metadata boosts
        for result in merged_results:
            original_score = result.score
            boosted_score = original_score
            
            if result.metadata:
                # Apply recency boost
                if boost_config.get('recency_boost') and 'last_modified' in result.metadata:
                    boost = self._calculate_recency_boost(
                        result.metadata['last_modified'],
                        boost_config['recency_max_boost'],
                        boost_config['recency_days_threshold']
                    )
                    boosted_score *= boost
                    result.metadata['recency_boost'] = boost
                
                # Apply status penalties
                if 'status' in result.metadata:
                    status = result.metadata['status']
                    if status in boost_config['status_penalties']:
                        penalty = boost_config['status_penalties'][status]
                        boosted_score *= penalty
                        result.metadata['status_penalty'] = penalty
                
                # Apply component type boosts
                if 'component_type' in result.metadata:
                    comp_type = result.metadata['component_type']
                    if comp_type in boost_config['component_boosts']:
                        boost = boost_config['component_boosts'][comp_type]
                        boosted_score *= boost
                        result.metadata['component_boost'] = boost
            
            # Update score and track changes
            result.metadata['pre_boost_score'] = original_score
            result.score = boosted_score
        
        # Re-sort after boosting
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        return merged_results[:top_k]
    
    def _calculate_recency_boost(
        self,
        last_modified: str,
        max_boost: float,
        days_threshold: int
    ) -> float:
        """Calculate recency boost based on last modified date"""
        try:
            last_mod_date = datetime.fromisoformat(last_modified)
            days_old = (datetime.now() - last_mod_date).days
            
            if days_old < 0:
                # Future date? No boost
                return 1.0
            elif days_old <= days_threshold:
                # Linear decay from max_boost to 1.0
                boost_range = max_boost - 1.0
                decay_factor = 1.0 - (days_old / days_threshold)
                return 1.0 + (boost_range * decay_factor)
            else:
                # Older than threshold, slight penalty
                return 0.95
                
        except Exception as e:
            logger.debug(f"Could not calculate recency boost: {e}")
            return 1.0
    
    def combine_with_exclusion_manager(
        self,
        search_results: List[Tuple[str, List[SearchResult]]],
        exclusion_manager: Any,  # Type hint avoided to prevent circular import
        top_k: int = 50
    ) -> List[SearchResult]:
        """
        Combine RRF with exclusion manager for comprehensive filtering and ranking
        
        Args:
            search_results: List of (source_name, ranked_results) tuples
            exclusion_manager: Instance of ExclusionManager
            top_k: Number of top results to return
            
        Returns:
            Filtered and re-ranked results
        """
        # First, filter out excluded files from each source
        filtered_results = []
        
        for source_name, results in search_results:
            filtered_source_results = []
            excluded_count = 0
            
            for result in results:
                if not exclusion_manager.should_exclude(result.file_path):
                    # Apply weight penalty for deprioritized files
                    penalty = exclusion_manager.get_weight_penalty(result.file_path)
                    if penalty > 0:
                        result.score *= (1.0 - penalty)
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata['exclusion_penalty'] = penalty
                    
                    filtered_source_results.append(result)
                else:
                    excluded_count += 1
            
            if excluded_count > 0:
                logger.info(f"Excluded {excluded_count} files from {source_name}")
            
            filtered_results.append((source_name, filtered_source_results))
        
        # Apply RRF with metadata boost
        return self.apply_with_metadata_boost(filtered_results, top_k)
    
    def _generate_cache_key(self, search_results: List[Tuple[str, List[SearchResult]]]) -> str:
        """Generate cache key from search results"""
        key_parts = []
        for source_name, results in search_results:
            result_ids = [r.file_path for r in results[:10]]  # Use top 10 for key
            key_parts.append(f"{source_name}:{','.join(result_ids)}")
        return '|'.join(key_parts)
    
    def _update_cache(self, key: str, results: List[SearchResult]):
        """Update cache with size limit"""
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get reranker metrics"""
        return {
            **self.metrics,
            'cache_size': len(self.cache),
            'cache_max_size': self.cache_max_size,
            'k_parameter': self.k
        }
    
    def clear_cache(self):
        """Clear the result cache"""
        self.cache.clear()
        logger.info("RRF cache cleared")