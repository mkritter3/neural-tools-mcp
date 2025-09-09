#!/usr/bin/env python3
"""
Asynchronous Haiku Dynamic Re-ranker (Phase 1.7+)
Fast initial response with progressive enhancement via background re-ranking
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.infrastructure.subscription_haiku_reranker import SubscriptionHaikuReRanker
from src.infrastructure.haiku_reranker import SearchResult, ReRankingRequest

logger = logging.getLogger(__name__)


class ReRankingMode(Enum):
    """Re-ranking execution modes"""
    IMMEDIATE = "immediate"      # Block and return re-ranked results (26s latency)
    PROGRESSIVE = "progressive"  # Return vector results immediately, re-rank in background
    CACHED_ONLY = "cached_only"  # Only use cached re-rankings, no new API calls
    SELECTIVE = "selective"      # Only re-rank if query meets certain criteria


@dataclass
class AsyncReRankingResult:
    """Result container for async re-ranking operations"""
    initial_results: List[Dict[str, Any]]
    reranking_id: str
    mode: ReRankingMode
    initial_response_time: float
    estimated_reranking_time: Optional[float] = None
    reranked_results: Optional[List[Dict[str, Any]]] = None
    reranking_complete: bool = False
    reranking_error: Optional[str] = None


class AsyncHaikuReRanker:
    """
    Asynchronous Haiku re-ranker with progressive enhancement
    
    Features:
    - Immediate vector results (< 100ms)
    - Background re-ranking for improved relevance
    - Configurable thresholds for when to re-rank
    - Real-time status updates via callbacks
    - Smart caching to minimize API calls
    """
    
    def __init__(
        self,
        subscription_reranker: Optional[SubscriptionHaikuReRanker] = None,
        default_mode: ReRankingMode = ReRankingMode.PROGRESSIVE,
        rerank_threshold_results: int = 5,  # Only re-rank if we have 5+ results
        rerank_threshold_confidence: float = 0.7,  # Only re-rank if vector confidence < 0.7
        max_concurrent_rerankings: int = 3
    ):
        """
        Initialize async re-ranker
        
        Args:
            subscription_reranker: Underlying subscription re-ranker
            default_mode: Default re-ranking mode
            rerank_threshold_results: Minimum results to trigger re-ranking
            rerank_threshold_confidence: Vector confidence threshold for re-ranking
            max_concurrent_rerankings: Maximum concurrent re-ranking operations
        """
        self.reranker = subscription_reranker or SubscriptionHaikuReRanker()
        self.default_mode = default_mode
        self.rerank_threshold_results = rerank_threshold_results
        self.rerank_threshold_confidence = rerank_threshold_confidence
        self.max_concurrent_rerankings = max_concurrent_rerankings
        
        # Track active re-ranking operations
        self.active_rerankings: Dict[str, asyncio.Task] = {}
        self.reranking_results: Dict[str, AsyncReRankingResult] = {}
        
        # Performance tracking
        self.stats = {
            'requests': 0,
            'immediate_responses': 0,
            'background_rerankings': 0,
            'cache_hits': 0,
            'skipped_rerankings': 0,
            'avg_initial_response_time': 0.0,
            'total_initial_response_time': 0.0
        }
    
    def _should_rerank(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        mode: ReRankingMode
    ) -> bool:
        """Determine if re-ranking should be performed"""
        if mode == ReRankingMode.CACHED_ONLY:
            # Check if we have a cached result
            search_results = [
                SearchResult(
                    content=r.get('content', str(r)),
                    score=r.get('score', 0.0),
                    metadata=r.get('metadata', {}),
                    source=r.get('source', 'unknown'),
                    original_rank=i + 1
                )
                for i, r in enumerate(results)
            ]
            
            request = ReRankingRequest(
                query=query,
                results=search_results,
                max_results=len(results)
            )
            
            cached_result = self.reranker._check_cache(request)
            return cached_result is not None
        
        if mode == ReRankingMode.IMMEDIATE:
            return True
        
        # Apply thresholds for progressive/selective modes
        if len(results) < self.rerank_threshold_results:
            return False
        
        # Check if vector results have low confidence
        avg_score = sum(r.get('score', 0.0) for r in results) / len(results)
        if avg_score > self.rerank_threshold_confidence:
            return False
        
        return True
    
    async def _background_rerank(
        self,
        reranking_id: str,
        query: str,
        results: List[Dict[str, Any]],
        context: Optional[str] = None,
        max_results: int = 10,
        callback: Optional[Callable] = None
    ):
        """Perform re-ranking in background"""
        try:
            logger.info(f"Starting background re-ranking {reranking_id}")
            
            start_time = time.perf_counter()
            reranked_results = await self.reranker.rerank_simple(
                query=query,
                results=results,
                context=context,
                max_results=max_results
            )
            reranking_time = time.perf_counter() - start_time
            
            # Update result
            if reranking_id in self.reranking_results:
                result = self.reranking_results[reranking_id]
                result.reranked_results = reranked_results
                result.reranking_complete = True
                result.estimated_reranking_time = reranking_time
                
                logger.info(f"Background re-ranking {reranking_id} completed in {reranking_time:.2f}s")
                
                # Call callback if provided
                if callback:
                    try:
                        await callback(result)
                    except Exception as e:
                        logger.error(f"Callback error for {reranking_id}: {e}")
            
            self.stats['background_rerankings'] += 1
            
        except Exception as e:
            logger.error(f"Background re-ranking {reranking_id} failed: {e}")
            
            if reranking_id in self.reranking_results:
                result = self.reranking_results[reranking_id]
                result.reranking_error = str(e)
                result.reranking_complete = True
        
        finally:
            # Clean up
            if reranking_id in self.active_rerankings:
                del self.active_rerankings[reranking_id]
    
    async def search_and_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context: Optional[str] = None,
        max_results: int = 10,
        mode: Optional[ReRankingMode] = None,
        callback: Optional[Callable] = None
    ) -> AsyncReRankingResult:
        """
        Perform search with async re-ranking
        
        Args:
            query: Search query
            results: Initial vector search results
            context: Optional context
            max_results: Maximum results to return
            mode: Re-ranking mode (uses default if None)
            callback: Optional callback for when re-ranking completes
            
        Returns:
            AsyncReRankingResult with immediate results and re-ranking status
        """
        start_time = time.perf_counter()
        self.stats['requests'] += 1
        
        mode = mode or self.default_mode
        reranking_id = str(uuid.uuid4())[:8]
        
        # Limit initial results
        initial_results = results[:max_results]
        initial_response_time = time.perf_counter() - start_time
        
        # Update stats
        self.stats['total_initial_response_time'] += initial_response_time
        self.stats['avg_initial_response_time'] = (
            self.stats['total_initial_response_time'] / self.stats['requests']
        )
        
        # Create result container
        async_result = AsyncReRankingResult(
            initial_results=initial_results,
            reranking_id=reranking_id,
            mode=mode,
            initial_response_time=initial_response_time
        )
        
        self.reranking_results[reranking_id] = async_result
        
        # Determine re-ranking strategy
        should_rerank = self._should_rerank(query, results, mode)
        
        if not should_rerank:
            logger.info(f"Skipping re-ranking for query '{query}' (mode: {mode.value})")
            async_result.reranking_complete = True
            async_result.reranked_results = initial_results
            self.stats['skipped_rerankings'] += 1
            return async_result
        
        if mode == ReRankingMode.IMMEDIATE:
            # Block and return re-ranked results
            logger.info(f"Performing immediate re-ranking for query '{query}'")
            
            rerank_start = time.perf_counter()
            reranked_results = await self.reranker.rerank_simple(
                query=query,
                results=results,
                context=context,
                max_results=max_results
            )
            reranking_time = time.perf_counter() - rerank_start
            
            async_result.reranked_results = reranked_results
            async_result.reranking_complete = True
            async_result.estimated_reranking_time = reranking_time
            
            return async_result
        
        elif mode == ReRankingMode.CACHED_ONLY:
            # Try to get cached result
            logger.info(f"Checking cache for query '{query}'")
            
            search_results = [
                SearchResult(
                    content=r.get('content', str(r)),
                    score=r.get('score', 0.0),
                    metadata=r.get('metadata', {}),
                    source=r.get('source', 'unknown'),
                    original_rank=i + 1
                )
                for i, r in enumerate(results)
            ]
            
            request = ReRankingRequest(
                query=query,
                results=search_results,
                context=context,
                max_results=max_results
            )
            
            cached_result = self.reranker._check_cache(request)
            if cached_result:
                # Convert back to dict format
                cached_dicts = []
                for result in cached_result.results:
                    original_data = next(
                        (r for r in results if r.get('content') == result.content),
                        {}
                    )
                    cached_data = {**original_data}
                    cached_data.update({
                        'content': result.content,
                        'score': result.score,
                        'metadata': {**cached_data.get('metadata', {}), **result.metadata},
                        'reranking_confidence': cached_result.confidence_score
                    })
                    cached_dicts.append(cached_data)
                
                async_result.reranked_results = cached_dicts
                async_result.reranking_complete = True
                async_result.estimated_reranking_time = 0.0
                self.stats['cache_hits'] += 1
            else:
                async_result.reranking_complete = True
                async_result.reranked_results = initial_results
            
            return async_result
        
        else:  # PROGRESSIVE or SELECTIVE
            # Start background re-ranking if we have capacity
            if len(self.active_rerankings) < self.max_concurrent_rerankings:
                logger.info(f"Starting background re-ranking for query '{query}'")
                
                task = asyncio.create_task(
                    self._background_rerank(
                        reranking_id=reranking_id,
                        query=query,
                        results=results,
                        context=context,
                        max_results=max_results,
                        callback=callback
                    )
                )
                
                self.active_rerankings[reranking_id] = task
                
                # Estimate completion time (rough)
                stats = self.reranker.get_stats()
                if stats['avg_processing_time'] > 0:
                    async_result.estimated_reranking_time = stats['avg_processing_time']
                
            else:
                logger.warning(f"Max concurrent re-rankings reached, skipping for query '{query}'")
                async_result.reranking_complete = True
                async_result.reranked_results = initial_results
                self.stats['skipped_rerankings'] += 1
        
        self.stats['immediate_responses'] += 1
        return async_result
    
    async def get_reranking_status(self, reranking_id: str) -> Optional[AsyncReRankingResult]:
        """Get status of a re-ranking operation"""
        return self.reranking_results.get(reranking_id)
    
    async def wait_for_reranking(self, reranking_id: str, timeout: float = 30.0) -> Optional[AsyncReRankingResult]:
        """Wait for a re-ranking operation to complete"""
        if reranking_id not in self.reranking_results:
            return None
        
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout:
            result = self.reranking_results[reranking_id]
            if result.reranking_complete:
                return result
            await asyncio.sleep(0.1)
        
        logger.warning(f"Re-ranking {reranking_id} timed out after {timeout}s")
        return self.reranking_results.get(reranking_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        base_stats = self.reranker.get_stats()
        
        return {
            **self.stats,
            **base_stats,
            'active_rerankings': len(self.active_rerankings),
            'pending_results': len(self.reranking_results)
        }
    
    async def cleanup_completed(self, max_age_seconds: float = 300):
        """Clean up completed re-ranking results older than max_age"""
        current_time = time.perf_counter()
        
        to_remove = []
        for reranking_id, result in self.reranking_results.items():
            if (result.reranking_complete and 
                current_time - result.initial_response_time > max_age_seconds):
                to_remove.append(reranking_id)
        
        for reranking_id in to_remove:
            del self.reranking_results[reranking_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed re-ranking results")


# Convenience functions
async def search_with_progressive_reranking(
    query: str,
    results: List[Dict[str, Any]],
    context: Optional[str] = None,
    max_results: int = 10,
    callback: Optional[Callable] = None
) -> AsyncReRankingResult:
    """
    Convenience function for progressive re-ranking
    Returns immediate results, re-ranks in background
    """
    async_reranker = AsyncHaikuReRanker(default_mode=ReRankingMode.PROGRESSIVE)
    return await async_reranker.search_and_rerank(
        query=query,
        results=results,
        context=context,
        max_results=max_results,
        callback=callback
    )


async def search_with_cached_reranking(
    query: str,
    results: List[Dict[str, Any]],
    context: Optional[str] = None,
    max_results: int = 10
) -> AsyncReRankingResult:
    """
    Convenience function for cached-only re-ranking
    Only uses cached results, no new API calls
    """
    async_reranker = AsyncHaikuReRanker(default_mode=ReRankingMode.CACHED_ONLY)
    return await async_reranker.search_and_rerank(
        query=query,
        results=results,
        context=context,
        max_results=max_results
    )