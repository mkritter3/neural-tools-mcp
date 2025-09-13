#!/usr/bin/env python3
"""
Enhanced Hybrid Retriever with Haiku Re-ranking (Phase 1.7)
Integrates HaikuReRanker with existing HybridRetriever for improved search relevance

Phase 2.1: Added OpenTelemetry instrumentation for production observability
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from src.servers.services.hybrid_retriever import HybridRetriever
from src.infrastructure.async_haiku_reranker import AsyncHaikuReRanker, ReRankingMode
from src.infrastructure.cross_encoder_reranker import CrossEncoderReranker, RerankConfig
from src.infrastructure.telemetry import get_telemetry

logger = logging.getLogger(__name__)
telemetry = get_telemetry()


class EnhancedHybridRetriever:
    """
    Enhanced hybrid retriever with Haiku-based re-ranking
    
    Features:
    - Integrates with existing HybridRetriever
    - Adds intelligent re-ranking using Haiku model
    - Maintains backward compatibility
    - Configurable re-ranking thresholds
    """
    
    def __init__(
        self, 
        hybrid_retriever: HybridRetriever,
        anthropic_api_key: Optional[str] = None,
        enable_reranking: bool = True,
        rerank_threshold: int = 3,  # Only re-rank if more than N results
        reranking_mode: ReRankingMode = ReRankingMode.PROGRESSIVE,
        prefer_local: bool = True,
        rerank_latency_budget_ms: int = 120,
        allow_haiku_fallback: bool = False,
    ):
        """
        Initialize enhanced hybrid retriever
        
        Args:
            hybrid_retriever: Existing HybridRetriever instance
            anthropic_api_key: API key for Haiku (None for mock mode)
            enable_reranking: Whether to enable re-ranking
            rerank_threshold: Minimum results needed to trigger re-ranking
            reranking_mode: Re-ranking execution mode (PROGRESSIVE, IMMEDIATE, CACHED_ONLY)
        """
        self.hybrid_retriever = hybrid_retriever
        self.async_reranker = AsyncHaikuReRanker(default_mode=reranking_mode)
        self.enable_reranking = enable_reranking
        self.rerank_threshold = rerank_threshold
        self.reranking_mode = reranking_mode
        self.prefer_local = prefer_local
        self.rerank_latency_budget_ms = rerank_latency_budget_ms
        self.allow_haiku_fallback = allow_haiku_fallback

        # Local cross-encoder reranker (optional, preferred)
        tenant_id = getattr(getattr(hybrid_retriever, "container", None), "tenant_id", None)
        try:
            cfg = RerankConfig(latency_budget_ms=rerank_latency_budget_ms)
            self.local_reranker = CrossEncoderReranker(cfg, tenant_id=tenant_id) if prefer_local else None
        except Exception as e:
            logger.warning(f"Failed to initialize local reranker, will use fallback: {e}")
            self.local_reranker = None
        
        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'reranked_queries': 0,
            'avg_improvement': 0.0,
            'processing_times': []
        }
    
    async def find_similar_with_context(
        self,
        query: str,
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2,
        enable_reranking: Optional[bool] = None,
        rerank_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar results with graph context and re-ranking
        
        Args:
            query: Search query
            limit: Maximum results to return
            include_graph_context: Whether to include graph relationships
            max_hops: Maximum graph traversal hops
            enable_reranking: Override global re-ranking setting
            rerank_context: Additional context for re-ranking
            
        Returns:
            List of enhanced search results with re-ranking
        """
        start_time = time.time()
        
        # Main operation span
        with telemetry.trace_operation("hybrid_retrieval", {
            "query_length": len(query),
            "limit": limit,
            "include_graph_context": include_graph_context,
            "max_hops": max_hops,
            "reranking_enabled": self.enable_reranking
        }) as main_span:
            
            self.stats['total_queries'] += 1
            
            # Get initial results from hybrid retriever
            with telemetry.trace_operation("initial_retrieval", {
                "requested_limit": min(limit * 2, 20)
            }):
                initial_start = time.time()
                initial_results = await self.hybrid_retriever.find_similar_with_context(
                    query=query,
                    limit=min(limit * 2, 20),  # Get more results for re-ranking
                    include_graph_context=include_graph_context,
                    max_hops=max_hops
                )
                
                initial_duration = time.time() - initial_start
                telemetry.record_retrieval("hybrid", len(initial_results), initial_duration)
                
                if main_span:
                    main_span.set_attribute("initial_results_count", len(initial_results))
            
            # Check if re-ranking should be applied
            should_rerank = (
                (enable_reranking if enable_reranking is not None else self.enable_reranking) and
                len(initial_results) >= self.rerank_threshold
            )
            
            if main_span:
                main_span.set_attribute("should_rerank", should_rerank)
            
            if not should_rerank:
                query_duration = time.time() - start_time
                telemetry.record_query(query, "default", query_duration, True)
                return initial_results[:limit]
            
            try:
                # Prefer local cross-encoder rerank when available
                self.stats['reranked_queries'] += 1
                
                if self.prefer_local and self.local_reranker is not None:
                    with telemetry.trace_operation("local_cross_encoder_rerank", {
                        "initial_count": len(initial_results),
                        "target_limit": limit,
                        "budget_ms": self.rerank_latency_budget_ms
                    }) as rerank_span:
                        
                        rerank_start = time.time()
                        # Use a slightly larger candidate set, then cut to limit
                        candidates = initial_results[: max(limit * 3, min(50, len(initial_results)))]
                        reranked_local = await self.local_reranker.rerank(
                            query=query,
                            results=candidates,
                            top_k=limit,
                            latency_budget_ms=self.rerank_latency_budget_ms,
                        )
                        
                        rerank_duration = time.time() - rerank_start
                        telemetry.record_rerank("local_cross_encoder", len(candidates), rerank_duration)
                        
                        if rerank_span:
                            rerank_span.set_attribute("output_count", len(reranked_local))
                            rerank_span.set_attribute("duration_ms", rerank_duration * 1000)
                        
                        for i, r in enumerate(reranked_local):
                            r.setdefault('metadata', {})
                            r['metadata'].update({
                                'enhanced_hybrid_retrieval': True,
                                'reranked': True,
                                'rerank_position': i + 1,
                                'reranking_id': 'local-ce',
                                'reranking_mode': 'local_cross_encoder'
                            })
                        
                        query_duration = time.time() - start_time
                        telemetry.record_query(query, "default", query_duration, True)
                        return reranked_local

                # Otherwise, apply Async Haiku re-ranking (only if allowed)
                if not self.allow_haiku_fallback:
                    query_duration = time.time() - start_time
                    telemetry.record_query(query, "default", query_duration, True)
                    return initial_results[:limit]

                with telemetry.trace_operation("haiku_rerank_fallback", {
                    "mode": self.reranking_mode.value,
                    "candidate_count": len(initial_results)
                }) as haiku_span:
                    
                    # Prepare context for re-ranking
                    context_parts = []
                    if rerank_context:
                        context_parts.append(rerank_context)
                    
                    # Add graph context if available
                    if include_graph_context and initial_results:
                        graph_context = self._extract_graph_context(initial_results)
                        if graph_context:
                            context_parts.append(f"Graph context: {graph_context}")
                    
                    context = " | ".join(context_parts) if context_parts else None
                    
                    haiku_start = time.time()
                    # Re-rank using Async Haiku (fast initial response)
                    async_result = await self.async_reranker.search_and_rerank(
                        query=query,
                        results=initial_results,
                        context=context,
                        max_results=limit,
                        mode=self.reranking_mode
                    )
                    
                    haiku_duration = time.time() - haiku_start
                    telemetry.record_rerank("haiku_async", len(initial_results), haiku_duration)
                    
                    # Use initial results immediately (progressive enhancement)
                    results_to_return = async_result.initial_results
                    if async_result.reranking_complete and async_result.reranked_results:
                        results_to_return = async_result.reranked_results
                    
                    if haiku_span:
                        haiku_span.set_attribute("reranking_complete", async_result.reranking_complete)
                        haiku_span.set_attribute("output_count", len(results_to_return))
                    
                    # Add enhanced metadata
                    for i, result in enumerate(results_to_return):
                        result.setdefault('metadata', {})
                        result['metadata'].update({
                            'enhanced_hybrid_retrieval': True,
                            'reranked': async_result.reranking_complete,
                            'rerank_position': i + 1,
                            'reranking_id': async_result.reranking_id,
                            'reranking_mode': async_result.mode.value,
                            'initial_response_time': async_result.initial_response_time
                        })
                    
                    query_duration = time.time() - start_time
                    telemetry.record_query(query, "default", query_duration, True)
                    return results_to_return
                    
            except Exception as e:
                # Record error in telemetry
                telemetry.record_error(type(e).__name__, "enhanced_hybrid_retriever")
                
                if main_span:
                    main_span.set_attribute("error", True)
                    main_span.set_attribute("error_type", type(e).__name__)
                    main_span.set_attribute("error_message", str(e))
                
                logger.warning(f"Re-ranking failed, returning original results: {e}")
                
                query_duration = time.time() - start_time
                telemetry.record_query(query, "default", query_duration, False)
                return initial_results[:limit]
    
    def _extract_graph_context(self, results: List[Dict[str, Any]]) -> Optional[str]:
        """Extract meaningful graph context from results"""
        try:
            graph_info = []
            
            for result in results:
                metadata = result.get('metadata', {})
                
                # Extract file relationships
                if 'file_relationships' in metadata:
                    relationships = metadata['file_relationships']
                    if relationships:
                        graph_info.append(f"Related files: {', '.join(relationships[:3])}")
                
                # Extract code dependencies
                if 'dependencies' in metadata:
                    deps = metadata['dependencies']
                    if deps:
                        graph_info.append(f"Dependencies: {', '.join(deps[:3])}")
                
                # Limit context length
                if len(graph_info) >= 3:
                    break
            
            return " | ".join(graph_info) if graph_info else None
            
        except Exception as e:
            logger.debug(f"Failed to extract graph context: {e}")
            return None
    
    async def search_code_chunks(
        self,
        query: str,
        limit: int = 10,
        file_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        enable_reranking: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for code chunks with optional filtering and re-ranking
        
        Args:
            query: Search query
            limit: Maximum results
            file_filter: Optional file path filter
            language_filter: Optional programming language filter
            enable_reranking: Override re-ranking setting
            
        Returns:
            List of code chunks with enhanced ranking
        """
        # Use hybrid retriever's search if available
        if hasattr(self.hybrid_retriever, 'search_code_chunks'):
            initial_results = await self.hybrid_retriever.search_code_chunks(
                query=query,
                limit=min(limit * 2, 20),
                file_filter=file_filter,
                language_filter=language_filter
            )
        else:
            # Fallback to basic similarity search
            initial_results = await self.hybrid_retriever.find_similar_with_context(
                query=query,
                limit=min(limit * 2, 20),
                include_graph_context=False
            )
        
        # Apply filters if needed
        if file_filter or language_filter:
            initial_results = self._apply_filters(initial_results, file_filter, language_filter)
        
        # Re-rank if enabled
        should_rerank = (
            (enable_reranking if enable_reranking is not None else self.enable_reranking) and
            len(initial_results) >= self.rerank_threshold
        )
        
        if should_rerank:
            try:
                # Create context for code search
                context_parts = []
                if language_filter:
                    context_parts.append(f"Language: {language_filter}")
                if file_filter:
                    context_parts.append(f"File pattern: {file_filter}")
                context_parts.append("Focus on code quality and semantic relevance")
                
                context = " | ".join(context_parts)
                
                reranked_results = await self.reranker.rerank_simple(
                    query=query,
                    results=initial_results,
                    context=context,
                    max_results=limit
                )
                
                return reranked_results
                
            except Exception as e:
                logger.warning(f"Code chunk re-ranking failed: {e}")
        
        return initial_results[:limit]
    
    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        file_filter: Optional[str],
        language_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Apply file and language filters to results"""
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Apply file filter
            if file_filter:
                file_path = metadata.get('file_path', '')
                if file_filter not in file_path:
                    continue
            
            # Apply language filter
            if language_filter:
                language = metadata.get('language', '').lower()
                if language_filter.lower() not in language:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    async def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Dict[str, Any]]],
        context: Optional[str] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch re-rank multiple query results
        
        Args:
            queries: List of search queries
            results_list: List of result lists for each query
            context: Optional shared context
            
        Returns:
            List of re-ranked result lists
        """
        if len(queries) != len(results_list):
            raise ValueError("Queries and results lists must have same length")
        
        # Process in parallel for better performance
        tasks = []
        for query, results in zip(queries, results_list):
            if len(results) >= self.rerank_threshold:
                task = self.async_reranker.search_and_rerank(
                    query=query,
                    results=results,
                    context=context,
                    max_results=len(results),
                    mode=ReRankingMode.IMMEDIATE  # Use immediate mode for batch processing
                )
                tasks.append(task)
            else:
                # Create a dummy task that returns original results
                tasks.append(asyncio.coroutine(lambda r=results: r)())
        
        reranked_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and extract results
        final_results = []
        for i, result in enumerate(reranked_lists):
            if isinstance(result, Exception):
                logger.warning(f"Batch re-ranking failed for query {i}: {result}")
                final_results.append(results_list[i])
            else:
                # Extract results from AsyncReRankingResult
                if hasattr(result, 'reranked_results') and result.reranked_results:
                    final_results.append(result.reranked_results)
                elif hasattr(result, 'initial_results'):
                    final_results.append(result.initial_results)
                else:
                    # Fallback for direct list results
                    final_results.append(result)
        
        return final_results
    
    async def get_reranking_status(self, reranking_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a background re-ranking operation"""
        result = await self.async_reranker.get_reranking_status(reranking_id)
        if result:
            return {
                'reranking_id': result.reranking_id,
                'mode': result.mode.value,
                'reranking_complete': result.reranking_complete,
                'initial_response_time': result.initial_response_time,
                'estimated_reranking_time': result.estimated_reranking_time,
                'reranking_error': result.reranking_error,
                'has_reranked_results': result.reranked_results is not None
            }
        return None
    
    async def wait_for_reranking(self, reranking_id: str, timeout: float = 30.0) -> Optional[List[Dict[str, Any]]]:
        """Wait for a re-ranking operation to complete and return final results"""
        result = await self.async_reranker.wait_for_reranking(reranking_id, timeout)
        if result and result.reranked_results:
            return result.reranked_results
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        rerank_rate = (
            (self.stats['reranked_queries'] / max(self.stats['total_queries'], 1)) * 100
        )
        
        async_stats = self.async_reranker.get_stats()
        
        return {
            **self.stats,
            'rerank_rate': rerank_rate,
            'async_reranker_stats': async_stats,
            'rerank_threshold': self.rerank_threshold,
            'reranking_enabled': self.enable_reranking,
            'reranking_mode': self.reranking_mode.value
        }
    
    def configure_reranking(
        self,
        enable: bool,
        threshold: Optional[int] = None,
        cache_ttl: Optional[int] = None
    ):
        """
        Configure re-ranking parameters
        
        Args:
            enable: Whether to enable re-ranking
            threshold: Minimum results for re-ranking
            cache_ttl: Cache time-to-live in seconds
        """
        self.enable_reranking = enable
        
        if threshold is not None:
            self.rerank_threshold = threshold
        
        if cache_ttl is not None:
            self.reranker.cache_ttl = cache_ttl
            
        logger.info(f"Re-ranking configured: enabled={enable}, threshold={self.rerank_threshold}")
    
    # Delegate other methods to underlying hybrid retriever
    def __getattr__(self, name):
        """Delegate unknown methods to hybrid retriever"""
        return getattr(self.hybrid_retriever, name)


# Convenience function to wrap existing HybridRetriever
def enhance_hybrid_retriever(
    hybrid_retriever: HybridRetriever,
    anthropic_api_key: Optional[str] = None,
    **kwargs
) -> EnhancedHybridRetriever:
    """
    Enhance an existing HybridRetriever with Haiku re-ranking
    
    Args:
        hybrid_retriever: Existing HybridRetriever to enhance
        anthropic_api_key: Anthropic API key
        **kwargs: Additional configuration options
        
    Returns:
        EnhancedHybridRetriever instance
    """
    return EnhancedHybridRetriever(
        hybrid_retriever=hybrid_retriever,
        anthropic_api_key=anthropic_api_key,
        **kwargs
    )
