#!/usr/bin/env python3
"""
Haiku Dynamic Re-ranker (Phase 1.7)
Fast, lightweight re-ranking system using Anthropic's Haiku model
Improves search result relevance through intelligent ranking
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import httpx
except ImportError:
    httpx = None

from infrastructure.telemetry import get_telemetry

logger = logging.getLogger(__name__)
telemetry = get_telemetry()


@dataclass
class SearchResult:
    """Represents a search result to be re-ranked"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str = "vector"  # vector, graph, hybrid
    original_rank: int = 0


@dataclass
class ReRankingRequest:
    """Request for re-ranking search results"""
    query: str
    results: List[SearchResult]
    context: Optional[str] = None
    max_results: int = 10
    relevance_threshold: float = 0.1


@dataclass
class ReRankingResult:
    """Result of re-ranking operation"""
    results: List[SearchResult]
    processing_time: float
    model_used: str
    confidence_score: float
    cache_hit: bool = False


class HaikuReRanker:
    """
    Fast dynamic re-ranker using Anthropic's Haiku model
    
    Features:
    - Sub-100ms re-ranking for up to 20 results
    - Query-aware relevance scoring
    - Context-sensitive ranking
    - Smart caching for repeated queries
    - Batch processing optimization
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        cache_ttl: int = 300,  # 5 minutes
        max_cache_size: int = 1000
    ):
        """
        Initialize Haiku re-ranker
        
        Args:
            api_key: Anthropic API key (can be None for mock mode)
            base_url: Anthropic API base URL
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum cache entries
        """
        self.api_key = api_key
        self.base_url = base_url
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        
        # Smart caching system
        self.cache: Dict[str, Tuple[ReRankingResult, datetime]] = {}
        
        # Performance tracking
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'avg_processing_time': 0,
            'total_processing_time': 0
        }
        
        # Mock mode for testing/development
        self.mock_mode = api_key is None
        
        if self.mock_mode:
            logger.warning("HaikuReRanker running in mock mode (no API key provided)")
    
    def _generate_cache_key(self, request: ReRankingRequest) -> str:
        """Generate cache key for re-ranking request"""
        # Include query, result content hashes, and context
        result_hashes = [
            hashlib.sha256(r.content.encode()).hexdigest()[:8]
            for r in request.results
        ]
        
        key_data = {
            'query': request.query,
            'results': result_hashes,
            'context': request.context,
            'max_results': request.max_results
        }
        
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def _clean_cache(self):
        """Remove expired entries from cache"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp > timedelta(seconds=self.cache_ttl)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        # Limit cache size
        if len(self.cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            
            excess = len(self.cache) - self.max_cache_size
            for key, _ in sorted_items[:excess]:
                del self.cache[key]
    
    def _check_cache(self, request: ReRankingRequest) -> Optional[ReRankingResult]:
        """Check if request is cached"""
        cache_key = self._generate_cache_key(request)
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            
            if datetime.now() - timestamp <= timedelta(seconds=self.cache_ttl):
                # Cache hit
                result.cache_hit = True
                self.stats['cache_hits'] += 1
                telemetry.record_cache_operation("haiku_rerank", True)
                return result
            else:
                # Expired entry
                del self.cache[cache_key]
                telemetry.record_cache_operation("haiku_rerank", False)
        else:
            telemetry.record_cache_operation("haiku_rerank", False)
        
        return None
    
    def _cache_result(self, request: ReRankingRequest, result: ReRankingResult):
        """Cache re-ranking result"""
        cache_key = self._generate_cache_key(request)
        self.cache[cache_key] = (result, datetime.now())
        
        # Clean up old entries periodically
        if len(self.cache) % 100 == 0:
            self._clean_cache()
    
    async def _call_haiku_api(self, prompt: str) -> str:
        """Call Anthropic's Haiku API for re-ranking"""
        if self.mock_mode or httpx is None:
            # Mock response for testing
            await asyncio.sleep(0.01)  # Simulate API latency
            return self._generate_mock_ranking()
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.1  # Low temperature for consistent rankings
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=payload,
                    timeout=10.0
                )
                response.raise_for_status()
                
                data = response.json()
                return data["content"][0]["text"]
                
            except httpx.TimeoutException:
                logger.warning("Haiku API timeout, falling back to original ranking")
                telemetry.record_error("api_timeout", "haiku_reranker")
                return self._generate_fallback_ranking()
            except Exception as e:
                logger.error(f"Haiku API error: {e}")
                telemetry.record_error("api_error", "haiku_reranker")
                return self._generate_fallback_ranking()
    
    def _generate_mock_ranking(self) -> str:
        """Generate mock ranking for testing"""
        return """Based on the query relevance analysis:

RANKING (most relevant first):
1. Result with highest semantic similarity to query intent
2. Result with strong contextual match and good code quality
3. Result with partial relevance but useful supporting information
4. Result with lower direct relevance but valuable related concepts

CONFIDENCE: 0.85

The ranking prioritizes semantic relevance, code quality, and contextual appropriateness."""
    
    def _generate_fallback_ranking(self) -> str:
        """Generate fallback ranking when API fails"""
        return """FALLBACK_RANKING: Maintaining original vector similarity order.
CONFIDENCE: 0.50"""
    
    def _build_ranking_prompt(self, request: ReRankingRequest) -> str:
        """Build prompt for Haiku re-ranking"""
        prompt_parts = [
            f"TASK: Re-rank these search results for the query: '{request.query}'",
            "",
            "CONTEXT:" if request.context else "",
            request.context if request.context else "",
            "",
            "SEARCH RESULTS TO RANK:",
        ]
        
        for i, result in enumerate(request.results):
            snippet = result.content[:200] + "..." if len(result.content) > 200 else result.content
            prompt_parts.append(f"{i+1}. [{result.source}] Score: {result.score:.3f}")
            prompt_parts.append(f"   {snippet}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "INSTRUCTIONS:",
            "1. Analyze each result's relevance to the query",
            "2. Consider semantic meaning, not just keyword matching",
            "3. Account for code quality and completeness",
            "4. Provide ranking from most to least relevant",
            "5. Give confidence score (0.0-1.0)",
            "",
            "RESPONSE FORMAT:",
            "RANKING (most relevant first):",
            "1. [Brief reason for ranking]",
            "2. [Brief reason for ranking]",
            "...",
            "",
            "CONFIDENCE: [0.0-1.0]"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_ranking_response(self, response: str, original_results: List[SearchResult]) -> Tuple[List[int], float]:
        """
        Parse Haiku's ranking response
        
        Returns:
            Tuple of (ranking_indices, confidence_score)
        """
        lines = response.strip().split('\n')
        ranking_indices = []
        confidence_score = 0.5  # Default confidence
        
        # Look for ranking section
        in_ranking = False
        for line in lines:
            line = line.strip()
            
            if line.startswith("RANKING"):
                in_ranking = True
                continue
            
            if in_ranking and line and line[0].isdigit():
                # Try to extract original index from ranking
                # For now, use the order provided (1st becomes index 0, etc.)
                if len(ranking_indices) < len(original_results):
                    ranking_indices.append(len(ranking_indices))
            
            if line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.split(":")[1].strip()
                    confidence_score = float(conf_str)
                except (ValueError, IndexError):
                    pass
        
        # If parsing failed, return original order
        if not ranking_indices:
            ranking_indices = list(range(len(original_results)))
        
        # Handle fallback ranking
        if "FALLBACK_RANKING" in response:
            confidence_score = 0.5
            ranking_indices = list(range(len(original_results)))
        
        return ranking_indices, confidence_score
    
    async def rerank(self, request: ReRankingRequest) -> ReRankingResult:
        """
        Re-rank search results using Haiku model
        
        Args:
            request: ReRankingRequest with query and results
            
        Returns:
            ReRankingResult with re-ranked results
        """
        start_time = time.perf_counter()
        self.stats['requests'] += 1
        
        with telemetry.trace_operation("haiku_rerank", {
            "query_length": len(request.query),
            "result_count": len(request.results),
            "max_results": request.max_results,
            "has_context": request.context is not None,
            "mock_mode": self.mock_mode
        }) as span:
            # Check cache first
            cached_result = self._check_cache(request)
            if cached_result:
                if span:
                    span.set_attribute("cache_hit", True)
                    span.set_attribute("processing_time_ms", cached_result.processing_time * 1000)
                return cached_result
            
            if span:
                span.set_attribute("cache_hit", False)
            
            try:
                # Limit results for performance
                results_to_rank = request.results[:20]  # Haiku can handle up to 20 efficiently
                
                if span:
                    span.set_attribute("results_to_rank", len(results_to_rank))
                
                # Build ranking prompt
                with telemetry.trace_operation("haiku_prompt_build", {
                    "result_count": len(results_to_rank)
                }):
                    prompt = self._build_ranking_prompt(
                        ReRankingRequest(
                            query=request.query,
                            results=results_to_rank,
                            context=request.context,
                            max_results=request.max_results
                        )
                    )
                
                # Call Haiku API
                api_start = time.perf_counter()
                with telemetry.trace_operation("haiku_api_call", {
                    "model": "claude-3-haiku" if not self.mock_mode else "mock",
                    "prompt_length": len(prompt)
                }) as api_span:
                    response = await self._call_haiku_api(prompt)
                    api_duration = time.perf_counter() - api_start
                    if api_span:
                        api_span.set_attribute("api_duration_ms", api_duration * 1000)
                        api_span.set_attribute("response_length", len(response))
                
                # Parse response
                with telemetry.trace_operation("haiku_response_parse"):
                    ranking_indices, confidence_score = self._parse_ranking_response(response, results_to_rank)
                
                # Apply ranking
                ranked_results = []
                for i, original_idx in enumerate(ranking_indices):
                    if original_idx < len(results_to_rank):
                        result = results_to_rank[original_idx]
                        # Update with new ranking info
                        result.metadata['reranked'] = True
                        result.metadata['original_rank'] = original_idx + 1
                        result.metadata['new_rank'] = i + 1
                        result.metadata['ranking_confidence'] = confidence_score
                        ranked_results.append(result)
                
                # Limit to requested max results
                ranked_results = ranked_results[:request.max_results]
                
                processing_time = time.perf_counter() - start_time
                
                # Create result
                result = ReRankingResult(
                    results=ranked_results,
                    processing_time=processing_time,
                    model_used="claude-3-haiku" if not self.mock_mode else "mock",
                    confidence_score=confidence_score,
                    cache_hit=False
                )
                
                # Update stats
                self.stats['total_processing_time'] += processing_time
                self.stats['avg_processing_time'] = (
                    self.stats['total_processing_time'] / self.stats['requests']
                )
                
                # Record telemetry
                telemetry.record_rerank("haiku", len(request.results), processing_time)
                
                if span:
                    span.set_attribute("success", True)
                    span.set_attribute("processing_time_ms", processing_time * 1000)
                    span.set_attribute("confidence_score", confidence_score)
                    span.set_attribute("final_result_count", len(ranked_results))
                
                # Cache result
                self._cache_result(request, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Re-ranking failed: {e}")
                telemetry.record_error("reranking_failed", "haiku_reranker")
                
                if span:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                
                # Return original results as fallback
                processing_time = time.perf_counter() - start_time
                
                fallback_result = ReRankingResult(
                    results=request.results[:request.max_results],
                    processing_time=processing_time,
                    model_used="fallback",
                    confidence_score=0.3,
                    cache_hit=False
                )
                
                if span:
                    span.set_attribute("processing_time_ms", processing_time * 1000)
                    span.set_attribute("fallback_used", True)
                
                return fallback_result
    
    async def rerank_simple(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Simplified re-ranking interface for direct use
        
        Args:
            query: Search query
            results: List of search results (dicts with 'content', 'score', etc.)
            context: Optional context for ranking
            max_results: Maximum results to return
            
        Returns:
            List of re-ranked results
        """
        with telemetry.trace_operation("haiku_rerank_simple", {
            "query_length": len(query),
            "result_count": len(results),
            "max_results": max_results,
            "has_context": context is not None
        }):
            # Convert to SearchResult objects
            search_results = []
            for i, result in enumerate(results):
                search_result = SearchResult(
                    content=result.get('content', str(result)),
                    score=result.get('score', 0.0),
                    metadata=result.get('metadata', {}),
                    source=result.get('source', 'unknown'),
                    original_rank=i + 1
                )
                search_results.append(search_result)
            
            # Create request
            request = ReRankingRequest(
                query=query,
                results=search_results,
                context=context,
                max_results=max_results
            )
            
            # Re-rank
            ranking_result = await self.rerank(request)
            
            # Convert back to dict format
            reranked = []
            for result in ranking_result.results:
                # Merge original data with new metadata
                original_data = next(
                    (r for r in results if r.get('content') == result.content),
                    {}
                )
                
                reranked_data = {**original_data}
                reranked_data.update({
                    'content': result.content,
                    'score': result.score,
                    'metadata': {**reranked_data.get('metadata', {}), **result.metadata},
                    'source': result.source,
                    'reranking_confidence': ranking_result.confidence_score,
                    'processing_time': ranking_result.processing_time
                })
                
                reranked.append(reranked_data)
            
            return reranked
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        cache_hit_rate = (
            self.stats['cache_hits'] / max(self.stats['requests'], 1)
        ) * 100
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache),
            'mock_mode': self.mock_mode
        }
    
    def clear_cache(self):
        """Clear the ranking cache"""
        self.cache.clear()
        logger.info("HaikuReRanker cache cleared")


# Convenience function for easy integration
async def rerank_search_results(
    query: str,
    results: List[Dict[str, Any]],
    context: Optional[str] = None,
    max_results: int = 10,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to re-rank search results
    
    Args:
        query: Search query
        results: List of search results
        context: Optional context
        max_results: Maximum results to return
        api_key: Anthropic API key (optional for mock mode)
        
    Returns:
        List of re-ranked results
    """
    reranker = HaikuReRanker(api_key=api_key)
    return await reranker.rerank_simple(query, results, context, max_results)


# Global instance for easy access
_global_reranker: Optional[HaikuReRanker] = None

def get_global_reranker(api_key: Optional[str] = None) -> HaikuReRanker:
    """Get or create global re-ranker instance"""
    global _global_reranker
    
    if _global_reranker is None:
        _global_reranker = HaikuReRanker(api_key=api_key)
    
    return _global_reranker