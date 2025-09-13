"""
Cache Warming Service - Phase 3 Intelligent Caching

Provides analytics-driven cache warming and optimization:
- Frequent query analysis and pre-loading
- TTL optimization based on access patterns  
- Cache invalidation strategies
- Performance analytics and recommendations
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CacheAnalytics:
    """Analytics engine for cache access patterns and optimization recommendations"""
    
    def __init__(self, redis_cache):
        self.redis = redis_cache
        self.analytics_key_prefix = "l9:prod:neural_tools:analytics"
        
    async def record_cache_access(self, cache_key: str, hit: bool, ttl_remaining: Optional[int] = None):
        """Record cache access for analytics"""
        try:
            # Record access pattern
            access_key = f"{self.analytics_key_prefix}:access:{cache_key}"
            timestamp = int(time.time())
            
            # Store access log with sliding window (keep last 1000 accesses)
            await self.redis.lpush(access_key, json.dumps({
                'timestamp': timestamp,
                'hit': hit,
                'ttl_remaining': ttl_remaining
            }))
            await self.redis.ltrim(access_key, 0, 999)  # Keep last 1000 entries
            await self.redis.expire(access_key, 86400 * 7)  # Keep for 7 days
            
            # Update frequency counters
            freq_key = f"{self.analytics_key_prefix}:frequency"
            await self.redis.zincrby(freq_key, 1, cache_key)
            await self.redis.expire(freq_key, 86400 * 7)
            
        except Exception as e:
            logger.warning(f"Failed to record cache access analytics: {e}")
    
    async def get_frequent_queries(self, limit: int = 100) -> List[Tuple[str, float]]:
        """Get most frequently accessed cache keys"""
        try:
            freq_key = f"{self.analytics_key_prefix}:frequency"
            # Get top N most frequent keys with scores
            frequent_keys = await self.redis.zrevrange(freq_key, 0, limit-1, withscores=True)
            return [(key.decode() if isinstance(key, bytes) else key, score) 
                   for key, score in frequent_keys]
        except Exception as e:
            logger.warning(f"Failed to get frequent queries: {e}")
            return []
    
    async def analyze_ttl_patterns(self, cache_key: str) -> Dict[str, Any]:
        """Analyze TTL patterns for a specific cache key"""
        try:
            access_key = f"{self.analytics_key_prefix}:access:{cache_key}"
            access_logs = await self.redis.lrange(access_key, 0, -1)
            
            if not access_logs:
                return {"error": "No access data available"}
            
            # Parse access logs
            accesses = []
            for log_entry in access_logs:
                try:
                    access_data = json.loads(log_entry)
                    accesses.append(access_data)
                except json.JSONDecodeError:
                    continue
            
            if not accesses:
                return {"error": "No valid access data"}
            
            # Calculate statistics
            total_accesses = len(accesses)
            hits = sum(1 for a in accesses if a.get('hit'))
            hit_ratio = hits / total_accesses if total_accesses > 0 else 0
            
            # Analyze TTL patterns
            ttl_values = [a.get('ttl_remaining') for a in accesses if a.get('ttl_remaining') is not None]
            avg_ttl_at_access = sum(ttl_values) / len(ttl_values) if ttl_values else 0
            
            # Determine optimal TTL (higher access frequency = longer TTL)
            access_frequency = total_accesses / 7  # accesses per day over last week
            
            if access_frequency > 10:  # High frequency
                recommended_ttl = 86400 * 3  # 3 days
            elif access_frequency > 1:   # Medium frequency  
                recommended_ttl = 86400     # 1 day
            else:  # Low frequency
                recommended_ttl = 3600 * 12  # 12 hours
            
            return {
                "total_accesses": total_accesses,
                "hit_ratio": round(hit_ratio, 3),
                "avg_ttl_at_access": round(avg_ttl_at_access),
                "access_frequency_per_day": round(access_frequency, 2),
                "recommended_ttl": recommended_ttl,
                "ttl_optimization": "increase" if avg_ttl_at_access > recommended_ttl * 0.8 else "maintain"
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze TTL patterns for {cache_key}: {e}")
            return {"error": str(e)}

class CacheWarmer:
    """
    Intelligent cache warming service with analytics-driven optimization
    
    Features:
    - Frequent query identification and pre-loading
    - TTL optimization based on access patterns
    - Configurable warming strategies
    - Performance monitoring and recommendations
    """
    
    def __init__(self, service_container):
        self.container = service_container
        self.analytics = None  # Will be initialized with Redis client
        
        # Configuration
        self.warming_batch_size = int(os.getenv('CACHE_WARMING_BATCH_SIZE', 50))
        self.warming_concurrency = int(os.getenv('CACHE_WARMING_CONCURRENCY', 5))
        self.warming_delay_seconds = float(os.getenv('CACHE_WARMING_DELAY', 0.1))
        
        # Performance tracking
        self.warming_stats = {
            'total_warmed': 0,
            'successful_warms': 0,
            'failed_warms': 0,
            'last_warming': None
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize cache warmer with Redis analytics"""
        try:
            redis_cache = await self.container.get_redis_cache_client()
            self.analytics = CacheAnalytics(redis_cache)
            
            logger.info("✅ Cache warmer initialized with analytics")
            return {"success": True, "message": "Cache warmer ready"}
            
        except Exception as e:
            logger.error(f"Failed to initialize cache warmer: {e}")
            return {"success": False, "message": str(e)}
    
    async def warm_embedding_cache(self, queries: List[str], model: str = "nomic-v2") -> Dict[str, Any]:
        """
        Warm embedding cache for specific queries
        
        Args:
            queries: List of text queries to cache embeddings for
            model: Model identifier for embeddings
            
        Returns:
            Warming results with success/failure counts
        """
        if not self.container.nomic:
            return {"error": "Nomic service not available"}
        
        results = {
            "total_queries": len(queries),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        # Process in batches with concurrency control
        semaphore = asyncio.Semaphore(self.warming_concurrency)
        
        async def warm_single_query(query: str) -> bool:
            async with semaphore:
                try:
                    # Use the existing caching logic in NomicService
                    await self.container.nomic.get_embedding(query, model)
                    results["successful"] += 1
                    self.warming_stats['successful_warms'] += 1
                    
                    # Small delay to prevent overwhelming the service
                    await asyncio.sleep(self.warming_delay_seconds)
                    return True
                    
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"{query[:50]}...: {str(e)}")
                    self.warming_stats['failed_warms'] += 1
                    logger.warning(f"Failed to warm cache for query: {e}")
                    return False
        
        # Execute warming tasks
        tasks = [warm_single_query(query) for query in queries]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.warming_stats['total_warmed'] += results["total_queries"]
        self.warming_stats['last_warming'] = datetime.now().isoformat()
        
        logger.info(f"Cache warming complete: {results['successful']}/{results['total_queries']} successful")
        return results
    
    async def auto_warm_frequent_queries(self, limit: int = 50) -> Dict[str, Any]:
        """
        Automatically warm cache for most frequently accessed queries
        
        Args:
            limit: Number of top frequent queries to warm
            
        Returns:
            Warming results
        """
        try:
            if not self.analytics:
                await self.initialize()
            
            # Get frequent queries from analytics
            frequent_queries = await self.analytics.get_frequent_queries(limit)
            
            if not frequent_queries:
                return {"message": "No frequent queries found for warming"}
            
            # Extract actual query text from cache keys
            embedding_queries = []
            for cache_key, frequency in frequent_queries:
                # Parse embedding cache keys: l9:prod:neural_tools:embeddings:model:version:hash
                if ":embeddings:" in cache_key:
                    # For now, we can't reverse the hash to get original text
                    # This would require storing query-to-hash mappings
                    continue
                embedding_queries.append(cache_key)
            
            if not embedding_queries:
                return {"message": "No embedding queries identified for warming"}
            
            # Warm the identified queries
            return await self.warm_embedding_cache(embedding_queries[:limit])
            
        except Exception as e:
            logger.error(f"Auto cache warming failed: {e}")
            return {"error": str(e)}
    
    async def optimize_ttl_policies(self) -> Dict[str, Any]:
        """
        Analyze cache access patterns and recommend TTL optimizations
        
        Returns:
            TTL optimization recommendations
        """
        try:
            if not self.analytics:
                await self.initialize()
            
            # Get frequent queries for analysis
            frequent_queries = await self.analytics.get_frequent_queries(100)
            
            recommendations = {
                "analyzed_keys": 0,
                "ttl_recommendations": [],
                "summary": {
                    "increase_ttl": 0,
                    "maintain_ttl": 0,
                    "high_frequency_keys": 0
                }
            }
            
            for cache_key, frequency in frequent_queries:
                # Analyze TTL patterns for each key
                analysis = await self.analytics.analyze_ttl_patterns(cache_key)
                
                if analysis.get("error"):
                    continue
                
                recommendations["analyzed_keys"] += 1
                
                # Collect recommendations
                if analysis.get("access_frequency_per_day", 0) > 5:  # High frequency
                    recommendations["summary"]["high_frequency_keys"] += 1
                
                if analysis.get("ttl_optimization") == "increase":
                    recommendations["summary"]["increase_ttl"] += 1
                    recommendations["ttl_recommendations"].append({
                        "cache_key": cache_key[:80] + "..." if len(cache_key) > 80 else cache_key,
                        "current_pattern": analysis,
                        "recommendation": "Increase TTL for better cache efficiency"
                    })
                else:
                    recommendations["summary"]["maintain_ttl"] += 1
            
            return recommendations
            
        except Exception as e:
            logger.error(f"TTL optimization analysis failed: {e}")
            return {"error": str(e)}
    
    async def get_warming_stats(self) -> Dict[str, Any]:
        """Get cache warming performance statistics"""
        redis_cache = await self.container.get_redis_cache_client()
        
        # Get Redis cache statistics
        cache_info = await redis_cache.info('stats')
        
        return {
            "warming_stats": self.warming_stats.copy(),
            "cache_performance": {
                "keyspace_hits": cache_info.get('keyspace_hits', 0),
                "keyspace_misses": cache_info.get('keyspace_misses', 0),
                "hit_ratio": cache_info.get('keyspace_hits', 0) / max(1, 
                    cache_info.get('keyspace_hits', 0) + cache_info.get('keyspace_misses', 0)
                )
            },
            "configuration": {
                "warming_batch_size": self.warming_batch_size,
                "warming_concurrency": self.warming_concurrency,
                "warming_delay_seconds": self.warming_delay_seconds
            }
        }
    
    async def invalidate_expired_patterns(self, pattern: str = "l9:prod:neural_tools:*") -> Dict[str, Any]:
        """
        Intelligent cache invalidation based on patterns and TTL analysis
        
        Args:
            pattern: Redis key pattern to analyze for invalidation
            
        Returns:
            Invalidation results
        """
        try:
            redis_cache = await self.container.get_redis_cache_client()
            
            # Get keys matching pattern
            keys = await redis_cache.keys(pattern)
            
            invalidation_results = {
                "total_keys_analyzed": len(keys),
                "keys_invalidated": 0,
                "keys_refreshed": 0,
                "errors": []
            }
            
            for key in keys:
                try:
                    ttl = await redis_cache.ttl(key)
                    
                    # Keys with TTL < 0 are expired or have no expiry
                    if ttl < 0:
                        await redis_cache.delete(key)
                        invalidation_results["keys_invalidated"] += 1
                    elif ttl < 3600:  # Less than 1 hour remaining
                        # Check if this is a frequently accessed key that should be refreshed
                        analysis = await self.analytics.analyze_ttl_patterns(key)
                        if analysis.get("access_frequency_per_day", 0) > 1:
                            # High frequency key - refresh TTL
                            await redis_cache.expire(key, analysis.get("recommended_ttl", 86400))
                            invalidation_results["keys_refreshed"] += 1
                        
                except Exception as e:
                    invalidation_results["errors"].append(f"{key}: {str(e)}")
            
            logger.info(f"Cache invalidation complete: {invalidation_results}")
            return invalidation_results
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return {"error": str(e)}

import os