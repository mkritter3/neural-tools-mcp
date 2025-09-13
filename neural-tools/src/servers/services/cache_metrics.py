"""
Cache Performance Analytics and Metrics Service - Phase 3 Intelligence

Provides comprehensive cache performance monitoring, analytics, and optimization metrics:
- Real-time hit/miss ratio tracking
- Service-specific cache performance analysis
- Cache size and memory usage monitoring  
- Performance trend analysis and recommendations
- Integration with cache warmer analytics
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CacheMetricsService:
    """
    Comprehensive cache performance analytics and monitoring service
    
    Features:
    - Real-time cache hit/miss ratio tracking
    - Service-specific performance breakdown (Neo4j, Qdrant, Nomic)
    - Memory usage and cache size monitoring
    - Performance trend analysis with recommendations
    - Integration with Redis native stats
    """
    
    def __init__(self, service_container):
        self.container = service_container
        self.metrics_key_prefix = "l9:prod:neural_tools:metrics"
        
        # Performance tracking windows
        self.short_window_seconds = int(os.getenv('CACHE_METRICS_SHORT_WINDOW', 300))  # 5 minutes
        self.long_window_seconds = int(os.getenv('CACHE_METRICS_LONG_WINDOW', 3600))   # 1 hour
        
        # Metrics collection intervals
        self.collection_interval = float(os.getenv('CACHE_METRICS_INTERVAL', 60))  # 1 minute
        
        # Background metrics collection task
        self.metrics_task = None
        self.initialized = False
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize cache metrics service and start background collection"""
        try:
            redis_cache = await self.container.get_redis_cache_client()
            
            # Test Redis connection
            await redis_cache.ping()
            
            # Start background metrics collection
            self.metrics_task = asyncio.create_task(self._collect_metrics_loop())
            self.initialized = True
            
            logger.info("✅ Cache metrics service initialized with background collection")
            return {"success": True, "message": "Cache metrics service ready"}
            
        except Exception as e:
            logger.error(f"Failed to initialize cache metrics service: {e}")
            return {"success": False, "message": str(e)}
    
    async def _collect_metrics_loop(self):
        """Background loop to collect cache metrics"""
        while True:
            try:
                await self._collect_redis_stats()
                await self._collect_service_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                logger.info("Cache metrics collection stopped")
                break
            except Exception as e:
                logger.warning(f"Metrics collection iteration failed: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_redis_stats(self):
        """Collect Redis cache statistics"""
        try:
            redis_cache = await self.container.get_redis_cache_client()
            stats = await redis_cache.info('stats')
            memory_info = await redis_cache.info('memory')
            
            timestamp = int(time.time())
            metrics_data = {
                'timestamp': timestamp,
                'keyspace_hits': int(stats.get('keyspace_hits', 0)),
                'keyspace_misses': int(stats.get('keyspace_misses', 0)),
                'expired_keys': int(stats.get('expired_keys', 0)),
                'evicted_keys': int(stats.get('evicted_keys', 0)),
                'used_memory': int(memory_info.get('used_memory', 0)),
                'used_memory_peak': int(memory_info.get('used_memory_peak', 0)),
                'connected_clients': int(stats.get('connected_clients', 0))
            }
            
            # Store metrics with sliding window
            metrics_key = f"{self.metrics_key_prefix}:redis_stats"
            await redis_cache.lpush(metrics_key, json.dumps(metrics_data))
            await redis_cache.ltrim(metrics_key, 0, 1439)  # Keep last 24 hours (assuming 1-minute intervals)
            await redis_cache.expire(metrics_key, 86400)  # 24 hours expiry
            
        except Exception as e:
            logger.warning(f"Failed to collect Redis stats: {e}")
    
    async def _collect_service_metrics(self):
        """Collect service-specific cache metrics"""
        try:
            redis_cache = await self.container.get_redis_cache_client()
            
            # Get cache keys by service type
            service_patterns = {
                'neo4j': 'l9:prod:neural_tools:neo4j:*',
                'qdrant': 'l9:prod:neural_tools:qdrant:*', 
                'embeddings': 'l9:prod:neural_tools:embeddings:*',
                'analytics': 'l9:prod:neural_tools:analytics:*'
            }
            
            service_stats = {}
            
            for service_name, pattern in service_patterns.items():
                keys = await redis_cache.keys(pattern)
                
                # Count keys and estimate memory usage
                key_count = len(keys)
                total_memory = 0
                total_ttl = 0
                ttl_count = 0
                
                # Sample subset for performance (max 100 keys for TTL analysis)
                sample_keys = keys[:100] if len(keys) > 100 else keys
                
                for key in sample_keys:
                    try:
                        ttl = await redis_cache.ttl(key)
                        if ttl > 0:
                            total_ttl += ttl
                            ttl_count += 1
                        
                        # Estimate memory (approximate)
                        key_size = await redis_cache.memory_usage(key)
                        if key_size:
                            total_memory += key_size
                    except Exception:
                        continue  # Skip problematic keys
                
                avg_ttl = total_ttl // ttl_count if ttl_count > 0 else 0
                
                service_stats[service_name] = {
                    'key_count': key_count,
                    'estimated_memory_bytes': total_memory,
                    'avg_ttl_seconds': avg_ttl,
                    'sample_size': len(sample_keys)
                }
            
            # Store service metrics
            timestamp = int(time.time())
            metrics_data = {
                'timestamp': timestamp,
                'services': service_stats
            }
            
            metrics_key = f"{self.metrics_key_prefix}:service_stats"
            await redis_cache.lpush(metrics_key, json.dumps(metrics_data))
            await redis_cache.ltrim(metrics_key, 0, 287)  # Keep last 24 hours (assuming 5-minute intervals)
            await redis_cache.expire(metrics_key, 86400)
            
        except Exception as e:
            logger.warning(f"Failed to collect service metrics: {e}")
    
    async def get_cache_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive cache performance summary"""
        try:
            redis_cache = await self.container.get_redis_cache_client()
            
            # Get recent Redis stats
            stats_key = f"{self.metrics_key_prefix}:redis_stats"
            recent_stats = await redis_cache.lrange(stats_key, 0, window_minutes - 1)
            
            if not recent_stats:
                return {"error": "No metrics data available"}
            
            # Parse metrics data
            parsed_stats = []
            for stat_entry in recent_stats:
                try:
                    parsed_stats.append(json.loads(stat_entry))
                except json.JSONDecodeError:
                    continue
            
            if not parsed_stats:
                return {"error": "No valid metrics data"}
            
            # Calculate performance metrics
            latest_stats = parsed_stats[0]
            oldest_stats = parsed_stats[-1] if len(parsed_stats) > 1 else latest_stats
            
            # Calculate deltas for rate-based metrics
            hits_delta = latest_stats['keyspace_hits'] - oldest_stats['keyspace_hits']
            misses_delta = latest_stats['keyspace_misses'] - oldest_stats['keyspace_misses']
            total_requests = hits_delta + misses_delta
            
            hit_ratio = hits_delta / total_requests if total_requests > 0 else 0
            
            # Get service breakdown
            service_stats = await self._get_service_breakdown()
            
            # Performance assessment
            performance_grade = self._assess_performance(hit_ratio, latest_stats)
            
            return {
                "summary": {
                    "hit_ratio": round(hit_ratio, 3),
                    "total_requests": total_requests,
                    "hits": hits_delta,
                    "misses": misses_delta,
                    "performance_grade": performance_grade,
                    "window_minutes": window_minutes
                },
                "memory": {
                    "used_memory_mb": round(latest_stats['used_memory'] / 1024 / 1024, 2),
                    "peak_memory_mb": round(latest_stats['used_memory_peak'] / 1024 / 1024, 2),
                    "memory_efficiency": round(latest_stats['used_memory'] / max(1, latest_stats['used_memory_peak']), 3)
                },
                "services": service_stats,
                "maintenance": {
                    "expired_keys": latest_stats['expired_keys'] - oldest_stats['expired_keys'],
                    "evicted_keys": latest_stats['evicted_keys'] - oldest_stats['evicted_keys'],
                    "connected_clients": latest_stats['connected_clients']
                },
                "recommendations": self._generate_recommendations(hit_ratio, latest_stats, service_stats)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {"error": str(e)}
    
    async def _get_service_breakdown(self) -> Dict[str, Any]:
        """Get service-specific cache statistics breakdown"""
        try:
            redis_cache = await self.container.get_redis_cache_client()
            service_key = f"{self.metrics_key_prefix}:service_stats"
            recent_service_data = await redis_cache.lrange(service_key, 0, 0)
            
            if not recent_service_data:
                return {"error": "No service metrics available"}
            
            latest_data = json.loads(recent_service_data[0])
            return latest_data.get('services', {})
            
        except Exception as e:
            logger.warning(f"Failed to get service breakdown: {e}")
            return {"error": str(e)}
    
    def _assess_performance(self, hit_ratio: float, stats: Dict[str, Any]) -> str:
        """Assess cache performance and assign grade"""
        if hit_ratio >= 0.9:
            return "Excellent"
        elif hit_ratio >= 0.8:
            return "Good"
        elif hit_ratio >= 0.7:
            return "Fair"
        elif hit_ratio >= 0.5:
            return "Poor"
        else:
            return "Critical"
    
    def _generate_recommendations(self, hit_ratio: float, stats: Dict[str, Any], service_stats: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Hit ratio recommendations
        if hit_ratio < 0.7:
            recommendations.append("Consider increasing cache TTL values for frequently accessed data")
            recommendations.append("Review cache warming strategies to pre-load popular queries")
        
        # Memory usage recommendations
        used_memory_mb = stats['used_memory'] / 1024 / 1024
        if used_memory_mb > 1000:  # >1GB
            recommendations.append("High memory usage detected - consider implementing cache size limits")
        
        # Eviction recommendations
        if stats.get('evicted_keys', 0) > 0:
            recommendations.append("Cache evictions detected - increase memory allocation or optimize TTL policies")
        
        # Service-specific recommendations
        if isinstance(service_stats, dict):
            for service_name, service_data in service_stats.items():
                if isinstance(service_data, dict) and service_data.get('key_count', 0) > 10000:
                    recommendations.append(f"High key count in {service_name} cache - consider data archival policies")
        
        if not recommendations:
            recommendations.append("Cache performance is optimal - no immediate optimizations needed")
        
        return recommendations
    
    async def get_trending_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get trending cache performance metrics over time"""
        try:
            redis_cache = await self.container.get_redis_cache_client()
            stats_key = f"{self.metrics_key_prefix}:redis_stats"
            
            # Calculate how many entries to fetch (assuming 1-minute intervals)
            entries_needed = min(hours * 60, 1440)  # Cap at 24 hours
            historical_stats = await redis_cache.lrange(stats_key, 0, entries_needed - 1)
            
            if len(historical_stats) < 2:
                return {"error": "Insufficient data for trending analysis"}
            
            # Parse and analyze trends
            parsed_data = []
            for entry in reversed(historical_stats):  # Reverse for chronological order
                try:
                    parsed_data.append(json.loads(entry))
                except json.JSONDecodeError:
                    continue
            
            if len(parsed_data) < 2:
                return {"error": "Insufficient valid data"}
            
            # Calculate trends
            hit_ratios = []
            memory_usage = []
            timestamps = []
            
            for i in range(1, len(parsed_data)):
                prev = parsed_data[i-1]
                curr = parsed_data[i]
                
                hits_delta = curr['keyspace_hits'] - prev['keyspace_hits']
                misses_delta = curr['keyspace_misses'] - prev['keyspace_misses']
                total_delta = hits_delta + misses_delta
                
                if total_delta > 0:
                    hit_ratio = hits_delta / total_delta
                    hit_ratios.append(hit_ratio)
                    memory_usage.append(curr['used_memory'] / 1024 / 1024)  # MB
                    timestamps.append(curr['timestamp'])
            
            if not hit_ratios:
                return {"error": "No request activity in the specified period"}
            
            # Calculate trend statistics
            avg_hit_ratio = sum(hit_ratios) / len(hit_ratios)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            # Trend direction (simple linear approximation)
            hit_ratio_trend = "stable"
            if len(hit_ratios) >= 5:
                recent_avg = sum(hit_ratios[-5:]) / 5
                earlier_avg = sum(hit_ratios[:5]) / 5
                
                if recent_avg > earlier_avg + 0.05:
                    hit_ratio_trend = "improving"
                elif recent_avg < earlier_avg - 0.05:
                    hit_ratio_trend = "declining"
            
            return {
                "period_hours": hours,
                "data_points": len(hit_ratios),
                "performance": {
                    "avg_hit_ratio": round(avg_hit_ratio, 3),
                    "min_hit_ratio": round(min(hit_ratios), 3),
                    "max_hit_ratio": round(max(hit_ratios), 3),
                    "trend": hit_ratio_trend
                },
                "memory": {
                    "avg_memory_mb": round(avg_memory, 2),
                    "min_memory_mb": round(min(memory_usage), 2),
                    "max_memory_mb": round(max(memory_usage), 2)
                },
                "time_range": {
                    "start": datetime.fromtimestamp(timestamps[0]).isoformat(),
                    "end": datetime.fromtimestamp(timestamps[-1]).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate trending metrics: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_metrics(self, days_to_keep: int = 7):
        """Clean up old metrics data to prevent unbounded growth"""
        try:
            redis_cache = await self.container.get_redis_cache_client()
            
            # Clean up metrics keys
            metrics_keys = [
                f"{self.metrics_key_prefix}:redis_stats",
                f"{self.metrics_key_prefix}:service_stats"
            ]
            
            cleanup_results = {}
            
            for key in metrics_keys:
                # Keep only recent entries (based on collection intervals)
                entries_to_keep = days_to_keep * 1440  # Assuming 1-minute intervals
                await redis_cache.ltrim(key, 0, entries_to_keep - 1)
                
                # Get current length after cleanup
                current_length = await redis_cache.llen(key)
                cleanup_results[key] = {
                    "entries_retained": current_length,
                    "max_entries": entries_to_keep
                }
            
            logger.info(f"Metrics cleanup completed: {cleanup_results}")
            return {"success": True, "cleanup_results": cleanup_results}
            
        except Exception as e:
            logger.error(f"Metrics cleanup failed: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown cache metrics service and stop background collection"""
        if self.metrics_task and not self.metrics_task.done():
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
        
        self.initialized = False
        logger.info("Cache metrics service shutdown")

import os