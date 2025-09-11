"""
L9 2025 Redis-Backed Rate Limiter for MCP Sessions
Distributed rate limiting with sliding window algorithm
"""

import time
import logging
from typing import Dict, Optional, Tuple
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RedisRateLimiter:
    """Distributed rate limiter using Redis sliding window"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.prefix = "rate_limit"
        
    async def check_rate_limit(
        self, 
        session_id: str, 
        limit: int = 60, 
        window_seconds: int = 60
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if session is within rate limits using sliding window
        
        Returns:
            (allowed: bool, info: dict with current usage)
        """
        key = f"{self.prefix}:{session_id}"
        current_time = time.time()
        window_start = current_time - window_seconds
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window_seconds + 10)  # Extra buffer
            
            results = await pipe.execute()
            current_count = results[1]  # Count after cleanup
            
            # Check if within limit
            allowed = current_count < limit
            
            if not allowed:
                # Remove the request we just added since it's rejected
                await self.redis_client.zrem(key, str(current_time))
                
            info = {
                "current_count": current_count + (1 if allowed else 0),
                "limit": limit,
                "window_seconds": window_seconds,
                "reset_time": current_time + window_seconds,
                "allowed": allowed
            }
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for session {session_id[:8]}... ({current_count}/{limit})")
            
            return allowed, info
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open - allow request if Redis is down
            return True, {
                "current_count": 0,
                "limit": limit,
                "window_seconds": window_seconds,
                "error": "Redis unavailable",
                "allowed": True
            }
    
    async def get_rate_limit_info(self, session_id: str, window_seconds: int = 60) -> Dict[str, any]:
        """Get current rate limit status for session"""
        key = f"{self.prefix}:{session_id}"
        current_time = time.time()
        window_start = current_time - window_seconds
        
        try:
            # Clean up expired entries and count
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = await pipe.execute()
            
            current_count = results[1]
            
            return {
                "current_count": current_count,
                "window_seconds": window_seconds,
                "reset_time": current_time + window_seconds,
                "last_check": current_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get rate limit info: {e}")
            return {
                "current_count": 0,
                "window_seconds": window_seconds,
                "error": str(e)
            }
    
    async def reset_rate_limit(self, session_id: str) -> bool:
        """Reset rate limit for session (admin function)"""
        key = f"{self.prefix}:{session_id}"
        try:
            await self.redis_client.delete(key)
            logger.info(f"Rate limit reset for session {session_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            return False
    
    async def get_all_sessions_info(self) -> Dict[str, Dict]:
        """Get rate limit info for all active sessions"""
        try:
            pattern = f"{self.prefix}:*"
            keys = await self.redis_client.keys(pattern)
            
            info = {}
            for key in keys:
                session_id = key.split(':', 1)[1]  # Remove prefix
                session_info = await self.get_rate_limit_info(session_id)
                info[session_id] = session_info
                
            return info
            
        except Exception as e:
            logger.error(f"Failed to get all sessions info: {e}")
            return {}


class SessionResourceQuota:
    """Enforce resource quotas per session"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.prefix = "session_quota"
        
    async def check_connection_quota(
        self, 
        session_id: str, 
        service: str, 
        max_connections: int = 3
    ) -> Tuple[bool, int]:
        """Check if session can open another connection to service"""
        key = f"{self.prefix}:{session_id}:{service}"
        
        try:
            current_connections = await self.redis_client.get(key)
            current_count = int(current_connections) if current_connections else 0
            
            allowed = current_count < max_connections
            
            if allowed:
                # Increment connection count
                await self.redis_client.incr(key)
                await self.redis_client.expire(key, 3600)  # 1 hour timeout
                
            return allowed, current_count + (1 if allowed else 0)
            
        except Exception as e:
            logger.error(f"Connection quota check failed: {e}")
            return True, 0  # Fail open
    
    async def release_connection(self, session_id: str, service: str):
        """Release a connection from session quota"""
        key = f"{self.prefix}:{session_id}:{service}"
        
        try:
            current = await self.redis_client.get(key)
            if current and int(current) > 0:
                await self.redis_client.decr(key)
                
        except Exception as e:
            logger.error(f"Failed to release connection: {e}")
    
    async def cleanup_session_quotas(self, session_id: str):
        """Cleanup all quotas for session"""
        pattern = f"{self.prefix}:{session_id}:*"
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
                logger.debug(f"Cleaned up quotas for session {session_id[:8]}...")
                
        except Exception as e:
            logger.error(f"Failed to cleanup session quotas: {e}")
    
    async def get_session_usage(self, session_id: str) -> Dict[str, int]:
        """Get current resource usage for session"""
        pattern = f"{self.prefix}:{session_id}:*"
        
        try:
            keys = await self.redis_client.keys(pattern)
            usage = {}
            
            for key in keys:
                service = key.split(':', 2)[2]  # Extract service name
                count = await self.redis_client.get(key)
                usage[service] = int(count) if count else 0
                
            return usage
            
        except Exception as e:
            logger.error(f"Failed to get session usage: {e}")
            return {}