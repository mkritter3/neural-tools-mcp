"""
L9 2025 Session Manager for MCP Multi-Client Support
Handles session isolation, rate limiting, and resource management
"""

import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class SessionContext:
    """Context for individual MCP session"""
    
    def __init__(self, session_id: str, client_info: Dict[str, Any], rate_limiter=None, quota_manager=None):
        self.session_id = session_id
        self.client_info = client_info
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.query_count = 0
        self.rate_limit_reset = time.time() + 60  # Reset every minute
        self.resource_limits = self._get_default_limits()
        
        # L9 2025: Redis-backed rate limiting and quotas
        self.rate_limiter = rate_limiter
        self.quota_manager = quota_manager
        
    def _get_default_limits(self) -> Dict[str, Any]:
        """Get default resource limits for session"""
        return {
            "queries_per_minute": 60,
            "concurrent_connections": 3,
            "max_result_size": 10_000,
            "session_timeout": 3600,  # 1 hour
            "max_query_duration": 30,  # 30 seconds
            "neo4j_max_connections": 3,
            "qdrant_max_connections": 2,
            "redis_max_connections": 2
        }
    
    async def check_rate_limit(self) -> bool:
        """Check if session is within rate limits (Redis-backed)"""
        if self.rate_limiter:
            # Use Redis-backed distributed rate limiting
            allowed, info = await self.rate_limiter.check_rate_limit(
                self.session_id,
                limit=self.resource_limits["queries_per_minute"],
                window_seconds=60
            )
            if allowed:
                self.last_activity = datetime.now()
            return allowed
        else:
            # Fallback to in-memory rate limiting
            current_time = time.time()
            
            # Reset counter if minute has passed
            if current_time > self.rate_limit_reset:
                self.query_count = 0
                self.rate_limit_reset = current_time + 60
                
            # Check limit
            if self.query_count >= self.resource_limits["queries_per_minute"]:
                logger.warning(f"Rate limit exceeded for session {self.session_id[:8]}...")
                return False
                
            self.query_count += 1
            self.last_activity = datetime.now()
            return True
    
    async def check_connection_quota(self, service: str) -> bool:
        """Check if session can open another connection to service"""
        if self.quota_manager:
            max_connections = self.resource_limits.get(f"{service}_max_connections", 3)
            allowed, current_count = await self.quota_manager.check_connection_quota(
                self.session_id, service, max_connections
            )
            return allowed
        return True  # Fallback: allow if quota manager unavailable
    
    async def release_connection_quota(self, service: str):
        """Release connection quota for service"""
        if self.quota_manager:
            await self.quota_manager.release_connection(self.session_id, service)
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        timeout = timedelta(seconds=self.resource_limits["session_timeout"])
        return datetime.now() - self.last_activity > timeout


class SessionManager:
    """L9 2025 Session Manager for MCP Server"""
    
    def __init__(self):
        self.sessions: Dict[str, SessionContext] = {}
        self.redis_client: Optional[redis.Redis] = None
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        
        # L9 2025: Redis-backed rate limiting and quotas
        self.rate_limiter = None
        self.quota_manager = None
        
    async def initialize(self):
        """Initialize session manager with Redis backing"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=46379,  # Cache Redis instance
                password='cache-secret-key',
                decode_responses=True,
                db=1  # Use db 1 for sessions
            )
            # Test connection
            await self.redis_client.ping()
            
            # Initialize L9 2025 Redis-backed services
            from servers.services.rate_limiter import RedisRateLimiter, SessionResourceQuota
            self.rate_limiter = RedisRateLimiter(self.redis_client)
            self.quota_manager = SessionResourceQuota(self.redis_client)
            
            logger.info("✅ SessionManager initialized with Redis-backed rate limiting and quotas")
        except Exception as e:
            logger.warning(f"Redis not available for sessions, using in-memory fallback: {e}")
            self.redis_client = None
            self.rate_limiter = None
            self.quota_manager = None
    
    async def create_session(self, client_info: Dict[str, Any] = None) -> str:
        """Create new session with secure ID"""
        session_id = secrets.token_urlsafe(32)  # 256-bit security
        
        if client_info is None:
            client_info = {"source": "mcp_client", "created": datetime.now().isoformat()}
            
        session = SessionContext(session_id, client_info, self.rate_limiter, self.quota_manager)
        self.sessions[session_id] = session
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"session:{session_id}",
                    session.resource_limits["session_timeout"],
                    f"created:{session.created_at.isoformat()}"
                )
            except Exception as e:
                logger.warning(f"Failed to store session in Redis: {e}")
        
        logger.info(f"🆔 Created session {session_id[:8]}... for {client_info.get('source', 'unknown')}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get existing session"""
        if not session_id:
            return None
            
        # Check in-memory first
        session = self.sessions.get(session_id)
        if session and not session.is_expired():
            return session
            
        # Remove expired session
        if session and session.is_expired():
            await self.cleanup_session(session_id)
            return None
            
        # Check Redis backup
        if self.redis_client:
            try:
                exists = await self.redis_client.exists(f"session:{session_id}")
                if exists:
                    # Recreate session from Redis
                    client_info = {"source": "mcp_client_restored"}
                    session = SessionContext(session_id, client_info)
                    self.sessions[session_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Failed to check Redis for session: {e}")
                
        return None
    
    async def get_or_create_session(self, session_id: str = None) -> SessionContext:
        """Get existing session or create new one"""
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session
                
        # Create new session
        new_session_id = await self.create_session()
        return self.sessions[new_session_id]
    
    async def cleanup_session(self, session_id: str):
        """Clean up session and its resources"""
        session = self.sessions.pop(session_id, None)
        if session:
            logger.info(f"🧹 Cleaning up session {session_id[:8]}...")
            
            # Remove from Redis
            if self.redis_client:
                try:
                    await self.redis_client.delete(f"session:{session_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove session from Redis: {e}")
    
    async def cleanup_expired_sessions(self):
        """Periodic cleanup of expired sessions"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
            
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
                
        for session_id in expired_sessions:
            await self.cleanup_session(session_id)
            
        if expired_sessions:
            logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
            
        self._last_cleanup = current_time
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        active_sessions = len(self.sessions)
        total_queries = sum(session.query_count for session in self.sessions.values())
        
        return {
            "active_sessions": active_sessions,
            "total_queries": total_queries,
            "avg_queries_per_session": total_queries / active_sessions if active_sessions > 0 else 0,
            "oldest_session": min(
                (session.created_at for session in self.sessions.values()), 
                default=None
            ),
            "redis_enabled": self.redis_client is not None
        }