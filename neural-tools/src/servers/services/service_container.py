"""
Minimal ServiceContainer implementation for multi-project GraphRAG
"""

import os
import logging
import secrets
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings

logger = logging.getLogger(__name__)

class ServiceContainer:
    """Simplified service container for multi-project GraphRAG support"""
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.config_base = Path("/app/config/.neural-tools")
        self.neo4j_client = None
        self.qdrant_client = None
        self.initialized = False
        
        # Add attributes expected by MCP server
        self.neo4j = None
        self.qdrant = None
        self.nomic = None  # Embedding client
        
        # Redis clients for resilience architecture
        self._redis_cache_client = None
        self._redis_queue_client = None  
        self._job_queue = None
        self._dlq_service = None
        
        # Phase 3: Intelligent caching services
        self._cache_warmer = None
        self._cache_metrics = None
        
        # L9 2025: Connection pooling and session management
        self.connection_pools = {}
        self.session_manager = None
        self._pool_initialized = False
        
        # L9 2025: Pool monitoring and metrics
        self.pool_monitor = None
        
        # Phase 3: Production security and monitoring
        self.auth_service = None
        self.error_handler = None
        self.health_monitor = None
    
    async def get_redis_cache_client(self):
        """Get async Redis client for caching"""
        if self._redis_cache_client is None:
            self._redis_cache_client = redis.Redis(
                host=os.getenv('REDIS_CACHE_HOST', 'localhost'),
                port=int(os.getenv('REDIS_CACHE_PORT', 46379)),
                password=os.getenv('REDIS_CACHE_PASSWORD', 'cache-secret-key'),
                decode_responses=True  # Auto-decode bytes to str
            )
        return self._redis_cache_client
    
    async def get_redis_queue_client(self):
        """Get async Redis client for DLQ streams"""
        if self._redis_queue_client is None:
            self._redis_queue_client = redis.Redis(
                host=os.getenv('REDIS_QUEUE_HOST', 'localhost'),
                port=int(os.getenv('REDIS_QUEUE_PORT', 46380)),
                password=os.getenv('REDIS_QUEUE_PASSWORD', 'queue-secret-key'),
                decode_responses=True
            )
        return self._redis_queue_client
    
    async def get_job_queue(self):
        """Get ARQ job queue using dedicated queue Redis instance"""
        if self._job_queue is None:
            redis_settings = RedisSettings(
                host=os.getenv('REDIS_QUEUE_HOST', 'localhost'),
                port=int(os.getenv('REDIS_QUEUE_PORT', 46380)),
                password=os.getenv('REDIS_QUEUE_PASSWORD', 'queue-secret-key'),
                database=0  # Use dedicated Redis instance, db 0
            )
            self._job_queue = await create_pool(redis_settings)
        return self._job_queue
    
    async def get_dlq_service(self):
        """Get dead letter queue service using queue Redis instance"""
        if self._dlq_service is None:
            redis_client = await self.get_redis_queue_client()
            from servers.services.dead_letter_service import DeadLetterService
            self._dlq_service = DeadLetterService(redis_client)
            # Initialize DLQ consumer groups
            await self._dlq_service.initialize()
        return self._dlq_service
    
    async def get_cache_warmer(self):
        """Get cache warming service"""
        if self._cache_warmer is None:
            from servers.services.cache_warmer import CacheWarmer
            self._cache_warmer = CacheWarmer(self)
            await self._cache_warmer.initialize()
        return self._cache_warmer
    
    async def get_cache_metrics(self):
        """Get cache metrics service"""
        if self._cache_metrics is None:
            from servers.services.cache_metrics import CacheMetricsService
            self._cache_metrics = CacheMetricsService(self)
            await self._cache_metrics.initialize()
        return self._cache_metrics
    
    async def initialize_connection_pools(self):
        """Initialize L9 2025 connection pools for MCP sessions"""
        if self._pool_initialized:
            return
            
        logger.info(f"🔄 Initializing L9 connection pools for project: {self.project_name}")
        
        # Get pool sizes from environment (L9 container-tuned defaults)
        neo4j_pool_size = int(os.getenv('NEO4J_POOL_SIZE', '50'))
        qdrant_pool_size = int(os.getenv('QDRANT_POOL_SIZE', '30'))
        redis_cache_pool_size = int(os.getenv('REDIS_CACHE_POOL_SIZE', '25'))
        redis_queue_pool_size = int(os.getenv('REDIS_QUEUE_POOL_SIZE', '15'))
        
        self.connection_pools = {
            'neo4j': {'max_size': neo4j_pool_size, 'min_idle': max(5, neo4j_pool_size // 10), 'active': 0, 'connections': {}},
            'qdrant': {'max_size': qdrant_pool_size, 'min_idle': max(3, qdrant_pool_size // 10), 'active': 0, 'connections': {}},
            'redis_cache': {'max_size': redis_cache_pool_size, 'min_idle': max(2, redis_cache_pool_size // 10), 'active': 0, 'connections': {}},
            'redis_queue': {'max_size': redis_queue_pool_size, 'min_idle': max(1, redis_queue_pool_size // 10), 'active': 0, 'connections': {}}
        }
        
        # Initialize session manager
        from servers.services.session_manager import SessionManager
        self.session_manager = SessionManager()
        
        # Initialize pool monitor
        from servers.services.pool_monitor import PoolMonitor
        self.pool_monitor = PoolMonitor(self)
        await self.pool_monitor.initialize()
        
        logger.info(f"✅ L9 connection pools initialized: Neo4j={neo4j_pool_size}, Qdrant={qdrant_pool_size}, Redis={redis_cache_pool_size}/{redis_queue_pool_size}")
        self._pool_initialized = True
    
    async def initialize_security_services(self):
        """Initialize Phase 3 security and monitoring services"""
        logger.info("🔐 Initializing Phase 3 security and monitoring services...")
        
        # Initialize authentication service
        try:
            from servers.services.auth_service import AuthenticationService
            redis_client = await self.get_redis_cache_client()
            self.auth_service = AuthenticationService(redis_client)
            logger.info("✅ Authentication service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize auth service: {e}")
            self.auth_service = None
        
        # Initialize error handler
        try:
            from servers.services.error_handler import ErrorHandler
            redis_client = await self.get_redis_cache_client()
            self.error_handler = ErrorHandler(redis_client)
            logger.info("✅ Error handler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize error handler: {e}")
            self.error_handler = None
        
        # Initialize health monitor
        try:
            from servers.services.health_monitor import HealthMonitor
            self.health_monitor = HealthMonitor(self)
            await self.health_monitor.initialize()
            await self.health_monitor.start_monitoring()
            logger.info("✅ Health monitor initialized and started")
        except Exception as e:
            logger.error(f"Failed to initialize health monitor: {e}")
            self.health_monitor = None
        
        logger.info("🎉 Phase 3 security and monitoring services initialized")
    
    async def get_pooled_connection(self, service: str, session_id: str):
        """Get connection from pool for specific session"""
        start_time = time.time()
        success = False
        
        try:
            if not self._pool_initialized:
                await self.initialize_connection_pools()
                
            pool = self.connection_pools.get(service)
            if not pool:
                raise ValueError(f"Unknown service: {service}")
                
            # Check if session already has a connection
            if session_id in pool['connections']:
                success = True
                return pool['connections'][session_id]
                
            # Check pool capacity
            if pool['active'] >= pool['max_size']:
                raise Exception(f"Connection pool exhausted for {service} (active: {pool['active']}, max: {pool['max_size']})")
                
            # Create new connection based on service type
            if service == 'neo4j':
                connection = await self._create_neo4j_connection()
            elif service == 'qdrant':
                connection = await self._create_qdrant_connection()
            elif service == 'redis_cache':
                connection = await self._create_redis_cache_connection()
            elif service == 'redis_queue':
                connection = await self._create_redis_queue_connection()
            else:
                raise ValueError(f"Unknown service: {service}")
                
            # Store in pool
            pool['connections'][session_id] = connection
            pool['active'] += 1
            
            logger.debug(f"🔗 Created {service} connection for session {session_id[:8]}... (pool: {pool['active']}/{pool['max_size']})")
            success = True
            return connection
            
        finally:
            # Record metrics
            duration = time.time() - start_time
            if self.pool_monitor:
                self.pool_monitor.record_connection_event(service, success, duration)
    
    async def return_pooled_connection(self, service: str, session_id: str):
        """Return connection to pool (session cleanup)"""
        pool = self.connection_pools.get(service)
        if pool and session_id in pool['connections']:
            # Close connection
            connection = pool['connections'].pop(session_id)
            try:
                if hasattr(connection, 'close'):
                    await connection.close()
                elif hasattr(connection, 'aclose'):
                    await connection.aclose()
            except Exception as e:
                logger.warning(f"Error closing {service} connection: {e}")
                
            pool['active'] -= 1
            logger.debug(f"♻️ Returned {service} connection for session {session_id[:8]}... (pool: {pool['active']}/{pool['max_size']})")
    
    async def _create_neo4j_connection(self):
        """Create new Neo4j connection"""
        from neo4j import GraphDatabase
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:47687")
        NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")  
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag-password")
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    async def _create_qdrant_connection(self):
        """Create new Qdrant connection"""
        from qdrant_client import QdrantClient
        QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
        QDRANT_PORT = int(os.getenv("QDRANT_PORT", "46333"))
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
    
    async def _create_redis_cache_connection(self):
        """Create new Redis cache connection"""
        return redis.Redis(
            host=os.getenv('REDIS_CACHE_HOST', 'localhost'),
            port=int(os.getenv('REDIS_CACHE_PORT', 46379)),
            password=os.getenv('REDIS_CACHE_PASSWORD', 'cache-secret-key'),
            decode_responses=True
        )
    
    async def _create_redis_queue_connection(self):
        """Create new Redis queue connection"""
        return redis.Redis(
            host=os.getenv('REDIS_QUEUE_HOST', 'localhost'),
            port=int(os.getenv('REDIS_QUEUE_PORT', 46380)),
            password=os.getenv('REDIS_QUEUE_PASSWORD', 'queue-secret-key'),
            decode_responses=True
        )
        
    def ensure_neo4j_client(self):
        """Initialize REAL Neo4j client connection"""
        try:
            from neo4j import GraphDatabase
            
            # Connect to REAL Neo4j instance running on docker-compose
            NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:47687")
            NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")  
            NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag-password")
            
            logger.info(f"Connecting to REAL Neo4j at {NEO4J_URI} for project {self.project_name}")
            
            # Create REAL connection to running Neo4j instance
            self.neo4j_client = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            
            # Test connection
            with self.neo4j_client.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info(f"✅ REAL Neo4j connection successful for {self.project_name}")
                    # Connection successful - service wrapper will be set later
                    return True
            
        except ImportError:
            logger.error("Neo4j driver not available - install with 'pip install neo4j'")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to REAL Neo4j: {e}")
            return False
    
    def ensure_qdrant_client(self):
        """Initialize REAL Qdrant vector database connection"""
        try:
            # Lightweight connectivity check to provide early feedback; wrapper handles real init
            from qdrant_client import QdrantClient
            # L9 2025: Force use of environment variables for MCP connectivity
            QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
            QDRANT_PORT = int(os.getenv("QDRANT_PORT", "46333"))
            
            logger.info(f"L9 MCP: Using Qdrant at {QDRANT_HOST}:{QDRANT_PORT} (MCP container ports)")

            logger.info(f"Connecting to Qdrant for connectivity check at {QDRANT_HOST}:{QDRANT_PORT}")
            tmp_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
            _ = tmp_client.get_collections()
            logger.info("✅ Qdrant connectivity check succeeded")
            # Do not assign raw client to self.qdrant; wrapper will be initialized below
            self.qdrant_client = None
            return True
            
        except ImportError:
            logger.error("Qdrant client not available - install with 'pip install qdrant-client'")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to REAL Qdrant: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize all services for this project"""
        if self.initialized:
            return True
            
        logger.info(f"Initializing ServiceContainer for project: {self.project_name}")
        
        neo4j_ok = self.ensure_neo4j_client()
        qdrant_ok = self.ensure_qdrant_client()
        
        # Initialize REAL service wrappers - NO MOCKS!
        if neo4j_ok:
            from servers.services.neo4j_service import Neo4jService
            self.neo4j = Neo4jService(self.project_name)
            # Connect service to container for cache access
            self.neo4j.set_service_container(self)
            # Initialize the service asynchronously later
        else:
            self.neo4j = None
            
        if qdrant_ok:
            from servers.services.qdrant_service import QdrantService
            self.qdrant = QdrantService(self.project_name)
            # Connect service to container for cache access
            self.qdrant.set_service_container(self)
            # Initialize the service asynchronously later
        else:
            self.qdrant = None
        
        # Initialize REAL Nomic embedding service (service wrapper)
        try:
            from servers.services.nomic_service import NomicService
            self.nomic = NomicService()
            # Connect service to container for queue/cache access
            self.nomic.set_service_container(self)
            logger.info("✅ Nomic service object created with container integration")
        except ImportError as e:
            logger.error(f"Failed to import NomicService: {e}")
            self.nomic = None
        
        self.initialized = True
        return neo4j_ok and qdrant_ok
    
    async def initialize_all_services(self) -> bool:
        """Async version of initialize for MCP server compatibility"""
        base_init = self.initialize()
        
        # Initialize service wrappers asynchronously
        if self.neo4j and hasattr(self.neo4j, 'initialize'):
            neo4j_result = await self.neo4j.initialize()
            if not neo4j_result.get('success'):
                logger.warning(f"Neo4j service initialization failed: {neo4j_result.get('message')}")
                self.neo4j = None
                
        if self.qdrant and hasattr(self.qdrant, 'initialize'):
            qdrant_result = await self.qdrant.initialize()
            if not qdrant_result.get('success'):
                logger.warning(f"Qdrant service initialization failed: {qdrant_result.get('message')}")
                self.qdrant = None
        # Initialize Nomic service wrapper
        if self.nomic and hasattr(self.nomic, 'initialize'):
            nomic_result = await self.nomic.initialize()
            if not nomic_result.get('success'):
                logger.warning(f"Nomic service initialization failed: {nomic_result.get('message')}")
                self.nomic = None

        # Return a dictionary for compatibility with the caller
        return {
            "success": base_init and self.neo4j is not None and self.qdrant is not None and self.nomic is not None,
            "services": {
                "neo4j": self.neo4j is not None,
                "qdrant": self.qdrant is not None,
                "nomic": self.nomic is not None
            }
        }
    
    def get_neo4j_client(self):
        """Get Neo4j client, initializing if needed"""
        if not self.neo4j_client:
            self.ensure_neo4j_client()
        return self.neo4j_client
    
    def get_qdrant_client(self):
        """Get Qdrant client, initializing if needed"""
        if not self.qdrant_client:
            self.ensure_qdrant_client()
        return self.qdrant_client


# ALL MOCKS REMOVED! 
# This service container now uses ONLY REAL service connections:
# - REAL Neo4j GraphDatabase connection via docker-compose  
# - REAL Qdrant vector database connection via docker-compose
# - REAL Nomic embedding service connection via HTTP API
