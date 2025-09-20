"""
Minimal ServiceContainer implementation for multi-project GraphRAG
"""

import os
import logging
import time
from pathlib import Path
from enum import Enum

import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state for progressive initialization"""
    DISCONNECTED = "disconnected"  # No services connected
    DEGRADED = "degraded"          # Only essential services (Neo4j + Qdrant)
    PARTIAL = "partial"             # Essential + some optional services
    FULL = "full"                   # All services connected


class ServiceContainer:
    """Simplified service container for multi-project GraphRAG support"""

    def __init__(self, context_manager=None, project_name: str = None):
        """
        Initialize with injected ProjectContextManager (ADR-0044)

        Args:
            context_manager: ProjectContextManager instance (dependency injection)
            project_name: Optional project name override (deprecated, use context_manager)
        """
        # ADR-0044: Use injected context manager if provided
        if context_manager:
            # Skip isinstance check - causes module identity issues
            # from servers.services.project_context_manager import ProjectContextManager
            # if not isinstance(context_manager, ProjectContextManager):
            #     raise TypeError("context_manager must be a ProjectContextManager instance")
            self.context_manager = context_manager
            self.project_name = context_manager.current_project or "default"
            self.project_path = context_manager.current_project_path
            logger.info(f"🔗 ServiceContainer using injected context for project: {self.project_name}")
        else:
            # Fallback for backward compatibility
            self.context_manager = None
            self.project_name = project_name or "default"
            self.project_path = None
            logger.warning("⚠️ ServiceContainer created without context_manager (deprecated)")

        # Configuration base path for storing project-specific configs
        self.config_base = Path("/app/config/.neural-tools")
        
        # Core database clients - initialized lazily when needed
        self.neo4j_client = None  # Neo4j driver for graph operations
        self.qdrant_client = None  # Qdrant client for vector search
        self.initialized = False  # Track if container is fully initialized
        
        # Circuit breakers for resilient connections (ADR 0018 Phase 4)
        from servers.services.connection_circuit_breaker import ServiceCircuitBreakerManager
        self.circuit_breakers = ServiceCircuitBreakerManager()
        
        # Service instances expected by MCP server protocol
        # These provide the actual functionality for each service
        self.neo4j = None     # Neo4jService wrapper for async operations
        self.qdrant = None    # QdrantService wrapper for vector operations
        self.nomic = None     # NomicService for text embeddings (768-dim)
        
        # Redis clients for resilience architecture (Phase 2)
        # Split into cache and queue for isolation and performance
        self._redis_cache_client = None  # For session cache, rate limiting
        self._redis_queue_client = None  # For job queues, dead letters
        self._job_queue = None           # ARQ job queue for async tasks
        self._dlq_service = None         # Dead letter queue for failed tasks
        
        # Phase 3: Intelligent caching services
        # These optimize performance through predictive caching
        self._cache_warmer = None   # Pre-warms cache with likely queries
        self._cache_metrics = None  # Tracks cache hit rates and patterns
        
        # L9 2025: Connection pooling and session management
        # Critical for handling 15+ concurrent MCP sessions
        self.connection_pools = {}       # Per-service connection pools
        self.session_manager = None      # Manages MCP session isolation
        self._pool_initialized = False   # Ensures pools init only once
        
        # L9 2025: Pool monitoring and metrics
        # Tracks utilization and provides optimization suggestions
        self.pool_monitor = None
        
        # Phase 3: Production security and monitoring
        # Essential for production deployment
        self.auth_service = None    # OAuth2/JWT authentication
        self.error_handler = None   # Structured error handling/logging
        self.health_monitor = None  # Service health checks and metrics
        
        # ADR-0030: Multi-container indexer orchestration
        # Manages lifecycle of per-project indexer containers
        self.indexer_orchestrator = None  # Lazy-loaded when needed
    
    async def get_redis_cache_client(self):
        """Get async Redis client for caching"""
        # Lazy initialization - create client only when first needed
        if self._redis_cache_client is None:
            # Connect to Redis cache container (exposed port 46379)
            # This Redis instance handles:
            # - Session data caching
            # - Rate limiting counters
            # - Temporary computation results
            self._redis_cache_client = redis.Redis(
                host=os.getenv('REDIS_CACHE_HOST', 'localhost'),  # Host machine
                port=int(os.getenv('REDIS_CACHE_PORT', 46379)),   # Exposed port
                password=os.getenv('REDIS_CACHE_PASSWORD', 'cache-secret-key'),
                decode_responses=True  # Auto-decode bytes to str for convenience
            )
        return self._redis_cache_client
    
    async def get_redis_queue_client(self):
        """Get async Redis client for DLQ streams"""
        # Separate Redis instance for queue operations
        if self._redis_queue_client is None:
            # Connect to Redis queue container (exposed port 46380)
            # This Redis instance handles:
            # - Job queues (ARQ)
            # - Dead letter queues for failed tasks
            # - Task scheduling and retry logic
            self._redis_queue_client = redis.Redis(
                host=os.getenv('REDIS_QUEUE_HOST', 'localhost'),   # Host machine
                port=int(os.getenv('REDIS_QUEUE_PORT', 46380)),    # Different port from cache
                password=os.getenv('REDIS_QUEUE_PASSWORD', 'queue-secret-key'),
                decode_responses=True  # String decoding for queue messages
            )
        return self._redis_queue_client
    
    async def get_job_queue(self):
        """Get ARQ job queue using dedicated queue Redis instance"""
        # ARQ is the async job queue framework we use for background tasks
        if self._job_queue is None:
            # Configure ARQ to use the queue Redis instance
            redis_settings = RedisSettings(
                host=os.getenv('REDIS_QUEUE_HOST', 'localhost'),
                port=int(os.getenv('REDIS_QUEUE_PORT', 46380)),
                password=os.getenv('REDIS_QUEUE_PASSWORD', 'queue-secret-key'),
                database=0  # Use db 0 in the dedicated Redis instance
            )
            # Create ARQ connection pool for job submission
            self._job_queue = await create_pool(redis_settings)
        return self._job_queue
    
    async def get_dlq_service(self):
        """Get dead letter queue service using queue Redis instance"""
        # Dead Letter Queue handles failed tasks for retry/analysis
        if self._dlq_service is None:
            # Use the queue Redis client (not cache) for DLQ operations
            redis_client = await self.get_redis_queue_client()
            
            # Import here to avoid circular dependencies
            from servers.services.dead_letter_service import DeadLetterService
            
            # Create DLQ service with Redis backend
            self._dlq_service = DeadLetterService(redis_client)
            
            # Initialize consumer groups for processing failed messages
            await self._dlq_service.initialize()
        return self._dlq_service
    
    async def get_cache_warmer(self):
        """Get cache warming service"""
        # Cache warmer pre-loads frequently accessed data
        if self._cache_warmer is None:
            # Import here to avoid circular dependencies
            from servers.services.cache_warmer import CacheWarmer
            
            # Create warmer with reference to this container
            self._cache_warmer = CacheWarmer(self)
            
            # Initialize warming patterns and schedules
            await self._cache_warmer.initialize()
        return self._cache_warmer
    
    async def get_cache_metrics(self):
        """Get cache metrics service"""
        # Cache metrics tracks hit rates and optimization opportunities
        if self._cache_metrics is None:
            # Import here to avoid circular dependencies
            from servers.services.cache_metrics import CacheMetricsService
            
            # Create metrics service with container reference
            self._cache_metrics = CacheMetricsService(self)
            
            # Initialize metrics collection and reporting
            await self._cache_metrics.initialize()
        return self._cache_metrics
    
    async def initialize_connection_pools(self):
        """Initialize L9 2025 connection pools for MCP sessions"""
        # Prevent double initialization which could leak connections
        if self._pool_initialized:
            return
            
        logger.info(f"🔄 Initializing L9 connection pools for project: {self.project_name}")
        
        # Get pool sizes from environment with L9-tuned defaults
        # These defaults are based on load testing with 15+ concurrent sessions
        neo4j_pool_size = int(os.getenv('NEO4J_POOL_SIZE', '50'))      # Graph queries are complex
        qdrant_pool_size = int(os.getenv('QDRANT_POOL_SIZE', '30'))    # Vector ops are fast
        redis_cache_pool_size = int(os.getenv('REDIS_CACHE_POOL_SIZE', '25'))  # Cache ops are lightweight
        redis_queue_pool_size = int(os.getenv('REDIS_QUEUE_POOL_SIZE', '15'))  # Queue ops less frequent
        
        # Initialize pool structures with L9 specifications
        # Each pool tracks:
        # - max_size: Maximum connections allowed
        # - min_idle: Minimum idle connections to maintain
        # - active: Currently active connections
        # - connections: Map of session_id to connection object
        self.connection_pools = {
            'neo4j': {
                'max_size': neo4j_pool_size,
                'min_idle': max(5, neo4j_pool_size // 10),  # Keep 10% idle minimum
                'active': 0,
                'connections': {}  # session_id -> connection mapping
            },
            'qdrant': {
                'max_size': qdrant_pool_size,
                'min_idle': max(3, qdrant_pool_size // 10),
                'active': 0,
                'connections': {}
            },
            'redis_cache': {
                'max_size': redis_cache_pool_size,
                'min_idle': max(2, redis_cache_pool_size // 10),
                'active': 0,
                'connections': {}
            },
            'redis_queue': {
                'max_size': redis_queue_pool_size,
                'min_idle': max(1, redis_queue_pool_size // 10),
                'active': 0,
                'connections': {}
            }
        }
        
        # Initialize session manager for MCP session isolation
        from servers.services.session_manager import SessionManager
        self.session_manager = SessionManager()
        
        # Initialize pool monitor for utilization tracking
        from servers.services.pool_monitor import PoolMonitor
        self.pool_monitor = PoolMonitor(self)
        await self.pool_monitor.initialize()
        
        logger.info(f"✅ L9 connection pools initialized: Neo4j={neo4j_pool_size}, Qdrant={qdrant_pool_size}, Redis={redis_cache_pool_size}/{redis_queue_pool_size}")
        self._pool_initialized = True
    
    async def initialize_security_services(self):
        """Initialize Phase 3 security and monitoring services"""
        logger.info("🔐 Initializing Phase 3 security and monitoring services...")
        
        # Initialize authentication service for OAuth2/JWT
        try:
            from servers.services.auth_service import AuthenticationService
            
            # Use Redis cache for storing auth tokens and sessions
            redis_client = await self.get_redis_cache_client()
            
            # Create auth service with Redis backend
            self.auth_service = AuthenticationService(redis_client)
            logger.info("✅ Authentication service initialized")
        except Exception as e:
            # Auth service is optional for local development
            logger.error(f"Failed to initialize auth service: {e}")
            self.auth_service = None
        
        # Initialize error handler for structured logging
        try:
            from servers.services.error_handler import ErrorHandler
            
            # Use Redis for error metrics and alerting
            redis_client = await self.get_redis_cache_client()
            
            # Create error handler with Redis backend
            self.error_handler = ErrorHandler(redis_client)
            logger.info("✅ Error handler initialized")
        except Exception as e:
            # Error handler is optional but recommended
            logger.error(f"Failed to initialize error handler: {e}")
            self.error_handler = None
        
        # Initialize health monitor for service health checks
        try:
            from servers.services.health_monitor import HealthMonitor
            
            # Health monitor needs reference to container for service checks
            self.health_monitor = HealthMonitor(self)
            
            # Initialize health check endpoints
            await self.health_monitor.initialize()
            
            # Start background monitoring (30s intervals)
            await self.health_monitor.start_monitoring()
            logger.info("✅ Health monitor initialized and started")
        except Exception as e:
            # Health monitor is critical for production
            logger.error(f"Failed to initialize health monitor: {e}")
            self.health_monitor = None
        
        logger.info("🎉 Phase 3 security and monitoring services initialized")
    
    async def get_pooled_connection(self, service: str, session_id: str):
        """Get connection from pool for specific session"""
        # Track timing for pool performance metrics
        start_time = time.time()
        success = False
        
        try:
            # Ensure pools are initialized before use
            if not self._pool_initialized:
                await self.initialize_connection_pools()
                
            # Get the pool for the requested service
            pool = self.connection_pools.get(service)
            if not pool:
                raise ValueError(f"Unknown service: {service}")
                
            # Check if this session already has a connection (connection reuse)
            if session_id in pool['connections']:
                success = True
                return pool['connections'][session_id]
                
            # Check if pool has capacity for new connection
            if pool['active'] >= pool['max_size']:
                # Pool exhausted - this triggers circuit breaker in production
                raise Exception(f"Connection pool exhausted for {service} (active: {pool['active']}, max: {pool['max_size']})")
                
            # Create new connection based on service type
            # Each service has its own connection creation logic
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
                
            # Store connection in pool mapped to session ID
            pool['connections'][session_id] = connection
            pool['active'] += 1
            
            logger.debug(f"🔗 Created {service} connection for session {session_id[:8]}... (pool: {pool['active']}/{pool['max_size']})")
            success = True
            return connection
            
        finally:
            # Record metrics for pool optimization analysis
            duration = time.time() - start_time
            if self.pool_monitor:
                self.pool_monitor.record_connection_event(service, success, duration)
    
    async def return_pooled_connection(self, service: str, session_id: str):
        """Return connection to pool (session cleanup)"""
        # Get the pool for this service
        pool = self.connection_pools.get(service)
        if pool and session_id in pool['connections']:
            # Remove connection from pool's session mapping
            connection = pool['connections'].pop(session_id)
            
            # Properly close the connection to prevent resource leaks
            try:
                if hasattr(connection, 'close'):
                    await connection.close()
                elif hasattr(connection, 'aclose'):
                    await connection.aclose()
            except Exception as e:
                logger.warning(f"Error closing {service} connection: {e}")
                
            # Update pool statistics
            pool['active'] -= 1
            logger.debug(f"♻️ Returned {service} connection for session {session_id[:8]}... (pool: {pool['active']}/{pool['max_size']})")
    
    async def _create_neo4j_connection(self):
        """Create new Neo4j connection"""
        from neo4j import GraphDatabase
        
        # Use exposed port 47687 (not internal 7687)
        NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:47687")
        NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")  
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag-password")  # Must match docker-compose
        
        # Create Neo4j driver with connection pooling
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    async def _create_qdrant_connection(self):
        """Create new Qdrant connection"""
        from qdrant_client import QdrantClient
        
        # Use exposed port 46333 (not internal 6333)
        QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
        QDRANT_PORT = int(os.getenv("QDRANT_PORT", "46333"))
        
        # Create Qdrant client with timeout for resilience
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)
    
    async def _create_redis_cache_connection(self):
        """Create new Redis cache connection"""
        # Connect to Redis cache container (exposed port 46379)
        return redis.Redis(
            host=os.getenv('REDIS_CACHE_HOST', 'localhost'),
            port=int(os.getenv('REDIS_CACHE_PORT', 46379)),
            password=os.getenv('REDIS_CACHE_PASSWORD', 'cache-secret-key'),
            decode_responses=True  # Automatic string decoding
        )
    
    async def _create_redis_queue_connection(self):
        """Create new Redis queue connection"""
        # Connect to Redis queue container (exposed port 46380)
        return redis.Redis(
            host=os.getenv('REDIS_QUEUE_HOST', 'localhost'),
            port=int(os.getenv('REDIS_QUEUE_PORT', 46380)),
            password=os.getenv('REDIS_QUEUE_PASSWORD', 'queue-secret-key'),
            decode_responses=True  # Automatic string decoding
        )
        
    def ensure_neo4j_client(self):
        """Initialize REAL Neo4j client connection with circuit breaker"""
        breaker = self.circuit_breakers.get_breaker("neo4j", failure_threshold=3, recovery_timeout=30)
        
        def _connect_neo4j():
            from neo4j import GraphDatabase
            
            # Connect to Neo4j container via exposed port
            # CRITICAL: Use localhost:47687, NOT neo4j:7687 or Docker IPs
            NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:47687")
            NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")  
            NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag-password")
            
            logger.info(f"Connecting to REAL Neo4j at {NEO4J_URI} for project {self.project_name}")
            logger.info(f"   User: {NEO4J_USER}, Password: {'***' + NEO4J_PASSWORD[-4:] if len(NEO4J_PASSWORD) > 4 else '***'}")
            
            # Create driver with connection pooling
            self.neo4j_client = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            
            # Test connection with simple query
            with self.neo4j_client.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info(f"✅ REAL Neo4j connection successful for {self.project_name}")
                    return True
            return False
        
        try:
            # Use circuit breaker for connection attempt
            from servers.services.connection_circuit_breaker import CircuitOpenError
            try:
                # Since this is sync code, we call the breaker synchronously
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, can't use run_until_complete
                    # Just try direct connection for now
                    return _connect_neo4j()
                except RuntimeError:
                    # No running loop, create one
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(breaker.call(_connect_neo4j))
                    return result
            except CircuitOpenError as e:
                logger.warning(f"Neo4j circuit breaker open: {e}")
                return False
                
        except ImportError:
            logger.error("Neo4j driver not available - install with 'pip install neo4j'")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to REAL Neo4j: {e}")
            return False
    
    def ensure_qdrant_client(self):
        """Initialize REAL Qdrant vector database connection"""
        try:
            # Import Qdrant client for vector operations
            from qdrant_client import QdrantClient
            
            # L9 2025: Use exposed ports for host-to-container communication
            # CRITICAL: Use localhost:46333, NOT qdrant:6333 or Docker IPs
            QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
            QDRANT_PORT = int(os.getenv("QDRANT_PORT", "46333"))  # Exposed port, not 6333
            
            logger.info(f"L9 MCP: Using Qdrant at {QDRANT_HOST}:{QDRANT_PORT} (MCP container ports)")

            # Perform connectivity check to verify Qdrant is accessible
            logger.info(f"Connecting to Qdrant for connectivity check at {QDRANT_HOST}:{QDRANT_PORT}")
            tmp_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
            
            # Test connection by fetching collections
            _ = tmp_client.get_collections()
            logger.info("✅ Qdrant connectivity check succeeded")
            
            # Store the client for direct use
            self.qdrant_client = tmp_client
            return True
            
        except ImportError:
            logger.error("Qdrant client not available - install with 'pip install qdrant-client'")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to REAL Qdrant: {e}")
            return False
    
    async def _check_neo4j_health(self) -> bool:
        """Check if Neo4j is healthy and responsive"""
        try:
            if not self.neo4j_client:
                return False
            with self.neo4j_client.session() as session:
                result = session.run("RETURN 1 as health_check")
                return result.single()["health_check"] == 1
        except Exception as e:
            logger.debug(f"Neo4j health check failed: {e}")
            return False
    
    async def _check_qdrant_health(self) -> bool:
        """Check if Qdrant is healthy and responsive"""
        try:
            if not self.qdrant_client:
                return False
            # Try to get collections as health check
            collections = self.qdrant_client.get_collections()
            return collections is not None
        except Exception as e:
            logger.debug(f"Qdrant health check failed: {e}")
            return False
    
    async def _check_redis_cache_health(self) -> bool:
        """Check if Redis cache is healthy"""
        try:
            if not self._redis_cache_client:
                return False
            return self._redis_cache_client.ping()
        except Exception:
            return False  # Redis cache is optional
    
    async def _check_redis_queue_health(self) -> bool:
        """Check if Redis queue is healthy"""
        try:
            if not self._redis_queue_client:
                return False
            return self._redis_queue_client.ping()
        except Exception:
            return False  # Redis queue is optional
    
    async def _check_nomic_health(self) -> bool:
        """Check if Nomic embedding service is healthy"""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                host = os.getenv('EMBEDDING_SERVICE_HOST', 'localhost')
                port = int(os.getenv('EMBEDDING_SERVICE_PORT', 48000))
                response = await client.get(f"http://{host}:{port}/health", timeout=2.0)
                return response.status_code == 200
        except Exception:
            return False  # Nomic is optional for basic functionality
    
    async def wait_for_services_healthy(self, timeout: int = 60) -> dict:
        """Wait for all services to pass health checks
        
        Returns:
            Dict with service health status and overall readiness
        """
        import time
        import asyncio
        start_time = time.time()
        
        health_status = {
            'neo4j': False,
            'qdrant': False,
            'redis_cache': False,
            'redis_queue': False,
            'nomic': False
        }
        
        while time.time() - start_time < timeout:
            # Check all services in parallel
            health_checks = await asyncio.gather(
                self._check_neo4j_health(),
                self._check_qdrant_health(),
                self._check_redis_cache_health(),
                self._check_redis_queue_health(),
                self._check_nomic_health(),
                return_exceptions=True
            )
            
            health_status = {
                'neo4j': health_checks[0] if not isinstance(health_checks[0], Exception) else False,
                'qdrant': health_checks[1] if not isinstance(health_checks[1], Exception) else False,
                'redis_cache': health_checks[2] if not isinstance(health_checks[2], Exception) else False,
                'redis_queue': health_checks[3] if not isinstance(health_checks[3], Exception) else False,
                'nomic': health_checks[4] if not isinstance(health_checks[4], Exception) else False
            }
            
            # Check if essential services are ready
            essential_ready = health_status['neo4j'] and health_status['qdrant']
            
            if essential_ready:
                optional_ready = sum([health_status['redis_cache'], health_status['redis_queue'], health_status['nomic']])
                logger.info(f"✅ Essential services healthy. Optional services: {optional_ready}/3")
                health_status['all_ready'] = essential_ready
                health_status['optional_count'] = optional_ready
                return health_status
            
            # Log current status
            unhealthy = [k for k, v in health_status.items() if not v]
            logger.debug(f"Waiting for services: {', '.join(unhealthy)}")
            await asyncio.sleep(2)
        
        logger.error(f"Services failed to become healthy within {timeout}s")
        health_status['all_ready'] = False
        return health_status
    
    async def progressive_initialization(self, essential_timeout: int = 30, optional_timeout: int = 10) -> ConnectionState:
        """Initialize services progressively with different timeouts
        
        Args:
            essential_timeout: Max time to wait for essential services (Neo4j, Qdrant)
            optional_timeout: Max time to wait for optional services
            
        Returns:
            ConnectionState indicating level of connectivity
        """
        import asyncio
        
        logger.info("Starting progressive service initialization...")
        
        # Phase 1: Connect essential services (Neo4j + Qdrant)
        essential_start = time.time()
        essential_connected = False
        
        while time.time() - essential_start < essential_timeout:
            neo4j_ok = self.ensure_neo4j_client()
            qdrant_ok = self.ensure_qdrant_client()
            
            if neo4j_ok and qdrant_ok:
                essential_connected = True
                logger.info("✅ Essential services connected (Neo4j + Qdrant)")
                break
                
            await asyncio.sleep(2)
            logger.debug(f"Waiting for essential services... Neo4j: {neo4j_ok}, Qdrant: {qdrant_ok}")
        
        if not essential_connected:
            logger.error("Failed to connect essential services")
            return ConnectionState.DISCONNECTED
        
        # Initialize essential service wrappers
        from servers.services.neo4j_service import Neo4jService
        from servers.services.qdrant_service import QdrantService
        
        self.neo4j = Neo4jService(self.project_name)
        self.neo4j.set_service_container(self)
        
        self.qdrant = QdrantService(self.project_name)
        self.qdrant.set_service_container(self)
        
        # Phase 2: Try to connect optional services with shorter timeout
        optional_start = time.time()
        optional_count = 0
        
        while time.time() - optional_start < optional_timeout:
            # Try to connect optional services
            health_checks = await asyncio.gather(
                self._check_redis_cache_health(),
                self._check_redis_queue_health(),
                self._check_nomic_health(),
                return_exceptions=True
            )
            
            redis_cache_ok = health_checks[0] if not isinstance(health_checks[0], Exception) else False
            redis_queue_ok = health_checks[1] if not isinstance(health_checks[1], Exception) else False
            nomic_ok = health_checks[2] if not isinstance(health_checks[2], Exception) else False
            
            optional_count = sum([redis_cache_ok, redis_queue_ok, nomic_ok])
            
            if optional_count == 3:
                logger.info("✅ All optional services connected")
                break
            elif optional_count > 0:
                logger.info(f"⚡ {optional_count}/3 optional services connected")
            
            if time.time() - optional_start < optional_timeout - 2:
                await asyncio.sleep(2)
            else:
                break
        
        # Initialize Nomic if available
        if nomic_ok:
            from servers.services.nomic_service import NomicService
            self.nomic = NomicService(self.project_name)
            logger.info("✅ Nomic embedding service initialized")
        else:
            self.nomic = None
            logger.warning("⚠️ Nomic service unavailable - embeddings will be limited")
        
        # Determine final connection state
        self.initialized = True
        
        if optional_count == 3:
            return ConnectionState.FULL
        elif optional_count > 0:
            return ConnectionState.PARTIAL
        else:
            return ConnectionState.DEGRADED
    
    def initialize(self, retry_on_failure: bool = True) -> bool:
        """Initialize all services for this project
        
        Args:
            retry_on_failure: If True, retry connection with exponential backoff
        """
        if self.initialized:
            return True
            
        logger.info(f"Initializing ServiceContainer for project: {self.project_name}")
        
        # Try with retry logic if enabled
        if retry_on_failure:
            import time
            max_retries = 5
            base_delay = 2  # seconds
            
            for attempt in range(max_retries):
                neo4j_ok = self.ensure_neo4j_client()
                qdrant_ok = self.ensure_qdrant_client()
                
                if neo4j_ok and qdrant_ok:
                    logger.info(f"✅ Services connected on attempt {attempt + 1}/{max_retries}")
                    break
                    
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16 seconds
                    logger.warning(f"⏳ Services not ready (Neo4j: {neo4j_ok}, Qdrant: {qdrant_ok}), retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    # IMPORTANT: Using synchronous sleep here because this is called from sync context
                    # The async version is in initialize_with_retry() method
                    time.sleep(delay)
            else:
                logger.error(f"Failed to connect to services after {max_retries} attempts")
        else:
            # Single attempt without retry
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
        """Async version of initialize for MCP server compatibility
        
        This method includes retry logic with exponential backoff to handle
        Docker container startup delays (ADR 0018)
        """
        # Use the existing initialize method with retry logic
        base_init = self.initialize(retry_on_failure=True)
        
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
    
    async def ensure_indexer_running(self, project_path: str = None):
        """
        Ensure indexer container is running for this project
        Implements ADR-0030 multi-container orchestration
        Uses ProjectContextManager for dynamic project detection (ADR-0034)
        
        Args:
            project_path: Absolute path to project directory
                         If None, uses current working directory
        
        Returns:
            Container ID of the running indexer
        """
        if not self.indexer_orchestrator:
            from servers.services.indexer_orchestrator import IndexerOrchestrator
            # ADR-0044: Pass context manager to orchestrator
            if self.context_manager:
                self.indexer_orchestrator = IndexerOrchestrator(context_manager=self.context_manager)
            else:
                # Fallback without context manager
                self.indexer_orchestrator = IndexerOrchestrator()
            await self.indexer_orchestrator.initialize()
            logger.info("IndexerOrchestrator initialized with dependency injection")
        
        # ADR-0044: Use injected context manager, don't create new instance!
        project_info = {}  # Initialize for later use
        if self.context_manager:
            # Use the injected context manager
            current_project_name = self.context_manager.current_project
            current_project_path = str(self.context_manager.current_project_path)
            logger.info(f"✅ Using injected context: project={current_project_name}")
            # Create project_info for consistency
            project_info = {
                "project": current_project_name,
                "path": current_project_path,
                "method": "injected",
                "confidence": 1.0
            }
        else:
            # Fallback: Get from singleton (should not happen in normal flow)
            from servers.services.project_context_manager import get_project_context_manager
            project_manager = await get_project_context_manager()
            project_info = await project_manager.get_current_project()
            current_project_name = project_info["project"]
            current_project_path = project_info["path"]
            logger.warning("⚠️ Had to get singleton context manager in ensure_indexer_running")

        # Use detected path if no path provided
        if not project_path:
            project_path = current_project_path

        logger.info(f"🔄 [Container Sync] Ensuring indexer for project: {current_project_name}")
        logger.info(f"🔄 [Container Sync] Project path: {project_path}")
        logger.info(f"🔄 [Container Sync] Detection method: {project_info.get('method', 'unknown')}")
        logger.info(f"🔄 [Container Sync] Detection confidence: {project_info.get('confidence', 0.0)}")
        
        # Validate project detection before container spawn
        if current_project_name == "default" and project_info.get('confidence', 0.0) < 0.5:
            logger.warning("⚠️ [Container Sync] Low confidence project detection - using fallback")
        
        # Ensure indexer is running with detected project name
        container_id = await self.indexer_orchestrator.ensure_indexer(
            current_project_name,
            project_path
        )
        
        logger.info(f"✅ [Container Sync] Indexer container for {current_project_name} is running: {container_id[:12]}")
        return container_id
    
    async def get_indexer_status(self):
        """
        Get status of all running indexers

        Returns:
            Dict with indexer status information
        """
        if not self.indexer_orchestrator:
            return {"status": "orchestrator_not_initialized"}

        return await self.indexer_orchestrator.get_status()

    async def teardown(self):
        """Gracefully close all managed connections and resources

        Implements ADR-0043: Project Context Lifecycle Management
        Ensures proper cleanup when switching projects to prevent:
        - Connection leaks
        - Stale state retention
        - Cross-project data contamination
        """
        logger.info("Tearing down service container...")

        # Close Neo4j async driver (validated by Gemini)
        try:
            if self.neo4j and hasattr(self.neo4j, 'client') and self.neo4j.client.driver:
                await self.neo4j.client.driver.close()
                logger.info("Neo4j connection closed")
        except Exception as e:
            logger.error(f"Failed to close Neo4j connection: {e}")

        # Close Qdrant client (Grok: handle both sync and async)
        try:
            if hasattr(self, 'qdrant') and self.qdrant:
                if hasattr(self.qdrant, 'close'):
                    # Check if async client
                    if 'Async' in str(type(self.qdrant)):
                        await self.qdrant.close()
                    else:
                        self.qdrant.close()
                    logger.info("Qdrant client closed")
        except Exception as e:
            logger.error(f"Failed to close Qdrant client: {e}")

        # Clear caches
        if hasattr(self, 'cache'):
            self.cache.clear()
            logger.info("Caches cleared")

        # Close Redis connections
        try:
            if self._redis_cache_client:
                await self._redis_cache_client.close()
                logger.info("Redis cache connection closed")
        except Exception as e:
            logger.error(f"Failed to close Redis cache: {e}")

        try:
            if self._redis_queue_client:
                await self._redis_queue_client.close()
                logger.info("Redis queue connection closed")
        except Exception as e:
            logger.error(f"Failed to close Redis queue: {e}")

        # Close job queue
        try:
            if self._job_queue:
                await self._job_queue.close()
                logger.info("Job queue closed")
        except Exception as e:
            logger.error(f"Failed to close job queue: {e}")

        # Stop indexer orchestrator
        try:
            if self.indexer_orchestrator:
                await self.indexer_orchestrator.cleanup()
                logger.info("Indexer orchestrator cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup indexer orchestrator: {e}")

        # Clear all references
        self.neo4j = None
        self.qdrant = None
        self.nomic = None
        self._redis_cache_client = None
        self._redis_queue_client = None
        self._job_queue = None
        self._dlq_service = None
        self._cache_warmer = None
        self._cache_metrics = None
        self.indexer_orchestrator = None
        self.initialized = False
        self._pool_initialized = False

        logger.info("Service container teardown complete")


# ALL MOCKS REMOVED! 
# This service container now uses ONLY REAL service connections:
# - REAL Neo4j GraphDatabase connection via docker-compose  
# - REAL Qdrant vector database connection via docker-compose
# - REAL Nomic embedding service connection via HTTP API
