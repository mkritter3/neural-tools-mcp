# ADR-0010: Redis-Based Resilience Architecture for Neural Tools

**Date**: 2025-09-09
**Status**: Proposed
**Decision Makers**: L9 Engineering Team
**Consulted**: Gemini-2.5-Pro architectural analysis
**Informed**: Neural Tools MCP stakeholders

## Context

The Neural Tools MCP server currently lacks resilience mechanisms for handling service failures, job persistence, and retry logic. Analysis of production patterns reveals several critical gaps:

1. **No Job Persistence**: Embedding requests are processed synchronously with no retry mechanism if Nomic service fails
2. **No Dead Letter Queue**: Failed requests are lost entirely, requiring manual reprocessing
3. **Limited Caching**: No intelligent caching of embedding results or intermediate processing states
4. **Fragile Dependencies**: Single points of failure in Neo4j, Qdrant, and Nomic services

Gemini-2.5-Pro architectural analysis identified three high-impact improvements for system resilience using Redis as the backbone.

## Decision

Implement a comprehensive Redis-based resilience architecture in three phases:

### Phase 1: Persistent Job Queues (Priority: Critical)
- Redis-backed job queuing using ARQ (python-arq/arq) library
- Persistent storage of embedding requests with automatic retry logic
- Job deduplication to prevent duplicate processing

### Phase 2: Dead Letter Queue & Error Handling (Priority: High)
- Redis Streams for failed job processing and analysis
- Comprehensive error categorization and retry policies
- Administrative interfaces for manual job recovery

### Phase 3: Intelligent Caching Layer (Priority: Medium)
- Multi-tier Redis caching for embeddings, Neo4j queries, and processed results
- TTL-based cache invalidation with intelligent refresh
- Cache warming strategies for frequently accessed data

## Technical Implementation Plan

### Phase 1: Redis Job Queues (Weeks 1-2)

#### 1.1 Infrastructure Integration - SEPARATE REDIS INSTANCES
```yaml
# docker-compose.yml - SEPARATE QUEUE VS CACHE REDIS
# Critical: Logical DBs don't isolate memory. Cache evictions can kill queue data.
services:
  # Redis for job queues - NO EVICTION, durable persistence
  redis-queue:
    image: redis:7-alpine
    ports:
      - "46380:6379"
    volumes:
      - redis_queue_data:/data
    command: >
      redis-server
      --appendonly yes
      --save 60 1
      --appendfsync everysec
      --no-appendfsync-on-rewrite yes
      --maxmemory-policy noeviction
      --requirepass ${REDIS_QUEUE_PASSWORD:-queue-secret-key}
    environment:
      - REDIS_PASSWORD=${REDIS_QUEUE_PASSWORD:-queue-secret-key}
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_QUEUE_PASSWORD:-queue-secret-key}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Redis for caching - EVICTION ENABLED, less critical persistence  
  redis-cache:
    image: redis:7-alpine
    ports:
      - "46379:6379"  # Keep existing port for cache
    volumes:
      - redis_cache_data:/data
    command: >
      redis-server
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --save 300 10
      --requirepass ${REDIS_CACHE_PASSWORD:-cache-secret-key}
    environment:
      - REDIS_PASSWORD=${REDIS_CACHE_PASSWORD:-cache-secret-key}
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_CACHE_PASSWORD:-cache-secret-key}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  redis_queue_data:  # Durable queue storage
  redis_cache_data:  # Cache storage
```

#### 1.2 ARQ Integration into ServiceContainer
**Research Status**: ARQ library verified current as of 2025 via Context7
- Job queuing with built-in retry logic and job uniqueness
- Async/await pattern compatible with existing MCP architecture
- Redis persistence ensures job survival across service restarts

```python
# neural-tools/src/servers/services/service_container.py - FIXED ASYNC + SEPARATE INSTANCES
import redis.asyncio as redis
from arq import create_pool
from arq.connections import RedisSettings

class ServiceContainer:
    def __init__(self):
        # ... existing initialization ...
        self._redis_cache_client = None
        self._redis_queue_client = None  
        self._job_queue = None
    
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
```

#### 1.3 Job Processing Integration
```python
# neural-tools/src/servers/services/embedding_service.py - ENHANCE WITH DEDUPLICATION
import hashlib
import json
from typing import List, Optional, Dict, Any
from arq.jobs import Job

class EmbeddingService:
    def __init__(self, service_container):
        self.container = service_container
        self.nomic_client = service_container.get_nomic_client()
        
    def _generate_job_id(self, text: str, model: str = "nomic-v2", options: Dict = None) -> str:
        """Generate deterministic job ID for deduplication"""
        options = options or {}
        content = f"{model}:{text}:{json.dumps(options, sort_keys=True)}"
        return f"embed_{hashlib.sha256(content.encode()).hexdigest()[:16]}"
        
    async def get_embedding_with_queue(self, text: str, model: str = "nomic-v2") -> Dict[str, Any]:
        """Enhanced embedding with job queue fallback and result delivery"""
        try:
            # Try direct call first (existing behavior)
            embedding = await self.nomic_client.get_embedding(text)
            return {"status": "success", "embedding": embedding, "source": "direct"}
        except (ConnectionError, TimeoutError) as e:
            # Fallback to queue for resilience
            job_queue = await self.container.get_job_queue()
            job_id = self._generate_job_id(text, model)
            
            # Check if job already exists/completed
            redis_cache = await self.container.get_redis_cache_client()
            result_key = f"l9:prod:neural_tools:job_result:{job_id}"
            cached_result = await redis_cache.get(result_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Enqueue with deduplication
            job: Job = await job_queue.enqueue_job(
                'process_embedding_job',
                text,
                model,
                _job_id=job_id,  # ARQ deduplication
                _job_timeout=300,
                _defer_until=None
            )
            
            # Return 202-style response for async processing
            return {
                "status": "queued", 
                "job_id": job_id,
                "message": f"Embedding queued due to service unavailability",
                "poll_url": f"/api/jobs/{job_id}/status"
            }
```

#### 1.4 ARQ Worker Definition & Process Management
```python
# neural-tools/src/workers/embedding_worker.py - COMPLETE WORKER IMPLEMENTATION
import asyncio
import logging
from arq import create_pool, Worker
from arq.connections import RedisSettings
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def process_embedding_job(ctx: Dict[str, Any], text: str, model: str = "nomic-v2") -> Dict[str, Any]:
    """ARQ worker function for processing embeddings with idempotency"""
    job_id = ctx.get('job_id')
    
    try:
        # Get services from context
        embedding_service = ctx['embedding_service']
        redis_cache = ctx['redis_cache']
        
        # Check if result already exists (idempotency)
        result_key = f"l9:prod:neural_tools:job_result:{job_id}"
        cached_result = await redis_cache.get(result_key)
        if cached_result:
            logger.info(f"Job {job_id} already completed, returning cached result")
            return json.loads(cached_result)
        
        # Process embedding
        logger.info(f"Processing embedding job {job_id} for model {model}")
        embedding = await embedding_service.get_embedding(text, model)
        
        # Store result with TTL
        result = {
            "status": "success", 
            "embedding": embedding, 
            "source": "worker",
            "job_id": job_id
        }
        await redis_cache.setex(result_key, 3600, json.dumps(result))  # 1 hour TTL
        
        logger.info(f"Job {job_id} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        # ARQ will handle retry logic based on configuration
        raise

# Worker configuration and startup
class EmbeddingWorker:
    def __init__(self, service_container):
        self.container = service_container
        
    async def startup(self, ctx: Dict[str, Any]):
        """Initialize worker context with services"""
        ctx['embedding_service'] = self.container.nomic
        ctx['redis_cache'] = await self.container.get_redis_cache_client()
        ctx['dlq_service'] = await self.container.get_dlq_service()
        logger.info("ARQ worker context initialized")
        
    async def shutdown(self, ctx: Dict[str, Any]):
        """Cleanup worker resources"""
        # Close Redis connections if needed
        logger.info("ARQ worker shutting down")

# Entry point for ARQ worker process
async def main():
    """ARQ worker process entry point"""
    from servers.services.service_container import ServiceContainer
    
    # Initialize service container
    container = ServiceContainer()
    await container.initialize_all_services()
    
    worker_instance = EmbeddingWorker(container)
    
    redis_settings = RedisSettings(
        host=os.getenv('REDIS_QUEUE_HOST', 'localhost'),
        port=int(os.getenv('REDIS_QUEUE_PORT', 46380)),
        password=os.getenv('REDIS_QUEUE_PASSWORD', 'queue-secret-key'),
    )
    
    worker = Worker(
        functions=[process_embedding_job],
        redis_settings=redis_settings,
        on_startup=worker_instance.startup,
        on_shutdown=worker_instance.shutdown,
        max_jobs=10,  # Concurrency limit
        job_timeout=300,  # 5 minute timeout
        keep_result=3600,  # Keep results for 1 hour
    )
    
    await worker.run()

if __name__ == '__main__':
    asyncio.run(main())
```

#### 1.5 Backpressure & Queue Management
```python
# neural-tools/src/servers/services/queue_manager.py - BACKPRESSURE IMPLEMENTATION
import asyncio
import time
from typing import Optional, Dict, Any

class QueueManager:
    def __init__(self, service_container):
        self.container = service_container
        self.max_queue_depth = int(os.getenv('MAX_QUEUE_DEPTH', 1000))
        self.backpressure_threshold = int(os.getenv('BACKPRESSURE_THRESHOLD', 800))
        
    async def get_queue_depth(self) -> int:
        """Get current queue depth from ARQ"""
        redis_queue = await self.container.get_redis_queue_client()
        # ARQ uses multiple keys, check main queue depth
        queue_length = await redis_queue.llen('arq:queue:default')
        return queue_length
        
    async def should_apply_backpressure(self) -> tuple[bool, str]:
        """Check if backpressure should be applied"""
        current_depth = await self.get_queue_depth()
        
        if current_depth >= self.max_queue_depth:
            return True, f"Queue full ({current_depth}/{self.max_queue_depth})"
        elif current_depth >= self.backpressure_threshold:
            return True, f"Queue under pressure ({current_depth}/{self.max_queue_depth})"
        else:
            return False, f"Queue healthy ({current_depth}/{self.max_queue_depth})"
            
    async def enqueue_with_backpressure(self, job_func: str, *args, **kwargs) -> Dict[str, Any]:
        """Enqueue job with backpressure control"""
        should_throttle, reason = await self.should_apply_backpressure()
        
        if should_throttle:
            # Return 429 Too Many Requests equivalent
            return {
                "status": "throttled",
                "error": "System overloaded", 
                "reason": reason,
                "retry_after": 30  # Seconds
            }
        
        # Proceed with normal enqueue
        job_queue = await self.container.get_job_queue()
        job = await job_queue.enqueue_job(job_func, *args, **kwargs)
        
        return {
            "status": "queued",
            "job_id": job.job_id,
            "queue_position": await self.get_queue_depth()
        }
```

### Phase 2: Dead Letter Queue (Weeks 3-4)

#### 2.1 Redis Streams Integration - ENHANCE ServiceContainer
**Research Status**: Redis Streams patterns verified via Context7 redis-py documentation
- XADD for adding failed jobs to error stream  
- Consumer groups for processing different error types
- Message acknowledgment for processed errors

```python
# neural-tools/src/servers/services/service_container.py - ADD TO EXISTING
import time

class ServiceContainer:
    # ... existing methods ...
    
    async def get_dlq_service(self):
        """Get dead letter queue service using queue Redis instance"""
        if not hasattr(self, '_dlq_service'):
            redis_client = await self.get_redis_queue_client()  # Use queue Redis
            self._dlq_service = DeadLetterService(redis_client)
        return self._dlq_service

class DeadLetterService:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.dlq_stream = "l9:prod:neural_tools:embedding_failures"  # Standardized key
        
    async def add_to_dlq(self, job_data: dict, error: Exception, retry_count: int = 0):
        """Add failed job to dead letter queue with proper data types"""
        # Redis Streams require string values - convert all fields
        stream_data = {
            'job_id': str(job_data.get('job_id', 'unknown')),
            'original_text': str(job_data.get('text', ''))[:1000],  # Truncate long text
            'model': str(job_data.get('model', 'nomic-v2')),
            'error_type': type(error).__name__,
            'error_message': str(error)[:500],  # Truncate long errors
            'retry_count': str(retry_count),
            'timestamp': str(int(time.time())),  # Integer timestamp as string
            'environment': 'prod',
            'service': 'neural_tools'
        }
        
        # Add to stream with automatic trimming to prevent unbounded growth
        stream_id = await self.redis.xadd(
            self.dlq_stream,
            stream_data,
            maxlen=100000,  # Keep last 100k failed jobs
            approximate=True  # Use ~ for efficiency
        )
        return stream_id
        
    async def get_dlq_stats(self) -> dict:
        """Get DLQ statistics for monitoring"""
        stream_length = await self.redis.xlen(self.dlq_stream)
        
        # Get error type distribution from recent entries
        recent_entries = await self.redis.xrevrange(self.dlq_stream, count=1000)
        error_types = {}
        for stream_id, fields in recent_entries:
            error_type = fields.get('error_type', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        return {
            'total_failures': stream_length,
            'error_type_distribution': error_types,
            'recent_sample_size': len(recent_entries)
        }
```

#### 2.2 Error Classification & Recovery
```python
class ErrorHandler:
    ERROR_POLICIES = {
        'ConnectionError': {'max_retries': 5, 'backoff': 'exponential'},
        'TimeoutError': {'max_retries': 3, 'backoff': 'linear'},
        'ValidationError': {'max_retries': 0, 'requires_manual': True}
    }
    
    async def handle_job_failure(self, job, error):
        """Intelligent error handling with retry policies"""
        error_type = type(error).__name__
        policy = self.ERROR_POLICIES.get(error_type, {'max_retries': 1})
        
        if job.retry_count < policy['max_retries']:
            await self.schedule_retry(job, policy['backoff'])
        else:
            await self.dlq_service.add_to_dlq(job.data, error)
```

### Phase 3: Intelligent Caching (Weeks 5-6)

#### 3.1 Cache Integration - ENHANCE Existing Services
```python
# neural-tools/src/servers/services/embedding_service.py - ENHANCED CACHING WITH MODEL VERSIONING
import hashlib
import json
from typing import List, Optional

class EmbeddingService:
    def __init__(self, service_container):
        self.container = service_container
        self.nomic_client = service_container.get_nomic_client()
        
    def _generate_cache_key(self, text: str, model: str = "nomic-v2", version: str = "1.0") -> str:
        """Generate cache key with model and version to prevent contamination"""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"l9:prod:neural_tools:embeddings:{model}:{version}:{content_hash}"
        
    async def get_embedding(self, text: str, model: str = "nomic-v2") -> List[float]:
        """Enhanced with intelligent caching using separate cache Redis"""
        redis_cache = await self.container.get_redis_cache_client()
        
        # Check cache first with model-specific key
        cache_key = self._generate_cache_key(text, model)
        cached = await redis_cache.get(cache_key)
        if cached:
            # Cache hit - refresh TTL and return
            await redis_cache.expire(cache_key, 86400)  
            return json.loads(cached)
            
        # Cache miss - get embedding (existing flow)
        try:
            embedding = await self.nomic_client.get_embedding(text)
            # Cache result with 24 hour TTL
            await redis_cache.setex(cache_key, 86400, json.dumps(embedding))
            return embedding
        except Exception as e:
            # Queue fallback logic handled in get_embedding_with_queue
            raise
            
    async def warm_cache_for_frequent_queries(self, queries: List[str], model: str = "nomic-v2"):
        """Pre-populate cache with frequently accessed queries"""
        for query in queries:
            cache_key = self._generate_cache_key(query, model)
            cached = await redis_cache.get(cache_key)
            if not cached:
                try:
                    embedding = await self.nomic_client.get_embedding(query)
                    await redis_cache.setex(cache_key, 86400, json.dumps(embedding))
                except Exception:
                    continue  # Skip failed cache warming
                    
    async def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        redis_cache = await self.container.get_redis_cache_client()
        info = await redis_cache.info('stats')
        
        return {
            'cache_hits': info.get('keyspace_hits', 0),
            'cache_misses': info.get('keyspace_misses', 0), 
            'hit_ratio': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)),
            'memory_usage': info.get('used_memory_human', '0B')
        }
```

#### 3.2 Cache Warming & Invalidation
```python
class CacheWarmer:
    async def warm_frequent_queries(self):
        """Pre-populate cache with frequently accessed data"""
        # Analytics-driven cache warming
        frequent_queries = await self.get_query_analytics()
        for query in frequent_queries:
            if not await self.cache.has_cached_result(query):
                await self.process_and_cache(query)
```

## Monitoring & Observability

### Key Metrics
1. **Queue Health**: Job queue depth, processing rate, failure rate
2. **Error Patterns**: DLQ volume by error type, manual intervention rate  
3. **Cache Performance**: Hit ratio, cache warming effectiveness, TTL optimization
4. **System Resilience**: Recovery time from failures, job completion rate

### Implementation
```python
# neural-tools/src/monitoring/redis_metrics.py
class RedisMetrics:
    async def collect_queue_metrics(self):
        return {
            'queue_depth': await self.redis.llen('arq:queue'),
            'processing_jobs': await self.get_active_jobs_count(),
            'dlq_volume': await self.redis.xlen('embedding_failures'),
            'cache_hit_ratio': await self.calculate_cache_hit_ratio()
        }
```

## Migration Strategy

### Week 1: Infrastructure Integration
- [ ] Add persistence to existing Redis service in docker-compose  
- [ ] Install ARQ dependencies in neural-tools requirements
- [ ] Enhance ServiceContainer with Redis client and job queue methods

### Week 2: Job Queue Integration  
- [ ] Enhance existing EmbeddingService with queue fallback
- [ ] Implement ARQ worker integration in existing services
- [ ] Add job deduplication logic to existing methods
- [ ] Test job persistence across service restarts

### Week 3: Error Handling Integration
- [ ] Add Redis Streams DLQ to existing ServiceContainer
- [ ] Enhance existing error handling with classification and retry policies  
- [ ] Extend existing MCP endpoints for manual recovery
- [ ] Test failure scenarios and recovery

### Week 4: Dead Letter Queue Refinement
- [ ] Add error analytics to existing monitoring/metrics
- [ ] Implement batch processing for DLQ items
- [ ] Integrate alerting with existing health check system

### Week 5: Caching Layer Enhancement
- [ ] Add intelligent caching to existing EmbeddingService
- [ ] Enhance existing Neo4j and Qdrant services with caching
- [ ] Add cache warming to existing initialization

### Week 6: Optimization & Monitoring Integration
- [ ] Enhance existing metrics collection with Redis metrics
- [ ] Add cache performance to existing health endpoints
- [ ] Optimize TTL policies based on usage patterns
- [ ] Load test complete resilience architecture

## Rollback Plan

Each phase includes a feature flag allowing immediate rollback:

```python
class ResilienceConfig:
    ENABLE_JOB_QUEUE = os.getenv('ENABLE_REDIS_QUEUE', 'false').lower() == 'true'
    ENABLE_DLQ = os.getenv('ENABLE_DLQ', 'false').lower() == 'true'  
    ENABLE_CACHE = os.getenv('ENABLE_REDIS_CACHE', 'false').lower() == 'true'
```

**Rollback triggers:**
- Queue processing latency > 10 seconds p95
- DLQ volume > 100 jobs/hour
- Cache hit ratio < 60%
- Any service availability < 99.5%

## Risk Assessment

### High Risk
- **Redis Single Point of Failure**: Mitigate with Redis Sentinel for HA
- **Queue Overflow**: Implement queue depth monitoring and backpressure

### Medium Risk  
- **Cache Invalidation Complexity**: Start with simple TTL, iterate to smarter policies
- **Job Processing Bottlenecks**: Horizontal scaling of ARQ workers

### Low Risk
- **Memory Usage**: Redis memory optimization and monitoring
- **Network Latency**: Local Redis deployment minimizes impact

## Success Criteria

### Phase 1 Success Metrics
- Zero job loss during service restarts
- < 30 second recovery time from Nomic service failures
- 95% job completion rate within 5 minutes

### Phase 2 Success Metrics  
- All failed jobs captured in DLQ with full context
- < 5% manual intervention rate for recoverable errors
- Mean time to error resolution < 2 hours

### Phase 3 Success Metrics
- > 80% cache hit ratio for embeddings
- 50% reduction in Nomic API calls
- < 100ms p95 latency for cached responses

## Dependencies

### External Services
- **Redis 7.2+**: Core persistence and queuing infrastructure
- **ARQ 0.25+**: Job queuing library (verified current via Context7)
- **redis-py 5.0+**: Python Redis client (verified current via Context7)

### Internal Integration Points
- ServiceContainer: Dependency injection for queue services
- EmbeddingService: Integration with job queue and cache
- MCP Server: Queue status and cache metrics endpoints

## Alternatives Considered

### PostgreSQL + Celery
**Rejected**: Higher operational complexity, less optimal for high-throughput queuing

### RabbitMQ + Custom Retry Logic  
**Rejected**: Additional service dependency, ARQ provides equivalent functionality with Redis

### In-Memory Queuing
**Rejected**: No persistence, fails L9 reliability requirements

---

**Confidence**: 85%

**Key Assumptions**:
1. Redis infrastructure can handle projected queue volume (estimated 1000 jobs/hour peak)
2. ARQ library stability continues through 2025 (verified via Context7 research)

**Verification Requirements**:
- Load testing with realistic embedding job volumes
- Redis memory usage analysis under full queue load
- ARQ worker scaling characteristics validation