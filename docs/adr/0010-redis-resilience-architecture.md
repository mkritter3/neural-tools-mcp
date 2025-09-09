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

#### 1.1 Infrastructure Integration (USE EXISTING REDIS)
```yaml
# docker-compose.yml - NO NEW SERVICES NEEDED
# Use existing Redis service at redis:6379 (mapped to localhost:46379)
# Add persistence to existing Redis:
services:
  redis:
    command: redis-server --appendonly yes --save 60 1  # Add persistence
```

#### 1.2 ARQ Integration into ServiceContainer
**Research Status**: ARQ library verified current as of 2025 via Context7
- Job queuing with built-in retry logic and job uniqueness
- Async/await pattern compatible with existing MCP architecture
- Redis persistence ensures job survival across service restarts

```python
# neural-tools/src/servers/services/service_container.py - INTEGRATE WITH EXISTING
class ServiceContainer:
    def __init__(self):
        # ... existing initialization ...
        self._redis_client = None
        self._job_queue = None
    
    async def get_redis_client(self):
        """Get Redis client using existing infrastructure"""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 46379)),  # Use existing Redis port
                db=0
            )
        return self._redis_client
    
    async def get_job_queue(self):
        """Get ARQ job queue integrated with existing Redis"""
        if self._job_queue is None:
            redis_settings = RedisSettings(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 46379)),
                database=1  # Use db 1 for queues, db 0 for cache
            )
            self._job_queue = await create_pool(redis_settings)
        return self._job_queue
```

#### 1.3 Job Processing Integration
```python
# neural-tools/src/servers/services/embedding_service.py - ENHANCE EXISTING
class EmbeddingService:
    def __init__(self, service_container):
        self.container = service_container
        self.nomic_client = service_container.get_nomic_client()
        
    async def get_embedding_with_queue(self, text: str) -> List[float]:
        """Enhanced embedding with job queue fallback"""
        try:
            # Try direct call first (existing behavior)
            return await self.nomic_client.get_embedding(text)
        except (ConnectionError, TimeoutError) as e:
            # Fallback to queue for resilience
            job_queue = await self.container.get_job_queue()
            job = await job_queue.enqueue_job(
                'process_embedding_job',
                text,
                _job_timeout=300
            )
            # Return job ID for tracking
            raise EmbeddingQueuedError(f"Queued as job {job.job_id}")
```

### Phase 2: Dead Letter Queue (Weeks 3-4)

#### 2.1 Redis Streams Integration - ENHANCE ServiceContainer
**Research Status**: Redis Streams patterns verified via Context7 redis-py documentation
- XADD for adding failed jobs to error stream  
- Consumer groups for processing different error types
- Message acknowledgment for processed errors

```python
# neural-tools/src/servers/services/service_container.py - ADD TO EXISTING
class ServiceContainer:
    # ... existing methods ...
    
    async def get_dlq_service(self):
        """Get dead letter queue service using existing Redis"""
        if not hasattr(self, '_dlq_service'):
            redis_client = await self.get_redis_client()
            self._dlq_service = DeadLetterService(redis_client)
        return self._dlq_service

class DeadLetterService:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.dlq_stream = "neural_tools:embedding_failures"
        
    async def add_to_dlq(self, job_data: dict, error: Exception):
        """Add failed job to dead letter queue using existing Redis"""
        await self.redis.xadd(
            self.dlq_stream,
            {
                'job_id': job_data.get('job_id'),
                'original_text': job_data.get('text'),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'retry_count': job_data.get('retry_count', 0),
                'timestamp': time.time()
            }
        )
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
# neural-tools/src/servers/services/embedding_service.py - ENHANCE EXISTING
class EmbeddingService:
    def __init__(self, service_container):
        self.container = service_container
        self.nomic_client = service_container.get_nomic_client()
        
    async def get_embedding(self, text: str) -> List[float]:
        """Enhanced with caching using existing Redis"""
        redis_client = await self.container.get_redis_client()
        
        # Check cache first
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cache_key = f"neural_tools:embeddings:{text_hash}"
        
        cached = await redis_client.get(cache_key)
        if cached:
            await redis_client.expire(cache_key, 86400)  # Refresh TTL
            return json.loads(cached)
            
        # Get embedding (existing flow)
        try:
            embedding = await self.nomic_client.get_embedding(text)
            # Cache result
            await redis_client.setex(cache_key, 86400, json.dumps(embedding))
            return embedding
        except Exception as e:
            # Existing queue fallback logic...
            raise
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