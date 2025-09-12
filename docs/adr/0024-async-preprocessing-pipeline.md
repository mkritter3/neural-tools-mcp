# ADR-0024: Async Preprocessing Pipeline with Pattern-Based Metadata Extraction

**Status:** Implemented (with modifications)  
**Date:** September 11, 2025  
**Author:** L9 Engineering Team

## Context

Our current indexing pipeline directly processes files from detection to embedding:
1. File detected → 2. Parse content → 3. Generate embedding → 4. Store in Qdrant

However, research confirms that metadata enrichment BEFORE embedding creates superior semantic representations. ~~We need to insert an LLM metadata tagging stage using Gemma 4B between file detection and embedding generation.~~

**UPDATE (Sep 11, 2025)**: Implemented with pattern-based extraction instead of LLM due to performance constraints.

## Decision

Implement a multi-stage async preprocessing pipeline with queue-based architecture that processes files through metadata enrichment before embedding generation.

**IMPLEMENTED WITH MODIFICATIONS**:
- Pattern-based extraction instead of LLM (100x faster)
- Same queue architecture maintained
- 12 metadata fields extracted deterministically
- No external LLM service dependency

## Architecture

### Pipeline Stages

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  File Watcher   │────▶│  Preprocessing   │────▶│  Metadata       │────▶│   Embedding      │
│   (Watchdog)    │     │     Queue        │     │  Extraction     │     │   Generation     │
└─────────────────┘     └──────────────────┘     │  (Patterns)     │     │   (Nomic)        │
                              │                   └─────────────────┘     └──────────────────┘
                              │                            │                        │
                              ▼                            ▼                        ▼
                        ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
                        │  Redis Queue     │     │  Redis Queue     │     │  Redis Queue     │
                        │  (Raw Files)     │     │  (Tagged Files)  │     │  (Embeddings)    │
                        └──────────────────┘     └──────────────────┘     └──────────────────┘
```

**ACTUAL IMPLEMENTATION**: Pattern-based extraction runs in-process within the indexer, no separate container needed

### Queue Design

Using Redis for distributed queue management with three separate queues:

1. **Raw File Queue** (`neural:queue:raw`)
   - Files pending metadata tagging
   - Priority: recency-based
   - TTL: 1 hour

2. **Tagged File Queue** (`neural:queue:tagged`)
   - Files with metadata, awaiting embedding
   - Includes metadata payload
   - TTL: 30 minutes

3. **Embedding Queue** (`neural:queue:embed`)
   - Ready for vector storage
   - Contains embeddings + metadata
   - TTL: 15 minutes

### Async Processing Components

```python
class AsyncPreprocessingPipeline:
    """
    Orchestrates the complete preprocessing pipeline
    """
    
    def __init__(self, container: ServiceContainer):
        self.container = container
        self.redis_queue = container.redis_queue  # Port 46380
        self.metadata_tagger = MetadataTaggerClient(
            base_url="http://localhost:48001"  # Gemma 4B container
        )
        self.embedding_service = container.nomic_service
        
        # Worker pools for each stage
        self.tagger_workers = 4  # Gemma 4B can handle ~4 concurrent
        self.embedder_workers = 8  # Nomic is faster, can handle more
        
        # Queue names
        self.RAW_QUEUE = "neural:queue:raw"
        self.TAGGED_QUEUE = "neural:queue:tagged"
        self.EMBED_QUEUE = "neural:queue:embed"
        
    async def start_workers(self):
        """Start all async worker pools"""
        tasks = []
        
        # Start metadata tagging workers
        for i in range(self.tagger_workers):
            tasks.append(asyncio.create_task(
                self.metadata_worker(f"tagger-{i}")
            ))
        
        # Start embedding workers
        for i in range(self.embedder_workers):
            tasks.append(asyncio.create_task(
                self.embedding_worker(f"embedder-{i}")
            ))
        
        # Start storage worker
        tasks.append(asyncio.create_task(self.storage_worker()))
        
        await asyncio.gather(*tasks)
    
    async def metadata_worker(self, worker_id: str):
        """Process files from raw queue through metadata tagging"""
        while True:
            try:
                # Fetch from raw queue (blocking pop with timeout)
                item = await self.redis_queue.blpop(
                    self.RAW_QUEUE, 
                    timeout=5
                )
                
                if not item:
                    continue
                
                file_data = json.loads(item[1])
                
                # Tag with Gemma 4B
                metadata = await self.metadata_tagger.tag_code(
                    content=file_data['content'],
                    file_path=file_data['path']
                )
                
                # Enrich file data with metadata
                file_data['metadata'] = metadata.dict()
                file_data['tagged_at'] = datetime.utcnow().isoformat()
                
                # Push to tagged queue
                await self.redis_queue.rpush(
                    self.TAGGED_QUEUE,
                    json.dumps(file_data)
                )
                
                logger.info(f"[{worker_id}] Tagged {file_data['path']}")
                
            except Exception as e:
                logger.error(f"[{worker_id}] Error in metadata tagging: {e}")
                await asyncio.sleep(1)
    
    async def embedding_worker(self, worker_id: str):
        """Process tagged files through embedding generation"""
        while True:
            try:
                # Fetch from tagged queue
                item = await self.redis_queue.blpop(
                    self.TAGGED_QUEUE,
                    timeout=5
                )
                
                if not item:
                    continue
                
                file_data = json.loads(item[1])
                
                # Create composite content for embedding
                # Metadata is embedded WITH content for richer representation
                composite_text = self._create_composite_text(
                    content=file_data['content'],
                    metadata=file_data['metadata']
                )
                
                # Generate embedding
                embedding = await self.embedding_service.embed_text(
                    composite_text
                )
                
                # Prepare for storage
                file_data['embedding'] = embedding
                file_data['embedded_at'] = datetime.utcnow().isoformat()
                
                # Push to embedding queue
                await self.redis_queue.rpush(
                    self.EMBED_QUEUE,
                    json.dumps(file_data)
                )
                
                logger.info(f"[{worker_id}] Embedded {file_data['path']}")
                
            except Exception as e:
                logger.error(f"[{worker_id}] Error in embedding: {e}")
                await asyncio.sleep(1)
    
    def _create_composite_text(self, content: str, metadata: dict) -> str:
        """
        Create composite text that includes metadata for richer embeddings
        Based on 2025 best practices for metadata-enriched embeddings
        """
        composite_parts = []
        
        # Add metadata context
        if metadata.get('status') == 'archived':
            composite_parts.append("[ARCHIVED CODE]")
        elif metadata.get('status') == 'deprecated':
            composite_parts.append("[DEPRECATED]")
        
        if metadata.get('component_type'):
            composite_parts.append(f"Component: {metadata['component_type']}")
        
        if metadata.get('dependencies'):
            composite_parts.append(f"Uses: {', '.join(metadata['dependencies'][:5])}")
        
        # Add main content
        composite_parts.append(content)
        
        # Add extracted questions (if any)
        if metadata.get('answers_questions'):
            composite_parts.append("Can answer: " + "; ".join(metadata['answers_questions'][:3]))
        
        return "\n\n".join(composite_parts)
    
    async def storage_worker(self):
        """Store embedded files in Qdrant"""
        while True:
            try:
                # Fetch from embedding queue
                item = await self.redis_queue.blpop(
                    self.EMBED_QUEUE,
                    timeout=5
                )
                
                if not item:
                    continue
                
                file_data = json.loads(item[1])
                
                # Store in Qdrant with metadata as payload
                await self.container.qdrant_service.upsert(
                    collection_name=f"project_{file_data['project']}_embeddings",
                    points=[{
                        'id': self._generate_point_id(file_data['path']),
                        'vector': file_data['embedding'],
                        'payload': {
                            'content': file_data['content'],
                            'file_path': file_data['path'],
                            'metadata': file_data['metadata'],
                            'indexed_at': datetime.utcnow().isoformat()
                        }
                    }]
                )
                
                logger.info(f"Stored {file_data['path']} in Qdrant")
                
            except Exception as e:
                logger.error(f"Error in storage: {e}")
                await asyncio.sleep(1)
```

### Integration with Existing Indexer

Modify the current `IncrementalIndexer` to queue files instead of directly processing:

```python
class IncrementalIndexer(FileSystemEventHandler):
    """Modified to queue files for async preprocessing pipeline"""
    
    async def _queue_change(self, file_path: str, action: str):
        """Queue file for preprocessing instead of direct processing"""
        if action in ['create', 'update']:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Queue for preprocessing
                file_data = {
                    'path': str(file_path),
                    'content': content,
                    'action': action,
                    'project': self.project_name,
                    'queued_at': datetime.utcnow().isoformat()
                }
                
                # Push to raw queue
                await self.container.redis_queue.rpush(
                    "neural:queue:raw",
                    json.dumps(file_data)
                )
                
                self.metrics['files_queued'] += 1
                logger.info(f"Queued {file_path} for preprocessing")
                
            except Exception as e:
                logger.error(f"Error queuing {file_path}: {e}")
                self.metrics['queue_errors'] += 1
```

### Gemma 4B Container Configuration

Add to `docker-compose.yml`:

```yaml
  gemma-tagger:
    image: ollama/ollama:latest
    container_name: neural-gemma-tagger
    volumes:
      - gemma_models:/root/.ollama
    ports:
      - "48001:11434"
    environment:
      - OLLAMA_HOST=0.0.0.0
    networks:
      - l9-graphrag-network
    deploy:
      resources:
        limits:
          memory: 6G  # Gemma 4B needs ~4.4GB + overhead
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
    command: >
      sh -c "ollama serve & 
             sleep 5 && 
             ollama pull gemma2:4b && 
             wait"
```

### Monitoring & Observability

```python
class PipelineMonitor:
    """Monitor async pipeline health and performance"""
    
    async def get_metrics(self) -> dict:
        return {
            'queue_depths': {
                'raw': await self.redis_queue.llen("neural:queue:raw"),
                'tagged': await self.redis_queue.llen("neural:queue:tagged"),
                'embed': await self.redis_queue.llen("neural:queue:embed")
            },
            'worker_status': {
                'taggers': self.get_worker_status('tagger'),
                'embedders': self.get_worker_status('embedder')
            },
            'throughput': {
                'tagging_per_min': self.tagging_rate,
                'embedding_per_min': self.embedding_rate
            },
            'errors': {
                'tagging_failures': self.tagging_errors,
                'embedding_failures': self.embedding_errors
            }
        }
```

## Benefits

1. **Richer Embeddings**: Metadata is embedded WITH content, creating superior semantic representations
2. **Async Processing**: Non-blocking pipeline with worker pools for each stage
3. **Scalability**: Can scale workers independently based on bottlenecks
4. **Fault Tolerance**: Queue-based design survives crashes and restarts
5. **Observability**: Clear metrics for each pipeline stage

## Consequences

### Positive
- Better search quality with metadata-enriched embeddings
- Can filter archived/deprecated code before it pollutes results
- Scalable architecture that can grow with demand
- Clear separation of concerns

### Negative
- Added complexity with multi-stage pipeline
- Additional latency (metadata tagging adds ~200-500ms per file)
- Requires more infrastructure (Gemma container, Redis queues)
- More failure points to monitor

## Implementation Plan

1. **Phase 1**: Deploy Gemma 4B container with Ollama
2. **Phase 2**: Implement AsyncPreprocessingPipeline class
3. **Phase 3**: Modify IncrementalIndexer to use queues
4. **Phase 4**: Add monitoring and observability
5. **Phase 5**: Performance tuning and optimization

## Performance Targets

- Metadata tagging: <500ms per file
- End-to-end latency: <2 seconds from file change to searchable
- Queue depth: <100 items per queue under normal load
- Worker utilization: 60-80% target

## References

- [Understanding What Matters for LLM Ingestion and Preprocessing (2025)](https://unstructured.io/blog/understanding-what-matters-for-llm-ingestion-and-preprocessing)
- [Metadata Enrichment in RAG Pipelines (2025)](https://www.deepset.ai/blog/leveraging-metadata-in-rag-customization)
- [Async Pipeline Architecture Patterns (2025)](https://arxiv.org/html/2407.11798v1)