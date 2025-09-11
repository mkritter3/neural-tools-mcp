# ADR-0014: Incremental Indexing Optimization

**Status:** Proposed  
**Date:** 2025-09-09  
**Deciders:** Engineering Team  

## Context

Our neural indexer currently re-processes files on every change. With tree-sitter extraction enabled, this becomes expensive. We need to optimize the existing `IndexerService` to perform incremental updates, leveraging our Redis cache and file state tracking.

## Decision

Enhance the **existing** `IndexerService` in `neural-tools/src/servers/services/indexer_service.py` to support incremental indexing using content hashing, change detection, and smart cache invalidation.

## Integration Approach

### 1. Enhance Existing IndexerService
```python
# EDIT neural-tools/src/servers/services/indexer_service.py
class IndexerService:
    def __init__(self):
        # Existing initialization...
        self.file_hashes = {}  # NEW: Track file content hashes
        self.symbol_cache = {}  # NEW: Cache extracted symbols
        self.index_state = self._load_index_state()  # NEW: Persistent state
    
    async def _should_reindex(self, file_path: str, content: str) -> bool:
        """Check if file needs reindexing"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check existing state
        if file_path in self.file_hashes:
            if self.file_hashes[file_path] == content_hash:
                return False  # No change
        
        # Store new hash
        self.file_hashes[file_path] = content_hash
        return True
```

### 2. Modify File Processing Pipeline
```python
# EDIT existing process_file method:
async def process_file(self, file_path: str, event_type: str):
    # Existing validation...
    
    if event_type == 'deleted':
        await self._handle_deletion(file_path)
        return
    
    content = await self._read_file(file_path)
    
    # NEW: Skip if unchanged
    if not await self._should_reindex(file_path, content):
        logger.debug(f"Skipping unchanged file: {file_path}")
        return
    
    # NEW: Incremental symbol extraction
    old_symbols = await self._get_cached_symbols(file_path)
    new_symbols = await self._extract_symbols(file_path, content)
    
    # Diff and update only changed symbols
    await self._update_changed_symbols(old_symbols, new_symbols)
```

### 3. Enhance Redis Cache Usage
```python
# EDIT existing redis integration:
async def _get_cached_symbols(self, file_path: str):
    """Retrieve cached symbols from Redis"""
    cache_key = f"symbols:{file_path}"
    cached = await self.redis_cache.get(cache_key)
    if cached:
        return json.loads(cached)
    return None

async def _cache_symbols(self, file_path: str, symbols: List[Dict]):
    """Cache symbols with TTL"""
    cache_key = f"symbols:{file_path}"
    await self.redis_cache.setex(
        cache_key,
        3600,  # 1 hour TTL
        json.dumps(symbols)
    )
```

### 4. Implement Smart Diffing
```python
# ADD to existing indexer_service.py:
async def _update_changed_symbols(
    self,
    old_symbols: Optional[List[Dict]],
    new_symbols: List[Dict]
):
    """Intelligently update only changed symbols"""
    
    if not old_symbols:
        # First time indexing
        await self._index_all_symbols(new_symbols)
        return
    
    # Build lookup maps
    old_map = {s['name']: s for s in old_symbols}
    new_map = {s['name']: s for s in new_symbols}
    
    # Find changes
    added = set(new_map.keys()) - set(old_map.keys())
    removed = set(old_map.keys()) - set(new_map.keys())
    modified = []
    
    for name in set(old_map.keys()) & set(new_map.keys()):
        if self._symbol_changed(old_map[name], new_map[name]):
            modified.append(name)
    
    # Update databases
    if removed:
        await self._remove_symbols([old_map[n] for n in removed])
    if added:
        await self._add_symbols([new_map[n] for n in added])
    if modified:
        await self._update_symbols([new_map[n] for n in modified])
```

### 5. Persistent State Management
```python
# EDIT existing state management:
def _load_index_state(self) -> Dict:
    """Load persistent indexing state"""
    state_file = Path("/app/state/index_state.json")
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {
        "file_hashes": {},
        "last_full_index": None,
        "version": "2.0"  # Incremental version
    }

async def _save_index_state(self):
    """Persist indexing state"""
    state_file = Path("/app/state/index_state.json")
    state_file.parent.mkdir(exist_ok=True)
    
    state = {
        "file_hashes": self.file_hashes,
        "last_full_index": datetime.now().isoformat(),
        "version": "2.0"
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
```

## Implementation Phases

### Phase 1: Hash-Based Change Detection
- Implement content hashing
- Skip unchanged files
- Add metrics for skip rate

### Phase 2: Symbol-Level Diffing
- Cache extracted symbols
- Implement symbol comparison
- Update only changed symbols

### Phase 3: Dependency Tracking
- Track symbol dependencies
- Cascade updates for dependent symbols
- Optimize Neo4j queries

## Consequences

### Positive
- 70-90% reduction in reindexing time
- Lower CPU/memory usage
- Faster file save response
- Better resource utilization

### Negative
- Increased state complexity
- More Redis memory usage
- Potential cache inconsistencies

### Neutral
- Uses existing Redis infrastructure
- Compatible with current architecture
- Maintains same API surface

## Performance Targets

- File change to index complete: <500ms (from 2-3s)
- Memory overhead: <100MB for 10k files
- Cache hit rate: >80%
- CPU reduction: 60-70%

## Testing Strategy

1. Unit tests for diff algorithms
2. Integration tests with file changes
3. Performance benchmarks
4. Cache consistency tests
5. State recovery tests

## Monitoring

```python
# Add metrics to existing indexer:
self.metrics.gauge('indexer.cache_hit_rate', hit_rate)
self.metrics.counter('indexer.files_skipped', skipped)
self.metrics.histogram('indexer.diff_time', diff_duration)
```

## Rollback Plan

- Feature flag: `INCREMENTAL_INDEXING_ENABLED`
- Falls back to full reindexing
- Clear cache to force reindex
- State version check for compatibility

## Edge Cases

1. **Corrupted cache**: Automatic full reindex
2. **State file missing**: Rebuild from scan
3. **Symbol extraction failure**: Mark file for retry
4. **Memory pressure**: Evict LRU cache entries

## References

- Current implementation: `neural-tools/src/servers/services/indexer_service.py`
- Redis cache: Already configured in docker-compose.yml
- State persistence: `/app/state` volume mounted