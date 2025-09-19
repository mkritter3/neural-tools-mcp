# ADR-0055: PostgreSQL Event Store Migration Decision

## Status
**DEFERRED** - Not needed at current scale

## Context
ADR-054 implemented event sourcing with SQLite (WAL mode) and included an abstraction layer for future PostgreSQL migration. We need to decide if/when to create a full ADR for PostgreSQL migration.

## Current State
- **Scale**: Single developer, ~10K files, 100-1000 events/minute
- **Performance**: SQLite handling load without issues
- **Infrastructure**: Already running Neo4j, Qdrant, Redis in containers
- **Abstraction**: Storage backend protocol defined and ready

## Decision
**DO NOT create PostgreSQL migration ADR now**. Wait for reactive signals.

## Migration Triggers (Monitor These)
When ANY of these occur, create full migration ADR:

### Performance Signals
- [ ] Write latency >100ms for event logging (current: <10ms)
- [ ] Idempotency checks >100ms average (current: <5ms)
- [ ] Circuit breaker trips >10/hour due to SQLite locks
- [ ] Event rate sustained >5,000/minute

### Scale Signals
- [ ] Database file size >10GB
- [ ] Files indexed >100,000
- [ ] Multiple concurrent users/sessions
- [ ] Event backlog >1,000 unprocessed

### Operational Signals
- [ ] Need for replication/HA
- [ ] Need for row-level locking
- [ ] Container deployment friction
- [ ] Backup/restore complexity

## Monitoring Setup Required
```python
# Add these metrics to health_monitor.py
metrics_to_track = {
    "event_write_latency_ms": Histogram,
    "event_db_size_bytes": Gauge,
    "sqlite_busy_errors": Counter,
    "events_per_minute": Gauge,
    "idempotency_check_ms": Histogram
}
```

## Pre-Migration Checklist
When triggers are met, before creating ADR:

1. **Load Test Current System**
   ```bash
   # Simulate 10x current load
   python3 scripts/load_test_event_store.py --rate=10000
   ```

2. **Validate PostgreSQL Stub**
   ```python
   # Test abstraction works with both backends
   pytest tests/test_storage_abstraction.py
   ```

3. **Benchmark Both Options**
   - SQLite with optimizations (indexes, vacuum settings)
   - PostgreSQL with connection pooling

## Migration Path (When Needed)
1. Implement PostgreSQLBackend fully
2. Add migration script (SQLite → PostgreSQL)
3. Test with production-like load
4. Deploy PostgreSQL container
5. Switch via environment variable
6. Keep SQLite as fallback for 30 days

## Why Not Now?
Per Grok 4's analysis:
- SQLite can handle 10K+ writes/sec (we're at ~17/sec)
- WAL mode provides sufficient concurrency
- No current multi-user requirements
- Premature optimization = wasted effort

## References
- SQLite limits: https://www.sqlite.org/limits.html
- WAL mode docs: https://www.sqlite.org/wal.html
- Grok 4 consultation: September 19, 2025

## Review Schedule
Check migration triggers monthly or when:
- User reports performance issues
- Planning multi-user features
- Scaling beyond single machine

---

**Confidence: 95%**
Assumptions:
- Current load patterns continue
- Single-developer use case remains
- SQLite WAL mode properly configured