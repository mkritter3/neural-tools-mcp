"""
Event Store for GraphRAG Synchronization (ADR-054)

Provides lightweight event sourcing with abstracted storage backend.
Supports SQLite (immediate) and PostgreSQL (future migration).
Includes production safety features from Grok 4 review.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
import json
import logging
from typing import Optional, List, Dict, Any, Protocol
from pathlib import Path
import hashlib
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyncEventType(Enum):
    """Types of synchronization events"""

    # Write operations
    WRITE_STARTED = "write_started"
    NEO4J_WRITTEN = "neo4j_written"
    QDRANT_WRITTEN = "qdrant_written"
    WRITE_COMPLETED = "write_completed"
    WRITE_FAILED = "write_failed"

    # Rollback operations
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_COMPLETED = "rollback_completed"
    ROLLBACK_FAILED = "rollback_failed"

    # Drift detection
    DRIFT_DETECTED = "drift_detected"
    DRIFT_SCAN_STARTED = "drift_scan_started"
    DRIFT_SCAN_COMPLETED = "drift_scan_completed"

    # Repair operations
    REPAIR_STARTED = "repair_started"
    REPAIR_COMPLETED = "repair_completed"
    REPAIR_FAILED = "repair_failed"
    REPAIR_RATE_LIMITED = "repair_rate_limited"

    # Circuit breaker events
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    CIRCUIT_HALF_OPEN = "circuit_half_open"


@dataclass
class SyncEvent:
    """Structured event data"""

    event_type: SyncEventType
    project: str
    correlation_id: str
    timestamp: datetime = None
    neo4j_id: Optional[str] = None
    qdrant_id: Optional[str] = None
    file_path: Optional[str] = None
    chunk_index: Optional[int] = None
    status: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None
    idempotency_key: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

        # Generate idempotency key for repair operations
        if self.event_type in [SyncEventType.REPAIR_STARTED, SyncEventType.WRITE_STARTED]:
            if self.idempotency_key is None:
                # Create deterministic key from event data
                key_data = f"{self.project}:{self.neo4j_id or ''}:{self.qdrant_id or ''}"
                self.idempotency_key = hashlib.md5(key_data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.metadata:
            data['metadata'] = json.dumps(self.metadata)
        return data


class EventStorageBackend(ABC):
    """Abstract interface for swappable storage backends (Grok 4 recommendation)"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend"""
        pass

    @abstractmethod
    async def store_event(self, event: SyncEvent) -> None:
        """Store a single event"""
        pass

    @abstractmethod
    async def get_events_by_correlation_id(self, correlation_id: str) -> List[SyncEvent]:
        """Get all events for a correlation ID (time-travel debugging)"""
        pass

    @abstractmethod
    async def get_failed_operations(self, project: str, since: datetime) -> List[SyncEvent]:
        """Get all failed operations for analysis"""
        pass

    @abstractmethod
    async def get_events_by_type(self, event_type: SyncEventType,
                                 project: str, limit: int = 100) -> List[SyncEvent]:
        """Get events of a specific type"""
        pass

    @abstractmethod
    async def check_idempotency(self, idempotency_key: str) -> bool:
        """Check if an operation with this key has already been performed"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources"""
        pass


class SQLiteBackend(EventStorageBackend):
    """SQLite implementation for immediate use with production optimizations"""

    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path("~/.graphrag/events.db")
        self.db_path = db_path.expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._lock = asyncio.Lock()  # For thread safety

    async def initialize(self):
        """Create schema with WAL mode for better concurrency (Grok 4 suggestion)"""
        async with self._lock:
            self.conn = sqlite3.connect(
                str(self.db_path),
                isolation_level=None,  # Autocommit mode
                check_same_thread=False
            )

            # Enable WAL mode for better concurrent access
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")

            # Create events table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    project TEXT NOT NULL,
                    correlation_id TEXT NOT NULL,
                    neo4j_id TEXT,
                    qdrant_id TEXT,
                    file_path TEXT,
                    chunk_index INTEGER,
                    status TEXT,
                    error TEXT,
                    metadata TEXT,
                    duration_ms INTEGER,
                    idempotency_key TEXT UNIQUE
                )
            """)

            # Create indexes for performance
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_correlation ON sync_events(correlation_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON sync_events(timestamp)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_project ON sync_events(project)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON sync_events(event_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_idempotency ON sync_events(idempotency_key)")

            logger.info(f"✅ SQLite event store initialized at {self.db_path}")

    async def store_event(self, event: SyncEvent) -> None:
        """Store event with idempotency checking"""
        async with self._lock:
            try:
                data = event.to_dict()

                # Handle idempotency for critical operations
                if event.idempotency_key:
                    # Check if already exists
                    cursor = self.conn.execute(
                        "SELECT id FROM sync_events WHERE idempotency_key = ?",
                        (event.idempotency_key,)
                    )
                    if cursor.fetchone():
                        logger.debug(f"Skipping duplicate event with key {event.idempotency_key}")
                        return

                # Insert event
                self.conn.execute("""
                    INSERT INTO sync_events (
                        timestamp, event_type, project, correlation_id,
                        neo4j_id, qdrant_id, file_path, chunk_index,
                        status, error, metadata, duration_ms, idempotency_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['timestamp'], data['event_type'], data['project'],
                    data['correlation_id'], data.get('neo4j_id'), data.get('qdrant_id'),
                    data.get('file_path'), data.get('chunk_index'),
                    data.get('status'), data.get('error'),
                    data.get('metadata'), data.get('duration_ms'),
                    data.get('idempotency_key')
                ))

                logger.debug(f"Stored event: {event.event_type.value} for project {event.project}")

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    logger.debug(f"Event already exists (idempotency): {event.idempotency_key}")
                else:
                    logger.error(f"Failed to store event: {e}")
                    raise
            except Exception as e:
                logger.error(f"Failed to store event: {e}")
                # Don't crash the system on logging failures (Grok 4 recommendation)

    async def get_events_by_correlation_id(self, correlation_id: str) -> List[SyncEvent]:
        """Get all events for debugging a specific operation"""
        async with self._lock:
            cursor = self.conn.execute("""
                SELECT * FROM sync_events
                WHERE correlation_id = ?
                ORDER BY timestamp ASC
            """, (correlation_id,))

            events = []
            for row in cursor.fetchall():
                events.append(self._row_to_event(row))

            return events

    async def get_failed_operations(self, project: str, since: datetime) -> List[SyncEvent]:
        """Get failed operations for analysis"""
        async with self._lock:
            cursor = self.conn.execute("""
                SELECT * FROM sync_events
                WHERE project = ?
                AND timestamp >= ?
                AND event_type IN (?, ?, ?)
                ORDER BY timestamp DESC
                LIMIT 1000
            """, (
                project,
                since.isoformat(),
                SyncEventType.WRITE_FAILED.value,
                SyncEventType.ROLLBACK_FAILED.value,
                SyncEventType.REPAIR_FAILED.value
            ))

            events = []
            for row in cursor.fetchall():
                events.append(self._row_to_event(row))

            return events

    async def get_events_by_type(self, event_type: SyncEventType,
                                 project: str, limit: int = 100) -> List[SyncEvent]:
        """Get events of a specific type"""
        async with self._lock:
            cursor = self.conn.execute("""
                SELECT * FROM sync_events
                WHERE event_type = ?
                AND project = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (event_type.value, project, limit))

            events = []
            for row in cursor.fetchall():
                events.append(self._row_to_event(row))

            return events

    async def check_idempotency(self, idempotency_key: str) -> bool:
        """Check if operation was already performed"""
        async with self._lock:
            cursor = self.conn.execute(
                "SELECT id FROM sync_events WHERE idempotency_key = ?",
                (idempotency_key,)
            )
            return cursor.fetchone() is not None

    async def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("SQLite event store closed")

    def _row_to_event(self, row) -> SyncEvent:
        """Convert database row to SyncEvent"""
        # SQLite returns rows as tuples, convert to dict
        columns = ['id', 'timestamp', 'event_type', 'project', 'correlation_id',
                  'neo4j_id', 'qdrant_id', 'file_path', 'chunk_index',
                  'status', 'error', 'metadata', 'duration_ms', 'idempotency_key']

        data = dict(zip(columns, row))

        # Parse special fields
        if data['metadata']:
            data['metadata'] = json.loads(data['metadata'])

        data['event_type'] = SyncEventType(data['event_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        # Remove id field (not part of SyncEvent)
        del data['id']

        return SyncEvent(**data)


class PostgreSQLBackend(EventStorageBackend):
    """PostgreSQL implementation stub for future migration"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
        logger.info("PostgreSQL backend initialized (not yet implemented)")

    async def initialize(self):
        """Initialize PostgreSQL connection pool"""
        # Implementation deferred to migration phase
        # Will use asyncpg for async PostgreSQL access
        logger.warning("PostgreSQL backend not yet implemented - use SQLite for now")
        raise NotImplementedError("PostgreSQL backend coming in Phase 2")

    # Other methods would follow similar pattern with asyncpg
    async def store_event(self, event: SyncEvent) -> None:
        raise NotImplementedError()

    async def get_events_by_correlation_id(self, correlation_id: str) -> List[SyncEvent]:
        raise NotImplementedError()

    async def get_failed_operations(self, project: str, since: datetime) -> List[SyncEvent]:
        raise NotImplementedError()

    async def get_events_by_type(self, event_type: SyncEventType,
                                 project: str, limit: int = 100) -> List[SyncEvent]:
        raise NotImplementedError()

    async def check_idempotency(self, idempotency_key: str) -> bool:
        raise NotImplementedError()

    async def close(self):
        if self.pool:
            await self.pool.close()


class SyncEventStore:
    """
    Main event store interface with backend abstraction.
    Provides audit trail and time-travel debugging capabilities.
    """

    def __init__(self, backend: Optional[EventStorageBackend] = None):
        """Initialize with specified backend or default to SQLite"""
        self.backend = backend or SQLiteBackend()
        self._initialized = False

    async def initialize(self):
        """Initialize the storage backend"""
        if not self._initialized:
            await self.backend.initialize()
            self._initialized = True
            logger.info("✅ Event store initialized")

    async def log_event(
        self,
        event_type: SyncEventType,
        project: str,
        correlation_id: str,
        **kwargs
    ) -> None:
        """
        Log a synchronization event.

        Args:
            event_type: Type of event
            project: Project name
            correlation_id: Correlation ID for tracking related events
            **kwargs: Additional event fields
        """
        if not self._initialized:
            await self.initialize()

        event = SyncEvent(
            event_type=event_type,
            project=project,
            correlation_id=correlation_id,
            **kwargs
        )

        try:
            await self.backend.store_event(event)
        except Exception as e:
            # Log but don't crash on storage failures (Grok 4 recommendation)
            logger.error(f"Failed to log event {event_type}: {e}")

    async def get_operation_history(self, correlation_id: str) -> List[SyncEvent]:
        """Get complete history of an operation for debugging"""
        if not self._initialized:
            await self.initialize()

        return await self.backend.get_events_by_correlation_id(correlation_id)

    async def get_failed_operations(
        self,
        project: str,
        hours_back: int = 24
    ) -> List[SyncEvent]:
        """Get recent failed operations for analysis"""
        if not self._initialized:
            await self.initialize()

        since = datetime.utcnow() - timedelta(hours=hours_back)
        return await self.backend.get_failed_operations(project, since)

    async def check_repair_idempotency(self, project: str, chunk_id: str) -> bool:
        """Check if a repair has already been performed"""
        if not self._initialized:
            await self.initialize()

        key_data = f"repair:{project}:{chunk_id}"
        idempotency_key = hashlib.md5(key_data.encode()).hexdigest()

        return await self.backend.check_idempotency(idempotency_key)

    async def get_drift_history(self, project: str, limit: int = 10) -> List[SyncEvent]:
        """Get recent drift detection events"""
        if not self._initialized:
            await self.initialize()

        return await self.backend.get_events_by_type(
            SyncEventType.DRIFT_DETECTED,
            project,
            limit
        )

    async def get_repair_history(self, project: str, limit: int = 20) -> List[SyncEvent]:
        """Get recent repair operations"""
        if not self._initialized:
            await self.initialize()

        completed = await self.backend.get_events_by_type(
            SyncEventType.REPAIR_COMPLETED,
            project,
            limit // 2
        )

        failed = await self.backend.get_events_by_type(
            SyncEventType.REPAIR_FAILED,
            project,
            limit // 2
        )

        # Combine and sort by timestamp
        all_repairs = completed + failed
        all_repairs.sort(key=lambda x: x.timestamp, reverse=True)

        return all_repairs[:limit]

    async def close(self):
        """Clean up resources"""
        if self.backend:
            await self.backend.close()

    @asynccontextmanager
    async def transaction(self):
        """Context manager for transactional operations"""
        # For now, just ensure initialization
        if not self._initialized:
            await self.initialize()

        try:
            yield self
        finally:
            # Future: could add transaction support here
            pass

    def is_operational(self) -> bool:
        """Check if event store is operational"""
        return self._initialized and self.backend is not None


# Convenience function for creating event store with config
def create_event_store(storage_type: str = "sqlite", **kwargs) -> SyncEventStore:
    """
    Factory function to create event store with specified backend.

    Args:
        storage_type: "sqlite" or "postgresql"
        **kwargs: Backend-specific configuration

    Returns:
        Configured SyncEventStore instance
    """
    if storage_type == "sqlite":
        backend = SQLiteBackend(kwargs.get('db_path'))
    elif storage_type == "postgresql":
        backend = PostgreSQLBackend(kwargs.get('connection_string'))
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")

    return SyncEventStore(backend)