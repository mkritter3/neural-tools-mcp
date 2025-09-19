#!/usr/bin/env python3
"""
Event Store Health Monitor - Track ADR-055 Migration Triggers

Run this periodically to check if PostgreSQL migration is needed.
"""

import sqlite3
import asyncio
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.servers.services.event_store import create_event_store, SyncEventType

class EventStoreMonitor:
    """Monitor event store health and migration triggers"""

    def __init__(self):
        self.db_path = Path.home() / '.graphrag' / 'events.db'
        self.triggers_met = []

    async def check_performance_metrics(self):
        """Check performance-related triggers"""
        store = create_event_store('sqlite')
        await store.initialize()

        # Test write latency
        start = time.time()
        await store.log_event(
            event_type=SyncEventType.WRITE_STARTED,
            project='monitor-test',
            correlation_id='perf-test'
        )
        write_latency_ms = (time.time() - start) * 1000

        # Test idempotency check latency
        start = time.time()
        await store.backend.check_idempotency('test-key')
        idempotency_latency_ms = (time.time() - start) * 1000

        await store.close()

        # Check triggers
        if write_latency_ms > 100:
            self.triggers_met.append(f"❌ Write latency: {write_latency_ms:.1f}ms > 100ms")
        else:
            print(f"✅ Write latency: {write_latency_ms:.1f}ms < 100ms")

        if idempotency_latency_ms > 100:
            self.triggers_met.append(f"❌ Idempotency check: {idempotency_latency_ms:.1f}ms > 100ms")
        else:
            print(f"✅ Idempotency check: {idempotency_latency_ms:.1f}ms < 100ms")

    def check_scale_metrics(self):
        """Check scale-related triggers"""
        if not self.db_path.exists():
            print("⚠️  Event database not found")
            return

        # Check database size
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
        db_size_gb = db_size_mb / 1024

        if db_size_gb > 10:
            self.triggers_met.append(f"❌ Database size: {db_size_gb:.1f}GB > 10GB")
        else:
            print(f"✅ Database size: {db_size_mb:.1f}MB < 10GB")

        # Check event count and rate
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Total events
        cursor.execute("SELECT COUNT(*) FROM sync_events")
        total_events = cursor.fetchone()[0]

        # Events in last minute
        one_minute_ago = (datetime.utcnow() - timedelta(minutes=1)).isoformat()
        cursor.execute(
            "SELECT COUNT(*) FROM sync_events WHERE timestamp > ?",
            (one_minute_ago,)
        )
        events_last_minute = cursor.fetchone()[0]

        conn.close()

        print(f"✅ Total events: {total_events:,}")

        if events_last_minute > 5000:
            self.triggers_met.append(f"❌ Event rate: {events_last_minute}/min > 5000/min")
        else:
            print(f"✅ Event rate: {events_last_minute}/min < 5000/min")

    def check_sqlite_health(self):
        """Check SQLite-specific health metrics"""
        if not self.db_path.exists():
            return

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Check WAL mode
        cursor.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]

        if journal_mode != 'wal':
            self.triggers_met.append(f"⚠️  Not using WAL mode: {journal_mode}")
        else:
            print(f"✅ WAL mode enabled")

        # Check page count and fragmentation
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]

        cursor.execute("PRAGMA freelist_count")
        freelist_count = cursor.fetchone()[0]

        fragmentation = (freelist_count / max(page_count, 1)) * 100

        if fragmentation > 30:
            print(f"⚠️  Database fragmentation: {fragmentation:.1f}% (consider VACUUM)")
        else:
            print(f"✅ Database fragmentation: {fragmentation:.1f}%")

        conn.close()

    async def run_checks(self):
        """Run all monitoring checks"""
        print("=" * 50)
        print("Event Store Health Monitor")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 50)
        print()

        print("Performance Metrics:")
        await self.check_performance_metrics()
        print()

        print("Scale Metrics:")
        self.check_scale_metrics()
        print()

        print("SQLite Health:")
        self.check_sqlite_health()
        print()

        print("=" * 50)
        if self.triggers_met:
            print("⚠️  MIGRATION TRIGGERS MET:")
            for trigger in self.triggers_met:
                print(f"  {trigger}")
            print()
            print("Consider creating full ADR-055 for PostgreSQL migration")
        else:
            print("✅ No migration triggers met - SQLite is sufficient")
        print("=" * 50)

        return len(self.triggers_met) == 0

async def main():
    monitor = EventStoreMonitor()
    healthy = await monitor.run_checks()

    # Exit code for CI/CD integration
    sys.exit(0 if healthy else 1)

if __name__ == "__main__":
    asyncio.run(main())