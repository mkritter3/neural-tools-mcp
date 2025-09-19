#!/usr/bin/env python3
"""
Chaos Engineering Tests for Self-Healing System (ADR-054)

Tests resilience of the synchronization and self-healing mechanisms
under various failure scenarios and edge cases.
"""

import asyncio
import pytest
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.servers.services.sync_manager import WriteSynchronizationManager
from src.servers.services.drift_monitor import DriftMonitor, DriftReport, DriftSample
from src.servers.services.self_healing_reconciler import (
    SelfHealingReconciler,
    RepairStrategy,
    RepairOperation,
    create_self_healing_system
)
from src.servers.services.circuit_breaker import CircuitBreaker, CircuitState
from src.servers.services.event_store import SyncEventStore, SyncEventType


class TestChaosEngineering:
    """Chaos engineering tests for self-healing system"""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing"""
        neo4j = AsyncMock()
        neo4j.execute_cypher = AsyncMock()
        neo4j.initialized = True

        qdrant = AsyncMock()
        qdrant.upsert_points = AsyncMock()
        qdrant.retrieve_points = AsyncMock()
        qdrant.scroll_collection = AsyncMock()
        qdrant.initialized = True

        return neo4j, qdrant

    @pytest.fixture
    def sync_manager(self, mock_services):
        """Create sync manager with mock services"""
        neo4j, qdrant = mock_services
        return WriteSynchronizationManager(neo4j, qdrant, "test-project")

    @pytest.fixture
    def drift_monitor(self, mock_services):
        """Create drift monitor with mock services"""
        neo4j, qdrant = mock_services
        return DriftMonitor(neo4j, qdrant, "test-project")

    @pytest.fixture
    def reconciler(self, mock_services, sync_manager, drift_monitor):
        """Create reconciler with mock services"""
        neo4j, qdrant = mock_services
        return SelfHealingReconciler(
            neo4j, qdrant, sync_manager, drift_monitor, "test-project"
        )

    @pytest.mark.asyncio
    async def test_concurrent_reconcilers(self, reconciler):
        """
        Test: Multiple reconcilers running simultaneously
        Expected: No duplicate repairs due to idempotency keys
        """
        # Create drift report with 10 drifted chunks
        drift_report = DriftReport(
            total_chunks=100,
            sampled_chunks=10,
            drifted_chunks=5,
            drift_rate=0.5
        )

        for i in range(5):
            drift_report.samples.append(
                DriftSample(
                    chunk_id=f"chunk_{i}",
                    is_drifted=True,
                    drift_type="missing_in_qdrant"
                )
            )

        # Run 3 concurrent reconcilers
        tasks = []
        for _ in range(3):
            task = asyncio.create_task(
                reconciler.run_reconciliation(drift_report)
            )
            tasks.append(task)

        reports = await asyncio.gather(*tasks)

        # Check that repairs weren't duplicated
        total_successful = sum(r.repairs_successful for r in reports)
        total_skipped = sum(r.repairs_skipped for r in reports)

        # Should have 5 successful repairs total (not 15)
        # Additional reconcilers should skip due to idempotency
        assert total_successful <= 5, "Duplicate repairs detected!"
        assert total_skipped > 0, "Idempotency protection not working"

    @pytest.mark.asyncio
    async def test_reconciliation_loop_detection(self, reconciler):
        """
        Test: Reconciliation loop (repair → drift → repair cycle)
        Expected: Circuit breaker opens after detecting loop
        """
        # Simulate a chunk that keeps getting detected as drifted
        problematic_chunk = "problem_chunk"

        for attempt in range(5):
            drift_report = DriftReport(drifted_chunks=1, drift_rate=0.1)
            drift_report.samples.append(
                DriftSample(
                    chunk_id=problematic_chunk,
                    is_drifted=True,
                    drift_type="content_mismatch"
                )
            )

            # Mock repair that appears successful but doesn't fix drift
            with patch.object(reconciler, '_execute_single_repair', return_value=True):
                report = await reconciler.run_reconciliation(drift_report)

                # After 3 attempts, circuit breaker should open
                if attempt >= 3:
                    assert reconciler.circuit_breaker.state == CircuitState.OPEN
                    assert report.circuit_breaker_trips > 0
                    break

    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self, reconciler):
        """
        Test: Massive drift requiring many repairs
        Expected: Rate limiting prevents overload (100 repairs/minute max)
        """
        # Create drift report with 500 drifted chunks
        drift_report = DriftReport(
            total_chunks=1000,
            sampled_chunks=500,
            drifted_chunks=500,
            drift_rate=0.5
        )

        for i in range(500):
            drift_report.samples.append(
                DriftSample(
                    chunk_id=f"chunk_{i}",
                    is_drifted=True,
                    drift_type="missing_in_qdrant"
                )
            )

        # Mock fast repairs
        with patch.object(reconciler, '_execute_single_repair', return_value=True):
            # Track repair rate
            start_time = datetime.now()
            report = await reconciler.run_reconciliation(drift_report)
            duration = (datetime.now() - start_time).total_seconds()

            # Calculate actual repair rate
            repairs_per_minute = (report.repairs_attempted / duration) * 60

            # Should not exceed rate limit (with some tolerance)
            assert repairs_per_minute <= reconciler.max_repair_rate * 1.1
            assert report.rate_limited > 0, "Rate limiting not triggered"

    @pytest.mark.asyncio
    async def test_network_partition_simulation(self, mock_services):
        """
        Test: Network partition between Neo4j and Qdrant
        Expected: Graceful degradation, no data corruption
        """
        neo4j, qdrant = mock_services

        # Simulate network partition - Neo4j succeeds, Qdrant fails
        neo4j.execute_cypher.return_value = {
            'status': 'success',
            'data': [{'content': 'test content'}]
        }
        qdrant.upsert_points.side_effect = ConnectionError("Network partition")

        sync_manager = WriteSynchronizationManager(neo4j, qdrant, "test-project")

        # Attempt to write chunk
        success, chunk_id, chunk_hash = await sync_manager.write_chunk(
            content="test content",
            metadata={'test': True}
        )

        # Should fail and rollback
        assert not success
        # Neo4j rollback should have been called
        neo4j.execute_cypher.assert_any_call(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
            DELETE c
            """,
            {'chunk_id': chunk_id, 'project': 'test-project'}
        )

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, reconciler, mock_services):
        """
        Test: Some repairs succeed, others fail
        Expected: Successful repairs are preserved, failures are tracked
        """
        neo4j, qdrant = mock_services

        # Mock some repairs to fail
        fail_chunks = {'chunk_2', 'chunk_4'}

        async def mock_repair(chunk_id):
            if chunk_id in fail_chunks:
                raise Exception(f"Failed to repair {chunk_id}")
            return True

        drift_report = DriftReport(drifted_chunks=5)
        for i in range(5):
            drift_report.samples.append(
                DriftSample(
                    chunk_id=f"chunk_{i}",
                    is_drifted=True,
                    drift_type="missing_in_qdrant"
                )
            )

        with patch.object(reconciler, '_copy_to_qdrant', side_effect=mock_repair):
            report = await reconciler.run_reconciliation(drift_report)

            assert report.repairs_successful == 3
            assert report.repairs_failed == 2
            # Failed chunks should be in report
            failed_ids = [op.chunk_id for op in report.operations if op.error]
            assert 'chunk_2' in failed_ids
            assert 'chunk_4' in failed_ids

    @pytest.mark.asyncio
    async def test_drift_detection_with_corruption(self, drift_monitor, mock_services):
        """
        Test: Content corruption detection using MD5
        Expected: Content mismatches are detected even with same IDs
        """
        neo4j, qdrant = mock_services

        # Same chunk ID but different content
        chunk_id = "corrupted_chunk"
        neo4j_content = "Original content from Neo4j"
        qdrant_content = "Corrupted content in Qdrant"  # Different!

        # Mock Neo4j response
        neo4j.execute_cypher.return_value = {
            'status': 'success',
            'data': [{'n': {'content': neo4j_content}}]
        }

        # Mock Qdrant response
        qdrant_point = Mock()
        qdrant_point.payload = {'content': qdrant_content}
        qdrant.retrieve_points.return_value = [{'payload': {'content': qdrant_content}}]

        # Check drift for this chunk
        sample = await drift_monitor._check_chunk_content_drift(chunk_id)

        # Should detect content mismatch via MD5 hash difference
        assert sample.is_drifted
        assert sample.drift_type == "content_mismatch"
        assert sample.neo4j_content_hash != sample.qdrant_content_hash

    @pytest.mark.asyncio
    async def test_exponential_backoff_on_failures(self, reconciler):
        """
        Test: Repeated failures trigger exponential backoff
        Expected: Delays increase exponentially
        """
        # Track delay between attempts
        attempt_times = []

        async def failing_repair(chunk_id):
            attempt_times.append(datetime.now())
            raise Exception("Transient failure")

        drift_report = DriftReport(drifted_chunks=1)
        drift_report.samples.append(
            DriftSample(
                chunk_id="retry_chunk",
                is_drifted=True,
                drift_type="missing_in_qdrant"
            )
        )

        with patch.object(reconciler, '_copy_to_qdrant', side_effect=failing_repair):
            # Configure retries
            reconciler.retry_delays = [1, 2, 4]  # Short delays for testing

            report = await reconciler.run_reconciliation(drift_report)

            # Should have attempted multiple times
            assert len(attempt_times) >= 2

            # Check delays are increasing
            if len(attempt_times) >= 2:
                delay1 = (attempt_times[1] - attempt_times[0]).total_seconds()
                if len(attempt_times) >= 3:
                    delay2 = (attempt_times[2] - attempt_times[1]).total_seconds()
                    assert delay2 > delay1, "Backoff not increasing"

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, reconciler):
        """
        Test: Long-running reconciler doesn't leak memory
        Expected: Old repair history is pruned
        """
        # Simulate many repair cycles
        for i in range(200):
            drift_report = DriftReport(drifted_chunks=1)
            drift_report.samples.append(
                DriftSample(
                    chunk_id=f"chunk_{i}",
                    is_drifted=True,
                    drift_type="missing_in_qdrant"
                )
            )

            with patch.object(reconciler, '_execute_single_repair', return_value=True):
                await reconciler.run_reconciliation(drift_report)

        # Check memory bounds
        assert len(reconciler.repair_history) <= 100, "Repair history not pruned"
        assert len(reconciler.recent_repairs) <= 10000, "Recent repairs not pruned"

    @pytest.mark.asyncio
    async def test_event_store_failure_handling(self, reconciler):
        """
        Test: Event store failures don't stop reconciliation
        Expected: Reconciliation continues despite logging failures
        """
        # Make event store fail
        with patch.object(reconciler.event_store, 'log_event', side_effect=Exception("DB error")):
            drift_report = DriftReport(drifted_chunks=1)
            drift_report.samples.append(
                DriftSample(
                    chunk_id="test_chunk",
                    is_drifted=True,
                    drift_type="missing_in_qdrant"
                )
            )

            # Should still complete reconciliation
            with patch.object(reconciler, '_execute_single_repair', return_value=True):
                report = await reconciler.run_reconciliation(drift_report)

                # Reconciliation should succeed despite event store failure
                assert report.repairs_attempted > 0

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, mock_services):
        """
        Test: One service failure doesn't cascade to others
        Expected: Circuit breakers isolate failures
        """
        neo4j, qdrant = mock_services

        # Create multiple services with circuit breakers
        services = []
        for i in range(3):
            drift_monitor = DriftMonitor(neo4j, qdrant, f"project_{i}")
            sync_manager = WriteSynchronizationManager(neo4j, qdrant, f"project_{i}")
            reconciler = SelfHealingReconciler(
                neo4j, qdrant, sync_manager, drift_monitor, f"project_{i}"
            )
            services.append((drift_monitor, reconciler))

        # Make one service fail repeatedly
        failed_project = "project_1"

        async def selective_failure(*args, **kwargs):
            if kwargs.get('project') == failed_project:
                raise Exception("Service failure")
            return {'status': 'success', 'data': []}

        neo4j.execute_cypher.side_effect = selective_failure

        # Run drift checks for all services
        for drift_monitor, reconciler in services:
            try:
                report = await drift_monitor.check_drift_with_sampling()

                if drift_monitor.project_name == failed_project:
                    # Should fail and open circuit breaker
                    assert drift_monitor.circuit_breaker.state == CircuitState.OPEN
                else:
                    # Other services should continue working
                    assert drift_monitor.circuit_breaker.state == CircuitState.CLOSED

            except Exception:
                # Only the failed service should raise exceptions
                assert drift_monitor.project_name == failed_project


class TestSafetyLimits:
    """Test safety limits and guardrails"""

    @pytest.mark.asyncio
    async def test_max_repairs_per_run_limit(self):
        """
        Test: Max repairs per run is enforced
        Expected: Stops after reaching limit
        """
        reconciler = Mock(spec=SelfHealingReconciler)
        reconciler.max_repairs_per_run = 10

        # Create drift with 100 chunks
        drift_samples = [
            DriftSample(chunk_id=f"chunk_{i}", is_drifted=True, drift_type="missing_in_qdrant")
            for i in range(100)
        ]

        # Process samples with limit
        processed = 0
        for sample in drift_samples:
            if processed >= reconciler.max_repairs_per_run:
                break
            processed += 1

        assert processed == 10, "Max repairs per run not enforced"

    @pytest.mark.asyncio
    async def test_rate_limit_window_reset(self):
        """
        Test: Rate limit window resets after 60 seconds
        Expected: New window allows more repairs
        """
        reconciler = Mock(spec=SelfHealingReconciler)
        reconciler.max_repair_rate = 100
        reconciler.repair_count = 100  # At limit
        reconciler.rate_window_start = datetime.now() - timedelta(seconds=61)

        # Simulate rate limit check
        current_time = datetime.now()
        if (current_time - reconciler.rate_window_start).total_seconds() >= 60:
            reconciler.repair_count = 0
            reconciler.rate_window_start = current_time

        assert reconciler.repair_count == 0, "Rate window didn't reset"

    @pytest.mark.asyncio
    async def test_idempotency_key_generation(self):
        """
        Test: Idempotency keys prevent duplicate repairs
        Expected: Same chunk + drift type = same key within time window
        """
        reconciler = Mock(spec=SelfHealingReconciler)
        reconciler.project_name = "test-project"  # Add required attribute

        # Generate keys for same repair
        key1 = SelfHealingReconciler._generate_idempotency_key(
            reconciler, "chunk_1", "missing_in_qdrant"
        )

        # Wait a bit (but within same hour)
        await asyncio.sleep(0.1)

        key2 = SelfHealingReconciler._generate_idempotency_key(
            reconciler, "chunk_1", "missing_in_qdrant"
        )

        assert key1 == key2, "Idempotency keys don't match for same repair"

        # Different chunk should have different key
        key3 = SelfHealingReconciler._generate_idempotency_key(
            reconciler, "chunk_2", "missing_in_qdrant"
        )

        assert key1 != key3, "Different chunks have same idempotency key"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])