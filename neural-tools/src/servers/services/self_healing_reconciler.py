"""
Self-Healing Reconciler Service (ADR-054)

Automatically repairs drift between Neo4j and Qdrant with safety limits.
Implements rate limiting, circuit breakers, and idempotency protection.
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .event_store import SyncEventStore, SyncEventType, create_event_store
from .circuit_breaker import SelfHealingCircuitBreaker
from .drift_monitor import DriftMonitor, DriftReport, DriftSample
from .sync_manager import WriteSynchronizationManager

logger = logging.getLogger(__name__)


class RepairStrategy(Enum):
    """Repair strategies for different drift types"""
    COPY_TO_QDRANT = "copy_to_qdrant"      # Neo4j → Qdrant
    COPY_TO_NEO4J = "copy_to_neo4j"        # Qdrant → Neo4j
    RECONCILE_CONTENT = "reconcile_content" # Fix content mismatch
    DELETE_ORPHAN = "delete_orphan"         # Remove orphaned data
    SKIP = "skip"                            # Cannot repair automatically


@dataclass
class RepairOperation:
    """Represents a single repair operation"""
    chunk_id: str
    drift_type: str
    strategy: RepairStrategy
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    completed: bool = False
    error: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass
class RepairReport:
    """Report of repair operations"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_drift: int = 0
    repairs_attempted: int = 0
    repairs_successful: int = 0
    repairs_failed: int = 0
    repairs_skipped: int = 0
    rate_limited: int = 0
    operations: List[RepairOperation] = field(default_factory=list)
    duration_ms: Optional[int] = None
    circuit_breaker_trips: int = 0


class SelfHealingReconciler:
    """
    Automatically repairs drift between Neo4j and Qdrant.
    Implements safety limits per ADR-054 and Grok 4 recommendations.
    """

    def __init__(
        self,
        neo4j_service,
        qdrant_service,
        sync_manager: WriteSynchronizationManager,
        drift_monitor: DriftMonitor,
        project_name: str,
        max_repair_rate: int = 100,  # Per minute per ADR-054
        max_repairs_per_run: int = 1000,
        repair_batch_size: int = 10
    ):
        """
        Initialize self-healing reconciler.

        Args:
            neo4j_service: Neo4j service instance
            qdrant_service: Qdrant service instance
            sync_manager: Write synchronization manager
            drift_monitor: Drift detection monitor
            project_name: Project to reconcile
            max_repair_rate: Max repairs per minute (safety limit)
            max_repairs_per_run: Max repairs in single run
            repair_batch_size: Repairs per batch
        """
        self.neo4j = neo4j_service
        self.qdrant = qdrant_service
        self.sync_manager = sync_manager
        self.drift_monitor = drift_monitor
        self.project_name = project_name
        self.collection_name = f'project-{project_name}'

        # Safety limits
        self.max_repair_rate = max_repair_rate
        self.max_repairs_per_run = max_repairs_per_run
        self.repair_batch_size = repair_batch_size

        # Rate limiting
        self.repair_count = 0
        self.rate_window_start = datetime.now()

        # Event store for audit trail
        self.event_store = create_event_store("sqlite")
        self._event_store_initialized = False

        # Circuit breaker for self-healing operations
        self.circuit_breaker = SelfHealingCircuitBreaker(
            service_name=f"reconciler_{project_name}",
            failure_threshold=3,  # Per ADR-054
            recovery_timeout=60,   # Per ADR-054
            max_repair_rate=max_repair_rate
        )

        # Track repair history for idempotency
        self.recent_repairs: Set[str] = set()
        self.repair_history: List[RepairReport] = []

        # Exponential backoff for retries
        self.retry_delays = [1, 2, 4, 8, 16, 32, 60]  # seconds

    async def _ensure_event_store_initialized(self):
        """Ensure event store is initialized"""
        if not self._event_store_initialized:
            await self.event_store.initialize()
            self._event_store_initialized = True

    def _generate_idempotency_key(self, chunk_id: str, drift_type: str) -> str:
        """Generate idempotency key for repair operation"""
        # Include timestamp rounded to hour to allow retries after time
        hour_timestamp = datetime.now().replace(minute=0, second=0, microsecond=0)
        key_data = f"{self.project_name}:{chunk_id}:{drift_type}:{hour_timestamp.isoformat()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = datetime.now()

        # Reset window if minute has passed
        if (current_time - self.rate_window_start).total_seconds() >= 60:
            self.repair_count = 0
            self.rate_window_start = current_time

        if self.repair_count >= self.max_repair_rate:
            logger.warning(
                f"Rate limit reached: {self.repair_count}/{self.max_repair_rate} repairs/min"
            )
            return False

        return True

    async def run_reconciliation(
        self,
        drift_report: Optional[DriftReport] = None,
        force: bool = False
    ) -> RepairReport:
        """
        Run self-healing reconciliation.

        Args:
            drift_report: Pre-computed drift report (optional)
            force: Force reconciliation even if drift is minimal

        Returns:
            RepairReport with results
        """
        await self._ensure_event_store_initialized()

        start_time = datetime.now()
        report = RepairReport()
        correlation_id = str(uuid.uuid4())

        # Log repair started
        await self.event_store.log_event(
            event_type=SyncEventType.REPAIR_STARTED,
            project=self.project_name,
            correlation_id=correlation_id,
            metadata={'force': force}
        )

        try:
            # Get drift report if not provided
            if not drift_report:
                logger.info("Checking for drift...")
                drift_report = await self.drift_monitor.check_drift_with_sampling()

            # Check if repair is needed
            if drift_report.drift_rate == 0 and not force:
                logger.info("No drift detected - skipping repair")
                report.total_drift = 0
                return report

            if drift_report.drift_rate < 0:
                logger.error("Drift check failed - cannot proceed with repair")
                report.total_drift = -1
                return report

            # Estimate total drift
            report.total_drift = int(drift_report.total_chunks * drift_report.drift_rate)

            logger.info(
                f"Starting repair: estimated {report.total_drift} drifted chunks "
                f"(rate: {drift_report.drift_rate:.2%})"
            )

            # Process drifted samples
            repair_operations = []
            for sample in drift_report.samples:
                if not sample.is_drifted:
                    continue

                # Determine repair strategy
                strategy = self._determine_repair_strategy(sample)

                if strategy == RepairStrategy.SKIP:
                    report.repairs_skipped += 1
                    continue

                # Create repair operation
                operation = RepairOperation(
                    chunk_id=sample.chunk_id,
                    drift_type=sample.drift_type,
                    strategy=strategy,
                    correlation_id=correlation_id
                )

                # Generate idempotency key
                operation.idempotency_key = self._generate_idempotency_key(
                    sample.chunk_id,
                    sample.drift_type
                )

                # Check if already repaired recently
                if operation.idempotency_key in self.recent_repairs:
                    logger.debug(f"Skipping duplicate repair for chunk {sample.chunk_id}")
                    report.repairs_skipped += 1
                    continue

                repair_operations.append(operation)

                # Limit repairs per run
                if len(repair_operations) >= self.max_repairs_per_run:
                    logger.warning(
                        f"Reached max repairs per run ({self.max_repairs_per_run})"
                    )
                    break

            # Execute repairs in batches
            for i in range(0, len(repair_operations), self.repair_batch_size):
                batch = repair_operations[i:i + self.repair_batch_size]

                # Check rate limit
                if not await self._check_rate_limit():
                    report.rate_limited += len(batch)

                    # Log rate limiting
                    await self.event_store.log_event(
                        event_type=SyncEventType.REPAIR_RATE_LIMITED,
                        project=self.project_name,
                        correlation_id=correlation_id,
                        metadata={'batch_size': len(batch)}
                    )

                    # Wait before retrying
                    await asyncio.sleep(60 - (datetime.now() - self.rate_window_start).total_seconds())

                # Execute batch with circuit breaker
                try:
                    await self.circuit_breaker.call_with_rate_limit(
                        self._execute_repair_batch,
                        batch,
                        report
                    )
                except Exception as e:
                    logger.error(f"Circuit breaker tripped: {e}")
                    report.circuit_breaker_trips += 1

                    # Log circuit breaker event
                    await self.event_store.log_event(
                        event_type=SyncEventType.CIRCUIT_OPENED,
                        project=self.project_name,
                        correlation_id=correlation_id,
                        error=str(e)
                    )

                    # Stop if circuit breaker opens
                    break

            # Calculate duration
            report.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Log repair completed
            if report.repairs_successful > 0:
                await self.event_store.log_event(
                    event_type=SyncEventType.REPAIR_COMPLETED,
                    project=self.project_name,
                    correlation_id=correlation_id,
                    metadata={
                        'successful': report.repairs_successful,
                        'failed': report.repairs_failed,
                        'skipped': report.repairs_skipped
                    },
                    duration_ms=report.duration_ms
                )

            # Store in history
            self.repair_history.append(report)
            if len(self.repair_history) > 100:
                self.repair_history.pop(0)

            logger.info(
                f"Repair complete: {report.repairs_successful} successful, "
                f"{report.repairs_failed} failed, {report.repairs_skipped} skipped"
            )

            return report

        except Exception as e:
            logger.error(f"Repair failed: {e}")

            await self.event_store.log_event(
                event_type=SyncEventType.REPAIR_FAILED,
                project=self.project_name,
                correlation_id=correlation_id,
                error=str(e)
            )

            report.repairs_failed = report.repairs_attempted
            return report

    async def _execute_repair_batch(
        self,
        batch: List[RepairOperation],
        report: RepairReport
    ):
        """Execute a batch of repair operations"""
        for operation in batch:
            report.repairs_attempted += 1
            self.repair_count += 1

            try:
                success = await self._execute_single_repair(operation)

                if success:
                    report.repairs_successful += 1
                    operation.completed = True

                    # Track for idempotency
                    self.recent_repairs.add(operation.idempotency_key)

                    # Clean old entries (keep last 10000)
                    if len(self.recent_repairs) > 10000:
                        self.recent_repairs = set(list(self.recent_repairs)[-5000:])
                else:
                    report.repairs_failed += 1
                    operation.error = "Repair failed"

            except Exception as e:
                logger.error(f"Failed to repair chunk {operation.chunk_id}: {e}")
                report.repairs_failed += 1
                operation.error = str(e)

            report.operations.append(operation)

    async def _execute_single_repair(self, operation: RepairOperation) -> bool:
        """Execute a single repair operation"""
        start_time = datetime.now()

        try:
            if operation.strategy == RepairStrategy.COPY_TO_QDRANT:
                success = await self._copy_to_qdrant(operation.chunk_id)

            elif operation.strategy == RepairStrategy.COPY_TO_NEO4J:
                success = await self._copy_to_neo4j(operation.chunk_id)

            elif operation.strategy == RepairStrategy.RECONCILE_CONTENT:
                success = await self._reconcile_content(operation.chunk_id)

            elif operation.strategy == RepairStrategy.DELETE_ORPHAN:
                success = await self._delete_orphan(operation.chunk_id, operation.drift_type)

            else:
                logger.warning(f"Unknown repair strategy: {operation.strategy}")
                success = False

            operation.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if success:
                logger.debug(f"Repaired chunk {operation.chunk_id} using {operation.strategy.value}")

            return success

        except Exception as e:
            logger.error(f"Repair operation failed for {operation.chunk_id}: {e}")
            operation.error = str(e)
            return False

    def _determine_repair_strategy(self, sample: DriftSample) -> RepairStrategy:
        """Determine the appropriate repair strategy"""
        if sample.drift_type == "missing_in_qdrant":
            return RepairStrategy.COPY_TO_QDRANT

        elif sample.drift_type == "missing_in_neo4j":
            # Be careful about adding to Neo4j - might be intentionally deleted
            return RepairStrategy.DELETE_ORPHAN

        elif sample.drift_type == "content_mismatch":
            # For content mismatches, prefer Neo4j as source of truth
            return RepairStrategy.COPY_TO_QDRANT

        else:
            return RepairStrategy.SKIP

    async def _copy_to_qdrant(self, chunk_id: str) -> bool:
        """Copy chunk from Neo4j to Qdrant"""
        try:
            # Get chunk from Neo4j
            cypher = """
            MATCH (c:Chunk {qdrant_id: $chunk_id, project: $project})
            RETURN c.content as content, c.chunk_hash as hash, c as properties
            """

            result = await self.neo4j.execute_cypher(
                cypher,
                {'chunk_id': chunk_id, 'project': self.project_name}
            )

            if result.get('status') != 'success' or not result.get('data'):
                logger.error(f"Chunk {chunk_id} not found in Neo4j")
                return False

            data = result['data'][0]
            content = data.get('content', '')
            properties = data.get('properties', {})

            # Generate embedding (placeholder - would call embedding service)
            vector = [0.0] * 768  # Placeholder

            # Write to Qdrant
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=int(chunk_id) if chunk_id.isdigit() else hash(chunk_id),
                vector=vector,
                payload={
                    'content': content,
                    'chunk_hash': data.get('hash'),
                    'project': self.project_name,
                    **properties
                }
            )

            await self.qdrant.upsert_points(
                collection_name=self.collection_name,
                points=[point]
            )

            return True

        except Exception as e:
            logger.error(f"Failed to copy chunk {chunk_id} to Qdrant: {e}")
            return False

    async def _copy_to_neo4j(self, chunk_id: str) -> bool:
        """Copy chunk from Qdrant to Neo4j"""
        try:
            # Get point from Qdrant
            points = await self.qdrant.retrieve_points(
                collection_name=self.collection_name,
                ids=[int(chunk_id) if chunk_id.isdigit() else hash(chunk_id)]
            )

            if not points:
                logger.error(f"Chunk {chunk_id} not found in Qdrant")
                return False

            payload = points[0].get('payload', {})
            content = payload.get('content', '')

            # Create in Neo4j
            cypher = """
            CREATE (c:Chunk {
                qdrant_id: $chunk_id,
                chunk_hash: $hash,
                project: $project,
                content: $content,
                indexed_at: datetime()
            })
            RETURN c.chunk_id as id
            """

            result = await self.neo4j.execute_cypher(
                cypher,
                {
                    'chunk_id': chunk_id,
                    'hash': payload.get('chunk_hash', ''),
                    'project': self.project_name,
                    'content': content
                }
            )

            return result.get('status') == 'success'

        except Exception as e:
            logger.error(f"Failed to copy chunk {chunk_id} to Neo4j: {e}")
            return False

    async def _reconcile_content(self, chunk_id: str) -> bool:
        """Reconcile content mismatch - Neo4j is source of truth"""
        # This is essentially copy_to_qdrant for content mismatches
        return await self._copy_to_qdrant(chunk_id)

    async def _delete_orphan(self, chunk_id: str, drift_type: str) -> bool:
        """Delete orphaned data"""
        try:
            if drift_type == "missing_in_neo4j":
                # Delete from Qdrant
                from qdrant_client.models import PointIdsList

                await self.qdrant.delete_points(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(
                        points=[int(chunk_id) if chunk_id.isdigit() else hash(chunk_id)]
                    )
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete orphan {chunk_id}: {e}")
            return False

    async def get_repair_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get repair statistics over time"""
        cutoff = datetime.now() - timedelta(hours=hours)

        recent_reports = [
            r for r in self.repair_history
            if r.timestamp >= cutoff
        ]

        if not recent_reports:
            return {
                'message': 'No repairs in the specified period',
                'total_repairs': 0
            }

        total_successful = sum(r.repairs_successful for r in recent_reports)
        total_failed = sum(r.repairs_failed for r in recent_reports)
        total_rate_limited = sum(r.rate_limited for r in recent_reports)
        total_circuit_trips = sum(r.circuit_breaker_trips for r in recent_reports)

        return {
            'period_hours': hours,
            'repair_runs': len(recent_reports),
            'total_successful': total_successful,
            'total_failed': total_failed,
            'total_rate_limited': total_rate_limited,
            'circuit_breaker_trips': total_circuit_trips,
            'success_rate': total_successful / max(1, total_successful + total_failed),
            'last_repair': recent_reports[-1].timestamp.isoformat() if recent_reports else None
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current reconciler status"""
        return {
            'project': self.project_name,
            'circuit_breaker_state': self.circuit_breaker.get_status(),
            'recent_repairs_tracked': len(self.recent_repairs),
            'repair_history_size': len(self.repair_history),
            'current_repair_rate': f"{self.repair_count}/min",
            'max_repair_rate': f"{self.max_repair_rate}/min",
            'safety_limits': {
                'max_repairs_per_run': self.max_repairs_per_run,
                'repair_batch_size': self.repair_batch_size,
                'circuit_breaker_threshold': 3,
                'recovery_timeout_seconds': 60
            }
        }


async def create_self_healing_system(
    neo4j_service,
    qdrant_service,
    project_name: str,
    enable_auto_repair: bool = False,
    repair_interval_minutes: int = 30
) -> Tuple[DriftMonitor, SelfHealingReconciler]:
    """
    Create a complete self-healing system.

    Args:
        neo4j_service: Neo4j service instance
        qdrant_service: Qdrant service instance
        project_name: Project to monitor and repair
        enable_auto_repair: Enable automatic repair on drift detection
        repair_interval_minutes: Minutes between auto repairs

    Returns:
        Tuple of (DriftMonitor, SelfHealingReconciler)
    """
    # Create sync manager
    sync_manager = WriteSynchronizationManager(
        neo4j_service,
        qdrant_service,
        project_name
    )

    # Create drift monitor
    drift_monitor = DriftMonitor(
        neo4j_service,
        qdrant_service,
        project_name
    )

    # Create reconciler
    reconciler = SelfHealingReconciler(
        neo4j_service,
        qdrant_service,
        sync_manager,
        drift_monitor,
        project_name
    )

    # Schedule automatic repair if enabled
    if enable_auto_repair:
        asyncio.create_task(
            _auto_repair_loop(
                drift_monitor,
                reconciler,
                repair_interval_minutes
            )
        )

    logger.info(
        f"✅ Self-healing system initialized for project '{project_name}' "
        f"(auto_repair={'enabled' if enable_auto_repair else 'disabled'})"
    )

    return drift_monitor, reconciler


async def _auto_repair_loop(
    drift_monitor: DriftMonitor,
    reconciler: SelfHealingReconciler,
    interval_minutes: int
):
    """Background task for automatic repair"""
    while True:
        try:
            # Wait for interval
            await asyncio.sleep(interval_minutes * 60)

            # Check drift
            logger.info("Running scheduled drift check...")
            drift_report = await drift_monitor.check_drift_with_sampling()

            # Repair if needed
            if drift_report.drift_rate > 0.01:  # >1% drift
                logger.info(f"Drift detected ({drift_report.drift_rate:.2%}) - starting repair...")
                repair_report = await reconciler.run_reconciliation(drift_report)

                logger.info(
                    f"Auto-repair complete: {repair_report.repairs_successful} successful, "
                    f"{repair_report.repairs_failed} failed"
                )

        except Exception as e:
            logger.error(f"Auto-repair loop error: {e}")
            # Continue loop despite errors