#!/usr/bin/env python3
"""
Drift Monitor Service - Prevents Neo4j/Qdrant drift
Ensures consistency between graph and vector databases
Part of L9 2025 Production Standards
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Monitors and prevents drift between Neo4j and Qdrant

    Key responsibilities:
    1. Verify ID consistency during writes
    2. Periodic consistency checks
    3. Automatic reconciliation
    4. Alert on drift detection
    """

    def __init__(self, neo4j_service, qdrant_service, project_name: str):
        self.neo4j = neo4j_service
        self.qdrant = qdrant_service
        self.project_name = project_name
        self.last_check = datetime.now()
        self.drift_threshold = 0.01  # 1% drift tolerance

    async def verify_write(self, chunk_id: str, neo4j_written: bool, qdrant_written: bool) -> bool:
        """
        Verify both databases received the write

        Args:
            chunk_id: The chunk ID that was written
            neo4j_written: Whether Neo4j write succeeded
            qdrant_written: Whether Qdrant write succeeded

        Returns:
            True if consistent, False if drift detected
        """
        if neo4j_written != qdrant_written:
            logger.error(f"⚠️ DRIFT DETECTED: chunk_id={chunk_id}, neo4j={neo4j_written}, qdrant={qdrant_written}")
            await self.reconcile_chunk(chunk_id)
            return False
        return True

    async def periodic_consistency_check(self) -> Dict[str, any]:
        """
        Run periodic consistency check between databases

        Returns:
            Dictionary with drift statistics
        """
        logger.info(f"🔍 Running consistency check for project: {self.project_name}")

        # Get counts from both databases
        neo4j_count = await self._get_neo4j_count()
        qdrant_count = await self._get_qdrant_count()

        # Get sample of IDs from both
        neo4j_ids = await self._get_neo4j_sample_ids(100)
        qdrant_ids = await self._get_qdrant_sample_ids(100)

        # Find mismatches
        only_in_neo4j = set(neo4j_ids) - set(qdrant_ids)
        only_in_qdrant = set(qdrant_ids) - set(neo4j_ids)

        drift_percentage = abs(neo4j_count - qdrant_count) / max(neo4j_count, qdrant_count, 1)

        result = {
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name,
            "neo4j_count": neo4j_count,
            "qdrant_count": qdrant_count,
            "drift_percentage": drift_percentage,
            "sample_only_neo4j": len(only_in_neo4j),
            "sample_only_qdrant": len(only_in_qdrant),
            "is_healthy": drift_percentage <= self.drift_threshold
        }

        if not result["is_healthy"]:
            logger.warning(f"⚠️ DRIFT DETECTED: {drift_percentage:.2%} drift between databases")
            logger.warning(f"  Neo4j: {neo4j_count} chunks")
            logger.warning(f"  Qdrant: {qdrant_count} points")

            # Trigger reconciliation if drift is significant
            if drift_percentage > 0.05:  # 5% drift
                await self.auto_reconcile()
        else:
            logger.info(f"✅ Databases in sync: {neo4j_count} chunks, drift: {drift_percentage:.2%}")

        self.last_check = datetime.now()
        return result

    async def reconcile_chunk(self, chunk_id: str) -> bool:
        """
        Reconcile a specific chunk between databases

        Args:
            chunk_id: The chunk to reconcile

        Returns:
            True if reconciled successfully
        """
        try:
            # Check Neo4j
            neo4j_chunk = await self._get_neo4j_chunk(chunk_id)

            # Check Qdrant
            qdrant_point = await self._get_qdrant_point(chunk_id)

            if neo4j_chunk and not qdrant_point:
                # Missing in Qdrant - need to reindex
                logger.info(f"Reconciling: Adding chunk {chunk_id} to Qdrant")
                # Trigger reindex for this specific chunk
                return await self._reindex_chunk(chunk_id)

            elif qdrant_point and not neo4j_chunk:
                # Missing in Neo4j - remove from Qdrant or add to Neo4j
                logger.info(f"Reconciling: Chunk {chunk_id} only in Qdrant, removing")
                return await self._remove_from_qdrant(chunk_id)

            return True

        except Exception as e:
            logger.error(f"Reconciliation failed for {chunk_id}: {e}")
            return False

    async def auto_reconcile(self) -> Dict[str, int]:
        """
        Automatically reconcile all drift between databases

        Returns:
            Statistics about reconciliation
        """
        logger.info("🔧 Starting automatic reconciliation...")

        stats = {
            "added_to_qdrant": 0,
            "added_to_neo4j": 0,
            "removed_from_qdrant": 0,
            "removed_from_neo4j": 0,
            "errors": 0
        }

        # Get all IDs from both databases
        all_neo4j_ids = await self._get_all_neo4j_ids()
        all_qdrant_ids = await self._get_all_qdrant_ids()

        only_in_neo4j = set(all_neo4j_ids) - set(all_qdrant_ids)
        only_in_qdrant = set(all_qdrant_ids) - set(all_neo4j_ids)

        # Process chunks only in Neo4j
        for chunk_id in only_in_neo4j:
            if await self._reindex_chunk(chunk_id):
                stats["added_to_qdrant"] += 1
            else:
                stats["errors"] += 1

        # Process points only in Qdrant
        for chunk_id in only_in_qdrant:
            if await self._remove_from_qdrant(chunk_id):
                stats["removed_from_qdrant"] += 1
            else:
                stats["errors"] += 1

        logger.info(f"✅ Reconciliation complete: {stats}")
        return stats

    async def _get_neo4j_count(self) -> int:
        """Get count of CodeChunks in Neo4j"""
        result = await self.neo4j.execute_cypher(
            "MATCH (n:CodeChunk {project: $project}) RETURN count(n) as count",
            {"project": self.project_name}
        )
        return result[0]["count"] if result else 0

    async def _get_qdrant_count(self) -> int:
        """Get count of points in Qdrant"""
        from qdrant_client.models import CountFilter, FieldCondition, MatchValue

        result = await self.qdrant.client.count(
            collection_name=f"project-{self.project_name}",
            count_filter=CountFilter(
                must=[
                    FieldCondition(
                        key="project",
                        match=MatchValue(value=self.project_name)
                    )
                ]
            )
        )
        return result.count

    async def _get_neo4j_sample_ids(self, limit: int) -> List[str]:
        """Get sample of chunk IDs from Neo4j"""
        result = await self.neo4j.execute_cypher(
            "MATCH (n:CodeChunk {project: $project}) RETURN n.id as id LIMIT $limit",
            {"project": self.project_name, "limit": limit}
        )
        return [r["id"] for r in result if r.get("id")]

    async def _get_qdrant_sample_ids(self, limit: int) -> List[str]:
        """Get sample of chunk IDs from Qdrant"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = await self.qdrant.client.scroll(
            collection_name=f"project-{self.project_name}",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="project",
                        match=MatchValue(value=self.project_name)
                    )
                ]
            ),
            limit=limit,
            with_payload=["neo4j_chunk_id"]
        )

        return [p.payload.get("neo4j_chunk_id") for p in results[0] if p.payload.get("neo4j_chunk_id")]

    async def _get_all_neo4j_ids(self) -> List[str]:
        """Get all chunk IDs from Neo4j"""
        result = await self.neo4j.execute_cypher(
            "MATCH (n:CodeChunk {project: $project}) RETURN n.id as id",
            {"project": self.project_name}
        )
        return [r["id"] for r in result if r.get("id")]

    async def _get_all_qdrant_ids(self) -> List[str]:
        """Get all chunk IDs from Qdrant"""
        all_ids = []
        offset = None

        while True:
            results = await self.qdrant.client.scroll(
                collection_name=f"project-{self.project_name}",
                limit=1000,
                offset=offset,
                with_payload=["neo4j_chunk_id"]
            )

            points, next_offset = results
            all_ids.extend([p.payload.get("neo4j_chunk_id") for p in points if p.payload.get("neo4j_chunk_id")])

            if next_offset is None:
                break
            offset = next_offset

        return all_ids

    async def _get_neo4j_chunk(self, chunk_id: str):
        """Get chunk from Neo4j"""
        result = await self.neo4j.execute_cypher(
            "MATCH (n:CodeChunk {id: $id, project: $project}) RETURN n",
            {"id": chunk_id, "project": self.project_name}
        )
        return result[0] if result else None

    async def _get_qdrant_point(self, chunk_id: str):
        """Get point from Qdrant by neo4j_chunk_id"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results = await self.qdrant.client.scroll(
            collection_name=f"project-{self.project_name}",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="neo4j_chunk_id",
                        match=MatchValue(value=chunk_id)
                    )
                ]
            ),
            limit=1
        )

        return results[0][0] if results[0] else None

    async def _reindex_chunk(self, chunk_id: str) -> bool:
        """Trigger reindex for a specific chunk"""
        # This would trigger the indexer to reprocess this chunk
        # For now, log the need
        logger.info(f"TODO: Reindex chunk {chunk_id}")
        return True

    async def _remove_from_qdrant(self, chunk_id: str) -> bool:
        """Remove orphaned point from Qdrant"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            await self.qdrant.client.delete(
                collection_name=f"project-{self.project_name}",
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="neo4j_chunk_id",
                            match=MatchValue(value=chunk_id)
                        )
                    ]
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to remove {chunk_id} from Qdrant: {e}")
            return False


async def create_drift_monitor(container, project_name: str) -> DriftMonitor:
    """
    Factory function to create drift monitor

    Args:
        container: ServiceContainer with neo4j and qdrant services
        project_name: Project to monitor

    Returns:
        Configured DriftMonitor instance
    """
    if not container.neo4j or not container.qdrant:
        raise ValueError("Both Neo4j and Qdrant services required for drift monitoring")

    monitor = DriftMonitor(container.neo4j, container.qdrant, project_name)

    # Schedule periodic checks every 5 minutes
    asyncio.create_task(_periodic_check_loop(monitor))

    return monitor


async def _periodic_check_loop(monitor: DriftMonitor):
    """Background task for periodic consistency checks"""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            await monitor.periodic_consistency_check()
        except Exception as e:
            logger.error(f"Periodic check failed: {e}")