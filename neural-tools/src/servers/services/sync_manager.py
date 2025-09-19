#!/usr/bin/env python3
"""
WriteSynchronizationManager - Ensures atomic writes to Neo4j and Qdrant
Implements ADR-053 for production-grade synchronization
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, UpdateStatus

logger = logging.getLogger(__name__)

class SyncStatus(Enum):
    """Synchronization status for operations"""
    SYNCED = "synced"
    NEO4J_ONLY = "neo4j_only"
    QDRANT_ONLY = "qdrant_only"
    FAILED = "failed"
    PENDING = "pending"
    RECOVERED = "recovered"

@dataclass
class SyncOperation:
    """Represents a synchronization operation"""
    chunk_id: int
    chunk_hash: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    neo4j_status: SyncStatus = SyncStatus.PENDING
    qdrant_status: SyncStatus = SyncStatus.PENDING
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0

class WriteSynchronizationManager:
    """
    Manages synchronized writes to Neo4j and Qdrant per ADR-053
    Ensures data consistency across both databases
    """

    def __init__(self, neo4j_service, qdrant_service, project_name: str):
        self.neo4j = neo4j_service
        self.qdrant = qdrant_service
        self.project_name = project_name
        self.collection_name = f'project-{project_name}'

        # Track pending operations for recovery
        self.pending_operations: Dict[int, SyncOperation] = {}

        # Metrics
        self.metrics = {
            'total_writes': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'recovered_syncs': 0,
            'neo4j_only': 0,
            'qdrant_only': 0
        }

    @staticmethod
    def generate_chunk_id(content: str) -> Tuple[int, str]:
        """Generate Qdrant-compatible integer ID and hex hash
        Returns: (integer_id, hex_hash)
        """
        if isinstance(content, str):
            content = content.encode()
        hex_hash = hashlib.sha256(content).hexdigest()
        # Convert first 15 hex chars to int (fits in Neo4j's int64)
        # Max value: int('f' * 15, 16) = 1152921504606846975 < 2^63-1
        integer_id = int(hex_hash[:15], 16)
        return integer_id, hex_hash

    async def write_chunk(self, content: str, metadata: Optional[Dict] = None,
                         vector: Optional[List[float]] = None,
                         collection_name: Optional[str] = None,
                         payload: Optional[Dict] = None) -> Tuple[bool, int, str]:
        """
        Write chunk atomically to both databases
        Returns: (success, chunk_id, chunk_hash)
        """
        chunk_id, chunk_hash = self.generate_chunk_id(content)

        # Create sync operation record
        operation = SyncOperation(
            chunk_id=chunk_id,
            chunk_hash=chunk_hash,
            content=content,
            metadata=metadata or {}
        )

        self.pending_operations[chunk_id] = operation
        self.metrics['total_writes'] += 1

        try:
            # Phase 1: Write to Neo4j with transaction
            neo4j_success = await self._write_to_neo4j(operation)

            if not neo4j_success:
                operation.neo4j_status = SyncStatus.FAILED
                await self._handle_failure(operation, "Neo4j write failed")
                return False, chunk_id, chunk_hash

            operation.neo4j_status = SyncStatus.SYNCED

            # Phase 2: Write to Qdrant
            qdrant_success = await self._write_to_qdrant(
                operation, vector, collection_name, payload
            )

            if not qdrant_success:
                operation.qdrant_status = SyncStatus.FAILED
                # Rollback Neo4j if Qdrant fails
                await self._rollback_neo4j(chunk_id)
                await self._handle_failure(operation, "Qdrant write failed, rolled back Neo4j")
                return False, chunk_id, chunk_hash

            operation.qdrant_status = SyncStatus.SYNCED

            # Both succeeded
            self.metrics['successful_syncs'] += 1
            del self.pending_operations[chunk_id]

            logger.info(f"✅ Successfully synced chunk {chunk_id} to both databases")
            return True, chunk_id, chunk_hash

        except Exception as e:
            logger.error(f"❌ Sync failed for chunk {chunk_id}: {e}")
            operation.error = str(e)
            await self._handle_failure(operation, str(e))
            return False, chunk_id, chunk_hash

    async def _write_to_neo4j(self, operation: SyncOperation) -> bool:
        """Write chunk to Neo4j"""
        try:
            # Merge metadata into properties for CREATE
            all_props = {
                'chunk_id': operation.chunk_hash,  # Use hash as Neo4j ID
                'chunk_hash': operation.chunk_hash,
                'qdrant_id': operation.chunk_id,  # Integer ID for Qdrant
                'project': self.project_name,
                'content': operation.content,
                'indexed_at': datetime.now().isoformat(),
                **operation.metadata
            }

            cypher = """
            CREATE (c:Chunk $props)
            RETURN c.chunk_id as chunk_id
            """

            params = {'props': all_props}

            result = await self.neo4j.execute_cypher(cypher, params)

            if result.get('status') == 'success':
                return True

            logger.error(f"Neo4j write failed: {result.get('message')}")
            return False

        except Exception as e:
            logger.error(f"Neo4j write exception: {e}")
            return False

    async def _write_to_qdrant(self, operation: SyncOperation,
                               vector: Optional[List[float]] = None,
                               collection_name: Optional[str] = None,
                               payload: Optional[Dict] = None) -> bool:
        """Write chunk to Qdrant"""
        try:
            if vector is None:
                # Generate a placeholder vector if not provided
                vector = [0.0] * 768

            # Use provided payload or construct default
            if payload:
                final_payload = payload
            else:
                final_payload = {
                    'content': operation.content,
                    'chunk_hash': operation.chunk_hash,
                    'project': self.project_name,
                    **operation.metadata
                }

            point = PointStruct(
                id=operation.chunk_id,
                vector=vector,
                payload=final_payload
            )

            # Use provided collection name or default
            coll_name = collection_name or self.collection_name

            # Use the service's upsert_points method which handles async properly
            result = await self.qdrant.upsert_points(coll_name, [point])

            # The upsert_points method returns None on success, raises on error
            # If we get here without exception, it succeeded
            return True

        except Exception as e:
            logger.error(f"Qdrant write exception: {e}")
            return False

    async def _rollback_neo4j(self, chunk_id: int) -> bool:
        """Rollback Neo4j write on failure"""
        try:
            cypher = """
            MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
            DELETE c
            """

            result = await self.neo4j.execute_cypher(
                cypher,
                {'chunk_id': chunk_id, 'project': self.project_name}
            )

            if result.get('status') == 'success':
                logger.info(f"Rolled back Neo4j chunk {chunk_id}")
                return True

            logger.error(f"Failed to rollback Neo4j: {result.get('message')}")
            return False

        except Exception as e:
            logger.error(f"Neo4j rollback exception: {e}")
            return False

    async def _handle_failure(self, operation: SyncOperation, error_msg: str):
        """Handle write failure and track for recovery"""
        operation.error = error_msg
        self.metrics['failed_syncs'] += 1

        # Track partial writes
        if operation.neo4j_status == SyncStatus.SYNCED and \
           operation.qdrant_status != SyncStatus.SYNCED:
            self.metrics['neo4j_only'] += 1
        elif operation.qdrant_status == SyncStatus.SYNCED and \
             operation.neo4j_status != SyncStatus.SYNCED:
            self.metrics['qdrant_only'] += 1

        logger.error(f"Sync failure for chunk {operation.chunk_id}: {error_msg}")

    async def batch_write_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch write multiple chunks with transactional semantics
        Returns summary of operations
        """
        results = {
            'total': len(chunks),
            'successful': 0,
            'failed': 0,
            'chunk_ids': [],
            'errors': []
        }

        # Neo4j batch operation
        neo4j_chunks = []
        qdrant_points = []

        for chunk_data in chunks:
            content = chunk_data['content']
            chunk_id, chunk_hash = self.generate_chunk_id(content)

            neo4j_chunks.append({
                'chunk_id': chunk_id,
                'chunk_hash': chunk_hash,
                'project': self.project_name,
                'content': content,
                'metadata': chunk_data.get('metadata', {})
            })

            qdrant_points.append(PointStruct(
                id=chunk_id,
                vector=chunk_data.get('vector', [0.0] * 768),
                payload={
                    'content': content,
                    'chunk_hash': chunk_hash,
                    'project': self.project_name,
                    **chunk_data.get('metadata', {})
                }
            ))

            results['chunk_ids'].append((chunk_id, chunk_hash))

        try:
            # Batch write to Neo4j
            cypher = """
            UNWIND $chunks as chunk
            CREATE (c:Chunk {
                chunk_id: chunk.chunk_id,
                chunk_hash: chunk.chunk_hash,
                project: chunk.project,
                content: chunk.content,
                indexed_at: datetime()
            })
            SET c += chunk.metadata
            RETURN count(c) as created
            """

            neo4j_result = await self.neo4j.execute_cypher(
                cypher,
                {'chunks': neo4j_chunks}
            )

            if neo4j_result.get('status') != 'success':
                raise Exception(f"Neo4j batch write failed: {neo4j_result.get('message')}")

            # Batch write to Qdrant
            qdrant_result = self.qdrant.client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points,
                wait=True
            )

            results['successful'] = len(chunks)
            self.metrics['successful_syncs'] += len(chunks)

            logger.info(f"✅ Batch synced {len(chunks)} chunks successfully")

        except Exception as e:
            # Rollback Neo4j on failure
            for chunk_id, _ in results['chunk_ids']:
                await self._rollback_neo4j(chunk_id)

            results['failed'] = len(chunks)
            results['errors'].append(str(e))
            self.metrics['failed_syncs'] += len(chunks)

            logger.error(f"❌ Batch sync failed: {e}")

        return results

    async def verify_sync(self, chunk_id: int) -> Dict[str, Any]:
        """Verify a chunk exists in both databases with same content"""
        result = {
            'chunk_id': chunk_id,
            'synced': False,
            'neo4j': None,
            'qdrant': None,
            'status': SyncStatus.PENDING
        }

        # Check Neo4j
        try:
            cypher = """
            MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
            RETURN c.content as content, c.chunk_hash as hash
            """
            neo4j_result = await self.neo4j.execute_cypher(
                cypher,
                {'chunk_id': chunk_id, 'project': self.project_name}
            )

            if neo4j_result.get('status') == 'success' and neo4j_result.get('data'):
                data = neo4j_result['data'][0]
                result['neo4j'] = {
                    'content': data.get('content'),
                    'hash': data.get('hash')
                }
        except Exception as e:
            result['neo4j_error'] = str(e)

        # Check Qdrant
        try:
            points = self.qdrant.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id]
            )

            if points:
                payload = points[0].payload
                result['qdrant'] = {
                    'content': payload.get('content'),
                    'hash': payload.get('chunk_hash')
                }
        except Exception as e:
            result['qdrant_error'] = str(e)

        # Determine sync status
        if result['neo4j'] and result['qdrant']:
            if (result['neo4j']['content'] == result['qdrant']['content'] and
                result['neo4j']['hash'] == result['qdrant']['hash']):
                result['synced'] = True
                result['status'] = SyncStatus.SYNCED
            else:
                result['status'] = SyncStatus.FAILED
        elif result['neo4j'] and not result['qdrant']:
            result['status'] = SyncStatus.NEO4J_ONLY
        elif result['qdrant'] and not result['neo4j']:
            result['status'] = SyncStatus.QDRANT_ONLY
        else:
            result['status'] = SyncStatus.FAILED

        return result

    async def recover_partial_writes(self) -> Dict[str, Any]:
        """Recover from partial write failures"""
        recovery_results = {
            'total_pending': len(self.pending_operations),
            'recovered': 0,
            'failed': 0,
            'details': []
        }

        for chunk_id, operation in list(self.pending_operations.items()):
            sync_status = await self.verify_sync(chunk_id)

            try:
                if sync_status['status'] == SyncStatus.NEO4J_ONLY:
                    # Neo4j has data, Qdrant missing - write to Qdrant
                    success = await self._write_to_qdrant(operation, None)
                    if success:
                        recovery_results['recovered'] += 1
                        operation.qdrant_status = SyncStatus.RECOVERED
                        self.metrics['recovered_syncs'] += 1
                        del self.pending_operations[chunk_id]
                    else:
                        recovery_results['failed'] += 1

                elif sync_status['status'] == SyncStatus.QDRANT_ONLY:
                    # Qdrant has data, Neo4j missing - write to Neo4j
                    success = await self._write_to_neo4j(operation)
                    if success:
                        recovery_results['recovered'] += 1
                        operation.neo4j_status = SyncStatus.RECOVERED
                        self.metrics['recovered_syncs'] += 1
                        del self.pending_operations[chunk_id]
                    else:
                        recovery_results['failed'] += 1

                elif sync_status['status'] == SyncStatus.SYNCED:
                    # Already synced, remove from pending
                    recovery_results['recovered'] += 1
                    del self.pending_operations[chunk_id]

            except Exception as e:
                recovery_results['failed'] += 1
                recovery_results['details'].append({
                    'chunk_id': chunk_id,
                    'error': str(e)
                })

        logger.info(f"Recovery complete: {recovery_results['recovered']}/{recovery_results['total_pending']} recovered")
        return recovery_results

    async def get_sync_metrics(self) -> Dict[str, Any]:
        """Get current synchronization metrics"""
        # Calculate sync rate
        total = await self._get_total_chunks()
        synced = await self._get_synced_chunks()

        sync_rate = synced / total if total > 0 else 1.0

        return {
            'sync_rate': sync_rate,
            'total_chunks': total,
            'synced_chunks': synced,
            'pending_operations': len(self.pending_operations),
            'metrics': self.metrics,
            'health': 'healthy' if sync_rate >= 0.95 else 'degraded'
        }

    async def _get_total_chunks(self) -> int:
        """Get total number of unique chunks across both databases"""
        neo4j_ids = set()
        qdrant_ids = set()

        # Get Neo4j chunk IDs
        cypher = """
        MATCH (c:Chunk {project: $project})
        RETURN collect(c.chunk_id) as ids
        """
        result = await self.neo4j.execute_cypher(cypher, {'project': self.project_name})
        if result.get('status') == 'success' and result.get('data'):
            neo4j_ids = set(result['data'][0].get('ids', []))

        # Get Qdrant chunk IDs
        try:
            scroll_result = self.qdrant.client.scroll(
                collection_name=self.collection_name,
                limit=10000
            )
            qdrant_ids = {point.id for point in scroll_result[0]}
        except:
            pass

        return len(neo4j_ids.union(qdrant_ids))

    async def _get_synced_chunks(self) -> int:
        """Get number of chunks that exist in both databases"""
        neo4j_ids = set()
        qdrant_ids = set()

        # Get Neo4j chunk IDs
        cypher = """
        MATCH (c:Chunk {project: $project})
        RETURN collect(c.chunk_id) as ids
        """
        result = await self.neo4j.execute_cypher(cypher, {'project': self.project_name})
        if result.get('status') == 'success' and result.get('data'):
            neo4j_ids = set(result['data'][0].get('ids', []))

        # Get Qdrant chunk IDs
        try:
            scroll_result = self.qdrant.client.scroll(
                collection_name=self.collection_name,
                limit=10000
            )
            qdrant_ids = {point.id for point in scroll_result[0]}
        except:
            pass

        return len(neo4j_ids.intersection(qdrant_ids))

    async def create_file_chunk_relationship(self, file_path: str, chunk_id: int) -> bool:
        """Create relationship between File and Chunk nodes"""
        try:
            cypher = """
            MATCH (f:File {path: $file_path, project: $project})
            MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
            MERGE (f)-[:HAS_CHUNK]->(c)
            RETURN count(*) as created
            """

            result = await self.neo4j.execute_cypher(
                cypher,
                {
                    'file_path': file_path,
                    'chunk_id': chunk_id,
                    'project': self.project_name
                }
            )

            return result.get('status') == 'success'

        except Exception as e:
            logger.error(f"Failed to create File->Chunk relationship: {e}")
            return False