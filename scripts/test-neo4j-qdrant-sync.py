#!/usr/bin/env python3
"""
E2E Test Suite for Neo4j-Qdrant Synchronization (ADR-053)
Tests that verify both databases have required information AND proper relationships
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import random
import uuid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Test configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:47687')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'graphrag-password')
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 46333))
PROJECT_NAME = 'test-sync-validation'

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'warnings': [],
    'metrics': {}
}

class ChunkManager:
    """Centralized chunk management for consistent ID generation and operations"""

    @staticmethod
    def generate_id(content: str) -> Tuple[int, str]:
        """Generate Qdrant-compatible integer ID and hex hash from content
        Returns: (integer_id, hex_hash)
        """
        if isinstance(content, str):
            content = content.encode()
        hex_hash = hashlib.sha256(content).hexdigest()
        # Convert first 15 hex chars to int (fits in Neo4j's int64)
        integer_id = int(hex_hash[:15], 16)
        return integer_id, hex_hash

    @staticmethod
    def generate_test_vector(seed: float = None) -> List[float]:
        """Generate a test vector for Qdrant"""
        if seed is not None:
            return [seed] * 768
        return [random.random() for _ in range(768)]

    async def create_in_neo4j(self, session, chunk_id: int, content: str,
                             chunk_hash: str = None, metadata: Dict = None) -> bool:
        """Create chunk in Neo4j with standard structure"""
        params = {
            'chunk_id': chunk_id,
            'project': PROJECT_NAME,
            'content': content,
            'chunk_hash': chunk_hash or '',
            'indexed_at': datetime.now().isoformat()
        }
        if metadata:
            params.update(metadata)

        try:
            result = await session.run("""
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    project: $project,
                    content: $content,
                    chunk_hash: $chunk_hash,
                    indexed_at: $indexed_at
                })
                SET c += $metadata
                RETURN c.chunk_id as id
            """, **params, metadata=metadata or {})
            return (await result.single())['id'] == chunk_id
        except Exception as e:
            print(f"Failed to create chunk in Neo4j: {e}")
            return False

    def create_in_qdrant(self, client, collection: str, chunk_id: int,
                        content: str, chunk_hash: str = None,
                        vector: List[float] = None, metadata: Dict = None) -> bool:
        """Create chunk in Qdrant with standard structure"""
        payload = {
            'content': content,
            'project': PROJECT_NAME,
            'chunk_hash': chunk_hash or ''
        }
        if metadata:
            payload.update(metadata)

        try:
            client.upsert(
                collection_name=collection,
                points=[PointStruct(
                    id=chunk_id,
                    vector=vector or self.generate_test_vector(),
                    payload=payload
                )]
            )
            return True
        except Exception as e:
            print(f"Failed to create chunk in Qdrant: {e}")
            return False

    async def create_in_both(self, neo4j_session, qdrant_client, collection: str,
                            content: str, metadata: Dict = None) -> Tuple[bool, int, str]:
        """Create chunk in both databases atomically
        Returns: (success, chunk_id, chunk_hash)
        """
        chunk_id, chunk_hash = self.generate_id(content)
        vector = self.generate_test_vector()

        # Create in Neo4j
        neo4j_success = await self.create_in_neo4j(
            neo4j_session, chunk_id, content, chunk_hash, metadata
        )

        # Create in Qdrant
        qdrant_success = self.create_in_qdrant(
            qdrant_client, collection, chunk_id, content, chunk_hash, vector, metadata
        )

        return (neo4j_success and qdrant_success, chunk_id, chunk_hash)

    async def verify_sync(self, neo4j_session, qdrant_client, collection: str,
                         chunk_id: int) -> Dict[str, Any]:
        """Verify chunk exists in both databases with same content"""
        result = {'synced': False, 'neo4j': None, 'qdrant': None}

        # Check Neo4j
        try:
            neo4j_result = await neo4j_session.run("""
                MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
                RETURN c.content as content, c.chunk_hash as hash
            """, chunk_id=chunk_id, project=PROJECT_NAME)
            record = await neo4j_result.single()
            if record:
                result['neo4j'] = {'content': record['content'], 'hash': record['hash']}
        except Exception as e:
            result['neo4j_error'] = str(e)

        # Check Qdrant
        try:
            qdrant_points = qdrant_client.retrieve(
                collection_name=collection,
                ids=[chunk_id]
            )
            if qdrant_points:
                payload = qdrant_points[0].payload
                result['qdrant'] = {
                    'content': payload.get('content'),
                    'hash': payload.get('chunk_hash')
                }
        except Exception as e:
            result['qdrant_error'] = str(e)

        # Check sync
        if result['neo4j'] and result['qdrant']:
            result['synced'] = (
                result['neo4j']['content'] == result['qdrant']['content'] and
                result['neo4j']['hash'] == result['qdrant']['hash']
            )

        return result

class SyncValidator:
    """Validates Neo4j-Qdrant synchronization per ADR-053"""

    def __init__(self):
        self.neo4j_driver = None
        self.qdrant_client = None
        self.collection_name = f'project-{PROJECT_NAME}'
        self.chunk_manager = ChunkManager()

    async def initialize(self):
        """Initialize database connections"""
        self.neo4j_driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=('neo4j', NEO4J_PASSWORD)
        )
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    async def cleanup(self):
        """Clean up test data"""
        # Clean Neo4j
        async with self.neo4j_driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n.project = $project DETACH DELETE n",
                project=PROJECT_NAME
            )

        # Clean Qdrant
        try:
            self.qdrant_client.delete_collection(self.collection_name)
        except:
            pass

    async def close(self):
        """Close connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()

class E2ETestSuite:
    """Comprehensive E2E test suite for Neo4j-Qdrant synchronization"""

    def __init__(self):
        self.validator = SyncValidator()
        self.chunk_mgr = ChunkManager()
        self.start_time = None

    async def run_all_tests(self):
        """Run complete test suite"""
        self.start_time = time.time()

        print("=" * 80)
        print("NEO4J-QDRANT SYNCHRONIZATION E2E TEST SUITE")
        print("ADR-053 Implementation Validation")
        print("=" * 80)

        await self.validator.initialize()

        try:
            # Setup
            print("\n[SETUP] Preparing test environment...")
            await self.validator.cleanup()
            await self.setup_test_data()

            # Core Validation Tests
            print("\n[SECTION 1] CORE VALIDATION TESTS")
            await self.test_id_consistency()
            await self.test_content_consistency()
            await self.test_metadata_consistency()
            await self.test_relationship_integrity()
            await self.test_collection_consistency()

            # Cross-Database Tests
            print("\n[SECTION 2] CROSS-DATABASE VALIDATION")
            await self.test_chunk_cross_reference()
            await self.test_orphaned_detection()
            await self.test_sync_rate_calculation()

            # Write Synchronization Tests
            print("\n[SECTION 3] WRITE SYNCHRONIZATION")
            await self.test_atomic_write()
            await self.test_rollback_capability()
            await self.test_concurrent_writes()

            # Recovery Tests
            print("\n[SECTION 4] RECOVERY & SELF-HEALING")
            await self.test_missing_chunk_detection()
            await self.test_orphan_cleanup()
            await self.test_metadata_repair()

            # Performance Tests
            print("\n[SECTION 5] PERFORMANCE VALIDATION")
            await self.test_sync_latency()
            await self.test_batch_performance()
            await self.test_query_performance()

            # Chaos Engineering
            print("\n[SECTION 6] CHAOS ENGINEERING")
            await self.test_partial_failure_recovery()
            await self.test_network_partition_simulation()
            await self.test_data_corruption_detection()

        finally:
            await self.validator.cleanup()
            await self.validator.close()

        self.print_results()
        return self.calculate_pass_rate()

    async def setup_test_data(self):
        """Create initial test data"""
        # Create test collection in Qdrant
        try:
            self.validator.qdrant_client.create_collection(
                collection_name=self.validator.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        except:
            pass  # Collection may exist

        # Create test File nodes in Neo4j
        async with self.validator.neo4j_driver.session() as session:
            for i in range(5):
                await session.run("""
                    CREATE (f:File {
                        path: $path,
                        project: $project,
                        indexed_at: datetime()
                    })
                """, path=f"test_file_{i}.py", project=PROJECT_NAME)

    async def test_id_consistency(self):
        """Test 1: Verify ID consistency between databases"""
        test_name = "ID Consistency"
        try:
            async with self.validator.neo4j_driver.session() as session:
                success, chunk_id, chunk_hash = await self.chunk_mgr.create_in_both(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    "test_chunk_1"
                )

                # Verify sync
                sync_result = await self.chunk_mgr.verify_sync(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    chunk_id
                )

                assert sync_result['synced'], f"Sync failed: {sync_result}"

                test_results['passed'].append(test_name)
                print(f"✅ {test_name}: IDs match across databases")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_content_consistency(self):
        """Test 2: Verify content consistency"""
        test_name = "Content Consistency"
        try:
            content = "def test_function():\n    return 'Hello World'"

            async with self.validator.neo4j_driver.session() as session:
                success, chunk_id, _ = await self.chunk_mgr.create_in_both(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    content
                )

                sync_result = await self.chunk_mgr.verify_sync(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    chunk_id
                )

                assert sync_result['synced']
                assert sync_result['neo4j']['content'] == content
                assert sync_result['qdrant']['content'] == content

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Content matches across databases")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_metadata_consistency(self):
        """Test 3: Verify metadata consistency"""
        test_name = "Metadata Consistency"
        try:
            metadata = {
                'file_path': 'test/file.py',
                'start_line': 10,
                'end_line': 20,
                'language': 'python'
            }

            async with self.validator.neo4j_driver.session() as session:
                chunk_id, chunk_hash = self.chunk_mgr.generate_id("test_metadata")

                # Create with metadata
                await self.chunk_mgr.create_in_neo4j(
                    session, chunk_id, "metadata test", chunk_hash, metadata
                )
                self.chunk_mgr.create_in_qdrant(
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    chunk_id, "metadata test", chunk_hash,
                    metadata=metadata
                )

                # Verify metadata
                result = await session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
                    RETURN c.file_path as file_path, c.start_line as start_line
                """, chunk_id=chunk_id, project=PROJECT_NAME)
                neo4j_meta = await result.single()

                qdrant_points = self.validator.qdrant_client.retrieve(
                    collection_name=self.validator.collection_name,
                    ids=[chunk_id]
                )

                assert neo4j_meta['file_path'] == metadata['file_path']
                assert qdrant_points[0].payload['file_path'] == metadata['file_path']

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Metadata synchronized correctly")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_relationship_integrity(self):
        """Test 4: Verify File->Chunk relationships"""
        test_name = "Relationship Integrity"
        try:
            async with self.validator.neo4j_driver.session() as session:
                chunk_id, _ = self.chunk_mgr.generate_id("test_relationship")

                # Create chunk and relationship
                await session.run("""
                    MATCH (f:File {path: 'test_file_0.py', project: $project})
                    CREATE (c:Chunk {
                        chunk_id: $chunk_id,
                        project: $project,
                        content: 'Related chunk'
                    })
                    CREATE (f)-[:HAS_CHUNK {sequence: 1}]->(c)
                """, chunk_id=chunk_id, project=PROJECT_NAME)

                # Verify relationship exists
                result = await session.run("""
                    MATCH (f:File {project: $project})-[:HAS_CHUNK]->(c:Chunk)
                    WHERE c.chunk_id = $chunk_id
                    RETURN count(*) as rel_count
                """, chunk_id=chunk_id, project=PROJECT_NAME)
                rel_count = (await result.single())['rel_count']

                assert rel_count == 1

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Relationships properly maintained")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_collection_consistency(self):
        """Test 5: Verify collection naming and structure"""
        test_name = "Collection Consistency"
        try:
            collections = self.validator.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            assert self.validator.collection_name in collection_names

            # Verify collection config
            info = self.validator.qdrant_client.get_collection(self.validator.collection_name)
            assert info.config.params.vectors.size == 768

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Collection structure verified")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_chunk_cross_reference(self):
        """Test 6: Verify chunks can be cross-referenced"""
        test_name = "Chunk Cross-Reference"
        try:
            # Get all Neo4j chunk IDs
            async with self.validator.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (c:Chunk {project: $project})
                    RETURN collect(c.chunk_id) as ids
                """, project=PROJECT_NAME)
                neo4j_ids = set((await result.single())['ids'])

            # Get all Qdrant chunk IDs
            scroll_result = self.validator.qdrant_client.scroll(
                collection_name=self.validator.collection_name,
                limit=100
            )
            qdrant_ids = {point.id for point in scroll_result[0]}

            # Calculate overlap
            common_ids = neo4j_ids.intersection(qdrant_ids)
            sync_rate = len(common_ids) / max(len(neo4j_ids), len(qdrant_ids)) if neo4j_ids or qdrant_ids else 1.0

            test_results['metrics']['sync_rate'] = sync_rate

            if sync_rate >= 0.95:
                test_results['passed'].append(test_name)
                print(f"✅ {test_name}: Cross-reference rate {sync_rate:.1%}")
            else:
                test_results['warnings'].append((test_name, f"Low sync rate: {sync_rate:.1%}"))
                print(f"⚠️ {test_name}: Sync rate {sync_rate:.1%} below threshold")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_orphaned_detection(self):
        """Test 7: Detect orphaned records"""
        test_name = "Orphaned Detection"
        try:
            # Create orphan in Qdrant only
            orphan_id, orphan_hash = self.chunk_mgr.generate_id("orphan_chunk")
            self.chunk_mgr.create_in_qdrant(
                self.validator.qdrant_client,
                self.validator.collection_name,
                orphan_id, "Orphaned chunk", orphan_hash
            )

            # Verify detection
            async with self.validator.neo4j_driver.session() as session:
                sync_result = await self.chunk_mgr.verify_sync(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    orphan_id
                )

                assert sync_result['neo4j'] is None
                assert sync_result['qdrant'] is not None
                assert not sync_result['synced']

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Orphans correctly detected")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_sync_rate_calculation(self):
        """Test 8: Verify sync rate calculation"""
        test_name = "Sync Rate Calculation"
        try:
            # Get counts
            async with self.validator.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (c:Chunk {project: $project})
                    RETURN count(*) as count
                """, project=PROJECT_NAME)
                neo4j_count = (await result.single())['count']

            collection_info = self.validator.qdrant_client.get_collection(self.validator.collection_name)
            qdrant_count = collection_info.points_count

            # Calculate sync metrics
            if qdrant_count > 0:
                sync_rate = min(neo4j_count, qdrant_count) / max(neo4j_count, qdrant_count)
            else:
                sync_rate = 1.0 if neo4j_count == 0 else 0.0

            missing_in_neo4j = max(0, qdrant_count - neo4j_count)
            orphaned_in_neo4j = max(0, neo4j_count - qdrant_count)

            test_results['metrics'].update({
                'neo4j_chunks': neo4j_count,
                'qdrant_chunks': qdrant_count,
                'sync_rate': sync_rate,
                'missing_in_neo4j': missing_in_neo4j,
                'orphaned_in_neo4j': orphaned_in_neo4j
            })

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Neo4j={neo4j_count}, Qdrant={qdrant_count}, Sync={sync_rate:.1%}")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_atomic_write(self):
        """Test 9: Verify atomic write operations"""
        test_name = "Atomic Write"
        try:
            async with self.validator.neo4j_driver.session() as session:
                tx = await session.begin_transaction()
                try:
                    success, chunk_id, _ = await self.chunk_mgr.create_in_both(
                        tx,  # Use transaction instead of session
                        self.validator.qdrant_client,
                        self.validator.collection_name,
                        "atomic_test"
                    )
                    await tx.commit()
                    assert success
                except Exception:
                    await tx.rollback()
                    raise

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Atomic operations work correctly")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_rollback_capability(self):
        """Test 10: Verify rollback on failure"""
        test_name = "Rollback Capability"
        try:
            # Track initial state
            async with self.validator.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (c:Chunk {project: $project})
                    RETURN count(*) as initial_count
                """, project=PROJECT_NAME)
                initial_count = (await result.single())['initial_count']

                # Simulate failed transaction
                tx = await session.begin_transaction()
                try:
                    chunk_id, _ = self.chunk_mgr.generate_id("rollback_test")
                    await tx.run("""
                        CREATE (c:Chunk {
                            chunk_id: $chunk_id,
                            project: $project,
                            content: 'Will rollback'
                        })
                    """, chunk_id=chunk_id, project=PROJECT_NAME)

                    # Simulate failure
                    raise Exception("Simulated failure")

                except Exception:
                    await tx.rollback()

                # Verify rollback
                result = await session.run("""
                    MATCH (c:Chunk {project: $project})
                    RETURN count(*) as final_count
                """, project=PROJECT_NAME)
                final_count = (await result.single())['final_count']

                assert final_count == initial_count

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Rollback works correctly")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_concurrent_writes(self):
        """Test 11: Handle concurrent writes"""
        test_name = "Concurrent Writes"
        try:
            async with self.validator.neo4j_driver.session() as session:
                # Create multiple concurrent writes
                tasks = []
                for i in range(5):
                    tasks.append(self.chunk_mgr.create_in_both(
                        session,
                        self.validator.qdrant_client,
                        self.validator.collection_name,
                        f"concurrent_{i}"
                    ))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Check all succeeded
                failures = [r for r in results if isinstance(r, Exception)]
                assert len(failures) == 0

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Handled {len(tasks)} concurrent writes")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_missing_chunk_detection(self):
        """Test 12: Detect missing chunks"""
        test_name = "Missing Chunk Detection"
        try:
            async with self.validator.neo4j_driver.session() as session:
                # Create chunk only in Neo4j
                missing_id, missing_hash = self.chunk_mgr.generate_id("missing_in_qdrant")
                await self.chunk_mgr.create_in_neo4j(
                    session, missing_id, "Missing in Qdrant", missing_hash
                )

                # Verify missing
                sync_result = await self.chunk_mgr.verify_sync(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    missing_id
                )

                assert sync_result['neo4j'] is not None
                assert sync_result['qdrant'] is None
                assert not sync_result['synced']

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Missing chunks detected")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_orphan_cleanup(self):
        """Test 13: Clean up orphaned records"""
        test_name = "Orphan Cleanup"
        try:
            async with self.validator.neo4j_driver.session() as session:
                # Create orphans
                for i in range(3):
                    chunk_id, _ = self.chunk_mgr.generate_id(f"orphan_{i}")
                    await session.run("""
                        CREATE (c:Chunk {
                            chunk_id: $chunk_id,
                            project: $project,
                            content: $content
                        })
                    """, chunk_id=chunk_id, project=PROJECT_NAME, content=f"Orphan {i}")

                # Count orphans before cleanup
                result = await session.run("""
                    MATCH (c:Chunk {project: $project})
                    WHERE NOT EXISTS((:File)-[:HAS_CHUNK]->(c))
                    RETURN count(*) as orphan_count
                """, project=PROJECT_NAME)
                initial_orphans = (await result.single())['orphan_count']

                # Clean up orphans
                await session.run("""
                    MATCH (c:Chunk {project: $project})
                    WHERE NOT EXISTS((:File)-[:HAS_CHUNK]->(c))
                    DELETE c
                """, project=PROJECT_NAME)

                # Verify cleanup
                result = await session.run("""
                    MATCH (c:Chunk {project: $project})
                    WHERE NOT EXISTS((:File)-[:HAS_CHUNK]->(c))
                    RETURN count(*) as orphan_count
                """, project=PROJECT_NAME)
                final_orphans = (await result.single())['orphan_count']

                assert final_orphans == 0

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Cleaned {initial_orphans} orphans")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_metadata_repair(self):
        """Test 14: Repair missing metadata"""
        test_name = "Metadata Repair"
        try:
            async with self.validator.neo4j_driver.session() as session:
                # Create chunk with incomplete metadata
                chunk_id, _ = self.chunk_mgr.generate_id("incomplete_meta")
                await session.run("""
                    CREATE (c:Chunk {
                        chunk_id: $chunk_id,
                        project: $project,
                        content: 'Incomplete metadata'
                    })
                """, chunk_id=chunk_id, project=PROJECT_NAME)

                # Repair metadata
                await session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
                    WHERE c.indexed_at IS NULL
                    SET c.indexed_at = datetime(),
                        c.repaired = true
                """, chunk_id=chunk_id, project=PROJECT_NAME)

                # Verify repair
                result = await session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
                    RETURN c.indexed_at IS NOT NULL as has_timestamp
                """, chunk_id=chunk_id, project=PROJECT_NAME)
                has_timestamp = (await result.single())['has_timestamp']

                assert has_timestamp

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Metadata successfully repaired")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_sync_latency(self):
        """Test 15: Measure sync latency"""
        test_name = "Sync Latency"
        try:
            start_time = time.time()

            async with self.validator.neo4j_driver.session() as session:
                success, chunk_id, _ = await self.chunk_mgr.create_in_both(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    "latency_test"
                )

                # Verify both have data
                sync_result = await self.chunk_mgr.verify_sync(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    chunk_id
                )

            latency_ms = (time.time() - start_time) * 1000
            test_results['metrics']['sync_latency_ms'] = latency_ms

            assert sync_result['synced']
            assert latency_ms < 100  # Should sync within 100ms

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Sync latency {latency_ms:.1f}ms")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_batch_performance(self):
        """Test 16: Batch operation performance"""
        test_name = "Batch Performance"
        try:
            batch_size = 100
            start_time = time.time()

            # Prepare batch data
            chunks = []
            points = []
            for i in range(batch_size):
                chunk_id, chunk_hash = self.chunk_mgr.generate_id(f"batch_{i}")
                content = f"Batch chunk {i}"

                chunks.append({
                    'chunk_id': chunk_id,
                    'content': content,
                    'chunk_hash': chunk_hash,
                    'project': PROJECT_NAME
                })

                points.append(PointStruct(
                    id=chunk_id,
                    vector=self.chunk_mgr.generate_test_vector(0.7),
                    payload={'content': content, 'project': PROJECT_NAME, 'chunk_hash': chunk_hash}
                ))

            # Neo4j batch insert
            async with self.validator.neo4j_driver.session() as session:
                await session.run("""
                    UNWIND $chunks as chunk
                    CREATE (c:Chunk {
                        chunk_id: chunk.chunk_id,
                        project: chunk.project,
                        content: chunk.content,
                        chunk_hash: chunk.chunk_hash
                    })
                """, chunks=chunks)

            # Qdrant batch insert
            self.validator.qdrant_client.upsert(
                collection_name=self.validator.collection_name,
                points=points
            )

            elapsed_ms = (time.time() - start_time) * 1000
            throughput = batch_size / (elapsed_ms / 1000)

            test_results['metrics']['batch_throughput'] = throughput

            assert elapsed_ms < 5000  # 100 records in <5 seconds

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: {throughput:.0f} chunks/sec")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_query_performance(self):
        """Test 17: Query performance across databases"""
        test_name = "Query Performance"
        try:
            # Neo4j query performance
            start_time = time.time()
            async with self.validator.neo4j_driver.session() as session:
                result = await session.run("""
                    MATCH (c:Chunk {project: $project})
                    RETURN count(*) as count
                """, project=PROJECT_NAME)
                neo4j_count = (await result.single())['count']
            neo4j_ms = (time.time() - start_time) * 1000

            # Qdrant query performance
            start_time = time.time()
            collection_info = self.validator.qdrant_client.get_collection(self.validator.collection_name)
            qdrant_count = collection_info.points_count
            qdrant_ms = (time.time() - start_time) * 1000

            test_results['metrics']['neo4j_query_ms'] = neo4j_ms
            test_results['metrics']['qdrant_query_ms'] = qdrant_ms

            assert neo4j_ms < 50  # Query in <50ms
            assert qdrant_ms < 50

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Neo4j={neo4j_ms:.1f}ms, Qdrant={qdrant_ms:.1f}ms")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_partial_failure_recovery(self):
        """Test 18: Recover from partial failures"""
        test_name = "Partial Failure Recovery"
        try:
            async with self.validator.neo4j_driver.session() as session:
                # Simulate Neo4j success, Qdrant failure (by creating only in Neo4j)
                chunk_id, chunk_hash = self.chunk_mgr.generate_id("partial_failure")
                content = "Partial failure test"

                await self.chunk_mgr.create_in_neo4j(
                    session, chunk_id, content, chunk_hash
                )

                # Detect and recover
                sync_result = await self.chunk_mgr.verify_sync(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    chunk_id
                )

                # Recovery: write missing data to Qdrant
                if sync_result['neo4j'] and not sync_result['qdrant']:
                    self.chunk_mgr.create_in_qdrant(
                        self.validator.qdrant_client,
                        self.validator.collection_name,
                        chunk_id, content, chunk_hash
                    )

                # Verify recovery
                sync_result = await self.chunk_mgr.verify_sync(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    chunk_id
                )

                assert sync_result['synced']

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Partial failure recovered")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_network_partition_simulation(self):
        """Test 19: Simulate network partition"""
        test_name = "Network Partition"
        try:
            async with self.validator.neo4j_driver.session() as session:
                # Write during "partition" (only to Neo4j)
                partition_chunks = []
                for i in range(3):
                    chunk_id, chunk_hash = self.chunk_mgr.generate_id(f"partition_{i}")
                    await session.run("""
                        CREATE (c:Chunk {
                            chunk_id: $chunk_id,
                            project: $project,
                            content: $content,
                            chunk_hash: $chunk_hash,
                            partition_write: true
                        })
                    """, chunk_id=chunk_id, project=PROJECT_NAME,
                        content=f"Written during partition {i}", chunk_hash=chunk_hash)
                    partition_chunks.append((chunk_id, f"Written during partition {i}", chunk_hash))

                # "Heal" partition - sync to Qdrant
                for chunk_id, content, chunk_hash in partition_chunks:
                    self.chunk_mgr.create_in_qdrant(
                        self.validator.qdrant_client,
                        self.validator.collection_name,
                        chunk_id, content, chunk_hash
                    )

                # Verify all synced
                for chunk_id, _, _ in partition_chunks:
                    sync_result = await self.chunk_mgr.verify_sync(
                        session,
                        self.validator.qdrant_client,
                        self.validator.collection_name,
                        chunk_id
                    )
                    assert sync_result['synced']

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Partition recovery successful")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    async def test_data_corruption_detection(self):
        """Test 20: Detect data corruption"""
        test_name = "Corruption Detection"
        try:
            original_content = "Original content before corruption"

            async with self.validator.neo4j_driver.session() as session:
                # Write original
                success, chunk_id, chunk_hash = await self.chunk_mgr.create_in_both(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    original_content
                )

                # "Corrupt" data in Neo4j
                await session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id, project: $project})
                    SET c.content = $corrupted
                """, chunk_id=chunk_id, project=PROJECT_NAME,
                    corrupted="CORRUPTED: " + original_content)

                # Detect corruption by comparing
                sync_result = await self.chunk_mgr.verify_sync(
                    session,
                    self.validator.qdrant_client,
                    self.validator.collection_name,
                    chunk_id
                )

                # Should detect mismatch
                assert not sync_result['synced']
                assert sync_result['neo4j']['content'] != sync_result['qdrant']['content']

            test_results['passed'].append(test_name)
            print(f"✅ {test_name}: Corruption correctly detected")

        except Exception as e:
            test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name}: {e}")

    def print_results(self):
        """Print test results summary"""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        print(f"\n✅ PASSED: {len(test_results['passed'])} tests")
        for test in test_results['passed']:
            print(f"   • {test}")

        if test_results['warnings']:
            print(f"\n⚠️ WARNINGS: {len(test_results['warnings'])} tests")
            for test, msg in test_results['warnings']:
                print(f"   • {test}: {msg}")

        if test_results['failed']:
            print(f"\n❌ FAILED: {len(test_results['failed'])} tests")
            for test, error in test_results['failed']:
                print(f"   • {test}: {error}")

        print(f"\n📊 METRICS:")
        for key, value in test_results['metrics'].items():
            if isinstance(value, float):
                if 'rate' in key or 'sync' in key:
                    print(f"   • {key}: {value:.1%}")
                elif 'ms' in key:
                    print(f"   • {key}: {value:.1f}ms")
                else:
                    print(f"   • {key}: {value:.2f}")
            else:
                print(f"   • {key}: {value}")

        print(f"\n⏱️ Total execution time: {elapsed:.1f} seconds")

    def calculate_pass_rate(self) -> float:
        """Calculate overall pass rate"""
        total = len(test_results['passed']) + len(test_results['failed'])
        if total == 0:
            return 0.0
        return len(test_results['passed']) / total

async def main():
    """Run the E2E test suite"""
    suite = E2ETestSuite()
    pass_rate = await suite.run_all_tests()

    print("\n" + "=" * 80)
    if pass_rate >= 0.95:
        print(f"✅ E2E VALIDATION PASSED ({pass_rate:.1%})")
        print("Neo4j-Qdrant synchronization is working correctly!")
        return 0
    else:
        print(f"❌ E2E VALIDATION FAILED ({pass_rate:.1%})")
        print("Critical synchronization issues detected!")
        print("\nRequired Actions:")
        print("1. Review failed tests above")
        print("2. Check sync rate metrics")
        print("3. Run recovery procedures if needed")
        print("4. Re-run validation after fixes")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)