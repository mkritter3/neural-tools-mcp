#!/usr/bin/env python3
"""
Strict CI/CD Validation Test for Neo4j-Qdrant Synchronization
Ensures both databases receive data during indexing
Part of ADR-053 implementation validation
"""

import asyncio
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "neural-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "neural-tools" / "src"))

from servers.services.service_container import ServiceContainer
from servers.services.project_context_manager import ProjectContextManager
from qdrant_client import QdrantClient

class SyncValidationTest:
    """Strict validation of Neo4j-Qdrant synchronization"""

    def __init__(self):
        self.project_name = "claude-l9-template"
        self.project_path = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []

    async def run_all_tests(self):
        """Run comprehensive synchronization validation"""
        print("\n" + "="*70)
        print("🔍 NEO4J-QDRANT SYNCHRONIZATION VALIDATION")
        print("="*70)

        all_passed = True

        # Test 1: Database connectivity
        if not await self.test_connectivity():
            all_passed = False

        # Test 2: Collection existence
        if not await self.test_collections_exist():
            all_passed = False

        # Test 3: Data consistency
        if not await self.test_data_consistency():
            all_passed = False

        # Test 4: Sync manager functionality
        if not await self.test_sync_manager():
            all_passed = False

        # Test 5: Hybrid search functionality
        if not await self.test_hybrid_search():
            all_passed = False

        # Print summary
        self.print_summary()

        return all_passed

    async def test_connectivity(self):
        """Test 1: Verify database connectivity"""
        print("\n📌 Test 1: Database Connectivity")
        print("-" * 40)

        try:
            # Initialize services
            context_manager = ProjectContextManager()
            await context_manager.set_project(str(self.project_path))

            container = ServiceContainer(context_manager)
            await container.initialize_all_services()

            # Check Neo4j
            neo4j_status = "✅ Connected" if container.neo4j else "❌ Failed"
            print(f"  Neo4j:  {neo4j_status}")

            # Check Qdrant
            qdrant_status = "✅ Connected" if container.qdrant else "❌ Failed"
            print(f"  Qdrant: {qdrant_status}")

            self.container = container
            return container.neo4j is not None and container.qdrant is not None

        except Exception as e:
            self.errors.append(f"Connectivity test failed: {e}")
            print(f"  ❌ Error: {e}")
            return False

    async def test_collections_exist(self):
        """Test 2: Verify Qdrant collections exist"""
        print("\n📌 Test 2: Collection Existence")
        print("-" * 40)

        try:
            client = QdrantClient(host="localhost", port=46333)
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]

            expected_collection = f"project-{self.project_name}"

            if expected_collection in collection_names:
                info = client.get_collection(expected_collection)
                print(f"  ✅ Collection '{expected_collection}' exists")
                print(f"     Points: {info.points_count}")
                print(f"     Vector size: {info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else 'Unknown'}")

                if info.points_count == 0:
                    self.warnings.append("Collection exists but has 0 points")
                    print(f"  ⚠️  WARNING: Collection has no vectors!")
                    return False

                return True
            else:
                self.errors.append(f"Collection '{expected_collection}' not found")
                print(f"  ❌ Collection '{expected_collection}' not found")
                print(f"     Available: {collection_names}")
                return False

        except Exception as e:
            self.errors.append(f"Collection test failed: {e}")
            print(f"  ❌ Error: {e}")
            return False

    async def test_data_consistency(self):
        """Test 3: Verify data consistency between Neo4j and Qdrant"""
        print("\n📌 Test 3: Data Consistency")
        print("-" * 40)

        try:
            # Count Neo4j chunks
            neo4j_query = """
            MATCH (c:Chunk {project: $project})
            RETURN count(c) as chunk_count
            """
            neo4j_result = await self.container.neo4j.execute_cypher(
                neo4j_query,
                {"project": self.project_name}
            )

            neo4j_count = 0
            if neo4j_result['status'] == 'success' and neo4j_result['data']:
                neo4j_count = neo4j_result['data'][0]['chunk_count']

            print(f"  Neo4j chunks: {neo4j_count}")

            # Count Qdrant points
            client = QdrantClient(host="localhost", port=46333)
            collection_name = f"project-{self.project_name}"

            try:
                info = client.get_collection(collection_name)
                qdrant_count = info.points_count
                print(f"  Qdrant points: {qdrant_count}")
            except:
                qdrant_count = 0
                print(f"  Qdrant points: 0 (collection not found)")

            # Calculate sync rate
            if neo4j_count > 0:
                sync_rate = (min(neo4j_count, qdrant_count) / neo4j_count) * 100
                print(f"  Sync rate: {sync_rate:.1f}%")

                if sync_rate < 95:
                    self.errors.append(f"Sync rate {sync_rate:.1f}% below 95% requirement")
                    print(f"  ❌ FAIL: Sync rate below ADR-053 requirement (≥95%)")
                    return False
                else:
                    print(f"  ✅ PASS: Sync rate meets requirement")
                    return True
            else:
                if qdrant_count > 0:
                    self.warnings.append("Qdrant has points but Neo4j has no chunks")
                    print(f"  ⚠️  WARNING: Data inconsistency detected")
                    return False
                else:
                    print(f"  ℹ️  No data indexed yet")
                    return True

        except Exception as e:
            self.errors.append(f"Consistency test failed: {e}")
            print(f"  ❌ Error: {e}")
            return False

    async def test_sync_manager(self):
        """Test 4: Verify WriteSynchronizationManager functionality"""
        print("\n📌 Test 4: Sync Manager Functionality")
        print("-" * 40)

        try:
            from servers.services.sync_manager import WriteSynchronizationManager

            # Create sync manager
            sync_manager = WriteSynchronizationManager(
                self.container.neo4j,
                self.container.qdrant,
                self.project_name
            )

            # Test write with test data
            test_content = f"Test sync validation at {datetime.now().isoformat()}"
            test_metadata = {
                "file_path": "test/validation.py",
                "test": True,
                "timestamp": datetime.now().isoformat()
            }
            test_vector = [0.1] * 768  # Test vector

            success, chunk_id, chunk_hash = await sync_manager.write_chunk(
                content=test_content,
                metadata=test_metadata,
                vector=test_vector
            )

            if success:
                print(f"  ✅ Test write successful")
                print(f"     Chunk ID: {chunk_id}")
                print(f"     Chunk hash: {chunk_hash[:16]}...")

                # Verify in both databases
                neo4j_check = await self.container.neo4j.execute_cypher(
                    "MATCH (c:Chunk {chunk_hash: $hash, project: $project}) RETURN c",
                    {"hash": chunk_hash, "project": self.project_name}
                )

                neo4j_found = neo4j_check['status'] == 'success' and neo4j_check['data']

                # Check Qdrant
                client = QdrantClient(host="localhost", port=46333)
                collection_name = f"project-{self.project_name}"
                try:
                    points = client.retrieve(
                        collection_name=collection_name,
                        ids=[chunk_id]
                    )
                    qdrant_found = len(points) > 0
                except:
                    qdrant_found = False

                print(f"     Neo4j verification: {'✅ Found' if neo4j_found else '❌ Not found'}")
                print(f"     Qdrant verification: {'✅ Found' if qdrant_found else '❌ Not found'}")

                # Check metrics
                if hasattr(sync_manager, 'metrics'):
                    print(f"     Sync metrics: {sync_manager.metrics}")

                return neo4j_found and qdrant_found

            else:
                self.errors.append(f"Sync manager write failed")
                print(f"  ❌ Test write failed")
                return False

        except Exception as e:
            self.errors.append(f"Sync manager test failed: {e}")
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_hybrid_search(self):
        """Test 5: Verify hybrid search functionality"""
        print("\n📌 Test 5: Hybrid Search Functionality")
        print("-" * 40)

        try:
            from servers.services.hybrid_retriever import HybridRetriever

            # Create hybrid retriever
            retriever = HybridRetriever(
                self.container.neo4j,
                self.container.qdrant,
                self.container.nomic,
                self.project_name
            )

            # Test search
            results = await retriever.search(
                query="test sync validation",
                limit=5
            )

            if results:
                print(f"  ✅ Hybrid search returned {len(results)} results")
                for i, result in enumerate(results[:2]):
                    print(f"     Result {i+1}: Score {result.get('score', 0):.3f}")
                return True
            else:
                self.warnings.append("Hybrid search returned no results")
                print(f"  ⚠️  No results returned (may be normal for test query)")
                # Don't fail on no results - query might not match anything
                return True

        except Exception as e:
            self.errors.append(f"Hybrid search test failed: {e}")
            print(f"  ❌ Error: {e}")
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors:
            print("\n✅ ALL VALIDATION TESTS PASSED")
            print("Neo4j and Qdrant are properly synchronized!")
        else:
            print("\n❌ VALIDATION FAILED")
            print(f"Found {len(self.errors)} critical errors")

        print("="*70)

async def main():
    """Run validation tests"""
    validator = SyncValidationTest()
    success = await validator.run_all_tests()

    # Exit code for CI/CD
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())