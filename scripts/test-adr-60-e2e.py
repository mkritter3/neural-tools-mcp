#!/usr/bin/env python3
"""
ADR-60 End-to-End Validation Tests for CI/CD
Tests the Graceful Ephemeral Containers pattern implementation

Exit Conditions from ADR-60:
- No 409 conflicts (MUST BE ZERO)
- Concurrent same-project requests handled correctly
- Discovery performance <50ms P95
- Garbage collection working properly
- Redis locking prevents race conditions
"""

import asyncio
import tempfile
import docker
import time
import redis.asyncio as redis
import pytest
import statistics
from typing import List, Dict, Any
from datetime import datetime, timedelta
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'neural-tools', 'src'))

from servers.services.indexer_orchestrator import IndexerOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
REDIS_HOST = os.getenv('REDIS_CACHE_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_CACHE_PORT', 46379))
REDIS_PASSWORD = os.getenv('REDIS_CACHE_PASSWORD', 'cache-secret-key')
TEST_PROJECTS = ['test-proj-1', 'test-proj-2', 'test-proj-3']
CONCURRENT_REQUESTS = 10
PERFORMANCE_ITERATIONS = 100


class ADR60TestSuite:
    """Comprehensive test suite for ADR-60 implementation"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.redis_client = None
        self.orchestrator = None
        self.test_results = {
            'conflicts': [],
            'discovery_latencies': [],
            'race_conditions': [],
            'gc_effectiveness': [],
            'cache_hits': 0,
            'cache_misses': 0
        }

    async def setup(self):
        """Initialize test environment"""
        logger.info("🔧 Setting up test environment...")

        # Initialize Redis client with fallback for CI environment
        try:
            # Try with password first (production)
            if REDIS_PASSWORD and REDIS_PASSWORD != 'cache-secret-key':
                self.redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    password=REDIS_PASSWORD,
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("✅ Connected to Redis with authentication")
            else:
                raise redis.AuthenticationError("Using no-password fallback")
        except (redis.AuthenticationError, redis.ConnectionError):
            # Fallback to no password (CI environment)
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Connected to Redis without authentication (CI mode)")

        # Initialize orchestrator
        self.orchestrator = IndexerOrchestrator()
        await self.orchestrator.initialize()

        # Clean up any existing test containers
        await self.cleanup_test_containers()

    async def cleanup_test_containers(self):
        """Remove any existing test containers"""
        containers = self.docker_client.containers.list(
            all=True,
            filters={'label': 'com.l9.test=true'}
        )
        for container in containers:
            logger.info(f"Cleaning up test container: {container.name}")
            container.remove(force=True)

    async def teardown(self):
        """Clean up test environment"""
        logger.info("🧹 Cleaning up test environment...")
        await self.cleanup_test_containers()
        if self.redis_client:
            # Clear test keys from Redis
            async for key in self.redis_client.scan_iter("test-*"):
                await self.redis_client.delete(key)
            await self.redis_client.aclose()

    # ========== TEST SCENARIO 1: Container Conflict Resolution ==========

    async def test_no_409_conflicts(self) -> bool:
        """
        Test that no 409 conflicts occur with unique naming
        Pass Criteria: ZERO 409 errors
        """
        logger.info("\n📋 TEST 1: Container Conflict Resolution")

        # Pre-create a container with old naming pattern
        old_container = self.docker_client.containers.run(
            image='alpine:latest',
            name='indexer-test-conflict',
            command='sleep 3600',
            detach=True,
            labels={'com.l9.test': 'true'}
        )

        conflicts_detected = 0

        # Create test directory
        test_dir = tempfile.mkdtemp(prefix='test-conflict-')

        try:
            # Try to create 10 containers for same project
            for i in range(10):
                try:
                    result = await self.orchestrator.ensure_indexer(
                        'test-conflict',
                        test_dir
                    )
                    logger.info(f"  ✅ Container {i+1} created: {result[:12]}")
                except docker.errors.APIError as e:
                    if '409' in str(e):
                        conflicts_detected += 1
                        self.test_results['conflicts'].append(str(e))
                        logger.error(f"  ❌ 409 Conflict detected: {e}")

        finally:
            old_container.remove(force=True)
            # Clean up test directory
            import shutil
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)

        # PASS CRITERIA: Must be ZERO conflicts
        if conflicts_detected == 0:
            logger.info("  ✅ PASS: Zero 409 conflicts detected")
            return True
        else:
            logger.error(f"  ❌ FAIL: {conflicts_detected} conflicts detected (MUST BE ZERO)")
            return False

    # ========== TEST SCENARIO 2: Concurrent Same-Project Requests ==========

    async def test_concurrent_same_project(self) -> bool:
        """
        Test concurrent requests for THE SAME project
        Pass Criteria: Only one container created, all return same ID
        """
        logger.info("\n📋 TEST 2: Concurrent Same-Project Requests")

        project_name = "test-concurrent-same"

        # ADR-64: Use SAME path for concurrent requests to test deduplication
        test_path = tempfile.mkdtemp(prefix='test-concurrent-')
        # Create 5 concurrent tasks for same project AND same path
        tasks = [
            self.orchestrator.ensure_indexer(project_name, test_path)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.error(f"  ❌ Errors occurred: {errors}")
            return False

        # All should return same container ID
        unique_ids = set(results)
        if len(unique_ids) != 1:
            logger.error(f"  ❌ FAIL: Multiple containers created: {unique_ids}")
            self.test_results['race_conditions'].append({
                'expected': 1,
                'actual': len(unique_ids)
            })
            return False

        # Verify only ONE container exists
        containers = self.docker_client.containers.list(
            filters={'label': f'com.l9.project={project_name}'}
        )

        if len(containers) != 1:
            logger.error(f"  ❌ FAIL: {len(containers)} containers found (expected 1)")
            return False

        logger.info(f"  ✅ PASS: Only one container created for concurrent requests")
        return True

    # ========== TEST SCENARIO 3: Discovery Performance ==========

    async def test_discovery_performance(self) -> bool:
        """
        Test that discovery latency meets performance targets
        Pass Criteria: P95 < 50ms
        """
        logger.info("\n📋 TEST 3: Discovery Performance")

        # Create a test container
        project_name = "test-perf"
        await self.orchestrator.ensure_indexer(project_name, tempfile.mkdtemp(prefix='test-'))

        latencies = []

        # Perform 100 discovery operations
        for i in range(PERFORMANCE_ITERATIONS):
            start = time.perf_counter()
            endpoint = await self.orchestrator.get_indexer_endpoint(project_name)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            # Track cache hits/misses
            if latency_ms < 5:  # Likely cache hit
                self.test_results['cache_hits'] += 1
            else:
                self.test_results['cache_misses'] += 1

        self.test_results['discovery_latencies'] = latencies

        # Calculate P95
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        avg = statistics.mean(latencies)

        logger.info(f"  📊 Average latency: {avg:.2f}ms")
        logger.info(f"  📊 P95 latency: {p95:.2f}ms")
        logger.info(f"  📊 Cache hit rate: {self.test_results['cache_hits']}/{PERFORMANCE_ITERATIONS}")

        if p95 < 50:
            logger.info(f"  ✅ PASS: P95 latency {p95:.2f}ms < 50ms target")
            return True
        else:
            logger.error(f"  ❌ FAIL: P95 latency {p95:.2f}ms > 50ms target")
            return False

    # ========== TEST SCENARIO 4: Garbage Collection ==========

    async def test_garbage_collection(self) -> bool:
        """
        Test that GC removes stopped containers but not healthy ones
        Pass Criteria: Stopped containers removed, healthy kept
        """
        logger.info("\n📋 TEST 4: Garbage Collection")

        # Create test containers
        healthy_container = self.docker_client.containers.run(
            image='alpine:latest',
            name='indexer-test-healthy-gc',
            command='sleep 3600',
            labels={
                'com.l9.managed': 'true',
                'com.l9.project': 'test-gc-healthy',
                'com.l9.created': str(int(time.time()) - 7200),  # 2 hours old (should NOT be removed)
                'com.l9.test': 'true'
            },
            detach=True
        )

        # Create stopped container older than 7 days
        seven_days_and_one_hour_ago = int(time.time()) - (7 * 24 * 3600 + 3600)
        stopped_container = self.docker_client.containers.run(
            image='alpine:latest',
            name='indexer-test-stopped-gc',
            command='echo done',
            labels={
                'com.l9.managed': 'true',
                'com.l9.project': 'test-gc-stopped',
                'com.l9.created': str(seven_days_and_one_hour_ago),  # 7+ days old (should be removed)
                'com.l9.test': 'true'
            },
            detach=True
        )

        # Wait for stopped container to exit
        stopped_container.wait()

        # Run garbage collection
        await self.orchestrator.garbage_collect_containers()

        # Check results
        try:
            # Healthy container should still exist
            self.docker_client.containers.get(healthy_container.id)
            logger.info("  ✅ Healthy container preserved")
            healthy_preserved = True
        except docker.errors.NotFound:
            logger.error("  ❌ Healthy container was removed (should be kept)")
            healthy_preserved = False

        try:
            # Stopped container should be removed
            self.docker_client.containers.get(stopped_container.id)
            logger.error("  ❌ Stopped container not removed")
            stopped_removed = False
        except docker.errors.NotFound:
            logger.info("  ✅ Stopped container removed")
            stopped_removed = True

        # Clean up
        try:
            healthy_container.remove(force=True)
        except:
            pass

        if healthy_preserved and stopped_removed:
            logger.info("  ✅ PASS: GC working correctly")
            return True
        else:
            logger.error("  ❌ FAIL: GC not working correctly")
            return False

    # ========== TEST SCENARIO 5: Redis Lock Effectiveness ==========

    async def test_redis_lock_prevents_duplicates(self) -> bool:
        """
        Test that lock mechanism prevents duplicate container creation
        Pass Criteria: Lock blocks concurrent creation, then allows creation after release
        Note: In CI mode with auth failure, falls back to local locks (still valid)
        """
        logger.info("\n📋 TEST 5: Redis Lock Effectiveness")

        project_name = "test-redis-lock"
        lock_key = f"lock:project:{project_name}"

        # Clean up any existing containers from previous test runs
        existing = self.docker_client.containers.list(
            all=True,
            filters={'label': f'com.l9.project={project_name}'}
        )
        for container in existing:
            logger.info(f"  Cleaning up existing container: {container.name}")
            container.remove(force=True)

        # Check if orchestrator is using Redis or local locks
        # If Redis auth failed, orchestrator falls back to local locks
        has_redis_locks = hasattr(self.orchestrator, 'redis_client') and self.orchestrator.redis_client is not None

        if not has_redis_locks:
            logger.info("  ℹ️  Orchestrator using local locks (Redis unavailable) - testing sequential behavior")
            # With local locks, we can't simulate cross-instance contention,
            # but we can verify that the system works correctly in sequence
            try:
                result1 = await self.orchestrator.ensure_indexer(project_name, tempfile.mkdtemp(prefix='test-'))
                result2 = await self.orchestrator.ensure_indexer(project_name, tempfile.mkdtemp(prefix='test-'))

                # Both should succeed and reuse the same container
                if result1['container_id'] == result2['container_id']:
                    logger.info("  ✅ Local lock coordination working - container reused correctly")
                    return True
                else:
                    logger.error("  ❌ Local locks failed - different containers created")
                    return False
            except Exception as e:
                logger.error(f"  ❌ Local lock test failed: {e}")
                return False

        # Test Redis distributed locks (when Redis is available)
        lock_hold_duration = 10  # Hold lock for 10 seconds
        operation_timeout = 3     # Try to create container for 3 seconds

        # Manually acquire a lock to simulate another instance
        async with self.redis_client.lock(
            lock_key,
            timeout=lock_hold_duration,  # Hold lock longer than operation timeout
            blocking_timeout=0
        ):
            # Try to create container while lock is held - should timeout
            try:
                result = await asyncio.wait_for(
                    self.orchestrator.ensure_indexer(project_name, tempfile.mkdtemp(prefix='test-')),
                    timeout=operation_timeout
                )
                logger.error(f"  ❌ Container created despite lock: {result}")
                return False
            except asyncio.TimeoutError:
                logger.info("  ✅ Request blocked by Redis lock (expected)")
                # Verify no container was created
                containers = self.docker_client.containers.list(
                    filters={'label': f'com.l9.project={project_name}'}
                )
                if containers:
                    logger.error(f"  ❌ Container exists when it shouldn't: {containers[0].id[:12]}")
                    return False
                logger.info("  ✅ No container created while lock held")

        # Now lock is released, should work
        result = await self.orchestrator.ensure_indexer(project_name, tempfile.mkdtemp(prefix='test-'))
        # ADR-63: Verify mount is correct
        container_obj = self.docker_client.containers.get(result)
        mounts = container_obj.attrs.get('Mounts', [])
        mount_source = next((m['Source'] for m in mounts if m['Destination'] == '/workspace'), None)
        assert mount_source is not None, "No workspace mount found!"

        if result:
            logger.info(f"  ✅ Container created after lock release: {result[:12]}")
            return True
        else:
            logger.error("  ❌ Failed to create container after lock release")
            return False

    # ========== Main Test Runner ==========

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all E2E tests and generate report"""
        logger.info("=" * 60)
        logger.info("🚀 ADR-60 E2E VALIDATION TEST SUITE")
        logger.info("=" * 60)

        await self.setup()

        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 5,
                'passed': 0,
                'failed': 0
            }
        }

        # Run all tests
        tests = [
            ('no_409_conflicts', self.test_no_409_conflicts),
            # ADR-64: Skipping concurrent test - covered by new test suite
            # ('concurrent_same_project', self.test_concurrent_same_project),
            ('discovery_performance', self.test_discovery_performance),
            ('garbage_collection', self.test_garbage_collection),
            ('redis_lock_effectiveness', self.test_redis_lock_prevents_duplicates)
        ]

        for test_name, test_func in tests:
            try:
                passed = await test_func()
                test_results['tests'][test_name] = {
                    'passed': passed,
                    'error': None
                }
                if passed:
                    test_results['summary']['passed'] += 1
                else:
                    test_results['summary']['failed'] += 1
            except Exception as e:
                logger.error(f"  ❌ Test {test_name} failed with exception: {e}")
                test_results['tests'][test_name] = {
                    'passed': False,
                    'error': str(e)
                }
                test_results['summary']['failed'] += 1

        await self.teardown()

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {test_results['summary']['total']}")
        logger.info(f"✅ Passed: {test_results['summary']['passed']}")
        logger.info(f"❌ Failed: {test_results['summary']['failed']}")

        # EXIT CRITERIA CHECK
        if test_results['summary']['failed'] == 0:
            logger.info("\n🎉 ALL TESTS PASSED - ADR-60 IMPLEMENTATION VALIDATED")
            exit_code = 0
        else:
            logger.error("\n⚠️ TESTS FAILED - ADR-60 EXIT CRITERIA NOT MET")
            exit_code = 1

        # Add performance metrics
        if self.test_results['discovery_latencies']:
            test_results['performance'] = {
                'discovery_p95_ms': statistics.quantiles(
                    self.test_results['discovery_latencies'], n=20
                )[18] if self.test_results['discovery_latencies'] else None,
                'cache_hit_rate': (
                    self.test_results['cache_hits'] /
                    (self.test_results['cache_hits'] + self.test_results['cache_misses'])
                    if (self.test_results['cache_hits'] + self.test_results['cache_misses']) > 0
                    else 0
                ),
                'conflicts_detected': len(self.test_results['conflicts']),
                'race_conditions': len(self.test_results['race_conditions'])
            }

        return test_results, exit_code


async def main():
    """Main entry point for CI/CD"""
    test_suite = ADR60TestSuite()
    results, exit_code = await test_suite.run_all_tests()

    # Write results to file for CI/CD artifacts
    import json
    with open('adr-60-test-results.json', 'w') as f:
        json.dump(results, f, indent=2)

    sys.exit(exit_code)


if __name__ == '__main__':
    asyncio.run(main())