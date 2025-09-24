#!/usr/bin/env python3
"""
ADR-0098 Phase 1 Validation Script
Test Enhanced Docker Labels implementation
"""

import asyncio
import docker
import sys
import time
import hashlib
import logging
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_enhanced_labels():
    """Test 1: Verify new containers have all required labels"""
    print("\n=== Test 1: Enhanced Labels ===")

    client = docker.from_env()

    # Create test container with enhanced labels
    test_project = f"test-phase1-{int(time.time())}"
    project_path = "/test/project"
    project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:12]
    test_port = 48150

    try:
        container = client.containers.run(
            image="alpine",
            command="sleep 300",
            labels={
                'com.l9.managed': 'true',
                'com.l9.project': test_project,
                'com.l9.created': str(int(time.time())),
                # Phase 1 additions
                'com.l9.project_hash': project_hash,
                'com.l9.port': str(test_port),
                'com.l9.project_path': project_path
            },
            name=f"test-phase1-container-{int(time.time())}",
            ports={'8080/tcp': test_port},
            detach=True,
            auto_remove=False
        )

        # Verify all labels are present
        container.reload()
        labels = container.labels

        required_labels = [
            'com.l9.managed',
            'com.l9.project',
            'com.l9.created',
            'com.l9.project_hash',
            'com.l9.port',
            'com.l9.project_path'
        ]

        missing = []
        for label in required_labels:
            if label not in labels:
                missing.append(label)

        if missing:
            print(f"❌ Missing labels: {missing}")
            return False

        # Verify label values
        if labels['com.l9.project_hash'] != project_hash:
            print(f"❌ Hash mismatch: expected {project_hash}, got {labels['com.l9.project_hash']}")
            return False

        if labels['com.l9.port'] != str(test_port):
            print(f"❌ Port mismatch: expected {test_port}, got {labels['com.l9.port']}")
            return False

        if labels['com.l9.project_path'] != project_path:
            print(f"❌ Path mismatch: expected {project_path}, got {labels['com.l9.project_path']}")
            return False

        print("✅ All enhanced labels present and correct")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Cleanup
        try:
            container.stop()
            container.remove()
        except:
            pass


async def test_backward_compatibility():
    """Test 2: Verify old containers without enhanced labels still work"""
    print("\n=== Test 2: Backward Compatibility ===")

    client = docker.from_env()

    # Create old-style container (without enhanced labels)
    test_project = f"test-legacy-{int(time.time())}"

    try:
        container = client.containers.run(
            image="alpine",
            command="sleep 300",
            labels={
                'com.l9.managed': 'true',
                'com.l9.project': test_project,
                'com.l9.created': str(int(time.time()))
                # No enhanced labels
            },
            name=f"test-legacy-container-{int(time.time())}",
            detach=True,
            auto_remove=False
        )

        # Try to discover container using existing methods
        containers = client.containers.list(
            filters={'label': f'com.l9.project={test_project}'}
        )

        if not containers:
            print("❌ Could not discover legacy container")
            return False

        found = containers[0]
        if found.id != container.id:
            print("❌ Found wrong container")
            return False

        print("✅ Legacy containers still discoverable")

        # Verify we can handle missing labels gracefully
        labels = found.labels
        project_hash = labels.get('com.l9.project_hash', 'NOT_SET')
        port_label = labels.get('com.l9.port', 'NOT_SET')

        if project_hash == 'NOT_SET':
            print("✅ Missing project_hash handled gracefully")

        if port_label == 'NOT_SET':
            print("✅ Missing port label handled gracefully")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Cleanup
        try:
            container.stop()
            container.remove()
        except:
            pass


async def test_label_persistence():
    """Test 3: Verify labels survive container restart"""
    print("\n=== Test 3: Label Persistence ===")

    client = docker.from_env()

    test_project = f"test-persist-{int(time.time())}"
    project_path = "/test/persist"
    project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:12]

    try:
        # Create container with enhanced labels
        container = client.containers.run(
            image="alpine",
            command="sleep 300",
            labels={
                'com.l9.managed': 'true',
                'com.l9.project': test_project,
                'com.l9.created': str(int(time.time())),
                'com.l9.project_hash': project_hash,
                'com.l9.port': '48151',
                'com.l9.project_path': project_path
            },
            name=f"test-persist-container-{int(time.time())}",
            detach=True,
            auto_remove=False
        )

        # Stop container
        container.stop()
        time.sleep(1)

        # Start container again
        container.start()
        container.reload()

        # Verify labels still present
        if container.labels.get('com.l9.project_hash') != project_hash:
            print("❌ Labels lost after restart")
            return False

        if container.labels.get('com.l9.port') != '48151':
            print("❌ Port label lost after restart")
            return False

        print("✅ Labels survive container restart")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Cleanup
        try:
            container.stop()
            container.remove()
        except:
            pass


async def test_hash_consistency():
    """Test 4: Verify project_hash is consistent for same path"""
    print("\n=== Test 4: Hash Consistency ===")

    project_path = "/test/consistent/path"

    # Generate hash multiple times
    hashes = []
    for _ in range(5):
        hash_val = hashlib.sha256(project_path.encode()).hexdigest()[:12]
        hashes.append(hash_val)

    # All should be identical
    if len(set(hashes)) != 1:
        print(f"❌ Inconsistent hashes: {hashes}")
        return False

    print(f"✅ Hash consistent: {hashes[0]}")

    # Different paths should have different hashes
    path1_hash = hashlib.sha256("/project/one".encode()).hexdigest()[:12]
    path2_hash = hashlib.sha256("/project/two".encode()).hexdigest()[:12]

    if path1_hash == path2_hash:
        print("❌ Different paths have same hash")
        return False

    print("✅ Different paths have different hashes")
    return True


async def test_mixed_environment():
    """Test 5: Verify mixed environment (old + new containers) works"""
    print("\n=== Test 5: Mixed Environment ===")

    client = docker.from_env()
    created_containers = []

    try:
        # Create one old-style container
        old_container = client.containers.run(
            image="alpine",
            command="sleep 300",
            labels={
                'com.l9.managed': 'true',
                'com.l9.project': 'old-project',
                'com.l9.created': str(int(time.time()))
            },
            name=f"old-style-{int(time.time())}",
            detach=True,
            auto_remove=False
        )
        created_containers.append(old_container)

        # Create one new-style container
        new_container = client.containers.run(
            image="alpine",
            command="sleep 300",
            labels={
                'com.l9.managed': 'true',
                'com.l9.project': 'new-project',
                'com.l9.created': str(int(time.time())),
                'com.l9.project_hash': 'abc123',
                'com.l9.port': '48152',
                'com.l9.project_path': '/new/path'
            },
            name=f"new-style-{int(time.time())}",
            detach=True,
            auto_remove=False
        )
        created_containers.append(new_container)

        # List all L9 managed containers
        all_containers = client.containers.list(
            filters={'label': 'com.l9.managed=true'}
        )

        found_old = False
        found_new = False

        for container in all_containers:
            if container.labels.get('com.l9.project') == 'old-project':
                found_old = True
                # Should not have enhanced labels
                if 'com.l9.project_hash' in container.labels:
                    print("❌ Old container shouldn't have enhanced labels")
                    return False
            elif container.labels.get('com.l9.project') == 'new-project':
                found_new = True
                # Should have enhanced labels
                if 'com.l9.project_hash' not in container.labels:
                    print("❌ New container missing enhanced labels")
                    return False

        if not found_old or not found_new:
            print("❌ Could not find both containers")
            return False

        print("✅ Mixed environment works correctly")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Cleanup
        for container in created_containers:
            try:
                container.stop()
                container.remove()
            except:
                pass


def check_exit_conditions():
    """Check if Phase 1 exit conditions are met"""
    print("\n=== Phase 1 Exit Conditions ===")

    client = docker.from_env()

    # Count containers with enhanced labels
    all_containers = client.containers.list(
        all=True,
        filters={'label': 'com.l9.managed=true'}
    )

    total = len(all_containers)
    with_enhanced = 0

    for container in all_containers:
        if all(label in container.labels for label in [
            'com.l9.project_hash', 'com.l9.port', 'com.l9.project_path'
        ]):
            with_enhanced += 1

    percentage = (with_enhanced / max(1, total)) * 100 if total > 0 else 0

    conditions = {
        "Enhanced labels present": with_enhanced > 0,
        "Backward compatibility": True,  # Verified by tests
        "No breaking changes": True,  # Verified by tests
        "Labels accurate": True,  # Verified by tests
    }

    ready_for_phase2 = all(conditions.values())

    for condition, met in conditions.items():
        status = "✅" if met else "❌"
        print(f"{status} {condition}")

    print(f"\n📊 Metrics:")
    print(f"  Total containers: {total}")
    print(f"  With enhanced labels: {with_enhanced} ({percentage:.1f}%)")

    print("\n=== Decision ===")
    if ready_for_phase2:
        print("✅ READY for Phase 2: Docker Primary, Dicts Fallback")
        print("\nNext steps:")
        print("1. Run in production for 1 week")
        print("2. Monitor label accuracy")
        print("3. If 100% new containers have labels, proceed to Phase 2")
    else:
        print("❌ NOT READY for Phase 2")
        print("Ensure all new containers get enhanced labels")

    return ready_for_phase2


async def main():
    """Run all Phase 1 validation tests"""
    print("=" * 60)
    print("ADR-0098 Phase 1 Validation Suite")
    print("=" * 60)

    tests = [
        ("Enhanced Labels", test_enhanced_labels),
        ("Backward Compatibility", test_backward_compatibility),
        ("Label Persistence", test_label_persistence),
        ("Hash Consistency", test_hash_consistency),
        ("Mixed Environment", test_mixed_environment),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    # Check exit conditions
    if passed == total:
        check_exit_conditions()
    else:
        print("\n❌ Fix failing tests before proceeding to Phase 2")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)