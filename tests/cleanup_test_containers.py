#!/usr/bin/env python3
"""
Utility to clean up test containers
Run this if tests fail to clean up properly
"""

import docker
import sys

def cleanup_test_containers():
    """Remove all test-related containers"""
    docker_client = docker.from_env()

    # Patterns that indicate test containers
    test_patterns = [
        'test-mount-change',
        'test-stale-mount',
        'test-env-change',
        'mount-test',
        'test-env',
        'test-concurrent',
        'test-conflict',
        'test-gc',
        'test-perf',
        'test-redis',
        'adr63-',
        'adr60-'
    ]

    print("🧹 Cleaning up test containers...")

    containers = docker_client.containers.list(all=True)
    removed_count = 0
    failed_count = 0

    for container in containers:
        project = container.labels.get('com.l9.project', '')
        name = container.name

        # Check if it matches test patterns
        is_test = any(pattern in project or pattern in name for pattern in test_patterns)

        if is_test:
            try:
                print(f"  Removing: {container.name} (project: {project})")
                container.remove(force=True)
                removed_count += 1
            except Exception as e:
                print(f"  ❌ Failed to remove {container.name}: {e}")
                failed_count += 1

    print(f"\n✅ Removed {removed_count} test containers")
    if failed_count > 0:
        print(f"❌ Failed to remove {failed_count} containers")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(cleanup_test_containers())