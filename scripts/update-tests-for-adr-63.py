#!/usr/bin/env python3
"""
Update all existing tests to properly validate mount paths per ADR-63

This script updates our test suite to:
1. Use different paths instead of static /tmp/test
2. Verify mounts are correct
3. Test real user scenarios (project switching, branch changes, etc.)
"""

import os
import re
import sys


def update_test_file(filepath):
    """Update a test file to use varying paths"""

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    changes_made = []

    # Pattern 1: Replace static /tmp/test paths with dynamic paths
    static_path_pattern = r"['\"]\/tmp\/test['\"]"
    if re.search(static_path_pattern, content):
        # Add import for tempfile if not present
        if 'import tempfile' not in content:
            content = content.replace('import asyncio', 'import asyncio\nimport tempfile')
            changes_made.append("Added tempfile import")

        # Replace static paths with dynamic ones
        content = re.sub(
            r"ensure_indexer\((.*?), ['\"]\/tmp\/test['\"]",
            r"ensure_indexer(\1, tempfile.mkdtemp(prefix='test-')",
            content
        )
        changes_made.append("Replaced static /tmp/test with dynamic paths")

    # Pattern 2: Add mount verification after container creation
    ensure_indexer_pattern = r"(container\w*|result) = await .*ensure_indexer\((.*?)\)"
    matches = list(re.finditer(ensure_indexer_pattern, content))

    for match in reversed(matches):  # Reverse to maintain positions
        var_name = match.group(1)
        # Check if mount verification already exists
        check_pos = match.end()
        next_100_chars = content[check_pos:check_pos+200]

        if 'assert' not in next_100_chars and 'Mounts' not in next_100_chars:
            # Add mount verification
            indent = len(match.group(0)) - len(match.group(0).lstrip())
            indent_str = ' ' * indent

            verification = f"""
{indent_str}# ADR-63: Verify mount is correct
{indent_str}container_obj = self.docker_client.containers.get({var_name})
{indent_str}mounts = container_obj.attrs.get('Mounts', [])
{indent_str}mount_source = next((m['Source'] for m in mounts if m['Destination'] == '/workspace'), None)
{indent_str}assert mount_source is not None, "No workspace mount found!"
"""
            # Insert after the ensure_indexer line
            content = content[:check_pos] + verification + content[check_pos:]
            changes_made.append(f"Added mount verification for {var_name}")

    # Pattern 3: Fix test scenarios that reuse same path
    if "ensure_indexer(project_name, '/tmp/test')" in content:
        # Create unique paths for each call
        counter = 0
        def replace_path(match):
            nonlocal counter
            counter += 1
            return f"ensure_indexer(project_name, tempfile.mkdtemp(prefix='test-{counter}-'))"

        content = re.sub(
            r"ensure_indexer\(project_name, '/tmp/test'\)",
            replace_path,
            content
        )
        changes_made.append("Made each test use unique paths")

    # Only write if changes were made
    if content != original_content:
        print(f"\n📝 Updating {filepath}")
        for change in changes_made:
            print(f"   ✓ {change}")

        # Backup original
        backup_path = filepath + '.pre-adr63'
        with open(backup_path, 'w') as f:
            f.write(original_content)
        print(f"   📦 Backup saved to {backup_path}")

        # Write updated content
        with open(filepath, 'w') as f:
            f.write(content)

        return True
    else:
        print(f"\n✅ {filepath} - Already compliant")
        return False


def add_mount_validation_test():
    """Add a specific test for mount validation regression"""

    test_content = '''#!/usr/bin/env python3
"""
ADR-63 Mount Validation Regression Test
CRITICAL: This test MUST be in CI/CD to prevent mount validation regression
"""

import asyncio
import tempfile
import docker
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'neural-tools', 'src'))
from servers.services.indexer_orchestrator import IndexerOrchestrator


async def test_mount_validation_regression():
    """
    Test that container mount paths are validated before reuse.
    This test would have caught the ADR-60 regression where containers
    were reused with wrong mount paths.
    """
    docker_client = docker.from_env()
    orchestrator = IndexerOrchestrator()
    orchestrator.docker_client = docker_client

    # Create two different paths
    path1 = tempfile.mkdtemp(prefix='mount-test-1-')
    path2 = tempfile.mkdtemp(prefix='mount-test-2-')

    print(f"Testing mount validation with paths:")
    print(f"  Path 1: {path1}")
    print(f"  Path 2: {path2}")

    try:
        # Create container with first path
        container1 = await orchestrator.ensure_indexer('mount-test', path1)

        # Verify mount
        c1_obj = docker_client.containers.get(container1)
        mounts1 = c1_obj.attrs['Mounts']
        mount1 = next((m['Source'] for m in mounts1 if m['Destination'] == '/workspace'), None)
        assert mount1 == path1, f"Container 1 mount wrong: {mount1} != {path1}"
        print(f"  ✅ Container 1 mount correct: {mount1}")

        # Request same project with DIFFERENT path
        container2 = await orchestrator.ensure_indexer('mount-test', path2)

        # CRITICAL ASSERTION: Must be different container
        assert container1 != container2, "REGRESSION: Container reused with wrong mount!"
        print(f"  ✅ Container recreated (not reused)")

        # Verify second container has correct mount
        c2_obj = docker_client.containers.get(container2)
        mounts2 = c2_obj.attrs['Mounts']
        mount2 = next((m['Source'] for m in mounts2 if m['Destination'] == '/workspace'), None)
        assert mount2 == path2, f"Container 2 mount wrong: {mount2} != {path2}"
        print(f"  ✅ Container 2 mount correct: {mount2}")

        # Verify first container was removed
        try:
            docker_client.containers.get(container1)
            assert False, "Old container should have been removed!"
        except docker.errors.NotFound:
            print(f"  ✅ Old container properly removed")

        print("\\n✅ Mount validation test PASSED - Regression prevented!")
        return True

    except AssertionError as e:
        print(f"\\n❌ Mount validation test FAILED: {e}")
        print("ADR-63 regression detected!")
        return False
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(path1, ignore_errors=True)
        shutil.rmtree(path2, ignore_errors=True)

        # Remove test containers
        for container in docker_client.containers.list(all=True):
            if 'mount-test' in container.name:
                try:
                    container.remove(force=True)
                except:
                    pass


if __name__ == "__main__":
    result = asyncio.run(test_mount_validation_regression())
    sys.exit(0 if result else 1)
'''

    # Write the critical regression test
    regression_test_path = 'tests/test_critical_mount_validation.py'
    with open(regression_test_path, 'w') as f:
        f.write(test_content)

    os.chmod(regression_test_path, 0o755)
    print(f"\n🆕 Created critical regression test: {regression_test_path}")
    print("   This test MUST be in CI/CD pipeline!")


def main():
    print("="*60)
    print("ADR-63 TEST SUITE UPDATE")
    print("Fixing tests to catch mount validation regressions")
    print("="*60)

    # Find all test files that need updating
    test_files = [
        'scripts/test-adr-60-e2e.py',
        'scripts/test-indexer-fix-adr0058.py',
        'tests/test_full_integration.py',
        'tests/test_mcp_neural_tools.py'
    ]

    updated_count = 0

    for test_file in test_files:
        if os.path.exists(test_file):
            if update_test_file(test_file):
                updated_count += 1
        else:
            print(f"\n⚠️  {test_file} not found")

    # Add the critical mount validation test
    add_mount_validation_test()

    print("\n" + "="*60)
    print(f"✅ Updated {updated_count} test files")
    print("🎯 Added critical mount validation regression test")
    print("\nNext steps:")
    print("1. Run the updated tests to verify they catch regressions")
    print("2. Add test_critical_mount_validation.py to CI/CD pipeline")
    print("3. Never use static paths like /tmp/test in tests again!")
    print("="*60)


if __name__ == "__main__":
    main()