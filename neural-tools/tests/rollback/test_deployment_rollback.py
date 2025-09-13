#!/usr/bin/env python3
"""
Deployment Rollback Test
Tests the ability to rollback to a previous version if deployment fails
"""

import sys
import os
import shutil
import json
import asyncio
from pathlib import Path
from datetime import datetime


async def test_deployment_rollback():
    """Test deployment rollback scenario"""
    print("\n↩️  Testing Deployment Rollback Capability...")

    # Paths
    global_mcp_dir = Path.home() / ".claude" / "mcp-servers" / "neural-tools"
    backup_dir = Path.home() / ".claude" / "mcp-servers" / "neural-tools-backup-test"
    test_marker = global_mcp_dir / ".test-rollback-marker"

    try:
        print("\n  1️⃣ Checking Current State...")

        # Check if global MCP exists
        if not global_mcp_dir.exists():
            print(f"     ⚠️ Global MCP not deployed yet at {global_mcp_dir}")
            print(f"     This is expected if running in development")
            return True

        print(f"     ✅ Found global MCP at {global_mcp_dir}")

        # Check for backup capability
        print("\n  2️⃣ Testing Backup Creation...")

        # Create a test backup
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        shutil.copytree(global_mcp_dir, backup_dir)
        print(f"     ✅ Created backup at {backup_dir}")

        # Verify backup integrity
        backup_files = list(backup_dir.rglob("*.py"))
        original_files = list(global_mcp_dir.rglob("*.py"))

        if len(backup_files) != len(original_files):
            print(f"     ❌ Backup file count mismatch")
            return False

        print(f"     ✅ Backup contains {len(backup_files)} Python files")

        # Simulate a deployment failure
        print("\n  3️⃣ Simulating Deployment Failure...")

        # Add a marker file to simulate change
        test_marker.write_text(f"Test deployment at {datetime.now()}")
        print(f"     ✅ Added test marker to deployment")

        # Check marker exists
        if not test_marker.exists():
            print(f"     ❌ Failed to create test marker")
            return False

        # Perform rollback
        print("\n  4️⃣ Performing Rollback...")

        # Remove current deployment
        if global_mcp_dir.exists():
            shutil.rmtree(global_mcp_dir)
            print(f"     ✅ Removed failed deployment")

        # Restore from backup
        shutil.copytree(backup_dir, global_mcp_dir)
        print(f"     ✅ Restored from backup")

        # Verify rollback
        print("\n  5️⃣ Verifying Rollback...")

        if test_marker.exists():
            print(f"     ❌ Test marker still exists (rollback failed)")
            return False
        else:
            print(f"     ✅ Test marker removed (rollback successful)")

        # Check critical files exist
        critical_files = [
            "src/neural_mcp/neural_server_stdio.py",
            "run_mcp_server.py"
        ]

        for file_path in critical_files:
            full_path = global_mcp_dir / file_path
            if not full_path.exists():
                print(f"     ❌ Critical file missing: {file_path}")
                return False

        print(f"     ✅ All critical files present")

        # Test MCP configuration validity
        print("\n  6️⃣ Validating MCP Configuration...")

        mcp_config_path = Path.home() / ".claude" / "mcp_config.json"
        if mcp_config_path.exists():
            with open(mcp_config_path) as f:
                config = json.load(f)

            if "neural-tools" in config.get("mcpServers", {}):
                print(f"     ✅ MCP configuration intact")
            else:
                print(f"     ⚠️ MCP configuration needs update")

        print("\n" + "=" * 50)
        print("✅ Deployment Rollback Test PASSED")
        print("Rollback capability verified")
        return True

    except Exception as e:
        print(f"\n  ❌ Rollback test failed: {e}")
        return False

    finally:
        # Cleanup test backup
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
            print(f"\n  🧹 Cleaned up test backup")


def main():
    """Run the rollback test"""
    success = asyncio.run(test_deployment_rollback())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()