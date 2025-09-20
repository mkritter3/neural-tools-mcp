#!/usr/bin/env python3
"""
Strict Import Validation Test
Ensures imports work in production MCP environment
This test mimics the EXACT sys.path setup that MCP uses
"""

import sys
import os
from pathlib import Path

def test_production_imports():
    """Test imports with production-like sys.path setup"""

    print("="*60)
    print("STRICT IMPORT VALIDATION TEST")
    print("="*60)

    # Clear sys.path to start fresh
    original_path = sys.path.copy()
    sys.path.clear()

    # Add ONLY what MCP server adds (mimics run_mcp_server.py)
    neural_tools_dir = Path(__file__).parent.parent / "neural-tools"
    sys.path.append(str(neural_tools_dir / "src"))

    # Add standard Python paths back
    sys.path.extend([p for p in original_path if 'python' in p.lower()])

    print(f"\n1. Production sys.path setup:")
    print(f"   Added: {neural_tools_dir / 'src'}")
    print(f"   NOT added: {neural_tools_dir}")

    success = True
    errors = []

    print("\n2. Testing critical imports:")

    # Test 1: sync_manager import
    try:
        from servers.services.sync_manager import WriteSynchronizationManager
        print("   ✅ sync_manager imports correctly")
    except ImportError as e:
        print(f"   ❌ sync_manager import failed: {e}")
        errors.append(f"sync_manager: {e}")
        success = False

    # Test 2: event_store import (from within sync_manager's perspective)
    try:
        from servers.services.event_store import SyncEventStore, create_event_store
        print("   ✅ event_store imports correctly")
    except ImportError as e:
        print(f"   ❌ event_store import failed: {e}")
        errors.append(f"event_store: {e}")
        success = False

    # Test 3: indexer_service import
    try:
        from servers.services.indexer_service import IncrementalIndexer
        print("   ✅ indexer_service imports correctly")
    except ImportError as e:
        print(f"   ❌ indexer_service import failed: {e}")
        errors.append(f"indexer_service: {e}")
        success = False

    # Test 4: service_container import
    try:
        from servers.services.service_container import ServiceContainer
        print("   ✅ service_container imports correctly")
    except ImportError as e:
        print(f"   ❌ service_container import failed: {e}")
        errors.append(f"service_container: {e}")
        success = False

    # Test 5: Verify no plain imports work (these should fail!)
    print("\n3. Testing that wrong imports FAIL (as they should):")
    try:
        # This SHOULD fail in production
        exec("from event_store import SyncEventStore")
        print("   ❌ PROBLEM: Plain 'from event_store' works (it shouldn't!)")
        errors.append("Plain imports are working - test env too permissive!")
        success = False
    except ImportError:
        print("   ✅ Plain 'from event_store' correctly fails")

    # Restore original sys.path
    sys.path = original_path

    print("\n" + "="*60)
    if success:
        print("✅ ALL IMPORTS ARE PRODUCTION-READY")
        print("Imports will work correctly in MCP server environment")
    else:
        print("❌ IMPORT VALIDATION FAILED")
        print("These imports will break in production MCP:")
        for error in errors:
            print(f"  - {error}")
        print("\nFIX: Use 'from servers.services.module' not 'from module'")
    print("="*60)

    return success

if __name__ == "__main__":
    success = test_production_imports()
    sys.exit(0 if success else 1)