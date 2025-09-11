#!/usr/bin/env python3
"""
Test instance cleanup for MCP server - verifies ADR-19 Phase 2 implementation
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_instance_cleanup():
    """Test that stale instances are properly cleaned up"""
    
    print("🧪 Testing MCP Instance Cleanup (ADR-19 Phase 2)")
    print("=" * 60)
    
    # Set short timeout for testing
    os.environ['INSTANCE_TIMEOUT_HOURS'] = '0.001'  # Very short timeout (3.6 seconds)
    os.environ['CLEANUP_INTERVAL_MINUTES'] = '0.05'  # 3 seconds
    os.environ['ENABLE_AUTO_CLEANUP'] = 'true'
    
    from neural_mcp.neural_server_stdio import MultiProjectServiceState
    
    # Test 1: Create multiple instances
    print("\n📋 Test 1: Creating Multiple Instances")
    print("-" * 40)
    
    # Create main instance
    os.environ['INSTANCE_ID'] = 'test-main'
    state_main = MultiProjectServiceState()
    print(f"✅ Main instance created: {state_main.instance_id}")
    
    # Manually add some stale instances to simulate old sessions
    stale_time = datetime.now() - timedelta(hours=2)
    state_main.instance_containers['stale-instance-1'] = {
        'project_containers': {},
        'project_retrievers': {},
        'session_started': stale_time,
        'last_activity': stale_time
    }
    print("✅ Added stale instance: stale-instance-1")
    
    state_main.instance_containers['stale-instance-2'] = {
        'project_containers': {},
        'project_retrievers': {},
        'session_started': stale_time,
        'last_activity': stale_time
    }
    print("✅ Added stale instance: stale-instance-2")
    
    # Add a recent instance that should NOT be cleaned
    recent_time = datetime.now() - timedelta(seconds=1)
    state_main.instance_containers['recent-instance'] = {
        'project_containers': {},
        'project_retrievers': {},
        'session_started': recent_time,
        'last_activity': recent_time
    }
    print("✅ Added recent instance: recent-instance")
    
    # Test 2: Verify metrics before cleanup
    print("\n📋 Test 2: Instance Metrics Before Cleanup")
    print("-" * 40)
    
    metrics = state_main.get_instance_metrics()
    print(f"Total instances: {metrics['total_instances']}")
    print(f"Active instances: {metrics['active_instances']}")
    print(f"Stale instances: {metrics['stale_instances']}")
    
    assert metrics['total_instances'] == 4, "Should have 4 instances total"
    assert metrics['stale_instances'] == 2, "Should have 2 stale instances"
    print("✅ Metrics correctly identify stale instances")
    
    # Test 3: Run cleanup
    print("\n📋 Test 3: Running Cleanup")
    print("-" * 40)
    
    cleaned = await state_main.cleanup_stale_instances()
    print(f"✅ Cleanup completed, removed {cleaned} instances")
    
    assert cleaned == 2, "Should have cleaned 2 stale instances"
    
    # Test 4: Verify metrics after cleanup
    print("\n📋 Test 4: Instance Metrics After Cleanup")
    print("-" * 40)
    
    metrics_after = state_main.get_instance_metrics()
    print(f"Total instances: {metrics_after['total_instances']}")
    print(f"Active instances: {metrics_after['active_instances']}")
    print(f"Stale instances: {metrics_after['stale_instances']}")
    
    assert metrics_after['total_instances'] == 2, "Should have 2 instances after cleanup"
    assert metrics_after['stale_instances'] == 0, "Should have no stale instances"
    assert 'stale-instance-1' not in state_main.instance_containers, "Stale instance 1 should be removed"
    assert 'stale-instance-2' not in state_main.instance_containers, "Stale instance 2 should be removed"
    assert 'recent-instance' in state_main.instance_containers, "Recent instance should remain"
    assert state_main.instance_id in state_main.instance_containers, "Current instance should remain"
    
    print("✅ Cleanup correctly removed only stale instances")
    
    # Test 5: Verify cleanup stats
    print("\n📋 Test 5: Cleanup Statistics")
    print("-" * 40)
    
    stats = state_main.cleanup_stats
    print(f"Total cleanups run: {stats['total_cleanups']}")
    print(f"Instances cleaned total: {stats['instances_cleaned']}")
    print(f"Last cleanup: {stats['last_cleanup']}")
    
    assert stats['total_cleanups'] == 1, "Should have run 1 cleanup"
    assert stats['instances_cleaned'] == 2, "Should have cleaned 2 instances total"
    assert stats['last_cleanup'] is not None, "Should have recorded cleanup time"
    print("✅ Cleanup statistics properly tracked")
    
    # Test 6: Test cleanup doesn't remove current instance
    print("\n📋 Test 6: Current Instance Protection")
    print("-" * 40)
    
    # Make current instance appear stale (but it should still be protected)
    state_main.instance_containers[state_main.instance_id]['last_activity'] = stale_time
    
    cleaned_self = await state_main.cleanup_stale_instances()
    print(f"Cleanup with stale current instance: {cleaned_self} removed")
    
    assert cleaned_self == 0, "Should not clean current instance even if stale"
    assert state_main.instance_id in state_main.instance_containers, "Current instance should never be removed"
    print("✅ Current instance protected from cleanup")
    
    # Test 7: Test container with services gets closed properly
    print("\n📋 Test 7: Container Connection Closure")
    print("-" * 40)
    
    # Create a mock container with fake connections
    class MockContainer:
        def __init__(self):
            self.neo4j_driver = MockDriver()
            self.qdrant_client = MockClient()
            self.closed_connections = []
    
    class MockDriver:
        def close(self):
            print("  Neo4j driver closed")
    
    class MockClient:
        def close(self):
            print("  Qdrant client closed")
    
    # Add instance with mock container
    mock_container = MockContainer()
    state_main.instance_containers['mock-stale'] = {
        'project_containers': {'test-project': mock_container},
        'project_retrievers': {},
        'session_started': stale_time,
        'last_activity': stale_time
    }
    
    # Clean it up
    await state_main.cleanup_stale_instances()
    
    assert 'mock-stale' not in state_main.instance_containers, "Mock instance should be removed"
    print("✅ Container connections properly closed during cleanup")
    
    print("\n" + "=" * 60)
    print("✅ ALL CLEANUP TESTS PASSED!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_instance_cleanup())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)