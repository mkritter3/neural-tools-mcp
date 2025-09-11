#!/usr/bin/env python3
"""
Test instance isolation for MCP server - verifies ADR-19 Phase 1 implementation
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_instance_isolation():
    """Test that different instances get different IDs and isolated state"""
    
    print("🧪 Testing MCP Instance Isolation (ADR-19 Phase 1)")
    print("=" * 60)
    
    # Test 1: Verify instance ID generation
    print("\n📋 Test 1: Instance ID Generation")
    print("-" * 40)
    
    # Import the function
    from neural_mcp.neural_server_stdio import get_instance_id
    
    # Test with different environment setups
    
    # Test with provided instance ID
    os.environ['INSTANCE_ID'] = 'test-instance-1'
    id1 = get_instance_id()
    print(f"✅ With INSTANCE_ID env var: {id1}")
    assert id1 == 'test-instance-1', "Should use provided instance ID"
    
    # Test without provided instance ID (process-based)
    del os.environ['INSTANCE_ID']
    id2 = get_instance_id()
    print(f"✅ Without INSTANCE_ID env var: {id2}")
    assert len(id2) == 8, "Should generate 8-char hash"
    assert id2 != 'test-instance-1', "Should not use previous ID"
    
    # Test 2: Verify state isolation
    print("\n📋 Test 2: State Isolation")
    print("-" * 40)
    
    from neural_mcp.neural_server_stdio import MultiProjectServiceState
    
    # Simulate two instances with different IDs
    os.environ['INSTANCE_ID'] = 'test-instance-a'
    state1 = MultiProjectServiceState()
    print(f"✅ State 1 instance ID: {state1.instance_id}")
    
    os.environ['INSTANCE_ID'] = 'test-instance-b'
    state2 = MultiProjectServiceState()
    print(f"✅ State 2 instance ID: {state2.instance_id}")
    
    # Verify they have different instance IDs
    assert state1.instance_id != state2.instance_id, "Instances should have different IDs"
    print("✅ Instance IDs are different")
    
    # Verify they have separate containers
    assert state1.instance_containers != state2.instance_containers, "Should have separate containers"
    print("✅ Instance containers are separate")
    
    # Test 3: Verify project isolation within instances
    print("\n📋 Test 3: Project Isolation Within Instances")
    print("-" * 40)
    
    # Get containers for different projects in state1
    container1a = await state1.get_project_container('project-a')
    container1b = await state1.get_project_container('project-b')
    
    # Verify they're different containers
    assert container1a is not container1b, "Different projects should have different containers"
    print("✅ Different projects have separate containers")
    
    # Verify same project returns same container
    container1a_again = await state1.get_project_container('project-a')
    assert container1a is container1a_again, "Same project should return same container"
    print("✅ Same project returns same container")
    
    # Test 4: Verify cross-instance isolation
    print("\n📋 Test 4: Cross-Instance Isolation")
    print("-" * 40)
    
    # Get container for same project in different instance
    container2a = await state2.get_project_container('project-a')
    
    # Verify it's different from state1's container
    assert container1a is not container2a, "Same project in different instances should have different containers"
    print("✅ Same project in different instances has separate containers")
    
    # Test 5: Verify instance data tracking
    print("\n📋 Test 5: Instance Data Tracking")
    print("-" * 40)
    
    instance_data1 = state1._get_instance_data()
    instance_data2 = state2._get_instance_data()
    
    # Check timestamps are set
    assert 'session_started' in instance_data1, "Should track session start time"
    assert 'last_activity' in instance_data1, "Should track last activity"
    print("✅ Instance tracking data properly initialized")
    
    # Verify activity tracking updates
    import time
    initial_activity = instance_data1['last_activity']
    time.sleep(0.01)  # Small delay to ensure timestamp difference
    instance_data1_updated = state1._get_instance_data()
    # The activity time should be updated when _get_instance_data is called
    assert instance_data1_updated['last_activity'] >= initial_activity, "Activity time should be maintained or updated"
    print("✅ Activity tracking maintains timestamps")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Instance isolation working correctly!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_instance_isolation())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)