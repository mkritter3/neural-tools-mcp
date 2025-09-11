#!/usr/bin/env python3
"""
Test Phase 3 features for MCP server - ADR-19 enhanced monitoring and debugging
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_phase3_features():
    """Test Phase 3 enhancements: verbose mode, metadata, migration"""
    
    print("🧪 Testing MCP Phase 3 Features (ADR-19)")
    print("=" * 60)
    
    # Test 1: Verbose mode configuration
    print("\n📋 Test 1: Verbose Mode Configuration")
    print("-" * 40)
    
    # Set verbose mode
    os.environ['MCP_VERBOSE'] = 'true'
    os.environ['INCLUDE_INSTANCE_METADATA'] = 'true'
    os.environ['INSTANCE_ID'] = 'test-phase3'
    
    from neural_mcp.neural_server_stdio import (
        MCP_VERBOSE, 
        INCLUDE_INSTANCE_METADATA,
        add_instance_metadata,
        MultiProjectServiceState,
        InstanceAwareLogger
    )
    
    assert MCP_VERBOSE == True, "Verbose mode should be enabled"
    assert INCLUDE_INSTANCE_METADATA == True, "Metadata inclusion should be enabled"
    print("✅ Verbose mode configuration loaded correctly")
    
    # Test 2: Instance metadata addition
    print("\n📋 Test 2: Instance Metadata Addition")
    print("-" * 40)
    
    test_response = {
        "status": "success",
        "data": "test"
    }
    
    enhanced_response = add_instance_metadata(test_response, instance_id="test-123", project_name="test-project")
    
    assert '_metadata' in enhanced_response, "Metadata should be added"
    assert enhanced_response['_metadata']['instance_id'] == 'test-123', "Instance ID should be included"
    assert enhanced_response['_metadata']['project'] == 'test-project', "Project should be included"
    assert 'timestamp' in enhanced_response['_metadata'], "Timestamp should be included"
    print(f"✅ Metadata added: {json.dumps(enhanced_response['_metadata'], indent=2)}")
    
    # Test 3: Instance-aware logging
    print("\n📋 Test 3: Instance-Aware Logging")
    print("-" * 40)
    
    import logging
    import io
    
    # Create a string buffer to capture log output
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.INFO)
    
    # Create test logger
    test_logger = logging.getLogger('test')
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)
    
    # Create instance-aware logger
    aware_logger = InstanceAwareLogger(test_logger, lambda: "test-instance-456")
    
    # Log a message
    aware_logger.info("Test message")
    
    # Check output
    log_output = log_buffer.getvalue()
    assert "[Instance test-instance-456]" in log_output, "Instance ID should be in log message"
    print(f"✅ Instance-aware logging working: {log_output.strip()}")
    
    # Test 4: State export/import
    print("\n📋 Test 4: Instance State Export/Import")
    print("-" * 40)
    
    # Create state and add some data
    state1 = MultiProjectServiceState()
    
    # Export state
    export_data = await state1.export_instance_state()
    
    assert 'instance_id' in export_data, "Export should include instance ID"
    assert 'configuration' in export_data, "Export should include configuration"
    assert 'export_timestamp' in export_data, "Export should include timestamp"
    print(f"✅ Exported state for instance: {export_data['instance_id']}")
    
    # Create new state and import
    os.environ['INSTANCE_ID'] = 'test-phase3-new'
    state2 = MultiProjectServiceState()
    
    # Import the exported data
    success = await state2.import_instance_state(export_data, merge=True)
    assert success == True, "Import should succeed"
    
    # Verify imported data exists
    assert state1.instance_id in state2.instance_containers, "Imported instance should exist"
    print(f"✅ Successfully imported state from {state1.instance_id} to {state2.instance_id}")
    
    # Test 5: Enhanced metrics with configuration
    print("\n📋 Test 5: Enhanced Metrics")
    print("-" * 40)
    
    metrics = state2.get_instance_metrics()
    
    assert 'current_instance_id' in metrics, "Should include current instance"
    assert 'total_instances' in metrics, "Should include total instances"
    assert metrics['total_instances'] >= 2, "Should have at least 2 instances after import"
    assert 'cleanup_stats' in metrics, "Should include cleanup stats"
    
    print(f"Current instance: {metrics['current_instance_id']}")
    print(f"Total instances: {metrics['total_instances']}")
    print(f"Active instances: {metrics['active_instances']}")
    print("✅ Enhanced metrics working correctly")
    
    # Test 6: Verbose mode toggle
    print("\n📋 Test 6: Verbose Mode Toggle")
    print("-" * 40)
    
    # Test with verbose off
    os.environ['MCP_VERBOSE'] = 'false'
    os.environ['INCLUDE_INSTANCE_METADATA'] = 'false'
    
    # Reimport to get new values
    import importlib
    import neural_mcp.neural_server_stdio as server_module
    importlib.reload(server_module)
    
    test_response_no_verbose = {"status": "success"}
    enhanced_no_verbose = server_module.add_instance_metadata(test_response_no_verbose)
    
    assert '_metadata' not in enhanced_no_verbose, "Metadata should not be added when verbose is off"
    print("✅ Verbose mode can be toggled off correctly")
    
    print("\n" + "=" * 60)
    print("✅ ALL PHASE 3 TESTS PASSED!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_phase3_features())
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)