#!/usr/bin/env python3
"""
Test Multi-hop Dependency Analysis - ADR-0075 Phase 3
Test the new dependency analysis MCP tool
"""

import asyncio
import sys
import json
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from neural_mcp.neural_server_stdio import dependency_analysis_impl
from mcp import types

async def test_dependency_analysis():
    """Test the new dependency analysis tool"""
    print("🧪 Testing Multi-hop Dependency Analysis")
    print("="*50)

    # Test with a file that should have dependencies
    test_file = "neural-tools/src/servers/services/service_container.py"

    print(f"🔍 Analyzing dependencies for: {test_file}")
    print()

    # Test all analysis types
    analysis_types = ['imports', 'dependents', 'calls', 'all']

    for analysis_type in analysis_types:
        print(f"📊 Testing {analysis_type} analysis...")

        try:
            results = await dependency_analysis_impl(
                target_file=test_file,
                analysis_type=analysis_type,
                max_depth=2
            )

            if results and len(results) > 0:
                response_data = json.loads(results[0].text)

                print(f"   Status: {response_data.get('status')}")

                if response_data.get('status') == 'success':
                    summary = response_data.get('summary', {})
                    print(f"   Imports: {summary.get('total_imports', 0)}")
                    print(f"   Dependents: {summary.get('total_dependents', 0)}")
                    print(f"   Calls: {summary.get('total_calls', 0)}")
                    print(f"   Max Depths: I:{summary.get('max_import_depth', 0)} D:{summary.get('max_dependent_depth', 0)} C:{summary.get('max_call_depth', 0)}")
                elif response_data.get('status') == 'no_data':
                    print(f"   No dependency data found")
                else:
                    print(f"   Error: {response_data.get('message', 'Unknown error')}")
            else:
                print("   No results returned")

        except Exception as e:
            print(f"   ❌ Test failed: {e}")

        print()

    print("✅ Multi-hop dependency analysis test completed!")

if __name__ == "__main__":
    asyncio.run(test_dependency_analysis())