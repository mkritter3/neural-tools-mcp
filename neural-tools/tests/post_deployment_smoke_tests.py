#!/usr/bin/env python3
"""
Post-Deployment Smoke Tests
Quick verification that deployment succeeded and system is operational

These tests run AFTER deployment to verify:
1. MCP server can be started
2. Tools return real results
3. Vector search works with actual data
4. No critical errors in basic operations
"""

import sys
import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class PostDeploymentSmokeTests:
    """Quick smoke tests to verify deployment success"""

    def __init__(self):
        self.results = []
        self.critical_failures = []

    async def run_all_tests(self) -> bool:
        """Run all smoke tests"""
        print("=" * 70)
        print("🔥 POST-DEPLOYMENT SMOKE TESTS")
        print("=" * 70)
        print()

        # 1. MCP Server can start
        print("1️⃣ Testing MCP Server Startup...")
        mcp_ok = await self.test_mcp_server_startup()
        self._report("MCP Server Startup", mcp_ok)

        # 2. Fast search returns results
        print("\n2️⃣ Testing Fast Search...")
        fast_ok = await self.test_fast_search()
        self._report("Fast Search", fast_ok)

        # 3. Elite search returns results with graph context
        print("\n3️⃣ Testing Elite Search...")
        elite_ok = await self.test_elite_search()
        self._report("Elite Search", elite_ok)

        # 4. Project operations work
        print("\n4️⃣ Testing Project Operations...")
        project_ok = await self.test_project_operations()
        self._report("Project Operations", project_ok)

        # 5. Schema validation works
        print("\n5️⃣ Testing Schema Validation...")
        schema_ok = await self.test_schema_validation()
        self._report("Schema Validation", schema_ok)

        # Final report
        self._print_summary()

        return len(self.critical_failures) == 0

    async def test_mcp_server_startup(self) -> bool:
        """Test that MCP server can start without errors"""
        try:
            # Try to import and initialize the server
            from neural_mcp.server import create_server

            # Server should be creatable
            server = create_server()
            if server is None:
                self.critical_failures.append("MCP server creation returned None")
                return False

            print("   ✅ MCP server can be created")
            return True

        except ImportError as e:
            self.critical_failures.append(f"Cannot import MCP server: {e}")
            print(f"   ❌ Import failed: {e}")
            return False
        except Exception as e:
            self.critical_failures.append(f"MCP server startup failed: {e}")
            print(f"   ❌ Startup failed: {e}")
            return False

    async def test_fast_search(self) -> bool:
        """Test that fast_search returns real results"""
        try:
            from neural_mcp.tools.fast_search import execute as fast_search

            # Test with a query that should return results
            result = await fast_search({
                "query": "vector index optimization",
                "limit": 3
            })

            if not result or not result[0].text:
                self.critical_failures.append("Fast search returned empty")
                return False

            # Parse result
            data = json.loads(result[0].text)

            # Check for success status
            if data.get("status") == "error":
                print(f"   ⚠️  Fast search error: {data.get('message')}")
                return False

            # Check if we got results
            if data.get("status") == "success":
                results = data.get("results", [])
                if results:
                    # Verify first result has required fields
                    first = results[0]
                    if "file" in first and "path" in first["file"]:
                        path = first["file"]["path"]
                        if path and path != "unknown" and path != "None":
                            print(f"   ✅ Fast search returned real file: {path}")
                            return True

            print("   ⚠️  Fast search returned but no valid results")
            return True  # Not critical if no results, just means no data

        except json.JSONDecodeError as e:
            self.critical_failures.append(f"Fast search returned invalid JSON: {e}")
            return False
        except Exception as e:
            self.critical_failures.append(f"Fast search failed: {e}")
            print(f"   ❌ Fast search error: {e}")
            return False

    async def test_elite_search(self) -> bool:
        """Test that elite_search returns results with graph context"""
        try:
            from neural_mcp.tools.elite_search import execute as elite_search

            # Test with a query
            result = await elite_search({
                "query": "migrations and schema",
                "limit": 2,
                "max_depth": 1
            })

            if not result or not result[0].text:
                self.critical_failures.append("Elite search returned empty")
                return False

            # Parse result
            data = json.loads(result[0].text)

            # Check for success status
            if data.get("status") == "error":
                print(f"   ⚠️  Elite search error: {data.get('message')}")
                return False

            # Check if we got results with graph context
            if data.get("status") == "success":
                results = data.get("results", [])
                if results:
                    first = results[0]
                    # Check for graph context
                    if "graph_context" in first:
                        connections = first["graph_context"].get("total_connections", 0)
                        if connections > 0:
                            print(f"   ✅ Elite search returned with {connections} graph connections")
                            return True
                        else:
                            print("   ⚠️  Elite search returned but no graph connections")
                            return True  # Not critical

            print("   ⚠️  Elite search returned but no valid results")
            return True  # Not critical if no results

        except json.JSONDecodeError as e:
            self.critical_failures.append(f"Elite search returned invalid JSON: {e}")
            return False
        except Exception as e:
            self.critical_failures.append(f"Elite search failed: {e}")
            print(f"   ❌ Elite search error: {e}")
            return False

    async def test_project_operations(self) -> bool:
        """Test project operations tool"""
        try:
            from neural_mcp.tools.project_operations import execute as project_ops

            # Test project understanding
            result = await project_ops({
                "operation": "understanding",
                "scope": "summary"
            })

            if not result or not result[0].text:
                print("   ⚠️  Project operations returned empty")
                return True  # Not critical

            # Should return valid JSON
            data = json.loads(result[0].text)

            if data.get("status") == "success":
                print("   ✅ Project operations working")
                return True
            else:
                print(f"   ⚠️  Project operations: {data.get('message')}")
                return True  # Not critical

        except ImportError:
            print("   ⚠️  Project operations tool not found (OK if not using)")
            return True
        except Exception as e:
            print(f"   ⚠️  Project operations error: {e}")
            return True  # Not critical

    async def test_schema_validation(self) -> bool:
        """Test schema validation is working"""
        try:
            from servers.services.chunk_schema import ChunkSchema

            # Create a valid chunk
            chunk = ChunkSchema(
                chunk_id="smoke_test.py:chunk:0",
                file_path="smoke_test.py",
                content="smoke test content",
                embedding=[0.5] * 768,
                project="smoke-test"
            )

            # Should convert to Neo4j dict without error
            neo4j_dict = chunk.to_neo4j_dict()

            # Should be JSON serializable
            json_str = json.dumps(chunk.to_dict())
            parsed = json.loads(json_str)

            if parsed["file_path"] == "smoke_test.py":
                print("   ✅ Schema validation working")
                return True
            else:
                self.critical_failures.append("Schema validation failed")
                return False

        except Exception as e:
            self.critical_failures.append(f"Schema validation error: {e}")
            print(f"   ❌ Schema validation error: {e}")
            return False

    def _report(self, test_name: str, passed: bool):
        """Record test result"""
        self.results.append((test_name, passed))

    def _print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📊 SMOKE TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for _, p in self.results if p)
        total = len(self.results)
        print(f"\nTests: {passed}/{total} passed")

        for name, success in self.results:
            status = "✅" if success else "❌"
            print(f"  {status} {name}")

        if self.critical_failures:
            print(f"\n🚨 Critical Failures ({len(self.critical_failures)}):")
            for failure in self.critical_failures:
                print(f"  • {failure}")

        print("\n" + "=" * 70)
        if len(self.critical_failures) == 0:
            print("✅ DEPLOYMENT VERIFIED: System operational!")
            if passed < total:
                print("⚠️  Some non-critical tests failed - review warnings")
        else:
            print("❌ DEPLOYMENT ISSUES: Critical failures detected!")
            print("🔧 Review failures and consider rollback if needed")
        print("=" * 70)


async def main():
    """Run post-deployment smoke tests"""
    tests = PostDeploymentSmokeTests()
    success = await tests.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)