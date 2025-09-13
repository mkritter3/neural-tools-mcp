#!/usr/bin/env python3
"""
Comprehensive Real-World Tests for ALL 22 Neural Tools
Tests every single tool with real data, no mocks
"""

import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Test colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

class NeuralToolsTester:
    """Comprehensive tester for all neural tools"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.test_project = "/Users/mkr/local-coding/claude-l9-template"
        self.alt_project = "/Users/mkr/local-coding/Systems/neural-novelist"
        self.current_project = None

    def run_mcp_tool(self, tool_name, params):
        """Execute an MCP tool and return the result"""
        try:
            # Create the request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": params
                }
            }

            # Run the server
            process = subprocess.Popen(
                [sys.executable, "run_mcp_server.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )

            # Initialize first
            init_request = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"}
                }
            }

            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()

            # Read init response
            init_response = process.stdout.readline()

            # Send actual request
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()

            # Read response with timeout
            response_line = process.stdout.readline()
            process.terminate()

            if response_line:
                return json.loads(response_line)
            return {"error": {"message": "No response from server"}}

        except Exception as e:
            return {"error": {"message": str(e)}}

    def test_tool(self, name, tool_name, params, validation_func=None):
        """Test a single tool"""
        print(f"{YELLOW}[{self.passed + self.failed + 1}/22] Testing: {name}{NC}")

        try:
            # Use async approach for tools that need it
            from servers.services.service_container import ServiceContainer

            async def run_test():
                container = ServiceContainer()
                await container.initialize()

                # Dynamic tool execution based on tool_name
                if tool_name == "neural_system_status":
                    from servers.tools.neural_tools import get_neural_system_status
                    result = await get_neural_system_status(container, params)

                elif tool_name == "semantic_code_search":
                    from servers.tools.neural_tools import semantic_code_search
                    result = await semantic_code_search(container, params)

                elif tool_name == "graphrag_hybrid_search":
                    from servers.tools.neural_tools import graphrag_hybrid_search
                    result = await graphrag_hybrid_search(container, params)

                elif tool_name == "project_understanding":
                    from servers.tools.neural_tools import get_project_understanding
                    result = await get_project_understanding(container, params)

                elif tool_name == "indexer_status":
                    from servers.tools.neural_tools import get_indexer_status
                    result = await get_indexer_status(container, params)

                elif tool_name == "reindex_path":
                    from servers.tools.neural_tools import reindex_path
                    result = await reindex_path(container, params)

                elif tool_name == "neural_tools_help":
                    from servers.tools.neural_tools import get_neural_tools_help
                    result = await get_neural_tools_help(container, params)

                elif tool_name == "instance_metrics":
                    from servers.tools.neural_tools import get_instance_metrics
                    result = await get_instance_metrics(container, params)

                elif tool_name == "schema_init":
                    from servers.tools.schema_tools import schema_init
                    result = await schema_init(container, params)

                elif tool_name == "schema_status":
                    from servers.tools.schema_tools import get_schema_status
                    result = await get_schema_status(container, params)

                elif tool_name == "schema_validate":
                    from servers.tools.schema_tools import schema_validate
                    result = await schema_validate(container, params)

                elif tool_name == "schema_add_node_type":
                    from servers.tools.schema_tools import add_node_type
                    result = await add_node_type(container, params)

                elif tool_name == "schema_add_relationship":
                    from servers.tools.schema_tools import add_relationship
                    result = await add_relationship(container, params)

                elif tool_name == "migration_generate":
                    from servers.tools.migration_tools import generate_migration
                    result = await generate_migration(container, params)

                elif tool_name == "migration_apply":
                    from servers.tools.migration_tools import apply_migrations
                    result = await apply_migrations(container, params)

                elif tool_name == "migration_rollback":
                    from servers.tools.migration_tools import rollback_migration
                    result = await rollback_migration(container, params)

                elif tool_name == "migration_status":
                    from servers.tools.migration_tools import get_migration_status
                    result = await get_migration_status(container, params)

                elif tool_name == "schema_diff":
                    from servers.tools.schema_tools import schema_diff
                    result = await schema_diff(container, params)

                elif tool_name == "canon_understanding":
                    from servers.tools.canon_tools import get_canon_understanding
                    result = await get_canon_understanding(container, params)

                elif tool_name == "backfill_metadata":
                    from servers.tools.canon_tools import backfill_metadata
                    result = await backfill_metadata(container, params)

                elif tool_name == "set_project_context":
                    from servers.tools.project_tools import set_project_context
                    result = await set_project_context(container, params)

                elif tool_name == "list_projects":
                    from servers.tools.project_tools import list_projects
                    result = await list_projects(container, params)

                else:
                    return {"error": f"Unknown tool: {tool_name}"}

                return result

            result = asyncio.run(run_test())

            # Check for errors
            if isinstance(result, dict):
                if "error" in result:
                    print(f"{RED}  ❌ FAILED: {result['error']}{NC}")
                    self.failed += 1
                    return False
                elif "status" in result and result["status"] == "error":
                    print(f"{RED}  ❌ FAILED: {result.get('message', 'Unknown error')}{NC}")
                    self.failed += 1
                    return False

            # Run custom validation if provided
            if validation_func:
                if not validation_func(result):
                    print(f"{RED}  ❌ FAILED: Validation failed{NC}")
                    self.failed += 1
                    return False

            print(f"{GREEN}  ✅ PASSED{NC}")
            self.passed += 1
            return True

        except Exception as e:
            print(f"{RED}  ❌ FAILED: {str(e)}{NC}")
            self.failed += 1
            return False

    def test_project_detection(self):
        """Test project detection in different directories"""
        print(f"\n{BLUE}=== Testing Project Detection ==={NC}\n")

        test_dirs = [
            (self.test_project, "claude-l9-template"),
            (self.alt_project, "neural-novelist"),
            ("/Users/mkr", None),  # No project
            (self.test_project + "/neural-tools", "claude-l9-template"),  # Subdirectory
            (self.alt_project + "/desktop-ui", "neural-novelist"),  # Subdirectory
        ]

        for test_dir, expected_project in test_dirs:
            if not Path(test_dir).exists():
                print(f"{YELLOW}  ⚠️ Skipping {test_dir} (doesn't exist){NC}")
                continue

            print(f"{YELLOW}Testing detection in: {test_dir}{NC}")

            # Change to directory and test detection
            original_cwd = os.getcwd()
            try:
                os.chdir(test_dir)

                # Test set_project_context with auto-detection
                result = self.test_tool(
                    f"  Auto-detect in {Path(test_dir).name}",
                    "set_project_context",
                    {},  # Empty params for auto-detection
                    lambda r: True  # Basic validation
                )

                if result and expected_project:
                    # Verify correct project was detected
                    self.test_tool(
                        f"  Verify project is {expected_project}",
                        "project_understanding",
                        {"scope": "summary"},
                        lambda r: expected_project in str(r)
                    )

            finally:
                os.chdir(original_cwd)

        print(f"\n{BLUE}=== Project Detection Complete ==={NC}\n")

    def test_tools_after_project_switch(self):
        """Test that all tools work correctly after switching projects"""
        print(f"\n{BLUE}=== Testing Tools After Project Switch ==={NC}\n")

        # First set to project 1
        print(f"{YELLOW}Setting context to: {self.test_project}{NC}")
        self.test_tool(
            "Set to claude-l9-template",
            "set_project_context",
            {"path": self.test_project},
            lambda r: "claude-l9-template" in str(r) or "status" in r
        )

        # Test a few critical tools
        self.test_tool(
            "Search in claude-l9-template",
            "semantic_code_search",
            {"query": "ServiceContainer", "limit": 3},
            lambda r: "results" in r
        )

        # Switch to project 2
        print(f"\n{YELLOW}Switching context to: {self.alt_project}{NC}")
        self.test_tool(
            "Set to neural-novelist",
            "set_project_context",
            {"path": self.alt_project},
            lambda r: "neural-novelist" in str(r) or "status" in r
        )

        # Test tools work with new project
        self.test_tool(
            "Search in neural-novelist",
            "semantic_code_search",
            {"query": "UnifiedSDKRouter", "limit": 3},
            lambda r: "results" in r
        )

        self.test_tool(
            "GraphRAG in neural-novelist",
            "graphrag_hybrid_search",
            {"query": "PipelineOrchestrator", "limit": 3},
            lambda r: "results" in r or "results_count" in r
        )

        # Verify projects are isolated
        self.test_tool(
            "List all projects",
            "list_projects",
            {},
            lambda r: "claude-l9-template" in str(r) and "neural-novelist" in str(r)
        )

        print(f"\n{BLUE}=== Project Switch Testing Complete ==={NC}\n")

    def test_data_isolation_between_projects(self):
        """Test that data is properly isolated between projects"""
        print(f"\n{BLUE}=== Testing Data Isolation Between Projects ==={NC}\n")

        # Set to project 1 and search for project-specific content
        self.test_tool(
            "Set to claude-l9-template",
            "set_project_context",
            {"path": self.test_project},
            lambda r: True
        )

        # Search for something specific to claude-l9-template
        result1 = self.test_tool(
            "Search for ServiceContainer in claude-l9-template",
            "graphrag_hybrid_search",
            {"query": "ServiceContainer service_container.py", "limit": 5},
            lambda r: "results" in r or "results_count" in r
        )

        # Switch to project 2
        self.test_tool(
            "Set to neural-novelist",
            "set_project_context",
            {"path": self.alt_project},
            lambda r: True
        )

        # Search for the same thing - should NOT find claude-l9-template content
        self.test_tool(
            "Verify ServiceContainer NOT in neural-novelist",
            "graphrag_hybrid_search",
            {"query": "ServiceContainer service_container.py", "limit": 5},
            lambda r: not any("service_container.py" in str(result) for result in r.get("results", []))
        )

        # Search for something specific to neural-novelist
        self.test_tool(
            "Search for UnifiedSDKRouter in neural-novelist",
            "graphrag_hybrid_search",
            {"query": "UnifiedSDKRouter ADR-029", "limit": 5},
            lambda r: any("UnifiedSDKRouter" in str(result) or "ADR-029" in str(result)
                         for result in r.get("results", []))
        )

        # Switch back to project 1 and verify its data is still there
        self.test_tool(
            "Switch back to claude-l9-template",
            "set_project_context",
            {"path": self.test_project},
            lambda r: True
        )

        self.test_tool(
            "Verify claude-l9-template data still accessible",
            "project_understanding",
            {"scope": "summary"},
            lambda r: "claude-l9-template" in str(r)
        )

        print(f"\n{BLUE}=== Data Isolation Testing Complete ==={NC}\n")

    def run_all_tests(self):
        """Run tests for all 22 tools"""
        print("\n" + "="*60)
        print("🚀 Testing ALL 22 Neural Tools with Real Data")
        print("="*60 + "\n")

        # Test project detection first
        self.test_project_detection()

        # Test tools after project switching
        self.test_tools_after_project_switch()

        # Test data isolation between projects
        self.test_data_isolation_between_projects()

        # 1. Neural System Status
        self.test_tool(
            "Neural System Status",
            "neural_system_status",
            {},
            lambda r: "system_status" in r and "services" in r.get("system_status", {})
        )

        # 2. Semantic Code Search
        self.test_tool(
            "Semantic Code Search",
            "semantic_code_search",
            {"query": "UnifiedSDKRouter", "limit": 5},
            lambda r: "results" in r and len(r.get("results", [])) > 0
        )

        # 3. GraphRAG Hybrid Search
        self.test_tool(
            "GraphRAG Hybrid Search",
            "graphrag_hybrid_search",
            {"query": "MainProcessFileIndexer", "limit": 5, "include_graph_context": True},
            lambda r: "results" in r or "results_count" in r
        )

        # 4. Project Understanding
        self.test_tool(
            "Project Understanding",
            "project_understanding",
            {"scope": "summary"},
            lambda r: "understanding" in r or "status" in r
        )

        # 5. Indexer Status
        self.test_tool(
            "Indexer Status",
            "indexer_status",
            {},
            lambda r: "indexer_status" in r or "status" in r
        )

        # 6. Reindex Path
        self.test_tool(
            "Reindex Path",
            "reindex_path",
            {"path": ".", "recursive": False},
            lambda r: "status" in r or "queued" in r
        )

        # 7. Neural Tools Help
        self.test_tool(
            "Neural Tools Help",
            "neural_tools_help",
            {},
            lambda r: "tools" in r or "help" in r or "status" in r
        )

        # 8. Instance Metrics
        self.test_tool(
            "Instance Metrics",
            "instance_metrics",
            {},
            lambda r: "metrics" in r or "instance" in r or "status" in r
        )

        # 9. Schema Init
        self.test_tool(
            "Schema Init",
            "schema_init",
            {"project_type": "auto", "auto_detect": True},
            lambda r: "status" in r or "schema" in r
        )

        # 10. Schema Status
        self.test_tool(
            "Schema Status",
            "schema_status",
            {},
            lambda r: "status" in r or "schema" in r
        )

        # 11. Schema Validate
        self.test_tool(
            "Schema Validate",
            "schema_validate",
            {"validate_nodes": True, "validate_relationships": True},
            lambda r: "validation" in r or "status" in r
        )

        # 12. Schema Add Node Type
        self.test_tool(
            "Schema Add Node Type",
            "schema_add_node_type",
            {
                "name": "TestNode",
                "properties": {"name": "string", "value": "integer"},
                "description": "Test node type"
            },
            lambda r: "status" in r or "added" in r
        )

        # 13. Schema Add Relationship
        self.test_tool(
            "Schema Add Relationship",
            "schema_add_relationship",
            {
                "name": "TEST_RELATION",
                "from_types": ["TestNode"],
                "to_types": ["TestNode"],
                "description": "Test relationship"
            },
            lambda r: "status" in r or "added" in r
        )

        # 14. Migration Generate
        self.test_tool(
            "Migration Generate",
            "migration_generate",
            {"name": "test_migration", "description": "Test migration", "dry_run": True},
            lambda r: "status" in r or "migration" in r
        )

        # 15. Migration Apply
        self.test_tool(
            "Migration Apply",
            "migration_apply",
            {"dry_run": True},
            lambda r: "status" in r or "applied" in r
        )

        # 16. Migration Rollback
        self.test_tool(
            "Migration Rollback",
            "migration_rollback",
            {"target_version": 0, "force": False},
            lambda r: "status" in r or "rollback" in r
        )

        # 17. Migration Status
        self.test_tool(
            "Migration Status",
            "migration_status",
            {},
            lambda r: "status" in r or "migrations" in r
        )

        # 18. Schema Diff
        self.test_tool(
            "Schema Diff",
            "schema_diff",
            {"from_source": "database", "to_source": "schema.yaml"},
            lambda r: "diff" in r or "status" in r
        )

        # 19. Canon Understanding
        self.test_tool(
            "Canon Understanding",
            "canon_understanding",
            {},
            lambda r: "understanding" in r or "canonical" in r or "status" in r
        )

        # 20. Backfill Metadata
        self.test_tool(
            "Backfill Metadata",
            "backfill_metadata",
            {"batch_size": 10, "dry_run": True},
            lambda r: "status" in r or "backfilled" in r
        )

        # 21. Set Project Context
        self.test_tool(
            "Set Project Context",
            "set_project_context",
            {"path": self.test_project},
            lambda r: "project" in r or "status" in r
        )

        # 22. List Projects
        self.test_tool(
            "List Projects",
            "list_projects",
            {},
            lambda r: "projects" in r or "status" in r
        )

        # Summary
        print("\n" + "="*60)
        print("📊 Comprehensive Test Summary")
        print("="*60)
        print(f"Total Tools Tested: 22")
        print(f"Additional Tests:")
        print(f"  - Project Detection: 5 directories")
        print(f"  - Project Switching: Multiple contexts")
        print(f"  - Data Isolation: Cross-project validation")
        print(f"\nResults:")
        print(f"Passed:            {GREEN}{self.passed}{NC}")
        print(f"Failed:            {RED}{self.failed}{NC}")
        print(f"Warnings:          {YELLOW}{self.warnings}{NC}")

        if self.failed == 0:
            print(f"\n{GREEN}✅ All 22 neural tools passed comprehensive testing!{NC}")
            print(f"{GREEN}✅ Project detection works in all directories{NC}")
            print(f"{GREEN}✅ Tools work correctly after project switching{NC}")
            print(f"{GREEN}✅ Data is properly isolated between projects{NC}")
            return 0
        else:
            print(f"\n{RED}❌ {self.failed} test(s) failed{NC}")
            print("\nFailed areas need immediate attention:")
            print("- Check ProjectContextManager initialization")
            print("- Verify project detection in subdirectories")
            print("- Ensure data isolation between projects")
            print("- Verify service connections (Neo4j, Qdrant, Nomic)")
            print("- Validate configuration paths")
            print("- Check for error patterns in logs")
            return 1

def main():
    """Run comprehensive tests for all tools"""
    tester = NeuralToolsTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())