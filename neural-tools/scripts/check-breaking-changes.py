#!/usr/bin/env python3
"""
Breaking Changes Detector
Identifies potential breaking changes before deployment
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

class BreakingChangeDetector:
    def __init__(self):
        self.neural_tools_dir = Path(__file__).parent.parent
        self.breaking_changes = []
        self.api_changes = []

    def detect_all(self) -> bool:
        """Run all breaking change detections"""
        print("🔍 Checking for breaking changes...")

        self.check_removed_functions()
        self.check_signature_changes()
        self.check_import_changes()
        self.check_config_changes()
        self.check_mcp_tool_changes()

        return self.report_results()

    def check_removed_functions(self):
        """Check if any public functions were removed"""
        # Define critical public APIs that must not be removed
        critical_apis = {
            "src/servers/services/project_context_manager.py": [
                "get_project_context_manager",
                "ProjectContextManager",
                "detect_project",
                "switch_project_with_teardown",
            ],
            "src/servers/services/service_container.py": [
                "ServiceContainer",
                "initialize",
                "ensure_indexer_running",
            ],
            "src/servers/services/indexer_orchestrator.py": [
                "IndexerOrchestrator",
                "ensure_indexer",
            ],
        }

        for file_path, required_functions in critical_apis.items():
            full_path = self.neural_tools_dir / file_path
            if not full_path.exists():
                self.breaking_changes.append(f"Critical file removed: {file_path}")
                continue

            content = full_path.read_text()
            tree = ast.parse(content)

            defined_functions = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_functions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined_functions.add(node.name)
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            defined_functions.add(f"{node.name}.{method.name}")

            for func in required_functions:
                if func not in defined_functions and f".{func}" not in str(defined_functions):
                    self.breaking_changes.append(f"Critical function removed: {func} from {file_path}")

    def check_signature_changes(self):
        """Check if function signatures have breaking changes"""
        # Check critical function signatures
        critical_signatures = {
            "ServiceContainer.__init__": ["context_manager", "project_name"],
            "IndexerOrchestrator.__init__": ["context_manager"],
            "ProjectContextManager.switch_project_with_teardown": ["project_path"],
        }

        for func_path, expected_params in critical_signatures.items():
            parts = func_path.split(".")
            if len(parts) == 2:
                class_name, method_name = parts
                file_map = {
                    "ServiceContainer": "src/servers/services/service_container.py",
                    "IndexerOrchestrator": "src/servers/services/indexer_orchestrator.py",
                    "ProjectContextManager": "src/servers/services/project_context_manager.py",
                }

                if class_name in file_map:
                    file_path = self.neural_tools_dir / file_map[class_name]
                    if file_path.exists():
                        content = file_path.read_text()
                        tree = ast.parse(content)

                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef) and node.name == class_name:
                                for method in node.body:
                                    if isinstance(method, ast.FunctionDef) and method.name == method_name:
                                        args = [arg.arg for arg in method.args.args if arg.arg != "self"]
                                        for param in expected_params:
                                            if param not in args:
                                                self.api_changes.append(
                                                    f"Parameter '{param}' missing from {func_path}"
                                                )

    def check_import_changes(self):
        """Check if import structure has changed"""
        # Verify critical imports are still available
        critical_imports = [
            ("src/neural_mcp/neural_server_stdio.py", "from servers.services.project_context_manager import get_project_context_manager"),
            ("src/servers/services/service_container.py", "from .project_context_manager import ProjectContextManager"),
            ("src/servers/services/indexer_orchestrator.py", "from .container_discovery import ContainerDiscoveryService"),
        ]

        for file_path, required_import in critical_imports:
            full_path = self.neural_tools_dir / file_path
            if full_path.exists():
                content = full_path.read_text()
                # Normalize import statement for checking
                import_check = required_import.replace("from ", "").replace("import ", "")
                if not any(import_check.replace(".", "") in line.replace(".", "") for line in content.split("\n")):
                    self.api_changes.append(f"Critical import missing: '{required_import}' in {file_path}")

    def check_config_changes(self):
        """Check for configuration breaking changes"""
        # Check environment variable usage
        env_vars = [
            "PROJECT_NAME",
            "PROJECT_PATH",
            "NEO4J_URI",
            "NEO4J_PASSWORD",
            "QDRANT_HOST",
            "QDRANT_PORT",
            "EMBEDDING_SERVICE_HOST",
            "EMBEDDING_SERVICE_PORT",
        ]

        config_files = [
            "src/servers/services/project_context_manager.py",
            "src/servers/config/runtime.py",
        ]

        for config_file in config_files:
            file_path = self.neural_tools_dir / config_file
            if file_path.exists():
                content = file_path.read_text()
                for env_var in env_vars:
                    if env_var in content:
                        # Check if the usage pattern changed
                        if f"os.getenv('{env_var}')" not in content and f'os.getenv("{env_var}")' not in content:
                            if f"os.environ.get('{env_var}')" not in content and f'os.environ.get("{env_var}")' not in content:
                                self.api_changes.append(f"Environment variable usage changed: {env_var} in {config_file}")

    def check_mcp_tool_changes(self):
        """Check if MCP tool definitions changed"""
        mcp_file = self.neural_tools_dir / "src/neural_mcp/neural_server_stdio.py"
        if not mcp_file.exists():
            self.breaking_changes.append("MCP server file missing!")
            return

        content = mcp_file.read_text()

        # Check for critical MCP tools
        required_tools = [
            "semantic_code_search",
            "graphrag_hybrid_search",
            "project_understanding",
            "neural_system_status",
            "set_project_context",
        ]

        for tool in required_tools:
            if f'name="{tool}"' not in content and f"name='{tool}'" not in content:
                self.breaking_changes.append(f"MCP tool removed or renamed: {tool}")

    def report_results(self) -> bool:
        """Report detection results"""
        print("\n" + "="*60)
        print("🔍 BREAKING CHANGES DETECTION RESULTS")
        print("="*60)

        if self.breaking_changes:
            print("\n🚨 BREAKING CHANGES DETECTED:")
            for change in self.breaking_changes:
                print(f"  • {change}")

        if self.api_changes:
            print("\n⚠️  API CHANGES (may affect compatibility):")
            for change in self.api_changes:
                print(f"  • {change}")

        if not self.breaking_changes and not self.api_changes:
            print("\n✅ No breaking changes detected")
            return True
        elif self.breaking_changes:
            print("\n❌ Breaking changes must be addressed before deployment")
            return False
        else:
            print("\n⚠️  API changes detected - review for compatibility")
            return True

if __name__ == "__main__":
    detector = BreakingChangeDetector()
    success = detector.detect_all()
    sys.exit(0 if success else 1)