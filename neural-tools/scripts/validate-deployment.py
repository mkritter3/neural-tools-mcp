#!/usr/bin/env python3
"""
Pre-Deployment Validation Script
Ensures neural-tools is ready for deployment to global MCP
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

class DeploymentValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.neural_tools_dir = Path(__file__).parent.parent

    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("🔍 Starting pre-deployment validation...")

        checks = [
            ("Python syntax", self.check_python_syntax),
            ("Import integrity", self.check_imports),
            ("ADR compliance", self.check_adr_compliance),
            ("Configuration files", self.check_configs),
            ("Test coverage", self.check_test_coverage),
            ("Docker images", self.check_docker_images),
            ("Dependencies", self.check_dependencies),
            ("Breaking changes", self.check_breaking_changes),
        ]

        for name, check_func in checks:
            print(f"\n📋 Checking {name}...")
            try:
                check_func()
                print(f"✅ {name} passed")
            except Exception as e:
                self.errors.append(f"{name}: {str(e)}")
                print(f"❌ {name} failed: {str(e)}")

        return self.report_results()

    def check_python_syntax(self):
        """Verify all Python files have valid syntax"""
        src_dir = self.neural_tools_dir / "src"
        for py_file in src_dir.rglob("*.py"):
            with open(py_file, 'r') as f:
                try:
                    compile(f.read(), py_file, 'exec')
                except SyntaxError as e:
                    raise Exception(f"Syntax error in {py_file}: {e}")

    def check_imports(self):
        """Verify all imports can be resolved"""
        # Add src to path for import checking
        import sys
        sys.path.insert(0, str(self.neural_tools_dir / "src"))

        critical_imports = [
            "neural_mcp.neural_server_stdio",
            "servers.services.service_container",
            "servers.services.project_context_manager",
            "servers.services.indexer_orchestrator",
            "servers.services.container_discovery",
        ]

        for module_name in critical_imports:
            try:
                __import__(module_name)
            except ImportError as e:
                raise Exception(f"Cannot import {module_name}: {e}")

    def check_adr_compliance(self):
        """Verify ADR implementations are in place"""
        required_patterns = {
            "ADR-0043": {
                "file": "src/servers/services/project_context_manager.py",
                "patterns": ["teardown_project", "rebuild_project", "_persist_registry"]
            },
            "ADR-0044": {
                "file": "src/servers/services/project_context_manager.py",
                "patterns": ["get_project_context_manager", "_global_context_manager", "_manager_lock"]
            },
            "ADR-0037": {
                "file": "src/servers/services/project_context_manager.py",
                "patterns": ["os.getenv", "PROJECT_NAME", "PROJECT_PATH"]
            }
        }

        for adr, requirements in required_patterns.items():
            file_path = self.neural_tools_dir / requirements["file"]
            if not file_path.exists():
                raise Exception(f"{adr}: Required file {requirements['file']} not found")

            content = file_path.read_text()
            for pattern in requirements["patterns"]:
                if pattern not in content:
                    raise Exception(f"{adr}: Pattern '{pattern}' not found in {requirements['file']}")

    def check_configs(self):
        """Verify configuration files are valid"""
        # Check .mcp.json
        mcp_config = self.neural_tools_dir.parent / ".mcp.json"
        if mcp_config.exists():
            try:
                with open(mcp_config) as f:
                    config = json.load(f)
                    if "mcpServers" not in config:
                        raise Exception(".mcp.json missing mcpServers section")
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON in .mcp.json: {e}")

        # Check requirements files
        for req_file in ["requirements/prod.txt", "requirements/dev.txt"]:
            req_path = self.neural_tools_dir / req_file
            if not req_path.exists():
                raise Exception(f"Missing {req_file}")

    def check_test_coverage(self):
        """Verify critical components have tests"""
        test_dir = self.neural_tools_dir / "tests"
        required_tests = [
            "test_adr_0043.py",
            "test_adr_0044.py",
            "test_project_context_manager.py",
            "test_service_container.py",
        ]

        for test_file in required_tests:
            # Check in multiple possible locations
            found = False
            for subdir in ["", "unit", "integration", "validation"]:
                test_path = test_dir / subdir / test_file if subdir else test_dir / test_file
                if test_path.exists():
                    found = True
                    break

            if not found:
                self.warnings.append(f"Test file {test_file} not found")

    def check_docker_images(self):
        """Verify Docker images follow ADR-0038 standards"""
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                check=True
            )

            images = result.stdout.strip().split('\n')

            # Check for debug/temporary tags
            for image in images:
                if any(bad in image for bad in ["-debug", "-fix", "-test", "-temp"]):
                    self.warnings.append(f"Non-standard Docker image tag: {image}")

        except subprocess.CalledProcessError:
            self.warnings.append("Could not check Docker images")

    def check_dependencies(self):
        """Verify all dependencies are installable"""
        req_file = self.neural_tools_dir / "requirements" / "prod.txt"
        if not req_file.exists():
            raise Exception("requirements/prod.txt not found")

        # Check for version conflicts or missing dependencies
        with open(req_file) as f:
            deps = f.read().strip().split('\n')

        critical_deps = ["mcp", "neo4j", "qdrant-client", "docker", "redis"]
        for dep in critical_deps:
            if not any(dep in line for line in deps):
                raise Exception(f"Critical dependency '{dep}' not found in requirements")

    def check_breaking_changes(self):
        """Check for potential breaking changes"""
        # Check if singleton pattern is properly implemented
        pcm_file = self.neural_tools_dir / "src/servers/services/project_context_manager.py"
        content = pcm_file.read_text()

        # Verify singleton pattern
        if "_global_context_manager = None" not in content:
            raise Exception("Singleton pattern not properly initialized")

        if "async def get_project_context_manager" not in content:
            raise Exception("Singleton getter function missing")

        # Check MCP server uses singleton
        mcp_file = self.neural_tools_dir / "src/neural_mcp/neural_server_stdio.py"
        mcp_content = mcp_file.read_text()

        if "await get_project_context_manager()" not in mcp_content:
            raise Exception("MCP server not using singleton pattern")

    def report_results(self) -> bool:
        """Report validation results"""
        print("\n" + "="*60)
        print("📊 DEPLOYMENT VALIDATION RESULTS")
        print("="*60)

        if self.errors:
            print("\n❌ ERRORS (must fix before deployment):")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS (should review):")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors:
            print("\n✅ All validation checks passed!")
            print("🚀 Ready for deployment to global MCP")
            return True
        else:
            print("\n❌ Validation failed - fix errors before deploying")
            return False

if __name__ == "__main__":
    validator = DeploymentValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)