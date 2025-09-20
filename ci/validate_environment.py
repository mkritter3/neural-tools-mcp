#!/usr/bin/env python3
"""
Production Environment Pre-flight Validator
Ensures code will work in production MCP environment BEFORE expensive tests run
Based on Gemini's comprehensive validation blind spot analysis
"""

import os
import sys
import importlib
from pathlib import Path

def validate_pythonpath():
    """Ensure PYTHONPATH matches production MCP setup"""
    print("--- Validating Python Path Setup ---")

    src_path = os.path.abspath("neural-tools/src")
    neural_tools_path = os.path.abspath("neural-tools")

    # Check that ONLY src is in path (not neural-tools itself)
    if src_path not in [os.path.abspath(p) for p in sys.path]:
        print(f"❌ ERROR: 'src' directory ({src_path}) not in PYTHONPATH")
        print("   Production MCP adds neural-tools/src to path")
        sys.exit(1)

    if neural_tools_path in [os.path.abspath(p) for p in sys.path]:
        print(f"❌ ERROR: Project root ({neural_tools_path}) is in PYTHONPATH")
        print("   This allows broken imports that will fail in production!")
        sys.exit(1)

    print("✅ Python path correctly configured (only src/ added)")
    return True

def validate_imports():
    """Test that all critical imports work with production paths"""
    print("\n--- Running Import Validation ---")

    # Critical modules that MUST work in production
    modules_to_check = [
        "servers.services.sync_manager",
        "servers.services.event_store",
        "servers.services.indexer_service",
        "servers.services.service_container",
        "servers.services.project_context_manager",
        "servers.services.neo4j_service",
        "servers.services.qdrant_service",
        "neural_mcp.neural_server_stdio",
    ]

    failed_imports = []
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"  ✅ Imported '{module}'")
        except ImportError as e:
            print(f"  ❌ Failed to import '{module}': {e}")
            failed_imports.append((module, str(e)))

    # Test that WRONG imports fail (as they should)
    wrong_imports = [
        "src.servers.services.sync_manager",  # Should fail - has 'src' prefix
        "event_store",  # Should fail - not in root of path
        "sync_manager",  # Should fail - not in root of path
    ]

    print("\n--- Validating Wrong Imports Fail ---")
    for module in wrong_imports:
        try:
            # Use exec to avoid syntax errors
            exec(f"import {module}")
            print(f"  ❌ PROBLEM: '{module}' imported (it shouldn't!)")
            failed_imports.append((module, "Should not be importable"))
        except ImportError:
            print(f"  ✅ '{module}' correctly fails to import")

    if failed_imports:
        print("\n❌ Import validation FAILED:")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
        sys.exit(1)

    print("\n✅ All imports validated for production")
    return True

def validate_configuration():
    """Validate configuration loading per ADR-0037"""
    print("\n--- Running Configuration Validation ---")

    required_env_vars = [
        "NEO4J_URI",
        "NEO4J_PASSWORD",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "EMBEDDING_SERVICE_HOST",
        "EMBEDDING_SERVICE_PORT",
    ]

    missing_vars = []

    # Check if we're in CI or local
    is_ci = os.getenv("CI") == "true"

    if is_ci:
        print("  Running in CI environment")
        # In CI, these should be set or have defaults
        for var in required_env_vars:
            value = os.getenv(var)
            if value:
                print(f"  ✅ {var} = {value[:20]}..." if len(value) > 20 else f"  ✅ {var} = {value}")
            else:
                # Check if there's a sensible default
                if var == "NEO4J_URI":
                    default = "bolt://localhost:47687"
                elif var == "QDRANT_HOST":
                    default = "localhost"
                elif var == "QDRANT_PORT":
                    default = "46333"
                else:
                    default = None

                if default:
                    print(f"  ⚠️  {var} not set, would use default: {default}")
                else:
                    missing_vars.append(var)
                    print(f"  ❌ {var} not set and no default available")
    else:
        print("  Running in local environment")
        print("  Environment variables would be loaded from .mcp.json in production")

    if missing_vars and is_ci:
        print(f"\n❌ Missing required environment variables: {missing_vars}")
        sys.exit(1)

    print("\n✅ Configuration validation passed")
    return True

def validate_docker_networking():
    """Check for localhost vs container networking issues"""
    print("\n--- Validating Network Configuration ---")

    # Check if we're using localhost or Docker service names
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:47687")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")

    in_container = Path("/.dockerenv").exists()

    if in_container:
        print("  Running inside Docker container")
        if "localhost" in neo4j_uri or qdrant_host == "localhost":
            print("  ⚠️  WARNING: Using 'localhost' inside container")
            print("     Should use 'host.docker.internal' or service names")
    else:
        print("  Running on host system")
        if "host.docker.internal" in neo4j_uri or "host.docker.internal" in qdrant_host:
            print("  ❌ ERROR: Using 'host.docker.internal' outside container")
            sys.exit(1)

    print("✅ Network configuration appropriate for environment")
    return True

def main():
    """Run all pre-flight validation checks"""
    print("="*60)
    print("🚀 PRODUCTION ENVIRONMENT PRE-FLIGHT CHECKS")
    print("="*60)

    # Set up path as MCP would
    neural_tools_src = Path(__file__).parent.parent / "neural-tools" / "src"
    if str(neural_tools_src) not in sys.path:
        sys.path.insert(0, str(neural_tools_src))

    all_passed = True

    try:
        validate_pythonpath()
    except SystemExit:
        all_passed = False

    try:
        validate_imports()
    except SystemExit:
        all_passed = False

    try:
        validate_configuration()
    except SystemExit:
        all_passed = False

    try:
        validate_docker_networking()
    except SystemExit:
        all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL PRE-FLIGHT CHECKS PASSED!")
        print("Code is ready for production MCP environment")
    else:
        print("❌ PRE-FLIGHT VALIDATION FAILED")
        print("Fix the issues above before deployment")
        sys.exit(1)
    print("="*60)

if __name__ == "__main__":
    main()