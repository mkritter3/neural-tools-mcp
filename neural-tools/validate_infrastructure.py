#!/usr/bin/env python3
"""
L9 Neural Tools Infrastructure Validation
Validates Docker infrastructure fixes without requiring full MCP dependencies
"""

import os
import sys
from pathlib import Path

def validate_file_structure():
    """Validate that required files exist in correct locations"""
    print("🔍 Validating file structure...")
    
    base_dir = Path(__file__).parent
    required_files = [
        "src/mcp/neural_server_stdio.py",  # canonical MCP server
        "src/clients/neo4j_client.py",
        "src/infrastructure/tree_sitter_ast.py",
        "Dockerfile.l9-minimal",
        "Dockerfile.neural-embeddings",
        ".dockerignore",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("  ✅ All required files present")
    return True

def validate_docker_compose_imports():
    """Validate Docker Compose file has correct import paths"""
    print("🔍 Validating Docker Compose import paths...")
    
    # Use top-level docker-compose.yml in repo root
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    if not compose_file.exists():
        print("  ❌ docker-compose.yml not found")
        return False
    content = compose_file.read_text()
    if "/app/src/mcp/neural_server_stdio.py" in content:
        print("  ✅ Supervisor runs canonical stdio server")
        return True
    else:
        print("  ❌ Supervisor command does not reference src/mcp/neural_server_stdio.py")
        return False

def validate_dockerfile_paths():
    """Validate Dockerfiles have correct copy paths"""
    print("🔍 Validating Dockerfile paths...")
    
    dockerfile_l9 = Path(__file__).parent / "Dockerfile.l9-minimal"
    if not dockerfile_l9.exists():
        print("  ❌ Dockerfile.l9-minimal not found")
        return False
    
    content = dockerfile_l9.read_text()
    
    # Check COPY paths
    if "COPY src/ ./src/" in content:
        print("  ✅ L9 Dockerfile COPY paths updated")
    else:
        print("  ❌ L9 Dockerfile COPY paths not updated")
        return False
    
    # Check PYTHONPATH
    if "ENV PYTHONPATH=/app:/app/src:/app/src/servers:/app/common" in content:
        print("  ✅ L9 Dockerfile PYTHONPATH configured")
    else:
        print("  ❌ L9 Dockerfile PYTHONPATH not configured")
        return False
    
    dockerfile_embeddings = Path(__file__).parent / "Dockerfile.neural-embeddings"
    if dockerfile_embeddings.exists():
        embed_content = dockerfile_embeddings.read_text()
        if "src.servers.nomic_embed_server:app" in embed_content:
            print("  ✅ Embeddings Dockerfile updated")
        else:
            print("  ❌ Embeddings Dockerfile not updated")
            return False
    
    return True

def validate_environment_setup():
    """Validate environment configuration"""
    print("🔍 Validating environment setup...")
    
    env_example = Path(__file__).parent / ".env.example"
    if not env_example.exists():
        print("  ❌ .env.example not found")
        return False
    
    content = env_example.read_text()
    if "NEO4J_PASSWORD=" in content:
        print("  ✅ Environment template includes Neo4j password")
    else:
        print("  ❌ Environment template missing Neo4j password")
        return False
    
    dockerignore = Path(__file__).parent / ".dockerignore"
    if dockerignore.exists():
        ignore_content = dockerignore.read_text()
        if "__pycache__/" in ignore_content and "*.py[cod]" in ignore_content:
            print("  ✅ .dockerignore configured for build optimization")
        else:
            print("  ❌ .dockerignore not properly configured")
            return False
    else:
        print("  ❌ .dockerignore not found")
        return False
    
    return True

def validate_mcp_tools_structure():
    """Basic check that canonical server exists (tool details validated by tests)."""
    print("🔍 Validating MCP tools structure...")
    server_file = Path(__file__).parent / "src" / "mcp" / "neural_server_stdio.py"
    if server_file.exists():
        print("  ✅ Canonical stdio server present")
        return True
    print("  ❌ Canonical stdio server missing")
    return False

def main():
    """Run all validation checks"""
    print("L9 Neural Tools Infrastructure Validation")
    print("=" * 50)
    
    checks = [
        ("File Structure", validate_file_structure),
        ("Docker Compose Imports", validate_docker_compose_imports),
        ("Dockerfile Paths", validate_dockerfile_paths),
        ("Environment Setup", validate_environment_setup),
        ("MCP Tools Structure", validate_mcp_tools_structure)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n📋 {name}")
        if check_func():
            passed += 1
            print(f"  ✅ {name} - PASSED")
        else:
            print(f"  ❌ {name} - FAILED")
    
    print("\n" + "=" * 50)
    print(f"VALIDATION SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 L9 Neural Tools infrastructure fixes are complete!")
        print("✅ Ready for Docker deployment")
        return 0
    else:
        print("❌ Infrastructure fixes need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
