#!/usr/bin/env python3
"""
Environment-aware installer for Neural Flow MCP Server
Automatically detects Python version and installs compatible dependencies
"""

import sys
import subprocess
import os
from pathlib import Path

def get_python_version():
    """Get the current Python version as a tuple"""
    return sys.version_info[:2]

def detect_existing_packages():
    """Detect already installed packages and their versions"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, check=True
        )
        import json
        packages = json.loads(result.stdout)
        return {pkg['name'].lower(): pkg['version'] for pkg in packages}
    except:
        return {}

def select_requirements_file():
    """Select the appropriate requirements file based on Python version"""
    major, minor = get_python_version()
    python_version = f"{major}.{minor}"
    
    if major == 3:
        if minor >= 13:
            return "requirements/requirements-py313.txt", "Python 3.13+"
        elif minor >= 10:
            return "requirements/requirements-py310.txt", "Python 3.10-3.12"
        elif minor >= 8:
            return "requirements/requirements-py38.txt", "Python 3.8-3.9"
        else:
            print(f"⚠️  Python {python_version} is too old. Minimum version is 3.8")
            print("   Please upgrade Python or use pyenv/conda to manage versions")
            return None, None
    else:
        print(f"❌ Python {major}.x is not supported. Python 3.8+ required")
        return None, None

def check_numpy_chromadb_compatibility(installed_packages):
    """Check if NumPy and ChromaDB versions are compatible"""
    numpy_version = installed_packages.get('numpy', '0.0.0')
    chromadb_version = installed_packages.get('chromadb', '0.0.0')
    
    issues = []
    
    # Parse major version of numpy
    try:
        numpy_major = int(numpy_version.split('.')[0])
        
        if numpy_major >= 2:
            # NumPy 2.x requires ChromaDB 1.0.20+
            if chromadb_version and chromadb_version < '1.0.20':
                issues.append(
                    f"⚠️  NumPy {numpy_version} requires ChromaDB >= 1.0.20 "
                    f"(you have {chromadb_version})"
                )
        else:
            # NumPy 1.x works best with older ChromaDB
            if chromadb_version and chromadb_version >= '1.0':
                issues.append(
                    f"⚠️  NumPy {numpy_version} works better with ChromaDB < 1.0 "
                    f"(you have {chromadb_version})"
                )
    except:
        pass
    
    return issues

def install_requirements(req_file):
    """Install requirements from the specified file"""
    print(f"\n📦 Installing from {req_file}...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def main():
    """Main installation process"""
    print("🚀 Neural Flow MCP Server - Environment-Aware Installer")
    print("=" * 60)
    
    # Check Python version
    major, minor = get_python_version()
    print(f"✅ Python version: {major}.{minor}.{sys.version_info[2]}")
    
    # Detect existing packages
    print("\n🔍 Checking existing packages...")
    installed = detect_existing_packages()
    
    if installed:
        relevant_packages = ['numpy', 'chromadb', 'mcp', 'onnxruntime', 'tiktoken']
        print("\n📋 Installed packages:")
        for pkg in relevant_packages:
            if pkg in installed:
                print(f"   • {pkg}: {installed[pkg]}")
    
    # Check compatibility issues
    issues = check_numpy_chromadb_compatibility(installed)
    if issues:
        print("\n⚠️  Compatibility warnings:")
        for issue in issues:
            print(f"   {issue}")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Installation cancelled")
            return
    
    # Select requirements file
    req_file, description = select_requirements_file()
    if not req_file:
        return
    
    print(f"\n✅ Selected requirements: {req_file} ({description})")
    
    # Check if file exists
    if not Path(req_file).exists():
        print(f"❌ {req_file} not found in current directory")
        print("   Make sure you're running this from the project root")
        return
    
    # Option to upgrade or fresh install
    if installed:
        print("\n📦 Installation options:")
        print("1. Upgrade/Install missing packages only")
        print("2. Force reinstall all packages")
        print("3. Cancel")
        
        choice = input("\nSelect option (1/2/3): ").strip()
        
        if choice == '3':
            print("Installation cancelled")
            return
        elif choice == '2':
            print("\n🔄 Force reinstalling all packages...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--force-reinstall", "-r", req_file],
                check=True
            )
        else:
            print("\n📦 Installing/Upgrading packages...")
            install_requirements(req_file)
    else:
        install_requirements(req_file)
    
    # Verify installation
    print("\n✅ Installation complete! Verifying imports...")
    
    try:
        sys.path.insert(0, '.claude/mcp-tools')
        from mcp_neural_server_preloaded import PreloadedNeuralServer
        print("   ✅ MCP server imports successfully")
        
        from neural_dynamic_memory_system import NeuralDynamicMemorySystem
        print("   ✅ Neural memory system imports successfully")
        
        print("\n🎉 Neural Flow MCP Server is ready to use!")
        print("\nNext steps:")
        print("1. Restart Claude to load the MCP server")
        print("2. The server will pre-load models on first start (~2-3 seconds)")
        print("3. Enjoy <50ms query performance!")
        
    except ImportError as e:
        print(f"\n⚠️  Import verification failed: {e}")
        print("   Try running: python3 -m pip install --upgrade pip")
        print("   Then run this installer again")

if __name__ == "__main__":
    main()