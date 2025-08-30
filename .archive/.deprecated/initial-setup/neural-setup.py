#!/usr/bin/env python3
"""
Neural Memory Setup Tool
Simple command-line interface for setting up projects with neural memory
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add both current directory and neural-system to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / '.claude' / 'neural-system'))

# Import with the correct module name
import importlib.util
spec = importlib.util.spec_from_file_location("neural_memory_mcp", current_dir / "neural-memory-mcp.py")
neural_memory_mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(neural_memory_mcp)

GlobalNeuralMemoryMCP = neural_memory_mcp.GlobalNeuralMemoryMCP

async def setup_project(project_path: str, enable_hooks: bool = True, enable_auto_indexing: bool = True):
    """Setup a project using the MCP server's setup_project functionality"""
    
    print("🚀 Neural Memory Project Setup")
    print("=" * 50)
    
    try:
        # Initialize MCP server
        mcp = GlobalNeuralMemoryMCP()
        await mcp.initialize()
        
        # Call setup_project tool directly
        target_path = Path(project_path).resolve() if project_path != "." else Path.cwd()
        
        print(f"📁 Setting up project: {target_path.name}")
        print(f"🎯 Path: {target_path}")
        print(f"📎 Hooks: {'enabled' if enable_hooks else 'disabled'}")
        print(f"🔄 Auto-indexing: {'enabled' if enable_auto_indexing else 'disabled'}")
        print()
        
        # Since we can't easily call the MCP tool directly, let's implement the logic here
        from project_isolation import ProjectIsolation
        import json
        
        # Create project configuration
        claude_dir = target_path / '.claude'
        claude_dir.mkdir(exist_ok=True)
        
        # Generate project-specific configuration
        project_isolation = ProjectIsolation()
        config = project_isolation.get_project_config()
        
        # Save project configuration
        config_path = claude_dir / 'project-config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Project '{target_path.name}' configuration created")
        print(f"🐳 Container: {config['container_name']}")
        print(f"🔌 Ports: REST {config['rest_port']}, gRPC {config['grpc_port']}")
        print(f"💾 Storage: {config['storage_path']}")
        
        # Create project-specific Qdrant container
        container_result = project_isolation.ensure_project_container(
            project_isolation.get_project_context()
        )
        print(f"🐳 Container status: {container_result['status']}")
        
        # Setup hooks if requested
        if enable_hooks:
            hooks_result = await mcp._setup_hooks(claude_dir, enable_auto_indexing)
            for result in hooks_result:
                print(result)
        
        # Setup .mcp.json if it doesn't exist
        mcp_json_path = target_path / '.mcp.json'
        if not mcp_json_path.exists():
            mcp_config = {
                "mcpServers": {
                    "neural-memory": {
                        "command": "python3",
                        "args": [str(Path(__file__).parent / "neural-memory-mcp.py")],
                        "env": {
                            "PYTHONPATH": str(Path(__file__).parent / ".claude" / "neural-system")
                        }
                    }
                }
            }
            with open(mcp_json_path, 'w') as f:
                json.dump(mcp_config, f, indent=2)
            print("📋 .mcp.json created for Claude Code integration")
        
        print("\n🎉 PROJECT SETUP COMPLETE!")
        print("🚀 Ready to use! Try:")
        print("  - Ask Claude to store a memory")
        print("  - Ask Claude to recall project information")
        print("  - Use global search across all projects")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    return True

def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="Setup neural memory system for a project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup current directory with hooks and auto-indexing
  python3 neural-setup.py
  
  # Setup specific project path
  python3 neural-setup.py /path/to/my-project
  
  # Setup without hooks
  python3 neural-setup.py --no-hooks
  
  # Setup without auto-indexing (but with other hooks)
  python3 neural-setup.py --no-auto-indexing
  
  # Minimal setup (no hooks)
  python3 neural-setup.py --no-hooks --no-auto-indexing
        """
    )
    
    parser.add_argument(
        'project_path', 
        nargs='?', 
        default='.', 
        help='Path to project directory (default: current directory)'
    )
    
    parser.add_argument(
        '--no-hooks', 
        action='store_true',
        help='Skip installing Claude Code hooks'
    )
    
    parser.add_argument(
        '--no-auto-indexing', 
        action='store_true',
        help='Disable automatic indexing after file changes'
    )
    
    parser.add_argument(
        '--version',
        action='version', 
        version='Neural Memory Setup 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Run setup
    success = asyncio.run(setup_project(
        args.project_path,
        enable_hooks=not args.no_hooks,
        enable_auto_indexing=not args.no_auto_indexing
    ))
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()