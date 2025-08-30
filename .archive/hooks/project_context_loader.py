#!/usr/bin/env python3
"""
Project Context Loader Hook
Loads project context at session start
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add neural-system to path
sys.path.append(str(Path(__file__).parent.parent / 'neural-system'))

try:
    from memory_system import MemorySystem
    from config_manager import get_config
except ImportError:
    MemorySystem = None
    get_config = None

async def load_project_context():
    """Load and display project context at session start"""
    
    try:
        # Basic project info
        project_path = Path.cwd()
        print(f"📁 Project: {project_path.name}")
        print(f"🎯 Path: {project_path}")
        
        if not MemorySystem:
            print("⚠️ Neural memory system not available")
            return
        
        # Get memory system stats
        memory = MemorySystem()
        await memory.initialize()
        
        stats = await memory.get_stats()
        print(f"🧠 Memories: {stats.get('total_memories', 0)}")
        print(f"📊 Status: {stats.get('status', 'unknown')}")
        
        # Show recent memories (last 3)
        recent_memories = await memory.search_memories(
            query="recent project activity",
            limit=3
        )
        
        if recent_memories:
            print("\n📝 Recent Activity:")
            for i, mem in enumerate(recent_memories, 1):
                content = mem['payload']['content'][:100]
                if len(mem['payload']['content']) > 100:
                    content += "..."
                timestamp = mem['payload'].get('timestamp', 'unknown')
                print(f"  {i}. {content} ({timestamp})")
        
        print(f"\n✅ Neural memory system ready for {project_path.name}")
        print("💡 Try: Store memories, search code, or query across all projects")
        
    except Exception as e:
        print(f"❌ Failed to load project context: {e}", file=sys.stderr)

def main():
    """Hook entry point"""
    # Run async context loading
    asyncio.run(load_project_context())

if __name__ == "__main__":
    main()