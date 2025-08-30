#!/usr/bin/env python3
"""
Session Memory Store Hook
Stores session insights when Claude stops
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
except ImportError:
    MemorySystem = None

async def store_session_insights():
    """Store session insights when Claude session ends"""
    
    if not MemorySystem:
        print("⚠️ Memory system not available for session storage", file=sys.stderr)
        return
    
    try:
        # Basic session summary
        project_path = Path.cwd()
        session_end_time = datetime.now().isoformat()
        
        session_summary = f"Claude session ended in project: {project_path.name}"
        session_summary += f"\nTime: {session_end_time}"
        session_summary += f"\nProject path: {project_path}"
        
        # Store session info
        memory = MemorySystem()
        await memory.initialize()
        
        metadata = {
            'type': 'session_end',
            'project_name': project_path.name,
            'project_path': str(project_path),
            'end_time': session_end_time
        }
        
        await memory.store_memory(session_summary, metadata)
        print(f"💾 Session insights stored for {project_path.name}", file=sys.stderr)
        
    except Exception as e:
        print(f"❌ Failed to store session insights: {e}", file=sys.stderr)

def main():
    """Hook entry point"""
    # Run async session storage
    asyncio.run(store_session_insights())

if __name__ == "__main__":
    main()