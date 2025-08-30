#!/usr/bin/env python3
"""
Context Injector Hook
Injects relevant context before user prompts
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

# Add neural-system to path
sys.path.append(str(Path(__file__).parent.parent / 'neural-system'))

try:
    from memory_system import MemorySystem
except ImportError:
    MemorySystem = None

async def inject_context(user_message: str) -> List[str]:
    """Inject relevant context based on user message"""
    
    if not MemorySystem or not user_message.strip():
        return []
    
    try:
        memory = MemorySystem()
        await memory.initialize()
        
        # Search for relevant memories (keep small to avoid overwhelming)
        relevant_memories = await memory.search_memories(
            query=user_message,
            limit=3  # Keep context injection lightweight
        )
        
        if not relevant_memories:
            return []
        
        context_lines = ["## 🧠 Relevant Context:"]
        
        for i, mem in enumerate(relevant_memories, 1):
            content = mem['payload']['content'][:200]  # Limit context length
            if len(mem['payload']['content']) > 200:
                content += "..."
            
            context_lines.append(f"{i}. {content}")
        
        context_lines.append("")  # Empty line separator
        return context_lines
        
    except Exception as e:
        print(f"❌ Context injection failed: {e}", file=sys.stderr)
        return []

def main():
    """Hook entry point"""
    if len(sys.argv) < 2:
        print("Usage: context_injector.py <user_message>", file=sys.stderr)
        sys.exit(1)
    
    user_message = sys.argv[1]
    
    # Run async context injection
    context = asyncio.run(inject_context(user_message))
    
    # Output context to stdout (Claude Code will prepend to prompt)
    for line in context:
        print(line)

if __name__ == "__main__":
    main()