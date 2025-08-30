#!/usr/bin/env python3
"""
Neural Auto Indexer Hook
Automatically indexes code changes after Edit/MultiEdit/Write operations
"""

import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

# Add neural-system to path
sys.path.append(str(Path(__file__).parent.parent / 'neural-system'))

try:
    from memory_system import MemorySystem
except ImportError:
    # Fallback if memory system not available
    MemorySystem = None

async def process_tool_use(tool_name: str, tool_input: Dict[str, Any], result: Any):
    """Process tool use for auto-indexing"""
    
    # Only process file modification tools
    if tool_name not in ['Edit', 'MultiEdit', 'Write']:
        return
    
    if not MemorySystem:
        print("⚠️ Memory system not available for auto-indexing", file=sys.stderr)
        return
    
    try:
        # Extract file path from tool input
        file_path = None
        if 'file_path' in tool_input:
            file_path = tool_input['file_path']
        elif 'edits' in tool_input and tool_input['edits']:
            # Handle MultiEdit case - we'll just note the file was modified
            file_path = tool_input['file_path']
        
        if not file_path:
            return
        
        # Create memory entry for the change
        change_summary = f"Modified file: {Path(file_path).name}"
        
        # Add more context based on tool type
        if tool_name == 'Edit':
            if 'old_string' in tool_input and 'new_string' in tool_input:
                old_preview = tool_input['old_string'][:100] + "..." if len(tool_input['old_string']) > 100 else tool_input['old_string']
                new_preview = tool_input['new_string'][:100] + "..." if len(tool_input['new_string']) > 100 else tool_input['new_string']
                change_summary += f"\nChanged: {old_preview} → {new_preview}"
        elif tool_name == 'MultiEdit':
            edit_count = len(tool_input.get('edits', []))
            change_summary += f"\n{edit_count} edits made to file"
        elif tool_name == 'Write':
            change_summary += "\nFile created or completely rewritten"
        
        # Store in memory system
        memory = MemorySystem()
        await memory.initialize()
        
        metadata = {
            'type': 'code_change',
            'tool_used': tool_name,
            'file_path': file_path,
            'auto_indexed': True
        }
        
        await memory.store_memory(change_summary, metadata)
        print(f"📝 Auto-indexed: {Path(file_path).name}", file=sys.stderr)
        
    except Exception as e:
        print(f"❌ Auto-indexing failed: {e}", file=sys.stderr)

def main():
    """Hook entry point"""
    if len(sys.argv) < 4:
        print("Usage: neural_auto_indexer.py <tool_name> <tool_input_json> <result_json>", file=sys.stderr)
        sys.exit(1)
    
    tool_name = sys.argv[1]
    tool_input = json.loads(sys.argv[2])
    result = json.loads(sys.argv[3]) if sys.argv[3] != 'null' else None
    
    # Run async processing
    asyncio.run(process_tool_use(tool_name, tool_input, result))

if __name__ == "__main__":
    main()