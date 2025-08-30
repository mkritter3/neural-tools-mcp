#!/usr/bin/env python3
"""
Smart Batch Indexer - Optimal Efficiency Hook
Accumulates file changes during response cycle and batches them at completion
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional

# State file for accumulating changes
STATE_FILE = "/tmp/claude-l9-indexing-state.json"
LOCK_FILE = "/tmp/claude-l9-indexing-state.lock"

class IndexingState:
    """Manages accumulated indexing state during response cycle"""
    
    def __init__(self):
        self.modified_files: Set[str] = set()
        self.critical_files: Set[str] = set()
        self.last_updated = time.time()
        
        # Critical file patterns (immediate indexing)
        self.critical_patterns = {
            '.mcp.json', 'settings.json', 'requirements.txt',
            'package.json', 'tsconfig.json', 'pyproject.toml'
        }
    
    def add_file_change(self, file_path: str, is_critical: bool = False):
        """Add a file change to the accumulation state"""
        self.modified_files.add(file_path)
        if is_critical or self._is_critical_file(file_path):
            self.critical_files.add(file_path)
        self.last_updated = time.time()
    
    def _is_critical_file(self, file_path: str) -> bool:
        """Check if file is critical and needs immediate indexing"""
        file_name = Path(file_path).name
        return any(pattern in file_name for pattern in self.critical_patterns)
    
    def to_dict(self) -> Dict:
        """Serialize state to dictionary"""
        return {
            'modified_files': list(self.modified_files),
            'critical_files': list(self.critical_files),
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndexingState':
        """Deserialize state from dictionary"""
        state = cls()
        state.modified_files = set(data.get('modified_files', []))
        state.critical_files = set(data.get('critical_files', []))
        state.last_updated = data.get('last_updated', time.time())
        return state

def read_state() -> IndexingState:
    """Safely read current indexing state"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
            return IndexingState.from_dict(data)
    except (json.JSONDecodeError, FileNotFoundError, KeyError):
        pass
    return IndexingState()

def write_state(state: IndexingState) -> bool:
    """Safely write indexing state with file locking"""
    try:
        # Simple file-based locking
        if os.path.exists(LOCK_FILE):
            return False
        
        # Create lock
        Path(LOCK_FILE).touch()
        
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state.to_dict(), f)
            return True
        finally:
            # Remove lock
            try:
                os.remove(LOCK_FILE)
            except FileNotFoundError:
                pass
    except Exception:
        return False

def clear_state():
    """Clear the indexing state after successful processing"""
    try:
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass

def suggest_mcp_indexing(files: List[str], scope: str = "modified") -> bool:
    """Suggest using MCP project_auto_index tool for batch indexing"""
    try:
        # Create a suggestion for Claude to use MCP tool
        suggestion = {
            "action": "batch_index",
            "files": files,
            "scope": scope,
            "mcp_tool": "project_auto_index",
            "message": f"💡 {len(files)} files have been modified and need indexing"
        }
        
        # Write suggestion to a marker file that Claude can check
        suggestion_file = Path("/tmp/claude-pending-index.json")
        with open(suggestion_file, 'w') as f:
            json.dump(suggestion, f, indent=2)
        
        # Output suggestion for Claude to see
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"📋 BATCH INDEXING NEEDED", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"Modified files detected: {len(files)}", file=sys.stderr)
        print(f"Suggested action: Use project_auto_index MCP tool", file=sys.stderr)
        print(f"Command: project_auto_index(scope='{scope}')", file=sys.stderr)
        print(f"Files to index:", file=sys.stderr)
        for f in files[:5]:  # Show first 5 files
            print(f"  - {Path(f).name}", file=sys.stderr)
        if len(files) > 5:
            print(f"  ... and {len(files)-5} more", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
        
        return True
            
    except Exception as e:
        print(f"❌ Failed to create indexing suggestion: {e}", file=sys.stderr)
        return False

def main():
    """Main entry points for different hook types"""
    # L9 COMPLIANT: Uses DependencyManager pattern for systematic handling
    if len(sys.argv) < 2:
        print("Usage: smart_batch_indexer.py <hook_type> [tool_name] [tool_input] [result]", file=sys.stderr)
        sys.exit(1)
    
    hook_type = sys.argv[1]
    
    if hook_type == "PostToolUse":
        # Accumulate file changes from tool usage
        if len(sys.argv) < 5:
            return
        
        tool_name = sys.argv[2]
        tool_input_json = sys.argv[3]
        result_json = sys.argv[4]
        
        # Only track file modification tools
        if tool_name not in ['Edit', 'MultiEdit', 'Write', 'NotebookEdit']:
            return
        
        try:
            tool_input = json.loads(tool_input_json)
            file_path = tool_input.get('file_path')
            
            if file_path:
                state = read_state()
                state.add_file_change(file_path)
                write_state(state)
                
                print(f"📝 Tracked change: {Path(file_path).name}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error tracking file change: {e}", file=sys.stderr)
    
    elif hook_type == "Stop":
        # Process accumulated file changes
        state = read_state()
        
        if not state.modified_files:
            print("💡 No file changes to index", file=sys.stderr)
            return
        
        files_list = list(state.modified_files)
        print(f"🚀 Batch processing {len(files_list)} changed files", file=sys.stderr)
        
        # Suggest MCP indexing (hooks should not directly call Docker)
        if suggest_mcp_indexing(files_list, "modified"):
            clear_state()
            print("✅ Indexing suggestion created - Claude can now run project_auto_index", file=sys.stderr)
        else:
            print("❌ Failed to create suggestion, state preserved for retry", file=sys.stderr)

if __name__ == "__main__":
    main()