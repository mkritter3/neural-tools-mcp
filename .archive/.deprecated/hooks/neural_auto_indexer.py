#!/usr/bin/env python3
"""
Neural Auto-Indexer Hook
Automatically indexes code changes after Edit/MultiEdit/Write operations
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Add neural-system to path
sys.path.append(os.path.dirname(__file__))

from memory_system import MemorySystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class NeuralAutoIndexer:
    """Automatically index code changes for memory system"""
    
    def __init__(self):
        self.memory_system = MemorySystem()
    
    async def process_file_change(self, tool_data: dict):
        """Process a file change and index relevant content"""
        try:
            file_path = tool_data.get('tool_input', {}).get('file_path')
            if not file_path:
                return
            
            # Skip certain file types and directories
            skip_patterns = [
                '.git/', '__pycache__/', 'node_modules/', '.env',
                '.log', '.tmp', '.cache', 'dist/', 'build/'
            ]
            
            if any(pattern in file_path for pattern in skip_patterns):
                return
            
            # Only index relevant file types
            relevant_extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.json', 
                '.yml', '.yaml', '.toml', '.sql', '.sh', '.go', 
                '.java', '.cpp', '.c', '.h', '.rs'
            ]
            
            if not any(file_path.endswith(ext) for ext in relevant_extensions):
                return
            
            # Read the file content
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip empty files or very small changes
                if len(content.strip()) < 10:
                    return
                
                # Extract change type and create meaningful description
                tool_name = tool_data.get('tool_name', 'unknown')
                old_string = tool_data.get('tool_input', {}).get('old_string', '')
                new_string = tool_data.get('tool_input', {}).get('new_string', '')
                
                # Create context-rich description
                relative_path = os.path.relpath(file_path)
                description = f"File modification in {relative_path}"
                
                if tool_name == 'Write':
                    description = f"New file created: {relative_path}"
                elif tool_name in ['Edit', 'MultiEdit'] and old_string and new_string:
                    change_size = abs(len(new_string) - len(old_string))
                    if change_size > 100:
                        description = f"Significant changes to {relative_path} (+{change_size} chars)"
                    else:
                        description = f"Minor edits to {relative_path}"
                
                # Store in memory with rich metadata
                metadata = {
                    'file_path': file_path,
                    'relative_path': relative_path,
                    'tool_name': tool_name,
                    'change_type': 'creation' if tool_name == 'Write' else 'modification',
                    'file_size': len(content),
                    'line_count': content.count('\n') + 1,
                    'language': self._detect_language(file_path),
                    'auto_indexed': True,
                    'indexed_at': datetime.now().isoformat()
                }
                
                # Index the change
                await self.memory_system.store_memory(
                    content=f"{description}\n\n{content[:2000]}...",  # First 2000 chars
                    metadata=metadata
                )
                
                logger.info(f"🔍 Auto-indexed: {relative_path}")
                
        except Exception as e:
            logger.error(f"❌ Auto-indexing failed: {e}")
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp', '.c': 'c', '.h': 'c',
            '.rs': 'rust',
            '.md': 'markdown',
            '.json': 'json',
            '.yml': 'yaml', '.yaml': 'yaml',
            '.sql': 'sql',
            '.sh': 'bash'
        }
        return language_map.get(ext, 'text')

async def main():
    """Main hook entry point"""
    try:
        # Read tool data from stdin (Claude Code hook format)
        input_data = json.load(sys.stdin)
        
        # Initialize and process
        indexer = NeuralAutoIndexer()
        await indexer.process_file_change(input_data)
        
    except json.JSONDecodeError:
        # Silent fail for non-JSON input (some hooks may not provide JSON)
        pass
    except Exception as e:
        # Log errors but don't fail the hook
        logger.error(f"Hook error: {e}")
        print(f"⚠️ Neural auto-indexer encountered an error: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())