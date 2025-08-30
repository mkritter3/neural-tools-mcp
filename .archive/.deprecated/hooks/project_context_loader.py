#!/usr/bin/env python3
"""
Project Context Loader Hook
Loads and displays project context when Claude Code starts a new session
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add neural-system to path
sys.path.append(os.path.dirname(__file__))

from config_manager import get_config
from memory_system import MemorySystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ProjectContextLoader:
    """Load project context on session start"""
    
    def __init__(self):
        self.config = get_config()
        self.memory_system = MemorySystem()
    
    async def load_project_context(self, session_data: dict):
        """Load and display project context"""
        try:
            # Get project information
            project_name = self.config.get_project_name()
            project_path = Path.cwd()
            
            # Initialize memory system
            await self.memory_system.initialize()
            
            # Get project statistics
            stats = await self.memory_system.get_stats()
            
            # Get recent project activity (last 10 memories)
            recent_memories = await self.memory_system.search_memories(
                query="recent changes configuration architecture",
                limit=5,
                similarity_threshold=0.3
            )
            
            # Detect project type and key files
            project_info = self._analyze_project_structure()
            
            # Format and display context
            context_info = self._format_project_context(
                project_name=project_name,
                project_path=project_path,
                stats=stats,
                recent_memories=recent_memories,
                project_info=project_info
            )
            
            # Output as system reminder for Claude
            print("<system-reminder>")
            print("🚀 **Neural Memory System Session Started**")
            print("")
            print(context_info)
            print("</system-reminder>")
            
            logger.info(f"✅ Loaded context for project: {project_name}")
            
        except Exception as e:
            logger.error(f"❌ Context loading failed: {e}")
            # Provide basic context even if memory system fails
            print("<system-reminder>")
            print(f"🚀 **Session Started** - Project: {self.config.get_project_name()}")
            print(f"⚠️  Neural memory system unavailable: {e}")
            print("</system-reminder>")
    
    def _analyze_project_structure(self) -> Dict[str, Any]:
        """Analyze project structure and detect type"""
        project_root = Path.cwd()
        
        # Check for common project files
        key_files = {
            'package.json': 'Node.js/JavaScript project',
            'requirements.txt': 'Python project',
            'Cargo.toml': 'Rust project',
            'go.mod': 'Go project',
            'pom.xml': 'Java/Maven project',
            'build.gradle': 'Java/Gradle project',
            'composer.json': 'PHP project',
            'Gemfile': 'Ruby project',
            '.claude/settings.json': 'Claude Code project'
        }
        
        detected_files = []
        project_types = []
        
        for filename, description in key_files.items():
            if (project_root / filename).exists():
                detected_files.append(filename)
                project_types.append(description)
        
        # Count source files
        source_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.rs', '.cpp', '.c']
        source_files = []
        
        for ext in source_extensions:
            files = list(project_root.glob(f"**/*{ext}"))
            # Filter out common ignore patterns
            filtered_files = [
                f for f in files 
                if not any(part.startswith('.') or part in ['node_modules', '__pycache__', 'dist', 'build', 'target'] 
                          for part in f.parts)
            ]
            source_files.extend(filtered_files)
        
        return {
            'project_types': project_types,
            'key_files': detected_files,
            'source_file_count': len(source_files),
            'has_docker': (project_root / 'Dockerfile').exists() or (project_root / 'docker-compose.yml').exists(),
            'has_tests': any(
                (project_root / pattern).exists() for pattern in 
                ['tests/', 'test/', '__tests__/', 'spec/']
            ) or len(list(project_root.glob('**/test_*.py'))) > 0,
            'has_ci': any(
                (project_root / pattern).exists() for pattern in
                ['.github/', '.gitlab-ci.yml', '.circleci/', 'Jenkinsfile']
            )
        }
    
    def _format_project_context(self, project_name: str, project_path: Path, 
                              stats: Dict, recent_memories: list, 
                              project_info: Dict) -> str:
        """Format project context for display"""
        lines = []
        
        # Project overview
        lines.append(f"**📁 Project: {project_name}**")
        lines.append(f"Path: `{project_path}`")
        lines.append("")
        
        # Project type detection
        if project_info['project_types']:
            lines.append("**🔍 Detected Project Types:**")
            for ptype in project_info['project_types'][:3]:  # Show max 3 types
                lines.append(f"• {ptype}")
            lines.append("")
        
        # Project characteristics
        lines.append("**📊 Project Overview:**")
        lines.append(f"• Source files: {project_info['source_file_count']}")
        lines.append(f"• Docker support: {'Yes' if project_info['has_docker'] else 'No'}")
        lines.append(f"• Tests present: {'Yes' if project_info['has_tests'] else 'No'}")
        lines.append(f"• CI/CD setup: {'Yes' if project_info['has_ci'] else 'No'}")
        lines.append("")
        
        # Memory system status
        lines.append("**🧠 Neural Memory Status:**")
        lines.append(f"• Total memories: {stats.get('total_memories', 0)}")
        lines.append(f"• System status: {stats.get('status', 'unknown')}")
        if stats.get('total_memories', 0) > 0:
            lines.append(f"• Last updated: {stats.get('last_updated', 'unknown')}")
        lines.append("")
        
        # Recent activity
        if recent_memories:
            lines.append("**🕒 Recent Activity:**")
            for memory in recent_memories[:3]:  # Show max 3 recent items
                payload = memory.get('payload', {})
                content = payload.get('content', 'No content')
                file_path = payload.get('relative_path', '')
                
                # Create short description
                preview = content[:100].replace('\n', ' ')
                if len(content) > 100:
                    preview += "..."
                
                if file_path:
                    lines.append(f"• `{file_path}`: {preview}")
                else:
                    lines.append(f"• {preview}")
            
            lines.append("")
        
        # Available tools
        lines.append("**🛠️ Available Neural Tools:**")
        lines.append("• Memory search and storage")
        lines.append("• Code indexing and retrieval")
        lines.append("• Cross-project context when needed")
        lines.append("• Automatic change tracking")
        
        return "\n".join(lines)

async def main():
    """Main hook entry point"""
    try:
        # Read session data from stdin (Claude Code hook format)
        input_data = json.load(sys.stdin) if sys.stdin.isatty() == False else {}
        
        # Initialize and load context
        loader = ProjectContextLoader()
        await loader.load_project_context(input_data)
        
    except json.JSONDecodeError:
        # Handle non-JSON input gracefully
        loader = ProjectContextLoader()
        await loader.load_project_context({})
    except Exception as e:
        # Log errors but don't fail the hook
        logger.error(f"Hook error: {e}")
        # Provide minimal context
        try:
            config = get_config()
            print("<system-reminder>")
            print(f"🚀 **Session Started** - Project: {config.get_project_name()}")
            print("⚠️  Neural memory system initialization failed")
            print("</system-reminder>")
        except:
            print("<system-reminder>")
            print("🚀 **Session Started** - Neural memory system unavailable")
            print("</system-reminder>")

if __name__ == "__main__":
    asyncio.run(main())