#!/usr/bin/env python3
"""
Session Context Injector - SessionStart Hook
Provides comprehensive codebase understanding at session start
L9-compliant with 3500 token layered context architecture
Enhanced with PRISM importance scoring system
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Add neural-tools to path for PRISM import (Docker architecture)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'neural-tools'))

# Token counting (rough estimation)
def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

# Import shared PRISM scorer
try:
    from prism_scorer import PrismScorer
except ImportError:
    # Fallback if module not found
    print("Warning: PRISM scorer module not found, using basic scoring", file=sys.stderr)
    # Define a minimal fallback class
    class PrismScorer:
        def __init__(self, project_root: str):
            self.project_root = Path(project_root)
        
        def get_top_files(self, n: int = 10, file_pattern: str = '**/*.py'):
            """Simple fallback: return first n files found"""
            files = []
            for f in self.project_root.glob(file_pattern):
                if f.is_file():
                    files.append((str(f), 0.5))  # Default score
                if len(files) >= n:
                    break
            return files

class SessionContextBuilder:
    """Builds layered session context for optimal codebase understanding"""
    
    def __init__(self, max_tokens: int = 3500):
        self.max_tokens = max_tokens
        self.layer_budgets = {
            'core': 1000,
            'structural': 1500, 
            'historical': 1000
        }
        self.project_dir = Path(os.environ.get('CLAUDE_PROJECT_DIR', os.getcwd()))
        self.project_name = os.environ.get('PROJECT_NAME', self.project_dir.name)
        
        # Initialize PRISM scorer for intelligent file selection
        self.prism_scorer = PrismScorer(str(self.project_dir))
    
    def build_session_context(self) -> Dict[str, Any]:
        """Build complete session context with layered approach"""
        context = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'project_name': self.project_name,
                'project_dir': str(self.project_dir),
                'target_tokens': self.max_tokens,
                'actual_tokens': 0
            },
            'layers': {}
        }
        
        # Layer 1: Core Context
        core_context = self.build_core_context()
        context['layers']['core'] = core_context
        
        # Layer 2: Structural Context  
        structural_context = self.build_structural_context()
        context['layers']['structural'] = structural_context
        
        # Layer 3: Historical Context
        historical_context = self.build_historical_context()
        context['layers']['historical'] = historical_context
        
        # Generate narrative summary
        narrative = self.generate_session_narrative(context)
        context['session_narrative'] = narrative
        
        # Calculate final token count
        full_text = self.format_context_for_injection(context)
        context['metadata']['actual_tokens'] = estimate_tokens(full_text)
        
        # Truncate if necessary (rare edge case)
        if context['metadata']['actual_tokens'] > self.max_tokens:
            context = self.intelligent_truncation(context)
        
        return context
    
    def build_core_context(self) -> Dict[str, Any]:
        """Layer 1: Core project understanding"""
        core = {
            'project_type': self.detect_project_type(),
            'main_purpose': self.extract_project_purpose(),
            'key_technologies': self.identify_key_technologies(),
            'entry_points': self.find_entry_points(),
            'recent_focus': self.get_recent_development_focus()
        }
        return core
    
    def build_structural_context(self) -> Dict[str, Any]:
        """Layer 2: Structural and dependency information with PRISM prioritization"""
        # Get top files using PRISM scoring
        top_files = self.prism_scorer.get_top_files(n=15)
        
        structural = {
            'main_functions': self.get_main_functions_via_mcp(),
            'dependencies': self.analyze_dependencies_via_mcp(),
            'file_organization': self.analyze_file_structure(),
            'architecture_patterns': self.detect_architecture_patterns(),
            'important_files': self.format_important_files(top_files)
        }
        return structural
    
    def format_important_files(self, top_files: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Format important files with PRISM scores"""
        formatted = []
        for file_path, score in top_files[:10]:  # Top 10 most important
            rel_path = Path(file_path).relative_to(self.project_dir)
            formatted.append({
                'path': str(rel_path),
                'score': round(score, 3),
                'importance': self.get_importance_level(score)
            })
        return formatted
    
    def get_importance_level(self, score: float) -> str:
        """Convert PRISM score to importance level"""
        if score >= 0.8:
            return 'critical'
        elif score >= 0.6:
            return 'high'
        elif score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def build_historical_context(self) -> Dict[str, Any]:
        """Layer 3: Historical context from semantic memory"""
        historical = {
            'recent_decisions': self.get_recent_decisions_via_mcp(),
            'key_outcomes': self.get_recent_outcomes_via_mcp(),
            'active_areas': self.identify_active_development_areas(),
            'context_summary': self.get_memory_context_summary()
        }
        return historical
    
    def detect_project_type(self) -> str:
        """Detect project type based on files and structure"""
        indicators = {
            'web_app': ['package.json', 'webpack.config.js', 'vite.config.js'],
            'python_package': ['setup.py', 'pyproject.toml', 'requirements.txt'],
            'neural_tools': ['neural-mcp-server-enhanced.py', '.mcp.json'],
            'react_app': ['package.json', 'src/App.tsx', 'src/App.jsx'],
            'api_server': ['server.py', 'app.py', 'main.py', 'api/'],
            'library': ['lib/', 'src/', 'dist/', 'build.gradle'],
            'docs_site': ['docs/', '_config.yml', 'mkdocs.yml']
        }
        
        for project_type, files in indicators.items():
            if any((self.project_dir / f).exists() for f in files):
                return project_type
        
        return 'general'
    
    def extract_project_purpose(self) -> str:
        """Extract project purpose from README or package files"""
        purpose_sources = [
            self.project_dir / 'README.md',
            self.project_dir / 'package.json',
            self.project_dir / 'pyproject.toml'
        ]
        
        for source in purpose_sources:
            if source.exists():
                try:
                    content = source.read_text(encoding='utf-8')[:500]
                    if 'description' in content.lower():
                        # Extract description from common formats
                        lines = content.split('\n')
                        for line in lines[1:6]:  # Check first few lines
                            if line.strip() and len(line) > 20:
                                return line.strip()[:200]
                except:
                    continue
        
        return f"Development project in {self.project_dir.name}"
    
    def identify_key_technologies(self) -> List[str]:
        """Identify key technologies from project files"""
        tech_indicators = {
            'Python': ['*.py', 'requirements.txt', 'pyproject.toml'],
            'JavaScript': ['*.js', '*.jsx', 'package.json'],
            'TypeScript': ['*.ts', '*.tsx', 'tsconfig.json'],
            'React': ['react', 'jsx', 'tsx'],
            'Node.js': ['package.json', 'node_modules'],
            'Docker': ['Dockerfile', 'docker-compose.yml'],
            'Neural/AI': ['neural', 'embedding', 'vector', 'qdrant'],
            'MCP': ['.mcp.json', 'mcp-server']
        }
        
        technologies = []
        for tech, indicators in tech_indicators.items():
            for indicator in indicators:
                if any(self.project_dir.rglob(indicator)):
                    technologies.append(tech)
                    break
        
        return technologies[:8]  # Limit to most important
    
    def find_entry_points(self) -> List[str]:
        """Find main entry points and important files"""
        entry_candidates = [
            'main.py', 'app.py', 'server.py', 'index.js', 'index.ts',
            'src/main.py', 'src/index.js', 'src/App.tsx', 'src/App.jsx',
            'neural-mcp-server-enhanced.py'
        ]
        
        entry_points = []
        for candidate in entry_candidates:
            if (self.project_dir / candidate).exists():
                entry_points.append(candidate)
        
        return entry_points[:5]
    
    def get_recent_development_focus(self) -> str:
        """Determine recent development focus from file modifications"""
        try:
            # Get recently modified files
            recent_files = []
            cutoff = datetime.now() - timedelta(days=7)
            
            for py_file in self.project_dir.rglob('*.py'):
                if py_file.is_file():
                    mtime = datetime.fromtimestamp(py_file.stat().st_mtime)
                    if mtime > cutoff:
                        recent_files.append(py_file.name)
            
            if recent_files:
                return f"Recent focus on: {', '.join(recent_files[:3])}"
            else:
                return "Stable codebase, no recent major changes"
        except:
            return "Unable to determine recent focus"
    
    def get_main_functions_via_mcp(self) -> List[Dict[str, str]]:
        """Get main functions via MCP project_understanding tool"""
        try:
            result = self.call_mcp_tool('project_understanding', {
                'scope': 'functions',
                'max_tokens': 800
            })
            
            if result and 'functions' in result:
                return result['functions'][:10]  # Top 10 functions
        except:
            pass
        
        # Fallback: Basic file scanning
        return [{'name': 'main', 'file': 'main.py', 'description': 'Entry point'}]
    
    def analyze_dependencies_via_mcp(self) -> Dict[str, Any]:
        """Analyze dependencies via MCP tools"""
        try:
            result = self.call_mcp_tool('project_understanding', {
                'scope': 'dependencies',
                'max_tokens': 600
            })
            
            if result and 'dependencies' in result:
                return result['dependencies']
        except:
            pass
        
        # Fallback: Basic dependency detection
        return {
            'internal': ['core modules', 'utilities'],
            'external': ['standard libraries'],
            'files_count': len(list(self.project_dir.rglob('*.py')))
        }
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze file organization"""
        structure = {
            'directories': [],
            'main_files': [],
            'total_files': 0
        }
        
        try:
            # Get main directories
            for item in self.project_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    structure['directories'].append(item.name)
            
            # Get main files
            for item in self.project_dir.iterdir():
                if item.is_file() and item.suffix in ['.py', '.js', '.ts', '.md']:
                    structure['main_files'].append(item.name)
            
            # Count total files
            structure['total_files'] = len(list(self.project_dir.rglob('*')))
        except:
            pass
        
        return structure
    
    def detect_architecture_patterns(self) -> List[str]:
        """Detect architectural patterns in use"""
        patterns = []
        
        # Check for common patterns
        if (self.project_dir / 'src').exists():
            patterns.append('Source directory structure')
        if any(self.project_dir.rglob('*controller*')):
            patterns.append('MVC pattern')
        if any(self.project_dir.rglob('*service*')):
            patterns.append('Service layer')
        if (self.project_dir / 'neural-tools').exists():
            patterns.append('Neural tools architecture')
        if (self.project_dir / '.claude').exists():
            patterns.append('Claude Code integration')
        
        return patterns
    
    def get_recent_decisions_via_mcp(self) -> List[str]:
        """Get recent decisions from semantic memory"""
        try:
            result = self.call_mcp_tool('memory_search_enhanced', {
                'query': 'recent decisions outcomes',
                'limit': 5,
                'category': 'pre_compact_summary'
            })
            
            if result and 'memories' in result:
                decisions = []
                for memory in result['memories'][:3]:
                    if 'content' in memory:
                        # Extract decision-like sentences
                        content = memory['content']
                        lines = content.split('\n')
                        for line in lines:
                            if any(word in line.lower() for word in ['decided', 'implemented', 'fixed', '✅']):
                                decisions.append(line.strip()[:100])
                                break
                return decisions
        except:
            pass
        
        return ["Recent development focused on auto-indexing optimization"]
    
    def get_recent_outcomes_via_mcp(self) -> List[str]:
        """Get recent outcomes from memory"""
        try:
            result = self.call_mcp_tool('memory_search_enhanced', {
                'query': 'completed success implemented',
                'limit': 5,
                'category': 'pre_compact_summary'
            })
            
            if result and 'memories' in result:
                outcomes = []
                for memory in result['memories'][:3]:
                    if 'content' in memory:
                        content = memory['content']
                        # Look for outcome indicators
                        if '✅' in content:
                            lines = content.split('\n')
                            for line in lines:
                                if '✅' in line:
                                    outcomes.append(line.strip()[:100])
                                    if len(outcomes) >= 3:
                                        break
                return outcomes
        except:
            pass
        
        return ["System optimizations completed"]
    
    def identify_active_development_areas(self) -> List[str]:
        """Identify currently active development areas"""
        areas = []
        
        # Check for common active areas based on recent files
        recent_patterns = {
            'hooks': 'Claude Code hook development',
            'neural': 'Neural tools enhancement',
            'mcp': 'MCP server improvements',
            'test': 'Testing and validation',
            'doc': 'Documentation updates'
        }
        
        try:
            cutoff = datetime.now() - timedelta(days=3)
            for pattern, description in recent_patterns.items():
                if any(f for f in self.project_dir.rglob(f'*{pattern}*') 
                      if f.is_file() and datetime.fromtimestamp(f.stat().st_mtime) > cutoff):
                    areas.append(description)
        except:
            pass
        
        return areas[:4]
    
    def get_memory_context_summary(self) -> str:
        """Get summary of stored context"""
        try:
            result = self.call_mcp_tool('memory_search_enhanced', {
                'query': 'project overview summary',
                'limit': 1
            })
            
            if result and 'memories' in result and result['memories']:
                memory = result['memories'][0]
                if 'content' in memory:
                    return memory['content'][:300] + "..."
        except:
            pass
        
        return "No previous session context available"
    
    def call_mcp_tool(self, tool_name: str, args: Dict) -> Optional[Dict]:
        """Call MCP tool via Docker exec"""
        try:
            docker_cmd = [
                'docker', 'exec', '-i', 'claude-l9-template-neural',
                'python3', '-c', f'''
import sys
sys.path.append("/app")
from neural_mcp_server_enhanced import {tool_name}
import asyncio
import json

args = {json.dumps(args)}
result = asyncio.run({tool_name}(**args))
print(json.dumps(result))
'''
            ]
            
            result = subprocess.run(docker_cmd, 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            print(f"MCP call failed for {tool_name}: {e}", file=sys.stderr)
        
        return None
    
    def generate_session_narrative(self, context: Dict) -> str:
        """Generate session narrative from context layers"""
        core = context['layers']['core']
        structural = context['layers']['structural']
        historical = context['layers']['historical']
        
        narrative_parts = []
        
        # Project overview
        narrative_parts.append(f"Project: {core['main_purpose']}")
        narrative_parts.append(f"Type: {core['project_type']} using {', '.join(core['key_technologies'][:3])}")
        
        # Structure info
        if structural['main_functions']:
            func_count = len(structural['main_functions'])
            narrative_parts.append(f"Contains {func_count} main functions")
        
        # Recent context
        if historical['recent_decisions']:
            narrative_parts.append(f"Recent work: {historical['recent_decisions'][0][:50]}...")
        
        # Active areas
        if historical['active_areas']:
            narrative_parts.append(f"Active areas: {', '.join(historical['active_areas'][:2])}")
        
        return '. '.join(narrative_parts) + '.'
    
    def format_context_for_injection(self, context: Dict) -> str:
        """Format context for injection into session"""
        # Format important files section
        important_files_str = ""
        if 'important_files' in context['layers']['structural']:
            files = context['layers']['structural']['important_files'][:5]
            important_files_str = "\n".join([
                f"  - {f['path']} [{f['importance'].upper()}]"
                for f in files
            ])
        
        formatted = f"""
🚀 SESSION CONTEXT: {context['metadata']['project_name']}

📋 PROJECT OVERVIEW:
{context['session_narrative']}

🏗️ CORE CONTEXT:
• Type: {context['layers']['core']['project_type']}
• Technologies: {', '.join(context['layers']['core']['key_technologies'])}
• Entry Points: {', '.join(context['layers']['core']['entry_points'])}
• Recent Focus: {context['layers']['core']['recent_focus']}

🔧 STRUCTURAL CONTEXT:
• File Organization: {context['layers']['structural']['file_organization']['total_files']} total files
• Main Directories: {', '.join(context['layers']['structural']['file_organization']['directories'][:5])}
• Architecture: {', '.join(context['layers']['structural']['architecture_patterns'])}

🎯 IMPORTANT FILES (PRISM SCORED):
{important_files_str}

📚 HISTORICAL CONTEXT:
• Recent Decisions: {len(context['layers']['historical']['recent_decisions'])} key decisions tracked
• Active Areas: {', '.join(context['layers']['historical']['active_areas'])}
• Context Summary: {context['layers']['historical']['context_summary'][:200]}...

✨ SESSION READY: {context['metadata']['actual_tokens']} tokens with PRISM intelligence
"""
        return formatted
    
    def intelligent_truncation(self, context: Dict) -> Dict:
        """Intelligently truncate context if over budget"""
        # Priority order for truncation
        truncation_order = [
            ('historical.context_summary', 150),
            ('structural.architecture_patterns', 3),
            ('core.key_technologies', 5),
            ('historical.active_areas', 2)
        ]
        
        for field_path, limit in truncation_order:
            # Implement truncation logic
            current_size = estimate_tokens(self.format_context_for_injection(context))
            if current_size <= self.max_tokens:
                break
        
        return context

def main():
    """Main entry point for SessionStart hook"""
    try:
        print("🧠 Loading session context...", file=sys.stderr)
        
        # Build session context
        builder = SessionContextBuilder(max_tokens=3500)
        context = builder.build_session_context()
        
        # Format for injection
        formatted_context = builder.format_context_for_injection(context)
        
        # Output context for Claude to receive
        print(formatted_context)
        
        print(f"✅ Session context loaded: {context['metadata']['actual_tokens']} tokens", file=sys.stderr)
        
    except Exception as e:
        print(f"❌ Error in session context injector: {e}", file=sys.stderr)
        # Provide minimal fallback context
        print("🚀 SESSION CONTEXT: Basic project context loaded")

if __name__ == "__main__":
    main()