#!/usr/bin/env python3
"""
L9 Session Context Injector - SessionStart Hook
Provides comprehensive codebase understanding at session start
Fully L9-compliant with systematic dependency management and shared utilities
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# L9 COMPLIANT: Pure package import - zero path manipulation
# Hooks must be run from directory where hook_utils is directly importable
from hook_utils import BaseHook


class SessionContextInjector(BaseHook):
    """L9-compliant session context injector with systematic architecture"""
    
    def __init__(self):
        super().__init__(max_tokens=3500, hook_name="SessionContextInjector")
        
        # Layer budgets for structured context
        self.layer_budgets = {
            'core': 1000,
            'structural': 1500, 
            'historical': 1000
        }
    
    def execute(self) -> Dict[str, Any]:
        """Main execution: build layered session context"""
        try:
            # Get PRISM scorer through DependencyManager (L9 systematic import handling)
            prism_class = self.dependency_manager.get_prism_scorer()
            self.prism_scorer = prism_class(str(self.project_dir))
            
            # Build complete session context
            context = self._build_session_context()
            
            # Generate user-facing output
            narrative = self._generate_session_narrative(context)
            
            # Log completion
            tokens_used = context['metadata']['actual_tokens']
            self.log_execution(f"Session context generated: {tokens_used} tokens")
            
            return {
                "status": "success",
                "content": narrative,
                "tokens_used": tokens_used,
                "context_data": context
            }
            
        except Exception as e:
            return self.handle_execution_error(e)
    
    def _build_session_context(self) -> Dict[str, Any]:
        """Build complete session context with layered approach"""
        context = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'project_name': self.project_dir.name,
                'project_dir': str(self.project_dir),
                'target_tokens': self.max_tokens,
                'actual_tokens': 0
            },
            'layers': {}
        }
        
        # Layer 1: Core Context
        context['layers']['core'] = self._build_core_context()
        
        # Layer 2: Structural Context  
        context['layers']['structural'] = self._build_structural_context()
        
        # Layer 3: Historical Context
        context['layers']['historical'] = self._build_historical_context()
        
        # L9 COMPLIANT: Use shared utilities instead of local functions
        from utilities import format_context
        full_text = format_context(context, self.max_tokens)
        context['metadata']['actual_tokens'] = self.estimate_content_tokens(full_text)
        
        # Intelligent truncation if needed
        if context['metadata']['actual_tokens'] > self.max_tokens:
            context = self._intelligent_truncation(context)
        
        return context
    
    def _build_core_context(self) -> Dict[str, Any]:
        """Build core project context"""
        try:
            # L9 COMPLIANT: Use shared utilities, no local implementations
            from utilities import get_project_metadata
            metadata = get_project_metadata(self.project_dir)
            
            # Get important files via PRISM
            top_files = self.prism_scorer.get_top_files(5, '**/*.py')
            
            # Build technology stack understanding
            tech_stack = self._analyze_technology_stack()
            
            return {
                "project_metadata": metadata,
                "important_files": [{"file": f[0], "score": f[1]} for f in top_files],
                "technology_stack": tech_stack,
                "layer_tokens": self.estimate_content_tokens(str(metadata) + str(top_files))
            }
            
        except Exception as e:
            self.log_execution(f"Core context build failed: {e}", "warning")
            return {"error": str(e), "layer_tokens": 0}
    
    def _build_structural_context(self) -> Dict[str, Any]:
        """Build structural understanding of codebase"""
        try:
            # L9 COMPLIANT: Use shared utilities, no local implementations
            from utilities import find_files_by_pattern
            
            py_files = find_files_by_pattern(self.project_dir, '*.py')
            js_files = find_files_by_pattern(self.project_dir, '*.js')
            ts_files = find_files_by_pattern(self.project_dir, '*.ts')
            
            # Analyze directory structure
            directories = set()
            for file_path in py_files + js_files + ts_files:
                directories.add(file_path.parent.name)
            
            structure = {
                "total_files": len(py_files) + len(js_files) + len(ts_files),
                "python_files": len(py_files),
                "js_ts_files": len(js_files) + len(ts_files),
                "main_directories": sorted(list(directories))[:10],
                "layer_tokens": 0
            }
            
            structure["layer_tokens"] = self.estimate_content_tokens(str(structure))
            return structure
            
        except Exception as e:
            self.log_execution(f"Structural context build failed: {e}", "warning")
            return {"error": str(e), "layer_tokens": 0}
    
    def _build_historical_context(self) -> Dict[str, Any]:
        """Build historical context from recent changes"""
        try:
            # Simple git analysis (no complex parsing)
            recent_files = []
            
            # Get recently modified files
            try:
                import time
                one_week_ago = time.time() - (7 * 24 * 3600)
                
                for file_path in self.project_dir.rglob('*.py'):
                    if file_path.stat().st_mtime > one_week_ago:
                        recent_files.append(str(file_path.relative_to(self.project_dir)))
                        
            except Exception:
                recent_files = ["Recent file analysis unavailable"]
            
            history = {
                "recent_files": recent_files[:10],
                "analysis_period": "7 days",
                "layer_tokens": 0
            }
            
            history["layer_tokens"] = self.estimate_content_tokens(str(history))
            return history
            
        except Exception as e:
            self.log_execution(f"Historical context build failed: {e}", "warning")
            return {"error": str(e), "layer_tokens": 0}
    
    def _generate_session_narrative(self, context: Dict[str, Any]) -> str:
        """Generate human-readable session narrative"""
        try:
            project_name = context['metadata']['project_name']
            tokens_used = context['metadata']['actual_tokens']
            
            # Core information
            core_layer = context['layers'].get('core', {})
            important_files = core_layer.get('important_files', [])
            tech_stack = core_layer.get('technology_stack', {})
            project_metadata = core_layer.get('project_metadata', {})
            
            # Structural information  
            structural = context['layers'].get('structural', {})
            total_files = structural.get('total_files', 0)
            directories = structural.get('main_directories', [])
            
            # Build narrative
            narrative_parts = [
                f"🚀 SESSION CONTEXT: {project_name}",
                "",
                "📋 PROJECT OVERVIEW:",
                f"Project: Development project in {project_name}. "
                f"Contains {total_files} files. "
                f"Main technologies: {', '.join(project_metadata.get('main_languages', []))}.",
                "",
                "🎯 IMPORTANT FILES (PRISM SCORED):"
            ]
            
            # Add important files
            for file_info in important_files[:5]:
                score = file_info.get('score', 0)
                level = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
                file_name = Path(file_info['file']).name
                narrative_parts.append(f"  - {file_name} [{level}]")
            
            narrative_parts.extend([
                "",
                "🔧 STRUCTURAL CONTEXT:",
                f"• File Organization: {total_files} total files",
                f"• Main Directories: {', '.join(directories[:5])}",
                "",
                f"✨ SESSION READY: {tokens_used} tokens with PRISM intelligence"
            ])
            
            return "\n".join(narrative_parts)
            
        except Exception as e:
            return f"Session context generated with {context.get('metadata', {}).get('actual_tokens', 0)} tokens"
    
    def _analyze_technology_stack(self) -> Dict[str, Any]:
        """Analyze technology stack from project files"""
        stack = {
            "languages": [],
            "frameworks": [],
            "tools": []
        }
        
        try:
            # Check for Python
            if list(self.project_dir.glob("**/*.py")):
                stack["languages"].append("Python")
            
            # Check for JavaScript/TypeScript
            if list(self.project_dir.glob("**/*.js")) or list(self.project_dir.glob("**/*.ts")):
                stack["languages"].append("JavaScript/TypeScript")
            
            # Check for Docker
            if (self.project_dir / "Dockerfile").exists():
                stack["tools"].append("Docker")
            
            # Check for MCP
            if list(self.project_dir.glob("**/neural-mcp-server*.py")):
                stack["frameworks"].append("MCP")
                
        except Exception:
            pass
        
        return stack
    
    def _intelligent_truncation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently truncate context to fit token budget"""
        # Simple truncation strategy - remove historical layer first
        if 'historical' in context['layers']:
            del context['layers']['historical']
            context['metadata']['truncation_applied'] = "historical_layer_removed"
        
        return context


# Main execution
def main():
    """Main execution function"""
    injector = SessionContextInjector()
    result = injector.run()
    
    if result.get('status') == 'success':
        print(result['content'])
        print("🧠 Loading session context...")
        print(f"✅ Session context loaded: {result['tokens_used']} tokens")
    else:
        print(f"❌ Error in session context injector: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())