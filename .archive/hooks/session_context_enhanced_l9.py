#!/usr/bin/env python3
"""
L9 Enhanced Session Context - SessionStart Hook
Combines rich codebase context with intelligent MCP tool suggestions
Full L9 compliance with systematic architecture and native Claude integration
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# L9 COMPLIANT: Pure package import - zero path manipulation
from hook_utils import BaseHook


class EnhancedSessionContextInjector(BaseHook):
    """L9-compliant session context with MCP integration suggestions"""
    
    def __init__(self):
        super().__init__(max_tokens=3500, hook_name="EnhancedSessionContextInjector")
        
        # Context layers with token budgets
        self.layer_budgets = {
            'project_overview': 800,
            'mcp_suggestions': 1200,
            'important_files': 600,
            'recent_activity': 500,
            'conversation_summary': 400
        }
    
    def execute(self) -> Dict[str, Any]:
        """Main execution: build enhanced context with MCP suggestions"""
        try:
            # Get PRISM scorer through DependencyManager
            prism_class = self.dependency_manager.get_prism_scorer()
            self.prism_scorer = prism_class(str(self.project_dir))
            
            # Build comprehensive context
            context = self._build_enhanced_context()
            
            # Generate intelligent narrative with MCP suggestions
            narrative = self._generate_enhanced_narrative(context)
            
            # Log completion
            tokens_used = context['metadata']['actual_tokens']
            self.log_execution(f"Enhanced session context: {tokens_used} tokens")
            
            return {
                "status": "success",
                "content": narrative,
                "tokens_used": tokens_used,
                "context_data": context
            }
            
        except Exception as e:
            return self.handle_execution_error(e)
    
    def _build_enhanced_context(self) -> Dict[str, Any]:
        """Build enhanced context with MCP integration points"""
        context = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'project_name': self.project_dir.name,
                'hook_version': 'L9-Enhanced-2025',
                'target_tokens': self.max_tokens,
                'actual_tokens': 0
            },
            'layers': {}
        }
        
        # Layer 1: Project Overview
        context['layers']['project_overview'] = self._build_project_overview()
        
        # Layer 2: MCP Tool Suggestions (Core Innovation)
        context['layers']['mcp_suggestions'] = self._build_mcp_suggestions()
        
        # Layer 3: Important Files (PRISM-scored)
        context['layers']['important_files'] = self._build_important_files()
        
        # Layer 4: Recent Activity Analysis
        context['layers']['recent_activity'] = self._build_recent_activity()
        
        # Layer 5: Previous Session Summary
        context['layers']['conversation_summary'] = self._build_conversation_summary()
        
        # Calculate total tokens and optimize if needed
        from utilities import estimate_tokens
        full_text = str(context)
        context['metadata']['actual_tokens'] = estimate_tokens(full_text)
        
        if context['metadata']['actual_tokens'] > self.max_tokens:
            context = self._intelligent_optimization(context)
        
        return context
    
    def _build_project_overview(self) -> Dict[str, Any]:
        """Build core project understanding"""
        try:
            from utilities import get_project_metadata
            metadata = get_project_metadata(self.project_dir)
            
            # Detect project architecture
            architecture = self._analyze_project_architecture()
            
            return {
                "metadata": metadata,
                "architecture": architecture,
                "layer_tokens": self.estimate_content_tokens(str(metadata) + str(architecture))
            }
            
        except Exception as e:
            self.log_execution(f"Project overview build failed: {e}", "warning")
            return {"error": str(e), "layer_tokens": 0}
    
    def _build_mcp_suggestions(self) -> Dict[str, Any]:
        """Core innovation: Intelligent MCP tool suggestions"""
        try:
            suggestions = {
                'priority_queries': [],
                'contextual_searches': [],
                'project_analysis': [],
                'memory_operations': []
            }
            
            # Analyze git changes for semantic search suggestions
            git_suggestions = self._analyze_git_for_mcp_suggestions()
            suggestions['contextual_searches'].extend(git_suggestions)
            
            # Detect architecture patterns for project understanding
            arch_suggestions = self._analyze_architecture_for_mcp_suggestions()
            suggestions['project_analysis'].extend(arch_suggestions)
            
            # Check for existing knowledge to build upon
            memory_suggestions = self._suggest_memory_operations()
            suggestions['memory_operations'].extend(memory_suggestions)
            
            # Priority queries based on project state
            priority_suggestions = self._determine_priority_queries()
            suggestions['priority_queries'].extend(priority_suggestions)
            
            return {
                "suggestions": suggestions,
                "total_suggested": sum(len(v) for v in suggestions.values()),
                "layer_tokens": self.estimate_content_tokens(str(suggestions))
            }
            
        except Exception as e:
            self.log_execution(f"MCP suggestions build failed: {e}", "warning")
            return {"error": str(e), "layer_tokens": 0}
    
    def _analyze_git_for_mcp_suggestions(self) -> List[Dict[str, str]]:
        """Analyze git changes to suggest semantic searches"""
        suggestions = []
        
        try:
            # Get recent changes
            result = subprocess.run(
                ['git', 'log', '--oneline', '--since=7.days.ago', '--name-only'],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                recent_files = [line for line in lines if '.' in line and not line.startswith(' ')]
                
                if recent_files:
                    # Suggest semantic search on recent files
                    file_sample = ' '.join(recent_files[:5])
                    suggestions.append({
                        'tool': 'memory_search_enhanced',
                        'query': f'recent work on: {file_sample}',
                        'purpose': 'Find related past work on these files',
                        'priority': 'high'
                    })
                    
                    # Suggest project understanding for recent changes
                    suggestions.append({
                        'tool': 'project_understanding',
                        'scope': 'recent_changes',
                        'purpose': 'Understand recent development patterns',
                        'priority': 'medium'
                    })
                        
        except Exception as e:
            self.log_execution(f"Git analysis failed: {e}", "warning")
        
        return suggestions
    
    def _analyze_architecture_for_mcp_suggestions(self) -> List[Dict[str, str]]:
        """Analyze project architecture for relevant MCP queries"""
        suggestions = []
        
        try:
            # Check for neural-tools directory (L9 specific)
            if (self.project_dir / 'neural-tools').exists():
                suggestions.append({
                    'tool': 'neo4j_semantic_graph_search',
                    'query': 'neural tools architecture dependencies',
                    'purpose': 'Map neural tools component relationships',
                    'priority': 'high'
                })
                
                suggestions.append({
                    'tool': 'neural_system_status',
                    'purpose': 'Check L9 neural system health',
                    'priority': 'medium'
                })
            
            # Check for Docker setup
            if (self.project_dir / 'docker-compose.yml').exists() or list(self.project_dir.glob('**/docker-compose*.yml')):
                suggestions.append({
                    'tool': 'semantic_code_search',
                    'query': 'docker configuration issues troubleshooting',
                    'purpose': 'Find Docker-related solutions and patterns',
                    'priority': 'low'
                })
            
            # Check for test directories
            test_dirs = list(self.project_dir.glob('**/test*')) + list(self.project_dir.glob('**/*test*'))
            if test_dirs:
                suggestions.append({
                    'tool': 'project_understanding',
                    'focus': 'testing_patterns',
                    'purpose': 'Understand testing strategy and coverage',
                    'priority': 'low'
                })
                        
        except Exception as e:
            self.log_execution(f"Architecture analysis failed: {e}", "warning")
        
        return suggestions
    
    def _suggest_memory_operations(self) -> List[Dict[str, str]]:
        """Suggest memory operations for knowledge continuity"""
        suggestions = []
        
        # Always suggest checking for previous context
        suggestions.append({
            'tool': 'memory_search_enhanced',
            'query': f'{self.project_dir.name} recent session context',
            'purpose': 'Restore context from previous sessions',
            'priority': 'high'
        })
        
        # Suggest storing this session's insights
        suggestions.append({
            'tool': 'memory_store_enhanced',
            'content': 'session_start_context',
            'purpose': 'Store session context for future reference',
            'priority': 'low'
        })
        
        return suggestions
    
    def _determine_priority_queries(self) -> List[Dict[str, str]]:
        """Determine highest priority MCP queries based on project state"""
        suggestions = []
        
        try:
            # Check if there are uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  cwd=self.project_dir, capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout.strip():
                suggestions.append({
                    'tool': 'memory_search_enhanced',
                    'query': 'uncommitted changes work in progress',
                    'purpose': 'Find context for current uncommitted work',
                    'priority': 'high'
                })
            
            # Check for error logs or issues
            log_files = list(self.project_dir.glob('**/*.log')) + list(self.project_dir.glob('**/logs/**'))
            if log_files:
                suggestions.append({
                    'tool': 'semantic_code_search',
                    'query': 'error handling debugging solutions',
                    'purpose': 'Find debugging patterns and solutions',
                    'priority': 'medium'
                })
                        
        except Exception as e:
            self.log_execution(f"Priority analysis failed: {e}", "warning")
        
        return suggestions
    
    def _build_important_files(self) -> Dict[str, Any]:
        """Build PRISM-scored important files list"""
        try:
            top_files = self.prism_scorer.get_top_files(10, '**/*.py')
            
            return {
                "files": [{"file": f[0], "score": f[1]} for f in top_files],
                "total_analyzed": len(top_files),
                "layer_tokens": self.estimate_content_tokens(str(top_files))
            }
            
        except Exception as e:
            self.log_execution(f"Important files build failed: {e}", "warning")
            return {"error": str(e), "layer_tokens": 0}
    
    def _build_recent_activity(self) -> Dict[str, Any]:
        """Build recent activity analysis"""
        try:
            recent_files = []
            
            # Get recently modified files
            try:
                import time
                one_week_ago = time.time() - (7 * 24 * 3600)
                
                for file_path in self.project_dir.rglob('*.py'):
                    if file_path.stat().st_mtime > one_week_ago:
                        recent_files.append({
                            'file': str(file_path.relative_to(self.project_dir)),
                            'days_ago': int((time.time() - file_path.stat().st_mtime) / (24 * 3600))
                        })
                        
                recent_files.sort(key=lambda x: x['days_ago'])
                        
            except Exception:
                recent_files = [{"error": "Recent file analysis unavailable"}]
            
            return {
                "recent_files": recent_files[:15],
                "analysis_period": "7 days",
                "layer_tokens": self.estimate_content_tokens(str(recent_files))
            }
            
        except Exception as e:
            self.log_execution(f"Recent activity build failed: {e}", "warning")
            return {"error": str(e), "layer_tokens": 0}
    
    def _build_conversation_summary(self) -> Dict[str, Any]:
        """Build previous conversation summary if available"""
        try:
            # Check for Claude Code session data (theoretical integration)
            summary_data = {
                "previous_session": "No previous session context available",
                "note": "Claude Code handles session continuity natively",
                "integration": "This hook complements Claude's native session management"
            }
            
            return {
                "summary": summary_data,
                "layer_tokens": self.estimate_content_tokens(str(summary_data))
            }
            
        except Exception as e:
            self.log_execution(f"Conversation summary build failed: {e}", "warning")
            return {"error": str(e), "layer_tokens": 0}
    
    def _generate_enhanced_narrative(self, context: Dict[str, Any]) -> str:
        """Generate enhanced narrative with MCP integration guidance"""
        try:
            project_name = context['metadata']['project_name']
            tokens_used = context['metadata']['actual_tokens']
            
            # Extract key information
            project_layer = context['layers'].get('project_overview', {})
            mcp_layer = context['layers'].get('mcp_suggestions', {})
            files_layer = context['layers'].get('important_files', {})
            activity_layer = context['layers'].get('recent_activity', {})
            
            # Build comprehensive narrative
            narrative_parts = [
                f"🚀 SESSION CONTEXT: {project_name}",
                "",
                "📋 PROJECT OVERVIEW:",
                self._format_project_overview(project_layer),
                "",
                "🔧 STRUCTURAL CONTEXT:",
                self._format_structural_context(files_layer, activity_layer),
                "",
                "🎯 IMPORTANT FILES (PRISM SCORED):",
                self._format_important_files(files_layer),
                "",
                "🤖 RECOMMENDED MCP QUERIES:",
                self._format_mcp_suggestions(mcp_layer),
                "",
                "💡 L9 NEURAL INTEGRATION:",
                self._format_neural_integration_notes(),
                "",
                f"✨ SESSION READY: {tokens_used} tokens with PRISM intelligence"
            ]
            
            return "\n".join(narrative_parts)
            
        except Exception as e:
            return f"Enhanced session context generated ({context.get('metadata', {}).get('actual_tokens', 0)} tokens)"
    
    def _format_project_overview(self, layer: Dict) -> str:
        """Format project overview section"""
        if 'error' in layer:
            return f"• Project analysis: {layer['error']}"
        
        metadata = layer.get('metadata', {})
        architecture = layer.get('architecture', {})
        
        lines = []
        
        if metadata:
            languages = metadata.get('main_languages', ['Unknown'])
            lines.append(f"• Languages: {', '.join(languages)}")
            
        if architecture:
            project_type = architecture.get('type', 'Unknown')
            lines.append(f"• Type: {project_type}")
            
            key_components = architecture.get('key_components', [])
            if key_components:
                lines.append(f"• Key Components: {', '.join(key_components[:3])}")
        
        return '\n'.join(lines) if lines else "• Project analysis in progress"
    
    def _format_structural_context(self, files_layer: Dict, activity_layer: Dict) -> str:
        """Format structural context section"""
        lines = []
        
        # File organization
        total_files = len(files_layer.get('files', []))
        if total_files > 0:
            lines.append(f"• File Organization: {total_files} important files identified")
        
        # Recent activity
        recent_files = activity_layer.get('recent_files', [])
        if recent_files and not any('error' in rf for rf in recent_files):
            active_count = len([rf for rf in recent_files if rf.get('days_ago', 0) <= 3])
            lines.append(f"• Recent Activity: {active_count} files modified in last 3 days")
        
        return '\n'.join(lines) if lines else "• Structural analysis in progress"
    
    def _format_important_files(self, layer: Dict) -> str:
        """Format important files section"""
        if 'error' in layer:
            return f"  - Error: {layer['error']}"
        
        files = layer.get('files', [])
        if not files:
            return "  - No important files identified"
        
        lines = []
        for file_info in files[:5]:  # Top 5 files
            score = file_info.get('score', 0)
            level = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
            file_path = file_info.get('file', 'unknown')
            file_name = Path(file_path).name if file_path != 'unknown' else 'unknown'
            lines.append(f"  - {file_name} [{level}]")
        
        return '\n'.join(lines)
    
    def _format_mcp_suggestions(self, layer: Dict) -> str:
        """Format MCP suggestions - Core innovation"""
        if 'error' in layer:
            return f"• Error generating suggestions: {layer['error']}"
        
        suggestions = layer.get('suggestions', {})
        if not suggestions:
            return "• No MCP suggestions available"
        
        lines = []
        
        # Priority queries first
        priority_queries = suggestions.get('priority_queries', [])
        for query in priority_queries:
            if query.get('priority') == 'high':
                lines.append(f"• PRIORITY: Use `{query['tool']}` → {query['purpose']}")
        
        # Contextual searches
        contextual = suggestions.get('contextual_searches', [])
        for search in contextual[:2]:  # Top 2
            lines.append(f"• SEARCH: `{search['tool']}` with query: \"{search['query']}\"")
        
        # Project analysis
        project_analysis = suggestions.get('project_analysis', [])
        for analysis in project_analysis[:2]:  # Top 2
            lines.append(f"• ANALYZE: `{analysis['tool']}` → {analysis['purpose']}")
        
        # Memory operations
        memory_ops = suggestions.get('memory_operations', [])
        for memory in memory_ops:
            if memory.get('priority') == 'high':
                lines.append(f"• MEMORY: `{memory['tool']}` → {memory['purpose']}")
        
        if not lines:
            lines.append("• MCP tools available - awaiting context analysis")
        
        return '\n'.join(lines)
    
    def _format_neural_integration_notes(self) -> str:
        """Format L9 Neural integration notes"""
        lines = [
            "• L9 Neural Tools: Available via MCP protocol",
            "• Semantic Memory: Use MCP suggestions above for optimal context",
            "• Session Continuity: Native Claude Code + MCP hybrid approach"
        ]
        
        # Check if neural-tools is active
        if (self.project_dir / 'neural-tools').exists():
            lines.append("• Neural Tools Detected: Enhanced analysis capabilities active")
        
        return '\n'.join(lines)
    
    def _analyze_project_architecture(self) -> Dict[str, Any]:
        """Analyze project architecture and type"""
        architecture = {
            "type": "development",
            "key_components": [],
            "patterns": []
        }
        
        try:
            # Check for specific architectures
            if (self.project_dir / 'neural-tools').exists():
                architecture["type"] = "neural_tools"
                architecture["key_components"].extend(["Neural Tools", "MCP", "Docker"])
            
            if list(self.project_dir.glob('**/docker-compose*.yml')):
                architecture["key_components"].append("Docker")
                architecture["patterns"].append("containerized")
            
            if (self.project_dir / '.claude').exists():
                architecture["key_components"].append("Claude Code")
                architecture["patterns"].append("ai_assisted")
            
            # Language detection
            if list(self.project_dir.glob('**/*.py')):
                architecture["key_components"].append("Python")
                
            if list(self.project_dir.glob('**/*.js')) or list(self.project_dir.glob('**/*.ts')):
                architecture["key_components"].append("JavaScript/TypeScript")
                
        except Exception as e:
            self.log_execution(f"Architecture analysis failed: {e}", "warning")
        
        return architecture
    
    def _intelligent_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently optimize context to fit token budget"""
        # Priority order: keep MCP suggestions, reduce other layers
        if context['metadata']['actual_tokens'] > self.max_tokens:
            # Reduce recent activity first
            if 'recent_activity' in context['layers']:
                activity_layer = context['layers']['recent_activity']
                if 'recent_files' in activity_layer:
                    activity_layer['recent_files'] = activity_layer['recent_files'][:5]  # Reduce to 5
            
            # Reduce important files if still too large
            if 'important_files' in context['layers']:
                files_layer = context['layers']['important_files']
                if 'files' in files_layer:
                    files_layer['files'] = files_layer['files'][:5]  # Top 5 only
            
            context['metadata']['optimization_applied'] = "reduced_secondary_layers"
        
        return context


# Main execution
def main():
    """Main execution function"""
    injector = EnhancedSessionContextInjector()
    result = injector.run()
    
    if result.get('status') == 'success':
        print(result['content'])
        return 0
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())