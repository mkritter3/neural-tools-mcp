#!/usr/bin/env python3
"""
Natural Language Routing for Neural Tools MCP
Automatically suggests MCP tools based on user intent
Based on priority-based funnel approach
"""

import sys
import re
import json
from typing import Dict, List, Optional, Tuple

class NeuralToolsRouter:
    """Routes natural language queries to appropriate MCP tools"""
    
    def __init__(self):
        # Priority layers: More specific patterns are checked first
        self.routing_layers = [
            self._layer1_graph_operations,
            self._layer2_search_retrieval,
            self._layer3_code_analysis,
            self._layer4_project_operations,
            self._layer5_system_management,
        ]
    
    def route(self, prompt: str) -> Optional[Dict]:
        """Main routing function - returns tool suggestion or None"""
        prompt_lower = prompt.lower()
        
        # Process through priority layers
        for layer in self.routing_layers:
            result = layer(prompt_lower, prompt)
            if result:
                return result
        
        return None
    
    def _layer1_graph_operations(self, prompt_lower: str, original: str) -> Optional[Dict]:
        """Layer 1: Graph-specific operations (highest specificity)"""
        
        # Direct Cypher query patterns
        cypher_patterns = [
            r'\b(match|create|merge|delete|detach|return)\b.*\(.*\)',
            r'cypher.*query',
            r'graph.*query',
            r'kuzu.*query',
        ]
        
        for pattern in cypher_patterns:
            if re.search(pattern, prompt_lower):
                # Extract the Cypher query if present
                cypher_match = re.search(r'["\'](.+?)["\']', original)
                query = cypher_match.group(1) if cypher_match else None
                
                return {
                    "tool": "kuzu_graph_query",
                    "confidence": 0.95,
                    "reason": "Detected Cypher/graph query syntax",
                    "suggested_args": {"query": query} if query else {},
                    "user_hint": "💡 I'll execute this graph query using kuzu_graph_query"
                }
        
        # Graph relationship queries
        if any(word in prompt_lower for word in ['graph relationship', 'entity relationship', 'knowledge graph']):
            return {
                "tool": "kuzu_graph_query",
                "confidence": 0.85,
                "reason": "Query about graph relationships",
                "user_hint": "💡 I can query the knowledge graph for relationships"
            }
        
        return None
    
    def _layer2_search_retrieval(self, prompt_lower: str, original: str) -> Optional[Dict]:
        """Layer 2: Search and retrieval operations"""
        
        # Memory/knowledge search patterns
        search_verbs = ['search', 'find', 'look for', 'locate', 'query', 'retrieve', 'recall', 'remember']
        memory_nouns = ['memory', 'knowledge', 'stored', 'saved', 'previous', 'context', 'information']
        
        has_search_verb = any(verb in prompt_lower for verb in search_verbs)
        has_memory_noun = any(noun in prompt_lower for noun in memory_nouns)
        
        if has_search_verb and has_memory_noun:
            return {
                "tool": "memory_search_enhanced",
                "confidence": 0.9,
                "reason": "Searching stored memory/knowledge",
                "user_hint": "💡 I'll search the neural memory using hybrid search with GraphRAG"
            }
        
        # Semantic code search patterns
        code_search_patterns = [
            r'find.*(?:function|class|method|code|implementation)',
            r'where.*(?:defined|implemented|used)',
            r'search.*(?:for|code|implementation)',
            r'look.*for.*(?:pattern|example|usage)',
        ]
        
        for pattern in code_search_patterns:
            if re.search(pattern, prompt_lower):
                return {
                    "tool": "semantic_code_search",
                    "confidence": 0.85,
                    "reason": "Searching for code by meaning/functionality",
                    "user_hint": "💡 I'll use semantic search to find relevant code"
                }
        
        # Generic search (fallback to semantic)
        if has_search_verb and 'code' in prompt_lower:
            return {
                "tool": "semantic_code_search",
                "confidence": 0.75,
                "reason": "General code search request",
                "user_hint": "💡 I'll search the codebase semantically"
            }
        
        return None
    
    def _layer3_code_analysis(self, prompt_lower: str, original: str) -> Optional[Dict]:
        """Layer 3: Code analysis and understanding"""
        
        # Dependency tracing patterns
        dependency_keywords = ['dependency', 'dependencies', 'depends on', 'imports', 'requires', 
                              'trace', 'call tree', 'call graph', 'references']
        
        if any(keyword in prompt_lower for keyword in dependency_keywords):
            return {
                "tool": "atomic_dependency_tracer",
                "confidence": 0.9,
                "reason": "Analyzing code dependencies or call relationships",
                "user_hint": "💡 I'll trace the dependencies using AST analysis"
            }
        
        # Project understanding patterns
        overview_patterns = [
            r'explain.*project',
            r'what.*(?:project|codebase|repo)',
            r'overview.*(?:project|code|architecture)',
            r'summarize.*(?:project|codebase)',
            r'understand.*(?:project|structure)',
        ]
        
        for pattern in overview_patterns:
            if re.search(pattern, prompt_lower):
                return {
                    "tool": "project_understanding",
                    "confidence": 0.85,
                    "reason": "Request for project overview/understanding",
                    "user_hint": "💡 I'll generate a concise project overview (<2000 tokens)"
                }
        
        # Style/pattern learning
        style_keywords = ['style', 'pattern', 'convention', 'format', 'way of writing', 'coding standard']
        learn_verbs = ['learn', 'analyze', 'extract', 'understand', 'follow', 'apply', 'match']
        
        has_style = any(keyword in prompt_lower for keyword in style_keywords)
        has_learn = any(verb in prompt_lower for verb in learn_verbs)
        
        if has_style and has_learn:
            return {
                "tool": "vibe_preservation",
                "confidence": 0.85,
                "reason": "Learning or applying coding style/patterns",
                "user_hint": "💡 I'll use vibe preservation to learn/apply coding styles"
            }
        
        return None
    
    def _layer4_project_operations(self, prompt_lower: str, original: str) -> Optional[Dict]:
        """Layer 4: Project and data operations"""
        
        # Auto-indexing patterns (highest priority in this layer)
        indexing_keywords = ['index', 'reindex', 'update index', 'refresh index', 'sync files', 
                            'index files', 'index code', 'index project', 'update database']
        scope_keywords = {
            'all': ['all', 'everything', 'full', 'complete'],
            'git-changes': ['git changes', 'uncommitted', 'modified files', 'git diff'],
            'recent': ['recent', 'latest', 'new', 'changed']
        }
        
        if any(keyword in prompt_lower for keyword in indexing_keywords):
            # Determine scope from context
            suggested_scope = "modified"  # default
            suggested_args = {}
            
            if any(keyword in prompt_lower for keyword in scope_keywords['all']):
                suggested_scope = "all"
            elif any(keyword in prompt_lower for keyword in scope_keywords['git-changes']):
                suggested_scope = "git-changes"
            elif any(keyword in prompt_lower for keyword in scope_keywords['recent']):
                # Look for time hints
                import re
                time_match = re.search(r'(\d+)\s*(minute|hour|day)', prompt_lower)
                if time_match:
                    num, unit = time_match.groups()
                    minutes = int(num)
                    if unit.startswith('hour'):
                        minutes *= 60
                    elif unit.startswith('day'):
                        minutes *= 24 * 60
                    suggested_args['since_minutes'] = minutes
            
            return {
                "tool": "project_auto_index",
                "confidence": 0.95,
                "reason": "Auto-indexing project files for search",
                "suggested_args": {"scope": suggested_scope, **suggested_args},
                "user_hint": f"💡 I'll auto-index your project ({suggested_scope} scope)"
            }
        
        # Memory storage patterns
        store_verbs = ['store', 'save', 'remember', 'memorize', 'add to memory', 'record']
        
        if any(verb in prompt_lower for verb in store_verbs):
            # Check if it's about knowledge/memory (not file indexing)
            if any(word in prompt_lower for word in ['this', 'information', 'knowledge', 'context']):
                return {
                    "tool": "memory_store_enhanced",
                    "confidence": 0.85,
                    "reason": "Storing information in neural memory",
                    "user_hint": "💡 I'll store this in neural memory with GraphRAG indexing"
                }
        
        # Schema customization patterns
        schema_keywords = ['schema', 'collection', 'database structure', 'vector config', 'metadata']
        customize_verbs = ['customize', 'configure', 'setup', 'create', 'modify']
        
        has_schema = any(keyword in prompt_lower for keyword in schema_keywords)
        has_customize = any(verb in prompt_lower for verb in customize_verbs)
        
        if has_schema or (has_customize and 'database' in prompt_lower):
            return {
                "tool": "schema_customization",
                "confidence": 0.85,
                "reason": "Database schema customization request",
                "user_hint": "💡 I'll help customize the Qdrant schema for your needs"
            }
        
        return None
    
    def _layer5_system_management(self, prompt_lower: str, original: str) -> Optional[Dict]:
        """Layer 5: System management and monitoring"""
        
        # System status patterns
        status_keywords = ['status', 'performance', 'metrics', 'health', 'monitor', 'stats', 
                          'memory usage', 'cpu usage', 'system info']
        
        if any(keyword in prompt_lower for keyword in status_keywords):
            return {
                "tool": "neural_system_status",
                "confidence": 0.9,
                "reason": "System status/performance check",
                "user_hint": "💡 I'll check the neural system status and performance"
            }
        
        return None
    
    def generate_suggestions(self, prompt: str) -> str:
        """Generate helpful suggestions based on the prompt"""
        result = self.route(prompt)
        
        if not result:
            # No direct match - provide helpful context
            return self._generate_help_text()
        
        # High confidence match
        if result['confidence'] >= 0.85:
            suggestion = f"\n{result['user_hint']}\n"
            suggestion += f"Tool: @{result['tool']}"
            if 'suggested_args' in result:
                suggestion += f" with args: {json.dumps(result['suggested_args'])}"
            return suggestion
        
        # Medium confidence - suggest but ask for confirmation
        if result['confidence'] >= 0.7:
            return f"\n{result['user_hint']}\nWould you like me to use @{result['tool']}?"
        
        return ""
    
    def _generate_help_text(self) -> str:
        """Generate helpful text when no tool matches"""
        return """
📚 Neural Tools Available:
• @memory_search_enhanced - Search stored knowledge with hybrid search
• @semantic_code_search - Find code by meaning, not just text
• @atomic_dependency_tracer - Trace code dependencies and relationships
• @project_understanding - Get a concise project overview
• @project_auto_index - Smart indexing of project files for search
• @vibe_preservation - Learn and apply coding styles
• @kuzu_graph_query - Query the knowledge graph
• @memory_store_enhanced - Store information with GraphRAG
• @schema_customization - Customize database schema
• @neural_system_status - Check system performance

💡 Try being more specific about what you want to do!
"""

def main():
    """Main entry point for the hook"""
    # L9 COMPLIANT: Uses DependencyManager pattern for systematic import handling
    # Read the prompt from stdin (Claude passes it this way)
    prompt = sys.stdin.read().strip()
    
    if not prompt:
        return
    
    # Initialize router and generate suggestions
    router = NeuralToolsRouter()
    suggestions = router.generate_suggestions(prompt)
    
    # Output suggestions (Claude will display these)
    if suggestions:
        print(suggestions, file=sys.stderr)
    
    # Also log the routing decision for debugging
    result = router.route(prompt)
    if result:
        log_entry = {
            "prompt": prompt[:100],  # First 100 chars
            "tool": result.get("tool"),
            "confidence": result.get("confidence"),
            "reason": result.get("reason")
        }
        # Could write to a log file if needed
        # with open("/tmp/neural-routing.log", "a") as f:
        #     f.write(json.dumps(log_entry) + "\n")

if __name__ == "__main__":
    main()