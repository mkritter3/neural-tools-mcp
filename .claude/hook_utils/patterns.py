#!/usr/bin/env python3
"""
L9 Hook Patterns - Shared Pattern Extraction Logic
Centralizes regex patterns and extraction logic used across hooks
"""

import re
from typing import Dict, List, Set, Pattern, Optional, Tuple, Any
import json
from datetime import datetime


class ConversationPatterns:
    """Compiled regex patterns for consistent conversation analysis across hooks"""
    
    def __init__(self):
        # Compile patterns for better performance
        self.file_operations = [
            re.compile(r'(Edit|MultiEdit|Write|Read)\([^)]*([^)]+\.(py|js|ts|tsx|jsx|json|yaml|yml|toml|md))[^)]*\)', re.IGNORECASE),
            re.compile(r'file_path["\\s]*:["\\s]*([^"]+\.(py|js|ts|tsx|jsx|json|yaml|yml|toml|md))', re.IGNORECASE),
            re.compile(r'([a-zA-Z0-9_/.-]+\.(py|js|ts|tsx|jsx|json|yaml|yml|toml|md)):\d+', re.IGNORECASE),
        ]
        
        self.decision_patterns = [
            re.compile(r'(decided|chosen|selected|implemented|fixed|resolved)', re.IGNORECASE),
            re.compile(r'(strategy|approach|solution|method|pattern)', re.IGNORECASE),
            re.compile(r'(because|since|due to|reason|rationale)', re.IGNORECASE),
        ]
        
        self.outcome_patterns = [
            re.compile(r'(✅|completed|successful|working|fixed|resolved)', re.IGNORECASE),
            re.compile(r'(❌|failed|error|issue|problem)', re.IGNORECASE),
            re.compile(r'(⚠️|warning|caution|note|important)', re.IGNORECASE),
        ]
        
        self.technical_patterns = [
            re.compile(r'\b(MCP|hook|Docker|Qdrant|embedding|neural|L9|batch|indexing)\b', re.IGNORECASE),
            re.compile(r'\b(API|REST|GraphQL|database|vector|semantic|hybrid)\b', re.IGNORECASE),
            re.compile(r'\b(async|await|promise|callback|thread|process)\b', re.IGNORECASE),
        ]
        
        self.tool_invocation_pattern = re.compile(r'<invoke name="([^"]+)">', re.IGNORECASE)
        
        self.user_intent_patterns = [
            re.compile(r'user:\s*"([^"]+)"', re.IGNORECASE),
            re.compile(r'user:\s*(.+)', re.IGNORECASE),
            re.compile(r'I want to (.+)', re.IGNORECASE),
            re.compile(r'I need to (.+)', re.IGNORECASE),
            re.compile(r'Can you (.+)', re.IGNORECASE),
            re.compile(r'Help me (.+)', re.IGNORECASE)
        ]


# Global instance for reuse across hooks
conversation_patterns = ConversationPatterns()


def extract_file_references(text: str, max_files: int = 50) -> Set[str]:
    """Extract unique file references from conversation text"""
    files = set()
    
    for pattern in conversation_patterns.file_operations:
        matches = pattern.findall(text)
        for match in matches:
            if isinstance(match, tuple):
                # Extract file path from tuple
                for item in match:
                    if any(ext in item for ext in ['.py', '.js', '.ts', '.json', '.md', '.yaml', '.yml']):
                        # Clean up the file path
                        clean_path = item.strip('"\'()[]{}')
                        if clean_path and len(clean_path) < 200:  # Reasonable path length
                            files.add(clean_path)
            else:
                clean_path = match.strip('"\'()[]{}')
                if clean_path and len(clean_path) < 200:
                    files.add(clean_path)
    
    # Convert to sorted list for consistency, limit to max_files
    return set(sorted(list(files))[:max_files])


def extract_key_decisions(text: str, max_decisions: int = 10) -> List[str]:
    """Extract key decisions and architectural choices from conversation"""
    decisions = []
    sentences = text.split('\n')
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:  # Skip very short lines
            continue
        
        # Look for decision patterns
        for pattern in conversation_patterns.decision_patterns:
            if pattern.search(sentence):
                if len(sentence) < 300:  # Keep it reasonable
                    # Clean up the sentence
                    clean_sentence = re.sub(r'^\d+→', '', sentence)  # Remove line numbers
                    clean_sentence = clean_sentence.strip()
                    if clean_sentence and clean_sentence not in decisions:
                        decisions.append(clean_sentence)
                break
        
        if len(decisions) >= max_decisions:
            break
    
    return decisions


def extract_technical_context(text: str) -> Dict[str, List[str]]:
    """Extract technical context including technologies, operations, and outcomes"""
    context = {
        'technologies': set(),
        'operations': set(), 
        'outcomes': set()
    }
    
    # Extract technical terms
    for pattern in conversation_patterns.technical_patterns:
        matches = pattern.findall(text)
        context['technologies'].update(matches)
    
    # Extract tool invocations
    tool_matches = conversation_patterns.tool_invocation_pattern.findall(text)
    context['operations'].update(tool_matches)
    
    # Extract outcomes (lines with success/error indicators)
    outcome_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if any(pattern.search(line) for pattern in conversation_patterns.outcome_patterns):
            if len(line) < 120:  # Reasonable length
                clean_line = re.sub(r'^\d+→', '', line).strip()  # Remove line numbers
                if clean_line:
                    outcome_lines.append(clean_line)
    
    context['outcomes'] = set(outcome_lines)
    
    # Convert sets to sorted lists and limit size
    return {
        'technologies': sorted(list(context['technologies']))[:15],
        'operations': sorted(list(context['operations']))[:15], 
        'outcomes': sorted(list(context['outcomes']))[:10]
    }


def classify_session_type(text: str) -> str:
    """Classify conversation type based on content patterns"""
    patterns = {
        'debugging': ['debug', 'error', 'fix', 'bug', 'issue', 'problem'],
        'feature_development': ['implement', 'feature', 'add', 'create', 'build', 'develop'],
        'refactoring': ['refactor', 'improve', 'optimize', 'cleanup', 'reorganize', 'restructure'],
        'configuration': ['config', 'setup', 'install', 'deploy', 'configure'],
        'analysis': ['analyze', 'review', 'understand', 'explore', 'investigate', 'examine'],
        'documentation': ['document', 'explain', 'describe', 'comment', 'readme'],
        'testing': ['test', 'validate', 'verify', 'check', 'unit test', 'integration'],
        'deployment': ['deploy', 'release', 'publish', 'launch', 'production']
    }
    
    text_lower = text.lower()
    scores = {}
    
    for session_type, keywords in patterns.items():
        score = sum(text_lower.count(keyword) for keyword in keywords)
        scores[session_type] = score
    
    # Return the type with highest score, default to 'development'
    return max(scores, key=scores.get) if any(scores.values()) else 'development'


def extract_user_intent(text: str, max_intents: int = 3) -> List[str]:
    """Extract user's main intents from early conversation"""
    intents = []
    early_lines = text.split('\n')[:30]  # Look at first 30 lines
    text_sample = '\n'.join(early_lines)
    
    for pattern in conversation_patterns.user_intent_patterns:
        matches = pattern.finditer(text_sample)
        for match in matches:
            intent = match.group(1).strip()
            if len(intent) < 150 and intent not in intents:  # Reasonable length, no duplicates
                intents.append(intent)
            
            if len(intents) >= max_intents:
                break
        
        if len(intents) >= max_intents:
            break
    
    return intents


def extract_implementation_details(text: str, max_details: int = 10) -> List[Dict[str, str]]:
    """Extract specific implementation details and code changes"""
    implementations = []
    
    # Pattern for code/file changes
    implementation_patterns = [
        (re.compile(r'(created|added|implemented|wrote|modified|updated)\s+([^.]+\.(py|js|ts|json))', re.IGNORECASE), 'file_change'),
        (re.compile(r'(function|class|method)\s+(\w+)', re.IGNORECASE), 'code_element'),
        (re.compile(r'(installed|configured|setup)\s+([^.]+)', re.IGNORECASE), 'setup'),
        (re.compile(r'(fixed|resolved|corrected)\s+([^.]+)', re.IGNORECASE), 'fix')
    ]
    
    for pattern, impl_type in implementation_patterns:
        matches = pattern.finditer(text)
        for match in matches:
            detail = match.group(0).strip()
            if len(detail) < 100:  # Reasonable length
                implementations.append({
                    'type': impl_type,
                    'description': detail,
                    'match': match.group(2) if len(match.groups()) >= 2 else match.group(1)
                })
            
            if len(implementations) >= max_details:
                break
        
        if len(implementations) >= max_details:
            break
    
    return implementations


def generate_mcp_suggestions(context: Dict[str, Any], max_suggestions: int = 5) -> List[Dict[str, str]]:
    """Generate MCP tool usage suggestions based on conversation context"""
    suggestions = []
    
    # Analyze context for suggestion opportunities
    technical_context = context.get('technical_context', {})
    file_references = context.get('file_references', [])
    decisions = context.get('key_decisions', [])
    
    # Memory search suggestions
    if len(decisions) > 2:
        suggestions.append({
            'tool': 'memory_search_enhanced',
            'query': f"decisions about {' '.join(technical_context.get('technologies', [])[:3])}",
            'priority': 'high',
            'reason': 'Multiple decisions made - search for related past decisions'
        })
    
    # Code search suggestions
    if len(file_references) > 3:
        main_files = [f.split('/')[-1] for f in file_references[:3]]
        suggestions.append({
            'tool': 'code_search_enhanced', 
            'query': f"files similar to {' '.join(main_files)}",
            'priority': 'medium',
            'reason': 'Multiple files modified - find related code patterns'
        })
    
    # Memory storage suggestions
    if context.get('session_type') in ['feature_development', 'refactoring']:
        suggestions.append({
            'tool': 'memory_store_enhanced',
            'query': f"Session outcome: {context.get('session_type')} in {', '.join(technical_context.get('technologies', [])[:2])}",
            'priority': 'medium',
            'reason': 'Significant development work - store for future reference'
        })
    
    # Graph analysis suggestions  
    if 'architecture' in ' '.join(decisions).lower() or 'design' in ' '.join(decisions).lower():
        suggestions.append({
            'tool': 'graph_analysis_enhanced',
            'query': 'architecture relationships and design patterns',
            'priority': 'high', 
            'reason': 'Architectural decisions made - analyze system relationships'
        })
    
    # Hybrid search suggestions
    if len(technical_context.get('technologies', [])) > 3:
        tech_list = ', '.join(technical_context['technologies'][:4])
        suggestions.append({
            'tool': 'hybrid_search_enhanced',
            'query': f"best practices for {tech_list}",
            'priority': 'low',
            'reason': 'Multiple technologies used - find cross-technology patterns'
        })
    
    return suggestions[:max_suggestions]


def create_conversation_summary(
    conversation_text: str, 
    max_tokens: int = 2000,
    include_mcp_suggestions: bool = True
) -> Dict[str, Any]:
    """
    Create a comprehensive conversation summary using all pattern extraction functions
    This is the main function that ties everything together
    """
    
    # Extract all components
    file_refs = extract_file_references(conversation_text)
    decisions = extract_key_decisions(conversation_text)
    tech_context = extract_technical_context(conversation_text)
    session_type = classify_session_type(conversation_text)
    user_intents = extract_user_intent(conversation_text)
    implementations = extract_implementation_details(conversation_text)
    
    # Build comprehensive summary
    summary = {
        'session_id': datetime.now().strftime('%x%x%x')[:12],  # Simple session ID
        'timestamp': datetime.now().isoformat(),
        'session_type': session_type,
        'user_intents': user_intents,
        'file_references': sorted(list(file_refs)),
        'key_decisions': decisions,
        'technical_context': tech_context,
        'implementations': implementations,
        'conversation_stats': {
            'total_lines': len(conversation_text.split('\n')),
            'total_characters': len(conversation_text),
            'estimated_tokens': len(conversation_text) // 4
        }
    }
    
    # Add MCP suggestions if requested
    if include_mcp_suggestions:
        summary['mcp_suggestions'] = generate_mcp_suggestions(summary)
    
    # Calculate token usage
    summary_text = json.dumps(summary, indent=2)
    summary['meta'] = {
        'summary_tokens': len(summary_text) // 4,
        'compression_ratio': len(summary_text) / len(conversation_text),
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    # Truncate if over token limit
    if summary['meta']['summary_tokens'] > max_tokens:
        summary = _truncate_summary(summary, max_tokens)
    
    return summary


def _truncate_summary(summary: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """Intelligently truncate summary to fit within token limit"""
    
    # Priority order for truncation (least to most important)
    truncation_order = [
        ('implementations', 5),
        ('mcp_suggestions', 3), 
        ('key_decisions', 5),
        ('file_references', 10),
        ('user_intents', 2),
        ('technical_context.outcomes', 5),
        ('technical_context.operations', 8),
        ('technical_context.technologies', 8)
    ]
    
    for field_path, limit in truncation_order:
        if '.' in field_path:
            parent, child = field_path.split('.')
            if parent in summary and child in summary[parent]:
                if len(summary[parent][child]) > limit:
                    summary[parent][child] = summary[parent][child][:limit]
        else:
            if field_path in summary and len(summary[field_path]) > limit:
                summary[field_path] = summary[field_path][:limit]
        
        # Recalculate tokens
        current_tokens = len(json.dumps(summary, indent=2)) // 4
        if current_tokens <= max_tokens:
            break
    
    # Update metadata
    summary['meta']['summary_tokens'] = len(json.dumps(summary, indent=2)) // 4
    summary['meta']['truncated'] = True
    
    return summary