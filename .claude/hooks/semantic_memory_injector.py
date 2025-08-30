#!/usr/bin/env python3
"""
Semantic Memory Injector - PreCompact Hook
Captures and stores semantic summary of conversation before compaction
"""

import os
import sys
import json
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime

# Token counting (rough estimation)
def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters)"""
    return len(text) // 4

class SemanticMemoryExtractor:
    """Extracts semantic information from conversation context"""
    
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.conversation_patterns = {
            'file_operations': [
                r'(Edit|MultiEdit|Write|Read)\([^)]*([^)]+\.(py|js|ts|tsx|jsx|json|yaml|yml|toml|md))[^)]*\)',
                r'file_path["\s]*:["\s]*([^"]+\.(py|js|ts|tsx|jsx|json|yaml|yml|toml|md))',
                r'([a-zA-Z0-9_/.-]+\.(py|js|ts|tsx|jsx|json|yaml|yml|toml|md)):\d+',
            ],
            'decisions': [
                r'(decided|chosen|selected|implemented|fixed|resolved)',
                r'(strategy|approach|solution|method|pattern)',
                r'(because|since|due to|reason|rationale)',
            ],
            'outcomes': [
                r'(✅|completed|successful|working|fixed|resolved)',
                r'(❌|failed|error|issue|problem)',
                r'(⚠️|warning|caution|note|important)',
            ],
            'technical_terms': [
                r'\b(MCP|hook|Docker|Qdrant|embedding|neural|L9|batch|indexing)\b',
                r'\b(API|REST|GraphQL|database|vector|semantic|hybrid)\b',
                r'\b(async|await|promise|callback|thread|process)\b',
            ]
        }
    
    def extract_file_references(self, text: str) -> Set[str]:
        """Extract unique file references from conversation"""
        files = set()
        for pattern in self.conversation_patterns['file_operations']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Extract file path from tuple
                    for item in match:
                        if any(ext in item for ext in ['.py', '.js', '.ts', '.json', '.md', '.yaml', '.yml']):
                            files.add(item)
                else:
                    files.add(match)
        return files
    
    def extract_key_decisions(self, text: str) -> List[str]:
        """Extract key decisions and outcomes"""
        decisions = []
        sentences = text.split('\n')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Look for decision patterns
            for pattern in self.conversation_patterns['decisions']:
                if re.search(pattern, sentence, re.IGNORECASE):
                    if len(sentence) < 200:  # Keep it concise
                        decisions.append(sentence)
                    break
        
        return list(set(decisions))  # Remove duplicates
    
    def extract_technical_context(self, text: str) -> Dict[str, List[str]]:
        """Extract technical context and terms"""
        context = {
            'technologies': set(),
            'operations': set(),
            'outcomes': set()
        }
        
        # Extract technical terms
        for pattern in self.conversation_patterns['technical_terms']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            context['technologies'].update(matches)
        
        # Extract operations (tool calls)
        tool_pattern = r'<invoke name="([^"]+)">'
        tools = re.findall(tool_pattern, text)
        context['operations'].update(tools)
        
        # Extract outcomes
        outcome_lines = []
        for line in text.split('\n'):
            if any(marker in line for marker in ['✅', '❌', '⚠️']):
                if len(line.strip()) < 100:
                    outcome_lines.append(line.strip())
        context['outcomes'].update(outcome_lines)
        
        return {k: list(v) for k, v in context.items()}
    
    def generate_semantic_summary(self, conversation_text: str) -> Dict[str, any]:
        """Generate semantic summary within token limit"""
        
        # Extract components
        file_refs = self.extract_file_references(conversation_text)
        decisions = self.extract_key_decisions(conversation_text)
        tech_context = self.extract_technical_context(conversation_text)
        
        # Build summary structure
        summary = {
            'session_type': self._classify_session_type(conversation_text),
            'file_references': list(file_refs)[:10],  # Limit to 10 most relevant
            'key_decisions': decisions[:5],  # Top 5 decisions
            'technical_context': tech_context,
            'timestamp': datetime.now().isoformat(),
            'token_count': 0  # Will be calculated
        }
        
        # Create narrative summary
        narrative = self._create_narrative(summary, conversation_text)
        summary['narrative'] = narrative
        
        # Calculate and enforce token limit
        full_text = json.dumps(summary, indent=2)
        estimated_tokens = estimate_tokens(full_text)
        
        if estimated_tokens > self.max_tokens:
            # Truncate while preserving structure
            summary = self._truncate_summary(summary, estimated_tokens)
        
        summary['token_count'] = estimate_tokens(json.dumps(summary, indent=2))
        return summary
    
    def _classify_session_type(self, text: str) -> str:
        """Classify the type of session based on content"""
        patterns = {
            'debugging': ['debug', 'error', 'fix', 'bug', 'issue', 'problem'],
            'feature_development': ['implement', 'feature', 'add', 'create', 'build'],
            'refactoring': ['refactor', 'improve', 'optimize', 'cleanup', 'reorganize'],
            'configuration': ['config', 'setup', 'install', 'deploy', 'configure'],
            'analysis': ['analyze', 'review', 'understand', 'explore', 'investigate'],
            'documentation': ['document', 'explain', 'describe', 'comment', 'readme']
        }
        
        text_lower = text.lower()
        scores = {}
        
        for session_type, keywords in patterns.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            scores[session_type] = score
        
        return max(scores, key=scores.get) if scores else 'general'
    
    def _create_narrative(self, summary: Dict, conversation_text: str) -> str:
        """Create a concise narrative of the session"""
        session_type = summary['session_type']
        file_count = len(summary['file_references'])
        decision_count = len(summary['key_decisions'])
        
        # Extract user's main goals from early conversation
        lines = conversation_text.split('\n')[:20]  # First 20 lines
        user_intent = self._extract_user_intent(lines)
        
        narrative_parts = []
        
        # Session overview
        if user_intent:
            narrative_parts.append(f"Session focused on {session_type}: {user_intent}")
        else:
            narrative_parts.append(f"Session type: {session_type}")
        
        # File scope
        if file_count > 0:
            narrative_parts.append(f"Modified {file_count} files")
            if summary['file_references']:
                key_files = summary['file_references'][:3]
                narrative_parts.append(f"Key files: {', '.join(key_files)}")
        
        # Key technologies
        if summary['technical_context']['technologies']:
            tech_list = summary['technical_context']['technologies'][:5]
            narrative_parts.append(f"Technologies: {', '.join(tech_list)}")
        
        # Major outcomes
        if summary['technical_context']['outcomes']:
            outcomes = [o for o in summary['technical_context']['outcomes'][:3]]
            if outcomes:
                narrative_parts.append(f"Outcomes: {'; '.join(outcomes)}")
        
        return '. '.join(narrative_parts) + '.'
    
    def _extract_user_intent(self, early_lines: List[str]) -> Optional[str]:
        """Extract user's main intent from early conversation"""
        user_patterns = [
            r'user:\s*"([^"]+)"',
            r'user:\s*(.+)',
            r'I want to (.+)',
            r'I need to (.+)',
            r'Can you (.+)',
            r'Help me (.+)'
        ]
        
        text = '\n'.join(early_lines)
        for pattern in user_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                intent = match.group(1).strip()
                if len(intent) < 100:  # Keep it concise
                    return intent
        return None
    
    def _truncate_summary(self, summary: Dict, current_tokens: int) -> Dict:
        """Truncate summary to fit token limit while preserving key info"""
        target_reduction = current_tokens - self.max_tokens
        
        # Priority order for truncation
        truncation_order = [
            ('key_decisions', 3),  # Keep top 3 decisions
            ('file_references', 7),  # Keep top 7 files
            ('technical_context.outcomes', 2),  # Keep top 2 outcomes
            ('technical_context.operations', 5),  # Keep top 5 operations
            ('technical_context.technologies', 5),  # Keep top 5 technologies
        ]
        
        for field_path, limit in truncation_order:
            if '.' in field_path:
                parent, child = field_path.split('.')
                if len(summary[parent][child]) > limit:
                    summary[parent][child] = summary[parent][child][:limit]
            else:
                if len(summary[field_path]) > limit:
                    summary[field_path] = summary[field_path][:limit]
            
            # Check if we're within limit
            current_size = estimate_tokens(json.dumps(summary, indent=2))
            if current_size <= self.max_tokens:
                break
        
        return summary

def trigger_memory_storage(summary: Dict[str, any]) -> bool:
    """Store semantic summary via MCP memory_store_enhanced"""
    try:
        # Prepare summary for storage
        summary_text = f"""
Session Summary ({summary['timestamp']}):

Type: {summary['session_type']}
{summary['narrative']}

Technical Context:
- Technologies: {', '.join(summary['technical_context']['technologies'])}
- Operations: {', '.join(summary['technical_context']['operations'])}

File References: {', '.join(summary['file_references'])}

Key Decisions:
{chr(10).join(f'• {decision}' for decision in summary['key_decisions'])}

Outcomes:
{chr(10).join(f'• {outcome}' for outcome in summary['technical_context']['outcomes'])}
"""
        
        # Use docker exec to call MCP memory storage
        docker_cmd = [
            'docker', 'exec', '-i', 'claude-l9-template-neural',
            'python3', '-c', f'''
import sys
sys.path.append("/app")
from neural_mcp_server_enhanced import memory_store_enhanced
import asyncio

summary_text = """{summary_text}"""
metadata = {json.dumps({
    'type': 'pre_compact_summary',
    'session_type': summary['session_type'], 
    'file_count': len(summary['file_references']),
    'timestamp': summary['timestamp'],
    'token_count': summary['token_count']
})}

result = asyncio.run(memory_store_enhanced(
    memory=summary_text,
    metadata=metadata
))
print("Semantic memory stored successfully")
'''
        ]
        
        result = subprocess.run(docker_cmd, 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"🧠 Stored semantic memory ({summary['token_count']} tokens)", file=sys.stderr)
            return True
        else:
            print(f"❌ Memory storage failed: {result.stderr}", file=sys.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to store semantic memory: {e}", file=sys.stderr)
        return False

def main():
    """Main entry point for PreCompact hook"""
    if len(sys.argv) < 2:
        print("Usage: semantic_memory_injector.py <conversation_context>", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Get conversation context (Claude passes this to PreCompact hooks)
        conversation_context = sys.argv[1] if len(sys.argv) > 1 else ""
        
        # If no context provided, try to read from stdin
        if not conversation_context.strip():
            conversation_context = sys.stdin.read()
        
        if not conversation_context.strip():
            print("💡 No conversation context available for semantic memory", file=sys.stderr)
            return
        
        print("🧠 Extracting semantic memory before compaction...", file=sys.stderr)
        
        # Extract semantic summary
        extractor = SemanticMemoryExtractor(max_tokens=2000)
        summary = extractor.generate_semantic_summary(conversation_context)
        
        print(f"📊 Generated summary: {summary['token_count']} tokens, {len(summary['file_references'])} files", file=sys.stderr)
        
        # Store via MCP
        if trigger_memory_storage(summary):
            print("✅ Semantic memory injection completed", file=sys.stderr)
        else:
            print("❌ Semantic memory injection failed", file=sys.stderr)
        
    except Exception as e:
        print(f"❌ Error in semantic memory injector: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()