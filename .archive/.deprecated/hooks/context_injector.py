#!/usr/bin/env python3
"""
Context Injector Hook
Injects relevant project context before user prompts are processed
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add neural-system to path  
sys.path.append(os.path.dirname(__file__))

from memory_system import MemorySystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ContextInjector:
    """Inject relevant context before user prompts"""
    
    def __init__(self):
        self.memory_system = MemorySystem()
        
    async def inject_context(self, prompt_data: dict):
        """Analyze user prompt and inject relevant context"""
        try:
            user_message = prompt_data.get('user_message', '')
            if not user_message or len(user_message.strip()) < 5:
                return
            
            # Skip injection for very short queries or commands
            skip_patterns = ['/help', '/version', '/status', '/list', '/ls', '/pwd']
            if any(user_message.strip().startswith(pattern) for pattern in skip_patterns):
                return
            
            # Initialize memory system
            await self.memory_system.initialize()
            
            # Extract key terms and concepts from user message
            context_query = self._extract_context_query(user_message)
            
            if not context_query:
                return
            
            # Search for relevant memories
            relevant_memories = await self.memory_system.search_memories(
                query=context_query,
                limit=3,
                similarity_threshold=0.6
            )
            
            if not relevant_memories:
                return
            
            # Format context for injection
            context_info = self._format_context(relevant_memories, context_query)
            
            # Output context as system reminder for Claude
            print(f"<system-reminder>")
            print(f"📋 **Project Context** (auto-injected based on query: \"{context_query}\")")
            print(f"")
            print(context_info)
            print(f"</system-reminder>")
            
            logger.info(f"🧠 Injected context for query: {context_query}")
            
        except Exception as e:
            logger.error(f"❌ Context injection failed: {e}")
            # Silently continue - don't interrupt user workflow
    
    def _extract_context_query(self, user_message: str) -> str:
        """Extract key terms from user message for context search"""
        # Remove common stop words and focus on meaningful terms
        stop_words = {
            'how', 'what', 'where', 'when', 'why', 'who', 'can', 'could', 
            'should', 'would', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 
            'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was',
            'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        # Extract meaningful words
        words = user_message.lower().split()
        meaningful_words = []
        
        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalnum())
            
            # Keep words that are likely meaningful
            if (len(clean_word) > 2 and 
                clean_word not in stop_words and 
                not clean_word.isdigit()):
                meaningful_words.append(clean_word)
        
        # Focus on first few meaningful terms
        context_terms = meaningful_words[:5]
        
        # Look for technical terms, file names, function names
        tech_patterns = []
        if any(term in user_message.lower() for term in ['function', 'class', 'method', 'api']):
            tech_patterns.append('code')
        if any(term in user_message.lower() for term in ['error', 'bug', 'issue', 'problem']):
            tech_patterns.append('error')
        if any(term in user_message.lower() for term in ['config', 'setup', 'install']):
            tech_patterns.append('configuration')
        
        # Combine terms
        all_terms = tech_patterns + context_terms
        return ' '.join(all_terms[:6])  # Limit to 6 terms
    
    def _format_context(self, memories: List[Dict], query: str) -> str:
        """Format relevant memories for context injection"""
        if not memories:
            return ""
        
        context_lines = []
        context_lines.append(f"Found {len(memories)} relevant memories for your query:")
        context_lines.append("")
        
        for i, memory in enumerate(memories, 1):
            payload = memory.get('payload', {})
            content = payload.get('content', 'No content')
            file_path = payload.get('relative_path', payload.get('file_path', ''))
            timestamp = payload.get('timestamp', '')
            score = memory.get('score', 0.0)
            
            # Format timestamp
            time_str = ""
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime(" (%Y-%m-%d)")
                except:
                    pass
            
            # Format entry
            context_lines.append(f"**{i}.** Score: {score:.2f}{time_str}")
            
            if file_path:
                context_lines.append(f"   📁 File: `{file_path}`")
            
            # Show relevant excerpt
            content_preview = content[:300]
            if len(content) > 300:
                content_preview += "..."
            
            context_lines.append(f"   💡 Content: {content_preview}")
            context_lines.append("")
        
        context_lines.append("This context is automatically provided to help with your query.")
        
        return "\n".join(context_lines)

async def main():
    """Main hook entry point"""
    try:
        # Read prompt data from stdin (Claude Code hook format)
        input_data = json.load(sys.stdin)
        
        # Initialize and process  
        injector = ContextInjector()
        await injector.inject_context(input_data)
        
    except json.JSONDecodeError:
        # Silent fail for non-JSON input
        pass  
    except Exception as e:
        # Log errors but don't fail the hook
        logger.error(f"Hook error: {e}")
        # Don't print error to stdout as it may interfere with Claude's input

if __name__ == "__main__":
    asyncio.run(main())