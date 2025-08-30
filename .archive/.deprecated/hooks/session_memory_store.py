#!/usr/bin/env python3
"""
Session Memory Store Hook
Stores session insights and learnings when Claude stops responding
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add neural-system to path
sys.path.append(os.path.dirname(__file__))

from memory_system import MemorySystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SessionMemoryStore:
    """Store session insights when Claude stops"""
    
    def __init__(self):
        self.memory_system = MemorySystem()
    
    async def store_session_insights(self, session_data: dict):
        """Store key insights from the session"""
        try:
            # Initialize memory system
            await self.memory_system.initialize()
            
            # Extract session insights
            insights = self._extract_session_insights(session_data)
            
            if not insights:
                logger.info("📝 No significant insights to store from session")
                return
            
            # Store each insight
            stored_count = 0
            for insight in insights:
                try:
                    memory_id = await self.memory_system.store_memory(
                        content=insight['content'],
                        metadata=insight['metadata']
                    )
                    stored_count += 1
                    logger.debug(f"💾 Stored insight: {memory_id}")
                except Exception as e:
                    logger.error(f"Failed to store insight: {e}")
            
            if stored_count > 0:
                logger.info(f"✅ Stored {stored_count} session insights")
            
        except Exception as e:
            logger.error(f"❌ Session memory storage failed: {e}")
    
    def _extract_session_insights(self, session_data: dict) -> List[Dict[str, Any]]:
        """Extract meaningful insights from session data"""
        insights = []
        
        # Current timestamp for all insights
        current_time = datetime.now()
        
        # Try to extract insights from conversation context
        conversation_summary = session_data.get('conversation_summary', '')
        recent_actions = session_data.get('recent_actions', [])
        
        # Create session summary insight
        if conversation_summary or recent_actions:
            session_content = self._create_session_summary(
                conversation_summary, recent_actions, current_time
            )
            
            if session_content:
                insights.append({
                    'content': session_content,
                    'metadata': {
                        'type': 'session_summary',
                        'timestamp': current_time.isoformat(),
                        'session_end': True,
                        'auto_stored': True,
                        'confidence': 'medium'
                    }
                })
        
        # Extract any decisions made during the session
        decisions = self._extract_decisions(session_data)
        for decision in decisions:
            insights.append({
                'content': decision,
                'metadata': {
                    'type': 'session_decision',
                    'timestamp': current_time.isoformat(),
                    'session_end': True,
                    'auto_stored': True,
                    'confidence': 'high'
                }
            })
        
        # Extract any patterns or learnings
        patterns = self._extract_patterns(session_data)
        for pattern in patterns:
            insights.append({
                'content': pattern,
                'metadata': {
                    'type': 'session_pattern',
                    'timestamp': current_time.isoformat(),
                    'session_end': True,
                    'auto_stored': True,
                    'confidence': 'low'
                }
            })
        
        return insights
    
    def _create_session_summary(self, conversation_summary: str, 
                               recent_actions: List, current_time: datetime) -> str:
        """Create a session summary for storage"""
        summary_parts = []
        
        # Add timestamp
        time_str = current_time.strftime("%Y-%m-%d %H:%M")
        summary_parts.append(f"Session completed at {time_str}")
        
        # Add conversation context if available
        if conversation_summary and len(conversation_summary.strip()) > 10:
            summary_parts.append(f"\nConversation Summary:")
            summary_parts.append(conversation_summary[:500])  # Limit to 500 chars
        
        # Add recent actions if available
        if recent_actions:
            summary_parts.append(f"\nKey Actions Taken:")
            for action in recent_actions[-5:]:  # Last 5 actions
                action_str = str(action)[:100]  # Limit action length
                summary_parts.append(f"• {action_str}")
        
        # Only return if we have meaningful content
        full_summary = "\n".join(summary_parts)
        return full_summary if len(full_summary) > 50 else ""
    
    def _extract_decisions(self, session_data: dict) -> List[str]:
        """Extract any decisions made during the session"""
        decisions = []
        
        # Look for decision keywords in session data
        decision_keywords = [
            'decided to', 'chose to', 'will implement', 'agreed to',
            'conclusion:', 'final decision:', 'approach:'
        ]
        
        # Check conversation content for decisions
        conversation = session_data.get('conversation_summary', '').lower()
        
        for keyword in decision_keywords:
            if keyword in conversation:
                # Extract context around the decision keyword
                start_idx = conversation.find(keyword)
                if start_idx != -1:
                    # Get surrounding context (50 chars before, 200 after)
                    context_start = max(0, start_idx - 50)
                    context_end = min(len(conversation), start_idx + 200)
                    
                    decision_context = conversation[context_start:context_end]
                    decisions.append(f"Decision made: {decision_context}")
                    
                    # Only extract first few decisions to avoid noise
                    if len(decisions) >= 2:
                        break
        
        return decisions
    
    def _extract_patterns(self, session_data: dict) -> List[str]:
        """Extract any patterns or recurring themes"""
        patterns = []
        
        # Look for repeated concepts or themes
        pattern_indicators = [
            'pattern:', 'recurring', 'consistent', 'always',
            'typically', 'usually', 'often', 'frequently'
        ]
        
        conversation = session_data.get('conversation_summary', '').lower()
        
        for indicator in pattern_indicators:
            if indicator in conversation:
                # Extract pattern context
                start_idx = conversation.find(indicator)
                if start_idx != -1:
                    context_start = max(0, start_idx - 30)
                    context_end = min(len(conversation), start_idx + 150)
                    
                    pattern_context = conversation[context_start:context_end]
                    patterns.append(f"Pattern observed: {pattern_context}")
                    
                    # Limit patterns to avoid noise
                    if len(patterns) >= 1:
                        break
        
        return patterns

async def main():
    """Main hook entry point"""
    try:
        # Read session data from stdin (Claude Code hook format)
        input_data = json.load(sys.stdin) if sys.stdin.isatty() == False else {}
        
        # Initialize and store session insights
        store = SessionMemoryStore()
        await store.store_session_insights(input_data)
        
    except json.JSONDecodeError:
        # Handle non-JSON input gracefully
        logger.debug("No JSON input provided to session store hook")
    except Exception as e:
        # Log errors but don't fail the hook
        logger.error(f"Hook error: {e}")
        # Don't print errors to stdout as it may interfere with Claude

if __name__ == "__main__":
    asyncio.run(main())