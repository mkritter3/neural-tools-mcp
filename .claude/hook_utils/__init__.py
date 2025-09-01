"""
L9 Hook Utilities - Shared Infrastructure for All Hooks
Eliminates code duplication and ensures consistent patterns
"""

from .base_hook import BaseHook
from .utilities import estimate_tokens, format_context
from .dependency_manager import DependencyManager
from .validators import validate_hook_compliance

# Phase 1: New shared utilities to eliminate code duplication
from .formatters import (
    format_session_metadata,
    format_file_references,
    format_technical_context,
    format_key_decisions,
    format_mcp_suggestions,
    format_project_overview,
    format_historical_context,
    format_session_outcomes,
    format_context_summary
)

from .patterns import (
    conversation_patterns,
    extract_file_references,
    extract_key_decisions,
    extract_technical_context,
    classify_session_type,
    extract_user_intent,
    extract_implementation_details,
    generate_mcp_suggestions,
    create_conversation_summary
)

from .project_analyzer import (
    analyze_project_structure,
    analyze_technology_stack,
    analyze_recent_activity,
    analyze_code_quality_indicators,
    create_project_summary
)

__all__ = [
    # Core infrastructure
    'BaseHook',
    'estimate_tokens', 
    'format_context',
    'DependencyManager',
    'validate_hook_compliance',
    
    # Shared formatters (eliminates 19 duplicate functions)
    'format_session_metadata',
    'format_file_references',
    'format_technical_context',
    'format_key_decisions',
    'format_mcp_suggestions',
    'format_project_overview',
    'format_historical_context',
    'format_session_outcomes',
    'format_context_summary',
    
    # Shared pattern extraction (eliminates duplicate regex logic)
    'conversation_patterns',
    'extract_file_references',
    'extract_key_decisions',
    'extract_technical_context',
    'classify_session_type',
    'extract_user_intent',
    'extract_implementation_details',
    'generate_mcp_suggestions',
    'create_conversation_summary',
    
    # Shared project analysis (eliminates duplicate analysis functions)
    'analyze_project_structure',
    'analyze_technology_stack',
    'analyze_recent_activity',
    'analyze_code_quality_indicators',
    'create_project_summary'
]