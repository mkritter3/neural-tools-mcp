"""
L9 Hook Utilities - Shared Infrastructure for All Hooks
Eliminates code duplication and ensures consistent patterns
"""

from .base_hook import BaseHook
from .utilities import estimate_tokens, format_context
from .dependency_manager import DependencyManager
from .validators import validate_hook_compliance

__all__ = [
    'BaseHook',
    'estimate_tokens', 
    'format_context',
    'DependencyManager',
    'validate_hook_compliance'
]