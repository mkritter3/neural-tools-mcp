"""
L9 Dependency Manager - Systematic Import and Fallback Handling
Eliminates manual sys.path manipulation and provides consistent fallbacks
"""

from pathlib import Path
from typing import Dict, Any, Optional, Type
import sys
import importlib
import logging


class DependencyManager:
    """Manages dependencies and fallbacks systematically for all hooks"""
    
    def __init__(self):
        self.logger = logging.getLogger("l9_dependency_manager")
        self.project_root = Path(__file__).parent.parent.parent
        self._add_search_paths()
        self._dependency_cache: Dict[str, Any] = {}
        
    def _add_search_paths(self):
        """Add necessary paths to Python path systematically"""
        search_paths = [
            self.project_root / "neural-tools",  # Docker MCP tools
            self.project_root / ".claude" / "hook_utils",  # Hook utilities
        ]
        
        for path in search_paths:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
    
    def import_with_fallback(self, module_name: str, fallback_class: Optional[Type] = None) -> Any:
        """
        Import module with systematic fallback handling
        
        Args:
            module_name: Module to import (e.g., 'prism_scorer')
            fallback_class: Fallback class if import fails
        
        Returns:
            Imported module or fallback class
        """
        # Check cache first
        cache_key = f"{module_name}_{fallback_class.__name__ if fallback_class else 'none'}"
        if cache_key in self._dependency_cache:
            return self._dependency_cache[cache_key]
        
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            self.logger.info(f"Successfully imported {module_name}")
            self._dependency_cache[cache_key] = module
            return module
            
        except ImportError as e:
            self.logger.warning(f"Failed to import {module_name}: {e}")
            
            if fallback_class:
                self.logger.info(f"Using fallback class for {module_name}")
                self._dependency_cache[cache_key] = fallback_class
                return fallback_class
            
            # Return None if no fallback available
            self._dependency_cache[cache_key] = None
            return None
    
    def get_prism_scorer(self):
        """Get PRISM scorer with systematic fallback"""
        try:
            from .fallbacks import SimplePrismScorer
        except ImportError:
            from fallbacks import SimplePrismScorer
        
        prism_module = self.import_with_fallback('prism_scorer', SimplePrismScorer)
        
        if prism_module and hasattr(prism_module, 'PrismScorer'):
            return prism_module.PrismScorer
        elif prism_module:
            # Using fallback class directly
            return prism_module
        else:
            return SimplePrismScorer
    
    def validate_all(self) -> bool:
        """Validate all common dependencies are available"""
        validations = {
            'pathlib': self._validate_stdlib_import('pathlib'),
            'json': self._validate_stdlib_import('json'),
            'os': self._validate_stdlib_import('os'),
            'sys': self._validate_stdlib_import('sys'),
        }
        
        # Log validation results
        failed = [k for k, v in validations.items() if not v]
        if failed:
            self.logger.error(f"Validation failed for: {', '.join(failed)}")
            return False
        
        self.logger.debug("All dependency validations passed")
        return True
    
    def _validate_stdlib_import(self, module_name: str) -> bool:
        """Validate standard library import"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    def get_mcp_tools_available(self) -> bool:
        """Check if MCP tools are available"""
        return self.import_with_fallback('neural_mcp_server_enhanced') is not None
    
    def clear_cache(self):
        """Clear dependency cache (useful for testing)"""
        self._dependency_cache.clear()
        self.logger.debug("Dependency cache cleared")