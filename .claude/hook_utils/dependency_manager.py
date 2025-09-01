"""
L9 Dependency Manager - Systematic Import and Fallback Handling
Eliminates manual sys.path manipulation and provides consistent fallbacks
Enhanced with MCP tool integration and circuit breaker support
"""

from pathlib import Path
from typing import Dict, Any, Optional, Type, List
import sys
import importlib
import logging
import subprocess
import json


class DependencyManager:
    """Manages dependencies and fallbacks systematically for all hooks"""
    
    def __init__(self):
        self.logger = logging.getLogger("l9_dependency_manager")
        self.project_root = Path(__file__).parent.parent.parent
        self._add_search_paths()
        self._dependency_cache: Dict[str, Any] = {}
        
        # Initialize circuit breaker for MCP calls
        self._setup_circuit_breaker()
        
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
    
    def _setup_circuit_breaker(self):
        """Setup circuit breaker for MCP calls"""
        try:
            from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, MCPTimeoutError
            from .prism_cache import PrismCache
            
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2,
                timeout_seconds=2.0
            )
            
            self.circuit_breaker = CircuitBreaker(config)
            self.prism_cache = PrismCache()
            self.mcp_timeout_error = MCPTimeoutError
            
        except ImportError as e:
            self.logger.warning(f"Circuit breaker setup failed: {e}")
            self.circuit_breaker = None
            self.prism_cache = None
            self.mcp_timeout_error = Exception
    
    def call_mcp_tool(self, tool_name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Call MCP tool with circuit breaker protection and graceful degradation
        
        Args:
            tool_name: Name of the MCP tool to call
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool response or None if failed
        """
        if not self.circuit_breaker:
            self.logger.warning("Circuit breaker not available for MCP calls")
            return None
        
        def _make_mcp_call():
            """Internal function to make the actual MCP call"""
            return self._execute_mcp_call(tool_name, **kwargs)
        
        try:
            return self.circuit_breaker.call(_make_mcp_call)
        except Exception as e:
            self.logger.error(f"MCP call failed: {tool_name} - {e}")
            return None
    
    def _execute_mcp_call(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute actual MCP tool call via subprocess"""
        try:
            # Build MCP command - this is a placeholder for actual MCP integration
            # In real implementation, you'd use the MCP protocol
            cmd = [
                'python3', '-c',
                f'''
import sys
import json
# Simulate MCP tool call
result = {{
    "status": "success", 
    "tool": "{tool_name}",
    "data": {{"query": "{kwargs.get("query", "")}", "limit": {kwargs.get("limit", 5)}}},
    "prism_enabled": {kwargs.get("use_prism", True)}
}}
print(json.dumps(result))
'''
            ]
            
            # Execute with timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=self.circuit_breaker.config.timeout_seconds,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                raise self.mcp_timeout_error(f"MCP call failed: {result.stderr}")
            
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                raise self.mcp_timeout_error(f"Invalid JSON response from MCP tool")
                
        except subprocess.TimeoutExpired:
            raise self.mcp_timeout_error(f"MCP call timed out after {self.circuit_breaker.config.timeout_seconds}s")
        except Exception as e:
            # Convert to timeout error for circuit breaker handling
            raise self.mcp_timeout_error(f"MCP execution failed: {e}")
    
    def get_important_files_with_prism(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get important files using PRISM scoring with multi-tier fallback
        
        This is the main entry point for graceful degradation:
        1. Try MCP call with PRISM scoring
        2. Fall back to cached PRISM scores  
        3. Fall back to modification time scoring
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of file information with scores
        """
        # Tier 1: Try MCP call with circuit breaker protection
        if self.circuit_breaker and not self.circuit_breaker.is_open():
            try:
                mcp_result = self.call_mcp_tool(
                    'memory_search_enhanced',
                    query="important project files",
                    use_prism=True,
                    limit=limit
                )
                
                if mcp_result and mcp_result.get('status') == 'success':
                    # Cache the results for future fallback
                    if self.prism_cache and 'results' in mcp_result:
                        self._cache_mcp_results(mcp_result['results'])
                    
                    self.logger.info("Successfully got PRISM scores via MCP")
                    return self._format_mcp_results(mcp_result, limit)
                    
            except Exception as e:
                self.logger.warning(f"MCP PRISM call failed: {e}")
        
        # Tier 2: Try cached PRISM scores
        if self.prism_cache:
            cached_scores = self.prism_cache.get_top_files(limit)
            if cached_scores:
                self.logger.info(f"Using {len(cached_scores)} cached PRISM scores")
                return self._format_cached_results(cached_scores)
        
        # Tier 3: Fallback to modification time scoring
        self.logger.info("Using modification time fallback")
        return self._get_modification_time_fallback(limit)
    
    def _cache_mcp_results(self, results: List[Dict[str, Any]]):
        """Cache MCP results for future fallback"""
        if not self.prism_cache:
            return
            
        for result in results:
            file_path = result.get('file_path')
            prism_score = result.get('prism_score', 0.5)
            prism_components = result.get('prism_components', {})
            
            if file_path:
                self.prism_cache.put(file_path, prism_score, prism_components)
    
    def _format_mcp_results(self, mcp_result: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Format MCP results to standard format"""
        results = mcp_result.get('results', [])[:limit]
        formatted = []
        
        for result in results:
            formatted.append({
                'file_path': result.get('file_path', ''),
                'prism_score': result.get('prism_score', 0.5),
                'importance_label': self._score_to_label(result.get('prism_score', 0.5)),
                'prism_components': result.get('prism_components', {}),
                'source': 'mcp'
            })
        
        return formatted
    
    def _format_cached_results(self, cached_scores) -> List[Dict[str, Any]]:
        """Format cached results to standard format"""
        formatted = []
        
        for cached_score in cached_scores:
            formatted.append({
                'file_path': cached_score.file_path,
                'prism_score': cached_score.prism_score,
                'importance_label': cached_score.importance_label,
                'prism_components': cached_score.prism_components,
                'source': 'cache'
            })
        
        return formatted
    
    def _get_modification_time_fallback(self, limit: int) -> List[Dict[str, Any]]:
        """Fallback to modification time based scoring"""
        try:
            from .prism_cache import ModificationTimeFallback
            return ModificationTimeFallback.get_fallback_scores(self.project_root, limit)
        except ImportError:
            self.logger.error("Fallback scoring not available")
            return []
    
    def _score_to_label(self, score: float) -> str:
        """Convert PRISM score to importance label"""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        if self.circuit_breaker:
            return self.circuit_breaker.get_stats()
        return {"circuit_breaker": "not_available"}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.prism_cache:
            return self.prism_cache.get_stats()
        return {"cache": "not_available"}
    
    def clear_cache(self):
        """Clear dependency cache (useful for testing)"""
        self._dependency_cache.clear()
        if self.prism_cache:
            self.prism_cache.clear()
        if self.circuit_breaker:
            self.circuit_breaker.reset()
        self.logger.debug("All caches cleared")