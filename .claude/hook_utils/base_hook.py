"""
L9 BaseHook - Abstract Base Class for All Hooks
Enforces consistent interface, dependency management, and error handling
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import logging

try:
    from .dependency_manager import DependencyManager
    from .utilities import estimate_tokens
except ImportError:
    # Handle relative import issues
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from dependency_manager import DependencyManager
    from utilities import estimate_tokens


class BaseHook(ABC):
    """Abstract base class for all L9 hooks with systematic patterns"""
    
    def __init__(self, max_tokens: int = 3500, hook_name: Optional[str] = None):
        self.max_tokens = max_tokens
        self.hook_name = hook_name or self.__class__.__name__
        self.project_dir = Path(self._get_project_dir())
        self.dependency_manager = DependencyManager()
        self._setup_logging()
        
    def _get_project_dir(self) -> str:
        """Get project directory with fallback"""
        return Path(__file__).parent.parent.parent.absolute()
    
    def _setup_logging(self):
        """Setup consistent logging for all hooks"""
        self.logger = logging.getLogger(f"l9_hook.{self.hook_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(f'[{self.hook_name}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Main execution method - must be implemented by each hook"""
        pass
    
    def validate_dependencies(self) -> bool:
        """Validate all required dependencies are available"""
        return self.dependency_manager.validate_all()
    
    def get_required_dependencies(self) -> list[str]:
        """Return list of required dependencies for this hook"""
        return []
    
    def handle_execution_error(self, error: Exception) -> Dict[str, Any]:
        """Consistent error handling across all hooks"""
        self.logger.error(f"Execution failed: {error}")
        return {
            "status": "error",
            "hook_name": self.hook_name,
            "error": str(error),
            "fallback_applied": True
        }
    
    def run(self) -> Dict[str, Any]:
        """Main entry point with error handling and validation"""
        try:
            # Pre-execution validation
            if not self.validate_dependencies():
                self.logger.warning("Dependencies not fully available, using fallback mode")
            
            # Execute hook logic
            result = self.execute()
            
            # Post-execution validation
            if isinstance(result, dict):
                result["hook_name"] = self.hook_name
                result["tokens_used"] = result.get("tokens_used", 0)
            
            return result
            
        except Exception as e:
            return self.handle_execution_error(e)
    
    def estimate_content_tokens(self, content: str) -> int:
        """Consistent token estimation across all hooks"""
        return estimate_tokens(content)
    
    def format_output(self, content: str, prefix: str = "") -> str:
        """Consistent output formatting"""
        if prefix:
            return f"{prefix} {content}"
        return content
    
    def log_execution(self, message: str, level: str = "info"):
        """Consistent execution logging"""
        getattr(self.logger, level)(message)