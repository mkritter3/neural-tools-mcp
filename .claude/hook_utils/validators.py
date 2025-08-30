"""
L9 Hook Validators - Automated Compliance and Quality Assurance
Ensures all hooks meet L9 engineering standards systematically
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import ast
import inspect
from abc import ABC


class HookComplianceValidator:
    """Validates hook compliance with L9 standards"""
    
    def __init__(self):
        self.violations: List[Dict[str, Any]] = []
        
    def validate_hook_compliance(self, hook_file: Path) -> Dict[str, Any]:
        """
        Comprehensive validation of a hook file
        
        Returns:
            Dict with compliance status and any violations found
        """
        self.violations = []
        
        try:
            # Read and parse the hook file
            content = hook_file.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Run all validation checks
            self._validate_imports(tree, content)
            self._validate_class_structure(tree)
            self._validate_no_code_duplication(tree, content)
            self._validate_error_handling(tree)
            self._validate_dependencies(tree, content)
            
            # Determine overall compliance
            compliance_score = self._calculate_compliance_score()
            
            return {
                "file": str(hook_file),
                "compliant": len(self.violations) == 0,
                "compliance_score": compliance_score,
                "violations": self.violations,
                "total_violations": len(self.violations)
            }
            
        except Exception as e:
            return {
                "file": str(hook_file),
                "compliant": False,
                "compliance_score": 0.0,
                "violations": [{"type": "parse_error", "message": str(e)}],
                "total_violations": 1
            }
    
    def _validate_imports(self, tree: ast.AST, content: str):
        """Validate import patterns follow L9 standards"""
        
        # Check for manual sys.path manipulation (should use DependencyManager)
        if 'sys.path.insert' in content or 'sys.path.append' in content:
            self.violations.append({
                "type": "import_violation",
                "severity": "high", 
                "message": "Manual sys.path manipulation found - should use DependencyManager"
            })
        
        # Check for direct neural-tools imports without dependency management
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if 'prism_scorer' in node.module and 'dependency_manager' not in content:
                        self.violations.append({
                            "type": "import_violation",
                            "severity": "medium",
                            "message": "Direct PRISM import without dependency management"
                        })
    
    def _validate_class_structure(self, tree: ast.AST):
        """Validate class structure follows L9 patterns"""
        
        hook_classes = []
        has_base_hook_import = False
        
        # Find all classes and check BaseHook usage
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                hook_classes.append(node)
            elif isinstance(node, ast.ImportFrom):
                if node.module and 'base_hook' in node.module:
                    has_base_hook_import = True
        
        # Check if hook classes inherit from BaseHook
        for cls in hook_classes:
            # Skip obviously non-hook classes
            if cls.name in ['PrismScorer', 'SemanticMemoryExtractor']:
                continue
                
            has_base_hook_inheritance = any(
                base.id == 'BaseHook' for base in cls.bases 
                if isinstance(base, ast.Name)
            )
            
            if not has_base_hook_inheritance and has_base_hook_import:
                self.violations.append({
                    "type": "structure_violation",
                    "severity": "high",
                    "message": f"Class {cls.name} should inherit from BaseHook"
                })
    
    def _validate_no_code_duplication(self, tree: ast.AST, content: str):
        """Validate no code duplication with shared utilities"""
        
        # Check for duplicate function definitions
        duplicate_functions = ['estimate_tokens', 'format_context']
        
        for func_name in duplicate_functions:
            if f'def {func_name}(' in content:
                self.violations.append({
                    "type": "duplication_violation", 
                    "severity": "medium",
                    "message": f"Function {func_name} duplicated - should use shared utilities"
                })
        
        # Check for fallback class definitions (should use shared fallbacks)
        if 'class PrismScorer:' in content and 'fallback' in content.lower():
            self.violations.append({
                "type": "duplication_violation",
                "severity": "low", 
                "message": "Custom PrismScorer fallback - should use shared fallbacks"
            })
    
    def _validate_error_handling(self, tree: ast.AST):
        """Validate consistent error handling patterns"""
        
        has_try_except = False
        has_proper_error_handling = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                has_try_except = True
                
                # Check if exception handling follows patterns
                for handler in node.handlers:
                    if handler.type is None or (
                        isinstance(handler.type, ast.Name) and 
                        handler.type.id == 'Exception'
                    ):
                        has_proper_error_handling = True
        
        if has_try_except and not has_proper_error_handling:
            self.violations.append({
                "type": "error_handling_violation",
                "severity": "medium",
                "message": "Inconsistent error handling - should use BaseHook patterns"
            })
    
    def _validate_dependencies(self, tree: ast.AST, content: str):
        """Validate dependency management follows L9 patterns"""
        
        # Check if using DependencyManager
        uses_dependency_manager = 'DependencyManager' in content
        has_imports = any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(tree)
        )
        
        if has_imports and not uses_dependency_manager:
            self.violations.append({
                "type": "dependency_violation",
                "severity": "low",
                "message": "Should use DependencyManager for systematic import handling"
            })
    
    def _calculate_compliance_score(self) -> float:
        """Calculate compliance score based on violations"""
        if not self.violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.6}
        
        total_penalty = sum(
            severity_weights.get(v.get('severity', 'medium'), 0.3)
            for v in self.violations
        )
        
        # Cap penalty and convert to score
        max_penalty = 2.0
        penalty_ratio = min(total_penalty / max_penalty, 1.0)
        
        return 1.0 - penalty_ratio


def validate_hook_compliance(hook_file: Path) -> Dict[str, Any]:
    """Convenience function for validating a single hook"""
    validator = HookComplianceValidator()
    return validator.validate_hook_compliance(hook_file)


def validate_all_hooks(hooks_directory: Path) -> Dict[str, Any]:
    """Validate compliance for all hooks in directory"""
    validator = HookComplianceValidator()
    results = {}
    
    for hook_file in hooks_directory.glob("*.py"):
        if hook_file.name.startswith('__'):
            continue
        
        results[hook_file.name] = validator.validate_hook_compliance(hook_file)
    
    # Calculate overall statistics
    total_hooks = len(results)
    compliant_hooks = sum(1 for r in results.values() if r['compliant'])
    avg_score = sum(r['compliance_score'] for r in results.values()) / total_hooks if total_hooks else 0
    
    return {
        "results": results,
        "summary": {
            "total_hooks": total_hooks,
            "compliant_hooks": compliant_hooks,
            "compliance_rate": compliant_hooks / total_hooks if total_hooks else 0,
            "average_score": avg_score
        }
    }