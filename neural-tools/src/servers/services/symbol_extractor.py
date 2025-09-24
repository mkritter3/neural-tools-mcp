"""
ADR-0090 Phase 2: Python AST-based Symbol Extraction
Simpler alternative to Tree-sitter for extracting code symbols
"""

import ast
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PythonSymbolExtractor:
    """Extract symbols from Python code using AST parsing"""

    def __init__(self):
        self.symbols = []
        self.current_class = None
        self.file_path = None

    def extract_symbols(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Extract symbols from Python file content
        Returns dict with symbols and relationships
        """
        self.symbols = []
        self.current_class = None
        self.file_path = file_path

        try:
            tree = ast.parse(content)
            self._visit_node(tree)

            return {
                "symbols": self.symbols,
                "relationships": self._extract_relationships()
            }
        except SyntaxError as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return {"symbols": [], "relationships": []}

    def _visit_node(self, node, parent_class=None):
        """Recursively visit AST nodes to extract symbols"""

        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            symbol = self._extract_function(node, parent_class)
            self.symbols.append(symbol)

            # Visit nested functions
            for child in ast.iter_child_nodes(node):
                self._visit_node(child, parent_class)

        elif isinstance(node, ast.ClassDef):
            symbol = self._extract_class(node)
            self.symbols.append(symbol)

            # Visit methods and nested classes
            for child in ast.iter_child_nodes(node):
                self._visit_node(child, node.name)

        else:
            # Continue traversing
            for child in ast.iter_child_nodes(node):
                self._visit_node(child, parent_class)

    def _extract_function(self, node: ast.FunctionDef, parent_class: Optional[str] = None) -> Dict[str, Any]:
        """Extract function/method information"""

        # Get function signature
        args = []
        if node.args.args:
            args = [arg.arg for arg in node.args.args]

        # Get return type if available
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

        # Calculate complexity (simplified McCabe)
        complexity = self._calculate_complexity(node)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Determine if it's a method or function
        symbol_type = "Method" if parent_class else "Function"

        # Extract decorators
        decorators = [ast.unparse(d) if hasattr(ast, 'unparse') else str(d) for d in node.decorator_list]

        # Extract function calls within this function
        calls = self._extract_calls(node)

        return {
            "type": symbol_type,
            "name": node.name,
            "qualified_name": f"{parent_class}.{node.name}" if parent_class else node.name,
            "line_start": node.lineno,
            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            "signature": f"({', '.join(args)})",
            "return_type": return_type,
            "docstring": docstring,
            "complexity": complexity,
            "parent_class": parent_class,
            "decorators": decorators,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "calls": calls
        }

    def _extract_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information"""

        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Count methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

        # Extract decorators
        decorators = [ast.unparse(d) if hasattr(ast, 'unparse') else str(d) for d in node.decorator_list]

        return {
            "type": "Class",
            "name": node.name,
            "qualified_name": node.name,
            "line_start": node.lineno,
            "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
            "bases": bases,
            "docstring": docstring,
            "methods": methods,
            "decorators": decorators
        }

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Each branch adds complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # and/or operators add complexity
                complexity += len(child.values) - 1

        return complexity

    def _extract_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls within a function"""
        calls = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(ast.unparse(child.func) if hasattr(ast, 'unparse') else str(child.func))

        return list(set(calls))  # Remove duplicates

    def _extract_relationships(self) -> List[Dict[str, Any]]:
        """Extract relationships between symbols"""
        relationships = []

        # Extract INHERITS relationships
        for symbol in self.symbols:
            if symbol["type"] == "Class" and symbol.get("bases"):
                for base in symbol["bases"]:
                    relationships.append({
                        "type": "INHERITS",
                        "from": symbol["name"],
                        "to": base,
                        "file": self.file_path
                    })

        # Extract CALLS relationships
        for symbol in self.symbols:
            if symbol["type"] in ["Function", "Method"] and symbol.get("calls"):
                for called in symbol["calls"]:
                    relationships.append({
                        "type": "CALLS",
                        "from": symbol["qualified_name"],
                        "to": called,
                        "file": self.file_path
                    })

        # Extract OVERRIDES relationships (methods with same name as parent)
        for symbol in self.symbols:
            if symbol["type"] == "Method" and symbol.get("parent_class"):
                # Check if parent class has this method
                parent_class = symbol["parent_class"]
                for other in self.symbols:
                    if other["type"] == "Class" and other["name"] == parent_class:
                        if other.get("bases"):
                            # Could check if method exists in base classes
                            # For now, mark common override patterns
                            if symbol["name"] in ["__init__", "__str__", "__repr__", "__eq__"]:
                                relationships.append({
                                    "type": "OVERRIDES",
                                    "from": symbol["qualified_name"],
                                    "to": f"object.{symbol['name']}",
                                    "file": self.file_path
                                })

        return relationships


class JavaScriptSymbolExtractor:
    """Extract symbols from JavaScript/TypeScript code (simplified)"""

    def extract_symbols(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Basic JavaScript/TypeScript symbol extraction using regex patterns
        Note: For production, use proper parser like Babel or TypeScript AST
        """
        import re

        symbols = []
        relationships = []

        # Extract functions
        function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer(function_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            symbols.append({
                "type": "Function",
                "name": match.group(1),
                "line_start": line_num,
                "qualified_name": match.group(1)
            })

        # Extract classes
        class_pattern = r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            class_name = match.group(1)
            base_class = match.group(2)

            symbols.append({
                "type": "Class",
                "name": class_name,
                "line_start": line_num,
                "qualified_name": class_name,
                "bases": [base_class] if base_class else []
            })

            if base_class:
                relationships.append({
                    "type": "INHERITS",
                    "from": class_name,
                    "to": base_class,
                    "file": file_path
                })

        # Extract arrow functions assigned to const/let/var
        arrow_pattern = r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>'
        for match in re.finditer(arrow_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            symbols.append({
                "type": "Function",
                "name": match.group(1),
                "line_start": line_num,
                "qualified_name": match.group(1)
            })

        # Extract React components (simplified)
        component_pattern = r'(?:export\s+)?(?:default\s+)?function\s+([A-Z]\w+)\s*\('
        for match in re.finditer(component_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            symbols.append({
                "type": "Component",
                "name": match.group(1),
                "line_start": line_num,
                "qualified_name": match.group(1)
            })

        # Extract imports (for USES relationships)
        import_pattern = r'import\s+(?:{[^}]+}|\w+)\s+from\s+[\'"]([^\'")]+)[\'"]'
        for match in re.finditer(import_pattern, content):
            relationships.append({
                "type": "IMPORTS",
                "from": file_path,
                "to": match.group(1),
                "file": file_path
            })

        return {
            "symbols": symbols,
            "relationships": relationships
        }


class SymbolExtractorFactory:
    """Factory to get appropriate symbol extractor for file type"""

    @staticmethod
    def get_extractor(file_path: str):
        """Get appropriate extractor based on file extension"""
        suffix = Path(file_path).suffix.lower()

        if suffix in ['.py', '.pyw']:
            return PythonSymbolExtractor()
        elif suffix in ['.js', '.jsx', '.ts', '.tsx', '.mjs']:
            return JavaScriptSymbolExtractor()
        else:
            return None

    @staticmethod
    def extract(content: str, file_path: str) -> Dict[str, Any]:
        """Extract symbols from file content"""
        extractor = SymbolExtractorFactory.get_extractor(file_path)

        if extractor:
            return extractor.extract_symbols(content, file_path)
        else:
            return {"symbols": [], "relationships": []}