#!/usr/bin/env python3
"""
L9 Tree-sitter AST Analysis - Multi-Language Code Understanding
Replaces basic Python ast module with full tree-sitter support
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from tree_sitter_languages import get_language, get_parser
    from tree_sitter import Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Define dummy types for when tree-sitter is not available
    class Node:
        pass
    def get_language(name):
        return None
    def get_parser(name):
        return None

logger = logging.getLogger(__name__)

class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript" 
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    CSS = "css"
    HTML = "html"
    JSON = "json"
    YAML = "yaml"
    BASH = "bash"
    SQL = "sql"
    UNKNOWN = "unknown"

@dataclass
class ASTPattern:
    """Structured AST pattern for semantic search"""
    pattern_type: str  # function, class, import, etc.
    name: str
    context: str  # surrounding context
    line_range: Tuple[int, int]
    metadata: Dict[str, Any]

@dataclass
class CodeStructure:
    """Complete code structure analysis"""
    file_path: str
    language: LanguageType
    functions: List[ASTPattern]
    classes: List[ASTPattern] 
    imports: List[ASTPattern]
    variables: List[ASTPattern]
    comments: List[ASTPattern]
    complexity_score: float
    dependencies: List[str]

class TreeSitterAnalyzer:
    """Advanced tree-sitter based code analysis"""
    
    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter-languages not available. Install with: pip install tree-sitter-languages")
        
        self.parsers = {}
        self.languages = {}
        self._initialize_parsers()
        
        # File extension mappings
        self.extension_map = {
            '.py': LanguageType.PYTHON,
            '.js': LanguageType.JAVASCRIPT,
            '.mjs': LanguageType.JAVASCRIPT,
            '.ts': LanguageType.TYPESCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.java': LanguageType.JAVA,
            '.go': LanguageType.GO,
            '.rs': LanguageType.RUST,
            '.cpp': LanguageType.CPP,
            '.cc': LanguageType.CPP,
            '.cxx': LanguageType.CPP,
            '.c': LanguageType.C,
            '.h': LanguageType.C,
            '.css': LanguageType.CSS,
            '.scss': LanguageType.CSS,
            '.html': LanguageType.HTML,
            '.htm': LanguageType.HTML,
            '.json': LanguageType.JSON,
            '.yml': LanguageType.YAML,
            '.yaml': LanguageType.YAML,
            '.sh': LanguageType.BASH,
            '.bash': LanguageType.BASH,
            '.sql': LanguageType.SQL,
        }
    
    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages"""
        language_mapping = {
            LanguageType.PYTHON: 'python',
            LanguageType.JAVASCRIPT: 'javascript', 
            LanguageType.TYPESCRIPT: 'typescript',
            LanguageType.JAVA: 'java',
            LanguageType.GO: 'go',
            LanguageType.RUST: 'rust',
            LanguageType.CPP: 'cpp',
            LanguageType.C: 'c',
            LanguageType.CSS: 'css',
            LanguageType.HTML: 'html',
            LanguageType.JSON: 'json',
            LanguageType.YAML: 'yaml',
            LanguageType.BASH: 'bash',
        }
        
        for lang_type, lang_name in language_mapping.items():
            try:
                # Simplified tree-sitter-languages API
                language = get_language(lang_name)
                parser = get_parser(lang_name)
                
                self.languages[lang_type] = language
                self.parsers[lang_type] = parser
                logger.debug(f"✅ Initialized parser for {lang_name}")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize {lang_name} parser: {e}")
    
    def detect_language(self, file_path: str) -> LanguageType:
        """Detect programming language from file extension"""
        path = Path(file_path)
        extension = path.suffix.lower()
        return self.extension_map.get(extension, LanguageType.UNKNOWN)
    
    def parse_code(self, code: str, language: LanguageType) -> Optional[Node]:
        """Parse code into AST tree"""
        if language not in self.parsers:
            return None
            
        try:
            parser = self.parsers[language]
            tree = parser.parse(bytes(code, 'utf8'))
            return tree.root_node
        except Exception as e:
            logger.warning(f"Failed to parse {language.value} code: {e}")
            return None
    
    def extract_functions(self, node: Node, code_bytes: bytes, language: LanguageType) -> List[ASTPattern]:
        """Extract function definitions"""
        functions = []
        
        # Language-specific function query patterns
        function_queries = {
            LanguageType.PYTHON: '(function_def name: (identifier) @name)',
            LanguageType.JAVASCRIPT: '(function_declaration name: (identifier) @name)',
            LanguageType.TYPESCRIPT: '[(function_declaration name: (identifier) @name) (method_definition name: (property_identifier) @name)]',
            LanguageType.GO: '(function_declaration name: (identifier) @name)',
            LanguageType.RUST: '(function_item name: (identifier) @name)',
            LanguageType.JAVA: '(method_declaration name: (identifier) @name)',
            LanguageType.CPP: '(function_definition declarator: (function_declarator declarator: (identifier) @name))',
        }
        
        query_str = function_queries.get(language)
        if not query_str:
            return functions
            
        try:
            query = self.languages[language].query(query_str)
            captures = query.captures(node)
            
            for captured_node, _ in captures:
                name = code_bytes[captured_node.start_byte:captured_node.end_byte].decode('utf8')
                
                # Get the full function node
                func_node = captured_node.parent
                while func_node and func_node.type not in ['function_def', 'function_declaration', 'method_declaration', 'function_item']:
                    func_node = func_node.parent
                
                if func_node:
                    start_line = func_node.start_point[0] + 1
                    end_line = func_node.end_point[0] + 1
                    
                    # Extract context (docstring, comments)
                    context = self._extract_context(func_node, code_bytes)
                    
                    functions.append(ASTPattern(
                        pattern_type="function",
                        name=name,
                        context=context,
                        line_range=(start_line, end_line),
                        metadata={
                            "language": language.value,
                            "node_type": func_node.type,
                            "byte_range": (func_node.start_byte, func_node.end_byte)
                        }
                    ))
                    
        except Exception as e:
            logger.warning(f"Failed to extract functions for {language.value}: {e}")
            
        return functions
    
    def extract_classes(self, node: Node, code_bytes: bytes, language: LanguageType) -> List[ASTPattern]:
        """Extract class definitions"""
        classes = []
        
        class_queries = {
            LanguageType.PYTHON: '(class_definition name: (identifier) @name)',
            LanguageType.JAVASCRIPT: '(class_declaration name: (identifier) @name)',
            LanguageType.TYPESCRIPT: '(class_declaration name: (type_identifier) @name)',
            LanguageType.JAVA: '(class_declaration name: (identifier) @name)',
            LanguageType.RUST: '[(struct_item name: (type_identifier) @name) (impl_item type: (type_identifier) @name)]',
            LanguageType.CPP: '(class_specifier name: (type_identifier) @name)',
        }
        
        query_str = class_queries.get(language)
        if not query_str:
            return classes
            
        try:
            query = self.languages[language].query(query_str)
            captures = query.captures(node)
            
            for captured_node, _ in captures:
                name = code_bytes[captured_node.start_byte:captured_node.end_byte].decode('utf8')
                
                class_node = captured_node.parent
                while class_node and class_node.type not in ['class_definition', 'class_declaration', 'struct_item', 'impl_item', 'class_specifier']:
                    class_node = class_node.parent
                
                if class_node:
                    start_line = class_node.start_point[0] + 1
                    end_line = class_node.end_point[0] + 1
                    
                    context = self._extract_context(class_node, code_bytes)
                    
                    classes.append(ASTPattern(
                        pattern_type="class",
                        name=name,
                        context=context,
                        line_range=(start_line, end_line),
                        metadata={
                            "language": language.value,
                            "node_type": class_node.type,
                            "byte_range": (class_node.start_byte, class_node.end_byte)
                        }
                    ))
                    
        except Exception as e:
            logger.warning(f"Failed to extract classes for {language.value}: {e}")
            
        return classes
    
    def extract_imports(self, node: Node, code_bytes: bytes, language: LanguageType) -> List[ASTPattern]:
        """Extract import statements"""
        imports = []
        
        import_queries = {
            LanguageType.PYTHON: '[(import_statement) (import_from_statement)]',
            LanguageType.JAVASCRIPT: '[(import_statement) (import_clause)]',
            LanguageType.TYPESCRIPT: '[(import_statement) (import_clause)]',
            LanguageType.GO: '(import_declaration)',
            LanguageType.RUST: '(use_declaration)',
            LanguageType.JAVA: '(import_declaration)',
            LanguageType.CPP: '(preproc_include)',
        }
        
        query_str = import_queries.get(language)
        if not query_str:
            return imports
            
        try:
            query = self.languages[language].query(query_str)
            captures = query.captures(node)
            
            for import_node, _ in captures:
                import_text = code_bytes[import_node.start_byte:import_node.end_byte].decode('utf8')
                start_line = import_node.start_point[0] + 1
                end_line = import_node.end_point[0] + 1
                
                imports.append(ASTPattern(
                    pattern_type="import",
                    name=import_text.strip(),
                    context="",
                    line_range=(start_line, end_line),
                    metadata={
                        "language": language.value,
                        "node_type": import_node.type,
                        "byte_range": (import_node.start_byte, import_node.end_byte)
                    }
                ))
                
        except Exception as e:
            logger.warning(f"Failed to extract imports for {language.value}: {e}")
            
        return imports
    
    def _extract_context(self, node: Node, code_bytes: bytes, max_chars: int = 200) -> str:
        """Extract surrounding context for a node (comments, docstrings)"""
        try:
            # Look for preceding comments or docstrings
            full_text = code_bytes[max(0, node.start_byte - max_chars):node.start_byte].decode('utf8')
            
            # Extract meaningful context (comments, docstrings)
            lines = full_text.split('\n')
            context_lines = []
            
            for line in reversed(lines):
                stripped = line.strip()
                if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                    context_lines.append(stripped)
                elif stripped.startswith('"""') or stripped.startswith("'''"):
                    context_lines.append(stripped)
                elif not stripped and context_lines:
                    break
                elif stripped and not context_lines:
                    break
                    
            return ' '.join(reversed(context_lines)) if context_lines else ""
            
        except:
            return ""
    
    def analyze_file(self, file_path: str) -> Optional[CodeStructure]:
        """Complete analysis of a code file"""
        try:
            # Read file
            path = Path(file_path)
            if not path.exists():
                return None
                
            code = path.read_text(encoding='utf-8')
            code_bytes = code.encode('utf8')
            
            # Detect language
            language = self.detect_language(file_path)
            if language == LanguageType.UNKNOWN or language not in self.parsers:
                return None
            
            # Parse AST
            root_node = self.parse_code(code, language)
            if not root_node:
                return None
            
            # Extract patterns
            functions = self.extract_functions(root_node, code_bytes, language)
            classes = self.extract_classes(root_node, code_bytes, language)
            imports = self.extract_imports(root_node, code_bytes, language)
            
            # Calculate complexity (simple metric based on nesting and patterns)
            complexity = self._calculate_complexity(root_node)
            
            # Extract dependencies from imports
            dependencies = [imp.name for imp in imports]
            
            return CodeStructure(
                file_path=str(path),
                language=language,
                functions=functions,
                classes=classes,
                imports=imports,
                variables=[],  # Could be extended
                comments=[],   # Could be extended
                complexity_score=complexity,
                dependencies=dependencies
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _calculate_complexity(self, node: Node) -> float:
        """Simple complexity calculation based on AST structure"""
        complexity = 0
        
        def traverse(n, depth=0):
            nonlocal complexity
            
            # Add complexity based on control structures
            if n.type in ['if_statement', 'for_statement', 'while_statement', 'try_statement']:
                complexity += 1 + (depth * 0.1)
            elif n.type in ['function_def', 'method_declaration', 'class_definition']:
                complexity += 0.5 + (depth * 0.1)
                
            # Traverse children
            for child in n.children:
                traverse(child, depth + 1)
        
        traverse(node)
        return round(complexity, 2)
    
    def extract_searchable_patterns(self, structure: CodeStructure) -> List[str]:
        """Extract patterns optimized for search"""
        patterns = []
        
        # Add function patterns
        for func in structure.functions:
            patterns.append(f"function:{func.name}")
            if func.context:
                patterns.append(f"context:{func.context}")
                
        # Add class patterns
        for cls in structure.classes:
            patterns.append(f"class:{cls.name}")
            if cls.context:
                patterns.append(f"context:{cls.context}")
                
        # Add import patterns  
        for imp in structure.imports:
            patterns.append(f"import:{imp.name}")
            
        # Add language pattern
        patterns.append(f"language:{structure.language.value}")
        
        # Add complexity pattern
        if structure.complexity_score > 5:
            patterns.append("complexity:high")
        elif structure.complexity_score > 2:
            patterns.append("complexity:medium")
        else:
            patterns.append("complexity:low")
            
        return patterns

# Example usage and testing
if __name__ == "__main__":
    if not TREE_SITTER_AVAILABLE:
        print("❌ tree-sitter-languages not installed")
        print("Install with: pip install tree-sitter-languages")
        exit(1)
    
    analyzer = TreeSitterAnalyzer()
    
    # Test with a Python file
    test_code = '''
def hello_world(name: str) -> str:
    """Greet a person by name"""
    if not name:
        return "Hello, World!"
    return f"Hello, {name}!"

class Greeter:
    """A class for greeting people"""
    
    def __init__(self, greeting="Hello"):
        self.greeting = greeting
    
    def greet(self, name):
        return f"{self.greeting}, {name}!"
'''
    
    # Write test file
    test_file = Path("/tmp/test_code.py")
    test_file.write_text(test_code)
    
    # Analyze
    structure = analyzer.analyze_file(str(test_file))
    if structure:
        print(f"✅ Analyzed {structure.language.value} file:")
        print(f"  Functions: {[f.name for f in structure.functions]}")
        print(f"  Classes: {[c.name for c in structure.classes]}")
        print(f"  Complexity: {structure.complexity_score}")
        
        patterns = analyzer.extract_searchable_patterns(structure)
        print(f"  Search patterns: {patterns[:5]}...")
    
    # Clean up
    test_file.unlink()
    
    print("\n🎯 Tree-sitter analyzer ready for multi-language code analysis!")