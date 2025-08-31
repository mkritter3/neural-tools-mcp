#!/usr/bin/env python3
"""
Tree-sitter AST Analysis for L9 Enhanced System
Multi-language code structure analysis using tree-sitter
"""

import os
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import tree-sitter
try:
    import tree_sitter_languages as tsl
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter-languages not available. Install with: pip install tree-sitter-languages")

# Dummy classes for when tree-sitter is not available
if not TREE_SITTER_AVAILABLE:
    class Node:
        def __init__(self):
            self.type = ""
            self.text = b""
            self.children = []
            self.start_point = (0, 0)
            self.end_point = (0, 0)

@dataclass
class CodeStructure:
    """Represents the structure of analyzed code"""
    file_path: str
    language: str
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    imports: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    comments: List[Dict[str, Any]]
    complexity_score: int
    total_lines: int

@dataclass
class ASTPattern:
    """Represents a pattern found in AST analysis"""
    pattern_type: str
    name: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    text: str
    metadata: Dict[str, Any]

class TreeSitterAnalyzer:
    """Multi-language AST analyzer using tree-sitter"""
    
    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter-languages not available. Install with: pip install tree-sitter-languages")
        
        self.parsers = {}
        self.languages = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize parsers for supported languages"""
        supported_languages = {
            'python': 'python',
            'javascript': 'javascript', 
            'typescript': 'typescript',
            'rust': 'rust',
            'go': 'go',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'csharp': 'c_sharp',
            'ruby': 'ruby',
            'php': 'php',
            'kotlin': 'kotlin',
            'swift': 'swift'
        }
        
        for lang_name, ts_name in supported_languages.items():
            try:
                language = tsl.get_language(ts_name)
                parser = tsl.get_parser(ts_name)
                self.languages[lang_name] = language
                self.parsers[lang_name] = parser
                logger.debug(f"Initialized parser for {lang_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize parser for {lang_name}: {e}")
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.kt': 'kotlin',
            '.kts': 'kotlin',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift'
        }
        
        suffix = Path(file_path).suffix.lower()
        return extension_map.get(suffix)
    
    def analyze_file(self, file_path: str) -> Optional[CodeStructure]:
        """Analyze a single file and extract its structure"""
        try:
            language = self.detect_language(file_path)
            if not language or language not in self.parsers:
                logger.debug(f"Unsupported language for file: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            return self.analyze_source(source_code, language, file_path)
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return None
    
    def analyze_source(self, source_code: str, language: str, file_path: str = "") -> Optional[CodeStructure]:
        """Analyze source code and extract structure"""
        if language not in self.parsers:
            return None
            
        try:
            parser = self.parsers[language]
            tree = parser.parse(bytes(source_code, "utf8"))
            
            # Extract different code elements
            functions = self._extract_functions(tree.root_node, source_code, language)
            classes = self._extract_classes(tree.root_node, source_code, language)
            imports = self._extract_imports(tree.root_node, source_code, language)
            variables = self._extract_variables(tree.root_node, source_code, language)
            comments = self._extract_comments(tree.root_node, source_code, language)
            
            # Calculate complexity
            complexity = self._calculate_complexity(tree.root_node, language)
            total_lines = len(source_code.split('\n'))
            
            return CodeStructure(
                file_path=file_path,
                language=language,
                functions=functions,
                classes=classes,
                imports=imports,
                variables=variables,
                comments=comments,
                complexity_score=complexity,
                total_lines=total_lines
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze source code: {e}")
            return None
    
    def _extract_functions(self, root_node, source_code: str, language: str) -> List[Dict[str, Any]]:
        """Extract function definitions from AST"""
        functions = []
        
        function_queries = {
            'python': ['function_definition'],
            'javascript': ['function_declaration', 'method_definition', 'arrow_function'],
            'typescript': ['function_declaration', 'method_definition', 'arrow_function'],
            'rust': ['function_item'],
            'go': ['function_declaration'],
            'java': ['method_declaration'],
            'cpp': ['function_definition'],
            'c': ['function_definition'],
            'csharp': ['method_declaration'],
            'ruby': ['method'],
            'php': ['function_definition', 'method_declaration'],
            'kotlin': ['function_declaration'],
            'swift': ['function_declaration']
        }
        
        query_types = function_queries.get(language, ['function_definition'])
        
        for node in self._walk_tree(root_node):
            if node.type in query_types:
                func_info = self._extract_function_info(node, source_code, language)
                if func_info:
                    functions.append(func_info)
        
        return functions
    
    def _extract_function_info(self, node, source_code: str, language: str) -> Dict[str, Any]:
        """Extract detailed information about a function"""
        try:
            name_node = None
            params_node = None
            body_node = None
            
            # Language-specific node finding
            for child in node.children:
                if child.type == 'identifier' and not name_node:
                    name_node = child
                elif 'parameter' in child.type or 'argument' in child.type:
                    params_node = child
                elif 'body' in child.type or 'block' in child.type:
                    body_node = child
            
            function_name = name_node.text.decode('utf-8') if name_node else 'anonymous'
            
            # Extract parameters
            parameters = []
            if params_node:
                for param_node in params_node.children:
                    if param_node.type == 'identifier':
                        parameters.append(param_node.text.decode('utf-8'))
            
            # Calculate function complexity
            complexity = self._calculate_node_complexity(body_node if body_node else node)
            
            return {
                'name': function_name,
                'parameters': parameters,
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1,
                'column_start': node.start_point[1],
                'column_end': node.end_point[1],
                'complexity': complexity,
                'text': node.text.decode('utf-8')[:200] + '...' if len(node.text) > 200 else node.text.decode('utf-8')
            }
        except Exception as e:
            logger.debug(f"Failed to extract function info: {e}")
            return None
    
    def _extract_classes(self, root_node, source_code: str, language: str) -> List[Dict[str, Any]]:
        """Extract class definitions from AST"""
        classes = []
        
        class_queries = {
            'python': ['class_definition'],
            'javascript': ['class_declaration'],
            'typescript': ['class_declaration'],
            'rust': ['struct_item', 'enum_item'],
            'go': ['type_declaration'],
            'java': ['class_declaration', 'interface_declaration'],
            'cpp': ['class_specifier', 'struct_specifier'],
            'c': ['struct_specifier'],
            'csharp': ['class_declaration', 'interface_declaration'],
            'ruby': ['class'],
            'php': ['class_declaration'],
            'kotlin': ['class_declaration'],
            'swift': ['class_declaration', 'struct_declaration']
        }
        
        query_types = class_queries.get(language, ['class_definition'])
        
        for node in self._walk_tree(root_node):
            if node.type in query_types:
                class_info = self._extract_class_info(node, source_code, language)
                if class_info:
                    classes.append(class_info)
        
        return classes
    
    def _extract_class_info(self, node, source_code: str, language: str) -> Dict[str, Any]:
        """Extract detailed information about a class"""
        try:
            name_node = None
            for child in node.children:
                if child.type == 'identifier' and not name_node:
                    name_node = child
                    break
            
            class_name = name_node.text.decode('utf-8') if name_node else 'anonymous'
            
            return {
                'name': class_name,
                'line_start': node.start_point[0] + 1,
                'line_end': node.end_point[0] + 1,
                'column_start': node.start_point[1],
                'column_end': node.end_point[1],
                'methods': self._count_methods_in_class(node, language)
            }
        except Exception as e:
            logger.debug(f"Failed to extract class info: {e}")
            return None
    
    def _extract_imports(self, root_node, source_code: str, language: str) -> List[Dict[str, Any]]:
        """Extract import statements from AST"""
        imports = []
        
        import_queries = {
            'python': ['import_statement', 'import_from_statement'],
            'javascript': ['import_statement'],
            'typescript': ['import_statement'],
            'rust': ['use_declaration'],
            'go': ['import_spec', 'import_declaration'],
            'java': ['import_declaration'],
            'cpp': ['preproc_include'],
            'c': ['preproc_include'],
            'csharp': ['using_directive'],
            'ruby': ['method_call'],  # for require statements
            'php': ['include_expression', 'require_expression'],
            'kotlin': ['import_header'],
            'swift': ['import_declaration']
        }
        
        query_types = import_queries.get(language, ['import_statement'])
        
        for node in self._walk_tree(root_node):
            if node.type in query_types:
                import_info = self._extract_import_info(node, source_code, language)
                if import_info:
                    imports.append(import_info)
        
        return imports
    
    def _extract_import_info(self, node, source_code: str, language: str) -> Dict[str, Any]:
        """Extract information about an import statement"""
        try:
            import_text = node.text.decode('utf-8')
            
            return {
                'text': import_text,
                'line': node.start_point[0] + 1,
                'column': node.start_point[1]
            }
        except Exception as e:
            logger.debug(f"Failed to extract import info: {e}")
            return None
    
    def _extract_variables(self, root_node, source_code: str, language: str) -> List[Dict[str, Any]]:
        """Extract variable declarations from AST"""
        variables = []
        
        # This is simplified - in a full implementation, you'd want more sophisticated variable tracking
        variable_queries = {
            'python': ['assignment'],
            'javascript': ['variable_declaration', 'lexical_declaration'],
            'typescript': ['variable_declaration', 'lexical_declaration'],
            'rust': ['let_declaration'],
            'go': ['var_declaration', 'short_var_declaration'],
            'java': ['variable_declarator'],
            'cpp': ['declaration'],
            'c': ['declaration']
        }
        
        query_types = variable_queries.get(language, [])
        
        for node in self._walk_tree(root_node):
            if node.type in query_types:
                var_info = self._extract_variable_info(node, source_code, language)
                if var_info:
                    variables.append(var_info)
        
        return variables
    
    def _extract_variable_info(self, node, source_code: str, language: str) -> Dict[str, Any]:
        """Extract information about a variable declaration"""
        try:
            return {
                'line': node.start_point[0] + 1,
                'column': node.start_point[1],
                'text': node.text.decode('utf-8')[:100] + '...' if len(node.text) > 100 else node.text.decode('utf-8')
            }
        except Exception as e:
            logger.debug(f"Failed to extract variable info: {e}")
            return None
    
    def _extract_comments(self, root_node, source_code: str, language: str) -> List[Dict[str, Any]]:
        """Extract comments from AST"""
        comments = []
        
        comment_queries = {
            'python': ['comment'],
            'javascript': ['comment'],
            'typescript': ['comment'],
            'rust': ['line_comment', 'block_comment'],
            'go': ['comment'],
            'java': ['line_comment', 'block_comment'],
            'cpp': ['comment'],
            'c': ['comment']
        }
        
        query_types = comment_queries.get(language, ['comment'])
        
        for node in self._walk_tree(root_node):
            if node.type in query_types:
                comment_info = {
                    'line': node.start_point[0] + 1,
                    'column': node.start_point[1],
                    'text': node.text.decode('utf-8')
                }
                comments.append(comment_info)
        
        return comments
    
    def _calculate_complexity(self, root_node, language: str) -> int:
        """Calculate cyclomatic complexity of the code"""
        complexity = 1  # Base complexity
        
        complexity_nodes = {
            'python': ['if_statement', 'for_statement', 'while_statement', 'try_statement'],
            'javascript': ['if_statement', 'for_statement', 'while_statement', 'switch_statement'],
            'typescript': ['if_statement', 'for_statement', 'while_statement', 'switch_statement'],
            'rust': ['if_expression', 'loop_expression', 'while_expression', 'match_expression'],
            'go': ['if_statement', 'for_statement', 'switch_statement'],
            'java': ['if_statement', 'for_statement', 'while_statement', 'switch_expression']
        }
        
        complexity_types = complexity_nodes.get(language, ['if_statement', 'for_statement', 'while_statement'])
        
        for node in self._walk_tree(root_node):
            if node.type in complexity_types:
                complexity += 1
        
        return complexity
    
    def _calculate_node_complexity(self, node) -> int:
        """Calculate complexity for a specific node"""
        if not node:
            return 1
            
        complexity = 1
        for child in self._walk_tree(node):
            if child.type in ['if_statement', 'for_statement', 'while_statement']:
                complexity += 1
        
        return complexity
    
    def _count_methods_in_class(self, class_node, language: str) -> int:
        """Count the number of methods in a class"""
        method_count = 0
        method_types = ['function_definition', 'method_declaration', 'method_definition']
        
        for child in self._walk_tree(class_node):
            if child.type in method_types:
                method_count += 1
        
        return method_count
    
    def _walk_tree(self, node):
        """Recursively walk through AST nodes"""
        yield node
        for child in node.children:
            yield from self._walk_tree(child)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages"""
        return list(self.parsers.keys())
    
    def analyze_project(self, project_path: str, file_extensions: Optional[Set[str]] = None) -> List[CodeStructure]:
        """Analyze all supported files in a project directory"""
        results = []
        
        if file_extensions is None:
            file_extensions = {'.py', '.js', '.ts', '.rs', '.go', '.java', '.cpp', '.c', '.cs', '.rb', '.php', '.kt', '.swift'}
        
        for root, dirs, files in os.walk(project_path):
            # Skip common build/cache directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', 'target', 'build', '.vscode'}]
            
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    result = self.analyze_file(file_path)
                    if result:
                        results.append(result)
        
        return results