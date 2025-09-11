#!/usr/bin/env python3
"""
Code Parser with Tree-sitter for Structure Extraction
Implements ADR 0017: GraphRAG True Hybrid Search
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re
try:
    import tree_sitter
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_typescript as tstypescript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("Tree-sitter not available. Structure extraction will be limited.")

logger = logging.getLogger(__name__)


class CodeParser:
    """
    Extract structured information from code using tree-sitter
    Supports Python, JavaScript, TypeScript for comprehensive GraphRAG
    """
    
    def __init__(self):
        self.parsers = {}
        self.languages = {}
        
        if TREE_SITTER_AVAILABLE:
            try:
                # Initialize Python parser
                PY_LANGUAGE = tree_sitter.Language(tspython.language())
                self.languages['.py'] = PY_LANGUAGE
                self.parsers['.py'] = tree_sitter.Parser(PY_LANGUAGE)
                
                # Initialize JavaScript parser  
                JS_LANGUAGE = tree_sitter.Language(tsjavascript.language())
                self.languages['.js'] = JS_LANGUAGE
                self.parsers['.js'] = self.parsers['.jsx'] = tree_sitter.Parser(JS_LANGUAGE)
                
                # Initialize TypeScript parser
                TS_LANGUAGE = tree_sitter.Language(tstypescript.language_typescript())
                self.languages['.ts'] = TS_LANGUAGE
                self.parsers['.ts'] = self.parsers['.tsx'] = tree_sitter.Parser(TS_LANGUAGE)
                
                logger.info("Tree-sitter parsers initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize tree-sitter: {e}")
                self.parsers = {}
        
        # Fallback regex patterns for when tree-sitter isn't available
        self.import_patterns = {
            '.py': [
                r'^import\s+([\w\.]+)',
                r'^from\s+([\w\.]+)\s+import',
            ],
            '.js': [
                r'^import\s+.*\s+from\s+[\'"]([^\'\"]+)[\'"]',
                r'^const\s+.*\s+=\s+require\([\'"]([^\'\"]+)[\'"]\)',
            ],
            '.ts': [
                r'^import\s+.*\s+from\s+[\'"]([^\'\"]+)[\'"]',
                r'^import\s+type\s+.*\s+from\s+[\'"]([^\'\"]+)[\'"]',
            ]
        }
        
        self.function_patterns = {
            '.py': [
                r'^(?:async\s+)?def\s+(\w+)\s*\([^)]*\)',
                r'^class\s+(\w+)',
            ],
            '.js': [
                r'^(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
                r'^const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
                r'^class\s+(\w+)',
            ],
            '.ts': [
                r'^(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
                r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)',
                r'^class\s+(\w+)',
                r'^interface\s+(\w+)',
            ]
        }
    
    @property
    def supported_extensions(self) -> set:
        """Get supported file extensions"""
        return {'.py', '.js', '.jsx', '.ts', '.tsx'}
    
    def extract_structure(self, content: str, file_ext: str) -> Dict[str, Any]:
        """
        Extract structured information from code
        
        Returns:
            {
                'imports': List of imported modules/files,
                'functions': List of function definitions,
                'classes': List of class definitions,
                'calls': List of function calls,
                'exports': List of exported items
            }
        """
        if not file_ext in self.supported_extensions:
            return {}
        
        # Try tree-sitter first
        if file_ext in self.parsers:
            try:
                return self._extract_with_tree_sitter(content, file_ext)
            except Exception as e:
                logger.warning(f"Tree-sitter extraction failed, falling back to regex: {e}")
        
        # Fallback to regex
        return self._extract_with_regex(content, file_ext)
    
    def _extract_with_tree_sitter(self, content: str, file_ext: str) -> Dict[str, Any]:
        """Extract structure using tree-sitter AST parsing"""
        parser = self.parsers[file_ext]
        tree = parser.parse(bytes(content, 'utf8'))
        
        result = {
            'imports': [],
            'functions': [],
            'classes': [],
            'calls': [],
            'exports': []
        }
        
        # Walk the AST
        cursor = tree.walk()
        
        def visit_node(cursor):
            node = cursor.node
            
            # Python imports
            if file_ext == '.py':
                if node.type == 'import_statement':
                    import_text = content[node.start_byte:node.end_byte]
                    result['imports'].append({
                        'statement': import_text,
                        'line': node.start_point[0] + 1
                    })
                elif node.type == 'import_from_statement':
                    import_text = content[node.start_byte:node.end_byte]
                    result['imports'].append({
                        'statement': import_text,
                        'line': node.start_point[0] + 1
                    })
                elif node.type == 'function_definition':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        func_name = content[name_node.start_byte:name_node.end_byte]
                        result['functions'].append({
                            'name': func_name,
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1,
                            'signature': content[node.start_byte:node.end_byte].split('\n')[0]
                        })
                elif node.type == 'class_definition':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        class_name = content[name_node.start_byte:name_node.end_byte]
                        result['classes'].append({
                            'name': class_name,
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1
                        })
                elif node.type == 'call':
                    func_node = node.child_by_field_name('function')
                    if func_node:
                        call_name = content[func_node.start_byte:func_node.end_byte]
                        result['calls'].append({
                            'name': call_name,
                            'line': node.start_point[0] + 1
                        })
            
            # JavaScript/TypeScript imports
            elif file_ext in ['.js', '.jsx', '.ts', '.tsx']:
                if node.type == 'import_statement':
                    import_text = content[node.start_byte:node.end_byte]
                    result['imports'].append({
                        'statement': import_text,
                        'line': node.start_point[0] + 1
                    })
                elif node.type in ['function_declaration', 'arrow_function']:
                    # Extract function name if available
                    if node.type == 'function_declaration':
                        name_node = node.child_by_field_name('name')
                        if name_node:
                            func_name = content[name_node.start_byte:name_node.end_byte]
                            result['functions'].append({
                                'name': func_name,
                                'start_line': node.start_point[0] + 1,
                                'end_line': node.end_point[0] + 1,
                                'signature': content[node.start_byte:node.end_byte].split('\n')[0]
                            })
                elif node.type == 'class_declaration':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        class_name = content[name_node.start_byte:name_node.end_byte]
                        result['classes'].append({
                            'name': class_name,
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1
                        })
                elif node.type == 'call_expression':
                    func_node = node.child_by_field_name('function')
                    if func_node:
                        call_name = content[func_node.start_byte:func_node.end_byte]
                        result['calls'].append({
                            'name': call_name,
                            'line': node.start_point[0] + 1
                        })
                elif node.type == 'export_statement':
                    export_text = content[node.start_byte:node.end_byte]
                    result['exports'].append({
                        'statement': export_text,
                        'line': node.start_point[0] + 1
                    })
            
            # Recursively visit children
            if cursor.goto_first_child():
                visit_node(cursor)
                while cursor.goto_next_sibling():
                    visit_node(cursor)
                cursor.goto_parent()
        
        visit_node(cursor)
        return result
    
    def _extract_with_regex(self, content: str, file_ext: str) -> Dict[str, Any]:
        """Fallback regex-based extraction"""
        result = {
            'imports': [],
            'functions': [],
            'classes': [],
            'calls': [],
            'exports': []
        }
        
        lines = content.split('\n')
        
        # Extract imports
        import_patterns = self.import_patterns.get(file_ext, [])
        for i, line in enumerate(lines):
            for pattern in import_patterns:
                match = re.search(pattern, line)
                if match:
                    result['imports'].append({
                        'statement': line.strip(),
                        'line': i + 1,
                        'module': match.group(1) if match.groups() else None
                    })
        
        # Extract functions and classes
        function_patterns = self.function_patterns.get(file_ext, [])
        for i, line in enumerate(lines):
            for pattern in function_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    if 'class' in pattern:
                        result['classes'].append({
                            'name': name,
                            'start_line': i + 1,
                            'end_line': i + 1  # Can't determine end with regex
                        })
                    else:
                        result['functions'].append({
                            'name': name,
                            'start_line': i + 1,
                            'end_line': i + 1,  # Can't determine end with regex
                            'signature': line.strip()
                        })
        
        # Simple call detection (very basic)
        call_pattern = r'(\w+)\s*\('
        for i, line in enumerate(lines):
            for match in re.finditer(call_pattern, line):
                func_name = match.group(1)
                # Filter out keywords
                if func_name not in ['if', 'for', 'while', 'with', 'def', 'class', 'function', 'async']:
                    result['calls'].append({
                        'name': func_name,
                        'line': i + 1
                    })
        
        return result
    
    def extract_imports_for_file(self, file_path: Path) -> List[str]:
        """Extract just the import statements from a file"""
        try:
            content = file_path.read_text()
            file_ext = file_path.suffix
            structure = self.extract_structure(content, file_ext)
            return [imp.get('module', imp.get('statement', '')) 
                   for imp in structure.get('imports', [])]
        except Exception as e:
            logger.error(f"Failed to extract imports from {file_path}: {e}")
            return []


# Testing
if __name__ == "__main__":
    parser = CodeParser()
    
    # Test Python code
    python_code = '''
import os
from pathlib import Path
from typing import List, Dict

class MyClass:
    def __init__(self):
        self.value = 42
    
    def process(self, data: List[str]) -> Dict:
        result = {}
        for item in data:
            result[item] = len(item)
        return result

async def fetch_data(url: str):
    response = await http_client.get(url)
    return response.json()
'''
    
    result = parser.extract_structure(python_code, '.py')
    print("Python extraction:", result)