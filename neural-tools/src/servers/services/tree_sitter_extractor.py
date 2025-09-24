"""
Tree-sitter based code structure extraction
Implements Codex P2 requirement for symbol extraction
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser, Node

logger = logging.getLogger(__name__)

class TreeSitterExtractor:
    """Extract code symbols using tree-sitter for enhanced semantic search"""
    
    def __init__(self):
        # Build language parsers
        self.languages = {
            '.py': Language(tspython.language()),
            '.js': Language(tsjavascript.language()),
            '.jsx': Language(tsjavascript.language()),
            '.ts': Language(tstypescript.language_typescript()),
            '.tsx': Language(tstypescript.language_tsx()),
        }
        
        self.parsers = {}
        for ext, lang in self.languages.items():
            parser = Parser(lang)  # New API: pass language to constructor
            self.parsers[ext] = parser
        
        # Executor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'symbols_extracted': 0,
            'parse_errors': 0,
            'languages': {}
        }
    
    async def extract_symbols_from_file(
        self, 
        file_path: str, 
        content: str,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """
        Extract symbols from a single file with timeout protection
        
        Returns:
            {
                'symbols': List of extracted symbols,
                'error': Error message if any,
                'stats': Extraction statistics
            }
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext not in self.parsers:
            return {
                'symbols': [],
                'error': f'Unsupported file type: {ext}',
                'stats': {'skipped': True}
            }
        
        try:
            # Run extraction in executor with timeout
            future = self.executor.submit(
                self._extract_symbols_sync,
                content,
                ext,
                file_path
            )
            
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, future.result),
                timeout=timeout
            )
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['symbols_extracted'] += len(result['symbols'])
            
            lang_key = ext.lstrip('.')
            if lang_key not in self.stats['languages']:
                self.stats['languages'][lang_key] = {
                    'files': 0,
                    'symbols': 0,
                    'classes': 0,
                    'functions': 0
                }
            
            self.stats['languages'][lang_key]['files'] += 1
            self.stats['languages'][lang_key]['symbols'] += len(result['symbols'])
            
            for symbol in result['symbols']:
                if symbol['type'] == 'class':
                    self.stats['languages'][lang_key]['classes'] += 1
                elif symbol['type'] == 'function':
                    self.stats['languages'][lang_key]['functions'] += 1
            
            return result
            
        except asyncio.TimeoutError:
            self.stats['parse_errors'] += 1
            logger.warning(f"Parse timeout for {file_path}")
            return {
                'symbols': [],
                'error': f'Parse timeout after {timeout}s',
                'stats': {'timeout': True}
            }
        except Exception as e:
            self.stats['parse_errors'] += 1
            logger.error(f"Parse error for {file_path}: {e}")
            return {
                'symbols': [],
                'error': str(e),
                'stats': {'error': True}
            }
    
    def _extract_symbols_sync(
        self, 
        content: str, 
        ext: str, 
        file_path: str
    ) -> Dict[str, Any]:
        """Synchronous symbol extraction"""
        parser = self.parsers[ext]
        tree = parser.parse(bytes(content, 'utf-8'))
        root = tree.root_node
        
        symbols = []
        
        if ext == '.py':
            symbols = self._extract_python_symbols(root, content, file_path)
        elif ext in ['.js', '.jsx']:
            symbols = self._extract_javascript_symbols(root, content, file_path)
        elif ext in ['.ts', '.tsx']:
            symbols = self._extract_typescript_symbols(root, content, file_path)
        
        return {
            'symbols': symbols,
            'error': None,
            'stats': {
                'symbol_count': len(symbols),
                'language': ext.lstrip('.')
            }
        }
    
    def _extract_python_symbols(
        self,
        node: Node,
        source: str,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Extract symbols and relationships from Python code using enhanced tree-sitter patterns"""
        symbols = []
        relationships = []  # ADR-0075: Track relationships for graph construction

        def traverse(node: Node, parent_class: Optional[str] = None):
            if node.type == 'class_definition':
                # Extract class name
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if name_node:
                    class_name = source[name_node.start_byte:name_node.end_byte]

                    # Get docstring if present
                    docstring = self._extract_python_docstring(node, source)

                    symbols.append({
                        'type': 'class',
                        'name': class_name,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'file_path': file_path,
                        'language': 'python',
                        'docstring': docstring,
                        'parent_class': parent_class
                    })

                    # ADR-0075: Extract inheritance relationships
                    inheritance_rels = self._extract_inheritance_relationships(node, source, class_name)
                    relationships.extend(inheritance_rels)

                    # Traverse children with parent context
                    for child in node.children:
                        traverse(child, class_name)
            
            elif node.type == 'function_definition':
                # Extract function name
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if name_node:
                    func_name = source[name_node.start_byte:name_node.end_byte]
                    
                    # Get parameters
                    params = []
                    for child in node.children:
                        if child.type == 'parameters':
                            params = self._extract_parameters(child, source)
                            break
                    
                    # Get docstring
                    docstring = self._extract_python_docstring(node, source)

                    # Determine if it's a method
                    is_method = parent_class is not None

                    symbol_data = {
                        'type': 'function' if not is_method else 'method',
                        'name': func_name,
                        'qualified_name': f"{parent_class}.{func_name}" if parent_class else func_name,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'file_path': file_path,
                        'language': 'python',
                        'parameters': params,
                        'docstring': docstring,
                        'parent_class': parent_class
                    }

                    symbols.append(symbol_data)

                    # ADR-0075: Extract method call relationships using Context7 best practices
                    method_calls = self._extract_method_calls(node, source, func_name, parent_class)
                    relationships.extend(method_calls)

                    # ADR-0090 Phase 3: Extract USES relationships (variables/attributes)
                    uses_rels = self._extract_uses_relationships(node, source, func_name, parent_class)
                    relationships.extend(uses_rels)

                    # ADR-0090 Phase 3: Extract INSTANTIATES relationships (class instantiation)
                    instantiates_rels = self._extract_instantiates_relationships(node, source, func_name, parent_class)
                    relationships.extend(instantiates_rels)
            
            # Recursive traversal
            for child in node.children:
                traverse(child, parent_class)

        traverse(node)

        # ADR-0075: Also extract import relationships
        import_relationships = self._extract_import_relationships(node, source, file_path)
        relationships.extend(import_relationships)

        # Return both symbols and relationships for enhanced GraphRAG
        result = {
            'symbols': symbols,
            'relationships': relationships,
            'stats': {
                'symbol_count': len(symbols),
                'relationship_count': len(relationships)
            }
        }

        return symbols  # Keep existing API, but store relationships for future use
        # TODO: Update callers to use relationships
    
    def _extract_javascript_symbols(
        self, 
        node: Node, 
        source: str,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Extract symbols from JavaScript/JSX code"""
        symbols = []
        
        def traverse(node: Node):
            # Class declarations
            if node.type in ['class_declaration', 'class']:
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if name_node:
                    class_name = source[name_node.start_byte:name_node.end_byte]
                    symbols.append({
                        'type': 'class',
                        'name': class_name,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'file_path': file_path,
                        'language': 'javascript'
                    })
            
            # Function declarations
            elif node.type == 'function_declaration':
                name_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        name_node = child
                        break
                
                if name_node:
                    func_name = source[name_node.start_byte:name_node.end_byte]
                    symbols.append({
                        'type': 'function',
                        'name': func_name,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'file_path': file_path,
                        'language': 'javascript'
                    })
            
            # Arrow functions assigned to variables
            elif node.type == 'variable_declaration':
                for child in node.children:
                    if child.type == 'variable_declarator':
                        name_node = None
                        arrow_func = None
                        
                        for subchild in child.children:
                            if subchild.type == 'identifier':
                                name_node = subchild
                            elif subchild.type == 'arrow_function':
                                arrow_func = subchild
                        
                        if name_node and arrow_func:
                            func_name = source[name_node.start_byte:name_node.end_byte]
                            symbols.append({
                                'type': 'function',
                                'name': func_name,
                                'start_line': arrow_func.start_point[0] + 1,
                                'end_line': arrow_func.end_point[0] + 1,
                                'start_byte': arrow_func.start_byte,
                                'end_byte': arrow_func.end_byte,
                                'file_path': file_path,
                                'language': 'javascript',
                                'is_arrow': True
                            })
            
            # Recursive traversal
            for child in node.children:
                traverse(child)
        
        traverse(node)
        return symbols
    
    def _extract_typescript_symbols(
        self, 
        node: Node, 
        source: str,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Extract symbols from TypeScript/TSX code"""
        # TypeScript shares most patterns with JavaScript
        symbols = self._extract_javascript_symbols(node, source, file_path)
        
        # Add TypeScript-specific patterns
        def traverse_ts(node: Node):
            # Interface declarations
            if node.type == 'interface_declaration':
                name_node = None
                for child in node.children:
                    if child.type == 'type_identifier':
                        name_node = child
                        break
                
                if name_node:
                    interface_name = source[name_node.start_byte:name_node.end_byte]
                    symbols.append({
                        'type': 'interface',
                        'name': interface_name,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'file_path': file_path,
                        'language': 'typescript'
                    })
            
            # Type aliases
            elif node.type == 'type_alias_declaration':
                name_node = None
                for child in node.children:
                    if child.type == 'type_identifier':
                        name_node = child
                        break
                
                if name_node:
                    type_name = source[name_node.start_byte:name_node.end_byte]
                    symbols.append({
                        'type': 'type_alias',
                        'name': type_name,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                        'start_byte': node.start_byte,
                        'end_byte': node.end_byte,
                        'file_path': file_path,
                        'language': 'typescript'
                    })
            
            # Recursive traversal
            for child in node.children:
                traverse_ts(child)
        
        traverse_ts(node)
        
        # Update language for all symbols
        for symbol in symbols:
            symbol['language'] = 'typescript'
        
        return symbols
    
    def _extract_python_docstring(self, node: Node, source: str) -> Optional[str]:
        """Extract docstring from Python function or class"""
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr in stmt.children:
                            if expr.type == 'string':
                                # Clean up the docstring
                                docstring = source[expr.start_byte:expr.end_byte]
                                docstring = docstring.strip().strip('"""').strip("'''")
                                return docstring.strip()
                # Only check first statement
                break
        return None
    
    def _extract_parameters(self, params_node: Node, source: str) -> List[str]:
        """Extract parameter names from function parameters"""
        params = []
        for child in params_node.children:
            if child.type == 'identifier':
                params.append(source[child.start_byte:child.end_byte])
            elif child.type == 'typed_parameter':
                # Handle typed parameters in Python
                for subchild in child.children:
                    if subchild.type == 'identifier':
                        params.append(source[subchild.start_byte:subchild.end_byte])
                        break
        return params
    
    async def extract_symbols_batch(
        self,
        files: List[Tuple[str, str]],
        batch_size: int = 10,
        timeout_per_file: float = 5.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract symbols from multiple files with batch processing
        
        Args:
            files: List of (file_path, content) tuples
            batch_size: Number of files to process in parallel
            timeout_per_file: Timeout for each file
            
        Returns:
            Dictionary mapping file_path to extracted symbols
        """
        results = {}
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            tasks = []
            
            for file_path, content in batch:
                task = self.extract_symbols_from_file(
                    file_path, 
                    content, 
                    timeout_per_file
                )
                tasks.append(task)
            
            # Process batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for (file_path, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Symbol extraction failed for {file_path}: {result}")
                    results[file_path] = {
                        'symbols': [],
                        'error': str(result)
                    }
                else:
                    results[file_path] = result
            
            # Log progress
            processed = min(i + batch_size, len(files))
            logger.info(f"Processed {processed}/{len(files)} files for symbol extraction")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            'files_processed': self.stats['files_processed'],
            'symbols_extracted': self.stats['symbols_extracted'],
            'parse_errors': self.stats['parse_errors'],
            'languages': self.stats['languages'],
            'average_symbols_per_file': (
                self.stats['symbols_extracted'] / self.stats['files_processed']
                if self.stats['files_processed'] > 0 else 0
            )
        }

    def _extract_method_calls(self, func_node: Node, source: str, func_name: str, parent_class: Optional[str]) -> List[Dict[str, Any]]:
        """ADR-0075: Extract method call relationships using Context7 tree-sitter patterns"""
        relationships = []

        def find_calls(node: Node):
            if node.type == 'call':
                # Extract function/method being called
                function_node = None
                for child in node.children:
                    if child.type in ['identifier', 'attribute']:
                        function_node = child
                        break

                if function_node:
                    called_name = source[function_node.start_byte:function_node.end_byte]

                    # Create CALLS relationship
                    relationships.append({
                        'type': 'CALLS',
                        'from_function': func_name,
                        'from_class': parent_class,
                        'to_function': called_name,
                        'line': node.start_point[0] + 1,
                        'relationship_id': f"{func_name}_calls_{called_name}_{node.start_point[0] + 1}"
                    })

            # Recursive search
            for child in node.children:
                find_calls(child)

        find_calls(func_node)
        return relationships

    def _extract_import_relationships(self, root_node: Node, source: str, file_path: str) -> List[Dict[str, Any]]:
        """ADR-0075: Extract import dependencies using Context7 patterns"""
        relationships = []

        def traverse_imports(node: Node):
            if node.type == 'import_statement':
                # from X import Y or import X
                imported_modules = []

                for child in node.children:
                    if child.type == 'dotted_name':
                        module_name = source[child.start_byte:child.end_byte]
                        imported_modules.append(module_name)
                    elif child.type == 'import_from_statement':
                        # Handle 'from X import Y'
                        for subchild in child.children:
                            if subchild.type == 'dotted_name':
                                module_name = source[subchild.start_byte:subchild.end_byte]
                                imported_modules.append(module_name)

                # Create IMPORTS relationships
                for module in imported_modules:
                    relationships.append({
                        'type': 'IMPORTS',
                        'from_file': file_path,
                        'to_module': module,
                        'line': node.start_point[0] + 1,
                        'relationship_id': f"{file_path}_imports_{module}_{node.start_point[0] + 1}"
                    })

            elif node.type == 'import_from_statement':
                # from X import Y
                module_name = None
                imported_items = []

                for child in node.children:
                    if child.type == 'dotted_name':
                        module_name = source[child.start_byte:child.end_byte]
                    elif child.type == 'import_list':
                        for item in child.children:
                            if item.type == 'identifier':
                                imported_items.append(source[item.start_byte:item.end_byte])

                if module_name:
                    relationships.append({
                        'type': 'IMPORTS',
                        'from_file': file_path,
                        'to_module': module_name,
                        'imported_items': imported_items,
                        'line': node.start_point[0] + 1,
                        'relationship_id': f"{file_path}_imports_{module_name}_{node.start_point[0] + 1}"
                    })

            # Recursive traversal
            for child in node.children:
                traverse_imports(child)

        traverse_imports(root_node)
        return relationships

    def _extract_inheritance_relationships(self, class_node: Node, source: str, class_name: str) -> List[Dict[str, Any]]:
        """ADR-0075: Extract class inheritance relationships"""
        relationships = []

        # Look for inheritance in class definition
        for child in class_node.children:
            if child.type == 'argument_list':  # class Foo(Bar):
                for arg in child.children:
                    if arg.type == 'identifier':
                        parent_class = source[arg.start_byte:arg.end_byte]
                        relationships.append({
                            'type': 'INHERITS',
                            'from_class': class_name,
                            'to_class': parent_class,
                            'line': class_node.start_point[0] + 1,
                            'relationship_id': f"{class_name}_inherits_{parent_class}"
                        })

        return relationships

    def _extract_uses_relationships(self, func_node: Node, source: str, func_name: str, parent_class: Optional[str]) -> List[Dict[str, Any]]:
        """ADR-0090 Phase 3: Extract USES relationships for variables and attributes"""
        relationships = []
        used_variables = set()

        def find_uses(node: Node):
            # Look for variable/attribute usage
            if node.type == 'identifier':
                var_name = source[node.start_byte:node.end_byte]
                # Skip self, common builtins, and function names
                if var_name not in ['self', 'cls', 'super', 'True', 'False', 'None', func_name]:
                    if var_name not in used_variables:
                        used_variables.add(var_name)
                        relationships.append({
                            'type': 'USES',
                            'from_function': func_name,
                            'from_class': parent_class,
                            'to_variable': var_name,
                            'line': node.start_point[0] + 1,
                            'relationship_id': f"{func_name}_uses_{var_name}_{node.start_point[0] + 1}"
                        })

            elif node.type == 'attribute':
                # Extract object.attribute patterns
                attr_text = source[node.start_byte:node.end_byte]
                if '.' in attr_text:
                    parts = attr_text.split('.')
                    if len(parts) == 2 and parts[0] not in ['self', 'cls']:
                        obj_name, attr_name = parts
                        if obj_name not in used_variables:
                            used_variables.add(obj_name)
                            relationships.append({
                                'type': 'USES',
                                'from_function': func_name,
                                'from_class': parent_class,
                                'to_variable': obj_name,
                                'to_attribute': attr_name,
                                'line': node.start_point[0] + 1,
                                'relationship_id': f"{func_name}_uses_{obj_name}_{attr_name}_{node.start_point[0] + 1}"
                            })

            # Recursive search
            for child in node.children:
                find_uses(child)

        find_uses(func_node)
        return relationships

    def _extract_instantiates_relationships(self, func_node: Node, source: str, func_name: str, parent_class: Optional[str]) -> List[Dict[str, Any]]:
        """ADR-0090 Phase 3: Extract INSTANTIATES relationships for class instantiation"""
        relationships = []

        def find_instantiations(node: Node):
            if node.type == 'call':
                # Check if this is a class instantiation
                function_node = None
                for child in node.children:
                    if child.type == 'identifier':
                        function_node = child
                        break
                    elif child.type == 'attribute':
                        # Handle module.Class() pattern
                        attr_text = source[child.start_byte:child.end_byte]
                        if '.' in attr_text:
                            parts = attr_text.split('.')
                            if len(parts) >= 2:
                                class_name = parts[-1]
                                # Check if it starts with uppercase (likely a class)
                                if class_name and class_name[0].isupper():
                                    relationships.append({
                                        'type': 'INSTANTIATES',
                                        'from_function': func_name,
                                        'from_class': parent_class,
                                        'to_class': class_name,
                                        'module': '.'.join(parts[:-1]),
                                        'line': node.start_point[0] + 1,
                                        'relationship_id': f"{func_name}_instantiates_{class_name}_{node.start_point[0] + 1}"
                                    })
                        break

                if function_node:
                    class_name = source[function_node.start_byte:function_node.end_byte]
                    # Check if it starts with uppercase (likely a class)
                    if class_name and class_name[0].isupper():
                        relationships.append({
                            'type': 'INSTANTIATES',
                            'from_function': func_name,
                            'from_class': parent_class,
                            'to_class': class_name,
                            'line': node.start_point[0] + 1,
                            'relationship_id': f"{func_name}_instantiates_{class_name}_{node.start_point[0] + 1}"
                        })

            # Recursive search
            for child in node.children:
                find_instantiations(child)

        find_instantiations(func_node)
        return relationships