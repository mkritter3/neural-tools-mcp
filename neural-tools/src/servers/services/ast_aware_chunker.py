#!/usr/bin/env python3
"""
AST-Aware Code Chunker - ADR-0047 Phase 3
Uses tree-sitter to parse code and create semantically meaningful chunks.
Ensures chunks contain complete code units (functions, classes, etc.).
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# Tree-sitter language mappings
LANGUAGE_PARSERS = {
    '.py': 'python',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.cpp': 'cpp',
    '.c': 'c',
    '.cs': 'c_sharp',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.r': 'r',
    '.lua': 'lua',
    '.dart': 'dart'
}

@dataclass
class CodeChunk:
    """Represents a semantically meaningful code chunk"""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'import_block', 'global', 'comment_block'
    name: Optional[str] = None  # Function/class/method name
    parent: Optional[str] = None  # Parent class for methods
    language: Optional[str] = None
    ast_path: Optional[str] = None  # Path in AST (e.g., "Module.Class.Method")
    dependencies: List[str] = None  # Imports/dependencies in this chunk
    complexity: int = 0  # Cyclomatic complexity estimate
    chunk_id: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if not self.chunk_id:
            # Generate deterministic chunk ID
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()
            self.chunk_id = f"{self.chunk_type}_{content_hash[:16]}"

class ASTAwareChunker:
    """
    Chunks code files using Abstract Syntax Tree parsing.
    Ensures chunks are semantically complete and meaningful.
    """

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 100):
        """
        Initialize AST-aware chunker.

        Args:
            max_chunk_size: Maximum lines per chunk (soft limit)
            min_chunk_size: Minimum lines per chunk (avoid tiny chunks)
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.parser = None
        self._init_parser()

    def _init_parser(self):
        """Initialize tree-sitter parser"""
        try:
            # Try to import tree-sitter
            import tree_sitter_languages
            self.parser = tree_sitter_languages
            logger.info("Tree-sitter languages loaded successfully")
        except ImportError:
            logger.warning("tree-sitter-languages not available, falling back to regex-based parsing")
            self.parser = None

    def chunk_file(self, file_path: str, content: str = None) -> List[CodeChunk]:
        """
        Chunk a code file into semantically meaningful pieces.

        Args:
            file_path: Path to the file
            content: Optional file content (will read file if not provided)

        Returns:
            List of CodeChunk objects
        """
        try:
            # Read file if content not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

            # Detect language from file extension
            ext = Path(file_path).suffix.lower()
            language = LANGUAGE_PARSERS.get(ext)

            if not language:
                # Unknown language - fall back to line-based chunking
                return self._fallback_chunk(content, file_path)

            # Use AST parsing if available
            if self.parser:
                return self._ast_chunk_with_tree_sitter(content, language, file_path)
            else:
                # Fall back to regex-based semantic chunking
                return self._regex_semantic_chunk(content, language, file_path)

        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            return self._fallback_chunk(content or "", file_path)

    def _ast_chunk_with_tree_sitter(self, content: str, language: str, file_path: str) -> List[CodeChunk]:
        """
        Chunk using tree-sitter AST parsing.

        Args:
            content: File content
            language: Programming language
            file_path: File path for context

        Returns:
            List of semantic code chunks
        """
        chunks = []

        try:
            # Get parser for language
            parser = self.parser.get_parser(language)
            tree = parser.parse(content.encode())
            root = tree.root_node

            # Extract semantic units
            chunks.extend(self._extract_imports(root, content, language))
            chunks.extend(self._extract_classes(root, content, language))
            chunks.extend(self._extract_functions(root, content, language))
            chunks.extend(self._extract_global_code(root, content, language, chunks))

            # Sort chunks by start line
            chunks.sort(key=lambda c: c.start_line)

            # Merge small adjacent chunks if needed
            chunks = self._merge_small_chunks(chunks)

            # Split large chunks if needed
            chunks = self._split_large_chunks(chunks)

        except Exception as e:
            logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}, using fallback")
            return self._regex_semantic_chunk(content, language, file_path)

        return chunks

    def _extract_imports(self, root, content: str, language: str) -> List[CodeChunk]:
        """Extract import statements as a single chunk"""
        chunks = []
        lines = content.split('\n')

        if language == 'python':
            import_nodes = self._find_nodes_by_type(root, ['import_statement', 'import_from_statement'])
        elif language in ['javascript', 'typescript']:
            import_nodes = self._find_nodes_by_type(root, ['import_statement', 'import_clause'])
        elif language == 'java':
            import_nodes = self._find_nodes_by_type(root, ['import_declaration'])
        elif language == 'go':
            import_nodes = self._find_nodes_by_type(root, ['import_declaration', 'import_spec'])
        else:
            import_nodes = []

        if import_nodes:
            # Group all imports into one chunk
            start_line = min(node.start_point[0] for node in import_nodes)
            end_line = max(node.end_point[0] for node in import_nodes)

            import_lines = lines[start_line:end_line + 1]
            import_content = '\n'.join(import_lines)

            # Extract imported modules
            dependencies = self._extract_import_names(import_content, language)

            chunks.append(CodeChunk(
                content=import_content,
                start_line=start_line + 1,  # Convert to 1-indexed
                end_line=end_line + 1,
                chunk_type='import_block',
                name='imports',
                language=language,
                dependencies=dependencies
            ))

        return chunks

    def _extract_classes(self, root, content: str, language: str) -> List[CodeChunk]:
        """Extract class definitions as chunks"""
        chunks = []
        lines = content.split('\n')

        if language == 'python':
            class_nodes = self._find_nodes_by_type(root, ['class_definition'])
        elif language in ['javascript', 'typescript']:
            class_nodes = self._find_nodes_by_type(root, ['class_declaration'])
        elif language in ['java', 'c_sharp']:
            class_nodes = self._find_nodes_by_type(root, ['class_declaration'])
        else:
            class_nodes = []

        for class_node in class_nodes:
            start_line = class_node.start_point[0]
            end_line = class_node.end_point[0]

            # Extract class name
            class_name = self._extract_node_name(class_node, language)

            # Get class content
            class_lines = lines[start_line:end_line + 1]
            class_content = '\n'.join(class_lines)

            # Check if class is too large
            if end_line - start_line > self.max_chunk_size:
                # Split class into method chunks
                method_chunks = self._extract_class_methods(class_node, content, language, class_name)
                chunks.extend(method_chunks)
            else:
                # Keep entire class as one chunk
                chunks.append(CodeChunk(
                    content=class_content,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    chunk_type='class',
                    name=class_name,
                    language=language,
                    ast_path=f"Module.{class_name}",
                    complexity=self._estimate_complexity(class_content)
                ))

        return chunks

    def _extract_class_methods(self, class_node, content: str, language: str, class_name: str) -> List[CodeChunk]:
        """Extract individual methods from a large class"""
        chunks = []
        lines = content.split('\n')

        if language == 'python':
            method_nodes = self._find_nodes_by_type(class_node, ['function_definition'])
        elif language in ['javascript', 'typescript']:
            method_nodes = self._find_nodes_by_type(class_node, ['method_definition'])
        elif language in ['java', 'c_sharp']:
            method_nodes = self._find_nodes_by_type(class_node, ['method_declaration'])
        else:
            method_nodes = []

        # Add class header as a chunk
        class_start = class_node.start_point[0]
        first_method_start = method_nodes[0].start_point[0] if method_nodes else class_node.end_point[0]

        header_lines = lines[class_start:first_method_start]
        header_content = '\n'.join(header_lines)

        if header_content.strip():
            chunks.append(CodeChunk(
                content=header_content,
                start_line=class_start + 1,
                end_line=first_method_start,
                chunk_type='class_header',
                name=f"{class_name}_header",
                parent=class_name,
                language=language,
                ast_path=f"Module.{class_name}"
            ))

        # Add each method as a chunk
        for method_node in method_nodes:
            method_start = method_node.start_point[0]
            method_end = method_node.end_point[0]

            method_name = self._extract_node_name(method_node, language)
            method_lines = lines[method_start:method_end + 1]
            method_content = '\n'.join(method_lines)

            chunks.append(CodeChunk(
                content=method_content,
                start_line=method_start + 1,
                end_line=method_end + 1,
                chunk_type='method',
                name=method_name,
                parent=class_name,
                language=language,
                ast_path=f"Module.{class_name}.{method_name}",
                complexity=self._estimate_complexity(method_content)
            ))

        return chunks

    def _extract_functions(self, root, content: str, language: str) -> List[CodeChunk]:
        """Extract standalone function definitions"""
        chunks = []
        lines = content.split('\n')

        if language == 'python':
            func_nodes = self._find_nodes_by_type(root, ['function_definition'])
        elif language in ['javascript', 'typescript']:
            func_nodes = self._find_nodes_by_type(root, ['function_declaration', 'arrow_function'])
        elif language in ['c', 'cpp']:
            func_nodes = self._find_nodes_by_type(root, ['function_definition'])
        elif language == 'go':
            func_nodes = self._find_nodes_by_type(root, ['function_declaration'])
        else:
            func_nodes = []

        # Filter out methods (functions inside classes)
        standalone_funcs = []
        for func in func_nodes:
            if not self._is_inside_class(func, root, language):
                standalone_funcs.append(func)

        for func_node in standalone_funcs:
            start_line = func_node.start_point[0]
            end_line = func_node.end_point[0]

            func_name = self._extract_node_name(func_node, language)
            func_lines = lines[start_line:end_line + 1]
            func_content = '\n'.join(func_lines)

            chunks.append(CodeChunk(
                content=func_content,
                start_line=start_line + 1,
                end_line=end_line + 1,
                chunk_type='function',
                name=func_name,
                language=language,
                ast_path=f"Module.{func_name}",
                complexity=self._estimate_complexity(func_content)
            ))

        return chunks

    def _extract_global_code(self, root, content: str, language: str, existing_chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Extract global code not in functions/classes"""
        chunks = []
        lines = content.split('\n')

        # Find lines not covered by existing chunks
        covered_lines = set()
        for chunk in existing_chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered_lines.add(line_num)

        # Group uncovered lines into chunks
        current_chunk_lines = []
        current_start = None

        for i, line in enumerate(lines, 1):
            if i not in covered_lines and line.strip():
                if current_start is None:
                    current_start = i
                current_chunk_lines.append(line)
            else:
                if current_chunk_lines and len(current_chunk_lines) >= 3:  # Min 3 lines for global chunk
                    chunks.append(CodeChunk(
                        content='\n'.join(current_chunk_lines),
                        start_line=current_start,
                        end_line=i - 1,
                        chunk_type='global',
                        name=f"global_{current_start}",
                        language=language
                    ))
                current_chunk_lines = []
                current_start = None

        # Add final chunk if exists
        if current_chunk_lines and len(current_chunk_lines) >= 3:
            chunks.append(CodeChunk(
                content='\n'.join(current_chunk_lines),
                start_line=current_start,
                end_line=len(lines),
                chunk_type='global',
                name=f"global_{current_start}",
                language=language
            ))

        return chunks

    def _regex_semantic_chunk(self, content: str, language: str, file_path: str) -> List[CodeChunk]:
        """
        Fall back to regex-based semantic chunking when tree-sitter unavailable.

        Args:
            content: File content
            language: Programming language
            file_path: File path

        Returns:
            List of semantic chunks
        """
        chunks = []
        lines = content.split('\n')

        # Language-specific regex patterns
        patterns = self._get_language_patterns(language)

        # Extract functions/methods
        for pattern in patterns['function_patterns']:
            chunks.extend(self._extract_by_regex(lines, pattern, 'function', language))

        # Extract classes
        for pattern in patterns['class_patterns']:
            chunks.extend(self._extract_by_regex(lines, pattern, 'class', language))

        # Sort by start line
        chunks.sort(key=lambda c: c.start_line)

        # Fill gaps with global code chunks
        chunks.extend(self._extract_global_code_regex(lines, chunks, language))

        return chunks

    def _get_language_patterns(self, language: str) -> Dict[str, List]:
        """Get regex patterns for language"""
        import re

        patterns = {
            'python': {
                'function_patterns': [
                    re.compile(r'^(async\s+)?def\s+(\w+)\s*\('),
                ],
                'class_patterns': [
                    re.compile(r'^class\s+(\w+)[\s\(:]'),
                ]
            },
            'javascript': {
                'function_patterns': [
                    re.compile(r'^(async\s+)?function\s+(\w+)\s*\('),
                    re.compile(r'^(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\('),
                    re.compile(r'^(export\s+)?(async\s+)?function\s+(\w+)'),
                ],
                'class_patterns': [
                    re.compile(r'^class\s+(\w+)'),
                    re.compile(r'^(export\s+)?class\s+(\w+)'),
                ]
            },
            'java': {
                'function_patterns': [
                    re.compile(r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\('),
                ],
                'class_patterns': [
                    re.compile(r'^\s*(public|private|protected)?\s*class\s+(\w+)'),
                ]
            }
        }

        # Default patterns if language not found
        default = {
            'function_patterns': [
                re.compile(r'(function|def|func)\s+(\w+)'),
            ],
            'class_patterns': [
                re.compile(r'class\s+(\w+)'),
            ]
        }

        return patterns.get(language, default)

    def _extract_by_regex(self, lines: List[str], pattern, chunk_type: str, language: str) -> List[CodeChunk]:
        """Extract code chunks using regex pattern"""
        chunks = []

        for i, line in enumerate(lines):
            match = pattern.search(line)
            if match:
                # Find the end of this block (simple heuristic based on indentation)
                start_line = i
                end_line = self._find_block_end(lines, i, language)

                # Extract name from match groups
                name = None
                for group in match.groups():
                    if group and group not in ['async', 'const', 'let', 'var', 'public', 'private', 'protected', 'static', 'export']:
                        name = group
                        break

                block_content = '\n'.join(lines[start_line:end_line + 1])

                chunks.append(CodeChunk(
                    content=block_content,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    chunk_type=chunk_type,
                    name=name or f"{chunk_type}_{start_line}",
                    language=language,
                    complexity=self._estimate_complexity(block_content)
                ))

        return chunks

    def _find_block_end(self, lines: List[str], start_idx: int, language: str) -> int:
        """Find the end of a code block based on indentation"""
        if start_idx >= len(lines) - 1:
            return start_idx

        # Get indentation of start line
        start_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())

        # For languages with braces, count braces
        if language in ['javascript', 'typescript', 'java', 'c', 'cpp', 'c_sharp', 'go', 'rust']:
            brace_count = 0
            for i in range(start_idx, len(lines)):
                brace_count += lines[i].count('{') - lines[i].count('}')
                if brace_count == 0 and i > start_idx and '{' in lines[start_idx:i+1]:
                    return i

        # For Python and other indentation-based languages
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if not line.strip():  # Skip empty lines
                continue

            line_indent = len(line) - len(line.lstrip())

            # If we're back to the same or less indentation, block ended
            if line_indent <= start_indent:
                return i - 1

        return len(lines) - 1

    def _extract_global_code_regex(self, lines: List[str], existing_chunks: List[CodeChunk], language: str) -> List[CodeChunk]:
        """Extract global code not covered by other chunks"""
        chunks = []

        # Find uncovered lines
        covered = set()
        for chunk in existing_chunks:
            for line_num in range(chunk.start_line, chunk.end_line + 1):
                covered.add(line_num)

        # Group uncovered lines
        current_lines = []
        current_start = None

        for i, line in enumerate(lines, 1):
            if i not in covered and line.strip():
                if current_start is None:
                    current_start = i
                current_lines.append(line)
            else:
                if len(current_lines) >= self.min_chunk_size // 10:  # At least 10 lines for global
                    chunks.append(CodeChunk(
                        content='\n'.join(current_lines),
                        start_line=current_start,
                        end_line=i - 1,
                        chunk_type='global',
                        name=f"global_{current_start}",
                        language=language
                    ))
                current_lines = []
                current_start = None

        # Add remaining lines
        if len(current_lines) >= self.min_chunk_size // 10:
            chunks.append(CodeChunk(
                content='\n'.join(current_lines),
                start_line=current_start,
                end_line=len(lines),
                chunk_type='global',
                name=f"global_{current_start}",
                language=language
            ))

        return chunks

    def _fallback_chunk(self, content: str, file_path: str) -> List[CodeChunk]:
        """Simple line-based chunking as ultimate fallback"""
        chunks = []
        lines = content.split('\n')

        for i in range(0, len(lines), self.max_chunk_size):
            chunk_lines = lines[i:i + self.max_chunk_size]
            if len(chunk_lines) >= self.min_chunk_size // 10 or i == 0:
                chunks.append(CodeChunk(
                    content='\n'.join(chunk_lines),
                    start_line=i + 1,
                    end_line=min(i + len(chunk_lines), len(lines)),
                    chunk_type='fallback',
                    name=f"chunk_{i // self.max_chunk_size}",
                    language='unknown'
                ))

        return chunks if chunks else [CodeChunk(
            content=content,
            start_line=1,
            end_line=len(lines),
            chunk_type='fallback',
            name='full_file',
            language='unknown'
        )]

    def _merge_small_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Merge adjacent small chunks to avoid fragmentation"""
        if not chunks:
            return chunks

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # Check if should merge
            current_size = current.end_line - current.start_line + 1
            next_size = next_chunk.end_line - next_chunk.start_line + 1

            if (current_size < self.min_chunk_size and
                next_size < self.min_chunk_size and
                current_size + next_size <= self.max_chunk_size and
                current.chunk_type == next_chunk.chunk_type):

                # Merge chunks
                current = CodeChunk(
                    content=current.content + '\n' + next_chunk.content,
                    start_line=current.start_line,
                    end_line=next_chunk.end_line,
                    chunk_type=current.chunk_type,
                    name=f"{current.name}_{next_chunk.name}",
                    language=current.language,
                    dependencies=current.dependencies + next_chunk.dependencies,
                    complexity=current.complexity + next_chunk.complexity
                )
            else:
                merged.append(current)
                current = next_chunk

        merged.append(current)
        return merged

    def _split_large_chunks(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Split chunks that are too large"""
        result = []

        for chunk in chunks:
            size = chunk.end_line - chunk.start_line + 1

            if size > self.max_chunk_size * 2:  # Way too large
                # Split into smaller chunks
                lines = chunk.content.split('\n')
                for i in range(0, len(lines), self.max_chunk_size):
                    sub_lines = lines[i:i + self.max_chunk_size]
                    if sub_lines:
                        result.append(CodeChunk(
                            content='\n'.join(sub_lines),
                            start_line=chunk.start_line + i,
                            end_line=min(chunk.start_line + i + len(sub_lines) - 1, chunk.end_line),
                            chunk_type=chunk.chunk_type,
                            name=f"{chunk.name}_part{i // self.max_chunk_size}",
                            parent=chunk.parent,
                            language=chunk.language,
                            ast_path=chunk.ast_path,
                            dependencies=chunk.dependencies if i == 0 else [],
                            complexity=self._estimate_complexity('\n'.join(sub_lines))
                        ))
            else:
                result.append(chunk)

        return result

    # Helper methods
    def _find_nodes_by_type(self, node, types: List[str]) -> List:
        """Recursively find all nodes of given types"""
        results = []

        def traverse(n):
            if n.type in types:
                results.append(n)
            for child in n.children:
                traverse(child)

        traverse(node)
        return results

    def _extract_node_name(self, node, language: str) -> str:
        """Extract name from AST node"""
        # This is simplified - actual implementation would vary by language
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode() if isinstance(child.text, bytes) else child.text

        return f"{node.type}_{node.start_point[0]}"

    def _is_inside_class(self, node, root, language: str) -> bool:
        """Check if node is inside a class definition"""
        parent = node.parent
        while parent and parent != root:
            if parent.type in ['class_definition', 'class_declaration', 'class_body']:
                return True
            parent = parent.parent
        return False

    def _extract_import_names(self, import_content: str, language: str) -> List[str]:
        """Extract imported module names from import statements"""
        imports = []
        lines = import_content.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if language == 'python':
                if line.startswith('import '):
                    # import module1, module2
                    modules = line[7:].split(',')
                    imports.extend([m.strip().split(' as ')[0] for m in modules])
                elif line.startswith('from '):
                    # from module import ...
                    parts = line.split(' import ')
                    if parts:
                        module = parts[0][5:].strip()
                        imports.append(module)

            elif language in ['javascript', 'typescript']:
                if 'import ' in line:
                    # Various JS import formats
                    if ' from ' in line:
                        module = line.split(' from ')[-1].strip(' ;\'"')
                        imports.append(module)

            elif language == 'java':
                if line.startswith('import '):
                    # import package.Class;
                    package = line[7:].strip(' ;')
                    imports.append(package)

        return imports

    def _estimate_complexity(self, content: str) -> int:
        """Estimate cyclomatic complexity of code"""
        # Simple heuristic: count control flow keywords
        complexity = 1  # Base complexity

        control_flow_keywords = [
            'if', 'else', 'elif', 'for', 'while', 'case', 'catch', 'except',
            'switch', 'try', '&&', '||', '?', 'and', 'or'
        ]

        content_lower = content.lower()
        for keyword in control_flow_keywords:
            # Count occurrences (simple approach)
            complexity += content_lower.count(f' {keyword} ')
            complexity += content_lower.count(f'\n{keyword} ')
            complexity += content_lower.count(f'({keyword} ')

        return complexity

# Test the chunker
if __name__ == "__main__":
    chunker = ASTAwareChunker()

    # Test Python code
    test_code = '''
import os
import sys
from typing import List

class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self, items: List[str]) -> List[str]:
        """Process a list of items"""
        result = []
        for item in items:
            if item:
                result.append(item.upper())
        return result

def main():
    processor = DataProcessor()
    data = ["hello", "world"]
    print(processor.process(data))

if __name__ == "__main__":
    main()
'''

    chunks = chunker.chunk_file("test.py", test_code)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}: {chunk.chunk_type} ({chunk.name}) ---")
        print(f"Lines {chunk.start_line}-{chunk.end_line}")
        print(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)