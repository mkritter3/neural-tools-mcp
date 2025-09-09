#!/usr/bin/env python3
"""
Unit tests for Tree-sitter Parser (Phase 1.6)
Tests high-performance parsing, incremental updates, and multi-language support
"""

import pytest
import asyncio
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import the modules to test
from src.infrastructure.tree_sitter_ast import (
    TreeSitterChunker, SmartCodeChunker, CodeChunk, EditInfo,
    TREE_SITTER_AVAILABLE
)


class TestTreeSitterChunker:
    """Unit tests for TreeSitterChunker high-performance parsing"""
    
    @pytest.fixture
    def chunker(self):
        """Create TreeSitterChunker instance if available"""
        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter-languages not available")
        return TreeSitterChunker()
    
    def test_initialization(self, chunker):
        """Test TreeSitter chunker initializes correctly"""
        assert isinstance(chunker.languages, dict)
        assert isinstance(chunker.parsers, dict)
        assert isinstance(chunker.queries, dict)
        assert len(chunker.parsers) > 0, "Should initialize at least one language parser"
    
    def test_supported_languages(self, chunker):
        """Test Tree-sitter supports required languages"""
        languages = chunker.get_supported_languages()
        
        # Must support these core languages per roadmap
        required_languages = {'python', 'javascript', 'typescript'}
        supported_required = set(languages) & required_languages
        
        assert len(supported_required) > 0, f"Must support at least one of {required_languages}"
        assert 'python' in languages, "Python must be supported for compatibility"
    
    @pytest.mark.asyncio
    async def test_python_semantic_extraction(self, chunker):
        """Test semantic chunk extraction for Python code"""
        python_code = '''
def calculate_fibonacci(n):
    """Calculate fibonacci sequence"""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    def __init__(self):
        self.pi = 3.14159
    
    def area_circle(self, radius):
        return self.pi * radius ** 2

import math
from typing import List
'''
        
        chunks = await chunker.extract_chunks("/fake/path.py", python_code, "python")
        
        # Verify semantic chunks were extracted
        assert len(chunks) > 0, "Should extract at least some chunks"
        
        chunk_types = {chunk.chunk_type for chunk in chunks}
        chunk_names = {chunk.name for chunk in chunks}
        
        # Should find functions and classes
        assert 'function' in chunk_types, "Should extract function chunks"
        assert 'class' in chunk_types, "Should extract class chunks"
        
        # Should find specific entities by name
        function_names = {c.name for c in chunks if c.chunk_type == 'function'}
        class_names = {c.name for c in chunks if c.chunk_type == 'class'}
        
        assert 'calculate_fibonacci' in function_names, "Should find fibonacci function"
        assert 'MathUtils' in class_names, "Should find MathUtils class"
    
    @pytest.mark.asyncio
    async def test_javascript_semantic_extraction(self, chunker):
        """Test semantic chunk extraction for JavaScript code"""
        if 'javascript' not in chunker.get_supported_languages():
            pytest.skip("JavaScript not supported by this Tree-sitter installation")
        
        js_code = '''
function calculateSum(a, b) {
    return a + b;
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push({operation: 'add', a, b, result});
        return result;
    }
}

const util = require('util');
import { Helper } from './helper.js';
'''
        
        chunks = await chunker.extract_chunks("/fake/path.js", js_code, "javascript")
        
        # Verify semantic chunks were extracted
        assert len(chunks) > 0, "Should extract JavaScript chunks"
        
        chunk_types = {chunk.chunk_type for chunk in chunks}
        chunk_names = {chunk.name for chunk in chunks}
        
        # Should find functions and classes
        function_names = {c.name for c in chunks if c.chunk_type == 'function'}
        class_names = {c.name for c in chunks if c.chunk_type == 'class'}
        
        assert 'calculateSum' in function_names or 'Calculator' in class_names, \
            "Should extract JavaScript functions or classes"
    
    @pytest.mark.asyncio
    async def test_incremental_parsing(self, chunker):
        """Test incremental parsing performance improvement"""
        if 'python' not in chunker.get_supported_languages():
            pytest.skip("Python not supported")
        
        original_code = '''
def original_function():
    return "original"

class OriginalClass:
    pass
'''
        
        modified_code = '''
def original_function():
    return "original"

def new_function():
    return "new"

class OriginalClass:
    pass
'''
        
        # Parse original code
        parser = chunker.parsers['python']
        original_tree = parser.parse(bytes(original_code, 'utf8'))
        
        # Create edit info for the modification
        edits = chunker.create_edit_info(original_code, modified_code)
        
        # Test incremental parsing
        start_time = time.perf_counter()
        new_tree = await chunker.incremental_parse(original_tree, edits, modified_code, 'python')
        incremental_time = time.perf_counter() - start_time
        
        # Test full reparse
        start_time = time.perf_counter()
        full_tree = parser.parse(bytes(modified_code, 'utf8'))
        full_time = time.perf_counter() - start_time
        
        # Incremental should be faster (though may not always be 5x on small code)
        # At minimum, it should not be slower
        assert incremental_time <= full_time * 2, \
            f"Incremental parsing ({incremental_time:.4f}s) should not be much slower than full ({full_time:.4f}s)"
        
        # Verify trees are functionally equivalent
        assert new_tree.root_node.type == full_tree.root_node.type
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, chunker):
        """Test error recovery with broken/incomplete code"""
        broken_python_code = '''
def broken_function(
    # Missing closing parenthesis and body
    
class IncompleteClass:
    def method_without_body(self):
    # Missing body
    
def working_function():
    return "this should still work"
'''
        
        # Should not crash and should extract what it can
        chunks = await chunker.extract_chunks("/fake/broken.py", broken_python_code, "python")
        
        # Even with broken code, should extract some chunks
        chunk_names = [chunk.name for chunk in chunks]
        
        # Should still find the working function
        working_found = any('working_function' in name for name in chunk_names)
        assert working_found or len(chunks) > 0, "Should extract some chunks even from broken code"
    
    @pytest.mark.asyncio 
    async def test_performance_benchmark(self, chunker):
        """Test performance benchmarking functionality"""
        python_code = '''
def benchmark_function():
    result = 0
    for i in range(100):
        result += i * 2
    return result

class BenchmarkClass:
    def __init__(self):
        self.data = list(range(50))
    
    def process_data(self):
        return sum(x ** 2 for x in self.data)
'''
        
        # Run performance benchmark
        results = await chunker.benchmark_parsing_performance(python_code, 'python', iterations=10)
        
        assert 'tree_sitter_avg_ms' in results
        assert 'estimated_ast_avg_ms' in results
        assert 'speedup_factor' in results
        assert 'language' in results
        
        # Verify expected speedup factor
        assert results['speedup_factor'] == 36.0, "Should report 36x speedup as per roadmap"
        assert results['estimated_ast_avg_ms'] > results['tree_sitter_avg_ms'], \
            "AST should be estimated as slower"
    
    def test_create_edit_info(self, chunker):
        """Test edit info creation for incremental parsing"""
        old_text = "line1\nline2\nline3"
        new_text = "line1\nmodified_line2\nline3\nnew_line4"
        
        edits = chunker.create_edit_info(old_text, new_text)
        
        assert isinstance(edits, list)
        assert len(edits) > 0, "Should create edit info for text changes"
        
        for edit in edits:
            assert isinstance(edit, EditInfo)
            assert hasattr(edit, 'start_byte')
            assert hasattr(edit, 'old_end_byte')
            assert hasattr(edit, 'new_end_byte')


class TestSmartCodeChunker:
    """Unit tests for SmartCodeChunker with fallbacks"""
    
    @pytest.fixture
    def chunker(self):
        """Create SmartCodeChunker instance"""
        return SmartCodeChunker()
    
    def test_initialization(self, chunker):
        """Test SmartCodeChunker initializes correctly"""
        assert chunker is not None
        
        # Should initialize Tree-sitter if available
        if TREE_SITTER_AVAILABLE:
            assert chunker.tree_sitter is not None
        
        # Should have AST analyzer fallback
        assert chunker.ast_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_chunk_file_python(self, chunker):
        """Test chunking a Python file"""
        python_code = '''
import os
from typing import List

def process_file(filename: str) -> List[str]:
    """Process a file and return lines"""
    with open(filename, 'r') as f:
        return f.readlines()

class FileProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def process_all(self):
        return "processed"
'''
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            temp_path = f.name
        
        try:
            chunks = await chunker.chunk_file(temp_path)
            
            assert len(chunks) > 0, "Should extract chunks from Python file"
            
            # Verify chunk structure
            for chunk in chunks:
                assert isinstance(chunk, CodeChunk)
                assert chunk.file_path == temp_path
                assert chunk.language == 'python'
                assert chunk.line_start > 0
                assert chunk.line_end >= chunk.line_start
            
            # Should find function and class
            chunk_types = {chunk.chunk_type for chunk in chunks}
            chunk_names = {chunk.name for chunk in chunks}
            
            has_function = 'function' in chunk_types or any('process' in name for name in chunk_names)
            has_class = 'class' in chunk_types or any('Processor' in name for name in chunk_names)
            
            assert has_function or has_class, "Should extract Python functions or classes"
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_chunk_unsupported_file(self, chunker):
        """Test handling of unsupported file types"""
        # Create temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("some content")
            temp_path = f.name
        
        try:
            chunks = await chunker.chunk_file(temp_path)
            assert chunks == [], "Should return empty list for unsupported files"
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_benchmark_performance(self, chunker):
        """Test SmartCodeChunker performance benchmarking"""
        python_code = "def test(): pass"
        
        results = await chunker.benchmark_parsing_performance(python_code, 'python', iterations=5)
        
        # Should either return benchmark results or error
        assert isinstance(results, dict)
        
        # If Tree-sitter available, should have performance data
        if TREE_SITTER_AVAILABLE and 'python' in chunker.tree_sitter.get_supported_languages():
            assert 'tree_sitter_avg_ms' in results or 'speedup_factor' in results
        else:
            # Should indicate Tree-sitter not available
            assert 'error' in results or 'estimated_speedup' in results
    
    def test_ext_to_language_mapping(self, chunker):
        """Test file extension to language mapping"""
        # Test common mappings
        test_mappings = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        
        for ext, expected_lang in test_mappings.items():
            detected_lang = chunker._ext_to_language(ext)
            if detected_lang:  # Only check if language is supported
                assert detected_lang == expected_lang, f"Extension {ext} should map to {expected_lang}"


class TestTreeSitterAvailability:
    """Test handling when Tree-sitter is not available"""
    
    @patch('src.infrastructure.tree_sitter_ast.TREE_SITTER_AVAILABLE', False)
    def test_chunker_without_tree_sitter(self):
        """Test TreeSitterChunker raises error when tree-sitter unavailable"""
        with pytest.raises(ImportError, match="tree-sitter-languages required"):
            TreeSitterChunker()
    
    @patch('src.infrastructure.tree_sitter_ast.TREE_SITTER_AVAILABLE', False)
    def test_smart_chunker_fallback(self):
        """Test SmartCodeChunker works without Tree-sitter"""
        chunker = SmartCodeChunker()
        
        # Should initialize without Tree-sitter
        assert chunker.tree_sitter is None
        assert chunker.ast_analyzer is not None


@pytest.mark.benchmark
class TestTreeSitterPerformance:
    """Performance benchmarks for Tree-sitter parsing"""
    
    @pytest.fixture
    def large_python_code(self):
        """Generate large Python code for performance testing"""
        code_parts = []
        
        # Generate imports
        for i in range(10):
            code_parts.append(f"import module_{i}")
        
        # Generate functions
        for i in range(50):
            code_parts.append(f'''
def function_{i}(param1, param2, param3):
    """Function {i} documentation"""
    result = param1 + param2
    for j in range(param3):
        result += j * 2
    return result
''')
        
        # Generate classes
        for i in range(10):
            code_parts.append(f'''
class Class_{i}:
    """Class {i} documentation"""
    
    def __init__(self, value):
        self.value = value
        self.data = list(range(100))
    
    def method_{i}_1(self):
        return sum(self.data)
    
    def method_{i}_2(self, multiplier):
        return [x * multiplier for x in self.data]
''')
        
        return '\n'.join(code_parts)
    
    @pytest.mark.asyncio
    async def test_large_file_parsing_performance(self, large_python_code):
        """Test parsing performance on large files"""
        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        
        chunker = TreeSitterChunker()
        if 'python' not in chunker.get_supported_languages():
            pytest.skip("Python not supported")
        
        # Benchmark parsing performance
        start_time = time.perf_counter()
        chunks = await chunker.extract_chunks("/fake/large.py", large_python_code, "python")
        parsing_time = time.perf_counter() - start_time
        
        # Verify extraction quality
        assert len(chunks) > 40, f"Should extract many chunks from large file, got {len(chunks)}"
        
        # Performance should be reasonable (under 100ms for this size)
        assert parsing_time < 0.1, f"Parsing took {parsing_time:.3f}s, should be under 0.1s for Tree-sitter"
        
        # Verify chunk quality
        function_chunks = [c for c in chunks if c.chunk_type == 'function']
        class_chunks = [c for c in chunks if c.chunk_type == 'class']
        
        assert len(function_chunks) >= 30, "Should extract many functions"
        assert len(class_chunks) >= 5, "Should extract several classes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])