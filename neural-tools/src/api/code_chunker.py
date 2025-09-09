#!/usr/bin/env python3
"""
AST-Based Code Chunking for Neural Search
Intelligently splits code files into semantic chunks (functions, classes)
Based on Gemini's recommendations using tree-sitter for language-agnostic parsing
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import ast
import re

logger = logging.getLogger(__name__)

class CodeChunk:
    """Represents a semantic chunk of code"""
    
    def __init__(
        self, 
        chunk_id: str,
        text: str,
        file_path: str,
        start_line: int,
        end_line: int,
        chunk_name: str,
        chunk_type: str,
        language: str = "python"
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_name = chunk_name
        self.chunk_type = chunk_type
        self.language = language
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Qdrant payload"""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "payload": {
                "file_path": self.file_path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "chunk_name": self.chunk_name,
                "chunk_type": self.chunk_type,
                "language": self.language,
                "text_length": len(self.text)
            }
        }

class PythonASTChunker:
    """
    Python-specific chunker using built-in AST module
    Fallback when tree-sitter is not available
    """
    
    def __init__(self):
        self.chunk_types = {
            ast.FunctionDef: "function",
            ast.AsyncFunctionDef: "async_function", 
            ast.ClassDef: "class",
            ast.Module: "module"
        }
    
    def generate_chunk_id(self, file_path: str, chunk_name: str, chunk_type: str) -> str:
        """Generate stable UUID for chunk"""
        content = f"{file_path}:{chunk_type}:{chunk_name}"
        # Create deterministic UUID from content hash
        hash_digest = hashlib.sha256(content.encode()).digest()
        return str(uuid.UUID(bytes=hash_digest[:16]))
    
    def extract_chunks(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """Extract semantic chunks from Python source code"""
        chunks = []
        
        try:
            # Parse the source code
            tree = ast.parse(source_code)
            lines = source_code.splitlines()
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if type(node) in [ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]:
                    chunk_type = self.chunk_types[type(node)]
                    chunk_name = node.name
                    
                    # Get line numbers
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    # Extract the actual code text
                    if end_line <= len(lines):
                        chunk_lines = lines[start_line - 1:end_line]
                        chunk_text = '\n'.join(chunk_lines)
                        
                        # Generate stable ID
                        chunk_id = self.generate_chunk_id(file_path, chunk_name, chunk_type)
                        
                        # Create chunk
                        chunk = CodeChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            chunk_name=chunk_name,
                            chunk_type=chunk_type,
                            language="python"
                        )
                        
                        chunks.append(chunk)
                        
                        logger.debug(f"Extracted {chunk_type} '{chunk_name}' from {Path(file_path).name}")
            
            # If no functions/classes found, create a module-level chunk
            if not chunks:
                chunk_id = self.generate_chunk_id(file_path, "module", "module")
                
                chunk = CodeChunk(
                    chunk_id=chunk_id,
                    text=source_code,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                    chunk_name=Path(file_path).stem,
                    chunk_type="module",
                    language="python"
                )
                
                chunks.append(chunk)
                logger.debug(f"Created module chunk for {Path(file_path).name}")
            
            return chunks
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            # Fallback to whole file
            return self._create_fallback_chunk(file_path, source_code)
        
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return self._create_fallback_chunk(file_path, source_code)
    
    def _create_fallback_chunk(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """Create a single chunk for the entire file as fallback"""
        chunk_id = self.generate_chunk_id(file_path, "file", "fallback")
        
        chunk = CodeChunk(
            chunk_id=chunk_id,
            text=source_code,
            file_path=file_path,
            start_line=1,
            end_line=len(source_code.splitlines()),
            chunk_name=Path(file_path).stem,
            chunk_type="fallback",
            language="python"
        )
        
        return [chunk]

class RegexChunker:
    """
    Simple regex-based chunker for non-Python files
    Looks for common patterns in various languages
    """
    
    def __init__(self):
        # Language-specific patterns
        self.patterns = {
            '.js': [
                (r'^(export\s+)?(async\s+)?function\s+(\w+)', 'function'),
                (r'^(export\s+)?(default\s+)?class\s+(\w+)', 'class'),
                (r'^const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', 'arrow_function'),
            ],
            '.ts': [
                (r'^(export\s+)?(async\s+)?function\s+(\w+)', 'function'),
                (r'^(export\s+)?(default\s+)?class\s+(\w+)', 'class'),
                (r'^(export\s+)?interface\s+(\w+)', 'interface'),
                (r'^(export\s+)?type\s+(\w+)', 'type'),
            ],
            '.go': [
                (r'^func\s+(\w+)', 'function'),
                (r'^func\s+\([^)]+\)\s+(\w+)', 'method'),
                (r'^type\s+(\w+)\s+struct', 'struct'),
                (r'^type\s+(\w+)\s+interface', 'interface'),
            ],
            '.rs': [
                (r'^(pub\s+)?fn\s+(\w+)', 'function'),
                (r'^(pub\s+)?(struct|enum)\s+(\w+)', 'type'),
                (r'^impl(?:\s*<[^>]+>)?\s+(\w+)', 'impl'),
            ]
        }
    
    def generate_chunk_id(self, file_path: str, chunk_name: str, chunk_type: str) -> str:
        """Generate stable UUID for chunk"""
        content = f"{file_path}:{chunk_type}:{chunk_name}"
        # Create deterministic UUID from content hash
        hash_digest = hashlib.sha256(content.encode()).digest()
        return str(uuid.UUID(bytes=hash_digest[:16]))
    
    def extract_chunks(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """Extract chunks using regex patterns"""
        file_ext = Path(file_path).suffix
        
        if file_ext not in self.patterns:
            return self._create_fallback_chunk(file_path, source_code)
        
        chunks = []
        lines = source_code.splitlines()
        patterns = self.patterns[file_ext]
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            for pattern, chunk_type in patterns:
                match = re.match(pattern, line)
                if match:
                    # Extract function/class name (usually the last capture group)
                    chunk_name = match.groups()[-1]
                    
                    # Find the end of this chunk (simple heuristic)
                    start_line = i + 1
                    end_line = self._find_chunk_end(lines, i, file_ext)
                    
                    chunk_lines = lines[i:end_line]
                    chunk_text = '\n'.join(chunk_lines)
                    
                    chunk_id = self.generate_chunk_id(file_path, chunk_name, chunk_type)
                    
                    chunk = CodeChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        chunk_name=chunk_name,
                        chunk_type=chunk_type,
                        language=file_ext[1:]  # Remove the dot
                    )
                    
                    chunks.append(chunk)
                    logger.debug(f"Extracted {chunk_type} '{chunk_name}' from {Path(file_path).name}")
                    break  # Only match first pattern per line
        
        # Fallback to whole file if no chunks found
        if not chunks:
            return self._create_fallback_chunk(file_path, source_code)
        
        return chunks
    
    def _find_chunk_end(self, lines: List[str], start_idx: int, file_ext: str) -> int:
        """Simple heuristic to find the end of a code chunk"""
        brace_count = 0
        i = start_idx
        
        # Look for opening brace
        while i < len(lines) and '{' not in lines[i]:
            i += 1
        
        if i >= len(lines):
            return min(start_idx + 20, len(lines))  # Default chunk size
        
        # Count braces to find matching closing brace
        for j in range(i, len(lines)):
            line = lines[j]
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and '{' in lines[i]:
                return j + 1
        
        # Default fallback
        return min(start_idx + 30, len(lines))
    
    def _create_fallback_chunk(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """Create a single chunk for the entire file as fallback"""
        chunk_id = self.generate_chunk_id(file_path, "file", "fallback")
        
        chunk = CodeChunk(
            chunk_id=chunk_id,
            text=source_code,
            file_path=file_path,
            start_line=1,
            end_line=len(source_code.splitlines()),
            chunk_name=Path(file_path).stem,
            chunk_type="fallback",
            language=Path(file_path).suffix[1:] if Path(file_path).suffix else "text"
        )
        
        return [chunk]

class SmartCodeChunker:
    """
    Smart code chunker that routes to appropriate chunker based on file type
    """
    
    def __init__(self):
        self.python_chunker = PythonASTChunker()
        self.regex_chunker = RegexChunker()
        
        logger.info("Initialized smart code chunker with AST and regex support")
    
    def chunk_file(self, file_path: str, source_code: str) -> List[CodeChunk]:
        """
        Chunk a file using the most appropriate method
        
        Args:
            file_path: Path to the file
            source_code: File content as string
            
        Returns:
            List of CodeChunk objects
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.py':
                # Use AST for Python files
                chunks = self.python_chunker.extract_chunks(file_path, source_code)
                logger.debug(f"AST chunking extracted {len(chunks)} chunks from {Path(file_path).name}")
                return chunks
            
            elif file_ext in ['.js', '.ts', '.jsx', '.tsx', '.go', '.rs']:
                # Use regex for other supported languages
                chunks = self.regex_chunker.extract_chunks(file_path, source_code)
                logger.debug(f"Regex chunking extracted {len(chunks)} chunks from {Path(file_path).name}")
                return chunks
            
            else:
                # Fallback for unsupported file types
                logger.debug(f"Using fallback chunking for {file_ext} file: {Path(file_path).name}")
                return self.regex_chunker._create_fallback_chunk(file_path, source_code)
        
        except Exception as e:
            logger.error(f"Chunking failed for {file_path}: {e}")
            # Ultimate fallback
            return self.regex_chunker._create_fallback_chunk(file_path, source_code)
    
    def chunk_directory(self, directory_path: str, file_extensions: Optional[List[str]] = None) -> List[CodeChunk]:
        """
        Chunk all supported files in a directory
        
        Args:
            directory_path: Path to directory
            file_extensions: List of extensions to include (e.g., ['.py', '.js'])
            
        Returns:
            List of all chunks from all files
        """
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs']
        
        directory = Path(directory_path)
        all_chunks = []
        
        for ext in file_extensions:
            for file_path in directory.rglob(f"*{ext}"):
                # Skip common ignore patterns
                if any(ignore in file_path.parts for ignore in ['__pycache__', '.venv', 'node_modules', '.git']):
                    continue
                
                try:
                    source_code = file_path.read_text(encoding='utf-8')
                    file_chunks = self.chunk_file(str(file_path), source_code)
                    all_chunks.extend(file_chunks)
                    
                except UnicodeDecodeError:
                    logger.warning(f"Could not decode {file_path} - skipping")
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Extracted {len(all_chunks)} total chunks from {directory_path}")
        return all_chunks

# Testing and example usage
if __name__ == "__main__":
    import tempfile
    
    def test_chunking():
        """Test the code chunker with sample files"""
        print("🧩 Testing Smart Code Chunker...")
        
        chunker = SmartCodeChunker()
        
        # Test Python code
        python_code = '''
class NeuralSearch:
    """Main neural search class"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.embeddings = {}
    
    async def embed_text(self, text):
        """Generate embeddings for text"""
        return [0.1] * 768
    
    def search(self, query, top_k=10):
        """Search for similar documents"""
        results = []
        return results

def main():
    """Main function"""
    search = NeuralSearch("nomic-embed")
    print("Neural search initialized")

if __name__ == "__main__":
    main()
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            temp_path = f.name
        
        try:
            chunks = chunker.chunk_file(temp_path, python_code)
            
            print(f"📊 Extracted {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. {chunk.chunk_type} '{chunk.chunk_name}' (lines {chunk.start_line}-{chunk.end_line})")
                print(f"     ID: {chunk.chunk_id}")
                print(f"     Length: {len(chunk.text)} chars")
                print()
            
        finally:
            Path(temp_path).unlink()
        
        print("✅ Code chunker test completed")
    
    test_chunking()