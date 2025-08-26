#!/usr/bin/env python3
"""
Project Neural Indexer - Sprint 1: Schema + Baseline Index
Intelligent project codebase indexing with neural embeddings

Based on Gemini's architecture analysis:
- Unified schema with file_type classification
- Neural embeddings via ONNX all-MiniLM-L6-v2
- ChromaDB for vector storage
- SQLite for metadata management
- Foundation for type weighting + freshness (Sprint 2-3)
"""

import os
import sys
import hashlib
import sqlite3
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add .claude/neural-system directory to path for neural system import
neural_system_dir = Path(__file__).parent.parent / "neural-system"
sys.path.insert(0, str(neural_system_dir))

from neural_embeddings import HybridEmbeddingSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProjectFile:
    """Represents a project file with metadata"""
    id: Optional[int]
    file_path: str
    file_hash: str
    file_type: str
    last_modified_git: int
    last_indexed_at: int
    chunk_count: int
    total_lines: int
    content: str = ""

@dataclass
class ProjectChunk:
    """Represents a searchable chunk of project content"""
    id: str
    file_path: str
    content: str
    chunk_index: int
    start_line: int
    end_line: int
    neural_score: float = 0.0

class ProjectFileClassifier:
    """Classifies project files into categories for intelligent weighting"""
    
    # File type mapping based on Gemini's analysis
    FILE_CATEGORIES = {
        # Core application logic - highest priority
        'CORE_CODE': {
            'weight': 1.0,
            'extensions': ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java', '.cpp', '.c', '.h'],
            'description': 'Primary application logic and implementation'
        },
        
        # Test code - high priority for understanding usage
        'TEST_CODE': {
            'weight': 0.7, 
            'extensions': ['.test.js', '.spec.js', '.test.ts', '.spec.ts'],
            'patterns': ['test_*.py', '*_test.py', '*_test.go', 'test*.py'],
            'description': 'Unit, integration, and e2e tests'
        },
        
        # Data structures and schemas - very high priority
        'DATA_ASSET': {
            'weight': 0.9,
            'extensions': ['.sql', '.proto', '.graphql', '.avro', '.thrift'],
            'description': 'Data schemas, migrations, and structure definitions'
        },
        
        # Configuration - important for project understanding
        'CONFIGURATION': {
            'weight': 0.8,
            'extensions': ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'],
            'patterns': ['Dockerfile', 'docker-compose*', '*.dockerfile', 'Makefile', 'CMakeLists.txt'],
            'description': 'Environment, deployment, and setup configuration'
        },
        
        # Build and deployment scripts
        'BUILD_SCRIPT': {
            'weight': 0.6,
            'extensions': ['.sh', '.bash', '.bat', '.ps1'],
            'patterns': ['webpack.config.*', 'rollup.config.*', 'vite.config.*'],
            'description': 'Build, compile, and deployment scripts'
        },
        
        # Documentation - lowest priority but still indexed
        'DOCUMENTATION': {
            'weight': 0.4,
            'extensions': ['.md', '.rst', '.txt', '.adoc'],
            'description': 'Markdown and text documentation'
        },
        
        # Other files
        'OTHER': {
            'weight': 0.2,
            'extensions': ['.gitignore', '.prettierrc', '.eslintrc'],
            'description': 'Miscellaneous configuration and ignore files'
        }
    }
    
    @classmethod
    def classify_file(cls, file_path: str) -> Tuple[str, float]:
        """
        Classify a file and return (category, weight)
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (category_name, weight)
        """
        file_path_lower = file_path.lower()
        filename = Path(file_path).name.lower()
        
        # Check each category
        for category, config in cls.FILE_CATEGORIES.items():
            # Check extensions
            if 'extensions' in config:
                for ext in config['extensions']:
                    if file_path_lower.endswith(ext.lower()):
                        return category, config['weight']
            
            # Check patterns
            if 'patterns' in config:
                for pattern in config['patterns']:
                    if cls._matches_pattern(filename, pattern.lower()):
                        return category, config['weight']
        
        # Default to OTHER if no match
        return 'OTHER', cls.FILE_CATEGORIES['OTHER']['weight']
    
    @classmethod
    def _matches_pattern(cls, filename: str, pattern: str) -> bool:
        """Simple pattern matching with * wildcard"""
        if '*' not in pattern:
            return filename == pattern
        
        if pattern.startswith('*') and pattern.endswith('*'):
            return pattern[1:-1] in filename
        elif pattern.startswith('*'):
            return filename.endswith(pattern[1:])
        elif pattern.endswith('*'):
            return filename.startswith(pattern[:-1])
        
        return filename == pattern

class ProjectNeuralIndexer:
    """
    Neural-powered project indexer for intelligent codebase context
    Sprint 1: Basic schema + neural indexing foundation
    """
    
    def __init__(self, project_root: str = ".", db_path: str = ".claude/memory/project-knowledge.db"):
        """Initialize project neural indexer"""
        
        self.project_root = Path(project_root).resolve()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize neural embeddings system
        logger.info("🧠 Initializing project neural indexer...")
        self.neural_system = HybridEmbeddingSystem()
        
        # Initialize database
        self._init_database()
        
        # ChromaDB collection for project embeddings  
        self.collection_name = "project_codebase"
        self._init_chromadb()
        
        logger.info("🚀 Project Neural Indexer ready")
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Balanced performance
        
        # Create unified project files table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS project_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                file_type TEXT NOT NULL,
                last_modified_git INTEGER NOT NULL,
                last_indexed_at INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                total_lines INTEGER NOT NULL
            )
        """)
        
        # Performance indexes
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_file_path ON project_files(file_path)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON project_files(file_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_last_modified ON project_files(last_modified_git)")
        
        self.conn.commit()
        logger.info("✅ Project database schema initialized")
    
    def _init_chromadb(self):
        """Initialize ChromaDB collection for embeddings"""
        try:
            # Use the vector store from the neural system
            self.vector_store = self.neural_system.vector_store
            
            # Get or create collection with custom name
            chroma_client = self.vector_store.client
            self.collection = chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Project codebase neural embeddings"}
            )
            logger.info(f"✅ ChromaDB collection '{self.collection_name}' ready")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path}: {e}")
            return ""
    
    def _get_git_last_modified(self, file_path: Path) -> int:
        """Get last Git commit timestamp for file (placeholder for Sprint 2)"""
        # For Sprint 1, use filesystem mtime as placeholder
        # Sprint 2 will implement: git log -1 --format=%ct -- <file_path>
        try:
            return int(file_path.stat().st_mtime)
        except:
            return int(time.time())
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines in file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def _should_index_file(self, file_path: Path) -> bool:
        """Determine if file should be indexed"""
        
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            # Allow .claude directory files
            if '.claude' in str(file_path):
                pass  # Continue checking other conditions
            else:
                return False
        
        # Skip common ignore patterns
        ignore_patterns = [
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            '.pytest_cache', '.coverage', 'dist', 'build', '.DS_Store'
        ]
        
        if any(pattern in str(file_path) for pattern in ignore_patterns):
            return False
        
        # Skip binary files by extension
        binary_extensions = [
            '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.bin',
            '.jpg', '.jpeg', '.png', '.gif', '.ico', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.zip', '.tar', '.gz'
        ]
        
        if file_path.suffix.lower() in binary_extensions:
            return False
        
        # Skip very large files (>1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                return False
        except:
            pass
        
        return True
    
    def _chunk_file_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunk file content into searchable segments
        Sprint 1: Simple paragraph-based chunking
        Sprint 4: Will implement language-aware Tree-sitter chunking
        """
        
        lines = content.split('\n')
        chunks = []
        
        # Simple chunking: ~50 lines per chunk with 10-line overlap
        chunk_size = 50
        overlap = 10
        
        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines).strip()
            
            if len(chunk_content) < 20:  # Skip tiny chunks
                continue
            
            chunks.append({
                'content': chunk_content,
                'chunk_index': len(chunks),
                'start_line': i + 1,
                'end_line': min(i + chunk_size, len(lines)),
                'file_path': file_path
            })
        
        return chunks
    
    def index_file(self, file_path: Path) -> bool:
        """
        Index a single file with neural embeddings
        
        Args:
            file_path: Path to file to index
            
        Returns:
            True if indexed successfully, False if skipped/failed
        """
        
        if not self._should_index_file(file_path):
            logger.debug(f"Skipping file: {file_path}")
            return False
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Calculate metadata
            relative_path = str(file_path.relative_to(self.project_root))
            file_hash = self._calculate_file_hash(file_path)
            file_type, type_weight = ProjectFileClassifier.classify_file(str(file_path))
            last_modified = self._get_git_last_modified(file_path)
            total_lines = self._count_lines(file_path)
            
            # Check if file needs re-indexing
            cursor = self.conn.execute(
                "SELECT file_hash FROM project_files WHERE file_path = ?",
                (relative_path,)
            )
            existing = cursor.fetchone()
            
            if existing and existing[0] == file_hash:
                # File unchanged, skip
                return True
            
            # Chunk file content
            chunks = self._chunk_file_content(content, relative_path)
            
            if not chunks:
                return False
            
            # Generate embeddings for each chunk
            chunk_ids = []
            chunk_embeddings = []
            chunk_metadatas = []
            
            for chunk in chunks:
                # Generate embedding with metadata
                metadata = {
                    'file_path': relative_path,
                    'chunk_index': chunk['chunk_index'],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'file_type': file_type,
                    'type_weight': type_weight
                }
                
                neural_result = self.neural_system.generate_embedding(chunk['content'], metadata)
                
                # Create unique chunk ID
                chunk_id = f"{relative_path}:chunk_{chunk['chunk_index']}"
                
                chunk_ids.append(chunk_id)
                chunk_embeddings.append(neural_result.embedding.tolist())
                
                # Filter metadata to only include ChromaDB-compatible types
                filtered_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        filtered_metadata[key] = value
                    else:
                        # Convert complex types to strings
                        filtered_metadata[f"{key}_str"] = str(value)
                
                chunk_metadatas.append(filtered_metadata)
            
            # Store embeddings in ChromaDB
            if existing:
                # Remove old embeddings first
                old_cursor = self.conn.execute(
                    "SELECT chunk_count FROM project_files WHERE file_path = ?",
                    (relative_path,)
                )
                old_chunk_count = old_cursor.fetchone()[0]
                
                old_chunk_ids = [f"{relative_path}:chunk_{i}" for i in range(old_chunk_count)]
                try:
                    self.collection.delete(ids=old_chunk_ids)
                except:
                    pass  # Ignore if chunks don't exist
            
            # Add new embeddings
            self.collection.add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadatas
            )
            
            # Update database record
            now = int(time.time())
            
            self.conn.execute("""
                INSERT OR REPLACE INTO project_files 
                (file_path, file_hash, file_type, last_modified_git, last_indexed_at, chunk_count, total_lines)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (relative_path, file_hash, file_type, last_modified, now, len(chunks), total_lines))
            
            self.conn.commit()
            
            logger.info(f"📁 Indexed {relative_path} ({file_type}, {len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            return False
    
    def index_project(self, max_files: int = None) -> Dict[str, int]:
        """
        Index entire project directory
        
        Args:
            max_files: Maximum files to index (None for unlimited)
            
        Returns:
            Dictionary with indexing statistics
        """
        
        logger.info(f"🔍 Starting project indexing: {self.project_root}")
        start_time = time.time()
        
        stats = {
            'total_files': 0,
            'indexed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_chunks': 0
        }
        
        # Walk project directory
        for file_path in self.project_root.rglob('*'):
            if not file_path.is_file():
                continue
            
            stats['total_files'] += 1
            
            if max_files and stats['indexed_files'] >= max_files:
                break
            
            # Index file
            if self.index_file(file_path):
                stats['indexed_files'] += 1
            else:
                stats['skipped_files'] += 1
        
        # Get chunk count
        cursor = self.conn.execute("SELECT SUM(chunk_count) FROM project_files")
        stats['total_chunks'] = cursor.fetchone()[0] or 0
        
        duration = time.time() - start_time
        logger.info(f"✅ Project indexing complete in {duration:.1f}s")
        logger.info(f"📊 Stats: {stats}")
        
        return stats
    
    def search_project(self, query: str, limit: int = 10) -> List[ProjectChunk]:
        """
        Search project with neural semantic similarity
        Sprint 1: Basic neural search only
        Sprint 3: Will add type weighting and freshness scoring
        
        Args:
            query: Search query
            limit: Maximum results to return
            
        Returns:
            List of relevant project chunks
        """
        
        start_time = time.time()
        
        # Generate query embedding
        query_result = self.neural_system.generate_embedding(query)
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_result.embedding.tolist()],
            n_results=limit,
            include=['metadatas', 'distances', 'documents']
        )
        
        # Convert to ProjectChunk objects
        chunks = []
        
        for i, (chunk_id, metadata, distance, content) in enumerate(zip(
            results['ids'][0],
            results['metadatas'][0], 
            results['distances'][0],
            results['documents'][0] if results['documents'] else []
        )):
            # Convert distance to similarity score (0-1, higher is better)
            neural_score = max(0, 1.0 - distance)
            
            chunk = ProjectChunk(
                id=chunk_id,
                file_path=metadata['file_path'],
                content=content or "",
                chunk_index=metadata.get('chunk_index', 0),
                start_line=metadata.get('start_line', 0),
                end_line=metadata.get('end_line', 0),
                neural_score=neural_score
            )
            chunks.append(chunk)
        
        duration = (time.time() - start_time) * 1000
        logger.info(f"🔍 Project search completed in {duration:.1f}ms ({len(chunks)} results)")
        
        return chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        
        cursor = self.conn.execute("""
            SELECT 
                file_type,
                COUNT(*) as count,
                SUM(chunk_count) as total_chunks,
                SUM(total_lines) as total_lines
            FROM project_files 
            GROUP BY file_type
            ORDER BY count DESC
        """)
        
        type_stats = {}
        for row in cursor.fetchall():
            file_type, count, chunks, lines = row
            type_stats[file_type] = {
                'files': count,
                'chunks': chunks,
                'lines': lines,
                'weight': ProjectFileClassifier.FILE_CATEGORIES.get(file_type, {}).get('weight', 0.0)
            }
        
        # Overall stats
        total_cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total_files,
                SUM(chunk_count) as total_chunks,
                SUM(total_lines) as total_lines
            FROM project_files
        """)
        total_stats = total_cursor.fetchone()
        
        return {
            'total_files': total_stats[0],
            'total_chunks': total_stats[1], 
            'total_lines': total_stats[2],
            'file_types': type_stats,
            'neural_dimensions': 384,  # ONNX all-MiniLM-L6-v2
            'collection_name': self.collection_name
        }
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'conn'):
            self.conn.close()
        logger.info("🔒 Project Neural Indexer closed")

def main():
    """CLI interface for project indexing"""
    
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    # Initialize indexer
    indexer = ProjectNeuralIndexer(project_root)
    
    try:
        # Index project
        stats = indexer.index_project(max_files=100)  # Limit for Sprint 1 testing
        
        # Show statistics
        print("\n📊 PROJECT INDEXING STATISTICS")
        print("=" * 50)
        
        project_stats = indexer.get_stats()
        print(f"Total Files: {project_stats['total_files']}")
        print(f"Total Chunks: {project_stats['total_chunks']}") 
        print(f"Total Lines: {project_stats['total_lines']}")
        
        print(f"\n📁 File Type Distribution:")
        for file_type, type_data in project_stats['file_types'].items():
            weight = type_data['weight']
            emoji = "🔥" if weight >= 0.9 else "⭐" if weight >= 0.7 else "📝" if weight >= 0.5 else "📄"
            print(f"   {emoji} {file_type}: {type_data['files']} files, {type_data['chunks']} chunks (weight: {weight})")
        
        # Test search
        test_queries = [
            "neural embeddings implementation",
            "database connection and setup", 
            "error handling and logging"
        ]
        
        print(f"\n🔍 Testing Neural Search:")
        for query in test_queries:
            results = indexer.search_project(query, limit=3)
            print(f"\nQuery: '{query}'")
            for i, chunk in enumerate(results):
                print(f"   {i+1}. {chunk.file_path} (score: {chunk.neural_score:.3f})")
                print(f"      Lines {chunk.start_line}-{chunk.end_line}: {chunk.content[:80]}...")
    
    finally:
        indexer.close()

if __name__ == "__main__":
    main()