#!/usr/bin/env python3
"""
Real-time file watcher for automatic Qdrant indexing
Monitors project files and automatically updates neural memory
"""

import os
import time
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, Set, Optional, List
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
import httpx
from qdrant_client import QdrantClient, models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralIndexer(FileSystemEventHandler):
    """Handles file system events and updates Qdrant index"""
    
    def __init__(self, project_path: str, qdrant_host: str = "localhost", qdrant_port: int = 6333,
                 embedding_service_url: str = "http://localhost:8081", project_name: str = "default"):
        self.project_path = Path(project_path)
        self.project_name = project_name
        self.collection_prefix = f"project_{project_name}_"
        
        # Qdrant client
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_url = embedding_service_url
        
        # Track indexed files and their hashes
        self.file_hashes: Dict[str, str] = {}
        self.indexing_queue: asyncio.Queue = asyncio.Queue()
        
        # File patterns to watch
        self.watch_patterns = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', 
            '.rs', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php',
            '.md', '.yaml', '.yml', '.json', '.toml', '.txt'
        }
        
        # Patterns to ignore
        self.ignore_patterns = {
            '__pycache__', '.git', 'node_modules', '.venv', 'venv',
            'dist', 'build', '.next', 'target', '.pytest_cache',
            '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.log'
        }
        
        # Batch settings
        self.batch_size = 10
        self.batch_timeout = 5.0  # seconds
        self.pending_files: Set[str] = set()
        self.last_batch_time = time.time()
        
        # Initialize collections
        self._ensure_collections()
        
    def _ensure_collections(self):
        """Ensure required collections exist"""
        code_collection = f"{self.collection_prefix}code"
        
        try:
            self.qdrant.get_collection(code_collection)
        except:
            # Create collection with hybrid search config
            self.qdrant.create_collection(
                collection_name=code_collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=768,  # Nomic v2 dimension
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False
                        )
                    )
                }
            )
            logger.info(f"Created collection: {code_collection}")
    
    def should_index(self, file_path: str) -> bool:
        """Check if file should be indexed"""
        path = Path(file_path)
        
        # Check if it's a file we care about
        if not any(str(path).endswith(ext) for ext in self.watch_patterns):
            return False
        
        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in str(path):
                return False
        
        return True
    
    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get hash of file contents"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    async def get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings from Nomic service"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embedding_url}/embed",
                    json={"texts": texts}
                )
                if response.status_code == 200:
                    return response.json()["embeddings"]
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
        return None
    
    async def index_file(self, file_path: str, action: str = "update"):
        """Index a single file"""
        if not self.should_index(file_path):
            return
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip empty files
            if not content.strip():
                return
            
            # Get file metadata
            path = Path(file_path)
            relative_path = path.relative_to(self.project_path)
            
            # Chunk large files (simple line-based chunking for now)
            max_chunk_size = 1000  # lines
            lines = content.split('\n')
            chunks = []
            
            for i in range(0, len(lines), max_chunk_size):
                chunk_lines = lines[i:i + max_chunk_size]
                chunk_text = '\n'.join(chunk_lines)
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'start_line': i + 1,
                        'end_line': min(i + max_chunk_size, len(lines))
                    })
            
            if not chunks:
                chunks = [{'text': content, 'start_line': 1, 'end_line': len(lines)}]
            
            # Get embeddings for all chunks
            texts = [f"File: {relative_path}\n{chunk['text'][:500]}" for chunk in chunks]
            embeddings = await self.get_embeddings(texts)
            
            if not embeddings:
                logger.warning(f"No embeddings generated for {file_path}")
                return
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = hash(f"{file_path}_{i}_{datetime.now()}") % (10**15)
                
                # Create sparse vector from keywords (simple TF-IDF style)
                words = chunk['text'].lower().split()
                word_freq = {}
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                vocab_size = 10000
                sparse_indices = []
                sparse_values = []
                for word, freq in list(word_freq.items())[:100]:  # Top 100 words
                    idx = hash(word) % vocab_size
                    sparse_indices.append(idx)
                    sparse_values.append(float(freq))
                
                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embedding,
                        "sparse": models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values
                        )
                    },
                    payload={
                        "file_path": str(relative_path),
                        "full_path": file_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "start_line": chunk['start_line'],
                        "end_line": chunk['end_line'],
                        "content": chunk['text'][:1000],  # Store first 1000 chars
                        "file_type": path.suffix,
                        "indexed_at": datetime.now().isoformat(),
                        "action": action,
                        "project": self.project_name
                    }
                )
                points.append(point)
            
            # Upsert to Qdrant
            collection_name = f"{self.collection_prefix}code"
            self.qdrant.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Update hash tracking
            self.file_hashes[file_path] = self.get_file_hash(file_path)
            
            logger.info(f"Indexed {file_path}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
    
    async def remove_file_from_index(self, file_path: str):
        """Remove file from index when deleted"""
        try:
            path = Path(file_path)
            relative_path = path.relative_to(self.project_path)
            
            # Search for all points related to this file
            collection_name = f"{self.collection_prefix}code"
            results = self.qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="full_path",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=100
            )
            
            # Delete all points
            point_ids = [point.id for point in results[0]]
            if point_ids:
                self.qdrant.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                logger.info(f"Removed {len(point_ids)} chunks for {file_path}")
            
            # Remove from hash tracking
            self.file_hashes.pop(file_path, None)
            
        except Exception as e:
            logger.error(f"Failed to remove {file_path} from index: {e}")
    
    async def process_batch(self):
        """Process a batch of file updates"""
        if not self.pending_files:
            return
        
        files_to_process = list(self.pending_files)
        self.pending_files.clear()
        
        logger.info(f"Processing batch of {len(files_to_process)} files")
        
        for file_path in files_to_process:
            if os.path.exists(file_path):
                # Check if file actually changed
                current_hash = self.get_file_hash(file_path)
                if current_hash and current_hash != self.file_hashes.get(file_path):
                    await self.index_file(file_path, "update")
            else:
                # File was deleted
                await self.remove_file_from_index(file_path)
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory and self.should_index(event.src_path):
            logger.debug(f"File modified: {event.src_path}")
            self.pending_files.add(event.src_path)
    
    def on_created(self, event):
        """Handle file creation"""
        if not event.is_directory and self.should_index(event.src_path):
            logger.debug(f"File created: {event.src_path}")
            self.pending_files.add(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion"""
        if not event.is_directory and self.should_index(event.src_path):
            logger.debug(f"File deleted: {event.src_path}")
            self.pending_files.add(event.src_path)
    
    async def initial_index(self):
        """Perform initial indexing of all files"""
        logger.info(f"Starting initial index of {self.project_path}")
        
        files_to_index = []
        for ext in self.watch_patterns:
            files_to_index.extend(self.project_path.rglob(f"*{ext}"))
        
        # Filter out ignored paths
        files_to_index = [f for f in files_to_index if self.should_index(str(f))]
        
        logger.info(f"Found {len(files_to_index)} files to index")
        
        # Index in batches
        for i in range(0, len(files_to_index), self.batch_size):
            batch = files_to_index[i:i + self.batch_size]
            for file_path in batch:
                await self.index_file(str(file_path), "initial")
            await asyncio.sleep(0.1)  # Small delay between batches
        
        logger.info("Initial indexing complete")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time file indexer for Neural Tools")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--project-name", default="default", help="Project name")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", default=6333, type=int, help="Qdrant port")
    parser.add_argument("--embedding-url", default="http://localhost:8081", help="Embedding service URL")
    parser.add_argument("--initial-index", action="store_true", help="Perform initial indexing")
    
    args = parser.parse_args()
    
    # Create indexer
    indexer = NeuralIndexer(
        project_path=args.project_path,
        project_name=args.project_name,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        embedding_service_url=args.embedding_url
    )
    
    # Perform initial index if requested
    if args.initial_index:
        await indexer.initial_index()
    
    # Set up file watcher
    observer = Observer()
    observer.schedule(indexer, args.project_path, recursive=True)
    observer.start()
    
    logger.info(f"Watching {args.project_path} for changes...")
    
    try:
        # Process batches periodically
        while True:
            await asyncio.sleep(indexer.batch_timeout)
            if indexer.pending_files:
                await indexer.process_batch()
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopping file watcher")
    
    observer.join()

if __name__ == "__main__":
    asyncio.run(main())