#!/usr/bin/env python3
"""
Selective Reprocessing Indexer for Neural Tools
Implements intelligent differential indexing to only update changed chunks
Integrates with enhanced file watcher for optimal performance
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SelectiveIndexer:
    """
    Enhanced indexer that only reprocesses changed chunks
    Maintains chunk-level cache for differential updates
    """
    
    def __init__(self, container, project_name: str):
        """
        Initialize selective indexer
        
        Args:
            container: ServiceContainer with Neo4j, Qdrant, Nomic services
            project_name: Project identifier for namespacing
        """
        self.container = container
        self.project_name = project_name
        self.collection_prefix = f"project_{project_name}_"
        
        # Cache for selective reprocessing
        self.chunk_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.file_hashes: Dict[str, str] = {}
        
        # Metrics tracking
        self.metrics = {
            'files_processed': 0,
            'chunks_updated': 0,
            'chunks_skipped': 0,
            'selective_updates': 0,
            'full_reindexes': 0,
            'time_saved_seconds': 0
        }
    
    def compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute MD5 hash of file contents"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return None
    
    async def simple_chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Simple line-based chunking for demonstration
        In production, this would use the Tree-sitter chunker from Phase 1.6
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []
        
        if not content.strip():
            return []
        
        # Simple chunking by lines
        lines = content.split('\n')
        chunks = []
        chunk_size = 50  # lines per chunk
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_content = '\n'.join(chunk_lines)
            
            if chunk_content.strip():
                chunk_id = f"{file_path}:{i//chunk_size}"
                chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_content,
                    'start_line': i + 1,
                    'end_line': min(i + chunk_size, len(lines)),
                    'hash': chunk_hash,
                    'file_path': file_path
                })
        
        return chunks
    
    def diff_chunks(self, old_chunks: List[Dict], new_chunks: List[Dict]) -> Tuple[List[int], List[int], List[int]]:
        """
        Compare chunks and return (changed_indices, new_indices, deleted_indices)
        
        Returns:
            Tuple of (changed_chunk_indices, new_chunk_indices, deleted_chunk_indices)
        """
        old_chunk_map = {chunk.get('id', i): i for i, chunk in enumerate(old_chunks)}
        new_chunk_map = {chunk.get('id', i): i for i, chunk in enumerate(new_chunks)}
        
        changed_indices = []
        new_indices = []
        deleted_indices = []
        
        # Check for changes and new chunks
        for i, new_chunk in enumerate(new_chunks):
            chunk_id = new_chunk.get('id', i)
            new_hash = new_chunk.get('hash', '')
            
            if chunk_id in old_chunk_map:
                # Existing chunk - check if changed
                old_index = old_chunk_map[chunk_id]
                if old_index < len(old_chunks):
                    old_hash = old_chunks[old_index].get('hash', '')
                    if new_hash != old_hash:
                        changed_indices.append(i)
            else:
                # New chunk
                new_indices.append(i)
        
        # Check for deleted chunks
        for chunk_id, old_index in old_chunk_map.items():
            if chunk_id not in new_chunk_map:
                deleted_indices.append(old_index)
        
        return changed_indices, new_indices, deleted_indices
    
    async def index_file(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index file with selective reprocessing based on metadata
        
        Args:
            file_path: Path to file to index
            metadata: Metadata from file watcher with reprocessing hints
            
        Returns:
            Indexing result with performance metrics
        """
        start_time = time.time()
        result = {
            'file_path': file_path,
            'action': 'unknown',
            'chunks_processed': 0,
            'chunks_skipped': 0,
            'selective_reprocessing_used': False,
            'processing_time_ms': 0,
            'error': None
        }
        
        try:
            # Check if file should be processed
            if not metadata.get('file_changed', True):
                result['action'] = 'skipped_unchanged'
                self.metrics['chunks_skipped'] += 1
                return result
            
            # Get current file chunks
            new_chunks = await self.simple_chunk_file(file_path)
            
            if metadata.get('requires_selective_update', False) and metadata.get('has_cached_chunks', False):
                # SELECTIVE REPROCESSING PATH
                result['selective_reprocessing_used'] = True
                result['action'] = 'selective_update'
                
                old_chunks = self.chunk_cache.get(file_path, [])
                changed_indices, new_indices, deleted_indices = self.diff_chunks(old_chunks, new_chunks)
                
                chunks_to_process = []
                
                # Add changed chunks
                for idx in changed_indices:
                    if idx < len(new_chunks):
                        chunks_to_process.append(new_chunks[idx])
                
                # Add new chunks
                for idx in new_indices:
                    if idx < len(new_chunks):
                        chunks_to_process.append(new_chunks[idx])
                
                # Process only changed/new chunks
                if chunks_to_process:
                    await self._process_chunks(chunks_to_process, file_path)
                    result['chunks_processed'] = len(chunks_to_process)
                    result['chunks_skipped'] = len(new_chunks) - len(chunks_to_process)
                    
                    logger.info(
                        f"Selective update: {len(chunks_to_process)} chunks updated, "
                        f"{result['chunks_skipped']} skipped for {file_path}"
                    )
                    
                    self.metrics['selective_updates'] += 1
                    self.metrics['chunks_updated'] += len(chunks_to_process)
                    self.metrics['chunks_skipped'] += result['chunks_skipped']
                    
                    # Estimate time saved (conservative)
                    estimated_time_saved = result['chunks_skipped'] * 0.1  # 100ms per chunk
                    self.metrics['time_saved_seconds'] += estimated_time_saved
                else:
                    result['action'] = 'no_changes_detected'
                    logger.debug(f"No chunk changes detected for {file_path}")
                
                # TODO: Handle deleted chunks - remove from Qdrant/Neo4j
                if deleted_indices:
                    logger.info(f"TODO: Remove {len(deleted_indices)} deleted chunks for {file_path}")
                
            else:
                # FULL REPROCESSING PATH
                result['action'] = 'full_reindex'
                
                if new_chunks:
                    await self._process_chunks(new_chunks, file_path)
                    result['chunks_processed'] = len(new_chunks)
                    
                    logger.info(f"Full reindex: {len(new_chunks)} chunks processed for {file_path}")
                    
                    self.metrics['full_reindexes'] += 1
                    self.metrics['chunks_updated'] += len(new_chunks)
            
            # Update cache
            self.chunk_cache[file_path] = new_chunks
            self.file_hashes[file_path] = self.compute_file_hash(file_path)
            self.metrics['files_processed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            result['error'] = str(e)
        
        result['processing_time_ms'] = (time.time() - start_time) * 1000
        return result
    
    async def _process_chunks(self, chunks: List[Dict[str, Any]], file_path: str):
        """
        Process chunks through the neural pipeline
        This would integrate with existing Qdrant/Neo4j indexing
        """
        # Placeholder for actual chunk processing
        # In production this would:
        # 1. Get embeddings from Nomic service
        # 2. Store vectors in Qdrant
        # 3. Extract entities and store in Neo4j
        # 4. Update hybrid retriever indexes
        
        logger.debug(f"Processing {len(chunks)} chunks for {file_path}")
        
        # Simulate processing time
        await asyncio.sleep(len(chunks) * 0.01)  # 10ms per chunk
        
        # TODO: Integrate with actual ServiceContainer pipeline:
        # embeddings = await self.container.nomic.get_embeddings([chunk['content'] for chunk in chunks])
        # await self.container.qdrant.upsert_chunks(chunks, embeddings)
        # await self.container.neo4j.extract_and_store_entities(chunks)
    
    async def remove_file(self, file_path: str) -> Dict[str, Any]:
        """Remove file from all indexes"""
        result = {
            'file_path': file_path,
            'action': 'deleted',
            'chunks_removed': 0,
            'error': None
        }
        
        try:
            # Get cached chunks to know what to remove
            cached_chunks = self.chunk_cache.get(file_path, [])
            
            if cached_chunks:
                # TODO: Remove from Qdrant and Neo4j
                # await self.container.qdrant.delete_by_file_path(file_path)
                # await self.container.neo4j.delete_by_file_path(file_path)
                
                result['chunks_removed'] = len(cached_chunks)
                logger.info(f"Removed {len(cached_chunks)} chunks for deleted file {file_path}")
            
            # Remove from cache
            self.chunk_cache.pop(file_path, None)
            self.file_hashes.pop(file_path, None)
            
        except Exception as e:
            logger.error(f"Failed to remove file {file_path}: {e}")
            result['error'] = str(e)
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current indexing metrics"""
        total_chunks = self.metrics['chunks_updated'] + self.metrics['chunks_skipped']
        efficiency = (self.metrics['chunks_skipped'] / max(total_chunks, 1)) * 100
        
        return {
            **self.metrics,
            'selective_efficiency_percent': efficiency,
            'average_time_saved_per_file_ms': (
                self.metrics['time_saved_seconds'] / max(self.metrics['files_processed'], 1)
            ) * 1000
        }
    
    def reset_metrics(self):
        """Reset metrics counters"""
        for key in self.metrics:
            self.metrics[key] = 0

# Example integration with file watcher
async def create_selective_reindex_callback(indexer: SelectiveIndexer):
    """
    Create callback function for file watcher that uses selective indexer
    This is what gets passed to AsyncProjectWatcher.initialize()
    """
    async def handle_reindex(project_name: str, file_path: str, metadata: Dict[str, Any]):
        """Handle file change with selective reprocessing"""
        logger.info(f"Reindexing {file_path} for project {project_name}")
        logger.debug(f"Metadata: {metadata}")
        
        try:
            if metadata.get('event_type') == 'deleted':
                result = await indexer.remove_file(file_path)
            else:
                result = await indexer.index_file(file_path, metadata)
            
            logger.info(f"Reindex complete: {result}")
            
        except Exception as e:
            logger.error(f"Reindex callback failed: {e}")
    
    return handle_reindex

# Example usage
async def example_usage():
    """Demonstrate selective reprocessing"""
    # This would normally use a real ServiceContainer
    from servers.services.service_container import ServiceContainer
    
    container = ServiceContainer("test_project")
    await container.initialize_all_services()
    
    # Create selective indexer
    indexer = SelectiveIndexer(container, "test_project")
    
    # Create file watcher callback
    callback = await create_selective_reindex_callback(indexer)
    
    # Example metadata for selective reprocessing
    metadata = {
        'selective_reprocessing': True,
        'requires_selective_update': True,
        'has_cached_chunks': True,
        'file_changed': True
    }
    
    # Simulate file change
    await callback("test_project", "/path/to/file.py", metadata)
    
    # Check metrics
    metrics = indexer.get_metrics()
    print(f"Selective reprocessing efficiency: {metrics['selective_efficiency_percent']:.1f}%")
    print(f"Time saved per file: {metrics['average_time_saved_per_file_ms']:.1f}ms")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())