#!/usr/bin/env python3
"""
Memory System Wrapper
Provides clean interface for global MCP server to access L9 memory functionality
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import the L9 memory system
from l9_qdrant_memory_v2 import L9QdrantMemoryV2, MemoryResult

logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Clean wrapper around L9QdrantMemoryV2 for global MCP server
    Provides simplified interface while maintaining all functionality
    """
    
    def __init__(self, qdrant_host: str = None, qdrant_port: int = None, 
                 embedder=None, collection_prefix: str = ""):
        """
        Initialize memory system wrapper
        Uses global unified container instead of L9 auto-creation
        """
        # Override L9 container creation with direct connection to global system
        self.qdrant_host = qdrant_host or "localhost"
        self.qdrant_port = qdrant_port or 6678  # Global unified container port
        self.collection_prefix = collection_prefix or "claude-l9-template"
        
        # Initialize L9 memory but prevent container auto-creation
        self.l9_memory = L9QdrantMemoryV2()
        self._initialized = False
        
    async def initialize(self):
        """Initialize the underlying memory system"""
        if not self._initialized:
            await self.l9_memory.initialize()
            self._initialized = True
            logger.info("✅ MemorySystem wrapper initialized")
    
    async def store_memory(self, content: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a memory and return its ID
        
        Args:
            content: The content to store
            metadata: Additional metadata dict
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        if not self._initialized:
            await self.initialize()
        
        # Generate unique memory ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        memory_id = f"mem_{timestamp}"
        
        # Store using L9 system
        success = await self.l9_memory.index_memory(
            memory_id=memory_id,
            content=content,
            metadata=metadata
        )
        
        if success:
            logger.info(f"💾 Stored memory: {memory_id}")
            return memory_id
        else:
            raise Exception("Failed to store memory")
    
    async def search_memories(self, query: str, limit: int = 10, 
                            similarity_threshold: float = 0.0) -> List[Dict]:
        """
        Search memories in current project
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (unused in L9 system)
            
        Returns:
            List of memory results as dicts with score and payload
        """
        if not self._initialized:
            await self.initialize()
        
        # Search using L9 system
        results = await self.l9_memory.search_project_memories(
            query=query,
            limit=limit,
            include_other_projects=False
        )
        
        # Convert MemoryResult objects to dict format expected by MCP
        formatted_results = []
        for result in results:
            formatted_results.append({
                'score': result.score,
                'payload': {
                    'content': result.content,
                    'timestamp': result.timestamp.isoformat(),
                    'project': result.project,
                    'token_count': result.token_count,
                    'entities': result.entities,
                    **result.metadata
                }
            })
        
        return formatted_results
    
    async def search_memories_global(self, query: str, limit: int = 10,
                                   similarity_threshold: float = 0.0) -> List[Dict]:
        """
        Search memories across all projects
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of memory results from all projects
        """
        if not self._initialized:
            await self.initialize()
        
        # Search across all projects using L9 system
        results = await self.l9_memory.search_project_memories(
            query=query,
            limit=limit,
            include_other_projects=True  # Enable cross-project search
        )
        
        # Convert to expected format
        formatted_results = []
        for result in results:
            formatted_results.append({
                'score': result.score,
                'project_name': result.project,  # Add project name for global search
                'payload': {
                    'content': result.content,
                    'timestamp': result.timestamp.isoformat(),
                    'project': result.project,
                    'token_count': result.token_count,
                    'entities': result.entities,
                    **result.metadata
                }
            })
        
        return formatted_results
    
    async def search_code(self, query: str, file_types: Optional[List[str]] = None,
                         limit: int = 10) -> List[Dict]:
        """
        Search code snippets (placeholder - would need code indexing system)
        For now, searches memories tagged as code
        
        Args:
            query: Search query
            file_types: File extensions to filter by
            limit: Maximum results
            
        Returns:
            List of code search results
        """
        if not self._initialized:
            await self.initialize()
        
        # For now, search for memories that might contain code
        # This would be enhanced with proper code indexing
        code_query = f"code {query} function class method"
        
        results = await self.search_memories(
            query=code_query,
            limit=limit
        )
        
        # Filter for code-like content (basic heuristic)
        code_results = []
        for result in results:
            content = result['payload']['content']
            # Simple heuristic to detect code content
            if any(keyword in content.lower() for keyword in 
                  ['function', 'class', 'def ', 'import', 'const ', 'var ', 'let ']):
                code_results.append({
                    'score': result['score'],
                    'payload': {
                        'file_path': result['payload'].get('file_path', 'unknown'),
                        'line_number': result['payload'].get('line_number', 'unknown'),
                        'content': content,
                        **result['payload']
                    }
                })
        
        return code_results
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics
        
        Returns:
            Dict with system stats
        """
        if not self._initialized:
            await self.initialize()
        
        # Get stats from L9 system
        l9_stats = await self.l9_memory.get_project_stats()
        
        # Format for MCP interface
        return {
            'total_memories': l9_stats.get('project_memories', 0),
            'code_snippets': 0,  # Would be tracked separately with code indexing
            'last_updated': datetime.now().isoformat(),
            'project': l9_stats.get('project'),
            'status': l9_stats.get('status', 'unknown'),
            'l9_details': l9_stats  # Include full L9 stats for debugging
        }
    
    def get_project_name(self) -> str:
        """Get the current project name"""
        return self.l9_memory.project_name if self._initialized else "unknown"
    
    def get_collection_name(self) -> str:
        """Get the current collection name"""
        return self.l9_memory.collection_name if self._initialized else "unknown"

# Legacy compatibility
class CodeSpecificEmbedder:
    """Placeholder embedder - L9 system handles embeddings internally"""
    
    def __init__(self):
        logger.info("🤖 CodeSpecificEmbedder initialized (L9 handles embeddings internally)")
    
    async def encode(self, text: str) -> List[float]:
        """Placeholder - L9 system handles encoding"""
        return []