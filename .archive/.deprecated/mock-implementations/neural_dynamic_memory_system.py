#!/usr/bin/env python3
"""
Simplified Neural Dynamic Memory System for benchmarking
Mock implementation for testing performance benchmarks
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MemorySearchResult:
    """Search result from memory system"""
    content: str
    metadata: Dict[str, Any]
    score: float


class NeuralDynamicMemorySystem:
    """Simplified memory system for benchmarking tests"""
    
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory
        self.memories: List[Dict[str, Any]] = []
        self.initialized = False
    
    async def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Store memory item (mock implementation)"""
        if not self.initialized:
            await self._initialize()
        
        memory_id = f"mem_{len(self.memories):04d}"
        memory_item = {
            "id": memory_id,
            "content": memory_data.get("content", ""),
            "metadata": memory_data.get("metadata", {}),
            "type": memory_data.get("type", "unknown"),
            "timestamp": time.time()
        }
        
        self.memories.append(memory_item)
        return memory_id
    
    async def search_code_context(self, 
                                query: str, 
                                limit: int = 10,
                                similarity_threshold: float = 0.3) -> List[MemorySearchResult]:
        """Search for relevant memories (mock implementation)"""
        if not self.initialized:
            await self._initialize()
        
        # Simulate search latency
        await asyncio.sleep(random.uniform(0.01, 0.05))
        
        results = []
        
        # Simple keyword matching for testing
        query_words = set(query.lower().split())
        
        for memory in self.memories:
            content = memory["content"].lower()
            content_words = set(content.split())
            
            # Calculate simple overlap score
            overlap = len(query_words & content_words)
            total_words = len(query_words | content_words)
            score = overlap / total_words if total_words > 0 else 0.0
            
            if score >= similarity_threshold:
                results.append(MemorySearchResult(
                    content=memory["content"],
                    metadata=memory["metadata"],
                    score=score
                ))
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    async def _initialize(self):
        """Initialize the memory system (mock)"""
        if self.initialized:
            return
        
        # Simulate initialization time
        await asyncio.sleep(0.1)
        self.initialized = True