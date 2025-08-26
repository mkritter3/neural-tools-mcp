#!/usr/bin/env python3
"""
Neural-Powered Dynamic Memory System
Combines neural embeddings (ONNX + ChromaDB) with dynamic relevance scoring
Provides sub-50ms retrieval with semantic similarity + temporal relevance
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import threading

# Add .claude/neural-system to path for neural embeddings
sys.path.insert(0, str(Path(__file__).parent.parent / 'neural-system'))

# Import neural system
from neural_embeddings import HybridEmbeddingSystem, NeuralEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NeuralMemoryChunk:
    """Enhanced memory chunk with neural embedding"""
    id: str
    conversation_id: str
    summary: str
    embedding: np.ndarray
    neural_score: float
    dynamic_score: float
    combined_score: float
    timestamp: float
    token_count: int
    storage_tier: str
    metadata: Dict[str, Any]

class NeuralDynamicMemorySystem:
    """
    Full Neural-Powered Dynamic Memory System
    Combines 384D ONNX embeddings with dynamic relevance scoring
    """
    
    def __init__(self, db_path: str = ".claude/memory/neural-dynamic-memory.db"):
        """Initialize neural-powered dynamic memory system"""
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize neural embeddings system
        logger.info("🧠 Initializing neural embeddings system...")
        self.neural_system = HybridEmbeddingSystem()
        logger.info("✅ Neural system ready: 384D ONNX embeddings")
        
        # Thread-safe database connection pool
        self._local = threading.local()
        self._init_database()
        
        # Performance optimization settings
        self.max_chunks_per_tier = {
            'hot': 100,    # Last 24 hours
            'warm': 500,   # Last 7 days  
            'cold': 2000   # Older than 7 days
        }
        
        # Scoring weights (tunable)
        self.neural_weight = 0.7  # Neural semantic similarity weight
        self.dynamic_weight = 0.3  # Dynamic relevance weight
        
        logger.info("🚀 Neural-Dynamic Memory System initialized")
    
    @property
    def conn(self):
        """Thread-safe database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=10000")
        return self._local.connection
    
    def _init_database(self):
        """Initialize database schema for neural + dynamic memory"""
        
        with self.conn as conn:
            # Neural embeddings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS neural_embeddings (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    embedding_vector BLOB NOT NULL,  -- 384D numpy array
                    created_at REAL NOT NULL,
                    metadata TEXT  -- JSON metadata
                )
            """)
            
            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_neural_conv ON neural_embeddings(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_neural_created ON neural_embeddings(created_at)")
            
            # Dynamic memory metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dynamic_metadata (
                    embedding_id TEXT PRIMARY KEY,
                    storage_tier TEXT NOT NULL,  -- hot/warm/cold
                    last_accessed REAL NOT NULL,
                    usage_count INTEGER DEFAULT 1,
                    context_score REAL DEFAULT 0.0,
                    temporal_decay REAL DEFAULT 1.0,
                    token_count INTEGER DEFAULT 0,
                    conversation_hash TEXT,
                    FOREIGN KEY(embedding_id) REFERENCES neural_embeddings(id)
                )
            """)
            
            # Fast lookup indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tier_access ON dynamic_metadata(storage_tier, last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_hash ON dynamic_metadata(conversation_hash)")
            
        logger.info("✅ Neural-dynamic database schema initialized")
    
    def store_memory(self, conversation_id: str, text: str, metadata: Dict[str, Any] = None) -> str:
        """Store memory with neural embedding and dynamic metadata"""
        
        if len(text.strip()) < 10:  # Skip very short texts
            return None
            
        start_time = time.time()
        
        # Generate neural embedding
        neural_result = self.neural_system.generate_embedding(text, metadata or {})
        
        # Create unique ID
        timestamp = datetime.now().timestamp()
        memory_id = hashlib.sha256(
            f"{conversation_id}:{timestamp}:{text[:100]}".encode()
        ).hexdigest()[:16]
        
        # Store in database
        with self.conn as conn:
            # Neural embeddings table
            conn.execute("""
                INSERT OR REPLACE INTO neural_embeddings 
                (id, conversation_id, summary, embedding_vector, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                conversation_id,
                text[:500] + "..." if len(text) > 500 else text,
                neural_result.embedding.tobytes(),  # Store as binary
                timestamp,
                json.dumps(neural_result.metadata)
            ))
            
            # Dynamic metadata table
            conn.execute("""
                INSERT OR REPLACE INTO dynamic_metadata
                (embedding_id, storage_tier, last_accessed, usage_count, 
                 context_score, temporal_decay, token_count, conversation_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                'hot',  # New memories start in HOT tier
                timestamp,
                1,
                0.8,  # Initial context score
                1.0,  # No decay initially
                len(text.split()),  # Rough token count
                hashlib.sha256(conversation_id.encode()).hexdigest()[:16]
            ))
        
        storage_time = (time.time() - start_time) * 1000
        logger.info(f"💾 Stored memory {memory_id} in {storage_time:.1f}ms")
        
        # Maintain tier limits asynchronously
        threading.Thread(target=self._maintain_storage_tiers, daemon=True).start()
        
        return memory_id
    
    def retrieve_relevant_memories(self, query: str, conversation_id: str = None, 
                                  limit: int = 10) -> List[NeuralMemoryChunk]:
        """
        Retrieve memories using neural similarity + dynamic relevance
        Target: Sub-50ms retrieval
        """
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.neural_system.generate_embedding(query)
        
        # Get all candidate memories (optimized query)
        with self.conn as conn:
            if conversation_id:
                cursor = conn.execute("""
                    SELECT ne.id, ne.conversation_id, ne.summary, ne.embedding_vector,
                           ne.created_at, ne.metadata,
                           dm.storage_tier, dm.last_accessed, dm.usage_count,
                           dm.context_score, dm.temporal_decay, dm.token_count
                    FROM neural_embeddings ne
                    JOIN dynamic_metadata dm ON ne.id = dm.embedding_id
                    WHERE ne.conversation_id = ?
                    ORDER BY dm.last_accessed DESC
                    LIMIT ?
                """, (conversation_id, limit * 3))  # Get more candidates
            else:
                cursor = conn.execute("""
                    SELECT ne.id, ne.conversation_id, ne.summary, ne.embedding_vector,
                           ne.created_at, ne.metadata,
                           dm.storage_tier, dm.last_accessed, dm.usage_count,
                           dm.context_score, dm.temporal_decay, dm.token_count
                    FROM neural_embeddings ne
                    JOIN dynamic_metadata dm ON ne.id = dm.embedding_id
                    ORDER BY dm.last_accessed DESC
                    LIMIT ?
                """, (limit * 3,))
            
            candidates = cursor.fetchall()
        
        # Compute combined scores in parallel
        results = []
        
        for row in candidates:
            (mem_id, conv_id, summary, embedding_bytes, created_at, metadata_json,
             storage_tier, last_accessed, usage_count, context_score, temporal_decay, token_count) = row
            
            # Reconstruct embedding from bytes
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(-1)
            
            # Calculate neural similarity score
            neural_score = self._cosine_similarity(query_embedding.embedding, embedding)
            
            # Calculate dynamic relevance score
            dynamic_score = self._calculate_dynamic_score(
                created_at, last_accessed, usage_count, context_score, 
                temporal_decay, storage_tier
            )
            
            # Combine scores
            combined_score = (self.neural_weight * neural_score + 
                            self.dynamic_weight * dynamic_score)
            
            results.append(NeuralMemoryChunk(
                id=mem_id,
                conversation_id=conv_id,
                summary=summary,
                embedding=embedding,
                neural_score=neural_score,  # NumpyJSONEncoder will handle type conversion
                dynamic_score=dynamic_score,
                combined_score=combined_score,
                timestamp=created_at,
                token_count=token_count,
                storage_tier=storage_tier,
                metadata=json.loads(metadata_json) if metadata_json else {}
            ))
        
        # Sort by combined score and take top results
        results.sort(key=lambda x: x.combined_score, reverse=True)
        final_results = results[:limit]
        
        # Update access patterns asynchronously
        accessed_ids = [r.id for r in final_results]
        threading.Thread(
            target=self._update_access_patterns, 
            args=(accessed_ids,),
            daemon=True
        ).start()
        
        retrieval_time = (time.time() - start_time) * 1000
        logger.info(f"🔍 Retrieved {len(final_results)} memories in {retrieval_time:.1f}ms")
        
        return final_results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fast cosine similarity calculation"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _calculate_dynamic_score(self, created_at: float, last_accessed: float, 
                                usage_count: int, context_score: float,
                                temporal_decay: float, storage_tier: str) -> float:
        """Calculate dynamic relevance score based on multiple factors"""
        
        now = datetime.now().timestamp()
        
        # Recency score (exponential decay)
        hours_since_access = (now - last_accessed) / 3600
        recency_score = np.exp(-hours_since_access / 24)  # Half-life of 24 hours
        
        # Usage frequency score (log scale)
        frequency_score = np.log(1 + usage_count) / 10
        
        # Storage tier bonus
        tier_bonus = {'hot': 1.0, 'warm': 0.7, 'cold': 0.4}[storage_tier]
        
        # Combine factors
        dynamic_score = (
            0.4 * recency_score +
            0.3 * frequency_score + 
            0.2 * context_score +
            0.1 * tier_bonus
        ) * temporal_decay
        
        return min(dynamic_score, 1.0)  # Cap at 1.0
    
    def _update_access_patterns(self, memory_ids: List[str]):
        """Update access patterns for retrieved memories"""
        
        now = datetime.now().timestamp()
        
        with self.conn as conn:
            for mem_id in memory_ids:
                conn.execute("""
                    UPDATE dynamic_metadata 
                    SET last_accessed = ?, usage_count = usage_count + 1
                    WHERE embedding_id = ?
                """, (now, mem_id))
    
    def _maintain_storage_tiers(self):
        """Maintain storage tier limits and perform tier migrations"""
        
        try:
            now = datetime.now().timestamp()
            
            with self.conn as conn:
                # Migrate hot -> warm (after 24 hours)
                conn.execute("""
                    UPDATE dynamic_metadata 
                    SET storage_tier = 'warm'
                    WHERE storage_tier = 'hot' 
                    AND last_accessed < ?
                """, (now - 86400,))  # 24 hours
                
                # Migrate warm -> cold (after 7 days)
                conn.execute("""
                    UPDATE dynamic_metadata 
                    SET storage_tier = 'cold'
                    WHERE storage_tier = 'warm' 
                    AND last_accessed < ?
                """, (now - 604800,))  # 7 days
                
                # Remove oldest cold entries if over limit
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM dynamic_metadata WHERE storage_tier = 'cold'
                """)
                cold_count = cursor.fetchone()[0]
                
                if cold_count > self.max_chunks_per_tier['cold']:
                    # Remove oldest cold memories
                    excess = cold_count - self.max_chunks_per_tier['cold']
                    conn.execute("""
                        DELETE FROM neural_embeddings 
                        WHERE id IN (
                            SELECT dm.embedding_id 
                            FROM dynamic_metadata dm
                            WHERE dm.storage_tier = 'cold'
                            ORDER BY dm.last_accessed ASC
                            LIMIT ?
                        )
                    """, (excess,))
                    
                    conn.execute("""
                        DELETE FROM dynamic_metadata 
                        WHERE storage_tier = 'cold' 
                        AND embedding_id NOT IN (SELECT id FROM neural_embeddings)
                    """)
                
        except Exception as e:
            logger.error(f"Tier maintenance error: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        with self.conn as conn:
            # Count by tiers
            cursor = conn.execute("""
                SELECT storage_tier, COUNT(*), AVG(usage_count), AVG(context_score)
                FROM dynamic_metadata 
                GROUP BY storage_tier
            """)
            tier_stats = {row[0]: {
                'count': row[1], 'avg_usage': row[2], 'avg_context': row[3]
            } for row in cursor.fetchall()}
            
            # Total embeddings
            cursor = conn.execute("SELECT COUNT(*) FROM neural_embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            # Vector store stats
            neural_stats = self.neural_system.vector_store.get_stats()
        
        return {
            'total_memories': total_embeddings,
            'tier_distribution': tier_stats,
            'neural_embeddings': neural_stats,
            'database_path': str(self.db_path),
            'neural_dimensions': 384,
            'scoring_weights': {
                'neural_weight': self.neural_weight,
                'dynamic_weight': self.dynamic_weight
            }
        }
    
    def close(self):
        """Clean close of all connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
        logger.info("🔒 Neural-Dynamic Memory System closed")

# Test function
def test_neural_dynamic_system():
    """Test the neural-powered dynamic memory system"""
    
    print("🧠 Testing Neural-Powered Dynamic Memory System")
    print("=" * 60)
    
    # Initialize system
    system = NeuralDynamicMemorySystem()
    
    # Store some test memories
    test_memories = [
        "Implemented JWT authentication with bcrypt password hashing",
        "Created REST API endpoints for user management and authorization", 
        "Added neural embeddings with ONNX models for semantic search",
        "Built dynamic memory system with storage tiers and relevance scoring",
        "Integrated ChromaDB vector store with 384-dimensional embeddings"
    ]
    
    print("📝 Storing test memories...")
    for i, memory in enumerate(test_memories):
        mem_id = system.store_memory(f"test_conv_{i//2}", memory, {
            'source': 'test',
            'importance': 0.8
        })
        print(f"   Stored: {mem_id}")
    
    # Test retrieval with neural similarity
    print("\n🔍 Testing neural-powered retrieval...")
    query = "authentication and security implementations"
    results = system.retrieve_relevant_memories(query, limit=3)
    
    print(f"Query: '{query}'")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result.summary[:80]}...")
        print(f"     Neural: {result.neural_score:.3f} | Dynamic: {result.dynamic_score:.3f} | Combined: {result.combined_score:.3f}")
        print(f"     Tier: {result.storage_tier} | Usage: {result.metadata}")
    
    # Show system stats
    print("\n📊 System Statistics:")
    stats = system.get_system_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    system.close()
    print("\n✅ Neural-Dynamic Memory System test completed!")

if __name__ == "__main__":
    test_neural_dynamic_system()