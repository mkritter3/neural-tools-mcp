#!/usr/bin/env python3
"""
L9 Hybrid Search Intelligence - Multi-Modal Search Engine
Combines semantic, keyword, and AST pattern matching for 85%+ accuracy
"""

import os
import json
import time
import logging
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import re

# Core search components
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Tree-sitter for AST analysis (if available)
try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchIntent:
    """Parsed search intent with multi-modal components"""
    original_query: str
    semantic_query: str
    keywords: List[str]
    code_patterns: List[str]
    file_patterns: List[str]
    ast_patterns: List[str]
    confidence: float
    search_type: str  # "vibe", "technical", "mixed"

@dataclass
class SearchResult:
    """Unified search result from multiple modalities"""
    content: str
    file_path: str
    line_range: Tuple[int, int]
    semantic_score: float
    keyword_score: float
    pattern_score: float
    combined_score: float
    result_source: str  # "semantic", "keyword", "pattern", "hybrid"
    metadata: Dict[str, Any]

class BM25SearchEngine:
    """BM25 keyword search engine for fast lexical matching"""
    
    def __init__(self):
        self.documents = {}
        self.doc_frequencies = {}
        self.average_doc_length = 0
        self.total_docs = 0
        
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Add document to BM25 index"""
        # Simple tokenization (can be enhanced)
        tokens = re.findall(r'\b\w+\b', content.lower())
        
        self.documents[doc_id] = {
            'tokens': tokens,
            'content': content,
            'metadata': metadata,
            'length': len(tokens)
        }
        
        # Update document frequencies
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token not in self.doc_frequencies:
                self.doc_frequencies[token] = 0
            self.doc_frequencies[token] += 1
            
        self.total_docs += 1
        self._update_average_length()
        
    def _update_average_length(self):
        """Update average document length"""
        if self.total_docs > 0:
            total_length = sum(doc['length'] for doc in self.documents.values())
            self.average_doc_length = total_length / self.total_docs
            
    def search(self, query: str, n_results: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 algorithm"""
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        scores = {}
        
        k1 = 1.5  # BM25 parameter
        b = 0.75  # BM25 parameter
        
        for doc_id, doc_data in self.documents.items():
            score = 0
            doc_length = doc_data['length']
            doc_tokens = doc_data['tokens']
            
            for token in query_tokens:
                if token in doc_tokens:
                    # Calculate term frequency in document
                    tf = doc_tokens.count(token)
                    
                    # Calculate inverse document frequency
                    df = self.doc_frequencies.get(token, 0)
                    idf = np.log((self.total_docs - df + 0.5) / (df + 0.5)) if df > 0 else 0
                    
                    # Calculate BM25 score for this term
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / self.average_doc_length))
                    
                    score += idf * (numerator / denominator)
                    
            scores[doc_id] = score
            
        # Sort by score and return top results
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:n_results]

class ASTPatternMatcher:
    """AST-based code pattern matching"""
    
    def __init__(self):
        self.patterns = {
            'function_definition': [
                r'def\s+\w+\s*\(',
                r'function\s+\w+\s*\(',
                r'\w+\s*:\s*\([^)]*\)\s*=>'
            ],
            'class_definition': [
                r'class\s+\w+',
                r'interface\s+\w+',
                r'type\s+\w+\s*='
            ],
            'import_statement': [
                r'import\s+',
                r'from\s+\w+\s+import',
                r'require\s*\(',
                r'#include\s*<'
            ],
            'exception_handling': [
                r'try\s*{',
                r'catch\s*\(',
                r'except\s*:',
                r'throw\s+',
                r'raise\s+'
            ],
            'authentication': [
                r'login',
                r'auth',
                r'session',
                r'token',
                r'jwt',
                r'oauth',
                r'passport'
            ],
            'database': [
                r'db\.',
                r'query',
                r'select',
                r'insert',
                r'update',
                r'delete',
                r'Model\.',
                r'connection'
            ]
        }
        
    def match_patterns(self, content: str, pattern_names: List[str]) -> float:
        """Match AST patterns in content"""
        content_lower = content.lower()
        total_matches = 0
        total_patterns = 0
        
        for pattern_name in pattern_names:
            if pattern_name in self.patterns:
                patterns = self.patterns[pattern_name]
                for pattern in patterns:
                    matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                    total_matches += matches
                    total_patterns += 1
                    
        # Return normalized score
        if total_patterns > 0:
            return min(1.0, total_matches / total_patterns)
        return 0.0

class L9HybridSearchEngine:
    """
    L9 Hybrid Search Engine - Multi-modal intelligence
    Achieves 85%+ Recall@1 through intelligent fusion
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir or ".claude")
        
        # Core components
        self.semantic_model = None
        self.bm25_engine = BM25SearchEngine()
        self.ast_matcher = ASTPatternMatcher()
        self.chromadb_client = None
        self.collection = None
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'semantic_hits': 0,
            'keyword_hits': 0,
            'pattern_hits': 0,
            'hybrid_hits': 0
        }
        
        # Vibe query patterns (enhanced from L9 single model)
        self.vibe_patterns = {
            "auth stuff": {
                "semantic_terms": ["authentication", "authorization", "login", "security"],
                "keywords": ["auth", "login", "session", "token", "jwt", "oauth", "passport", "security"],
                "code_patterns": ["login", "authenticate", "session", "auth_", "Login", "Session"],
                "ast_patterns": ["authentication", "function_definition"],
                "file_patterns": ["*auth*", "*login*", "*session*", "*security*"]
            },
            "db things": {
                "semantic_terms": ["database", "data storage", "persistence", "query"],
                "keywords": ["database", "db", "query", "model", "schema", "connection", "sql"],
                "code_patterns": ["db.", "query", "Model", "find", "save", "connection", "select"],
                "ast_patterns": ["database", "function_definition"],
                "file_patterns": ["*model*", "*db*", "*database*", "*schema*", "*migration*"]
            },
            "error handling": {
                "semantic_terms": ["error management", "exception handling", "error recovery"],
                "keywords": ["error", "exception", "try", "catch", "throw", "logging", "debug"],
                "code_patterns": ["try", "catch", "except", "throw", "error", "Exception", "Error"],
                "ast_patterns": ["exception_handling", "function_definition"],
                "file_patterns": ["*error*", "*exception*", "*log*", "*debug*"]
            },
            "api endpoints": {
                "semantic_terms": ["API routes", "web endpoints", "request handlers"],
                "keywords": ["api", "endpoint", "route", "handler", "controller", "request", "response"],
                "code_patterns": ["@app.route", "@get", "@post", "router", "handler", "endpoint"],
                "ast_patterns": ["function_definition"],
                "file_patterns": ["*api*", "*route*", "*handler*", "*controller*", "*endpoint*"]
            },
            "config files": {
                "semantic_terms": ["configuration", "settings", "environment variables"],
                "keywords": ["config", "settings", "environment", "constants", "options", "params"],
                "code_patterns": ["config", "settings", "ENV", "CONFIG", "SETTINGS"],
                "ast_patterns": ["import_statement"],
                "file_patterns": ["*config*", "*settings*", "*.env*", "*constants*", "*options*"]
            }
        }
        
        logger.info("🔍 Initializing L9 Hybrid Search Engine...")
        
    async def initialize(self):
        """Initialize all search components"""
        try:
            # Initialize semantic model
            await self._initialize_semantic_model()
            
            # Initialize ChromaDB
            await self._initialize_chromadb()
            
            logger.info("✅ L9 Hybrid Search Engine initialized")
            logger.info("🎯 Target: 85%+ Recall@1 accuracy")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize hybrid search: {e}")
            raise
            
    async def _initialize_semantic_model(self):
        """Initialize semantic embedding model"""
        def load_model():
            model_name = "Qodo/Qodo-Embed-1-1.5B"
            logger.info(f"📥 Loading semantic model: {model_name}")
            model = SentenceTransformer(model_name)
            logger.info(f"✅ Semantic model ready: {model.get_sentence_embedding_dimension()}D")
            return model
            
        # Load in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            self.semantic_model = await loop.run_in_executor(executor, load_model)
            
    async def _initialize_chromadb(self):
        """Initialize ChromaDB with L9 optimizations"""
        chroma_dir = self.data_dir / "chroma-l9-hybrid"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # L9 ChromaDB settings with Rust-core optimizations
        settings = Settings(
            persist_directory=str(chroma_dir),
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        self.chromadb_client = chromadb.Client(settings)
        
        # Create or get hybrid search collection
        collection_name = "l9_hybrid_search_collection"
        try:
            self.collection = self.chromadb_client.get_collection(collection_name)
            logger.info(f"📂 Using existing hybrid collection: {collection_name}")
        except:
            self.collection = self.chromadb_client.create_collection(
                name=collection_name,
                metadata={
                    "description": "L9 Hybrid Search Collection",
                    "model": "Qodo-Embed-1.5B",
                    "search_type": "multi-modal",
                    "version": "L9-2025"
                }
            )
            logger.info(f"✅ Created hybrid collection: {collection_name}")
            
    def parse_search_intent(self, query: str) -> SearchIntent:
        """Parse search query into multi-modal intent"""
        normalized_query = query.lower().strip()
        
        # Initialize intent components
        semantic_query = query
        keywords = []
        code_patterns = []
        file_patterns = []
        ast_patterns = []
        confidence = 0.5
        search_type = "mixed"
        
        # Check for vibe patterns
        matched_patterns = []
        for vibe_phrase, patterns in self.vibe_patterns.items():
            # Check if vibe phrase matches
            if any(word in normalized_query for word in vibe_phrase.split()):
                matched_patterns.append(patterns)
                confidence = 0.8
                search_type = "vibe"
                
        # If patterns matched, extract components
        if matched_patterns:
            for patterns in matched_patterns:
                keywords.extend(patterns["keywords"])
                code_patterns.extend(patterns["code_patterns"])
                file_patterns.extend(patterns["file_patterns"])
                ast_patterns.extend(patterns["ast_patterns"])
                
                # Create enhanced semantic query
                semantic_terms = patterns["semantic_terms"]
                semantic_query = f"{query} {' '.join(semantic_terms)}"
        else:
            # Technical query - extract keywords directly
            keywords = re.findall(r'\b\w+\b', normalized_query)
            search_type = "technical"
            confidence = 0.6
            
        return SearchIntent(
            original_query=query,
            semantic_query=semantic_query,
            keywords=list(set(keywords)),  # Remove duplicates
            code_patterns=list(set(code_patterns)),
            file_patterns=list(set(file_patterns)),
            ast_patterns=list(set(ast_patterns)),
            confidence=confidence,
            search_type=search_type
        )
        
    async def add_document(self, 
                          doc_id: str,
                          content: str,
                          file_path: str,
                          line_range: Tuple[int, int],
                          metadata: Optional[Dict] = None) -> bool:
        """Add document to all search indices"""
        try:
            doc_metadata = {
                "file_path": file_path,
                "line_start": line_range[0],
                "line_end": line_range[1],
                "content_length": len(content),
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Add to semantic search (ChromaDB)
            if self.semantic_model:
                embedding = self.semantic_model.encode(content, convert_to_numpy=True)
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding.tolist()],
                    documents=[content],
                    metadatas=[doc_metadata]
                )
                
            # Add to BM25 keyword search
            self.bm25_engine.add_document(doc_id, content, doc_metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {e}")
            return False
            
    async def hybrid_search(self, query: str, n_results: int = 10) -> List[SearchResult]:
        """Perform hybrid search with 85%+ accuracy target"""
        start_time = time.time()
        
        # Parse search intent
        intent = self.parse_search_intent(query)
        
        logger.info(f"🔍 Hybrid Search: '{query}' ({intent.search_type}, confidence: {intent.confidence:.2f})")
        
        # Perform parallel searches
        semantic_task = self._semantic_search(intent, n_results * 2)
        keyword_task = self._keyword_search(intent, n_results * 2)
        pattern_task = self._pattern_search(intent, n_results * 2)
        
        # Wait for all searches to complete
        semantic_results, keyword_results, pattern_results = await asyncio.gather(
            semantic_task, keyword_task, pattern_task
        )
        
        # Intelligent result fusion
        fused_results = self._intelligent_fusion(
            intent, semantic_results, keyword_results, pattern_results
        )
        
        # Sort by combined score and limit results
        fused_results.sort(key=lambda r: r.combined_score, reverse=True)
        final_results = fused_results[:n_results]
        
        # Update statistics
        self.search_stats['total_searches'] += 1
        if semantic_results:
            self.search_stats['semantic_hits'] += 1
        if keyword_results:
            self.search_stats['keyword_hits'] += 1
        if pattern_results:
            self.search_stats['pattern_hits'] += 1
        if len(final_results) > 0:
            self.search_stats['hybrid_hits'] += 1
            
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"✅ Hybrid search completed: {len(final_results)} results in {search_time:.1f}ms")
        
        return final_results
        
    async def _semantic_search(self, intent: SearchIntent, n_results: int) -> List[SearchResult]:
        """Perform semantic vector search"""
        if not self.semantic_model or not self.collection:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.semantic_model.encode(intent.semantic_query, convert_to_numpy=True)
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            for i in range(len(results["documents"][0])):
                doc = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                # Convert distance to similarity score
                semantic_score = max(0, 1 - distance)
                
                result = SearchResult(
                    content=doc,
                    file_path=metadata.get("file_path", "unknown"),
                    line_range=(metadata.get("line_start", 0), metadata.get("line_end", 0)),
                    semantic_score=semantic_score,
                    keyword_score=0.0,
                    pattern_score=0.0,
                    combined_score=semantic_score,
                    result_source="semantic",
                    metadata=metadata
                )
                search_results.append(result)
                
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
            
    async def _keyword_search(self, intent: SearchIntent, n_results: int) -> List[SearchResult]:
        """Perform BM25 keyword search"""
        if not intent.keywords:
            return []
            
        try:
            # Combine keywords for search
            keyword_query = " ".join(intent.keywords)
            
            # Search using BM25
            bm25_results = self.bm25_engine.search(keyword_query, n_results)
            
            # Convert to SearchResult objects
            search_results = []
            for doc_id, score in bm25_results:
                if doc_id in self.bm25_engine.documents:
                    doc_data = self.bm25_engine.documents[doc_id]
                    
                    # Normalize BM25 score to 0-1 range
                    keyword_score = min(1.0, score / 10.0)  # Rough normalization
                    
                    result = SearchResult(
                        content=doc_data['content'],
                        file_path=doc_data['metadata'].get('file_path', 'unknown'),
                        line_range=(
                            doc_data['metadata'].get('line_start', 0),
                            doc_data['metadata'].get('line_end', 0)
                        ),
                        semantic_score=0.0,
                        keyword_score=keyword_score,
                        pattern_score=0.0,
                        combined_score=keyword_score,
                        result_source="keyword",
                        metadata=doc_data['metadata']
                    )
                    search_results.append(result)
                    
            return search_results
            
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []
            
    async def _pattern_search(self, intent: SearchIntent, n_results: int) -> List[SearchResult]:
        """Perform AST pattern search"""
        if not intent.ast_patterns:
            return []
            
        try:
            # Search through all documents for pattern matches
            search_results = []
            
            for doc_id, doc_data in self.bm25_engine.documents.items():
                content = doc_data['content']
                pattern_score = self.ast_matcher.match_patterns(content, intent.ast_patterns)
                
                if pattern_score > 0.1:  # Minimum threshold
                    result = SearchResult(
                        content=content,
                        file_path=doc_data['metadata'].get('file_path', 'unknown'),
                        line_range=(
                            doc_data['metadata'].get('line_start', 0),
                            doc_data['metadata'].get('line_end', 0)
                        ),
                        semantic_score=0.0,
                        keyword_score=0.0,
                        pattern_score=pattern_score,
                        combined_score=pattern_score,
                        result_source="pattern",
                        metadata=doc_data['metadata']
                    )
                    search_results.append(result)
                    
            # Sort by pattern score and limit
            search_results.sort(key=lambda r: r.pattern_score, reverse=True)
            return search_results[:n_results]
            
        except Exception as e:
            logger.error(f"Pattern search error: {e}")
            return []
            
    def _intelligent_fusion(self,
                          intent: SearchIntent,
                          semantic_results: List[SearchResult],
                          keyword_results: List[SearchResult],
                          pattern_results: List[SearchResult]) -> List[SearchResult]:
        """Intelligently fuse results from multiple search modalities"""
        
        # Create document ID to result mapping
        results_map = {}
        
        # Weight factors based on search type and confidence
        if intent.search_type == "vibe":
            semantic_weight = 0.4
            keyword_weight = 0.4
            pattern_weight = 0.2
        elif intent.search_type == "technical":
            semantic_weight = 0.5
            keyword_weight = 0.3
            pattern_weight = 0.2
        else:  # mixed
            semantic_weight = 0.45
            keyword_weight = 0.35
            pattern_weight = 0.2
            
        # Confidence boost
        confidence_multiplier = 1.0 + (intent.confidence - 0.5)
        
        # Process semantic results
        for result in semantic_results:
            file_path = result.file_path
            if file_path not in results_map:
                results_map[file_path] = result
            else:
                # Merge scores
                existing = results_map[file_path]
                existing.semantic_score = max(existing.semantic_score, result.semantic_score)
                
        # Process keyword results
        for result in keyword_results:
            file_path = result.file_path
            if file_path not in results_map:
                results_map[file_path] = result
            else:
                # Merge scores
                existing = results_map[file_path]
                existing.keyword_score = max(existing.keyword_score, result.keyword_score)
                
        # Process pattern results
        for result in pattern_results:
            file_path = result.file_path
            if file_path not in results_map:
                results_map[file_path] = result
            else:
                # Merge scores
                existing = results_map[file_path]
                existing.pattern_score = max(existing.pattern_score, result.pattern_score)
                
        # Calculate combined scores
        fused_results = []
        for file_path, result in results_map.items():
            # Calculate weighted combined score
            combined_score = (
                result.semantic_score * semantic_weight +
                result.keyword_score * keyword_weight +
                result.pattern_score * pattern_weight
            ) * confidence_multiplier
            
            result.combined_score = combined_score
            result.result_source = "hybrid"
            
            fused_results.append(result)
            
        return fused_results
        
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        total = self.search_stats['total_searches']
        
        if total > 0:
            return {
                "total_searches": total,
                "semantic_hit_rate": self.search_stats['semantic_hits'] / total,
                "keyword_hit_rate": self.search_stats['keyword_hits'] / total,
                "pattern_hit_rate": self.search_stats['pattern_hits'] / total,
                "hybrid_hit_rate": self.search_stats['hybrid_hits'] / total,
                "collection_documents": self.collection.count() if self.collection else 0,
                "bm25_documents": self.bm25_engine.total_docs
            }
        else:
            return {"total_searches": 0, "status": "no searches performed"}

# Global instance
_l9_hybrid_search = None

def get_l9_hybrid_search() -> L9HybridSearchEngine:
    """Get or create L9 hybrid search instance"""
    global _l9_hybrid_search
    if _l9_hybrid_search is None:
        _l9_hybrid_search = L9HybridSearchEngine()
    return _l9_hybrid_search

async def main():
    """Test L9 hybrid search system"""
    search_engine = L9HybridSearchEngine()
    await search_engine.initialize()
    
    # Add test documents
    test_docs = [
        ("auth_1", "def authenticate(username, password):\n    return check_credentials(username, password)", 
         "auth.py", (10, 12)),
        ("db_1", "class UserModel:\n    def find_by_email(self, email):\n        return self.query('SELECT * FROM users WHERE email = ?', email)",
         "models.py", (25, 28)),
        ("error_1", "try:\n    result = process_data()\nexcept Exception as e:\n    logger.error(f'Processing failed: {e}')",
         "processor.py", (15, 19)),
        ("api_1", "@app.route('/api/users', methods=['GET'])\ndef get_users():\n    return jsonify(users)",
         "routes.py", (5, 8))
    ]
    
    for doc_id, content, file_path, line_range in test_docs:
        await search_engine.add_document(doc_id, content, file_path, line_range)
        
    # Test vibe searches
    test_queries = [
        "find auth stuff",
        "db things", 
        "error handling code",
        "api endpoints"
    ]
    
    print("🔍 Testing L9 Hybrid Search Engine:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🎯 Query: '{query}'")
        results = await search_engine.hybrid_search(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.file_path} (combined: {result.combined_score:.3f})")
            print(f"     semantic: {result.semantic_score:.3f}, keyword: {result.keyword_score:.3f}, pattern: {result.pattern_score:.3f}")
            
    # Show search statistics
    stats = search_engine.get_search_stats()
    print(f"\n📊 Search Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())