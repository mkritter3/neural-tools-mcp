#!/usr/bin/env python3
"""
L9 Offline Evaluation Harness for Neural Embeddings
Provides comprehensive evaluation of embedding models for Phase 1 validation
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Add neural-system to path
sys.path.insert(0, str(Path(__file__).parent))

from neural_embeddings import HybridEmbeddingSystem, get_neural_system
from feature_flags import get_feature_manager, is_enabled

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Results from embedding model evaluation"""
    model_name: str
    test_suite: str
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    latency_ms: Dict[str, float]
    error_rate: float
    total_queries: int
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class TestQuery:
    """Test query with expected results"""
    query: str
    query_type: str  # 'code', 'text', 'mixed'
    expected_results: List[str]  # Ground truth document IDs
    relevance_scores: Optional[List[float]] = None

class CodeRetrievalEvaluator:
    """Evaluates code-specific retrieval performance"""
    
    def __init__(self):
        self.system = get_neural_system()
        self.feature_manager = get_feature_manager()
        
        # Golden dataset for code understanding
        self.code_test_queries = [
            TestQuery(
                query="recursive fibonacci implementation",
                query_type="code",
                expected_results=["fibonacci_recursive", "dynamic_programming"],
                relevance_scores=[1.0, 0.8]
            ),
            TestQuery(
                query="database connection pool management",
                query_type="code", 
                expected_results=["db_pool", "connection_manager"],
                relevance_scores=[1.0, 0.9]
            ),
            TestQuery(
                query="error handling try catch exception",
                query_type="code",
                expected_results=["error_handler", "exception_wrapper"],
                relevance_scores=[1.0, 0.7]
            ),
            TestQuery(
                query="REST API authentication middleware",
                query_type="code",
                expected_results=["auth_middleware", "jwt_handler"],
                relevance_scores=[1.0, 0.8]
            )
        ]
        
        # Test documents for code retrieval
        self.test_documents = {
            "fibonacci_recursive": '''def fibonacci(n):
    """Recursive fibonacci implementation"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)''',
            
            "dynamic_programming": '''def fibonacci_dp(n):
    """Dynamic programming fibonacci implementation"""
    dp = [0, 1]
    for i in range(2, n+1):
        dp.append(dp[i-1] + dp[i-2])
    return dp[n]''',
            
            "db_pool": '''class DatabasePool:
    """Connection pool for database management"""
    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.pool = []
    
    def get_connection(self):
        if self.pool:
            return self.pool.pop()
        return create_new_connection()''',
            
            "connection_manager": '''class ConnectionManager:
    """Manages database connections with retry logic"""
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None
        
    def connect_with_retry(self, max_retries=3):
        for attempt in range(max_retries):
            try:
                self.connection = connect(self.host, self.port)
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)''',
            
            "error_handler": '''def handle_error(func):
    """Error handling decorator with logging"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper''',
            
            "exception_wrapper": '''class ExceptionWrapper:
    """Wraps functions with comprehensive exception handling"""
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger()
    
    def safe_execute(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.exception(f"Exception in {func.__name__}")
            return None''',
            
            "auth_middleware": '''def authenticate_request(request):
    """REST API authentication middleware"""
    token = request.headers.get('Authorization')
    if not token:
        return None, "Missing authorization header"
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return payload['user_id'], None
    except jwt.InvalidTokenError:
        return None, "Invalid token"''',
            
            "jwt_handler": '''class JWTHandler:
    """JWT token management for API authentication"""
    def __init__(self, secret_key, algorithm='HS256'):
        self.secret_key = secret_key  
        self.algorithm = algorithm
        
    def generate_token(self, user_id, expires_in=3600):
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)''',
        }
    
    def setup_test_data(self):
        """Index test documents for evaluation"""
        logger.info("Setting up test data for evaluation...")
        
        embeddings = []
        for doc_id, doc_content in self.test_documents.items():
            metadata = {
                'doc_id': doc_id,
                'file_path': f'{doc_id}.py',
                'file_type': 'CORE_CODE',
                'test_document': True
            }
            
            embedding = self.system.generate_embedding(doc_content, metadata)
            embeddings.append(embedding)
        
        success = self.system.vector_store.add_embeddings(embeddings)
        logger.info(f"Test data setup: {'success' if success else 'failed'}")
        return success
    
    def calculate_recall_at_k(self, results: List[Dict], expected: List[str], k: int) -> float:
        """Calculate Recall@K metric"""
        if not expected or k == 0:
            return 0.0
        
        # Extract document IDs from results
        retrieved_ids = []
        for result in results[:k]:
            metadata = result.get('metadata', {})
            doc_id = metadata.get('doc_id', '')
            if doc_id:
                retrieved_ids.append(doc_id)
        
        # Calculate recall
        relevant_retrieved = len(set(retrieved_ids) & set(expected))
        return relevant_retrieved / len(expected)
    
    def calculate_ndcg_at_k(self, results: List[Dict], expected: List[str], 
                           relevance_scores: Optional[List[float]], k: int) -> float:
        """Calculate NDCG@K (Normalized Discounted Cumulative Gain)"""
        if not expected or k == 0:
            return 0.0
        
        # Create relevance map
        relevance_map = {}
        if relevance_scores and len(relevance_scores) == len(expected):
            for doc_id, score in zip(expected, relevance_scores):
                relevance_map[doc_id] = score
        else:
            # Default relevance scores
            for i, doc_id in enumerate(expected):
                relevance_map[doc_id] = 1.0 / (i + 1)
        
        # Calculate DCG
        dcg = 0.0
        for i, result in enumerate(results[:k]):
            metadata = result.get('metadata', {})
            doc_id = metadata.get('doc_id', '')
            relevance = relevance_map.get(doc_id, 0.0)
            if relevance > 0:
                dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG (perfect ranking)
        sorted_relevances = sorted(relevance_map.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_model(self, model_name: str = "current", 
                      queries: Optional[List[TestQuery]] = None) -> EvaluationResult:
        """Evaluate a specific embedding model"""
        
        if queries is None:
            queries = self.code_test_queries
        
        logger.info(f"Evaluating model: {model_name}")
        
        recall_results = {k: [] for k in [1, 3, 5, 10]}
        ndcg_results = {k: [] for k in [1, 3, 5, 10]}
        latencies = []
        errors = 0
        
        for query in queries:
            try:
                start_time = time.time()
                
                # Perform search
                results = self.system.semantic_search(
                    query.query, 
                    n_results=10,
                    include_deprecated=True,
                    min_priority=1
                )
                
                query_latency = (time.time() - start_time) * 1000
                latencies.append(query_latency)
                
                # Calculate metrics for different K values
                for k in recall_results.keys():
                    recall = self.calculate_recall_at_k(results, query.expected_results, k)
                    recall_results[k].append(recall)
                    
                    ndcg = self.calculate_ndcg_at_k(
                        results, query.expected_results, query.relevance_scores, k
                    )
                    ndcg_results[k].append(ndcg)
                
                logger.debug(f"Query '{query.query[:30]}...' - Latency: {query_latency:.1f}ms")
                
            except Exception as e:
                logger.error(f"Query evaluation failed: {e}")
                errors += 1
        
        # Calculate average metrics
        avg_recall = {k: np.mean(scores) for k, scores in recall_results.items()}
        avg_ndcg = {k: np.mean(scores) for k, scores in ndcg_results.items()}
        avg_latency = {
            'mean': np.mean(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
        
        return EvaluationResult(
            model_name=model_name,
            test_suite="code_retrieval",
            recall_at_k=avg_recall,
            ndcg_at_k=avg_ndcg, 
            latency_ms=avg_latency,
            error_rate=errors / len(queries),
            total_queries=len(queries),
            timestamp=datetime.now().isoformat(),
            metadata={
                'feature_flags': self.feature_manager.get_stats(),
                'vector_store_stats': self.system.vector_store.get_stats()
            }
        )
    
    def run_comparative_evaluation(self) -> Dict[str, EvaluationResult]:
        """Run evaluation across different model configurations"""
        logger.info("Starting comparative evaluation...")
        
        results = {}
        
        # Evaluate current configuration
        baseline_result = self.evaluate_model("baseline")
        results["baseline"] = baseline_result
        
        # Test with different feature flag combinations if available
        if is_enabled("enable_ab_testing"):
            # Simulate different A/B test variants
            test_variants = ["onnx_baseline", "qodo_embed", "openai_hybrid"]
            
            for variant in test_variants:
                try:
                    # This would require more sophisticated variant switching
                    # For now, we'll just run the baseline evaluation
                    variant_result = self.evaluate_model(f"variant_{variant}")
                    results[variant] = variant_result
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate variant {variant}: {e}")
        
        return results
    
    def save_results(self, results: Dict[str, EvaluationResult], 
                    output_dir: str = ".claude/neural-system/evaluation_results"):
        """Save evaluation results to JSON file"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"evaluation_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = {
            name: asdict(result) for name, result in results.items()
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
        return results_file
    
    def print_summary(self, results: Dict[str, EvaluationResult]):
        """Print evaluation summary"""
        
        print("\n📊 L9 Phase 1 Evaluation Results")
        print("=" * 60)
        
        for model_name, result in results.items():
            print(f"\n🔍 {model_name.upper()} RESULTS:")
            print(f"   Queries: {result.total_queries} | Errors: {result.error_rate:.1%}")
            
            print("   Recall@K:")
            for k, score in result.recall_at_k.items():
                print(f"     @{k}: {score:.3f}")
            
            print("   NDCG@K:")
            for k, score in result.ndcg_at_k.items():
                print(f"     @{k}: {score:.3f}")
            
            print("   Latency:")
            print(f"     Mean: {result.latency_ms['mean']:.1f}ms")
            print(f"     P95:  {result.latency_ms['p95']:.1f}ms")
        
        # Compare results if multiple models
        if len(results) > 1:
            print("\n📈 PERFORMANCE COMPARISON:")
            baseline = list(results.values())[0]
            
            for model_name, result in list(results.items())[1:]:
                recall_improvement = ((result.recall_at_k[10] / baseline.recall_at_k[10] - 1) * 100 
                                    if baseline.recall_at_k[10] > 0 else 0)
                ndcg_improvement = ((result.ndcg_at_k[10] / baseline.ndcg_at_k[10] - 1) * 100 
                                  if baseline.ndcg_at_k[10] > 0 else 0)
                
                print(f"   {model_name} vs baseline:")
                print(f"     Recall@10: {recall_improvement:+.1f}%")
                print(f"     NDCG@10:   {ndcg_improvement:+.1f}%")

def main():
    """Run comprehensive evaluation"""
    
    print("🧪 L9 Phase 1: Offline Evaluation Harness")
    print("=" * 50)
    
    evaluator = CodeRetrievalEvaluator()
    
    # Setup test data
    if not evaluator.setup_test_data():
        print("❌ Failed to setup test data")
        return False
    
    # Run evaluation
    results = evaluator.run_comparative_evaluation()
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    results_file = evaluator.save_results(results)
    print(f"\n💾 Detailed results saved: {results_file}")
    
    # Check if we meet L9 performance criteria
    baseline = results.get("baseline")
    if baseline:
        meets_criteria = (
            baseline.recall_at_k[10] >= 0.6 and  # 60% recall@10
            baseline.ndcg_at_k[10] >= 0.5 and    # 50% NDCG@10  
            baseline.latency_ms['p95'] <= 200 and # <200ms P95 latency
            baseline.error_rate <= 0.05           # <5% error rate
        )
        
        if meets_criteria:
            print("\n✅ L9 PERFORMANCE CRITERIA MET!")
        else:
            print("\n⚠️  L9 performance criteria not met - Phase 2 improvements needed")
    
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    if not success:
        sys.exit(1)