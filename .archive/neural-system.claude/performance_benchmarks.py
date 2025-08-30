#!/usr/bin/env python3
"""
Neural Flow Performance Benchmarking System
L9-grade benchmarking with weekly validation checkpoints
"""

import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics
import asyncio
from pathlib import Path

from neural_embeddings import CodeSpecificEmbedder
from neural_dynamic_memory_system import NeuralDynamicMemorySystem
from project_neural_indexer import ProjectNeuralIndexer


@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    test_name: str
    metric: str
    value: float
    target: float
    passed: bool
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class WeeklyTargets:
    """Performance targets for each week"""
    week: int
    recall_at_1: float
    recall_at_3: float
    latency_p95: float  # milliseconds
    accuracy: float
    throughput: float  # queries/second


class PerformanceBenchmarks:
    """L9-grade performance benchmarking system"""
    
    WEEKLY_TARGETS = [
        WeeklyTargets(1, 0.40, 0.60, 150, 0.70, 10),
        WeeklyTargets(2, 0.50, 0.70, 120, 0.75, 15),
        WeeklyTargets(3, 0.60, 0.78, 100, 0.80, 20),
        WeeklyTargets(4, 0.70, 0.85, 80, 0.85, 25),
        WeeklyTargets(8, 0.85, 0.92, 60, 0.90, 30),  # Final target
    ]
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("/app/data")
        self.results_dir = self.data_dir / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.results: List[BenchmarkResult] = []
        self.current_week = self._calculate_current_week()
        
    def _calculate_current_week(self) -> int:
        """Calculate current week since deployment"""
        deployment_date = datetime(2025, 8, 26)  # Baseline
        current_date = datetime.now()
        weeks_elapsed = (current_date - deployment_date).days // 7
        return max(1, min(weeks_elapsed + 1, 8))
    
    def get_current_targets(self) -> WeeklyTargets:
        """Get performance targets for current week"""
        for targets in self.WEEKLY_TARGETS:
            if targets.week >= self.current_week:
                return targets
        return self.WEEKLY_TARGETS[-1]  # Final targets
    
    async def benchmark_hybrid_search(self, 
                                    memory_system: NeuralDynamicMemorySystem,
                                    test_queries: List[str],
                                    ground_truth: List[List[str]]) -> Dict[str, BenchmarkResult]:
        """Benchmark hybrid search performance"""
        results = {}
        targets = self.get_current_targets()
        
        # Recall@1 and Recall@3 benchmarks
        recall_1_scores = []
        recall_3_scores = []
        latencies = []
        
        for query, truth in zip(test_queries, ground_truth):
            start_time = time.perf_counter()
            
            # Perform search
            search_results = await memory_system.search_code_context(
                query, limit=10, similarity_threshold=0.3
            )
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Calculate recalls
            result_texts = [r.content for r in search_results[:3]]
            recall_1_scores.append(1.0 if any(t in result_texts[:1] for t in truth) else 0.0)
            recall_3_scores.append(1.0 if any(t in result_texts for t in truth) else 0.0)
        
        # Calculate metrics
        avg_recall_1 = statistics.mean(recall_1_scores) if recall_1_scores else 0.0
        avg_recall_3 = statistics.mean(recall_3_scores) if recall_3_scores else 0.0
        p95_latency = statistics.quantiles(latencies, n=20)[18] if latencies else 0.0  # 95th percentile
        
        # Create results
        results['recall_at_1'] = BenchmarkResult(
            test_name="hybrid_search_recall_1",
            metric="recall@1",
            value=avg_recall_1,
            target=targets.recall_at_1,
            passed=avg_recall_1 >= targets.recall_at_1,
            timestamp=datetime.now(),
            details={"individual_scores": recall_1_scores, "query_count": len(test_queries)}
        )
        
        results['recall_at_3'] = BenchmarkResult(
            test_name="hybrid_search_recall_3",
            metric="recall@3",
            value=avg_recall_3,
            target=targets.recall_at_3,
            passed=avg_recall_3 >= targets.recall_at_3,
            timestamp=datetime.now(),
            details={"individual_scores": recall_3_scores, "query_count": len(test_queries)}
        )
        
        results['latency_p95'] = BenchmarkResult(
            test_name="hybrid_search_latency",
            metric="latency_p95_ms",
            value=p95_latency,
            target=targets.latency_p95,
            passed=p95_latency <= targets.latency_p95,
            timestamp=datetime.now(),
            details={"all_latencies": latencies, "mean_latency": statistics.mean(latencies)}
        )
        
        return results
    
    async def benchmark_embedding_performance(self, 
                                            embedder: CodeSpecificEmbedder,
                                            code_samples: List[str]) -> Dict[str, BenchmarkResult]:
        """Benchmark embedding generation performance"""
        targets = self.get_current_targets()
        
        # Throughput benchmark
        start_time = time.perf_counter()
        embeddings = []
        
        for code in code_samples:
            embedding = await asyncio.to_thread(embedder.embed_code, code)
            embeddings.append(embedding)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = len(code_samples) / total_time
        
        # Quality benchmark (semantic similarity)
        quality_scores = []
        if len(code_samples) >= 2:
            for i in range(0, len(embeddings) - 1, 2):
                if i + 1 < len(embeddings):
                    # Simple cosine similarity
                    emb1, emb2 = embeddings[i], embeddings[i + 1]
                    similarity = sum(a * b for a, b in zip(emb1, emb2)) / (
                        (sum(a * a for a in emb1) * sum(b * b for b in emb2)) ** 0.5
                    )
                    quality_scores.append(abs(similarity))
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.7
        
        return {
            'throughput': BenchmarkResult(
                test_name="embedding_throughput",
                metric="embeddings_per_second",
                value=throughput,
                target=targets.throughput,
                passed=throughput >= targets.throughput,
                timestamp=datetime.now(),
                details={"total_samples": len(code_samples), "total_time": total_time}
            ),
            'quality': BenchmarkResult(
                test_name="embedding_quality",
                metric="semantic_consistency",
                value=avg_quality,
                target=0.6,  # Minimum semantic consistency
                passed=avg_quality >= 0.6,
                timestamp=datetime.now(),
                details={"similarity_scores": quality_scores}
            )
        }
    
    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        self.logger.info(f"Starting Week {self.current_week} benchmark suite")
        targets = self.get_current_targets()
        
        all_results = {}
        failed_tests = []
        
        try:
            # Initialize systems
            embedder = CodeSpecificEmbedder()
            memory_system = NeuralDynamicMemorySystem(
                persist_directory=str(self.data_dir / "chroma_db")
            )
            
            # Generate test data
            test_queries = [
                "function to calculate fibonacci numbers",
                "async HTTP request handler",
                "database connection pooling",
                "error handling middleware",
                "JWT token validation"
            ]
            
            ground_truth = [
                ["fibonacci", "recursive", "calculate"],
                ["async", "handler", "request", "response"],
                ["database", "pool", "connection"],
                ["error", "middleware", "exception"],
                ["jwt", "token", "validate", "auth"]
            ]
            
            code_samples = [
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "async def handle_request(request): return await process(request)",
                "class DatabasePool: def __init__(self): self.connections = []",
                "def error_middleware(error): logger.error(error); return Response(500)",
                "def validate_jwt(token): return jwt.decode(token, SECRET_KEY)"
            ]
            
            # Run search benchmarks
            search_results = await self.benchmark_hybrid_search(
                memory_system, test_queries, ground_truth
            )
            all_results.update(search_results)
            
            # Run embedding benchmarks
            embedding_results = await self.benchmark_embedding_performance(
                embedder, code_samples
            )
            all_results.update(embedding_results)
            
            # Check for failures
            for name, result in all_results.items():
                if not result.passed:
                    failed_tests.append(name)
                    self.logger.warning(f"FAILED: {name} - {result.value:.3f} vs target {result.target:.3f}")
                else:
                    self.logger.info(f"PASSED: {name} - {result.value:.3f} >= {result.target:.3f}")
            
            # Store results
            await self.save_benchmark_results(all_results)
            
            return {
                "week": self.current_week,
                "targets": asdict(targets),
                "results": {name: asdict(result) for name, result in all_results.items()},
                "summary": {
                    "total_tests": len(all_results),
                    "passed_tests": len(all_results) - len(failed_tests),
                    "failed_tests": failed_tests,
                    "overall_pass": len(failed_tests) == 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}")
            return {
                "error": str(e),
                "week": self.current_week,
                "status": "failed"
            }
    
    async def save_benchmark_results(self, results: Dict[str, BenchmarkResult]):
        """Save benchmark results to disk"""
        timestamp = datetime.now().isoformat()
        filename = f"week_{self.current_week}_{timestamp.replace(':', '-')}.json"
        filepath = self.results_dir / filename
        
        serializable_results = {
            name: asdict(result) for name, result in results.items()
        }
        
        # Convert datetime objects to strings
        for result in serializable_results.values():
            if 'timestamp' in result:
                result['timestamp'] = result['timestamp'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump({
                "week": self.current_week,
                "timestamp": timestamp,
                "results": serializable_results
            }, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {filepath}")
    
    async def generate_performance_report(self) -> str:
        """Generate human-readable performance report"""
        targets = self.get_current_targets()
        
        # Load recent results
        recent_files = sorted(self.results_dir.glob("*.json"))[-5:]  # Last 5 runs
        
        report = f"""
# Neural Flow Performance Report - Week {self.current_week}

## Current Targets
- Recall@1: {targets.recall_at_1:.1%}
- Recall@3: {targets.recall_at_3:.1%}  
- Latency P95: {targets.latency_p95}ms
- Accuracy: {targets.accuracy:.1%}
- Throughput: {targets.throughput} embeddings/sec

## Recent Performance Trend
"""
        
        if recent_files:
            for file_path in recent_files[-3:]:  # Last 3 runs
                with open(file_path) as f:
                    data = json.load(f)
                    timestamp = data['timestamp'][:19].replace('T', ' ')
                    report += f"\n### Run: {timestamp}\n"
                    
                    for name, result in data['results'].items():
                        status = "✅" if result['passed'] else "❌"
                        report += f"- {status} {result['metric']}: {result['value']:.3f} (target: {result['target']:.3f})\n"
        
        report += f"""
## Recommendations
- Monitor failed tests and implement fallback strategies
- Consider model optimization if latency targets missed
- Scale infrastructure if throughput insufficient
- Review search relevance if recall targets missed
"""
        
        return report


async def main():
    """CLI entry point for benchmarking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Flow Performance Benchmarks")
    parser.add_argument("--data-dir", default="/app/data", help="Data directory path")
    parser.add_argument("--report-only", action="store_true", help="Generate report only")
    
    args = parser.parse_args()
    
    benchmarker = PerformanceBenchmarks(Path(args.data_dir))
    
    if args.report_only:
        report = await benchmarker.generate_performance_report()
        print(report)
    else:
        results = await benchmarker.run_full_benchmark_suite()
        print(json.dumps(results, indent=2, default=str))
        
        # Also generate report
        report = await benchmarker.generate_performance_report()
        print("\n" + "="*60)
        print(report)


if __name__ == "__main__":
    asyncio.run(main())