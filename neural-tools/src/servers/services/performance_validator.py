"""
L9 2025 Performance Validator for MCP Architecture
Validates pool optimization and performance benchmarks
"""

import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .load_tester import LoadTestResults, LoadTestConfig

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

@dataclass
class ValidationRule:
    """Performance validation rule"""
    name: str
    metric_path: str
    threshold: float
    operator: str  # 'gt', 'lt', 'gte', 'lte', 'eq'
    severity: ValidationStatus
    description: str

@dataclass
class ValidationResult:
    """Result of a validation rule"""
    rule: ValidationRule
    actual_value: float
    expected_value: float
    status: ValidationStatus
    message: str
    timestamp: datetime

class PerformanceValidator:
    """L9 2025 Performance Validation Framework"""
    
    def __init__(self):
        self.validation_rules = self._setup_validation_rules()
        self.l9_benchmarks = self._setup_l9_benchmarks()
    
    def _setup_validation_rules(self) -> List[ValidationRule]:
        """Setup L9 2025 performance validation rules"""
        return [
            # Success Rate Rules
            ValidationRule(
                name="overall_success_rate",
                metric_path="overall_success_rate",
                threshold=95.0,
                operator="gte",
                severity=ValidationStatus.FAIL,
                description="Overall request success rate must be >= 95%"
            ),
            ValidationRule(
                name="session_success_rate_variance",
                metric_path="session_success_variance",
                threshold=10.0,
                operator="lte",
                severity=ValidationStatus.WARN,
                description="Session success rate variance should be <= 10%"
            ),
            
            # Response Time Rules
            ValidationRule(
                name="avg_response_time",
                metric_path="overall_avg_response_time",
                threshold=500.0,  # 500ms
                operator="lte",
                severity=ValidationStatus.WARN,
                description="Average response time should be <= 500ms"
            ),
            ValidationRule(
                name="p95_response_time",
                metric_path="p95_response_time",
                threshold=1000.0,  # 1 second
                operator="lte",
                severity=ValidationStatus.FAIL,
                description="95th percentile response time must be <= 1000ms"
            ),
            
            # Throughput Rules
            ValidationRule(
                name="requests_per_second",
                metric_path="requests_per_second",
                threshold=10.0,
                operator="gte",
                severity=ValidationStatus.WARN,
                description="System should handle >= 10 requests per second"
            ),
            ValidationRule(
                name="concurrent_session_support",
                metric_path="concurrent_sessions",
                threshold=15.0,
                operator="gte",
                severity=ValidationStatus.FAIL,
                description="System must support >= 15 concurrent sessions"
            ),
            
            # Resource Utilization Rules
            ValidationRule(
                name="connection_pool_efficiency",
                metric_path="avg_pool_utilization",
                threshold=70.0,
                operator="lte",
                severity=ValidationStatus.WARN,
                description="Average pool utilization should be <= 70%"
            ),
            ValidationRule(
                name="system_cpu_usage",
                metric_path="cpu_percent",
                threshold=80.0,
                operator="lte",
                severity=ValidationStatus.WARN,
                description="CPU usage should be <= 80% during load test"
            ),
            ValidationRule(
                name="system_memory_usage",
                metric_path="memory_percent",
                threshold=85.0,
                operator="lte",
                severity=ValidationStatus.FAIL,
                description="Memory usage must be <= 85% during load test"
            ),
            
            # Health Rules
            ValidationRule(
                name="health_percentage",
                metric_path="health_percentage",
                threshold=90.0,
                operator="gte",
                severity=ValidationStatus.FAIL,
                description="System health percentage must be >= 90%"
            )
        ]
    
    def _setup_l9_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Setup L9 2025 performance benchmarks"""
        return {
            "target": {
                "success_rate": 99.0,
                "avg_response_time_ms": 200.0,
                "p95_response_time_ms": 500.0,
                "requests_per_second": 50.0,
                "concurrent_sessions": 20,
                "pool_utilization": 50.0,
                "cpu_usage": 60.0,
                "memory_usage": 70.0
            },
            "minimum": {
                "success_rate": 95.0,
                "avg_response_time_ms": 500.0,
                "p95_response_time_ms": 1000.0,
                "requests_per_second": 10.0,
                "concurrent_sessions": 15,
                "pool_utilization": 70.0,
                "cpu_usage": 80.0,
                "memory_usage": 85.0
            }
        }
    
    def validate_load_test_results(self, results: LoadTestResults) -> Dict[str, Any]:
        """Validate load test results against L9 standards"""
        logger.info("📊 Validating load test results against L9 2025 benchmarks")
        
        # Extract metrics for validation
        metrics = self._extract_metrics(results)
        
        # Run validation rules
        validation_results = []
        for rule in self.validation_rules:
            result = self._validate_rule(rule, metrics)
            validation_results.append(result)
        
        # Analyze results
        analysis = self._analyze_validation_results(validation_results, metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis, metrics)
        
        return {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "test_config": {
                "concurrent_sessions": results.config.concurrent_sessions,
                "test_duration": results.config.test_duration_seconds,
                "total_requests": results.total_requests
            },
            "validation_results": [
                {
                    "rule": result.rule.name,
                    "status": result.status.value,
                    "actual": result.actual_value,
                    "expected": result.expected_value,
                    "message": result.message,
                    "description": result.rule.description
                }
                for result in validation_results
            ],
            "analysis": analysis,
            "recommendations": recommendations,
            "l9_benchmarks": self.l9_benchmarks,
            "performance_grade": self._calculate_performance_grade(validation_results)
        }
    
    def _extract_metrics(self, results: LoadTestResults) -> Dict[str, float]:
        """Extract key metrics from load test results"""
        metrics = {}
        
        # Basic metrics
        metrics["overall_success_rate"] = results.overall_success_rate
        metrics["overall_avg_response_time"] = results.overall_avg_response_time
        metrics["requests_per_second"] = results.requests_per_second
        metrics["concurrent_sessions"] = results.config.concurrent_sessions
        
        # Calculate P95 response time across all sessions
        all_response_times = []
        session_success_rates = []
        
        for session in results.session_metrics:
            all_response_times.extend(session.response_times)
            session_success_rates.append(session.success_rate)
        
        if all_response_times:
            metrics["p95_response_time"] = statistics.quantiles(all_response_times, n=20)[18]
        else:
            metrics["p95_response_time"] = 0.0
        
        # Session variance
        if session_success_rates:
            metrics["session_success_variance"] = statistics.stdev(session_success_rates)
        else:
            metrics["session_success_variance"] = 0.0
        
        # Connection pool metrics
        if results.container_stats.get("connection_pools"):
            pool_utilizations = [
                pool["utilization"] 
                for pool in results.container_stats["connection_pools"].values()
            ]
            metrics["avg_pool_utilization"] = statistics.mean(pool_utilizations) if pool_utilizations else 0.0
        else:
            metrics["avg_pool_utilization"] = 0.0
        
        # System metrics
        system_stats = results.system_stats
        metrics["cpu_percent"] = system_stats.get("cpu_percent", 0.0)
        metrics["memory_percent"] = system_stats.get("memory_percent", 0.0)
        
        # Health metrics
        health_status = results.container_stats.get("health_status", {})
        metrics["health_percentage"] = health_status.get("health_percentage", 0.0)
        
        return metrics
    
    def _validate_rule(self, rule: ValidationRule, metrics: Dict[str, float]) -> ValidationResult:
        """Validate a single rule against metrics"""
        actual_value = metrics.get(rule.metric_path, 0.0)
        expected_value = rule.threshold
        
        # Apply operator
        operators = {
            "gt": lambda a, e: a > e,
            "lt": lambda a, e: a < e,
            "gte": lambda a, e: a >= e,
            "lte": lambda a, e: a <= e,
            "eq": lambda a, e: abs(a - e) < 0.01  # Float equality
        }
        
        condition_met = operators[rule.operator](actual_value, expected_value)
        status = ValidationStatus.PASS if condition_met else rule.severity
        
        # Generate message
        if condition_met:
            message = f"✅ {rule.name}: {actual_value:.2f} {rule.operator} {expected_value:.2f}"
        else:
            message = f"❌ {rule.name}: {actual_value:.2f} {rule.operator} {expected_value:.2f} - {rule.description}"
        
        return ValidationResult(
            rule=rule,
            actual_value=actual_value,
            expected_value=expected_value,
            status=status,
            message=message,
            timestamp=datetime.utcnow()
        )
    
    def _analyze_validation_results(
        self, 
        validation_results: List[ValidationResult], 
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze validation results and provide insights"""
        
        pass_count = sum(1 for r in validation_results if r.status == ValidationStatus.PASS)
        warn_count = sum(1 for r in validation_results if r.status == ValidationStatus.WARN)
        fail_count = sum(1 for r in validation_results if r.status == ValidationStatus.FAIL)
        
        total_count = len(validation_results)
        
        # Performance comparison with benchmarks
        target_benchmark = self.l9_benchmarks["target"]
        minimum_benchmark = self.l9_benchmarks["minimum"]
        
        benchmark_comparison = {}
        for metric, target_value in target_benchmark.items():
            actual = metrics.get(metric, 0.0)
            minimum = minimum_benchmark.get(metric, target_value)
            
            if metric in ["success_rate", "requests_per_second", "concurrent_sessions", "health_percentage"]:
                # Higher is better
                if actual >= target_value:
                    performance = "exceeds_target"
                elif actual >= minimum:
                    performance = "meets_minimum"
                else:
                    performance = "below_minimum"
            else:
                # Lower is better (response times, utilization, etc.)
                if actual <= target_value:
                    performance = "exceeds_target"
                elif actual <= minimum:
                    performance = "meets_minimum"
                else:
                    performance = "below_minimum"
            
            benchmark_comparison[metric] = {
                "actual": actual,
                "target": target_value,
                "minimum": minimum,
                "performance": performance
            }
        
        return {
            "validation_summary": {
                "total_rules": total_count,
                "passed": pass_count,
                "warnings": warn_count,
                "failures": fail_count,
                "pass_rate": round((pass_count / total_count) * 100, 1) if total_count > 0 else 0
            },
            "benchmark_comparison": benchmark_comparison,
            "critical_issues": [
                r.message for r in validation_results 
                if r.status == ValidationStatus.FAIL
            ],
            "warnings": [
                r.message for r in validation_results 
                if r.status == ValidationStatus.WARN
            ]
        }
    
    def _generate_recommendations(
        self, 
        analysis: Dict[str, Any], 
        metrics: Dict[str, float]
    ) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        benchmark_comparison = analysis["benchmark_comparison"]
        
        # Pool optimization recommendations
        pool_utilization = benchmark_comparison.get("pool_utilization", {})
        if pool_utilization.get("performance") == "below_minimum":
            recommendations.append({
                "category": "connection_pools",
                "priority": "high",
                "title": "Optimize Connection Pools",
                "description": f"Pool utilization is {pool_utilization['actual']:.1f}%, consider reducing pool sizes to improve efficiency",
                "action": "Reduce NEO4J_POOL_SIZE, QDRANT_POOL_SIZE, and REDIS_POOL_SIZE by 20-30%"
            })
        
        # Response time recommendations
        response_time = benchmark_comparison.get("avg_response_time_ms", {})
        if response_time.get("performance") == "below_minimum":
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "title": "Improve Response Times",
                "description": f"Average response time is {response_time['actual']:.1f}ms, implement caching and query optimization",
                "action": "Enable Redis caching, optimize Neo4j queries, add connection pooling warmup"
            })
        
        # Throughput recommendations
        rps = benchmark_comparison.get("requests_per_second", {})
        if rps.get("performance") == "below_minimum":
            recommendations.append({
                "category": "scalability",
                "priority": "medium",
                "title": "Increase Throughput",
                "description": f"Current throughput is {rps['actual']:.1f} RPS, optimize for higher concurrency",
                "action": "Increase connection pool sizes, implement request batching, optimize async operations"
            })
        
        # Resource recommendations
        cpu_usage = benchmark_comparison.get("cpu_usage", {})
        memory_usage = benchmark_comparison.get("memory_usage", {})
        
        if cpu_usage.get("performance") == "below_minimum":
            recommendations.append({
                "category": "resources",
                "priority": "medium",
                "title": "Reduce CPU Usage",
                "description": f"CPU usage is {cpu_usage['actual']:.1f}%, optimize computational efficiency",
                "action": "Profile CPU-intensive operations, implement caching, optimize algorithms"
            })
        
        if memory_usage.get("performance") == "below_minimum":
            recommendations.append({
                "category": "resources",
                "priority": "high",
                "title": "Reduce Memory Usage",
                "description": f"Memory usage is {memory_usage['actual']:.1f}%, implement memory optimization",
                "action": "Review memory leaks, optimize data structures, implement memory pooling"
            })
        
        # Success rate recommendations
        success_rate = benchmark_comparison.get("success_rate", {})
        if success_rate.get("performance") == "below_minimum":
            recommendations.append({
                "category": "reliability",
                "priority": "critical",
                "title": "Improve Success Rate",
                "description": f"Success rate is {success_rate['actual']:.1f}%, implement better error handling",
                "action": "Add circuit breakers, implement retry logic, improve timeout handling"
            })
        
        return recommendations
    
    def _calculate_performance_grade(self, validation_results: List[ValidationResult]) -> str:
        """Calculate overall performance grade (A-F)"""
        fail_count = sum(1 for r in validation_results if r.status == ValidationStatus.FAIL)
        warn_count = sum(1 for r in validation_results if r.status == ValidationStatus.WARN)
        pass_count = sum(1 for r in validation_results if r.status == ValidationStatus.PASS)
        
        total_count = len(validation_results)
        
        if fail_count == 0 and warn_count == 0:
            return "A"  # Perfect
        elif fail_count == 0 and warn_count <= 2:
            return "B"  # Excellent
        elif fail_count <= 1 and warn_count <= 3:
            return "C"  # Good
        elif fail_count <= 2 and warn_count <= 5:
            return "D"  # Acceptable
        else:
            return "F"  # Needs improvement
    
    def get_pool_optimization_suggestions(
        self, 
        results: LoadTestResults
    ) -> Dict[str, Dict[str, int]]:
        """Generate specific pool size optimization suggestions"""
        
        current_pools = results.container_stats.get("connection_pools", {})
        suggestions = {}
        
        for service, pool_data in current_pools.items():
            current_size = pool_data["max_size"]
            utilization = pool_data["utilization"]
            
            # Calculate optimal size based on utilization and L9 guidelines
            if utilization < 30:
                # Under-utilized - reduce by 30%
                suggested_size = max(int(current_size * 0.7), 5)
                reason = f"Low utilization ({utilization:.1f}%) - reduce to improve efficiency"
            elif utilization > 80:
                # Over-utilized - increase by 50%
                suggested_size = int(current_size * 1.5)
                reason = f"High utilization ({utilization:.1f}%) - increase to prevent bottlenecks"
            elif utilization > 70:
                # Slightly over-utilized - increase by 20%
                suggested_size = int(current_size * 1.2)
                reason = f"Moderate utilization ({utilization:.1f}%) - small increase for safety margin"
            else:
                # Optimal range - no change
                suggested_size = current_size
                reason = f"Optimal utilization ({utilization:.1f}%) - no change needed"
            
            suggestions[service] = {
                "current_size": current_size,
                "suggested_size": suggested_size,
                "current_utilization": utilization,
                "change": suggested_size - current_size,
                "reason": reason
            }
        
        return suggestions