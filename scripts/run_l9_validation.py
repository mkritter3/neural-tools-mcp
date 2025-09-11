#!/usr/bin/env python3
"""
L9 2025 Validation Script for MCP Architecture
Runs comprehensive load testing and performance validation
"""

import os
import sys
import asyncio
import logging
import json
from pathlib import Path

# Add neural-tools to path
neural_tools_path = Path(__file__).parent.parent / "neural-tools" / "src"
sys.path.insert(0, str(neural_tools_path))

from servers.services.service_container import ServiceContainer
from servers.services.load_tester import LoadTester, LoadTestConfig
from servers.services.performance_validator import PerformanceValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_l9_validation():
    """Run complete L9 2025 validation suite"""
    
    logger.info("🚀 Starting L9 2025 MCP Architecture Validation")
    
    # Initialize service container
    logger.info("1. Initializing service container...")
    container = ServiceContainer("l9-validation")
    container.initialize()
    
    # Initialize all services
    await container.initialize_connection_pools()
    if container.session_manager:
        await container.session_manager.initialize()
    if hasattr(container, 'initialize_security_services'):
        await container.initialize_security_services()
    
    logger.info("✅ Service container initialized")
    
    # Initialize load tester
    logger.info("2. Initializing load tester...")
    load_tester = LoadTester(container)
    await load_tester.initialize()
    
    # Initialize performance validator
    validator = PerformanceValidator()
    
    # Configure load test
    test_configs = [
        # Light load test
        LoadTestConfig(
            concurrent_sessions=5,
            test_duration_seconds=60,
            requests_per_session=5,
            ramp_up_time=10
        ),
        # Medium load test
        LoadTestConfig(
            concurrent_sessions=10,
            test_duration_seconds=120,
            requests_per_session=8,
            ramp_up_time=20
        ),
        # Heavy load test (L9 requirement)
        LoadTestConfig(
            concurrent_sessions=15,
            test_duration_seconds=180,
            requests_per_session=10,
            ramp_up_time=30
        )
    ]
    
    all_results = []
    
    for i, config in enumerate(test_configs, 1):
        logger.info(f"3.{i} Running load test {i}/3: {config.concurrent_sessions} sessions...")
        
        # Run load test
        results = await load_tester.run_load_test(config)
        
        # Validate results
        validation = validator.validate_load_test_results(results)
        
        # Store combined results
        combined_result = {
            "test_number": i,
            "load_test_results": {
                "config": config.__dict__,
                "summary": {
                    "total_requests": results.total_requests,
                    "success_rate": results.overall_success_rate,
                    "avg_response_time": results.overall_avg_response_time,
                    "requests_per_second": results.requests_per_second
                },
                "session_count": len(results.session_metrics),
                "container_stats": results.container_stats,
                "system_stats": results.system_stats
            },
            "validation": validation
        }
        
        all_results.append(combined_result)
        
        # Print immediate results
        print(f"\n📊 Test {i} Results:")
        print(f"   Success Rate: {results.overall_success_rate:.1f}%")
        print(f"   Avg Response: {results.overall_avg_response_time:.1f}ms")
        print(f"   Throughput: {results.requests_per_second:.1f} RPS")
        print(f"   Performance Grade: {validation['performance_grade']}")
        
        # Short break between tests
        if i < len(test_configs):
            logger.info("⏸️ Short break between tests...")
            await asyncio.sleep(30)
    
    # Generate final report
    logger.info("4. Generating L9 validation report...")
    final_report = generate_final_report(all_results, validator)
    
    # Save report
    report_path = Path(__file__).parent.parent / "docs" / "l9_validation_report.json"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info(f"📄 Report saved to: {report_path}")
    
    # Print summary
    print_validation_summary(final_report)
    
    # Cleanup
    if container.health_monitor:
        await container.health_monitor.stop_monitoring()
    
    logger.info("🎉 L9 validation completed!")
    
    return final_report

def generate_final_report(all_results, validator):
    """Generate comprehensive validation report"""
    
    # Extract key metrics from all tests
    test_summaries = []
    best_performance = None
    worst_performance = None
    
    for result in all_results:
        summary = result["load_test_results"]["summary"]
        validation = result["validation"]
        
        test_summary = {
            "test_number": result["test_number"],
            "concurrent_sessions": result["load_test_results"]["config"]["concurrent_sessions"],
            "success_rate": summary["success_rate"],
            "avg_response_time": summary["avg_response_time"],
            "requests_per_second": summary["requests_per_second"],
            "performance_grade": validation["performance_grade"],
            "validation_pass_rate": validation["analysis"]["validation_summary"]["pass_rate"]
        }
        
        test_summaries.append(test_summary)
        
        # Track best/worst performance
        if best_performance is None or summary["success_rate"] > best_performance["success_rate"]:
            best_performance = test_summary
        
        if worst_performance is None or summary["success_rate"] < worst_performance["success_rate"]:
            worst_performance = test_summary
    
    # Calculate overall assessment
    l9_requirements_met = {
        "concurrent_sessions_15": any(t["concurrent_sessions"] >= 15 for t in test_summaries),
        "success_rate_95": all(t["success_rate"] >= 95.0 for t in test_summaries),
        "response_time_500ms": all(t["avg_response_time"] <= 500.0 for t in test_summaries),
        "throughput_10rps": all(t["requests_per_second"] >= 10.0 for t in test_summaries)
    }
    
    l9_compliance = all(l9_requirements_met.values())
    
    # Get pool optimization suggestions from the heaviest test
    heaviest_test = max(all_results, key=lambda x: x["load_test_results"]["config"]["concurrent_sessions"])
    pool_suggestions = validator.get_pool_optimization_suggestions(
        type('Results', (), heaviest_test["load_test_results"])()
    )
    
    return {
        "validation_timestamp": all_results[0]["validation"]["validation_timestamp"],
        "l9_architecture_version": "2025-09-10",
        "test_summary": {
            "total_tests": len(all_results),
            "tests_performed": test_summaries,
            "best_performance": best_performance,
            "worst_performance": worst_performance
        },
        "l9_compliance": {
            "overall_compliant": l9_compliance,
            "requirements_met": l9_requirements_met,
            "compliance_percentage": round(sum(l9_requirements_met.values()) / len(l9_requirements_met) * 100, 1)
        },
        "detailed_results": all_results,
        "optimization_recommendations": {
            "connection_pools": pool_suggestions,
            "priority_actions": extract_priority_actions(all_results)
        },
        "benchmarks": validator.l9_benchmarks
    }

def extract_priority_actions(all_results):
    """Extract priority actions from all validation results"""
    all_recommendations = []
    
    for result in all_results:
        recommendations = result["validation"].get("recommendations", [])
        all_recommendations.extend(recommendations)
    
    # Group by category and priority
    priority_actions = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": []
    }
    
    seen_actions = set()
    
    for rec in all_recommendations:
        action_key = f"{rec['category']}:{rec['title']}"
        if action_key not in seen_actions:
            priority_actions[rec["priority"]].append(rec)
            seen_actions.add(action_key)
    
    return priority_actions

def print_validation_summary(report):
    """Print validation summary to console"""
    
    print("\n" + "="*60)
    print("🏆 L9 2025 MCP ARCHITECTURE VALIDATION SUMMARY")
    print("="*60)
    
    compliance = report["l9_compliance"]
    print(f"\n📋 L9 Compliance: {'✅ COMPLIANT' if compliance['overall_compliant'] else '❌ NON-COMPLIANT'}")
    print(f"   Compliance Rate: {compliance['compliance_percentage']}%")
    
    requirements = compliance["requirements_met"]
    print(f"\n📊 Requirements Check:")
    print(f"   ✅ 15+ Concurrent Sessions: {'PASS' if requirements['concurrent_sessions_15'] else 'FAIL'}")
    print(f"   ✅ 95%+ Success Rate: {'PASS' if requirements['success_rate_95'] else 'FAIL'}")
    print(f"   ✅ <500ms Response Time: {'PASS' if requirements['response_time_500ms'] else 'FAIL'}")
    print(f"   ✅ 10+ RPS Throughput: {'PASS' if requirements['throughput_10rps'] else 'FAIL'}")
    
    summary = report["test_summary"]
    print(f"\n🎯 Performance Summary:")
    print(f"   Best Test: #{summary['best_performance']['test_number']} - {summary['best_performance']['success_rate']:.1f}% success")
    print(f"   Worst Test: #{summary['worst_performance']['test_number']} - {summary['worst_performance']['success_rate']:.1f}% success")
    
    # Show critical recommendations
    critical_actions = report["optimization_recommendations"]["priority_actions"]["critical"]
    if critical_actions:
        print(f"\n🚨 Critical Actions Required:")
        for action in critical_actions[:3]:  # Show top 3
            print(f"   • {action['title']}: {action['description']}")
    
    print(f"\n📄 Full report saved to: docs/l9_validation_report.json")
    print("="*60)

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.update({
        "NEO4J_URI": "bolt://localhost:47687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "graphrag-password",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "46333",
        "REDIS_CACHE_HOST": "localhost",
        "REDIS_CACHE_PORT": "46379",
        "REDIS_CACHE_PASSWORD": "cache-secret-key",
        "REDIS_QUEUE_HOST": "localhost", 
        "REDIS_QUEUE_PORT": "46380",
        "REDIS_QUEUE_PASSWORD": "queue-secret-key"
    })
    
    try:
        asyncio.run(run_l9_validation())
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)