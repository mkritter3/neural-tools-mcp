#!/usr/bin/env python3
"""
Simplified MCP Compliance Monitoring System
Phase 3 of ADR-0007: Compliance validation without external dependencies
"""

import sys
import os
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Add current directory to Python path
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"  
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class SimpleComplianceMonitor:
    """Simplified MCP compliance monitoring system"""
    
    def __init__(self):
        self.alert_thresholds = {
            'mcp_tool_success_rate': 80.0,
            'infrastructure_health': 90.0,
            'overall_compliance': 75.0
        }
        
        # Set up environment
        self._setup_environment()
        
    def _setup_environment(self):
        """Set up monitoring environment"""
        os.environ['PROJECT_NAME'] = 'default'
        os.environ['QDRANT_URL'] = 'http://default-neural-storage:6333'
        os.environ['NEO4J_URI'] = 'bolt://neo4j-graph:7687'
        os.environ['NEO4J_USERNAME'] = 'neo4j'
        os.environ['NEO4J_PASSWORD'] = 'neural-l9-2025'
        os.environ['NOMIC_ENDPOINT'] = 'http://neural-embeddings:8000/embed'
    
    async def run_compliance_check(self) -> Dict[str, Any]:
        """Run comprehensive compliance check"""
        logger.info("🔍 Running automated compliance monitoring...")
        
        start_time = datetime.now()
        results = {
            "timestamp": start_time.isoformat(),
            "metrics": {},
            "overall_status": ComplianceStatus.UNKNOWN.value,
            "overall_score": 0.0,
            "alerts": [],
            "recommendations": []
        }
        
        try:
            # 1. Quick MCP Tools Health Check
            mcp_metrics = await self._quick_mcp_health_check()
            results["metrics"]["mcp_tools"] = mcp_metrics
            
            # 2. Infrastructure Health Check  
            infra_metrics = await self._check_infrastructure_health()
            results["metrics"]["infrastructure"] = infra_metrics
            
            # 3. Architecture Compliance
            arch_metrics = self._check_basic_architecture_compliance()
            results["metrics"]["architecture"] = arch_metrics
            
            # 4. Calculate overall compliance
            overall_score, overall_status, alerts, recommendations = self._calculate_overall_compliance(results["metrics"])
            
            results.update({
                "overall_score": overall_score,
                "overall_status": overall_status.value,
                "alerts": alerts,
                "recommendations": recommendations,
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            })
            
            logger.info(f"✅ Compliance check complete: {overall_score:.1f}% ({overall_status.value})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Compliance check failed: {e}")
            results.update({
                "overall_status": ComplianceStatus.CRITICAL.value,
                "error": str(e),
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            })
            return results
    
    async def _quick_mcp_health_check(self) -> Dict[str, Any]:
        """Quick MCP tools health check using system status"""
        logger.info("📊 Quick MCP health check...")
        
        try:
            # Check if neural system status tool works
            neural_server_module = __import__('neural-mcp-server-enhanced')
            await neural_server_module.initialize()
            
            # Test just the system status tool for speed
            try:
                status_func = getattr(neural_server_module, 'neural_system_status', None)
                if status_func:
                    result = await asyncio.wait_for(status_func(), timeout=10.0)
                    
                    if isinstance(result, dict) and result.get('status') == 'success':
                        # Parse system status for health indicators
                        system_data = result.get('system_overview', {})
                        
                        return {
                            "success_rate": 100.0,
                            "status": ComplianceStatus.COMPLIANT.value,
                            "tools_available": system_data.get('total_tools', 15),
                            "system_healthy": True
                        }
                    else:
                        return {
                            "success_rate": 50.0,
                            "status": ComplianceStatus.WARNING.value,
                            "tools_available": "unknown",
                            "system_healthy": False,
                            "error": "System status tool returned unexpected result"
                        }
                else:
                    return {
                        "success_rate": 0.0,
                        "status": ComplianceStatus.CRITICAL.value,
                        "error": "System status tool not found"
                    }
                    
            except asyncio.TimeoutError:
                return {
                    "success_rate": 25.0,
                    "status": ComplianceStatus.WARNING.value,
                    "error": "System status tool timeout"
                }
                
        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            return {
                "success_rate": 0.0,
                "status": ComplianceStatus.CRITICAL.value,
                "error": str(e)
            }
    
    async def _check_infrastructure_health(self) -> Dict[str, Any]:
        """Check infrastructure components health"""
        logger.info("🏗️ Checking infrastructure health...")
        
        components = {
            "qdrant": await self._ping_qdrant(),
            "neo4j": await self._ping_neo4j(),
            "nomic": await self._ping_nomic()
        }
        
        healthy_count = sum(1 for comp in components.values() if comp.get('healthy', False))
        total_count = len(components)
        health_score = (healthy_count / total_count) * 100
        
        status = ComplianceStatus.COMPLIANT if health_score >= self.alert_thresholds['infrastructure_health'] else ComplianceStatus.WARNING
        
        return {
            "overall_score": health_score,
            "status": status.value,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "components": components
        }
    
    async def _ping_qdrant(self) -> Dict[str, Any]:
        """Quick Qdrant health ping"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://default-neural-storage:6333/")
                
                return {
                    "healthy": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _ping_neo4j(self) -> Dict[str, Any]:
        """Quick Neo4j health ping"""
        try:
            from neo4j import AsyncGraphDatabase
            
            driver = AsyncGraphDatabase.driver(
                "bolt://neo4j-graph:7687",
                auth=("neo4j", "neural-l9-2025")
            )
            
            async with driver.session() as session:
                start_time = datetime.now()
                await session.run("RETURN 1")
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                await driver.close()
                return {
                    "healthy": True,
                    "response_time_ms": response_time
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _ping_nomic(self) -> Dict[str, Any]:
        """Quick Nomic embedding service ping"""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=3.0) as client:
                # Try to get health endpoint rather than full embedding
                response = await client.get("http://neural-embeddings:8000/health")
                
                return {
                    "healthy": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    def _check_basic_architecture_compliance(self) -> Dict[str, Any]:
        """Check basic L9 architecture compliance"""
        logger.info("🏛️ Checking architecture compliance...")
        
        checks = {
            "docker_environment": self._check_docker_environment(),
            "mcp_server_structure": self._check_mcp_server_structure(),
            "file_organization": self._check_file_organization()
        }
        
        total_score = sum(check.get('score', 0) for check in checks.values())
        average_score = total_score / len(checks)
        
        status = ComplianceStatus.COMPLIANT if average_score >= 80 else ComplianceStatus.WARNING
        
        return {
            "overall_score": average_score,
            "status": status.value,
            "checks": checks
        }
    
    def _check_docker_environment(self) -> Dict[str, Any]:
        """Check if running in proper Docker environment"""
        try:
            # Check if we're in a container
            with open('/proc/1/cgroup', 'r') as f:
                cgroup_content = f.read()
            
            in_docker = 'docker' in cgroup_content or 'containerd' in cgroup_content
            
            # Check for expected Docker environment variables
            expected_vars = ['PROJECT_NAME', 'QDRANT_URL', 'NEO4J_URI']
            vars_present = sum(1 for var in expected_vars if os.getenv(var))
            
            score = (50 if in_docker else 0) + (vars_present / len(expected_vars)) * 50
            
            return {
                "score": score,
                "in_docker": in_docker,
                "env_vars_present": f"{vars_present}/{len(expected_vars)}"
            }
            
        except Exception as e:
            return {
                "score": 25.0,
                "error": str(e)
            }
    
    def _check_mcp_server_structure(self) -> Dict[str, Any]:
        """Check MCP server file structure"""
        mcp_server_path = Path("/app/neural-mcp-server-enhanced.py")
        
        if not mcp_server_path.exists():
            return {
                "score": 0.0,
                "error": "MCP server file not found"
            }
        
        try:
            with open(mcp_server_path, 'r') as f:
                content = f.read()
            
            # Check for key patterns
            patterns = {
                "mcp_tools": "@mcp.tool" in content,
                "async_functions": "async def" in content,
                "server_setup": "mcp.Server" in content,
                "initialization": "initialize" in content
            }
            
            score = (sum(patterns.values()) / len(patterns)) * 100
            
            return {
                "score": score,
                "patterns_found": patterns,
                "file_size_kb": len(content) / 1024
            }
            
        except Exception as e:
            return {
                "score": 25.0,
                "error": str(e)
            }
    
    def _check_file_organization(self) -> Dict[str, Any]:
        """Check basic file organization"""
        expected_files = [
            "/app/neural-mcp-server-enhanced.py",
            "/app/neo4j_client.py", 
            "/app/requirements-l9-enhanced.txt"
        ]
        
        files_present = sum(1 for file_path in expected_files if Path(file_path).exists())
        score = (files_present / len(expected_files)) * 100
        
        return {
            "score": score,
            "files_present": f"{files_present}/{len(expected_files)}",
            "expected_files": expected_files
        }
    
    def _calculate_overall_compliance(self, metrics: Dict[str, Any]) -> tuple:
        """Calculate overall compliance score and status"""
        
        # Weight different metric categories
        weights = {
            "mcp_tools": 0.5,      # 50% - most critical
            "infrastructure": 0.3,  # 30% - operational health
            "architecture": 0.2     # 20% - structural compliance
        }
        
        total_score = 0.0
        alerts = []
        recommendations = []
        
        for category, weight in weights.items():
            if category in metrics:
                metric_data = metrics[category]
                score = metric_data.get('overall_score', metric_data.get('success_rate', 0))
                total_score += score * weight
                
                # Check for alerts
                if category == "mcp_tools" and score < self.alert_thresholds['mcp_tool_success_rate']:
                    alerts.append(f"MCP tools health below threshold: {score:.1f}% < {self.alert_thresholds['mcp_tool_success_rate']}%")
                    recommendations.append("Investigate MCP tool initialization and service connections")
                
                elif category == "infrastructure" and score < self.alert_thresholds['infrastructure_health']:
                    alerts.append(f"Infrastructure health below threshold: {score:.1f}% < {self.alert_thresholds['infrastructure_health']}%")
                    recommendations.append("Check database connections and container networking")
        
        # Determine overall status
        if total_score >= self.alert_thresholds['overall_compliance']:
            status = ComplianceStatus.COMPLIANT
        elif total_score >= 50:
            status = ComplianceStatus.WARNING  
        else:
            status = ComplianceStatus.CRITICAL
        
        return total_score, status, alerts, recommendations
    
    def generate_compliance_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable compliance report"""
        
        report_lines = [
            "# MCP System Compliance Monitor - Phase 3 ADR-0007",
            f"Generated: {results['timestamp']}",
            f"Duration: {results.get('duration_seconds', 0):.1f} seconds",
            "",
            "## Executive Summary",
            f"- **Overall Score**: {results['overall_score']:.1f}%",
            f"- **Status**: {results['overall_status'].upper()}",
            ""
        ]
        
        # Status interpretation
        status_emoji = {
            'compliant': '✅',
            'warning': '⚠️', 
            'critical': '❌',
            'unknown': '❓'
        }
        
        emoji = status_emoji.get(results['overall_status'], '❓')
        report_lines.append(f"**System Status**: {emoji} {results['overall_status'].upper()}")
        report_lines.append("")
        
        # Alerts section
        if results.get('alerts'):
            report_lines.extend([
                "## 🚨 Active Alerts",
                ""
            ])
            for alert in results['alerts']:
                report_lines.append(f"- ⚠️ {alert}")
            report_lines.append("")
        
        # Recommendations section
        if results.get('recommendations'):
            report_lines.extend([
                "## 💡 Recommended Actions",
                ""
            ])
            for rec in results['recommendations']:
                report_lines.append(f"- 🔧 {rec}")
            report_lines.append("")
        
        # Metrics breakdown
        report_lines.extend([
            "## 📊 Component Health",
            ""
        ])
        
        for category, data in results.get('metrics', {}).items():
            if isinstance(data, dict):
                score = data.get('overall_score', data.get('success_rate', 'N/A'))
                status = data.get('status', 'unknown')
                status_icon = status_emoji.get(status, '❓')
                
                category_name = category.title().replace('_', ' ')
                report_lines.append(f"### {status_icon} {category_name}")
                
                if isinstance(score, (int, float)):
                    report_lines.append(f"- **Score**: {score:.1f}%")
                else:
                    report_lines.append(f"- **Score**: {score}")
                
                report_lines.append(f"- **Status**: {status.upper()}")
                
                # Add specific details
                if category == 'infrastructure':
                    healthy = data.get('healthy_components', 0)
                    total = data.get('total_components', 0)
                    report_lines.append(f"- **Components Healthy**: {healthy}/{total}")
                
                elif category == 'mcp_tools':
                    tools = data.get('tools_available', 'unknown')
                    report_lines.append(f"- **Tools Available**: {tools}")
                
                report_lines.append("")
        
        # Footer
        report_lines.extend([
            "---",
            "*Generated by L9 Automated Compliance Monitor*",
            f"*ADR-0007 Phase 3 Implementation - {datetime.now().strftime('%Y-%m-%d')}*"
        ])
        
        return "\n".join(report_lines)

async def main():
    """Main monitoring function"""
    monitor = SimpleComplianceMonitor()
    
    try:
        # Run compliance check
        results = await monitor.run_compliance_check()
        
        # Generate and save report
        report = monitor.generate_compliance_report(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        results_file = f"/app/compliance_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save report
        report_file = f"/app/compliance_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save as latest
        with open("/app/compliance_results_latest.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        with open("/app/compliance_report_latest.md", 'w') as f:
            f.write(report)
        
        # Print results
        print("\n" + "="*70)
        print("AUTOMATED COMPLIANCE MONITORING - ADR-0007 PHASE 3 COMPLETE")
        print("="*70)
        print(report)
        print("="*70)
        
        logger.info(f"📁 Results saved: {results_file}")
        logger.info(f"📁 Report saved: {report_file}")
        
        return results['overall_score'] >= monitor.alert_thresholds['overall_compliance']
        
    except Exception as e:
        logger.error(f"❌ Monitoring failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)