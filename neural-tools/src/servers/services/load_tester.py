"""
L9 2025 Load Testing and Validation Framework for MCP Server
Concurrent session testing and performance benchmarking
"""

import time
import asyncio
import logging
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import redis.asyncio as redis
import json

logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    concurrent_sessions: int = 15
    test_duration_seconds: int = 300  # 5 minutes
    requests_per_session: int = 10
    ramp_up_time: int = 30  # seconds
    cool_down_time: int = 10  # seconds
    test_scenarios: List[str] = None
    
    def __post_init__(self):
        if self.test_scenarios is None:
            self.test_scenarios = [
                'semantic_search',
                'graphrag_query', 
                'hybrid_search',
                'knowledge_graph',
                'health_check'
            ]

@dataclass
class SessionMetrics:
    """Metrics for individual session"""
    session_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = None
    errors: List[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time"""
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile

@dataclass
class LoadTestResults:
    """Aggregated load test results"""
    config: LoadTestConfig
    session_metrics: List[SessionMetrics]
    start_time: datetime
    end_time: datetime
    container_stats: Dict[str, Any]
    system_stats: Dict[str, Any]
    
    @property
    def total_requests(self) -> int:
        return sum(m.total_requests for m in self.session_metrics)
    
    @property
    def total_successful(self) -> int:
        return sum(m.successful_requests for m in self.session_metrics)
    
    @property
    def overall_success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_successful / self.total_requests) * 100
    
    @property
    def overall_avg_response_time(self) -> float:
        all_times = []
        for m in self.session_metrics:
            all_times.extend(m.response_times)
        return statistics.mean(all_times) if all_times else 0.0
    
    @property
    def requests_per_second(self) -> float:
        duration = (self.end_time - self.start_time).total_seconds()
        return self.total_requests / duration if duration > 0 else 0.0

class LoadTester:
    """L9 2025 Load Testing Framework"""
    
    def __init__(self, service_container):
        self.container = service_container
        self.redis_client: Optional[redis.Redis] = None
        self.test_scenarios = {}
        self._setup_test_scenarios()
    
    async def initialize(self):
        """Initialize load tester"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=46379,
                password='cache-secret-key',
                decode_responses=True,
                db=4  # Use db 4 for load test data
            )
            await self.redis_client.ping()
            logger.info("✅ LoadTester initialized with Redis storage")
        except Exception as e:
            logger.warning(f"Redis not available for load testing, using in-memory: {e}")
            self.redis_client = None
    
    def _setup_test_scenarios(self):
        """Setup test scenarios for load testing"""
        self.test_scenarios = {
            'semantic_search': self._semantic_search_scenario,
            'graphrag_query': self._graphrag_query_scenario,
            'hybrid_search': self._hybrid_search_scenario,
            'knowledge_graph': self._knowledge_graph_scenario,
            'health_check': self._health_check_scenario
        }
    
    async def run_load_test(self, config: LoadTestConfig) -> LoadTestResults:
        """Run comprehensive load test"""
        logger.info(f"🧪 Starting L9 load test: {config.concurrent_sessions} sessions, {config.test_duration_seconds}s duration")
        
        start_time = datetime.utcnow()
        
        # Collect baseline metrics
        baseline_stats = await self._collect_system_stats()
        
        # Create session tasks
        session_tasks = []
        session_metrics = []
        
        for i in range(config.concurrent_sessions):
            session_id = f"load_test_session_{i:03d}"
            metrics = SessionMetrics(session_id=session_id)
            session_metrics.append(metrics)
            
            task = asyncio.create_task(
                self._run_session_test(session_id, config, metrics)
            )
            session_tasks.append(task)
            
            # Ramp up delay
            if config.ramp_up_time > 0:
                await asyncio.sleep(config.ramp_up_time / config.concurrent_sessions)
        
        # Wait for all sessions to complete
        logger.info(f"⏳ Running {len(session_tasks)} concurrent sessions...")
        await asyncio.gather(*session_tasks, return_exceptions=True)
        
        # Cool down period
        if config.cool_down_time > 0:
            logger.info(f"❄️ Cool down period: {config.cool_down_time}s")
            await asyncio.sleep(config.cool_down_time)
        
        end_time = datetime.utcnow()
        
        # Collect final metrics
        final_stats = await self._collect_system_stats()
        container_stats = await self._collect_container_stats()
        
        results = LoadTestResults(
            config=config,
            session_metrics=session_metrics,
            start_time=start_time,
            end_time=end_time,
            container_stats=container_stats,
            system_stats=final_stats
        )
        
        # Store results
        await self._store_test_results(results)
        
        logger.info(f"✅ Load test completed: {results.overall_success_rate:.1f}% success rate, {results.requests_per_second:.1f} RPS")
        
        return results
    
    async def _run_session_test(
        self, 
        session_id: str, 
        config: LoadTestConfig, 
        metrics: SessionMetrics
    ):
        """Run test for individual session"""
        metrics.start_time = datetime.utcnow()
        
        try:
            # Create session context
            session_manager = self.container.session_manager
            if session_manager:
                actual_session_id = await session_manager.create_session({
                    'source': 'load_test',
                    'test_session': session_id
                })
            else:
                actual_session_id = session_id
            
            # Run test scenarios
            for _ in range(config.requests_per_session):
                scenario_name = config.test_scenarios[
                    metrics.total_requests % len(config.test_scenarios)
                ]
                
                await self._execute_scenario(
                    scenario_name, 
                    actual_session_id, 
                    metrics
                )
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            # Cleanup session
            if session_manager and actual_session_id != session_id:
                await session_manager.cleanup_session(actual_session_id)
                
        except Exception as e:
            metrics.errors.append(f"Session error: {e}")
            logger.error(f"Session {session_id} failed: {e}")
        
        finally:
            metrics.end_time = datetime.utcnow()
    
    async def _execute_scenario(
        self, 
        scenario_name: str, 
        session_id: str, 
        metrics: SessionMetrics
    ):
        """Execute single test scenario"""
        start_time = time.time()
        metrics.total_requests += 1
        
        try:
            scenario_func = self.test_scenarios.get(scenario_name)
            if not scenario_func:
                raise ValueError(f"Unknown scenario: {scenario_name}")
            
            result = await scenario_func(session_id)
            
            # Validate result
            if isinstance(result, dict) and result.get('success'):
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                metrics.errors.append(f"{scenario_name}: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            metrics.failed_requests += 1
            metrics.errors.append(f"{scenario_name}: {e}")
        
        finally:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            metrics.response_times.append(response_time)
    
    async def _semantic_search_scenario(self, session_id: str) -> Dict[str, Any]:
        """Test semantic search scenario"""
        try:
            if not self.container.qdrant:
                return {'success': False, 'error': 'Qdrant not available'}
            
            # Create a simple test vector for search
            test_vector = [0.1] * 768  # Simple 768-dim vector
            
            # Simulate semantic search with correct API
            search_result = await self.container.qdrant.search_vectors(
                collection_name="default",
                query_vector=test_vector,
                limit=5
            )
            
            return {'success': True, 'results': len(search_result.get('results', []))}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _graphrag_query_scenario(self, session_id: str) -> Dict[str, Any]:
        """Test GraphRAG query scenario"""
        try:
            if not self.container.neo4j:
                return {'success': False, 'error': 'Neo4j not available'}
            
            # Simulate GraphRAG query
            result = await self.container.neo4j.execute_query(
                "MATCH (n) RETURN count(n) as node_count LIMIT 1"
            )
            
            return {'success': True, 'node_count': result.get('results', [{}])[0].get('node_count', 0)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _hybrid_search_scenario(self, session_id: str) -> Dict[str, Any]:
        """Test hybrid search scenario"""
        try:
            # Combine semantic + graph search
            tasks = []
            
            if self.container.qdrant:
                tasks.append(self._semantic_search_scenario(session_id))
            
            if self.container.neo4j:
                tasks.append(self._graphrag_query_scenario(session_id))
            
            if not tasks:
                return {'success': False, 'error': 'No services available'}
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
            
            return {
                'success': success_count > 0,
                'semantic_result': results[0] if len(results) > 0 else None,
                'graph_result': results[1] if len(results) > 1 else None
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _knowledge_graph_scenario(self, session_id: str) -> Dict[str, Any]:
        """Test knowledge graph operations"""
        try:
            if not self.container.neo4j:
                return {'success': False, 'error': 'Neo4j not available'}
            
            # Simulate knowledge graph traversal
            result = await self.container.neo4j.execute_query(
                "MATCH (n)-[r]->(m) RETURN count(r) as relationship_count LIMIT 1"
            )
            
            return {'success': True, 'relationships': result.get('results', [{}])[0].get('relationship_count', 0)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _health_check_scenario(self, session_id: str) -> Dict[str, Any]:
        """Test health check scenario"""
        try:
            if self.container.health_monitor:
                health = await self.container.health_monitor.get_overall_health()
                return {
                    'success': health['status'] in ['healthy', 'degraded'],
                    'status': health['status'],
                    'health_percentage': health['summary']['health_percentage']
                }
            else:
                return {'success': True, 'status': 'no_monitor'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _collect_system_stats(self) -> Dict[str, Any]:
        """Collect system performance statistics"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2)
            }
            
        except ImportError:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': 'psutil not available'
            }
        except Exception as e:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    async def _collect_container_stats(self) -> Dict[str, Any]:
        """Collect container performance statistics"""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'connection_pools': {},
            'session_stats': {},
            'health_status': {}
        }
        
        # Connection pool stats
        if hasattr(self.container, 'connection_pools'):
            for service, pool_data in self.container.connection_pools.items():
                stats['connection_pools'][service] = {
                    'active': pool_data['active'],
                    'max_size': pool_data['max_size'],
                    'utilization': round((pool_data['active'] / pool_data['max_size']) * 100, 2)
                }
        
        # Session stats
        if self.container.session_manager:
            session_stats = self.container.session_manager.get_session_stats()
            stats['session_stats'] = session_stats
        
        # Health status
        if self.container.health_monitor:
            try:
                health = await self.container.health_monitor.get_overall_health()
                stats['health_status'] = {
                    'status': health['status'],
                    'health_percentage': health['summary']['health_percentage'],
                    'service_count': health['summary']['total']
                }
            except Exception as e:
                stats['health_status'] = {'error': str(e)}
        
        return stats
    
    async def _store_test_results(self, results: LoadTestResults):
        """Store test results for analysis"""
        if not self.redis_client:
            return
        
        try:
            # Convert to JSON-serializable format
            results_data = {
                'config': asdict(results.config),
                'session_metrics': [asdict(m) for m in results.session_metrics],
                'start_time': results.start_time.isoformat(),
                'end_time': results.end_time.isoformat(),
                'container_stats': results.container_stats,
                'system_stats': results.system_stats,
                'summary': {
                    'total_requests': results.total_requests,
                    'success_rate': results.overall_success_rate,
                    'avg_response_time': results.overall_avg_response_time,
                    'requests_per_second': results.requests_per_second
                }
            }
            
            # Store with timestamp key
            timestamp = results.start_time.strftime('%Y%m%d_%H%M%S')
            key = f"load_test_results:{timestamp}"
            
            await self.redis_client.setex(
                key,
                86400 * 7,  # 7 day retention
                json.dumps(results_data, default=str)
            )
            
            # Add to index
            await self.redis_client.lpush('load_test_index', key)
            await self.redis_client.ltrim('load_test_index', 0, 99)  # Keep last 100 tests
            
            logger.info(f"📊 Load test results stored: {key}")
            
        except Exception as e:
            logger.error(f"Failed to store load test results: {e}")
    
    async def get_historical_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical load test results"""
        if not self.redis_client:
            return []
        
        try:
            test_keys = await self.redis_client.lrange('load_test_index', 0, limit - 1)
            results = []
            
            for key in test_keys:
                data = await self.redis_client.get(key)
                if data:
                    results.append(json.loads(data))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get historical results: {e}")
            return []