"""
L9 2025 Connection Pool Monitor with Metrics
Real-time monitoring and alerting for connection pools
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class PoolMetrics:
    """Metrics collection for connection pools"""
    
    def __init__(self):
        self.metrics = {
            'pool_utilization': {},
            'connection_requests': {},
            'connection_failures': {},
            'average_hold_time': {},
            'peak_utilization': {},
            'last_updated': time.time()
        }
        
    def record_connection_request(self, service: str, success: bool, hold_time: float = 0):
        """Record connection request metrics"""
        current_time = time.time()
        
        if service not in self.metrics['connection_requests']:
            self.metrics['connection_requests'][service] = {'total': 0, 'successful': 0}
            self.metrics['connection_failures'][service] = 0
            self.metrics['average_hold_time'][service] = []
            
        self.metrics['connection_requests'][service]['total'] += 1
        
        if success:
            self.metrics['connection_requests'][service]['successful'] += 1
            if hold_time > 0:
                self.metrics['average_hold_time'][service].append(hold_time)
                # Keep only last 100 measurements
                if len(self.metrics['average_hold_time'][service]) > 100:
                    self.metrics['average_hold_time'][service] = self.metrics['average_hold_time'][service][-100:]
        else:
            self.metrics['connection_failures'][service] += 1
            
        self.metrics['last_updated'] = current_time
    
    def update_pool_utilization(self, service: str, active: int, max_size: int):
        """Update pool utilization metrics"""
        utilization = (active / max_size) * 100 if max_size > 0 else 0
        
        self.metrics['pool_utilization'][service] = {
            'active': active,
            'max_size': max_size,
            'utilization_percent': utilization,
            'timestamp': time.time()
        }
        
        # Track peak utilization
        if service not in self.metrics['peak_utilization']:
            self.metrics['peak_utilization'][service] = 0
        
        if utilization > self.metrics['peak_utilization'][service]:
            self.metrics['peak_utilization'][service] = utilization
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'services': {}
        }
        
        for service in self.metrics['pool_utilization']:
            pool_data = self.metrics['pool_utilization'][service]
            request_data = self.metrics['connection_requests'].get(service, {'total': 0, 'successful': 0})
            hold_times = self.metrics['average_hold_time'].get(service, [])
            
            summary['services'][service] = {
                'pool': {
                    'active_connections': pool_data['active'],
                    'max_connections': pool_data['max_size'],
                    'utilization_percent': round(pool_data['utilization_percent'], 2),
                    'peak_utilization_percent': round(self.metrics['peak_utilization'].get(service, 0), 2)
                },
                'requests': {
                    'total': request_data['total'],
                    'successful': request_data['successful'],
                    'failed': self.metrics['connection_failures'].get(service, 0),
                    'success_rate_percent': round(
                        (request_data['successful'] / request_data['total'] * 100) 
                        if request_data['total'] > 0 else 100, 2
                    )
                },
                'performance': {
                    'avg_hold_time_seconds': round(
                        sum(hold_times) / len(hold_times), 3
                    ) if hold_times else 0,
                    'samples': len(hold_times)
                }
            }
            
        return summary


class PoolMonitor:
    """Real-time connection pool monitoring"""
    
    def __init__(self, service_container):
        self.container = service_container
        self.metrics = PoolMetrics()
        self.redis_client: Optional[redis.Redis] = None
        self._monitoring = False
        self._monitor_task = None
        
        # Alert thresholds
        self.thresholds = {
            'utilization_warning': 70,  # %
            'utilization_critical': 90,  # %
            'failure_rate_warning': 5,  # %
            'failure_rate_critical': 15,  # %
        }
        
    async def initialize(self):
        """Initialize pool monitor"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=46379,
                password='cache-secret-key',
                decode_responses=True,
                db=2  # Use db 2 for metrics
            )
            await self.redis_client.ping()
            logger.info("✅ PoolMonitor initialized with Redis metrics storage")
        except Exception as e:
            logger.warning(f"Redis not available for metrics, using in-memory: {e}")
            self.redis_client = None
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Start periodic pool monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval_seconds))
        logger.info(f"🔍 Pool monitoring started (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Stop pool monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Pool monitoring stopped")
    
    async def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await self._store_metrics()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_metrics(self):
        """Collect current pool metrics"""
        if not hasattr(self.container, 'connection_pools'):
            return
            
        for service, pool_data in self.container.connection_pools.items():
            self.metrics.update_pool_utilization(
                service,
                pool_data['active'],
                pool_data['max_size']
            )
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        summary = self.metrics.get_summary()
        
        for service, data in summary['services'].items():
            utilization = data['pool']['utilization_percent']
            success_rate = data['requests']['success_rate_percent']
            failure_rate = 100 - success_rate
            
            # Utilization alerts
            if utilization >= self.thresholds['utilization_critical']:
                logger.error(f"🚨 CRITICAL: {service} pool utilization {utilization}% (threshold: {self.thresholds['utilization_critical']}%)")
                await self._send_alert('critical', f"{service} pool utilization critical: {utilization}%")
            elif utilization >= self.thresholds['utilization_warning']:
                logger.warning(f"⚠️ WARNING: {service} pool utilization {utilization}% (threshold: {self.thresholds['utilization_warning']}%)")
                await self._send_alert('warning', f"{service} pool utilization warning: {utilization}%")
            
            # Failure rate alerts
            if failure_rate >= self.thresholds['failure_rate_critical']:
                logger.error(f"🚨 CRITICAL: {service} failure rate {failure_rate}% (threshold: {self.thresholds['failure_rate_critical']}%)")
                await self._send_alert('critical', f"{service} failure rate critical: {failure_rate}%")
            elif failure_rate >= self.thresholds['failure_rate_warning']:
                logger.warning(f"⚠️ WARNING: {service} failure rate {failure_rate}% (threshold: {self.thresholds['failure_rate_warning']}%)")
                await self._send_alert('warning', f"{service} failure rate warning: {failure_rate}%")
    
    async def _send_alert(self, level: str, message: str):
        """Send alert (store in Redis for now)"""
        if self.redis_client:
            try:
                alert_data = {
                    'level': level,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    'service': 'mcp_connection_pools'
                }
                
                await self.redis_client.lpush('alerts', str(alert_data))
                await self.redis_client.ltrim('alerts', 0, 99)  # Keep last 100 alerts
                
            except Exception as e:
                logger.error(f"Failed to store alert: {e}")
    
    async def _store_metrics(self):
        """Store metrics in Redis"""
        if self.redis_client:
            try:
                summary = self.metrics.get_summary()
                await self.redis_client.setex(
                    'pool_metrics:current',
                    300,  # 5 minute expiry
                    str(summary)
                )
                
                # Store historical data
                timestamp = int(time.time())
                await self.redis_client.zadd(
                    'pool_metrics:history',
                    {str(summary): timestamp}
                )
                
                # Keep only last 24 hours of history
                cutoff = timestamp - (24 * 3600)
                await self.redis_client.zremrangebyscore('pool_metrics:history', 0, cutoff)
                
            except Exception as e:
                logger.error(f"Failed to store metrics: {e}")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current pool metrics"""
        return self.metrics.get_summary()
    
    async def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        if self.redis_client:
            try:
                alerts = await self.redis_client.lrange('alerts', 0, limit - 1)
                return [eval(alert) for alert in alerts]  # Note: eval is safe here since we control the data
            except Exception as e:
                logger.error(f"Failed to get alerts: {e}")
        return []
    
    def record_connection_event(self, service: str, success: bool, hold_time: float = 0):
        """Record connection event for metrics"""
        self.metrics.record_connection_request(service, success, hold_time)