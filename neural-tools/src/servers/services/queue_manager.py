"""
Queue Management Service with Backpressure Control

Provides intelligent queue management and backpressure mechanisms:
- Queue depth monitoring and alerts
- Backpressure thresholds to prevent system overload
- Job enqueue controls with rate limiting
- Queue health metrics for operational insights
"""

import logging
import os
import time
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class QueueManager:
    """
    Manages ARQ job queue health and applies backpressure when needed
    
    Features:
    - Real-time queue depth monitoring
    - Configurable backpressure thresholds
    - Graceful degradation under load
    - Operational metrics for monitoring
    """
    
    def __init__(self, service_container):
        self.container = service_container
        
        # Configuration from environment
        self.max_queue_depth = int(os.getenv('MAX_QUEUE_DEPTH', 1000))
        self.backpressure_threshold = int(os.getenv('BACKPRESSURE_THRESHOLD', 800))
        self.critical_threshold = int(os.getenv('CRITICAL_THRESHOLD', 950))
        
        # Metrics tracking
        self._last_depth_check = 0
        self._cached_depth = 0
        self._cache_ttl = 5  # Cache depth for 5 seconds
        
    async def get_queue_depth(self) -> int:
        """
        Get current queue depth from ARQ with caching for performance
        
        Returns:
            Current number of jobs in queue
        """
        try:
            # Use cached value if recent
            current_time = time.time()
            if current_time - self._last_depth_check < self._cache_ttl:
                return self._cached_depth
            
            redis_queue = await self.container.get_redis_queue_client()
            
            # ARQ uses multiple Redis keys for different job states
            # Check the main pending queue
            pending_jobs = await redis_queue.llen('arq:queue:default')
            
            # Also check deferred jobs (scheduled for later)
            try:
                deferred_jobs = await redis_queue.zcard('arq:deferred:default')
            except Exception:
                deferred_jobs = 0  # Fallback if key doesn't exist
            
            total_depth = pending_jobs + deferred_jobs
            
            # Update cache
            self._cached_depth = total_depth
            self._last_depth_check = current_time
            
            return total_depth
            
        except Exception as e:
            logger.error(f"Failed to get queue depth: {e}")
            # Return last known value or 0
            return self._cached_depth
    
    async def get_active_workers(self) -> int:
        """
        Get count of active worker processes
        
        Returns:
            Number of currently active workers
        """
        try:
            redis_queue = await self.container.get_redis_queue_client()
            
            # ARQ stores worker heartbeats in a sorted set
            active_workers = await redis_queue.zcard('arq:health-check')
            return active_workers
            
        except Exception as e:
            logger.error(f"Failed to get active worker count: {e}")
            return 0
    
    async def should_apply_backpressure(self) -> Tuple[bool, str, str]:
        """
        Check if backpressure should be applied based on queue health
        
        Returns:
            Tuple of (should_throttle, severity_level, reason)
        """
        try:
            current_depth = await self.get_queue_depth()
            active_workers = await self.get_active_workers()
            
            # Critical overload - reject requests
            if current_depth >= self.max_queue_depth:
                return True, "critical", f"Queue full ({current_depth}/{self.max_queue_depth})"
            
            # Severe pressure - strong backpressure
            elif current_depth >= self.critical_threshold:
                return True, "severe", f"Queue critically high ({current_depth}/{self.max_queue_depth})"
            
            # Moderate pressure - light backpressure
            elif current_depth >= self.backpressure_threshold:
                return True, "moderate", f"Queue under pressure ({current_depth}/{self.max_queue_depth})"
            
            # Check worker availability
            elif active_workers == 0 and current_depth > 0:
                return True, "no-workers", f"No active workers available (queue: {current_depth})"
            
            else:
                return False, "healthy", f"Queue healthy ({current_depth}/{self.max_queue_depth}, workers: {active_workers})"
                
        except Exception as e:
            logger.error(f"Failed to check backpressure: {e}")
            # Fail safe - apply backpressure if we can't determine state
            return True, "unknown", f"Unable to determine queue state: {e}"
    
    async def enqueue_with_backpressure(
        self, 
        job_func: str, 
        *args, 
        priority: str = "normal",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enqueue job with intelligent backpressure control
        
        Args:
            job_func: Function name to execute
            *args: Function arguments
            priority: Job priority (high, normal, low)
            **kwargs: Additional job parameters
            
        Returns:
            Dictionary with enqueue result and status
        """
        try:
            should_throttle, severity, reason = await self.should_apply_backpressure()
            
            # Handle backpressure based on priority and severity
            if should_throttle:
                # High priority jobs get preferential treatment
                if priority == "high" and severity != "critical":
                    logger.warning(f"Allowing high priority job despite backpressure: {reason}")
                else:
                    # Calculate retry delay based on severity
                    retry_delays = {
                        "moderate": 30,    # 30 seconds
                        "severe": 60,      # 1 minute
                        "critical": 300,   # 5 minutes
                        "no-workers": 120, # 2 minutes
                        "unknown": 60      # 1 minute
                    }
                    
                    return {
                        "status": "throttled",
                        "severity": severity,
                        "error": "System overloaded", 
                        "reason": reason,
                        "retry_after": retry_delays.get(severity, 60)
                    }
            
            # Proceed with normal enqueue
            job_queue = await self.container.get_job_queue()
            
            # Set job priority if supported
            job_kwargs = kwargs.copy()
            if priority == "high":
                job_kwargs['_defer_by'] = -1  # Negative defer for higher priority
            elif priority == "low":
                job_kwargs['_defer_by'] = 10   # Small delay for lower priority
            
            job = await job_queue.enqueue_job(job_func, *args, **job_kwargs)
            
            current_depth = await self.get_queue_depth()
            
            logger.info(f"Job {job.job_id} enqueued successfully (priority: {priority}, queue depth: {current_depth})")
            
            return {
                "status": "queued",
                "job_id": job.job_id,
                "priority": priority,
                "queue_position": current_depth,
                "estimated_delay": self._estimate_processing_delay(current_depth)
            }
            
        except Exception as e:
            logger.error(f"Failed to enqueue job: {e}")
            return {
                "status": "error",
                "error": "Internal queue error",
                "message": str(e)
            }
    
    def _estimate_processing_delay(self, queue_depth: int) -> int:
        """
        Estimate processing delay based on queue depth
        
        Args:
            queue_depth: Current queue depth
            
        Returns:
            Estimated delay in seconds
        """
        # Assume average processing time of 30 seconds per job
        avg_processing_time = 30
        
        # Assume we have at least 1 worker, up to 10
        estimated_workers = min(10, max(1, queue_depth // 100))
        
        estimated_delay = (queue_depth * avg_processing_time) // estimated_workers
        
        return max(0, estimated_delay)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive queue statistics for monitoring
        
        Returns:
            Dictionary with queue health metrics
        """
        try:
            current_depth = await self.get_queue_depth()
            active_workers = await self.get_active_workers()
            should_throttle, severity, reason = await self.should_apply_backpressure()
            
            redis_queue = await self.container.get_redis_queue_client()
            
            # Get additional Redis stats
            redis_info = await redis_queue.info('memory')
            memory_usage = redis_info.get('used_memory', 0)
            
            return {
                "queue_depth": current_depth,
                "max_queue_depth": self.max_queue_depth,
                "backpressure_threshold": self.backpressure_threshold,
                "critical_threshold": self.critical_threshold,
                "active_workers": active_workers,
                "backpressure_status": {
                    "enabled": should_throttle,
                    "severity": severity,
                    "reason": reason
                },
                "estimated_delay_seconds": self._estimate_processing_delay(current_depth),
                "queue_utilization": current_depth / self.max_queue_depth,
                "redis_memory_bytes": memory_usage,
                "health_score": self._calculate_health_score(current_depth, active_workers)
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {
                "error": str(e),
                "queue_depth": -1
            }
    
    def _calculate_health_score(self, queue_depth: int, active_workers: int) -> float:
        """
        Calculate overall queue health score (0.0 to 1.0)
        
        Args:
            queue_depth: Current queue depth
            active_workers: Number of active workers
            
        Returns:
            Health score between 0.0 (unhealthy) and 1.0 (healthy)
        """
        # Base score from queue utilization (inverted - lower is better)
        utilization = queue_depth / self.max_queue_depth
        queue_score = max(0.0, 1.0 - utilization)
        
        # Worker availability score
        worker_score = 1.0 if active_workers > 0 else 0.0
        
        # Weighted average (queue health is more important)
        health_score = (queue_score * 0.7) + (worker_score * 0.3)
        
        return round(health_score, 3)
    
    async def clear_old_completed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed jobs to prevent memory bloat
        
        Args:
            max_age_hours: Maximum age of jobs to keep in hours
            
        Returns:
            Number of jobs cleaned up
        """
        try:
            redis_queue = await self.container.get_redis_queue_client()
            
            cutoff_timestamp = int(time.time()) - (max_age_hours * 3600)
            
            # Clean up completed jobs older than cutoff
            removed_count = await redis_queue.zremrangebyscore(
                'arq:result',
                0,
                cutoff_timestamp
            )
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old completed jobs (older than {max_age_hours}h)")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to clean up old jobs: {e}")
            return 0