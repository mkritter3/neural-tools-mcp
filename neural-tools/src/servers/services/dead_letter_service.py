"""
Dead Letter Queue Service using Redis Streams

Provides robust error handling and failed job management:
- Redis Streams for persistent error storage
- Error classification and statistics
- Consumer groups for different error types
- Administrative recovery interfaces
"""

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DeadLetterService:
    """
    Redis Streams-based Dead Letter Queue for failed job management
    
    Features:
    - Persistent error storage with automatic trimming
    - Error classification and retry policy application
    - Statistics and monitoring for operational insights
    - Consumer group support for distributed processing
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.dlq_stream = "l9:prod:neural_tools:embedding_failures"
        self.consumer_group = "dlq_processors"
        self.max_stream_length = 100000  # Keep last 100k failed jobs
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize DLQ consumer group if it doesn't exist"""
        try:
            # Try to create consumer group (idempotent operation)
            try:
                await self.redis.xgroup_create(
                    self.dlq_stream, 
                    self.consumer_group,
                    id='0',  # Start from beginning
                    mkstream=True  # Create stream if it doesn't exist
                )
                logger.info(f"Created DLQ consumer group: {self.consumer_group}")
            except Exception as e:
                # Group likely already exists, which is fine
                if "BUSYGROUP" not in str(e):
                    logger.warning(f"Error creating consumer group: {e}")
            
            return {"success": True, "message": "DLQ service initialized"}
            
        except Exception as e:
            logger.error(f"Failed to initialize DLQ service: {e}")
            return {"success": False, "message": str(e)}
    
    async def add_to_dlq(self, job_data: Dict[str, Any], error: Exception, retry_count: int = 0) -> str:
        """
        Add failed job to dead letter queue with comprehensive error context
        
        Args:
            job_data: Original job data including job_id, text, model
            error: Exception that caused the failure
            retry_count: Number of retry attempts made
            
        Returns:
            Stream ID of the added DLQ entry
        """
        try:
            # Classify error type for retry policy application
            error_classification = self._classify_error(error)
            
            # Redis Streams require string values - convert all fields
            stream_data = {
                'job_id': str(job_data.get('job_id', 'unknown')),
                'original_text': str(job_data.get('text', ''))[:1000],  # Truncate long text
                'text_length': str(len(str(job_data.get('text', '')))),
                'model': str(job_data.get('model', 'nomic-v2')),
                'error_type': error_classification['type'],
                'error_category': error_classification['category'],
                'error_message': str(error)[:500],  # Truncate long errors
                'retry_count': str(retry_count),
                'max_retries': str(error_classification['max_retries']),
                'requires_manual': str(error_classification['requires_manual']),
                'timestamp': str(int(time.time())),
                'environment': os.getenv('ENVIRONMENT', 'prod'),
                'service': 'neural_tools',
                'project': os.getenv('PROJECT_NAME', 'default')
            }
            
            # Add to stream with automatic trimming to prevent unbounded growth
            stream_id = await self.redis.xadd(
                self.dlq_stream,
                stream_data,
                maxlen=self.max_stream_length,
                approximate=True  # Use ~ for efficiency
            )
            
            logger.warning(f"Job {job_data.get('job_id')} added to DLQ with stream ID {stream_id} "
                         f"(error: {error_classification['type']}, retries: {retry_count})")
            
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to add job to DLQ: {e}")
            # Don't raise - DLQ failures shouldn't break the main flow
            return None
    
    def _classify_error(self, error: Exception) -> Dict[str, Any]:
        """
        Classify error and determine retry policy
        
        Args:
            error: Exception to classify
            
        Returns:
            Dictionary with error classification and retry policy
        """
        error_type = type(error).__name__
        
        # Error classification policies
        ERROR_POLICIES = {
            'ConnectionError': {
                'category': 'transient',
                'max_retries': 5,
                'requires_manual': False,
                'description': 'Network connectivity issue'
            },
            'TimeoutError': {
                'category': 'transient', 
                'max_retries': 3,
                'requires_manual': False,
                'description': 'Service timeout'
            },
            'HTTPError': {
                'category': 'service',
                'max_retries': 3,
                'requires_manual': False,
                'description': 'HTTP service error'
            },
            'ValidationError': {
                'category': 'permanent',
                'max_retries': 0,
                'requires_manual': True,
                'description': 'Input validation failure'
            },
            'ValueError': {
                'category': 'permanent',
                'max_retries': 0,
                'requires_manual': True,
                'description': 'Invalid input data'
            },
            'AuthenticationError': {
                'category': 'configuration',
                'max_retries': 1,
                'requires_manual': True,
                'description': 'Authentication failure'
            }
        }
        
        policy = ERROR_POLICIES.get(error_type, {
            'category': 'unknown',
            'max_retries': 1,
            'requires_manual': True,
            'description': 'Unknown error type'
        })
        
        return {
            'type': error_type,
            'category': policy['category'],
            'max_retries': policy['max_retries'],
            'requires_manual': policy['requires_manual'],
            'description': policy['description']
        }
    
    async def get_dlq_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive DLQ statistics for monitoring and alerting
        
        Returns:
            Dictionary with DLQ metrics and error analysis
        """
        try:
            # Basic stream information
            stream_length = await self.redis.xlen(self.dlq_stream)
            
            if stream_length == 0:
                return {
                    'total_failures': 0,
                    'error_type_distribution': {},
                    'error_category_distribution': {},
                    'recent_sample_size': 0,
                    'requires_manual_count': 0,
                    'avg_retry_count': 0
                }
            
            # Get recent entries for analysis (last 1000 or all if fewer)
            sample_size = min(1000, stream_length)
            recent_entries = await self.redis.xrevrange(self.dlq_stream, count=sample_size)
            
            # Analyze error patterns
            error_types = {}
            error_categories = {}
            manual_required = 0
            retry_counts = []
            
            for stream_id, fields in recent_entries:
                # Error type distribution
                error_type = fields.get('error_type', 'Unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                # Error category distribution  
                error_category = fields.get('error_category', 'unknown')
                error_categories[error_category] = error_categories.get(error_category, 0) + 1
                
                # Manual intervention tracking
                if fields.get('requires_manual', 'false').lower() == 'true':
                    manual_required += 1
                    
                # Retry count analysis
                try:
                    retry_count = int(fields.get('retry_count', 0))
                    retry_counts.append(retry_count)
                except (ValueError, TypeError):
                    pass
            
            # Calculate averages
            avg_retry_count = sum(retry_counts) / len(retry_counts) if retry_counts else 0
            
            return {
                'total_failures': stream_length,
                'error_type_distribution': error_types,
                'error_category_distribution': error_categories,
                'recent_sample_size': len(recent_entries),
                'requires_manual_count': manual_required,
                'manual_intervention_rate': manual_required / len(recent_entries) if recent_entries else 0,
                'avg_retry_count': round(avg_retry_count, 2),
                'stream_name': self.dlq_stream
            }
            
        except Exception as e:
            logger.error(f"Failed to get DLQ stats: {e}")
            return {
                'error': str(e),
                'total_failures': -1
            }
    
    async def get_failed_jobs(self, limit: int = 100, error_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve failed jobs for manual review and recovery
        
        Args:
            limit: Maximum number of jobs to return
            error_type: Filter by specific error type (optional)
            
        Returns:
            List of failed job entries with full context
        """
        try:
            entries = await self.redis.xrevrange(self.dlq_stream, count=limit)
            
            failed_jobs = []
            for stream_id, fields in entries:
                # Filter by error type if specified
                if error_type and fields.get('error_type') != error_type:
                    continue
                
                job_entry = {
                    'stream_id': stream_id,
                    'job_id': fields.get('job_id'),
                    'original_text': fields.get('original_text'),
                    'text_length': fields.get('text_length'),
                    'model': fields.get('model'),
                    'error_type': fields.get('error_type'),
                    'error_category': fields.get('error_category'),
                    'error_message': fields.get('error_message'),
                    'retry_count': fields.get('retry_count'),
                    'requires_manual': fields.get('requires_manual', 'false').lower() == 'true',
                    'timestamp': int(fields.get('timestamp', 0)),
                    'project': fields.get('project'),
                    'environment': fields.get('environment')
                }
                failed_jobs.append(job_entry)
                
            return failed_jobs
            
        except Exception as e:
            logger.error(f"Failed to retrieve failed jobs: {e}")
            return []
    
    async def requeue_failed_job(self, stream_id: str) -> Dict[str, Any]:
        """
        Requeue a failed job for reprocessing
        
        Args:
            stream_id: Redis stream ID of the failed job
            
        Returns:
            Dictionary with requeue result
        """
        try:
            # Get the failed job entry
            entries = await self.redis.xrange(self.dlq_stream, min=stream_id, max=stream_id)
            
            if not entries:
                return {"success": False, "message": f"Job with stream ID {stream_id} not found"}
            
            stream_id, fields = entries[0]
            
            # Extract job data for requeuing
            job_data = {
                'text': fields.get('original_text', ''),
                'model': fields.get('model', 'nomic-v2')
            }
            
            # Get job queue and requeue
            from servers.services.service_container import ServiceContainer
            container = ServiceContainer()
            job_queue = await container.get_job_queue()
            
            new_job = await job_queue.enqueue_job(
                'process_embedding_job',
                job_data['text'],
                job_data['model'],
                _job_timeout=300
            )
            
            logger.info(f"Requeued failed job {fields.get('job_id')} as new job {new_job.job_id}")
            
            return {
                "success": True,
                "message": f"Job requeued successfully",
                "original_job_id": fields.get('job_id'),
                "new_job_id": new_job.job_id
            }
            
        except Exception as e:
            logger.error(f"Failed to requeue job {stream_id}: {e}")
            return {"success": False, "message": str(e)}
    
    async def close(self):
        """Cleanup resources (placeholder for future use)"""
        pass