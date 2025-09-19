#!/usr/bin/env python3
"""
Neural Tools Metrics Exporter for Prometheus
Implements ADR-053 monitoring requirements
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

from prometheus_client import (
    start_http_server,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)
from aiohttp import web
from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics definitions
sync_total = Counter('neural_tools_sync_total', 'Total number of sync operations')
sync_successful = Counter('neural_tools_sync_successful', 'Number of successful sync operations')
sync_failed = Counter('neural_tools_sync_failed', 'Number of failed sync operations')
sync_rollbacks = Counter('neural_tools_sync_rollbacks', 'Number of sync rollbacks')
sync_duration = Histogram('neural_tools_sync_duration_seconds', 'Duration of sync operations')

neo4j_chunk_count = Gauge('neural_tools_neo4j_chunk_count', 'Number of chunks in Neo4j')
qdrant_point_count = Gauge('neural_tools_qdrant_point_count', 'Number of points in Qdrant')
orphaned_chunks = Gauge('neural_tools_orphaned_chunks', 'Number of orphaned chunks')

indexer_health = Gauge('neural_tools_indexer_health', 'Indexer health status (1=healthy, 0=unhealthy)')
sync_manager_health = Gauge('neural_tools_sync_manager_health', 'Sync manager health (1=healthy, 0=unhealthy)')
indexer_queue_depth = Gauge('neural_tools_indexer_queue_depth', 'Number of items in indexer queue')

deployment_info = Gauge('neural_tools_deployment_info', 'Deployment information', ['version', 'commit'])


class MetricsCollector:
    """Collects metrics from various neural-tools components"""

    def __init__(self):
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://host.docker.internal:47687')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'graphrag-password')
        self.qdrant_host = os.getenv('QDRANT_HOST', 'host.docker.internal')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '46333'))
        self.redis_host = os.getenv('REDIS_HOST', 'host.docker.internal')
        self.redis_port = int(os.getenv('REDIS_PORT', '46379'))

        self.neo4j_driver = None
        self.qdrant_client = None
        self.redis_client = None

    async def initialize(self):
        """Initialize connections to services"""
        try:
            # Neo4j connection
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=("neo4j", self.neo4j_password)
            )
            await self.neo4j_driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.neo4j_uri}")

            # Qdrant connection
            self.qdrant_client = AsyncQdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            collections = await self.qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")

            # Redis connection
            self.redis_client = await redis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}"
            )
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")

        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise

    async def collect_neo4j_metrics(self):
        """Collect metrics from Neo4j"""
        try:
            async with self.neo4j_driver.session() as session:
                # Count total chunks
                result = await session.run(
                    "MATCH (c:CHUNK) RETURN count(c) as count"
                )
                record = await result.single()
                chunk_count = record["count"] if record else 0
                neo4j_chunk_count.set(chunk_count)

                # Check for sync metrics stored in Neo4j
                result = await session.run("""
                    MATCH (m:SyncMetric)
                    WHERE m.timestamp > datetime() - duration('PT5M')
                    RETURN
                        sum(m.successful) as successful,
                        sum(m.failed) as failed,
                        sum(m.rollbacks) as rollbacks
                """)
                record = await result.single()
                if record:
                    sync_successful._value.set(record["successful"] or 0)
                    sync_failed._value.set(record["failed"] or 0)
                    sync_rollbacks._value.set(record["rollbacks"] or 0)

        except Exception as e:
            logger.error(f"Failed to collect Neo4j metrics: {e}")

    async def collect_qdrant_metrics(self):
        """Collect metrics from Qdrant"""
        try:
            # Get all collections
            collections = await self.qdrant_client.get_collections()
            total_points = 0

            for collection in collections.collections:
                info = await self.qdrant_client.get_collection(collection.name)
                total_points += info.points_count

            qdrant_point_count.set(total_points)

        except Exception as e:
            logger.error(f"Failed to collect Qdrant metrics: {e}")

    async def collect_redis_metrics(self):
        """Collect metrics from Redis queues"""
        try:
            # Check indexer queue depth
            queue_length = await self.redis_client.llen("indexer:queue")
            indexer_queue_depth.set(queue_length)

            # Check sync metrics in Redis
            sync_stats = await self.redis_client.hgetall("sync:stats")
            if sync_stats:
                if b'total' in sync_stats:
                    sync_total._value.set(int(sync_stats[b'total']))
                if b'success_rate' in sync_stats:
                    # Calculate successful count from rate
                    rate = float(sync_stats[b'success_rate'])
                    total = int(sync_stats.get(b'total', 0))
                    successful = int(total * rate / 100)
                    sync_successful._value.set(successful)

        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")

    async def check_data_consistency(self):
        """Check for data consistency between Neo4j and Qdrant"""
        try:
            # Get counts
            neo4j_count = neo4j_chunk_count._value.get()
            qdrant_count = qdrant_point_count._value.get()

            # Calculate drift
            drift = abs(neo4j_count - qdrant_count)

            # Check for orphaned chunks (simplified check)
            if drift > 0:
                orphaned_chunks.set(drift)
            else:
                orphaned_chunks.set(0)

        except Exception as e:
            logger.error(f"Failed to check data consistency: {e}")

    async def check_service_health(self):
        """Check health of various services"""
        try:
            # Check indexer health
            # This would normally check the actual indexer service
            # For now, we'll check if Redis queue is processing
            queue_depth = indexer_queue_depth._value.get()
            if queue_depth < 1000:  # Healthy if queue not too deep
                indexer_health.set(1)
            else:
                indexer_health.set(0)

            # Check sync manager health based on sync rate
            total = sync_total._value.get()
            successful = sync_successful._value.get()
            if total > 0:
                rate = (successful / total) * 100
                if rate >= 95:  # ADR-053 requirement
                    sync_manager_health.set(1)
                else:
                    sync_manager_health.set(0)
            else:
                sync_manager_health.set(1)  # No syncs yet, assume healthy

        except Exception as e:
            logger.error(f"Failed to check service health: {e}")

    async def collect_all_metrics(self):
        """Collect all metrics"""
        try:
            await self.collect_neo4j_metrics()
            await self.collect_qdrant_metrics()
            await self.collect_redis_metrics()
            await self.check_data_consistency()
            await self.check_service_health()

            # Set deployment info (static)
            deployment_info.labels(
                version=os.getenv('DEPLOYMENT_VERSION', 'unknown'),
                commit=os.getenv('DEPLOYMENT_COMMIT', 'unknown')
            ).set(1)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def cleanup(self):
        """Clean up connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.qdrant_client:
            await self.qdrant_client.close()
        if self.redis_client:
            await self.redis_client.close()


async def metrics_handler(request):
    """HTTP handler for /metrics endpoint"""
    metrics = generate_latest()
    return web.Response(body=metrics, content_type=CONTENT_TYPE_LATEST)


async def health_handler(request):
    """HTTP handler for /health endpoint"""
    return web.Response(text="OK", status=200)


async def background_collector(collector: MetricsCollector):
    """Background task to collect metrics periodically"""
    while True:
        try:
            await collector.collect_all_metrics()
            logger.info("Metrics collected successfully")
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")

        # Wait 15 seconds before next collection
        await asyncio.sleep(15)


async def main():
    """Main entry point"""
    # Initialize collector
    collector = MetricsCollector()
    await collector.initialize()

    # Start background collection
    asyncio.create_task(background_collector(collector))

    # Create web app for metrics endpoint
    app = web.Application()
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/health', health_handler)

    # Start server
    port = int(os.getenv('METRICS_PORT', '9200'))
    logger.info(f"Starting metrics exporter on port {port}")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()

    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter")
    finally:
        await collector.cleanup()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())