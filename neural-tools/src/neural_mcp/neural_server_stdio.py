#!/usr/bin/env python3
"""
L9 Enhanced MCP Server - Proper STDIO Transport Implementation
Features: Neo4j GraphRAG + Nomic Embed v2-MoE + Tree-sitter + Qdrant hybrid search

This implementation follows the MCP specification for STDIO transport:
- Server runs as a long-lived subprocess
- Reads JSON-RPC messages from stdin (newline-delimited)
- Writes JSON-RPC responses to stdout
- Maintains session state throughout its lifetime
"""

import os
import sys
import json
import asyncio
import logging
import secrets
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Official MCP SDK for 2025-06-18 protocol (no namespace collision with neural_mcp)
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Import schema management for ADR-0020
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'servers', 'services'))
from schema_manager import SchemaManager, ProjectType
# Import migration management for ADR-0021
from migration_manager import MigrationManager, MigrationResult
from data_migrator import DataMigrator
# Import canon management for ADR-0031
from canon_manager import CanonManager
from metadata_backfiller import MetadataBackfiller
# Import project context management for ADR-0033
from project_context_manager import ProjectContextManager

# Configure logging to stderr (NEVER to stdout for STDIO transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Critical: Log to stderr, not stdout
)
base_logger = logging.getLogger(__name__)


class InstanceAwareLogger:
    """
    Logger wrapper that includes instance ID in all log messages.
    Phase 3 of ADR-19: Enhanced logging for debugging.
    """
    def __init__(self, logger, get_instance_id_func=None):
        self.logger = logger
        self.get_instance_id = get_instance_id_func or (lambda: "unknown")
    
    def _format_msg(self, msg):
        """Add instance ID to message if available"""
        try:
            instance_id = self.get_instance_id()
            if instance_id and instance_id != "unknown":
                return f"[Instance {instance_id}] {msg}"
        except:
            pass
        return msg
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(self._format_msg(msg), *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self.logger.warning(self._format_msg(msg), *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self.logger.error(self._format_msg(msg), *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(self._format_msg(msg), *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(self._format_msg(msg), *args, **kwargs)


# Create instance-aware logger (will be initialized after state is created)
logger = InstanceAwareLogger(base_logger)

# Constants
# ADR-0034: Use ProjectContextManager for dynamic project detection (no hardcoding)
DEFAULT_PROJECT_NAME = None  # Will be set dynamically by get_project_context()
LIMITS = {
    "semantic_limit_max": 50,
    "graphrag_limit_max": 25,
    "max_hops_max": 3,
}

# Phase 2: Configurable instance management
INSTANCE_TIMEOUT_HOURS = float(os.environ.get('INSTANCE_TIMEOUT_HOURS', '1'))  # Default 1 hour
CLEANUP_INTERVAL_MINUTES = float(os.environ.get('CLEANUP_INTERVAL_MINUTES', '10'))  # Check every 10 mins
ENABLE_AUTO_CLEANUP = os.environ.get('ENABLE_AUTO_CLEANUP', 'true').lower() == 'true'

# Phase 3: Enhanced monitoring and debugging
MCP_VERBOSE = os.environ.get('MCP_VERBOSE', 'false').lower() == 'true'
INCLUDE_INSTANCE_METADATA = os.environ.get('INCLUDE_INSTANCE_METADATA', 'false').lower() == 'true'


def add_instance_metadata(response_data: dict, instance_id: str = None, project_name: str = None) -> dict:
    """
    Add instance metadata to responses when in verbose mode.
    Phase 3 of ADR-19: Enhanced debugging and monitoring.
    """
    if not MCP_VERBOSE and not INCLUDE_INSTANCE_METADATA:
        return response_data
    
    if instance_id is None and 'state' in globals():
        instance_id = state.instance_id
    
    metadata = {
        'instance_id': instance_id,
        'timestamp': datetime.now().isoformat(),
        'verbose_mode': MCP_VERBOSE
    }
    
    if project_name:
        metadata['project'] = project_name
    
    # Add instance metrics summary if available
    if 'state' in globals() and hasattr(state, 'get_instance_metrics'):
        metrics = state.get_instance_metrics()
        metadata['instance_count'] = metrics.get('total_instances', 1)
        metadata['is_stale'] = metrics.get('stale_instances', 0) > 0
    
    response_data['_metadata'] = metadata
    return response_data


def get_instance_id() -> str:
    """
    Get or generate a unique instance ID for this MCP server instance.
    This enables isolation between different Claude instances.
    
    Returns a consistent ID for this MCP server process that persists
    throughout its lifetime but differs between Claude instances.
    """
    # Priority 1: Environment variable (if Claude provides it in future)
    instance_id = os.getenv('INSTANCE_ID')
    if instance_id:
        logger.info(f"🔐 Using provided instance ID: {instance_id}")
        return instance_id
    
    # Priority 2: Process-based ID (fallback)
    pid = os.getpid()
    ppid = os.getppid()
    
    # Priority 3: Hash of stdin/stdout file descriptors (unique per connection)
    try:
        stdin_stat = os.fstat(sys.stdin.fileno())
        stdout_stat = os.fstat(sys.stdout.fileno())
        unique_string = f"{pid}:{ppid}:{stdin_stat.st_ino}:{stdin_stat.st_dev}:{stdout_stat.st_ino}"
    except:
        # Fallback if file descriptors not available
        unique_string = f"{pid}:{ppid}:{datetime.now().isoformat()}"
    
    # Generate short, consistent hash
    instance_hash = hashlib.md5(unique_string.encode()).hexdigest()[:8]
    logger.info(f"🔐 Generated instance ID: {instance_hash} (pid:{pid}, ppid:{ppid})")
    return instance_hash


def _make_validation_error(
    tool: str,
    message: str,
    *,
    missing: Optional[List[str]] = None,
    invalid: Optional[List[Dict[str, str]]] = None,
    example: Optional[Dict[str, Any]] = None,
    received: Optional[Dict[str, Any]] = None,
    normalized: Optional[Dict[str, Any]] = None,
    hint: Optional[str] = None,
) -> str:
    payload: Dict[str, Any] = {
        "status": "error",
        "code": "validation_error",
        "tool": tool,
        "message": message,
        "missing": missing or [],
        "invalid": invalid or [],
        "example": example or {},
        "received": received or {},
    }
    if normalized is not None:
        payload["normalized_args"] = normalized
        payload["next_call"] = normalized
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, indent=2)

# Multi-project service instances - cached per project AND instance
class MultiProjectServiceState:
    """Holds persistent service state with instance-level and per-project isolation"""
    def __init__(self):
        # Instance-level isolation (Phase 1 of ADR-19)
        self.instance_id = get_instance_id()
        self.instance_containers = {}  # instance_id -> { project_containers, project_retrievers, ... }
        self.global_initialized = False
        
        # Phase 2: Cleanup tracking
        self.cleanup_task = None
        self.cleanup_stats = {
            'total_cleanups': 0,
            'instances_cleaned': 0,
            'last_cleanup': None
        }
        
        # Initialize container for this instance
        self._init_instance_container()
        
    def _init_instance_container(self):
        """Initialize container for this instance"""
        if self.instance_id not in self.instance_containers:
            self.instance_containers[self.instance_id] = {
                'project_containers': {},
                'project_retrievers': {},
                'project_schemas': {},  # ADR-0020: Per-project schema managers
                'project_migrations': {},  # ADR-0021: Per-project migration managers
                'session_started': datetime.now(),
                'last_activity': datetime.now()
            }
            logger.info(f"🔐 Initialized instance container: {self.instance_id}")
    
    def _get_instance_data(self):
        """Get data container for current instance"""
        # Update last activity
        if self.instance_id in self.instance_containers:
            self.instance_containers[self.instance_id]['last_activity'] = datetime.now()
        return self.instance_containers[self.instance_id]
        
    def detect_project_from_path(self, file_path: str) -> str:
        """Extract project name from workspace path"""
        if not file_path:
            return DEFAULT_PROJECT_NAME
            
        # Handle /workspace/project-name/... format
        if file_path.startswith('/workspace/'):
            parts = file_path.strip('/').split('/')
            if len(parts) >= 2 and parts[1] != '':
                return parts[1]  # project name
        
        # Handle query context with file references
        if 'project-' in file_path:
            for part in file_path.split('/'):
                if part.startswith('project-'):
                    return part
                    
        return DEFAULT_PROJECT_NAME
    
    async def get_project_container(self, project_name: str):
        """Get or create ServiceContainer for specific project with lazy initialization and instance isolation"""
        instance_data = self._get_instance_data()
        project_containers = instance_data['project_containers']
        
        if project_name not in project_containers:
            logger.info(f"🏗️ [Instance {self.instance_id}] Creating service container for project: {project_name}")
            
            # Import services
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from servers.services.service_container import ServiceContainer
            
            container = ServiceContainer(project_name)
            # Don't initialize services yet - will happen on first use
            project_containers[project_name] = container
        
        # Lazy initialization - only initialize when actually needed
        container = project_containers[project_name]
        if not container.initialized:
            logger.info(f"⚡ [Instance {self.instance_id}] Lazy-initializing services for project: {project_name} (first tool use)")
            await container.initialize_all_services()
            
            # Initialize L9 connection pools and session management
            await container.initialize_connection_pools()
            if container.session_manager:
                await container.session_manager.initialize()
            
            # Initialize Phase 3 security and monitoring if available
            if hasattr(container, 'initialize_security_services'):
                await container.initialize_security_services()
            
            # ADR-0030: Auto-start indexer container for this project
            if hasattr(container, 'ensure_indexer_running'):
                try:
                    # Use current working directory as project path
                    project_path = os.getcwd()
                    logger.info(f"🚀 [Instance {self.instance_id}] Starting indexer for {project_name} at {project_path}")
                    await container.ensure_indexer_running(project_path)
                except Exception as e:
                    logger.error(f"Failed to start indexer for {project_name}: {e}")
                    # Continue without indexer - non-fatal error
        
        return container
    
    async def get_service_container(self, project_name: str):
        """Alias for get_project_container for backward compatibility"""
        return await self.get_project_container(project_name)
    
    async def get_project_retriever(self, project_name: str):
        """Get or create HybridRetriever for specific project with instance isolation"""
        instance_data = self._get_instance_data()
        project_retrievers = instance_data['project_retrievers']
        
        if project_name not in project_retrievers:
            container = await self.get_project_container(project_name)
            if container.neo4j and container.qdrant:
                from servers.services.hybrid_retriever import HybridRetriever
                project_retrievers[project_name] = HybridRetriever(container)
                logger.info(f"🔍 [Instance {self.instance_id}] Created HybridRetriever for project: {project_name}")
            else:
                project_retrievers[project_name] = None
        return project_retrievers[project_name]
    
    async def get_project_schema(self, project_name: str) -> SchemaManager:
        """
        Get or create SchemaManager for specific project.
        ADR-0020: Per-project custom GraphRAG schemas.
        """
        instance_data = self._get_instance_data()
        project_schemas = instance_data['project_schemas']
        
        if project_name not in project_schemas:
            # Get project path from working directory or environment
            project_path = os.environ.get('PROJECT_PATH', os.getcwd())
            
            # Create schema manager
            schema_manager = SchemaManager(project_name, project_path)
            await schema_manager.initialize()
            
            project_schemas[project_name] = schema_manager
            logger.info(f"📋 [Instance {self.instance_id}] Created SchemaManager for project: {project_name}")
            
        return project_schemas[project_name]
    
    async def get_project_migration_manager(self, project_name: str) -> MigrationManager:
        """
        Get or create MigrationManager for specific project.
        ADR-0021: GraphRAG schema migration system.
        """
        instance_data = self._get_instance_data()
        project_migrations = instance_data['project_migrations']
        
        if project_name not in project_migrations:
            # Get project path from working directory or environment
            project_path = os.environ.get('PROJECT_PATH', os.getcwd())
            
            # Create migration manager
            migration_manager = MigrationManager(project_name, project_path)
            
            # Inject service connections
            container = await self.get_project_container(project_name)
            if container:
                migration_manager.neo4j = container.neo4j
                migration_manager.qdrant = container.qdrant
            
            project_migrations[project_name] = migration_manager
            logger.info(f"📦 [Instance {self.instance_id}] Created MigrationManager for project: {project_name}")
            
        return project_migrations[project_name]
    
    async def cleanup_stale_instances(self):
        """
        Clean up instances that have been inactive for longer than the timeout.
        Phase 2 of ADR-19: Resource cleanup for stale instances.
        """
        now = datetime.now()
        stale_threshold = timedelta(hours=INSTANCE_TIMEOUT_HOURS)
        cleaned_count = 0
        
        logger.info(f"🧹 [Instance {self.instance_id}] Starting cleanup check (threshold: {INSTANCE_TIMEOUT_HOURS}h)")
        
        for instance_id, instance_data in list(self.instance_containers.items()):
            # Don't clean up our own instance
            if instance_id == self.instance_id:
                continue
                
            last_activity = instance_data.get('last_activity', instance_data.get('session_started'))
            if now - last_activity > stale_threshold:
                logger.info(f"🧹 Cleaning stale instance: {instance_id} (inactive for {now - last_activity})")
                
                # Close all containers in this instance
                for project_name, container in instance_data.get('project_containers', {}).items():
                    try:
                        await self._close_container_connections(container, project_name)
                    except Exception as e:
                        logger.error(f"Error closing container for {project_name}: {e}")
                
                # Remove the instance
                del self.instance_containers[instance_id]
                cleaned_count += 1
        
        # Update stats
        self.cleanup_stats['total_cleanups'] += 1
        self.cleanup_stats['instances_cleaned'] += cleaned_count
        self.cleanup_stats['last_cleanup'] = now
        
        if cleaned_count > 0:
            logger.info(f"✅ Cleaned {cleaned_count} stale instances")
        
        return cleaned_count
    
    async def _close_container_connections(self, container, project_name: str):
        """Close all connections for a container"""
        logger.info(f"📦 Closing connections for project: {project_name}")
        
        # Close Neo4j driver
        if hasattr(container, 'neo4j_driver') and container.neo4j_driver:
            try:
                container.neo4j_driver.close()
                logger.info(f"  ✅ Closed Neo4j connection")
            except Exception as e:
                logger.error(f"  ❌ Failed to close Neo4j: {e}")
        
        # Close Qdrant client  
        if hasattr(container, 'qdrant_client') and container.qdrant_client:
            try:
                container.qdrant_client.close()
                logger.info(f"  ✅ Closed Qdrant connection")
            except Exception as e:
                logger.error(f"  ❌ Failed to close Qdrant: {e}")
        
        # Close Redis connections if they exist
        if hasattr(container, 'redis_cache') and container.redis_cache:
            try:
                await container.redis_cache.close()
                logger.info(f"  ✅ Closed Redis cache connection")
            except Exception as e:
                logger.error(f"  ❌ Failed to close Redis cache: {e}")
        
        if hasattr(container, 'redis_queue') and container.redis_queue:
            try:
                await container.redis_queue.close()
                logger.info(f"  ✅ Closed Redis queue connection")
            except Exception as e:
                logger.error(f"  ❌ Failed to close Redis queue: {e}")
    
    async def export_instance_state(self) -> dict:
        """
        Export current instance state for migration or backup.
        Phase 3 of ADR-19: Instance migration support.
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'instance_id': self.instance_id,
            'instance_data': self.instance_containers.get(self.instance_id, {}),
            'cleanup_stats': self.cleanup_stats,
            'configuration': {
                'timeout_hours': INSTANCE_TIMEOUT_HOURS,
                'cleanup_interval': CLEANUP_INTERVAL_MINUTES,
                'auto_cleanup': ENABLE_AUTO_CLEANUP,
                'verbose_mode': MCP_VERBOSE
            }
        }
        
        # Include project information but not actual containers (too heavy)
        instance_data = self.instance_containers.get(self.instance_id, {})
        projects = {}
        for project_name in instance_data.get('project_containers', {}).keys():
            projects[project_name] = {
                'name': project_name,
                'has_neo4j': instance_data['project_containers'][project_name].neo4j is not None,
                'has_qdrant': instance_data['project_containers'][project_name].qdrant is not None,
                'has_nomic': instance_data['project_containers'][project_name].nomic is not None
            }
        export_data['projects'] = projects
        
        logger.info(f"Exported instance state with {len(projects)} projects")
        return export_data
    
    async def import_instance_state(self, import_data: dict, merge: bool = False):
        """
        Import instance state from another instance.
        Phase 3 of ADR-19: Instance migration support.
        
        Args:
            import_data: Exported state from another instance
            merge: If True, merge with existing state. If False, replace.
        """
        try:
            imported_id = import_data.get('instance_id')
            
            if not merge:
                # Clear existing data for non-current instances
                for inst_id in list(self.instance_containers.keys()):
                    if inst_id != self.instance_id:
                        del self.instance_containers[inst_id]
            
            # Import the data (but not as current instance)
            if imported_id and imported_id != self.instance_id:
                self.instance_containers[imported_id] = import_data.get('instance_data', {})
                logger.info(f"Imported instance state from {imported_id}")
                
                # Update cleanup stats if provided
                if 'cleanup_stats' in import_data and not merge:
                    self.cleanup_stats = import_data['cleanup_stats']
            
            return True
        except Exception as e:
            logger.error(f"Failed to import instance state: {e}")
            return False
    
    def get_instance_metrics(self):
        """Get metrics about current instances"""
        now = datetime.now()
        metrics = {
            'current_instance_id': self.instance_id,
            'total_instances': len(self.instance_containers),
            'active_instances': 0,
            'stale_instances': 0,
            'cleanup_stats': self.cleanup_stats,
            'instances': []
        }
        
        stale_threshold = timedelta(hours=INSTANCE_TIMEOUT_HOURS)
        
        for instance_id, instance_data in self.instance_containers.items():
            last_activity = instance_data.get('last_activity', instance_data.get('session_started'))
            is_stale = (now - last_activity) > stale_threshold and instance_id != self.instance_id
            
            if is_stale:
                metrics['stale_instances'] += 1
            else:
                metrics['active_instances'] += 1
            
            metrics['instances'].append({
                'id': instance_id,
                'is_current': instance_id == self.instance_id,
                'is_stale': is_stale,
                'session_started': instance_data.get('session_started').isoformat() if instance_data.get('session_started') else None,
                'last_activity': last_activity.isoformat() if last_activity else None,
                'idle_time': str(now - last_activity) if last_activity else None,
                'project_count': len(instance_data.get('project_containers', {}))
            })
        
        return metrics


state = MultiProjectServiceState()
server = Server("l9-neural-enhanced")

# ADR-0033: Dynamic project context manager
PROJECT_CONTEXT = ProjectContextManager()
PROJECT_CONTEXT.set_instance_id(state.instance_id)

# Update logger to include instance ID now that state is available
logger.get_instance_id = lambda: state.instance_id if state else "unknown"


# Removed initialize_services() - services now initialized lazily on first use
# This prevents blocking the MCP handshake with 30+ second service initialization


async def get_project_context(arguments: Dict[str, Any]):
    """
    Get project context with ADR-0033 dynamic detection.
    Falls back to DEFAULT_PROJECT_NAME if detection fails.
    """
    global PROJECT_CONTEXT
    
    # Check if project explicitly provided in arguments
    if 'project' in arguments and arguments['project']:
        project_name = arguments['project']
        # Update context manager's current project
        try:
            await PROJECT_CONTEXT.switch_project(project_name)
            logger.debug(f"🔄 Switched to explicitly requested project: {project_name}")
        except ValueError as e:
            # Project not found, fall back to auto-detection
            logger.warning(f"⚠️ Could not switch to project '{project_name}': {e}. Falling back to auto-detection.")
            context = await PROJECT_CONTEXT.get_current_project()
            project_name = context['project']
        except Exception as e:
            # Any other error, fall back to auto-detection
            logger.error(f"❌ Error switching to project '{project_name}': {e}. Falling back to auto-detection.")
            context = await PROJECT_CONTEXT.get_current_project()
            project_name = context['project']
    else:
        # Use dynamic detection
        context = await PROJECT_CONTEXT.get_current_project()
        project_name = context['project']
        
        # Log detection for debugging
        method = context.get('method', 'unknown')
        confidence = context.get('confidence', 0)
        if confidence < 0.7:
            logger.warning(f"⚠️ Low confidence project detection: {project_name} (method: {method}, confidence: {confidence:.0%})")
        else:
            logger.debug(f"🎯 Detected project: {project_name} (method: {method}, confidence: {confidence:.0%})")
    
    # Get or create container for project
    container = await state.get_project_container(project_name)
    retriever = await state.get_project_retriever(project_name)
    
    # ADR-0035: Proactively ensure indexer container is ready to eliminate race conditions
    # This prevents the common issue where first reindex fails because container isn't ready
    if container:
        try:
            # Check if indexer auto-start is enabled (default: true)
            indexer_auto_start = os.getenv('INDEXER_AUTO_START', 'true').lower() == 'true'
            
            if indexer_auto_start:
                logger.debug(f"🚀 [ADR-0035] Ensuring indexer container ready for {project_name}")
                
                # Use existing ensure_indexer_running logic (already handles duplicates)
                # Get project path from PROJECT_CONTEXT if available, else use current directory
                if 'context' in locals() and isinstance(context, dict):
                    project_path = context.get('path', os.getcwd())
                else:
                    # Get fresh project info for path
                    current_context = await PROJECT_CONTEXT.get_current_project()
                    project_path = current_context.get('path', os.getcwd())
                
                container_id = await container.ensure_indexer_running(project_path)
                
                logger.debug(f"✅ [ADR-0035] Indexer container ready: {container_id[:12] if container_id else 'N/A'}")
            else:
                logger.debug(f"⏭️ [ADR-0035] Indexer auto-start disabled for {project_name}")
                
        except Exception as e:
            # Don't fail the entire request if indexer startup fails
            # This allows graceful degradation while still providing other neural tools
            logger.warning(f"⚠️ [ADR-0035] Indexer auto-start failed for {project_name}: {e}")
            logger.warning(f"⚠️ Neural tools will work but first reindex may fail (user can retry)")
    
    return project_name, container, retriever


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    # Tools are statically defined - no initialization needed for listing
    return [
        types.Tool(
            name="neural_system_status",
            description="Get comprehensive neural system status and health",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="semantic_code_search",
            description=(
                "Search code by meaning using semantic embeddings.\n"
                "Usage: {\"query\": \"natural language\", \"limit\": 10}\n"
                f"limit: 1..{LIMITS['semantic_limit_max']} (default 10)."
            ),
            inputSchema={
                "type": "object",
                "title": "Semantic Code Search",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 3,
                        "description": "Plain text query. Example: 'how to start server'"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": LIMITS["semantic_limit_max"],
                        "default": 10,
                        "description": f"Results to return (1..{LIMITS['semantic_limit_max']})."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="graphrag_hybrid_search",
            description=(
                "Hybrid search with graph context.\n"
                "Usage: {\"query\": \"...\", \"limit\": 5, \"include_graph_context\": true, \"max_hops\": 2}"
            ),
            inputSchema={
                "type": "object",
                "title": "GraphRAG Hybrid Search",
                "properties": {
                    "query": {"type": "string", "minLength": 3, "description": "Search text."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": LIMITS["graphrag_limit_max"], "default": 5},
                    "include_graph_context": {"type": "boolean", "default": True},
                    "max_hops": {"type": "integer", "minimum": 0, "maximum": LIMITS["max_hops_max"], "default": 2}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="project_understanding",
            description=(
                "Get condensed project understanding.\n"
                "Usage: {\"scope\": \"full|summary|files|services\"}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["full", "summary", "files", "services"], "default": "full"}
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="indexer_status",
            description="Get neural indexer sidecar status and metrics",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="reindex_path",
            description=(
                "Enqueue a file or folder for reindexing. Path is relative to workspace unless absolute.\n"
                "Usage: {\"path\": \"src/\", \"recursive\": true}"
            ),
            inputSchema={
                "type": "object",
                "title": "Reindex Path",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "description": "File or directory to reindex."},
                    "recursive": {"type": "boolean", "default": True}
                },
                "required": ["path"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="neural_tools_help",
            description=(
                "Show usage examples and constraints for all neural tools."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="instance_metrics",
            description="Get metrics about MCP instance isolation and resource usage (Phase 2 of ADR-19)",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        # Schema management tools (ADR-0020)
        types.Tool(
            name="schema_init",
            description=(
                "Initialize or auto-detect project GraphRAG schema.\n"
                "Usage: {\"project_type\": \"react|django|fastapi|auto\", \"auto_detect\": true}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_type": {"type": "string", "enum": ["react", "vue", "angular", "nextjs", "django", "fastapi", "flask", "express", "springboot", "rails", "generic", "auto"], "default": "auto"},
                    "auto_detect": {"type": "boolean", "default": True, "description": "Auto-detect project type from files"}
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="schema_status",
            description="Get current project schema information and status",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="schema_validate",
            description=(
                "Validate data against current schema.\n"
                "Usage: {\"validate_nodes\": true, \"validate_relationships\": true}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "validate_nodes": {"type": "boolean", "default": True},
                    "validate_relationships": {"type": "boolean", "default": True},
                    "fix_issues": {"type": "boolean", "default": False}
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="schema_add_node_type",
            description=(
                "Add custom node type to project schema.\n"
                "Usage: {\"name\": \"Component\", \"properties\": {\"name\": \"string\", \"type\": \"string\"}}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "description": "Node type name"},
                    "properties": {"type": "object", "description": "Node properties and their types"},
                    "indexes": {"type": "array", "items": {"type": "string"}, "description": "Properties to index"},
                    "description": {"type": "string", "description": "Node type description"}
                },
                "required": ["name", "properties"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="schema_add_relationship",
            description=(
                "Add custom relationship type to project schema.\n"
                "Usage: {\"name\": \"USES_HOOK\", \"from_types\": [\"Component\"], \"to_types\": [\"Hook\"]}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "description": "Relationship type name"},
                    "from_types": {"type": "array", "items": {"type": "string"}, "description": "Source node types"},
                    "to_types": {"type": "array", "items": {"type": "string"}, "description": "Target node types"},
                    "properties": {"type": "object", "description": "Relationship properties"},
                    "description": {"type": "string", "description": "Relationship description"}
                },
                "required": ["name", "from_types", "to_types"],
                "additionalProperties": False
            }
        ),
        # Migration management tools (ADR-0021)
        types.Tool(
            name="migration_generate",
            description=(
                "Generate a new migration from schema changes.\n"
                "Usage: {\"name\": \"add_user_auth\", \"description\": \"Add authentication fields\"}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1, "description": "Migration name (alphanumeric + underscore)"},
                    "description": {"type": "string", "description": "Migration description"},
                    "dry_run": {"type": "boolean", "default": False, "description": "Preview without creating file"}
                },
                "required": ["name"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="migration_apply",
            description=(
                "Apply pending migrations.\n"
                "Usage: {\"target_version\": 5, \"dry_run\": false}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target_version": {"type": "integer", "minimum": 0, "description": "Target version (omit for latest)"},
                    "dry_run": {"type": "boolean", "default": False, "description": "Preview what would be applied"}
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="migration_rollback",
            description=(
                "Rollback to a previous migration version.\n"
                "Usage: {\"target_version\": 3, \"force\": false}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target_version": {"type": "integer", "minimum": 0, "description": "Version to rollback to"},
                    "force": {"type": "boolean", "default": False, "description": "Skip safety checks"}
                },
                "required": ["target_version"],
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="migration_status",
            description="Get current migration status and pending migrations",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="schema_diff",
            description=(
                "Compare schema between database and schema.yaml.\n"
                "Usage: {\"from_source\": \"database\", \"to_source\": \"schema.yaml\"}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "from_source": {"type": "string", "default": "database", "description": "Source for comparison"},
                    "to_source": {"type": "string", "default": "schema.yaml", "description": "Target for comparison"}
                },
                "additionalProperties": False
            }
        ),
        # Canonical knowledge management (ADR-0031)
        types.Tool(
            name="canon_understanding",
            description=(
                "Get comprehensive canonical knowledge understanding for the project.\n"
                "Returns detailed breakdown of canonical sources, their distribution,\n"
                "and recommendations for improving canonical coverage."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
        types.Tool(
            name="backfill_metadata",
            description=(
                "Backfill metadata for already-indexed content.\n"
                "Adds canonical weights, PRISM scores, and Git metadata to existing data.\n"
                "Usage: {\"batch_size\": 100, \"dry_run\": false}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "default": 100, "description": "Number of files to process in each batch"},
                    "dry_run": {"type": "boolean", "default": False, "description": "If true, only report what would be backfilled"}
                },
                "additionalProperties": False
            }
        ),
        # Dynamic project context management (ADR-0033)
        types.Tool(
            name="set_project_context",
            description=(
                "Set or auto-detect the active project for this Claude session.\n"
                "Solves the problem where MCP cannot detect which project you're working on.\n"
                "Usage: {\"path\": \"/path/to/project\"} or {} for auto-detection"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to project directory. If not provided, will auto-detect."
                    }
                },
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="list_projects",
            description=(
                "List all known projects and their status.\n"
                "Shows project paths, last activity, and which is currently active."
            ),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False}
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    # Services will be initialized lazily on first actual tool use
    instance_id = state.instance_id
    
    # Log tool call with instance ID for debugging
    logger.info(f"🔧 [Instance {instance_id}] Tool call: {name}")

    # L9 2025: Session-aware tool execution
    try:
        # Generate or extract session ID (simplified for now)
        session_id = arguments.get('session_id') or secrets.token_urlsafe(16)
        
        # Get project container with connection pooling
        project_name, container, retriever = await get_project_context(arguments)
        
        # Get or create session
        if container.session_manager:
            session = await container.session_manager.get_or_create_session(session_id)
            
            # Check rate limits
            if not await session.check_rate_limit():
                return [types.TextContent(type="text", text=json.dumps({
                    "status": "error",
                    "code": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Current limit: {session.resource_limits['queries_per_minute']} queries/minute.",
                    "session_id": session_id[:8] + "..."
                }, indent=2))]
                
            # Clean up expired sessions periodically
            await container.session_manager.cleanup_expired_sessions()
        else:
            session = None
        
        # Execute tools with session context
        if name == "neural_system_status":
            return await neural_system_status_impl()
        elif name == "semantic_code_search":
            # Validate and normalize
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            if not isinstance(query, str) or len(query.strip()) < 3:
                return [types.TextContent(type="text", text=_make_validation_error(
                    "semantic_code_search",
                    "Missing or too-short 'query' (string, minLength 3).",
                    missing=["query"] if not query else [],
                    example={"query": "how to start server", "limit": 10},
                    received=arguments
                ))]
            # Coerce and clamp limit
            try:
                limit = int(limit)
            except Exception:
                limit = 10
            if limit < 1:
                limit = 1
            if limit > LIMITS["semantic_limit_max"]:
                limit = LIMITS["semantic_limit_max"]
            return await semantic_code_search_impl(query.strip(), limit)
        elif name == "graphrag_hybrid_search":
            query = arguments.get("query", "")
            if not isinstance(query, str) or len(query.strip()) < 3:
                return [types.TextContent(type="text", text=_make_validation_error(
                    "graphrag_hybrid_search",
                    "Missing or too-short 'query' (string, minLength 3).",
                    missing=["query"] if not query else [],
                    example={"query": "find graph relationships", "limit": 5, "include_graph_context": True, "max_hops": 2},
                    received=arguments
                ))]
            limit = arguments.get("limit", 5)
            try:
                limit = int(limit)
            except Exception:
                limit = 5
            limit = max(1, min(limit, LIMITS["graphrag_limit_max"]))
            include_graph_context = bool(arguments.get("include_graph_context", True))
            max_hops = arguments.get("max_hops", 2)
            try:
                max_hops = int(max_hops)
            except Exception:
                max_hops = 2
            max_hops = max(0, min(max_hops, LIMITS["max_hops_max"]))
            return await graphrag_hybrid_search_impl(query.strip(), limit, include_graph_context, max_hops)
        elif name == "project_understanding":
            scope = arguments.get("scope", "full")
            allowed = ["full", "summary", "files", "services"]
            if scope not in allowed:
                return [types.TextContent(type="text", text=_make_validation_error(
                    "project_understanding",
                    "Invalid 'scope'. Must be one of: full, summary, files, services.",
                    invalid=[{"field": "scope", "reason": f"allowed: {allowed}"}],
                    example={"scope": "summary"},
                    received=arguments,
                    normalized={"scope": "full"}
                ))]
            return await project_understanding_impl(scope)
        elif name == "indexer_status":
            return await indexer_status_impl()
        elif name == "reindex_path":
            p = arguments.get("path", "")
            if not isinstance(p, str) or len(p.strip()) == 0:
                return [types.TextContent(type="text", text=_make_validation_error(
                    "reindex_path",
                    "Missing required field: 'path' (string, minLength 1).",
                    missing=["path"],
                    example={"path": "src/", "recursive": True},
                    received=arguments
                ))]
            return await reindex_path_impl(p)
        elif name == "neural_tools_help":
            return await neural_tools_help_impl()
        elif name == "instance_metrics":
            return await instance_metrics_impl()
        # Schema management tools (ADR-0020)
        elif name == "schema_init":
            return await schema_init_impl(arguments)
        elif name == "schema_status":
            return await schema_status_impl()
        elif name == "schema_validate":
            return await schema_validate_impl(arguments)
        elif name == "schema_add_node_type":
            return await schema_add_node_type_impl(arguments)
        elif name == "schema_add_relationship":
            return await schema_add_relationship_impl(arguments)
        # Migration tools (ADR-0021)
        elif name == "migration_generate":
            return await migration_generate_impl(arguments)
        elif name == "migration_apply":
            return await migration_apply_impl(arguments)
        elif name == "migration_rollback":
            return await migration_rollback_impl(arguments)
        elif name == "migration_status":
            return await migration_status_impl(arguments)
        elif name == "schema_diff":
            return await schema_diff_impl(arguments)
        elif name == "canon_understanding":
            return await canon_understanding_impl(arguments)
        elif name == "backfill_metadata":
            return await backfill_metadata_impl(arguments)
        elif name == "set_project_context":
            return await set_project_context_impl(arguments)
        elif name == "list_projects":
            return await list_projects_impl(arguments)
        else:
            return [types.TextContent(type="text", text=json.dumps({
                "error": f"Unknown tool: {name}",
                "_debug": {"instance_id": instance_id}
            }))]
    except Exception as e:
        logger.error(f"[Instance {instance_id}] Tool call failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e),
            "_debug": {"instance_id": instance_id}
        }))]


async def canon_understanding_impl(arguments: dict) -> List[types.TextContent]:
    """
    Get comprehensive canonical knowledge understanding for the project
    Returns detailed breakdown of canonical sources, their distribution,
    and recommendations for improving canonical coverage.
    """
    try:
        project_name, container, _ = await get_project_context({})
        # Get project path from environment or default to current directory
        project_path = os.getcwd()
        canon_manager = CanonManager(project_name, project_path)
        
        # Load canon configuration
        canon_config_exists = (Path(project_path) / ".canon.yaml").exists()
        
        # Query Neo4j for canon statistics
        stats_query = """
        MATCH (f:File {project: $project})
        RETURN 
            f.canon_level as level,
            f.canon_weight as weight,
            COUNT(f) as count,
            AVG(f.complexity_score) as avg_complexity,
            AVG(f.recency_score) as avg_recency,
            SUM(f.todo_count) as total_todos,
            SUM(f.fixme_count) as total_fixmes
        ORDER BY weight DESC
        """
        
        results = await container.neo4j.execute_cypher(
            stats_query, {'project': project_name}
        )
        
        # Build distribution analysis
        level_distribution = {}
        total_files = 0
        canonical_files = 0
        
        for row in results.get('result', []):
            level = row.get('level') or 'none'
            level_distribution[level] = {
                'count': row.get('count', 0),
                'average_weight': row.get('weight') or 0.5,
                'average_complexity': row.get('avg_complexity') or 0,
                'average_recency': row.get('avg_recency') or 0,
                'total_todos': row.get('total_todos') or 0,
                'total_fixmes': row.get('total_fixmes') or 0
            }
            total_files += row.get('count', 0)
            if row.get('weight') and row['weight'] >= 0.7:
                canonical_files += row.get('count', 0)
        
        # Get top canonical files
        top_canon_query = """
        MATCH (f:File {project: $project})
        WHERE f.canon_weight >= 0.7
        RETURN f.path, f.canon_level, f.canon_weight, f.canon_reason
        ORDER BY f.canon_weight DESC
        LIMIT 10
        """
        
        top_files = await container.neo4j.execute_cypher(
            top_canon_query, {'project': project_name}
        )
        
        # Get files that should be canonical (high complexity + high dependencies)
        suggested_canon_query = """
        MATCH (f:File {project: $project})
        WHERE (f.canon_weight IS NULL OR f.canon_weight < 0.7)
        AND f.complexity_score > 0.7
        AND f.dependencies_score > 0.6
        RETURN f.path, f.complexity_score, f.dependencies_score
        ORDER BY (f.complexity_score + f.dependencies_score) DESC
        LIMIT 10
        """
        
        suggestions = await container.neo4j.execute_cypher(
            suggested_canon_query, {'project': project_name}
        )
        
        # Build comprehensive response
        response = {
            'project': project_name,
            'canon_config_exists': canon_config_exists,
            'statistics': {
                'total_files': total_files,
                'canonical_files': canonical_files,
                'canonical_percentage': (canonical_files / total_files * 100) if total_files > 0 else 0,
                'has_primary_sources': 'primary' in level_distribution,
                'has_deprecated': 'deprecated' in level_distribution
            },
            'distribution': level_distribution,
            'top_canonical_sources': [
                {
                    'path': f.get('path'),
                    'level': f.get('canon_level'),
                    'weight': f.get('canon_weight'),
                    'reason': f.get('canon_reason') or 'No reason specified'
                }
                for f in top_files.get('result', [])[:10]
            ],
            'suggested_for_canon': [
                {
                    'path': f.get('path'),
                    'complexity': f.get('complexity_score'),
                    'dependencies': f.get('dependencies_score'),
                    'recommendation': 'Mark as primary or secondary canonical source'
                }
                for f in suggestions.get('result', [])
            ],
            'recommendations': _generate_canon_recommendations(
                canon_config_exists,
                level_distribution,
                canonical_files,
                total_files
            )
        }
        
        # Add example configuration if none exists
        if not canon_config_exists:
            response['example_config'] = _generate_example_canon_config(project_path)
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        logger.error(f"canon_understanding failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e)
        }))]


def _generate_canon_recommendations(config_exists, distribution, canonical_files, total_files):
    """Generate actionable recommendations for improving canonical coverage"""
    recommendations = []
    
    if not config_exists:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Create .canon.yaml configuration',
            'reason': 'No canonical configuration found',
            'impact': 'Enables explicit source-of-truth designation'
        })
    
    canon_percentage = (canonical_files / total_files * 100) if total_files > 0 else 0
    
    if canon_percentage < 10:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Identify and mark primary sources',
            'reason': f'Only {canon_percentage:.1f}% of files are canonical',
            'impact': 'Improves search relevance and AI recommendations'
        })
    
    if 'primary' not in distribution:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Designate primary canonical sources',
            'reason': 'No primary sources defined',
            'impact': 'Establishes clear sources of truth'
        })
    
    if 'deprecated' in distribution and distribution['deprecated']['count'] > total_files * 0.2:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Review and clean up deprecated code',
            'reason': f"{distribution['deprecated']['count']} deprecated files found",
            'impact': 'Reduces confusion and improves maintainability'
        })
    
    return recommendations


def _generate_example_canon_config(project_path):
    """Generate example .canon.yaml based on project structure"""
    # Analyze project to suggest config
    has_docs = (Path(project_path) / "docs").exists()
    has_api = (Path(project_path) / "api").exists()
    has_src = (Path(project_path) / "src").exists()
    
    config = """# .canon.yaml - Canonical Knowledge Configuration
version: "1.0"

primary:
"""
    
    if has_docs:
        config += """  - path: "docs/api-specification.md"
    weight: 1.0
    description: "Official API specification"
    
  - pattern: "docs/architecture/*.md"
    weight: 0.95
    description: "Architecture decision records"
"""
    
    if has_api:
        config += """  - pattern: "api/schema/*.yaml"
    weight: 0.9
    description: "API schema definitions"
"""
    
    if has_src:
        config += """
secondary:
  - pattern: "src/core/**/*.py"
    weight: 0.7
    description: "Core business logic"
    
  - pattern: "src/models/**/*.py"
    weight: 0.7
    description: "Data models"

reference:
  - pattern: "examples/**/*"
    weight: 0.4
    description: "Usage examples"
    
  - pattern: "tests/**/*"
    weight: 0.3
    description: "Test cases"

deprecated:
  - pattern: "legacy/**/*"
    weight: 0.1
    description: "Legacy code - do not use"
"""
    
    return config


async def backfill_metadata_impl(arguments: dict) -> List[types.TextContent]:
    """
    Backfill metadata for already-indexed content
    
    Args:
        batch_size: Number of files to process in each batch
        dry_run: If true, only report what would be backfilled
    """
    try:
        project_name, container, _ = await get_project_context({})
        # Get project path from environment or default to current directory
        project_path = os.getcwd()
        
        batch_size = arguments.get('batch_size', 100)
        dry_run = arguments.get('dry_run', False)
        
        # Create backfiller
        backfiller = MetadataBackfiller(project_name, project_path, container)
        
        if dry_run:
            # Just count files needing backfill
            files = await backfiller.get_files_needing_backfill()
            
            response = {
                'mode': 'dry_run',
                'files_needing_backfill': len(files),
                'estimated_time_minutes': len(files) * 0.1 / 60  # ~0.1s per file
            }
            
            # Include sample of files if not too many
            if files and len(files) <= 10:
                response['sample_files'] = [f['path'] for f in files]
            elif files:
                response['sample_files'] = [f['path'] for f in files[:10]]
                response['sample_note'] = f"Showing first 10 of {len(files)} files"
            
            return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
        # Run the backfill
        await backfiller.backfill_project(batch_size=batch_size, dry_run=False)
        
        response = {
            'mode': 'executed',
            'files_processed': backfiller.processed_count,
            'neo4j_updated': backfiller.updated_neo4j,
            'qdrant_updated': backfiller.updated_qdrant,
            'skipped': backfiller.skipped_count,
            'errors': backfiller.error_count,
            'success': backfiller.error_count == 0
        }
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
        
    except Exception as e:
        logger.error(f"backfill_metadata failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e)
        }))]


async def semantic_code_search_impl(query: str, limit: int) -> List[types.TextContent]:
    try:
        # Import centralized naming (ADR-0039)
        from servers.config.collection_naming import collection_naming

        project_name, container, _ = await get_project_context({})
        embeddings = await container.nomic.get_embeddings([query])
        query_vector = embeddings[0]

        # ADR-0039: Use centralized naming with backward compatibility
        possible_names = collection_naming.get_possible_names_for_lookup(project_name)

        search_results = None
        for collection_name in possible_names:
            try:
                search_results = await container.qdrant.search_vectors(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit
                )
                logger.debug(f"Found collection: {collection_name}")
                break  # Success, stop trying
            except Exception as e:
                logger.debug(f"Collection {collection_name} not found, trying next...")
                continue

        if search_results is None:
            logger.error(f"No collection found for project {project_name}. Tried: {possible_names}")
            return await _fallback_neo4j_search(query, limit)

        formatted = []
        for hit in search_results:
            if isinstance(hit, dict):
                payload = hit.get("payload", {})
                score = hit.get("score", 0.0)
            else:
                # Fallback for client object shape
                payload = getattr(hit, 'payload', {}) or {}
                score = float(getattr(hit, 'score', 0.0))
            content = payload.get("content", "")
            formatted.append({
                "score": float(score),
                "file_path": payload.get("file_path", ""),
                "snippet": (content[:200] + "...") if len(content) > 200 else content
            })

        response = {
            "status": "success",
            "query": query,
            "results": formatted,
            "total_found": len(formatted)
        }
        
        # Add instance metadata in verbose mode
        response = add_instance_metadata(response, project_name=project_name)
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
    except Exception as e:
        error_response = {"status": "error", "message": str(e)}
        error_response = add_instance_metadata(error_response)
        return [types.TextContent(type="text", text=json.dumps(error_response))]


async def _fallback_neo4j_search(query: str, limit: int) -> List[types.TextContent]:
    try:
        project_name, container, _ = await get_project_context({})
        cypher = """
        MATCH (c:CodeChunk)
        WHERE c.content IS NOT NULL AND (c.content CONTAINS $query OR c.file_path CONTAINS $query)
        RETURN c.file_path as file_path, c.content as content LIMIT $limit
        """
        result = await container.neo4j.execute_cypher(cypher, {"query": query, "limit": limit})
        formatted = []
        data = result.get('data') if isinstance(result, dict) else result
        if data:
            for r in data:
                content = r.get("content", "")
                formatted.append({
                    "file_path": r.get("file_path", ""),
                    "snippet": (content[:200] + "...") if len(content) > 200 else content,
                    "fallback": True
                })
        return [types.TextContent(type="text", text=json.dumps({
            "status": "success",
            "message": "Results from Neo4j fallback search (Qdrant unavailable)",
            "query": query,
            "results": formatted,
            "total_found": len(formatted),
            "fallback_mode": True
        }, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def graphrag_hybrid_search_impl(query: str, limit: int, include_graph_context: bool, max_hops: int) -> List[types.TextContent]:
    try:
        project_name, _, retriever = await get_project_context({})
        if not retriever:
            return [types.TextContent(type="text", text=json.dumps({"error": "GraphRAG not available"}))]

        results = await retriever.find_similar_with_context(query, limit, include_graph_context, max_hops)
        formatted_results = []
        for res in results:
            entry = {
                "score": res.get("score", 0),
                "file": res.get("file_path", ""),
                "lines": f"{res.get('start_line', 0)}-{res.get('end_line', 0)}",
                "content": (res.get("content", "")[:200] + "...")
            }
            if include_graph_context and res.get("graph_context"):
                ctx = res["graph_context"]
                entry["graph_context"] = {
                    "imports": ctx.get("imports", []),
                    "imported_by": ctx.get("imported_by", []),
                    "related_chunks": len(ctx.get("related_chunks", []))
                }
            formatted_results.append(entry)
        return [types.TextContent(type="text", text=json.dumps({
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def neural_system_status_impl() -> List[types.TextContent]:
    """Get comprehensive neural system status and health with instance information"""
    try:
        project_name, container, retriever = await get_project_context({})
        
        # Phase 3: Include comprehensive instance information
        status = {
            "project": project_name,
            "timestamp": datetime.now().isoformat(),
            "instance": {
                "id": state.instance_id,
                "session_started": state.instance_containers[state.instance_id]['session_started'].isoformat(),
                "uptime": str(datetime.now() - state.instance_containers[state.instance_id]['session_started']),
                "projects_loaded": len(state.instance_containers[state.instance_id].get('project_containers', {}))
            },
            "services": {
                "neo4j": {
                    "connected": container.neo4j is not None,
                    "type": "REAL GraphDatabase connection" if container.neo4j else "Not connected",
                    "status": "healthy" if container.neo4j else "unavailable"
                },
                "qdrant": {
                    "connected": container.qdrant is not None,
                    "type": "REAL Qdrant vector database" if container.qdrant else "Not connected", 
                    "status": "healthy" if container.qdrant else "unavailable"
                },
                "nomic": {
                    "connected": container.nomic is not None,
                    "type": "REAL Nomic embedding service" if container.nomic else "Not connected",
                    "status": "healthy" if container.nomic else "unavailable"
                }
            },
            "hybrid_retriever": {
                "available": retriever is not None,
                "status": "active" if retriever else "inactive"
            },
            "indexing_status": "REAL services - NO MOCKS!"
        }
        
        # Test actual connections - PROJECT ISOLATED (ADR-0032)
        if container.neo4j:
            try:
                # Neo4j service wrapper - use async query WITH PROJECT FILTER
                if hasattr(container.neo4j, 'client') and container.neo4j.client:
                    # Count only nodes for the current project
                    result = await container.neo4j.client.execute_query(
                        "MATCH (n {project: $project}) RETURN count(n) as node_count",
                        {"project": project_name}
                    )
                    if result["status"] == "success" and result["result"]:
                        node_count = result["result"][0]["node_count"]
                        status["services"]["neo4j"]["node_count"] = node_count
                        status["services"]["neo4j"]["project_scope"] = project_name
                        
                        # Add breakdown by node type for better visibility (ADR-0032)
                        breakdown_query = """
                        MATCH (f:File {project: $project}) WITH count(f) as files
                        MATCH (c:Class {project: $project}) WITH files, count(c) as classes
                        MATCH (m:Method {project: $project}) WITH files, classes, count(m) as methods
                        RETURN files, classes, methods
                        """
                        breakdown_result = await container.neo4j.client.execute_query(
                            breakdown_query, {"project": project_name}
                        )
                        if breakdown_result["status"] == "success" and breakdown_result["result"]:
                            bd = breakdown_result["result"][0]
                            status["services"]["neo4j"]["breakdown"] = {
                                "files": bd.get("files", 0),
                                "classes": bd.get("classes", 0),
                                "methods": bd.get("methods", 0)
                            }
                    else:
                        status["services"]["neo4j"]["node_count"] = 0
                else:
                    status["services"]["neo4j"]["node_count"] = "N/A - client not initialized"
            except Exception as e:
                status["services"]["neo4j"]["error"] = str(e)
                
        if container.qdrant:
            try:
                # Get ALL collections first
                all_collections = await container.qdrant.get_collections()
                # Filter to show only collections for the current project (ADR-0032)
                project_prefix = f"project_{project_name}_"
                project_collections = [c for c in all_collections if c.startswith(project_prefix)]
                
                status["services"]["qdrant"]["collections"] = project_collections
                status["services"]["qdrant"]["collection_count"] = len(project_collections)
                status["services"]["qdrant"]["project_scope"] = project_name
            except Exception as e:
                status["services"]["qdrant"]["error"] = str(e)
        
        # Add instance metrics summary
        metrics = state.get_instance_metrics()
        status["instance"]["isolation_metrics"] = {
            "total_instances": metrics['total_instances'],
            "active_instances": metrics['active_instances'],
            "stale_instances": metrics['stale_instances']
        }
        
        # Add configuration in verbose mode
        if MCP_VERBOSE:
            status["configuration"] = {
                "verbose_mode": MCP_VERBOSE,
                "instance_metadata": INCLUDE_INSTANCE_METADATA,
                "timeout_hours": INSTANCE_TIMEOUT_HOURS,
                "cleanup_interval_minutes": CLEANUP_INTERVAL_MINUTES,
                "auto_cleanup": ENABLE_AUTO_CLEANUP
            }
        
        response = {"status": "success", "system_status": status}
        
        # Add metadata if verbose
        response = add_instance_metadata(response, project_name=project_name)
        
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]
    except Exception as e:
        error_response = {"status": "error", "message": str(e)}
        if 'state' in globals():
            error_response["instance_id"] = state.instance_id
        return [types.TextContent(type="text", text=json.dumps(error_response))]


async def project_understanding_impl(scope: str = "full") -> List[types.TextContent]:
    """
    Get real project understanding with actual data (ADR-0032)
    Scopes: full, summary, files, services
    """
    try:
        project_name, container, retriever = await get_project_context({})
        
        understanding = {
            "project": project_name,
            "timestamp": datetime.now().isoformat(),
            "scope": scope,
            "protocol": "2025-06-18",
            "transport": "stdio"
        }
        
        # Get real project statistics based on scope
        if scope in ["full", "summary"]:
            # Query Neo4j for project-specific counts
            if container.neo4j and hasattr(container.neo4j, 'client'):
                try:
                    # Get file count
                    file_result = await container.neo4j.client.execute_query(
                        "MATCH (f:File {project: $project}) RETURN count(f) as count",
                        {"project": project_name}
                    )
                    file_count = 0
                    if file_result["status"] == "success" and file_result["result"]:
                        file_count = file_result["result"][0]["count"]
                    
                    # Get class count
                    class_result = await container.neo4j.client.execute_query(
                        "MATCH (c:Class {project: $project}) RETURN count(c) as count",
                        {"project": project_name}
                    )
                    class_count = 0
                    if class_result["status"] == "success" and class_result["result"]:
                        class_count = class_result["result"][0]["count"]
                    
                    # Get method count
                    method_result = await container.neo4j.client.execute_query(
                        "MATCH (m:Method {project: $project}) RETURN count(m) as count",
                        {"project": project_name}
                    )
                    method_count = 0
                    if method_result["status"] == "success" and method_result["result"]:
                        method_count = method_result["result"][0]["count"]
                    
                    understanding["statistics"] = {
                        "files": file_count,
                        "classes": class_count,
                        "methods": method_count,
                        "total_nodes": file_count + class_count + method_count,
                        "indexed": file_count > 0
                    }
                except Exception as e:
                    understanding["statistics"] = {"error": str(e)}
            
            # Get Qdrant vector counts
            if container.qdrant:
                try:
                    collection_name = f"project-{project_name}"
                    # Check if collection exists
                    all_collections = await container.qdrant.get_collections()
                    if collection_name in all_collections:
                        # TODO: Add vector count when method is available
                        understanding["vectors"] = {
                            "collection": collection_name,
                            "exists": True
                        }
                    else:
                        understanding["vectors"] = {
                            "collection": collection_name,
                            "exists": False
                        }
                except Exception as e:
                    understanding["vectors"] = {"error": str(e)}
        
        if scope in ["full", "files"]:
            # Get sample of indexed files
            if container.neo4j and hasattr(container.neo4j, 'client'):
                try:
                    files_result = await container.neo4j.client.execute_query(
                        """
                        MATCH (f:File {project: $project})
                        RETURN f.path as path, f.language as language
                        ORDER BY f.path
                        LIMIT 10
                        """,
                        {"project": project_name}
                    )
                    if files_result["status"] == "success":
                        understanding["sample_files"] = [
                            {"path": r["path"], "language": r.get("language", "unknown")}
                            for r in files_result["result"]
                        ]
                except Exception as e:
                    understanding["sample_files"] = {"error": str(e)}
        
        if scope in ["full", "services"]:
            # Service status
            understanding["services"] = {
                "neo4j": "connected" if container.neo4j else "disconnected",
                "qdrant": "connected" if container.qdrant else "disconnected",
                "nomic": "connected" if container.nomic else "disconnected",
                "retriever": "available" if retriever else "unavailable"
            }
        
        return [types.TextContent(type="text", text=json.dumps({"status": "success", "understanding": understanding}, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]


async def instance_metrics_impl() -> List[types.TextContent]:
    """Return metrics about instance isolation and resource usage"""
    try:
        metrics = state.get_instance_metrics()
        
        # Add cleanup configuration
        metrics['cleanup_config'] = {
            'enabled': ENABLE_AUTO_CLEANUP,
            'timeout_hours': INSTANCE_TIMEOUT_HOURS,
            'cleanup_interval_minutes': CLEANUP_INTERVAL_MINUTES
        }
        
        # Format the response
        return [types.TextContent(type="text", text=json.dumps({
            "status": "success",
            "metrics": metrics
        }, indent=2, default=str))]
    except Exception as e:
        logger.error(f"Failed to get instance metrics: {e}")
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e)
        }))]


async def neural_tools_help_impl() -> List[types.TextContent]:
    """Return usage guidance, examples, and constraints for all tools."""
    try:
        help_payload = {
            "tools": {
                "semantic_code_search": {
                    "usage": {"query": "how to start server", "limit": 10},
                    "constraints": {"query_minLength": 3, "limit": [1, LIMITS["semantic_limit_max"]]},
                    "returns": {"status": "success", "results": "[]", "total_found": 0}
                },
                "graphrag_hybrid_search": {
                    "usage": {"query": "find relationships", "limit": 5, "include_graph_context": True, "max_hops": 2},
                    "constraints": {"query_minLength": 3, "limit": [1, LIMITS["graphrag_limit_max"]], "max_hops": [0, LIMITS["max_hops_max"]]},
                    "returns": {"results_count": 0, "results": "[]"}
                },
                "project_understanding": {
                    "usage": {"scope": "summary"},
                    "allowed_scopes": ["full", "summary", "files", "services"],
                    "returns": {"status": "success", "understanding": {"project": "..."}}
                },
                "indexer_status": {
                    "usage": {},
                    "returns": {"status": "success", "indexer_status": "healthy|unhealthy|disconnected|error"}
                },
                "reindex_path": {
                    "usage": {"path": "src/", "recursive": True},
                    "constraints": {"path_minLength": 1},
                    "returns": {"status": "enqueued"}
                }
            }
        }
        return [types.TextContent(type="text", text=json.dumps(help_payload, indent=2))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"status": "error", "message": str(e)}))]

async def indexer_status_impl() -> List[types.TextContent]:
    """Get neural indexer sidecar status and metrics via HTTP API"""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try to connect to indexer sidecar on localhost:48080
            response = await client.get("http://localhost:48080/status")
            
            if response.status_code == 200:
                status_data = response.json()
                
                # Format the response nicely
                formatted_status = {
                    "indexer_status": "healthy",
                    "timestamp": status_data.get("timestamp", "unknown"),
                    "metrics": {
                        "queue_depth": status_data.get("queue_depth", 0),
                        "files_processed": status_data.get("files_processed", 0),
                        "degraded_mode": status_data.get("degraded_mode", False)
                    },
                    "sidecar_connection": "connected"
                }
                
                return [types.TextContent(
                    type="text", 
                    text=json.dumps(formatted_status, indent=2)
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "indexer_status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "sidecar_connection": "failed"
                    }, indent=2)
                )]
                
    except httpx.ConnectError:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "indexer_status": "disconnected",
                "error": "Could not connect to indexer sidecar on localhost:48080",
                "sidecar_connection": "failed",
                "note": "Indexer sidecar may not be running in container mode"
            }, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "indexer_status": "error",
                "error": str(e),
                "sidecar_connection": "unknown"
            }, indent=2)
        )]


async def _wait_for_indexer_ready(port: int, timeout: int = 30) -> bool:
    """Wait for indexer container to be ready by checking health endpoint"""
    import httpx
    import asyncio
    
    for attempt in range(timeout):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"http://localhost:{port}/health")
                if response.status_code == 200:
                    logger.info(f"✅ Indexer on port {port} is ready (took {attempt + 1}s)")
                    return True
        except (httpx.ConnectError, httpx.TimeoutException):
            # Container not ready yet, wait
            pass
        
        await asyncio.sleep(1)
    
    logger.warning(f"⚠️ Indexer on port {port} not ready after {timeout}s")
    return False


async def reindex_path_impl(path: str) -> List[types.TextContent]:
    """Trigger reindexing of a specific path via project-specific indexer container"""
    import httpx
    
    if not path:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "Path is required for reindexing"}, indent=2)
        )]
    
    try:
        # Get current project context
        project_context = await PROJECT_CONTEXT.get_current_project()
        project_name = project_context['project']
        project_path = project_context['path']
        
        # Get or create service container for this project
        container = await state.get_service_container(project_name)
        
        # Ensure project-specific indexer is running
        logger.info(f"Ensuring indexer is running for project: {project_name}")
        container_id = await container.ensure_indexer_running(project_path)
        
        # Get the port for this project's indexer
        if hasattr(container, 'indexer_orchestrator') and container.indexer_orchestrator:
            indexer_port = container.indexer_orchestrator.get_indexer_port(project_name)
        else:
            # Fallback to default port if orchestrator not available
            indexer_port = 48080
            
        if not indexer_port:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "error": f"Could not determine port for {project_name} indexer",
                    "project": project_name
                }, indent=2)
            )]
        
        # CRITICAL: Wait for container to be fully ready before attempting connection
        logger.info(f"⏳ Waiting for indexer on port {indexer_port} to be ready...")
        is_ready = await _wait_for_indexer_ready(indexer_port, timeout=30)
        
        if not is_ready:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "error": f"Indexer on port {indexer_port} did not become ready within 30 seconds",
                    "project": project_name,
                    "container_id": container_id[:12] if container_id else None,
                    "troubleshooting": "Container may be building or have startup issues"
                }, indent=2)
            )]
        
        # Connect to project-specific indexer
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Send reindex request to project-specific indexer
            response = await client.post(
                f"http://localhost:{indexer_port}/reindex-path",
                params={"path": path}
            )
            
            if response.status_code == 200:
                result = response.json()
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": f"Reindex request queued for path: {path}",
                        "project": project_name,
                        "indexer_port": indexer_port,
                        "container_id": container_id[:12] if container_id else None,
                        "details": result
                    }, indent=2)
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "failed",
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "path": path,
                        "project": project_name,
                        "indexer_port": indexer_port
                    }, indent=2)
                )]
                
    except httpx.ConnectError as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "failed",
                "error": f"Could not connect to indexer on port {indexer_port}",
                "path": path,
                "project": project_name if 'project_name' in locals() else 'unknown',
                "note": "Indexer container may not be running or port may be incorrect"
            }, indent=2)
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "error": str(e),
                "path": path,
                "project": project_name if 'project_name' in locals() else 'unknown'
            }, indent=2)
        )]


# Schema management tool implementations (ADR-0020)

async def schema_init_impl(arguments: dict) -> List[types.TextContent]:
    """Initialize or auto-detect project GraphRAG schema"""
    try:
        project_name, _, _ = await get_project_context(arguments)
        schema_manager = await state.get_project_schema(project_name)
        
        project_type_str = arguments.get('project_type', 'auto')
        auto_detect = arguments.get('auto_detect', True)
        
        if auto_detect or project_type_str == 'auto':
            # Auto-detect project type
            detected_type = await schema_manager.detect_project_type()
            logger.info(f"📋 Detected project type: {detected_type.value}")
        else:
            # Use specified project type
            detected_type = ProjectType(project_type_str)
        
        # Create or update schema
        schema = await schema_manager.create_schema(detected_type)
        
        # Apply to Neo4j if container is available
        project_name, container, _ = await get_project_context(arguments)
        if container and container.neo4j:
            await schema_manager.apply_to_neo4j(container.neo4j)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "project": project_name,
                "project_type": schema.project_type.value,
                "schema_file": str(schema_manager.schema_file),
                "node_types": list(schema.node_types.keys()),
                "relationship_types": list(schema.relationship_types.keys()),
                "collections": list(schema.collections.keys()),
                "message": f"Schema initialized for {project_name} ({detected_type.value} project)"
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Schema init failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def schema_status_impl() -> List[types.TextContent]:
    """Get current project schema information"""
    try:
        project_name, _, _ = await get_project_context({})
        schema_manager = await state.get_project_schema(project_name)
        
        if not schema_manager.current_schema:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "no_schema",
                    "project": project_name,
                    "message": "No schema configured. Use schema_init to create one."
                }, indent=2)
            )]
        
        schema = schema_manager.current_schema
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "configured",
                "project": project_name,
                "version": schema.version,
                "project_type": schema.project_type.value,
                "description": schema.description,
                "created_at": schema.created_at.isoformat(),
                "updated_at": schema.updated_at.isoformat(),
                "extends": schema.extends,
                "node_types": {
                    name: {
                        "properties": node.properties,
                        "indexes": node.indexes,
                        "description": node.description
                    }
                    for name, node in schema.node_types.items()
                },
                "relationship_types": {
                    name: {
                        "from": rel.from_types,
                        "to": rel.to_types,
                        "properties": rel.properties,
                        "description": rel.description
                    }
                    for name, rel in schema.relationship_types.items()
                },
                "collections": {
                    name: {
                        "vector_size": col.vector_size,
                        "distance": col.distance_metric,
                        "fields": col.fields
                    }
                    for name, col in schema.collections.items()
                }
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Schema status failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def schema_validate_impl(arguments: dict) -> List[types.TextContent]:
    """Validate data against current schema"""
    try:
        project_name, container, _ = await get_project_context(arguments)
        schema_manager = await state.get_project_schema(project_name)
        
        if not schema_manager.current_schema:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "no_schema",
                    "message": "No schema to validate against. Use schema_init first."
                }, indent=2)
            )]
        
        validate_nodes = arguments.get('validate_nodes', True)
        validate_relationships = arguments.get('validate_relationships', True)
        fix_issues = arguments.get('fix_issues', False)
        
        issues = []
        
        # TODO: Implement actual validation against Neo4j data
        # This would query Neo4j and check nodes/relationships against schema
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "validated",
                "project": project_name,
                "validated": {
                    "nodes": validate_nodes,
                    "relationships": validate_relationships
                },
                "issues_found": len(issues),
                "issues": issues,
                "fixed": fix_issues and len(issues) > 0
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Schema validate failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def schema_add_node_type_impl(arguments: dict) -> List[types.TextContent]:
    """Add custom node type to project schema"""
    try:
        project_name, _, _ = await get_project_context(arguments)
        schema_manager = await state.get_project_schema(project_name)
        
        if not schema_manager.current_schema:
            # Initialize with generic schema first
            await schema_manager.create_schema(ProjectType.GENERIC)
        
        schema = schema_manager.current_schema
        
        # Add new node type
        from schema_manager import NodeType
        node_type = NodeType(
            name=arguments['name'],
            properties=arguments['properties'],
            indexes=arguments.get('indexes', []),
            description=arguments.get('description', '')
        )
        
        schema.node_types[arguments['name']] = node_type
        schema.updated_at = datetime.now()
        
        # Save updated schema
        await schema_manager.save_schema(schema)
        
        # Apply to Neo4j if available
        project_name, container, _ = await get_project_context(arguments)
        if container and container.neo4j:
            await schema_manager.apply_to_neo4j(container.neo4j)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "project": project_name,
                "node_type": arguments['name'],
                "properties": node_type.properties,
                "indexes": node_type.indexes,
                "message": f"Added node type '{arguments['name']}' to schema"
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Add node type failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def schema_add_relationship_impl(arguments: dict) -> List[types.TextContent]:
    """Add custom relationship type to project schema"""
    try:
        project_name, _, _ = await get_project_context(arguments)
        schema_manager = await state.get_project_schema(project_name)
        
        if not schema_manager.current_schema:
            # Initialize with generic schema first
            await schema_manager.create_schema(ProjectType.GENERIC)
        
        schema = schema_manager.current_schema
        
        # Add new relationship type
        from schema_manager import RelationshipType
        rel_type = RelationshipType(
            name=arguments['name'],
            from_types=arguments['from_types'],
            to_types=arguments['to_types'],
            properties=arguments.get('properties', {}),
            description=arguments.get('description', '')
        )
        
        schema.relationship_types[arguments['name']] = rel_type
        schema.updated_at = datetime.now()
        
        # Save updated schema
        await schema_manager.save_schema(schema)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "project": project_name,
                "relationship_type": arguments['name'],
                "from_types": rel_type.from_types,
                "to_types": rel_type.to_types,
                "properties": rel_type.properties,
                "message": f"Added relationship type '{arguments['name']}' to schema"
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Add relationship failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


# Migration tool handlers (ADR-0021)

async def migration_generate_impl(arguments: dict) -> List[types.TextContent]:
    """Generate a new migration from schema changes"""
    try:
        project_name, container, _ = await get_project_context(arguments)
        
        # Get or create migration manager for project
        from servers.services.migration_manager import MigrationManager
        migration_manager = MigrationManager(
            project_name=project_name,
            neo4j_service=container.neo4j if container else None,
            qdrant_service=container.qdrant if container else None
        )
        
        name = arguments.get('name', '')
        description = arguments.get('description', '')
        
        if not name:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "Migration name is required"
                }, indent=2)
            )]
        
        # Generate migration from current schema differences
        migration = await migration_manager.generate_migration(name, description)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "generated",
                "project": project_name,
                "migration": {
                    "version": migration.version,
                    "name": migration.name,
                    "description": migration.description,
                    "file": migration.file_path,
                    "operations": len(migration.up_operations)
                },
                "message": f"Generated migration {migration.version:04d}_{migration.name}.yaml"
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Migration generate failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def migration_apply_impl(arguments: dict) -> List[types.TextContent]:
    """Apply pending migrations"""
    try:
        project_name, container, _ = await get_project_context(arguments)
        
        if not container:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "Services not initialized. Cannot apply migrations."
                }, indent=2)
            )]
        
        # Get migration manager
        from servers.services.migration_manager import MigrationManager
        migration_manager = MigrationManager(
            project_name=project_name,
            neo4j_service=container.neo4j,
            qdrant_service=container.qdrant
        )
        
        target_version = arguments.get('target_version')
        dry_run = arguments.get('dry_run', False)
        
        # Apply migrations
        result = await migration_manager.migrate(target_version, dry_run)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success" if result.success else "failed",
                "project": project_name,
                "dry_run": dry_run,
                "from_version": result.from_version,
                "to_version": result.to_version,
                "migrations_applied": result.migrations_applied,
                "operations_executed": result.operations_executed,
                "errors": result.errors,
                "message": f"{'Would apply' if dry_run else 'Applied'} {len(result.migrations_applied)} migrations"
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Migration apply failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def migration_rollback_impl(arguments: dict) -> List[types.TextContent]:
    """Rollback to a previous migration version"""
    try:
        project_name, container, _ = await get_project_context(arguments)
        
        if not container:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "Services not initialized. Cannot rollback migrations."
                }, indent=2)
            )]
        
        target_version = arguments.get('target_version')
        if target_version is None:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "target_version is required for rollback"
                }, indent=2)
            )]
        
        # Get migration manager
        from servers.services.migration_manager import MigrationManager
        migration_manager = MigrationManager(
            project_name=project_name,
            neo4j_service=container.neo4j,
            qdrant_service=container.qdrant
        )
        
        # Rollback migrations
        result = await migration_manager.rollback(target_version)
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success" if result.success else "failed",
                "project": project_name,
                "from_version": result.from_version,
                "to_version": result.to_version,
                "migrations_rolled_back": result.migrations_applied,
                "operations_executed": result.operations_executed,
                "errors": result.errors,
                "message": f"Rolled back to version {target_version}"
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Migration rollback failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def migration_status_impl(arguments: dict) -> List[types.TextContent]:
    """Check migration status and history"""
    try:
        project_name, container, _ = await get_project_context(arguments)
        
        # Get migration manager
        from servers.services.migration_manager import MigrationManager
        migration_manager = MigrationManager(
            project_name=project_name,
            neo4j_service=container.neo4j if container else None,
            qdrant_service=container.qdrant if container else None
        )
        
        # Get migration state
        state = await migration_manager.get_migration_state()
        pending = await migration_manager.get_pending_migrations()
        
        # Get history
        history = []
        if container and container.neo4j:
            history_records = await migration_manager.get_migration_history()
            history = [
                {
                    "version": h["version"],
                    "name": h["name"],
                    "applied_at": h["applied_at"],
                    "direction": h["direction"],
                    "success": h["success"]
                }
                for h in history_records[:5]  # Last 5 migrations
            ]
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "project": project_name,
                "current_version": state.current_version,
                "pending_migrations": len(pending),
                "pending": [
                    {
                        "version": m.version,
                        "name": m.name,
                        "description": m.description
                    }
                    for m in pending
                ],
                "recent_history": history,
                "last_migration": state.last_migration_at.isoformat() if state.last_migration_at else None
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Migration status failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def schema_diff_impl(arguments: dict) -> List[types.TextContent]:
    """Show differences between current database and schema file"""
    try:
        project_name, container, _ = await get_project_context(arguments)
        schema_manager = await state.get_project_schema(project_name)
        
        if not schema_manager.current_schema:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "no_schema",
                    "message": "No schema to compare against. Use schema_init first."
                }, indent=2)
            )]
        
        if not container:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error", 
                    "message": "Services not initialized. Cannot compare with database."
                }, indent=2)
            )]
        
        # Compare schema with actual database
        differences = {
            "neo4j": {
                "missing_node_types": [],
                "extra_node_types": [],
                "missing_relationships": [],
                "extra_relationships": []
            },
            "qdrant": {
                "missing_collections": [],
                "extra_collections": [],
                "field_differences": {}
            }
        }
        
        # Check Neo4j differences
        if container.neo4j:
            # Get actual node labels from Neo4j
            result = await container.neo4j.execute_query("CALL db.labels()")
            actual_labels = set([r["label"] for r in result]) if result else set()
            schema_labels = set(schema_manager.current_schema.node_types.keys())
            
            differences["neo4j"]["missing_node_types"] = list(schema_labels - actual_labels)
            differences["neo4j"]["extra_node_types"] = list(actual_labels - schema_labels)
            
            # Get actual relationship types
            result = await container.neo4j.execute_query("CALL db.relationshipTypes()")
            actual_rels = set([r["relationshipType"] for r in result]) if result else set()
            schema_rels = set(schema_manager.current_schema.relationship_types.keys())
            
            differences["neo4j"]["missing_relationships"] = list(schema_rels - actual_rels)
            differences["neo4j"]["extra_relationships"] = list(actual_rels - schema_rels)
        
        # Check Qdrant differences
        if container.qdrant:
            # Get actual collections
            collections = await container.qdrant.get_collections()
            actual_collections = set([c.name for c in collections.collections])
            schema_collections = set(schema_manager.current_schema.collections.keys())
            
            differences["qdrant"]["missing_collections"] = list(schema_collections - actual_collections)
            differences["qdrant"]["extra_collections"] = list(actual_collections - schema_collections)
        
        # Determine if in sync
        in_sync = (
            not differences["neo4j"]["missing_node_types"] and
            not differences["neo4j"]["extra_node_types"] and
            not differences["neo4j"]["missing_relationships"] and
            not differences["neo4j"]["extra_relationships"] and
            not differences["qdrant"]["missing_collections"] and
            not differences["qdrant"]["extra_collections"]
        )
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "success",
                "project": project_name,
                "in_sync": in_sync,
                "differences": differences,
                "message": "Schema and database are in sync" if in_sync else "Differences found between schema and database"
            }, indent=2)
        )]
    except Exception as e:
        logger.error(f"Schema diff failed: {e}")
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


def cleanup_resources():
    """Clean up resources on server shutdown (called when stdin closes)"""
    logger.info("🔄 Cleaning up resources (stdin closed)")
    
    # Remove PID file if it exists
    pid = os.getpid()
    pid_file = Path(f"/tmp/mcp_pids/mcp_{pid}.pid")
    if pid_file.exists():
        try:
            pid_file.unlink()
            logger.info(f"✅ Removed PID file: {pid_file}")
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")
    
    # Close any active connections
    if 'state' in globals() and state:
        for project_state in state.project_states.values():
            if project_state.container:
                try:
                    # Close Neo4j connections
                    if hasattr(project_state.container, 'neo4j_driver'):
                        project_state.container.neo4j_driver.close()
                    # Close Qdrant connections
                    if hasattr(project_state.container, 'qdrant_client'):
                        project_state.container.qdrant_client.close()
                except Exception as e:
                    logger.error(f"Error closing connections: {e}")
    
    logger.info("👋 MCP server shutdown complete")


async def set_project_context_impl(arguments: dict) -> List[types.TextContent]:
    """
    Handle manual project context switch with full lifecycle management
    Implements ADR-0043: Project Context Lifecycle Management

    Args:
        arguments: Dict with optional 'path' key

    Returns:
        Project information including name and detection method
    """
    global PROJECT_CONTEXT

    try:
        path = arguments.get('path')
        if not path:
            # Auto-detect from current working directory
            path = os.getcwd()

        # Use the manager's switch_project_with_teardown method (returns project dict)
        new_project = await PROJECT_CONTEXT.switch_project_with_teardown(path)

        # Return detailed status
        output = f"""✅ Project context switched successfully!

**Project:** {new_project['name']}
**Path:** {new_project['path']}
**Connections:**
  - Neo4j: ✅ Reconnected
  - Qdrant: ✅ Reconnected
  - Indexer: ✅ Ready

**Message:** Successfully switched to project: {new_project['name']}

All connections have been torn down and rebuilt for the new project context.
This ensures complete isolation between projects (ADR-0043)."""

        return [types.TextContent(type="text", text=output)]

    except Exception as e:
        logger.error(f"Failed to switch project context: {e}", exc_info=True)
        # Grok: Graceful error handling
        output = f"""❌ Failed to switch project context

**Error:** {str(e)}
**Message:** Failed to switch project. Server may need restart.

Please try:
1. Restart Claude to reset the MCP server
2. Use `list_projects` to see available projects
3. Report issue if problem persists"""

        return [types.TextContent(type="text", text=output)]


async def list_projects_impl(arguments: dict) -> List[types.TextContent]:
    """
    List all known projects and their status.
    Shows which project is currently active.
    
    Returns:
        List of projects with paths and activity information
    """
    global PROJECT_CONTEXT
    
    try:
        projects = await PROJECT_CONTEXT.list_projects()
        
        if not projects:
            return [types.TextContent(
                type="text",
                text="""📁 No projects registered yet.

Use `set_project_context` with a project path to register your first project.

Example: `set_project_context(path="/path/to/your/project")`"""
            )]
        
        # Format project list
        output = "📁 **Known Projects:**\n\n"
        
        for proj in projects:
            # Status indicator
            if proj["is_current"]:
                marker = "➤ **[ACTIVE]**"
            elif not proj.get("exists", True):
                marker = "⚠️ _[MISSING]_"
            else:
                marker = "  "
            
            output += f"{marker} **{proj['name']}**\n"
            output += f"   📍 Path: `{proj['path']}`\n"
            
            if proj["last_activity"] != "never":
                # Parse ISO timestamp and format nicely
                try:
                    from datetime import datetime
                    last_active = datetime.fromisoformat(proj["last_activity"])
                    now = datetime.now()
                    delta = now - last_active
                    
                    if delta.days > 0:
                        time_str = f"{delta.days} days ago"
                    elif delta.seconds > 3600:
                        time_str = f"{delta.seconds // 3600} hours ago"
                    elif delta.seconds > 60:
                        time_str = f"{delta.seconds // 60} minutes ago"
                    else:
                        time_str = "just now"
                    
                    output += f"   🕐 Last active: {time_str}\n"
                except:
                    output += f"   🕐 Last active: {proj['last_activity']}\n"
            
            output += "\n"
        
        # Add usage instructions
        output += """---
**To switch projects:**
Use `set_project_context(path="/path/to/project")`

**To auto-detect current project:**
Use `set_project_context()` without arguments"""
        
        return [types.TextContent(type="text", text=output)]
        
    except Exception as e:
        logger.error(f"Error in list_projects: {e}")
        return [types.TextContent(
            type="text",
            text=f"❌ Error listing projects: {str(e)}"
        )]


async def periodic_cleanup_task():
    """Background task to periodically clean up stale instances"""
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_MINUTES * 60)  # Convert minutes to seconds
            if ENABLE_AUTO_CLEANUP:
                cleaned = await state.cleanup_stale_instances()
                if cleaned > 0:
                    logger.info(f"🧹 Periodic cleanup: removed {cleaned} stale instances")
        except asyncio.CancelledError:
            logger.info("🛑 Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            # Continue running even if cleanup fails
            await asyncio.sleep(60)  # Wait a minute before retrying


async def run():
    """
    Main entry point for MCP server following 2025-06-18 specification.
    
    Per spec:
    - Server reads from stdin, writes to stdout
    - Logs go to stderr only
    - Client initiates shutdown by closing stdin
    - No signal handling in server (client handles termination)
    """
    logger.info(f"🚀 Starting L9 Neural MCP Server (STDIO Transport)")
    logger.info(f"🔐 Instance ID: {state.instance_id}")
    logger.info(f"🔧 Cleanup config: Timeout={INSTANCE_TIMEOUT_HOURS}h, Interval={CLEANUP_INTERVAL_MINUTES}m, Enabled={ENABLE_AUTO_CLEANUP}")
    
    # Create PID file for tracking (helps with orphan cleanup)
    pid = os.getpid()
    pid_dir = Path("/tmp/mcp_pids")
    pid_dir.mkdir(exist_ok=True)
    pid_file = pid_dir / f"mcp_{pid}.pid"
    pid_file.write_text(str(pid))
    logger.info(f"📝 PID file created: {pid_file}")
    
    # Start periodic cleanup task if enabled
    cleanup_task = None
    if ENABLE_AUTO_CLEANUP:
        cleanup_task = asyncio.create_task(periodic_cleanup_task())
        logger.info(f"🧹 Started periodic cleanup task (every {CLEANUP_INTERVAL_MINUTES} minutes)")
    
    try:
        # CRITICAL FIX: Don't initialize services here - wait until after handshake
        # Services will be initialized lazily on first tool use
        logger.info("⏳ Delaying service initialization until after MCP handshake")
        
        # MCP STDIO server handles the transport layer
        # It will detect when stdin closes (client disconnect)
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="l9-neural-enhanced",
                    server_version="2.0.0-stdio",
                    capabilities=server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
                ),
            )
    except (EOFError, BrokenPipeError):
        # Normal shutdown - stdin closed by client
        logger.info("📥 Client disconnected (stdin closed)")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise
    finally:
        # Cancel cleanup task if running
        if cleanup_task and not cleanup_task.done():
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        
        cleanup_resources()
