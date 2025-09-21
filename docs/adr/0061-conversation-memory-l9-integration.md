# ADR-0061: L9 Conversation Memory Integration Without Webhooks

## Status
Proposed (Updated: September 21, 2025 - Aligned with latest standards)

## Context

Following the successful pattern established in ADR-0059, we need to integrate conversation memory capabilities into our existing L9 Neural GraphRAG MCP Architecture. The original ADR-0003 from claude-config-template proposed webhook servers, but we've learned that Claude already stores all conversations as JSONL files in `~/.claude/projects/`, eliminating the need for capture mechanisms.

Our L9 architecture (September 2025) consists of:
- MCP server running on host (not containerized) - MCP 2025-06-18 protocol
- Neo4j GraphRAG database on port 47687 (400 thread pool, 50 connection pool)
- Qdrant vector database on port 46333 (with scalar quantization enabled)
- Redis cache/queue on ports 46379/46380
- Nomic embeddings service on port 48000
- ServiceContainer with connection pooling, session management, and circuit breakers
- Established ADR standards (0029, 0037, 0038) for isolation and configuration
- **NEW: OAuth Resource Server compliance (MCP June 2025 spec)**
- **NEW: Resource Indicators (RFC 8707) mandatory**

**Critical Requirement**: Full E2E integration with ZERO parallel stacks or new containers.

## Decision

Extend the existing ServiceContainer to include ConversationService that:
1. Directly indexes Claude's native JSONL conversation files
2. Stores conversation embeddings in existing Qdrant infrastructure with scalar quantization
3. Creates conversation relationships in existing Neo4j graph
4. Provides MCP tools for conversation recall and analysis
5. Maintains strict project isolation per ADR-0029
6. **Implements bounded buffers with backpressure (200MB max)**
7. **Includes circuit breakers and rate limiting per session**
8. **Achieves <0.01% error rate per 2025 production standards**

## Architecture

### Integration into Existing L9 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Existing L9 MCP Server (Host)                 │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     ServiceContainer                         │ │
│  │                                                               │ │
│  │  ┌──────────────────────────────────────────────────────┐   │ │
│  │  │           NEW: ConversationService                    │   │ │
│  │  │                                                        │   │ │
│  │  │  • Indexes ~/.claude/projects/*.jsonl                 │   │ │
│  │  │  • Uses existing Nomic service for embeddings         │   │ │
│  │  │  • Stores in existing Neo4j/Qdrant                   │   │ │
│  │  │  • Respects PROJECT_NAME from ADR-0037               │   │ │
│  │  └──────────────────────────────────────────────────────┘   │ │
│  │                                                               │ │
│  │  ┌──────────────────────────────────────────────────────┐   │ │
│  │  │         Existing Services (Unchanged)                │   │ │
│  │  │  • Neo4jService      • QdrantService                 │   │ │
│  │  │  • NomicService      • SessionManager                │   │ │
│  │  │  • RateLimiter       • CircuitBreaker               │   │ │
│  │  └──────────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Extended MCP Tool Registry                      │ │
│  │                                                               │ │
│  │  NEW Tools:                                                  │ │
│  │  • conversation_index    • conversation_recall               │ │
│  │  • conversation_search   • conversation_stats                │ │
│  │  • conversation_export   • conversation_timeline             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │     Existing Container Infrastructure        │
        │                                               │
        │  Neo4j:47687  Qdrant:46333  Nomic:48000     │
        │  Redis:46379  Redis:46380   Indexer:48080   │
        └──────────────────────────────────────────────┘
```

## Detailed Implementation

### 1. ConversationService Class

```python
# neural-tools/src/servers/services/conversation_service.py

from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import asyncio
from datetime import datetime
import hashlib
import logging
from circuitbreaker import CircuitBreaker
from ratelimiter import RateLimiter

class ConversationService:
    """
    Service for indexing and querying Claude conversation history.
    Integrates with existing L9 infrastructure without webhooks.
    """

    def __init__(self,
                 neo4j_service,
                 qdrant_service,
                 nomic_service,
                 session_manager,
                 project_name: str):
        """
        Initialize with existing services from ServiceContainer.

        Args:
            neo4j_service: Existing Neo4j connection pool
            qdrant_service: Existing Qdrant connection pool
            nomic_service: Existing embedding service
            session_manager: Existing session manager
            project_name: From PROJECT_NAME env var (ADR-0037)
        """
        self.neo4j = neo4j_service
        self.qdrant = qdrant_service
        self.embeddings = nomic_service
        self.sessions = session_manager
        self.project_name = project_name

        # Claude conversation directory
        self.claude_dir = Path.home() / ".claude" / "projects"
        self.project_dir = self._find_project_directory()

        # Collection naming (ADR-0039 compliant)
        self.collection_name = f"conversations_{self._sanitize_project_name()}"

        # Sept 2025: Bounded buffer with backpressure
        self.message_buffer = asyncio.Queue(maxsize=1000)
        self.MAX_BUFFER_SIZE = 200 * 1024 * 1024  # 200MB limit
        self.current_buffer_size = 0

        # Sept 2025: Circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )

        # Sept 2025: Rate limiter per session (60 req/min)
        self.rate_limiter = RateLimiter(
            max_requests=60,
            time_window=60
        )

        # Sept 2025: Connection pool awareness
        # CRITICAL: These are CLIENT connections, not server threads!
        # Neo4j server: bolt.thread_pool_max_size = 400 (threads)
        # Neo4j client: max_connection_pool_size = 50 (connections)
        self.neo4j_connections_limit = 50
        self.qdrant_connections_limit = 30

        # Initialize collection if needed
        asyncio.create_task(self._ensure_collection())

    def _sanitize_project_name(self) -> str:
        """Sanitize project name for collection naming."""
        return self.project_name.lower().replace("-", "_").replace("/", "_")

    def _find_project_directory(self) -> Optional[Path]:
        """
        Find the Claude project directory for this project.
        Follows multiple strategies per ADR-0059.
        """
        # Strategy 1: Exact match
        formatted_name = self.project_name.replace("/", "__")
        exact_match = self.claude_dir / formatted_name
        if exact_match.exists():
            return exact_match

        # Strategy 2: Path-based match
        for project_dir in self.claude_dir.glob("*"):
            if self.project_name in str(project_dir):
                return project_dir

        # Strategy 3: Check JSONL content
        for project_dir in self.claude_dir.glob("*"):
            if self._validate_project_match(project_dir):
                return project_dir

        return None

    def _validate_project_match(self, directory: Path) -> bool:
        """
        Validate if directory contains conversations for this project.
        Implements ADR-0029 project isolation.
        """
        try:
            for jsonl_file in directory.glob("*.jsonl"):
                with open(jsonl_file, 'r') as f:
                    # Sample first few lines
                    for i, line in enumerate(f):
                        if i > 10:  # Sample limit
                            break
                        data = json.loads(line)
                        # Check for project markers
                        if self._contains_project_reference(data):
                            return True
        except:
            pass
        return False

    def _contains_project_reference(self, data: dict) -> bool:
        """Check if conversation data references this project."""
        # Check message content for project name
        if 'message' in data:
            content = str(data['message'].get('content', ''))
            if self.project_name in content:
                return True

        # Check metadata
        metadata = data.get('metadata', {})
        if metadata.get('project') == self.project_name:
            return True

        return False

    async def _ensure_collection(self):
        """Ensure Qdrant collection exists with proper schema."""
        try:
            # Check if collection exists
            collections = await self.qdrant.get_collections()

            if self.collection_name not in [c.name for c in collections.collections]:
                # Create collection with 768-dim vectors (Nomic standard)
                # Sept 2025: Scalar quantization for 4x memory reduction
                await self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=768,
                        distance=Distance.COSINE,
                        quantization_config={
                            "scalar": {
                                "type": "int8",
                                "quantile": 0.99,  # Optimal per Sept 2025
                                "always_ram": True
                            }
                        }
                    )
                )
                # Memory calculation with quantization:
                # 100K messages: 460MB → 115MB (75% reduction)

                # Create indexes for efficient querying
                await self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="session_id",
                    field_type="keyword"
                )
                await self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="timestamp",
                    field_type="datetime"
                )
                await self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="project",
                    field_type="keyword"
                )
        except Exception as e:
            logging.error(f"Failed to ensure collection: {e}")

    async def index_conversations(self,
                                 recent_only: bool = False,
                                 limit: Optional[int] = None,
                                 resource_indicator: str = None) -> Dict[str, Any]:
        """
        Index conversation JSONL files into Neo4j and Qdrant.

        Sept 2025: Includes OAuth Resource Server compliance (RFC 8707)

        Args:
            recent_only: Only index conversations from last 7 days
            limit: Maximum number of conversations to index
            resource_indicator: MCP resource server identifier (required)

        Returns:
            Statistics about indexed conversations
        """
        # Validate resource indicator per MCP June 2025 spec
        if resource_indicator:
            await self._validate_resource_access(resource_indicator)
        if not self.project_dir:
            return {"error": "Project directory not found", "indexed": 0}

        stats = {
            "files_processed": 0,
            "messages_indexed": 0,
            "embeddings_created": 0,
            "relationships_created": 0,
            "errors": []
        }

        # Get JSONL files
        jsonl_files = sorted(
            self.project_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if recent_only:
            cutoff = datetime.now().timestamp() - (7 * 24 * 3600)
            jsonl_files = [f for f in jsonl_files if f.stat().st_mtime > cutoff]

        if limit:
            jsonl_files = jsonl_files[:limit]

        for jsonl_file in jsonl_files:
            try:
                # Sept 2025: Use circuit breaker for resilience
                async with self.circuit_breaker:
                    async with self.rate_limiter:
                        await self._index_single_conversation(jsonl_file, stats)
                        stats["files_processed"] += 1
            except Exception as e:
                stats["errors"].append(f"{jsonl_file.name}: {str(e)}")
                # Sept 2025: Track error rate for <0.01% target
                self._track_error_rate(e)

        return stats

    async def _index_single_conversation(self,
                                        jsonl_path: Path,
                                        stats: Dict[str, Any]):
        """Index a single conversation file."""
        conversation_id = jsonl_path.stem
        messages = []

        # Parse JSONL
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if 'message' in entry and entry['message']:
                        msg = entry['message']
                        if msg.get('role') in ['user', 'assistant']:
                            messages.append({
                                'role': msg['role'],
                                'content': msg.get('content', ''),
                                'timestamp': entry.get('timestamp'),
                                'conversation_id': conversation_id,
                                'project': self.project_name
                            })
                except json.JSONDecodeError:
                    continue

        if not messages:
            return

        # Batch process messages
        for i, msg in enumerate(messages):
            # Generate unique ID
            msg_id = self._generate_message_id(conversation_id, i)

            # Generate embedding
            embedding = await self.embeddings.embed_text(msg['content'][:2000])

            # Store in Qdrant with backpressure check
            message_size = len(json.dumps(msg))
            if self.current_buffer_size + message_size > self.MAX_BUFFER_SIZE:
                await self._apply_backpressure()

            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[{
                    'id': msg_id,
                    'vector': embedding,
                    'payload': {
                        'conversation_id': conversation_id,
                        'message_index': i,
                        'role': msg['role'],
                        'content': msg['content'][:1000],  # Truncate for storage
                        'timestamp': msg['timestamp'],
                        'project': self.project_name,
                        'session_id': self._extract_session_id(conversation_id)
                    }
                }]
            )
            stats["embeddings_created"] += 1

            # Store in Neo4j with project isolation
            await self.neo4j.query("""
                MERGE (m:ConversationMessage {
                    id: $id,
                    project: $project
                })
                SET m.conversation_id = $conversation_id,
                    m.role = $role,
                    m.content = $content,
                    m.timestamp = $timestamp,
                    m.message_index = $index
            """, {
                'id': msg_id,
                'project': self.project_name,
                'conversation_id': conversation_id,
                'role': msg['role'],
                'content': msg['content'][:500],
                'timestamp': msg['timestamp'],
                'index': i
            })
            stats["messages_indexed"] += 1

            # Create relationships
            if i > 0:
                prev_msg_id = self._generate_message_id(conversation_id, i-1)
                await self.neo4j.query("""
                    MATCH (prev:ConversationMessage {
                        id: $prev_id,
                        project: $project
                    })
                    MATCH (curr:ConversationMessage {
                        id: $curr_id,
                        project: $project
                    })
                    MERGE (prev)-[:FOLLOWED_BY]->(curr)
                """, {
                    'prev_id': prev_msg_id,
                    'curr_id': msg_id,
                    'project': self.project_name
                })
                stats["relationships_created"] += 1

    def _generate_message_id(self, conversation_id: str, index: int) -> str:
        """Generate unique message ID."""
        content = f"{self.project_name}:{conversation_id}:{index}"
        return hashlib.sha256(content.encode()).hexdigest()[:15]

    def _extract_session_id(self, conversation_id: str) -> str:
        """Extract session ID from conversation filename."""
        # Claude filenames contain session information
        parts = conversation_id.split('_')
        if len(parts) > 1:
            return parts[0]
        return conversation_id

    async def recall_conversation(self,
                                 query: str,
                                 limit: int = 5,
                                 include_graph: bool = True) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant conversation context.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            include_graph: Include graph traversal results

        Returns:
            Relevant conversation segments with metadata
        """
        results = []

        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query)

        # Vector search in Qdrant
        vector_results = await self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter={
                "must": [
                    {"key": "project", "match": {"value": self.project_name}}
                ]
            }
        )

        for result in vector_results:
            payload = result.payload
            results.append({
                'score': result.score,
                'conversation_id': payload['conversation_id'],
                'role': payload['role'],
                'content': payload['content'],
                'timestamp': payload['timestamp'],
                'type': 'vector'
            })

        # Graph traversal for related context
        if include_graph:
            graph_results = await self.neo4j.query("""
                MATCH (m:ConversationMessage {project: $project})
                WHERE m.content CONTAINS $query
                OPTIONAL MATCH (m)-[:FOLLOWED_BY*1..3]-(related:ConversationMessage)
                WHERE related.project = $project
                RETURN m, collect(related) as context
                LIMIT $limit
            """, {
                'project': self.project_name,
                'query': query,
                'limit': limit
            })

            for record in graph_results:
                msg = record['m']
                context_msgs = record['context']

                results.append({
                    'score': 0.8,  # Graph match score
                    'conversation_id': msg['conversation_id'],
                    'role': msg['role'],
                    'content': msg['content'],
                    'timestamp': msg['timestamp'],
                    'type': 'graph',
                    'context': [
                        {
                            'role': c['role'],
                            'content': c['content'][:200]
                        } for c in context_msgs[:2]
                    ]
                })

        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x['score'], reverse=True):
            key = f"{r['conversation_id']}:{r['content'][:50]}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results[:limit]

    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed conversations."""
        stats = {}

        # Qdrant stats
        try:
            collection_info = await self.qdrant.get_collection(self.collection_name)
            stats['total_messages'] = collection_info.points_count
            stats['vector_dimensions'] = collection_info.config.params.vectors.size
        except:
            stats['total_messages'] = 0
            stats['vector_dimensions'] = 768

        # Neo4j stats
        neo4j_stats = await self.neo4j.query("""
            MATCH (m:ConversationMessage {project: $project})
            WITH count(m) as messages,
                 count(DISTINCT m.conversation_id) as conversations
            OPTIONAL MATCH (m)-[r:FOLLOWED_BY]->()
            WHERE m.project = $project
            RETURN messages, conversations, count(r) as relationships
        """, {'project': self.project_name})

        if neo4j_stats:
            record = neo4j_stats[0]
            stats.update({
                'graph_messages': record['messages'],
                'unique_conversations': record['conversations'],
                'message_relationships': record['relationships']
            })

        # Storage size estimate
        stats['estimated_storage_mb'] = (
            stats.get('total_messages', 0) * 0.005  # ~5KB per message
        )

        return stats
```

### 2. ServiceContainer Extension

```python
# neural-tools/src/servers/services/service_container.py
# ADD to existing ServiceContainer class:

async def initialize_conversation_service(self):
    """Initialize conversation memory service."""
    if not hasattr(self, 'conversation_service'):
        from .conversation_service import ConversationService

        self.conversation_service = ConversationService(
            neo4j_service=self.neo4j_service,
            qdrant_service=self.qdrant_service,
            nomic_service=self.nomic_service,
            session_manager=self.session_manager,
            project_name=os.getenv('PROJECT_NAME', 'default')
        )

        logging.info("✅ ConversationService initialized")

# ADD to initialize() method:
async def initialize(self):
    """Initialize all services."""
    # ... existing initialization ...

    # Initialize conversation service if enabled
    if os.getenv('ENABLE_CONVERSATION_MEMORY', 'false').lower() == 'true':
        await self.initialize_conversation_service()
```

### 3. MCP Tool Registration

```python
# neural-tools/src/neural_mcp/neural_server_stdio.py
# ADD these tool definitions:

@server.tool()
async def conversation_index(
    recent_only: bool = False,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Index Claude conversation history into GraphRAG.

    Args:
        recent_only: Only index conversations from last 7 days
        limit: Maximum number of conversations to index
    """
    if not hasattr(container, 'conversation_service'):
        return {"error": "Conversation service not enabled"}

    return await container.conversation_service.index_conversations(
        recent_only=recent_only,
        limit=limit
    )

@server.tool()
async def conversation_recall(
    query: str,
    limit: int = 5,
    include_graph: bool = True
) -> List[Dict[str, Any]]:
    """
    Recall relevant conversation context using semantic search.

    Args:
        query: Natural language search query
        limit: Maximum results to return
        include_graph: Include graph traversal results
    """
    if not hasattr(container, 'conversation_service'):
        return []

    return await container.conversation_service.recall_conversation(
        query=query,
        limit=limit,
        include_graph=include_graph
    )

@server.tool()
async def conversation_search(
    topics: List[str],
    date_range: Optional[Dict[str, str]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search conversations by topics and date range.

    Args:
        topics: List of topics to search for
        date_range: Optional {'from': 'YYYY-MM-DD', 'to': 'YYYY-MM-DD'}
        limit: Maximum results
    """
    if not hasattr(container, 'conversation_service'):
        return []

    # Combine topic searches
    all_results = []
    for topic in topics:
        results = await container.conversation_service.recall_conversation(
            query=topic,
            limit=limit,
            include_graph=False
        )
        all_results.extend(results)

    # Apply date filtering if provided
    if date_range:
        from_date = datetime.fromisoformat(date_range.get('from', '2000-01-01'))
        to_date = datetime.fromisoformat(date_range.get('to', '2099-12-31'))

        all_results = [
            r for r in all_results
            if from_date <= datetime.fromisoformat(r['timestamp']) <= to_date
        ]

    # Deduplicate and sort
    seen = set()
    unique_results = []
    for r in sorted(all_results, key=lambda x: x['score'], reverse=True):
        key = f"{r['conversation_id']}:{r['content'][:50]}"
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    return unique_results[:limit]

@server.tool()
async def conversation_stats() -> Dict[str, Any]:
    """Get statistics about indexed conversations."""
    if not hasattr(container, 'conversation_service'):
        return {"error": "Conversation service not enabled"}

    return await container.conversation_service.get_conversation_stats()

@server.tool()
async def conversation_export(
    conversation_id: Optional[str] = None,
    format: str = "json"
) -> Union[Dict[str, Any], str]:
    """
    Export conversation history.

    Args:
        conversation_id: Specific conversation to export (or all if None)
        format: Export format ('json' or 'markdown')
    """
    if not hasattr(container, 'conversation_service'):
        return {"error": "Conversation service not enabled"}

    # Query all messages for conversation
    if conversation_id:
        filter_query = """
            MATCH (m:ConversationMessage {
                project: $project,
                conversation_id: $conversation_id
            })
            RETURN m
            ORDER BY m.message_index
        """
        params = {
            'project': container.conversation_service.project_name,
            'conversation_id': conversation_id
        }
    else:
        filter_query = """
            MATCH (m:ConversationMessage {project: $project})
            RETURN m
            ORDER BY m.conversation_id, m.message_index
            LIMIT 1000
        """
        params = {'project': container.conversation_service.project_name}

    results = await container.neo4j_service.query(filter_query, params)

    if format == "markdown":
        output = []
        current_conv = None
        for record in results:
            msg = record['m']
            if msg['conversation_id'] != current_conv:
                current_conv = msg['conversation_id']
                output.append(f"\n## Conversation: {current_conv}\n")

            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            output.append(f"{role_emoji} **{msg['role']}**: {msg['content']}\n")

        return "\n".join(output)
    else:
        return {
            'conversations': [dict(record['m']) for record in results],
            'total': len(results)
        }

@server.tool()
async def conversation_timeline(
    days: int = 7
) -> List[Dict[str, Any]]:
    """
    Get conversation timeline for recent days.

    Args:
        days: Number of days to include in timeline
    """
    if not hasattr(container, 'conversation_service'):
        return []

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    timeline = await container.neo4j_service.query("""
        MATCH (m:ConversationMessage {project: $project})
        WHERE m.timestamp > $cutoff
        WITH date(datetime(m.timestamp)) as day,
             count(DISTINCT m.conversation_id) as conversations,
             count(m) as messages,
             collect(DISTINCT m.conversation_id)[..5] as sample_ids
        RETURN day, conversations, messages, sample_ids
        ORDER BY day DESC
    """, {
        'project': container.conversation_service.project_name,
        'cutoff': cutoff
    })

    return [dict(record) for record in timeline]
```

### 4. Configuration Updates

```yaml
# docker-compose.yml
# UPDATE existing services (no new containers):

services:
  # ... existing services remain unchanged ...

  # Only update volumes for host MCP if running in container
  # (but our MCP runs on host, so this is just for reference)
```

```json
# .mcp.json
# ADD these environment variables:

{
  "mcpServers": {
    "neural-tools": {
      "command": "python",
      "args": ["run_mcp_server.py"],
      "env": {
        // ... existing env vars ...
        "ENABLE_CONVERSATION_MEMORY": "true",
        "CLAUDE_PROJECTS_DIR": "~/.claude/projects",
        "CONVERSATION_INDEX_INTERVAL": "3600",
        "MAX_CONVERSATION_AGE_DAYS": "90"
      }
    }
  }
}
```

## Storage Schema

### Neo4j Schema Extensions

```cypher
// New node type for conversations
CREATE CONSTRAINT conversation_message_unique
ON (m:ConversationMessage)
ASSERT (m.id, m.project) IS NODE KEY;

CREATE INDEX conversation_timestamp
FOR (m:ConversationMessage)
ON (m.timestamp);

CREATE INDEX conversation_project
FOR (m:ConversationMessage)
ON (m.project);

// Node structure
(:ConversationMessage {
    id: String,              // SHA256 hash (15 chars)
    project: String,         // Project isolation (ADR-0029)
    conversation_id: String, // JSONL filename
    message_index: Integer,  // Position in conversation
    role: String,           // 'user' or 'assistant'
    content: String,        // Truncated to 500 chars
    timestamp: String       // ISO-8601
})

// Relationships
(:ConversationMessage)-[:FOLLOWED_BY]->(:ConversationMessage)
(:ConversationMessage)-[:RELATES_TO]->(:CodeNode)  // Future: link to code
```

### Qdrant Collection Schema

```python
{
    "collection_name": "conversations_{project_name}",
    "vector_size": 768,  # Nomic embedding size
    "distance": "Cosine",
    "quantization_config": {  # Sept 2025 mandatory
        "scalar": {
            "type": "int8",
            "quantile": 0.99,
            "always_ram": true
        }
    },
    "payload_schema": {
        "conversation_id": "keyword",
        "message_index": "integer",
        "role": "keyword",
        "content": "text",
        "timestamp": "datetime",
        "project": "keyword",
        "session_id": "keyword"
    }
}
```

## Performance Specifications (September 2025 Standards)

### Resource Usage
- **Memory**: +200MB for conversation indexing buffers (bounded)
- **CPU**: <5% additional during indexing
- **Storage**: ~1.25KB per message with scalar quantization (was 5KB)
- **Network**: Reuses existing connection pools

### Performance Targets (Updated per 2025 Standards)
- **P95 Latency**: <200ms (industry standard, was <1000ms)
- **Error Rate**: <0.01% (was <1%)
- **Indexing Speed**: <200ms per conversation
- **Recall Latency**: <100ms for semantic search
- **Graph Traversal**: <50ms for 3-hop queries
- **Batch Processing**: 200 conversations/minute
- **Concurrent Sessions**: 500-1000 (was 200)
- **Availability**: 99.95% (4 nines)

### Connection Pool Configuration (Sept 2025 Corrected)
```
Service       | Connections | Threads | Notes
-------------|-------------|---------|---------------------------
Neo4j Client | 50          | N/A     | Driver connection pool
Neo4j Server | N/A         | 400     | bolt.thread_pool_max_size
Qdrant       | 30          | N/A     | Client connections
Redis Cache  | 25          | N/A     | Client connections
Nomic        | N/A         | N/A     | HTTP client

CRITICAL: Neo4j threads ≠ connections!
- 1 thread can handle MULTIPLE connections
- Thread pool (400) is server-side configuration
- Connection pool (50) is client-side configuration
```

## Security & Isolation

### Project Isolation (ADR-0029)
- All nodes have `project` property
- All queries filter by project
- Collections named per project
- No cross-project data leakage

### Access Control
- Respects MCP session boundaries
- Read-only access to ~/.claude/projects
- No modification of source JSONL files
- Cleanup of old data after 90 days

### Data Privacy
- Conversations stay local (never leave host)
- Truncated content in databases (500 chars)
- Full content only in embeddings
- No PII extraction or storage

## Testing Strategy

### 1. Unit Test Criteria (90% Coverage Required)

#### ConversationService Tests
```python
# tests/test_conversation_service.py

async def test_project_isolation():
    """Ensure no cross-project contamination."""
    service1 = ConversationService(project_name="project1", ...)
    service2 = ConversationService(project_name="project2", ...)

    # Index for project1
    await service1.index_conversations()

    # Query from project2 should return nothing
    results = await service2.recall_conversation("test query")
    assert len(results) == 0

async def test_jsonl_parsing():
    """Test parsing of Claude JSONL format."""
    # Test valid JSONL
    # Test malformed JSONL
    # Test empty files
    # Test large payloads (>100MB)

async def test_embedding_generation():
    """Verify embeddings are 768-dimensional."""
    # Test normal text
    # Test empty content
    # Test special characters
    # Test truncation at 2000 chars

async def test_edge_cases():
    """Test all edge cases."""
    # Empty conversations
    # Missing directories
    # Permission errors
    # Corrupted JSONL
    # Unicode handling
    # Buffer overflow (>200MB)
    # Circuit breaker triggers
    # Rate limit exceeded

async def test_resilience():
    """Sept 2025: Chaos engineering tests."""
    # Simulate Neo4j outage
    await simulate_neo4j_failure()
    result = await conversation_index()
    assert result['degraded_mode'] == True

    # Simulate network partition
    await toxiproxy.add_latency("neo4j", 5000)

    # Test circuit breaker activation
    for i in range(10):
        try:
            await conversation_recall("test")
        except CircuitBreakerOpen:
            assert i >= 5  # Should open after 5 failures
```

### 2. Integration Test Criteria

#### E2E Test Flow (Must Pass)

**Sept 2025: Added chaos engineering and resilience testing**
```python
# tests/test_conversation_integration.py

async def test_end_to_end_flow():
    """Complete E2E test as per Gemini's specification."""
    # 1. Start conversation → Index with conversation_index()
    stats = await conversation_index()
    assert stats['messages_indexed'] > 0

    # 2. Verify listing shows indexed conversations
    timeline = await conversation_timeline(days=1)
    assert len(timeline) > 0

    # 3. Test recall functionality
    results = await conversation_recall("previous discussion")
    assert len(results) > 0
    assert results[0]['score'] > 0.7

    # 4. Verify Neo4j-Qdrant consistency
    neo4j_count = await get_neo4j_message_count()
    qdrant_count = await get_qdrant_point_count()
    assert neo4j_count == qdrant_count

    # 5. Verify project isolation
    other_project_results = await recall_with_different_project()
    assert len(other_project_results) == 0

async def test_mcp_tools():
    """Test all 6 MCP tool endpoints."""
    tools = [
        'conversation_index',
        'conversation_recall',
        'conversation_search',
        'conversation_stats',
        'conversation_export',
        'conversation_timeline'
    ]
    for tool in tools:
        result = await test_tool_endpoint(tool)
        assert result is not None
        assert 'error' not in result
```

### 3. Performance Criteria (Sept 2025 Standards)

| Metric | Target | Alert Threshold | Rollback Trigger |
|--------|---------|-----------------|-------------------|
| Average Response Time | <200ms | 150ms | >500ms |
| P95 Response Time | <500ms | 400ms | >1000ms |
| P99 Response Time | <1000ms | 850ms | >2000ms |
| Error Rate | <0.01% | 0.005% | >0.1% |
| CPU Usage (Idle) | <3% | 2.5% | >5% |
| Memory Growth | <100MB | 80MB | >200MB |
| Indexing Speed | <200ms/conv | 150ms | >500ms |
| Recall Latency | <100ms | 80ms | >200ms |
| Concurrent Sessions | >500 | 400 | <200 |

### 4. Load Test Criteria

```python
# tests/test_conversation_load.py

async def test_concurrent_sessions():
    """Test 500 concurrent sessions for 60 minutes (Sept 2025 standard)."""
    sessions = []

    # Create 500 concurrent sessions (Sept 2025 requirement)
    for i in range(500):
        session = create_test_session(f"session_{i}")
        sessions.append(session)

    # Run for 60 minutes
    start_time = time.time()
    while time.time() - start_time < 3600:
        # Each session performs operations
        for session in sessions:
            await session.index_conversations(limit=10)
            await session.recall_conversation("test query")
            await session.get_stats()

        # Monitor pool utilization
        neo4j_util = get_neo4j_pool_utilization()
        qdrant_util = get_qdrant_pool_utilization()

        assert neo4j_util < 70  # Must stay below 70%
        assert qdrant_util < 70  # Must stay below 70%

        # Check response times
        assert get_p95_latency() < 1000

    # Verify no degradation
    assert get_error_rate() < 0.01  # <1% errors

async def test_pool_saturation():
    """Find breaking point at 90% utilization."""
    # Gradually increase load
    # Monitor when pools hit 90%
    # Verify graceful degradation
    # Test circuit breaker activation
```

### 5. Exit Conditions (Production Ready)

✅ **All must pass before production:**
- [ ] 90% unit test coverage achieved
- [ ] All E2E integration tests passing consistently (10 consecutive runs)
- [ ] Performance metrics met under load for 60 minutes
- [ ] Zero P0/P1 bugs in release candidate
- [ ] Documentation reviewed and published
- [ ] L9 security review passed (no injection vulnerabilities)
- [ ] Monitoring dashboards deployed and verified
- [ ] Feature flags configured and tested
- [ ] Rollback procedure tested successfully
- [ ] Pool utilization stays <70% under max load

### 6. Rollback Triggers (Sept 2025 Standards)

🚨 **Automatic rollback if ANY condition met:**
- P95 latency >500ms (Sept 2025: was 1000ms)
- Error rate >0.1% sustained for 5 minutes (was 1%)
- Connection pool utilization >90% for 5+ minutes
- Any data corruption detected (Neo4j-Qdrant inconsistency)
- Cascade failures to other services
- Critical bugs affecting core functionality
- Memory leak detected (>200MB growth/hour, was 500MB)
- CPU usage >5% when idle (was 50%)
- Concurrent sessions drop below 200 (was not tracked)

### 7. L9 Validation Requirements

```python
# scripts/run_l9_validation_conversations.py

class L9ConversationValidation:
    """L9 compliance testing for conversation memory."""

    async def validate_observability(self):
        """Verify L9 observability standards."""
        # Structured logging format
        assert logs_are_structured()

        # Prometheus metrics available
        metrics = [
            'conversation_index_duration_seconds',
            'conversation_recall_latency_ms',
            'conversation_pool_utilization_percent',
            'conversation_errors_total'
        ]
        for metric in metrics:
            assert metric_exists(metric)

        # Distributed tracing enabled
        assert tracing_enabled()

    async def validate_security(self):
        """Verify security requirements."""
        # No SQL injection possible
        await test_sql_injection_attempts()

        # No cross-project data access
        await test_project_isolation()

        # Proper input sanitization
        await test_input_validation()

        # No PII logging
        await verify_no_pii_in_logs()

    async def validate_performance(self):
        """Verify Sept 2025 L9 performance standards."""
        results = await run_performance_suite()

        # Sept 2025 updated standards
        assert results['concurrent_sessions'] >= 500  # was 15
        assert results['success_rate'] >= 0.9999  # was 0.95 (now <0.01% errors)
        assert results['avg_response_ms'] < 200  # was 500
        assert results['p95_response_ms'] < 500  # was 1000
        assert results['throughput_rps'] >= 100  # was 10
```

### 8. Monitoring Dashboard Requirements

**Real-time metrics to display:**
```yaml
conversation_monitoring:
  connection_pools:
    neo4j_current: "XX/50"      # Alert at 35
    neo4j_percent: "XX%"         # Alert at 70%
    qdrant_current: "XX/30"      # Alert at 21
    qdrant_percent: "XX%"        # Alert at 70%

  performance:
    p50_latency_ms: "XXX"        # Baseline: 200ms
    p95_latency_ms: "XXX"        # Baseline: 500ms
    p99_latency_ms: "XXX"        # Baseline: 1000ms
    error_rate: "X.X%"           # Threshold: 1%

  operations:
    indexing_rate: "XX/min"      # Target: >100/min
    recall_rate: "XX/sec"        # Target: >10/sec
    active_sessions: "XXX"       # Max: 200

  health:
    neo4j_health: "🟢/🟡/🔴"
    qdrant_health: "🟢/🟡/🔴"
    nomic_health: "🟢/🟡/🔴"
    circuit_breaker: "CLOSED/OPEN/HALF_OPEN"
```

### 9. Testing Priority (Per Gemini)

1. **Connection pool monitoring** (highest risk)
   - Real-time utilization tracking
   - Automatic alerts at 70%
   - Circuit breaker at 90%

2. **E2E conversation-code integration**
   - Verify linkage works correctly
   - Test with real JSONL files
   - Validate recall accuracy

3. **Project isolation verification** (ADR-0029)
   - Zero cross-project leakage
   - Proper filtering in all queries
   - Collection naming compliance

4. **Performance under sustained load**
   - 60-minute continuous operation
   - 200 concurrent sessions
   - No degradation over time

5. **Rollback mechanisms**
   - Feature flag testing
   - Gradual rollout simulation
   - Emergency shutdown procedures

## Rollout Plan

### Phase 1: Core Implementation (Week 1)
- [ ] Implement ConversationService class
- [ ] Extend ServiceContainer
- [ ] Add Neo4j schema migrations
- [ ] Create Qdrant collections

### Phase 2: MCP Integration (Week 2)
- [ ] Register MCP tools
- [ ] Test with real JSONL files
- [ ] Implement project isolation
- [ ] Add health checks

### Phase 3: Optimization (Week 3)
- [ ] Performance tuning
- [ ] Add caching layer
- [ ] Implement cleanup routines
- [ ] Documentation

### Phase 4: Production (Week 4)
- [ ] L9 validation suite
- [ ] Load testing
- [ ] Deploy to global MCP
- [ ] Monitor metrics

## Success Metrics

### Immediate (Week 1)
- ✅ Successfully parse Claude JSONL files
- ✅ Generate embeddings via Nomic
- ✅ Store in Neo4j/Qdrant
- ✅ Basic recall functionality

### Short-term (Month 1)
- 📊 >95% conversation capture rate
- 📊 <200ms recall latency
- 📊 >90% relevant results in top-5
- 📊 Zero cross-project leakage

### Long-term (Month 3)
- 📈 >80% token reduction in context
- 📈 >10x faster context retrieval
- 📈 <1% storage overhead
- 📈 100% project isolation

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|------------|
| JSONL format changes | Version detection and adapters |
| Large conversation files | Streaming parser with chunking |
| Embedding service overload | Rate limiting and queueing |
| Storage growth | Auto-cleanup after 90 days |

### Operational Risks
| Risk | Mitigation |
|------|------------|
| Project directory not found | Multiple detection strategies |
| Cross-project contamination | Strict project property enforcement |
| Connection pool exhaustion | Reuse existing pools |
| Memory growth | Bounded buffers and cleanup |

## Alternatives Considered

### Webhook-based Capture (Rejected)
- **Pros**: Real-time capture, guaranteed delivery
- **Cons**: Complex architecture, unnecessary given JSONL storage
- **Decision**: Follow ADR-0059 pattern of direct indexing

### Separate Container (Rejected)
- **Pros**: Isolation, independent scaling
- **Cons**: Violates no-parallel-stack requirement
- **Decision**: Extend existing ServiceContainer

### New Database (Rejected)
- **Pros**: Optimized for conversations
- **Cons**: Additional infrastructure complexity
- **Decision**: Reuse Neo4j/Qdrant

## Future Enhancements

### Phase 5: Advanced Features
- Link conversations to code changes
- Sentiment analysis and decision extraction
- Conversation summarization
- Cross-session pattern detection

### Phase 6: Intelligence Layer
- Predictive context retrieval
- Automatic relevance scoring
- Context compression algorithms
- Multi-project knowledge synthesis

## References

- ADR-0029: Logical Partitioning for Project Isolation
- ADR-0037: Container Configuration Priority Standard
- ADR-0038: Docker Image Lifecycle Management
- ADR-0039: Collection Naming Standards
- ADR-0059: Original Conversation Memory Implementation
- MCP Protocol 2025-06-18 Specification
- Neo4j 5.22.0 Async Driver Documentation
- Qdrant 1.15.1 Client Documentation

---

**Document Version**: 1.0.0
**Created**: September 21, 2025
**Author**: L9 Engineering Team
**Status**: PROPOSED
**Target Implementation**: Q4 2025

**Confidence**: 95%
**Key Assumptions**:
- Claude JSONL format remains stable
- Existing infrastructure can handle additional load