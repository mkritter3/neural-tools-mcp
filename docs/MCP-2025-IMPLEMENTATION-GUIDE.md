# MCP 2025-06-18 Protocol Implementation Guide

**Created: September 20, 2025**
**Based on official MCP specification 2025-06-18**

## Key Learnings from Research

### 1. MCP Transport Mechanisms

#### STDIO Transport (What we use)
- Client launches MCP server as subprocess
- Messages exchanged via stdin/stdout
- Messages are newline-delimited JSON-RPC 2.0
- stderr used for logging only
- **CRITICAL**: stdout must ONLY contain valid MCP messages

#### Streamable HTTP Transport (Alternative)
- Server operates as independent process
- Uses HTTP POST and GET requests
- Optional Server-Sent Events (SSE) for streaming
- Single MCP endpoint path required

### 2. Session Lifecycle

The correct initialization sequence is:

1. Client sends `initialize` request with:
   - `protocolVersion`: "2025-06-18"
   - `capabilities`: client capabilities
   - `clientInfo`: name, title, version

2. Server responds with `InitializeResult`:
   - `protocolVersion`: agreed version
   - `capabilities`: server capabilities
   - `serverInfo`: server details
   - `instructions`: optional guidance

3. Client sends `notifications/initialized` notification

4. Normal operations begin

### 3. Service Container Architecture

Based on MCP Python SDK patterns:

```python
# CORRECT: Lifespan management with context
@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict[str, Any]]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize resources on startup
    db = await Database.connect()
    try:
        yield {"db": db}
    finally:
        # Clean up on shutdown
        await db.disconnect()

# Access lifespan context in tools
@server.call_tool()
async def query_db(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    ctx = server.request_context
    db = ctx.lifespan_context["db"]
    results = await db.query(arguments["query"])
    return [types.TextContent(type="text", text=f"Results: {results}")]
```

### 4. Tool Registration

Tools must be properly registered with schemas:

```python
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_db",
            description="Query the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"}
                },
                "required": ["query"],
            },
        )
    ]
```

### 5. Critical Implementation Mistakes to Avoid

#### ❌ WRONG: Creating separate indexers/services
```python
# DON'T create new components outside ServiceContainer
class ConversationIndexer:
    def __init__(self):
        self.neo4j = Neo4jService()  # NO! Separate instance
        self.qdrant = QdrantService()  # NO! Separate instance
```

#### ✅ CORRECT: Using existing ServiceContainer
```python
# DO use the existing ServiceContainer
class ConversationMemory:
    def __init__(self, container: ServiceContainer):
        self.container = container
        self.neo4j = container.neo4j  # YES! Shared instance
        self.qdrant = container.qdrant  # YES! Shared instance
```

#### ❌ WRONG: Hardcoded connection parameters
```python
# DON'T hardcode IPs or internal ports
host = "172.18.0.5"  # NO! Docker internal IP
port = 8000  # NO! Internal port
```

#### ✅ CORRECT: Using environment variables with proper defaults
```python
# DO use environment variables with localhost defaults
host = os.getenv('EMBEDDING_SERVICE_HOST', 'localhost')
port = int(os.getenv('EMBEDDING_SERVICE_PORT', 48000))  # Exposed port
```

### 6. Neo4j Index Syntax (Critical!)

#### ❌ WRONG: Generic index without label
```cypher
CREATE INDEX project_property_index IF NOT EXISTS FOR (n) ON (n.project)
```
**Error**: "Invalid input ')': expected ':'"

#### ✅ CORRECT: Label-specific indexes
```cypher
CREATE INDEX file_project_index IF NOT EXISTS FOR (f:File) ON (f.project)
CREATE INDEX class_project_index IF NOT EXISTS FOR (c:Class) ON (c.project)
CREATE INDEX message_project_index IF NOT EXISTS FOR (m:Message) ON (m.project)
```

### 7. Import Path Issues

#### ❌ WRONG: Mixed import styles causing module identity issues
```python
# In one file:
from src.servers.services.project_context_manager import ProjectContextManager
# In another:
from servers.services.project_context_manager import ProjectContextManager
# These create DIFFERENT module identities!
```

#### ✅ CORRECT: Consistent absolute imports
```python
# Always use the same import path throughout
from servers.services.project_context_manager import ProjectContextManager
```

### 8. MCP Server Capabilities

Servers declare capabilities during initialization:

- **logging**: Send log messages to client
- **prompts**: Provide prompt templates
- **resources**: Expose data resources
- **tools**: Offer executable functions
- **completions**: Provide auto-completion

Example:
```json
{
  "capabilities": {
    "tools": {
      "listChanged": true
    },
    "resources": {
      "subscribe": true,
      "listChanged": true
    }
  }
}
```

### 9. Conversation Memory Implementation Strategy

For ADR-0059, the correct approach should be:

1. **Parse JSONL files** from `~/.claude/projects/{project}/*.jsonl`
2. **Use existing ServiceContainer** for Neo4j/Qdrant access
3. **Leverage WriteSynchronizationManager** (ADR-053) for atomic writes
4. **Store in both databases**:
   - Neo4j: Graph structure (Conversation → Message nodes)
   - Qdrant: Vector embeddings for semantic search
5. **Expose via MCP tools** using existing tool registration patterns

### 10. Testing & Validation

Before deploying to global MCP:

1. **Test locally** with project's .mcp.json
2. **Verify all services connect** (Neo4j, Qdrant, Nomic)
3. **Run validation scripts** (test-sync-manager-integration.py)
4. **Check import paths** are consistent
5. **Ensure Neo4j syntax** is correct (labels required!)
6. **Deploy only after** all tests pass

## Summary

The Model Context Protocol 2025-06-18 is a sophisticated standard for LLM-system integration. Key principles:

- **STDIO transport** with JSON-RPC 2.0 over newline-delimited JSON
- **Proper lifecycle management** with initialize → initialized → operations
- **Service container pattern** for resource management
- **Tool registration** with proper schemas
- **Capability negotiation** during initialization
- **Session isolation** for multi-client support

Most importantly: **Always use existing infrastructure, don't create parallel systems!**

## Next Steps

1. When implementing conversation memory (ADR-0059):
   - Use existing IncrementalIndexer patterns
   - Leverage ServiceContainer for all service access
   - Follow MCP tool registration patterns
   - Test thoroughly before deployment

2. Always verify against official specification:
   - MCP 2025-06-18: https://modelcontextprotocol.io/specification/2025-06-18/
   - Python SDK: https://github.com/modelcontextprotocol/python-sdk

3. Check our ADRs for architecture decisions:
   - ADR-0029: Neo4j logical partitioning
   - ADR-0053: WriteSynchronizationManager
   - ADR-0036: Neo4j primitive properties only

**Confidence: 95%**
Assumptions: MCP specification remains stable, Python SDK patterns are canonical