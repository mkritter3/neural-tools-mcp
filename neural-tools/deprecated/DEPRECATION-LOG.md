# Neural Tools Deprecation Log

## Date: 2025-08-29

### Files Deprecated
The following files were moved to `deprecated/` as part of streamlining the Neural Tools Docker architecture:

#### Docker Compose Files (replaced by `docker-compose.neural-tools.yml`)
- `docker-compose.l9-enhanced.yml` - Old compose configuration
- `docker-compose.l9-enhanced.yml.bak` - Backup of old compose

#### Unused Directories
- `embedding_server.py/` - Empty directory (using `nomic_embed_server.py`)
- `.docker/` - Old Docker volumes (using `.neural-tools/` now)
- `deprecated-dockerfiles/` - Previously deprecated Dockerfiles

#### Documentation
- `README.md` - Old neural-tools specific README (content moved to main README)

### Active Files Retained
The following files are **actively used** and must NOT be deleted:

#### Docker Build Files
- `docker-compose.neural-tools.yml` - Active compose configuration
- `Dockerfile.l9-minimal` - Builds MCP server image
- `Dockerfile.neural-embeddings` - Builds embedding service image

#### Python Services
- `neural-mcp-server-enhanced.py` - **DEPRECATED** - Old MCP server (moved to legacy-mcp-servers/)
- `nomic_embed_server.py` - Nomic v2-MoE embedding service
- `tree_sitter_ast.py` - AST parser for code analysis

#### Dependencies
- `requirements-l9-minimal.txt` - MCP server dependencies
- `requirements-l9-enhanced.txt` - Embedding service dependencies

#### Data Volumes
- `.neural-tools/` - Active volume mounts for containers

### Current Architecture
```
neural-tools/
├── docker-compose.neural-tools.yml    # Main orchestration
├── Dockerfile.l9-minimal               # MCP server build
├── Dockerfile.neural-embeddings        # Embedding service build
├── neural-mcp-server-enhanced.py      # **DEPRECATED** - moved to legacy-mcp-servers/
├── nomic_embed_server.py              # Embedding API
├── tree_sitter_ast.py                 # Code parser
├── requirements-*.txt                 # Dependencies
├── .neural-tools/                     # Volume data
└── deprecated/                        # Old files
```

This structure provides a clean, maintainable Docker-based system for the L9 Neural Tools.

## Date: 2025-09-02

### MCP Server Implementations Deprecated

#### Files Moved to `deprecated/legacy-mcp-servers/`
- `neural_server_2025.py` - Initial attempt with incorrect session state handling
- `neural_server_2025_fixed.py` - Attempted fix using lifespan context (wrong approach for STDIO)
- `neural_server_refactored.py` - Refactored version still with session issues
- `neural-mcp-server-enhanced.py` - Original monolithic implementation (from main directory)

#### Reason for Deprecation
These implementations incorrectly handled MCP session state by:
- Creating new server instances for each request
- Not maintaining persistent state across JSON-RPC messages
- Misunderstanding the STDIO transport model (treating it like HTTP)

#### Current Working Implementation
**File**: `src/servers/neural_server_stdio.py`

**Key Improvements**:
- Proper STDIO transport implementation (long-lived subprocess)
- Persistent `ServiceState` object throughout server lifetime
- Services initialized once at startup, not per request
- Correct newline-delimited JSON-RPC message handling
- Logging to stderr only (stdout reserved for JSON-RPC)

**Architecture**:
```
Claude Code <--[STDIO]--> MCP Server (neural_server_stdio.py)
                              |
                              ├── Neo4j GraphRAG (persistent connection)
                              ├── Qdrant Vector DB (persistent connection)
                              └── Nomic Embeddings (persistent connection)
```

**Configuration**: Updated in `.mcp.json` to use `/app/neural-tools-src/servers/neural_server_stdio.py`