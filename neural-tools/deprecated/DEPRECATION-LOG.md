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
- `neural-mcp-server-enhanced.py` - MCP server with 9 vibe coder tools
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
├── neural-mcp-server-enhanced.py      # 9 MCP tools
├── nomic_embed_server.py              # Embedding API
├── tree_sitter_ast.py                 # Code parser
├── requirements-*.txt                 # Dependencies
├── .neural-tools/                     # Volume data
└── deprecated/                        # Old files
```

This structure provides a clean, maintainable Docker-based system for the L9 Neural Tools.