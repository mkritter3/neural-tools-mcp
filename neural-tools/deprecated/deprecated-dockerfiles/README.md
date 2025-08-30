# Deprecated Dockerfiles

**Deprecated Date**: 2025-08-29  
**Reason**: Moved to focused, single-responsibility containers following L9 principles

## What was deprecated:
- `Dockerfile.l9-enhanced` - Multi-stage build (MCP server + embedding service)
- `Dockerfile.l9-enhanced-cached` - Multi-stage build with Docker cache optimization
- `model-server/` - Legacy model server implementation

## Current Active Dockerfiles:
- `../Dockerfile.l9-minimal` - Builds `l9-mcp-enhanced:minimal-fixed` (MCP server only)
- `../Dockerfile.neural-embeddings` - Builds `neural-flow:nomic-v2-production` (embedding service only)

## Migration Notes:
The functionality from these deprecated multi-stage builds has been split into:
1. **MCP Server**: Minimal build using external embedding service
2. **Embedding Service**: Focused Nomic Embed v2-MoE container

This follows L9 single responsibility principle and reduces maintenance overhead.