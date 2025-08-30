# Legacy Neural Flow Components

This folder contains deprecated components from the Neural Flow system that have been replaced by the L9 optimized architecture.

## Deprecated Components

### Core System (Replaced by L9 Single Model)
- `ast_analyzer.py` - Multi-language AST analysis (replaced by L9 Hybrid Search)
- `evaluation_harness.py` - Phase 1 evaluation system (replaced by L9 Certification Suite)

### Servers & Infrastructure
- `mcp_neural_server_v2.py` - Experimental v2 server (replaced by main MCP server)

### Testing & Validation
- `test_shadow_indexing.py` - Shadow indexing tests (replaced by L9 certification)
- `test_benchmarks.py` - Legacy benchmark tests (replaced by L9 certification)

## L9 Migration Notes

The L9 Neural Flow architecture represents a complete reimplementation focused on:

1. **Single Model Architecture**: Qodo-Embed-1.5B replaces dual embedding system
2. **Hybrid Search Intelligence**: Combines semantic, keyword, and AST pattern matching
3. **Auto-Safety System**: Zero-config protection with comprehensive rule generation
4. **Container Optimization**: 92% size reduction (23.9GB → 1.85GB)
5. **MCP 2025 Compliance**: Full JSON-RPC 2.0 protocol support

## Preservation Rationale

These files are preserved for:
- Historical reference and debugging
- Migration assistance for custom configurations
- Research and analysis of evolution path
- Rollback capability (if needed)

**Status**: These components are no longer actively maintained but preserved for reference.