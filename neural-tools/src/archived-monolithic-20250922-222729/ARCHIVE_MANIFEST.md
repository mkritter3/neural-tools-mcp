# Archived Monolithic MCP Servers - September 22, 2025

## Purpose
This directory contains the old monolithic MCP server implementations that were
replaced by the ADR-0076 modular architecture on September 22, 2025.

## Archived Files
- unified_neural_server_stdio.py.backup-20250922 (1800+ lines monolithic server)
- neural_server_stdio.py.backup-20250922 (older version)
- old-mcp-local-backup/ (previous backup directory)

## Replacement
These monolithic servers have been replaced by:
- neural-tools/src/neural_mcp/server.py (150 lines orchestration)
- neural-tools/src/neural_mcp/tools/ (7 modular tools)
- neural-tools/src/neural_mcp/shared/ (shared utilities)

## Tool Consolidation Achieved
- Original: 24 tools in monolithic architecture
- New: 7 consolidated super-tools in modular architecture
- Reduction: 70% fewer tools with same functionality

## Do Not Use
These files are archived for historical reference only.
Use the new modular architecture instead.

ADR-0076: MCP Tool Standardization and Optimization
