# Deprecations

We have consolidated to a single MCP server implementation.

## Canonical

- Entrypoint: `neural-tools/run_mcp_server.py`
- Server: `src/mcp/neural_server_stdio.py`

## Deprecated

- `neural-tools/src/servers/neural_server_stdio.py` (kept as a wrapper; do not extend)
- `neural-tools/src/servers/mcp_proxy/*` (HTTP proxy variant)
- `neural-tools/src/servers/mcp_stdio_wrapper.py` (legacy stdio wrapper)

These remain in the repo to avoid breaking existing references, but all docs, scripts, and configs now point to the canonical server. New work must target `src/mcp/neural_server_stdio.py`.

