# MCP Server Guide (Canonical STDIO)

This guide documents the single, canonical MCP server and how to run, extend, and troubleshoot it.

## Canonical Server

- Entrypoint: `neural-tools/run_mcp_server.py`
- Server module: `neural-tools/src/mcp/neural_server_stdio.py` (canonical)
- Transport: STDIO (JSON-RPC over stdin/stdout)

## Run Locally

```bash
python neural-tools/run_mcp_server.py
```

## Run in Docker

- Supervisor launches: `python3 -u /app/src/mcp/neural_server_stdio.py`
- Logs: stderr only for logging; stdout is reserved for JSON-RPC

## Extend: Add a New Tool

1. Edit `src/mcp/neural_server_stdio.py` and add a new tool in the `@server.list_tools()` registry and in `@server.call_tool()` dispatch.
2. Enforce `additionalProperties: false` in input schemas and validate required fields.
3. Return `mcp.types.TextContent` containing a compact JSON string.

## Best Practices

- Always log to stderr; never print to stdout.
- Keep payloads compact; trim long content and avoid huge blobs.
- Use strict parameter validation and return structured errors.
- Apply latency budgets to downstream calls; surface degraded mode clearly.

## Troubleshooting

- “No output” from server: verify stdout isn’t polluted by logs.
- Protocol errors: check `initialize` is sent first; validate method names & payload shapes.
- Timeouts: increase client read timeout; ensure newline-terminated JSON per message.

## Tests

See `neural-tools/tests/mcp/` for stdio protocol, loop safety, and degradation tests.
