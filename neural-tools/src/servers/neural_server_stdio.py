#!/usr/bin/env python3
"""
DEPRECATED WRAPPER

This legacy server path is retained for backward compatibility only.
Canonical MCP server is located at: src/mcp/neural_server_stdio.py

This module delegates to the canonical server's run() when executed.
Do not add tools here; extend src/mcp/neural_server_stdio.py instead.
"""

import asyncio
from importlib import import_module


def main():
    srv = import_module('mcp.neural_server_stdio')
    asyncio.run(srv.run())


if __name__ == '__main__':
    main()

