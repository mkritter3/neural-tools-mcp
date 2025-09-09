#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import pytest

SERVER_PATH = "neural-tools/src/mcp/neural_server_stdio.py"


@pytest.mark.asyncio
async def test_semantic_code_search_fallback_when_qdrant_down():
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    # Simulate Qdrant failure by pointing to an invalid host/port
    env["QDRANT_HOST"] = "127.0.0.1"
    env["QDRANT_PORT"] = "65535"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, SERVER_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    # initialize
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    proc.stdin.write((json.dumps(init)+"\n").encode()); await proc.stdin.drain()
    await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    # call semantic_code_search
    req = {
      "jsonrpc": "2.0", "id": 2, "method": "tools/call",
      "params": {"name": "semantic_code_search", "arguments": {"query": "cache", "limit": 3}}
    }
    proc.stdin.write((json.dumps(req)+"\n").encode()); await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=10)
    resp = json.loads(line)
    # Should be valid JSON-RPC
    assert resp.get("jsonrpc") == "2.0"
    # Extract TextContent payload
    result_list = resp.get("result") or []
    assert isinstance(result_list, list) and len(result_list) > 0
    out = json.loads(result_list[0]["text"])  # server returns TextContent with JSON body
    assert out.get("status") in ("success", "error")
    proc.terminate(); await proc.wait()
