#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import pytest

SERVER_PATH = "neural-tools/src/mcp/neural_server_stdio.py"


def _jsonl(obj):
    return (json.dumps(obj) + "\n").encode()


@pytest.mark.asyncio
async def test_mcp_stdio_initialize_and_list_tools():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, SERVER_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # initialize
    init = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"}
    }
    proc.stdin.write(_jsonl(init))
    await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    resp = json.loads(line)
    assert resp.get("jsonrpc") == "2.0"
    assert resp.get("id") == 1
    assert "result" in resp

    # tools/list
    tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    proc.stdin.write(_jsonl(tools_req))
    await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    resp = json.loads(line)
    assert resp.get("id") == 2
    tools = resp.get("result", {}).get("tools", []) if isinstance(resp.get("result"), dict) else resp.get("result", [])
    assert isinstance(tools, list) and len(tools) > 0

    proc.terminate()
    await proc.wait()


@pytest.mark.asyncio
async def test_mcp_no_stdout_logging():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
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
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    # stdout must be valid JSON only
    json.loads(line)
    proc.terminate(); await proc.wait()
