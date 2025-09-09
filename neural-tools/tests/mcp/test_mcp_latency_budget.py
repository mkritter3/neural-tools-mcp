#!/usr/bin/env python3
import asyncio, json, os, sys, pytest, time

SERVER_PATH = "neural-tools/src/mcp/neural_server_stdio.py"


@pytest.mark.asyncio
async def test_tools_call_roundtrip_under_budget():
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, SERVER_PATH,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    init = {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
    proc.stdin.write((json.dumps(init)+"\n").encode()); await proc.stdin.drain()
    await proc.stdout.readline()
    req = {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"project_understanding","arguments":{}}}
    t0 = time.perf_counter()
    proc.stdin.write((json.dumps(req)+"\n").encode()); await proc.stdin.drain()
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
    elapsed_ms = (time.perf_counter()-t0)*1000
    assert elapsed_ms < 500
    proc.terminate(); await proc.wait()
