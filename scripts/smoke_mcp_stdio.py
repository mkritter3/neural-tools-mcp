#!/usr/bin/env python3
"""
MCP STDIO smoke test:
- Starts the MCP server (neural_server_stdio.py) as a subprocess
- Sends initialize and list_tools
- Verifies expected tools are present

This test is designed to run without external services (Neo4j/Qdrant). The server
should degrade gracefully; we only require that it starts and lists tools.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    server_path = repo_root / "neural-tools" / "src" / "neural_mcp" / "neural_server_stdio.py"

    env = os.environ.copy()
    # Ensure Python can import our packages
    env["PYTHONPATH"] = f"{repo_root / 'neural-tools' / 'src'}:{env.get('PYTHONPATH','')}"
    # Minimize external deps; allow graceful degradation
    env.setdefault("PROJECT_NAME", "smoke")
    env.setdefault("NEO4J_URI", "bolt://localhost:7687")
    env.setdefault("QDRANT_HOST", "localhost")
    env.setdefault("QDRANT_PORT", "6333")
    env.setdefault("EMBED_DIM", "768")
    env.setdefault("PYTHONUNBUFFERED", "1")

    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "SmokeMCP", "version": "0.1.0"},
        },
    }

    list_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

    proc = subprocess.Popen(
        [sys.executable, "-u", str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(repo_root),
        text=True,
        bufsize=1,
    )

    try:
        # Send initialize
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()
        init_line = proc.stdout.readline().strip()
        init_resp = json.loads(init_line)
        assert "result" in init_resp or "error" not in init_resp, f"Invalid init response: {init_resp}"

        # Send list_tools
        proc.stdin.write(json.dumps(list_request) + "\n")
        proc.stdin.flush()
        list_line = proc.stdout.readline().strip()
        list_resp = json.loads(list_line)
        assert "result" in list_resp, f"Invalid list_tools response: {list_resp}"

        tools = list_resp["result"].get("tools", [])
        names = {t.get("name") for t in tools}
        expected = {"semantic_code_search", "graphrag_hybrid_search", "neural_system_status"}
        missing = expected - names
        assert not missing, f"Missing expected tools: {missing}; got {names}"

        print("OK: MCP STDIO smoke passed")
        return 0
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())

