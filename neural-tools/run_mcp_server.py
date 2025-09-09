#!/usr/bin/env python3
"""
Unified MCP entrypoint (STDIO transport)

Launches the canonical MCP stdio server implemented in:
  neural-tools/src/neural_mcp/neural_server_stdio.py

Bootstrap behavior:
- Ensures `neural-tools/src` is on PYTHONPATH
- If `mcp` SDK is not installed in the current interpreter, attempts to
  bootstrap a local virtualenv under `<repo>/.neural/.venv` and install
  production requirements, then re-executes itself using that venv.
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_src_to_path():
    # Ensure neural-tools/src is on sys.path for canonical server import
    neural_tools_dir = Path(__file__).parent
    sys.path.insert(0, str(neural_tools_dir / "src"))


def _have_mcp() -> bool:
    try:
        import mcp  # type: ignore
        return True
    except Exception:
        return False


def _ensure_venv_and_deps() -> str:
    """Create a local venv and install prod requirements if missing.

    Returns the path to the venv's python executable.
    """
    repo = _repo_root()
    venv_dir = repo / ".neural" / ".venv"
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    python_bin = sys.executable

    # Create venv if missing
    if not (venv_dir / "bin" / "python").exists() and not (venv_dir / "Scripts" / "python.exe").exists():
        subprocess.check_call([python_bin, "-m", "venv", str(venv_dir)])

    venv_python = str((venv_dir / "bin" / "python").resolve()) if (venv_dir / "bin" / "python").exists() else str((venv_dir / "Scripts" / "python.exe").resolve())

    # Install requirements into the venv
    req = repo / "requirements" / "prod.txt"
    if req.exists():
        subprocess.check_call([venv_python, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])  # ensure tooling
        subprocess.check_call([venv_python, "-m", "pip", "install", "-r", str(req)])

    return venv_python


def _maybe_bootstrap_and_exec():
    # If mcp SDK is unavailable, attempt to bootstrap a local venv and re-exec
    if _have_mcp():
        return
    try:
        venv_python = _ensure_venv_and_deps()
        # Re-exec this script under the venv python
        os.execv(venv_python, [venv_python, __file__])
    except Exception as e:
        # Last resort: log to stderr and exit non-zero so caller sees failure
        sys.stderr.write(f"[neural-tools] Failed to bootstrap MCP environment: {e}\n")
        sys.exit(1)


def main():
    _maybe_bootstrap_and_exec()
    _add_src_to_path()
    # Prefer canonical stdio server under src/neural_mcp
    from neural_mcp.neural_server_stdio import run
    asyncio.run(run())


if __name__ == "__main__":
    main()
