#!/usr/bin/env python3
"""
Clean up stale MCP server processes that aren't connected to active Claude instances.
This helps prevent the "can't connect to MCP" issue for new Claude instances.
"""

import os
import sys
import psutil
import signal
from pathlib import Path
import time

def is_process_alive(pid):
    """Check if a process is alive"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def has_active_stdio(pid):
    """Check if process has active STDIO connections"""
    try:
        p = psutil.Process(pid)
        # Check for open file descriptors
        for conn in p.connections(kind='all'):
            if conn.status == 'ESTABLISHED':
                return True
        # Check if process is reading from stdin
        for f in p.open_files():
            if 'stdin' in f.path.lower():
                return True
        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def cleanup_stale_mcp_processes():
    """Clean up MCP processes that aren't connected to Claude"""
    cleaned = 0
    kept = 0
    
    # Find all run_mcp_server.py processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and 'run_mcp_server.py' in ' '.join(cmdline):
                pid = proc.info['pid']
                create_time = proc.info['create_time']
                age_seconds = time.time() - create_time
                
                # Skip very new processes (< 30 seconds old)
                if age_seconds < 30:
                    print(f"⏳ PID {pid}: Too new ({age_seconds:.0f}s old), skipping")
                    kept += 1
                    continue
                
                # Check if it has active connections
                if has_active_stdio(pid):
                    print(f"✅ PID {pid}: Active connections, keeping")
                    kept += 1
                else:
                    print(f"🧹 PID {pid}: No active connections, cleaning up")
                    try:
                        # Per MCP spec: close stdin first, then SIGTERM if needed
                        proc = psutil.Process(pid)
                        
                        # Try to close stdin if possible (simulate client disconnect)
                        # Since we can't close stdin of another process, go straight to SIGTERM
                        os.kill(pid, signal.SIGTERM)
                        cleaned += 1
                        time.sleep(2)  # Give more time for graceful shutdown
                        
                        if is_process_alive(pid):
                            print(f"   Process didn't exit, sending SIGKILL")
                            os.kill(pid, signal.SIGKILL)
                    except OSError as e:
                        print(f"   Failed to kill {pid}: {e}")
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Clean up PID files
    pid_dir = Path("/tmp/mcp_pids")
    if pid_dir.exists():
        for pid_file in pid_dir.glob("mcp_*.pid"):
            try:
                pid = int(pid_file.read_text().strip())
                if not is_process_alive(pid):
                    pid_file.unlink()
                    print(f"🗑️  Removed stale PID file for {pid}")
            except (ValueError, FileNotFoundError):
                pid_file.unlink()
                print(f"🗑️  Removed invalid PID file {pid_file}")
    
    print(f"\n📊 Summary: Cleaned {cleaned} stale processes, kept {kept} active")
    return cleaned

if __name__ == "__main__":
    print("🔍 Checking for stale MCP processes...")
    cleaned = cleanup_stale_mcp_processes()
    sys.exit(0)