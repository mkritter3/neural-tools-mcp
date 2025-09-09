#!/usr/bin/env python3
"""
Neural Search CLI
Command-line interface for managing the neural search server
"""

import typer
from pathlib import Path
import subprocess
import psutil
import json
import httpx
import time
import sys
from typing import Optional

# Import our config manager
from config_manager import ConfigManager

app = typer.Typer(help="Neural Search - Local semantic code search")
service_app = typer.Typer(help="Service management commands")
projects_app = typer.Typer(help="Project management commands")

app.add_typer(service_app, name="service")
app.add_typer(projects_app, name="projects")

# Global configuration
config_manager = ConfigManager()
PID_FILE = config_manager.config_dir / "server.pid"
LOG_FILE = config_manager.config_dir / "server.log"

def get_server_url() -> str:
    """Get the server URL from configuration"""
    server_config = config_manager.get_server_config()
    host = server_config.get("host", "localhost")
    port = server_config.get("port", 8000)
    return f"http://{host}:{port}"

def is_server_running() -> tuple[bool, Optional[int]]:
    """
    Check if the server is running
    Returns: (is_running, pid)
    """
    if not PID_FILE.exists():
        return False, None
    
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
            
        if psutil.pid_exists(pid):
            # Double-check by trying to connect
            try:
                response = httpx.get(f"{get_server_url()}/health", timeout=1.0)
                if response.status_code == 200:
                    return True, pid
            except:
                pass
        
        # PID file exists but process is dead - clean up
        PID_FILE.unlink()
        return False, None
        
    except (ValueError, FileNotFoundError):
        return False, None

def wait_for_server(timeout: int = 10) -> bool:
    """Wait for server to become ready"""
    for _ in range(timeout):
        try:
            response = httpx.get(f"{get_server_url()}/health", timeout=1.0)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

# Service management commands
@service_app.command("start")
def start_server(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run server on"),
    host: str = typer.Option("localhost", "--host", help="Host to bind to")
):
    """Start the neural search server"""
    running, pid = is_server_running()
    if running:
        typer.echo(f"✅ Server is already running (PID: {pid})")
        return
    
    typer.echo("🚀 Starting neural search server...")
    
    # Update server config
    config_manager.update_server_config(host=host, port=port)
    
    # Find the API directory and main.py
    api_dir = Path(__file__).parent
    main_file = api_dir / "main.py"
    
    if not main_file.exists():
        typer.echo(f"❌ Cannot find main.py at {main_file}", err=True)
        raise typer.Exit(1)
    
    # Start the server process
    try:
        # Create log file
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(LOG_FILE, 'w') as log_file:
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "main:app",
                "--host", host,
                "--port", str(port),
                "--log-level", "info"
            ], 
            cwd=api_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Detach from parent
            )
        
        # Write PID file
        with open(PID_FILE, 'w') as f:
            f.write(str(process.pid))
        
        # Wait for server to be ready
        typer.echo("⏳ Waiting for server to start...")
        if wait_for_server():
            typer.echo(f"✅ Server started successfully on {get_server_url()}")
            typer.echo(f"📋 Process ID: {process.pid}")
            typer.echo(f"📝 Logs: {LOG_FILE}")
        else:
            typer.echo("❌ Server failed to start properly", err=True)
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Failed to start server: {e}", err=True)
        raise typer.Exit(1)

@service_app.command("stop")
def stop_server():
    """Stop the neural search server"""
    running, pid = is_server_running()
    if not running:
        typer.echo("⚠️  Server is not running")
        return
    
    typer.echo(f"🛑 Stopping server (PID: {pid})...")
    
    try:
        process = psutil.Process(pid)
        process.terminate()
        
        # Wait up to 5 seconds for graceful shutdown
        try:
            process.wait(timeout=5)
            typer.echo("✅ Server stopped successfully")
        except psutil.TimeoutExpired:
            typer.echo("⚠️  Server didn't stop gracefully, force killing...")
            process.kill()
            process.wait()
            typer.echo("✅ Server force stopped")
            
        # Clean up PID file
        if PID_FILE.exists():
            PID_FILE.unlink()
            
    except psutil.NoSuchProcess:
        typer.echo("⚠️  Process no longer exists")
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception as e:
        typer.echo(f"❌ Failed to stop server: {e}", err=True)
        raise typer.Exit(1)

@service_app.command("status")
def server_status():
    """Check server status"""
    running, pid = is_server_running()
    
    if running:
        typer.echo(f"✅ Server is running")
        typer.echo(f"🔗 URL: {get_server_url()}")
        typer.echo(f"📋 PID: {pid}")
        
        # Get additional info from server
        try:
            response = httpx.get(f"{get_server_url()}/health", timeout=2.0)
            if response.status_code == 200:
                health = response.json()
                typer.echo(f"🔍 Status: {health.get('status', 'unknown')}")
                
            # Get project count
            projects_response = httpx.get(f"{get_server_url()}/projects", timeout=2.0)
            if projects_response.status_code == 200:
                projects_data = projects_response.json()
                typer.echo(f"📁 Projects: {projects_data.get('total_projects', 0)}")
                
        except Exception as e:
            typer.echo(f"⚠️  Could not get server details: {e}")
    else:
        typer.echo("❌ Server is not running")
        typer.echo("💡 Start with: neural-search service start")

@service_app.command("logs")
def show_logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show")
):
    """Show server logs"""
    if not LOG_FILE.exists():
        typer.echo("❌ No log file found", err=True)
        return
    
    try:
        # Simple tail implementation
        with open(LOG_FILE, 'r') as f:
            log_lines = f.readlines()
            
        # Show last N lines
        for line in log_lines[-lines:]:
            typer.echo(line.rstrip())
            
    except Exception as e:
        typer.echo(f"❌ Failed to read logs: {e}", err=True)

# Project management commands
@projects_app.command("add")
def add_project(
    project_path: str = typer.Argument(..., help="Path to project directory"),
    project_name: Optional[str] = typer.Option(None, "--name", "-n", help="Project name (defaults to directory name)")
):
    """Add a project to neural search"""
    project_path = Path(project_path).expanduser().resolve()
    
    if not project_path.exists():
        typer.echo(f"❌ Project path does not exist: {project_path}", err=True)
        raise typer.Exit(1)
    
    if not project_name:
        project_name = project_path.name
    
    # Check if server is running
    running, _ = is_server_running()
    if not running:
        typer.echo("❌ Server is not running. Start with: neural-search service start", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"📚 Adding project '{project_name}' from {project_path}...")
    
    try:
        response = httpx.post(
            f"{get_server_url()}/index",
            json={
                "project_path": str(project_path),
                "project_name": project_name
            },
            timeout=30.0
        )
        response.raise_for_status()
        
        result = response.json()
        job_id = result["job_id"]
        
        typer.echo(f"✅ Indexing started (Job ID: {job_id})")
        typer.echo("⏳ Waiting for indexing to complete...")
        
        # Poll job status
        for _ in range(30):  # Max 60 seconds
            time.sleep(2)
            status_response = httpx.get(f"{get_server_url()}/status/{job_id}", timeout=10.0)
            if status_response.status_code == 200:
                job_status = status_response.json()
                if job_status["status"] == "completed":
                    typer.echo(f"🎉 Project '{project_name}' indexed successfully!")
                    if job_status.get("details"):
                        typer.echo(f"📊 {job_status['details']}")
                    return
                elif job_status["status"] == "failed":
                    typer.echo(f"❌ Indexing failed: {job_status.get('details', 'Unknown error')}", err=True)
                    raise typer.Exit(1)
                else:
                    typer.echo(f"⏳ Status: {job_status['status']}")
        
        typer.echo("⚠️  Indexing is taking longer than expected. Check status with: neural-search service status")
        
    except httpx.HTTPError as e:
        typer.echo(f"❌ Failed to add project: {e}", err=True)
        raise typer.Exit(1)

@projects_app.command("list")
def list_projects():
    """List all indexed projects"""
    running, _ = is_server_running()
    if not running:
        typer.echo("❌ Server is not running. Start with: neural-search service start", err=True)
        raise typer.Exit(1)
    
    try:
        response = httpx.get(f"{get_server_url()}/projects", timeout=10.0)
        response.raise_for_status()
        
        data = response.json()
        projects = data.get("projects", [])
        
        if not projects:
            typer.echo("📭 No projects found")
            typer.echo("💡 Add a project with: neural-search projects add /path/to/project")
            return
        
        typer.echo(f"📁 Found {len(projects)} projects:")
        for project in projects:
            name = project["project_name"]
            points = project["points_count"]
            status = project["status"]
            typer.echo(f"  • {name}: {points} vectors ({status})")
            
    except httpx.HTTPError as e:
        typer.echo(f"❌ Failed to list projects: {e}", err=True)
        raise typer.Exit(1)

# Search command (top-level)
@app.command("search")
def search_code(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to return"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Search specific project only")
):
    """Search code across indexed projects"""
    running, _ = is_server_running()
    if not running:
        typer.echo("❌ Server is not running. Start with: neural-search service start", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"🔍 Searching for: '{query}'")
    
    try:
        params = {"query": query, "limit": limit}
        if project:
            params["project_names"] = project
            
        response = httpx.get(f"{get_server_url()}/search", params=params, timeout=15.0)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            typer.echo("📭 No results found")
            return
        
        typer.echo(f"🎯 Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            file_path = Path(result["file_path"]).name
            project_name = result["project_name"]
            score = result["score"]
            snippet = result["snippet"][:150] + "..." if len(result["snippet"]) > 150 else result["snippet"]
            
            typer.echo(f"\n{i}. {file_path} (score: {score:.3f}) [{project_name}]")
            typer.echo(f"   {snippet}")
            
    except httpx.HTTPError as e:
        typer.echo(f"❌ Search failed: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()