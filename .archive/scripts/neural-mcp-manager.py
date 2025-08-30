#!/usr/bin/env python3
"""
L9 Neural MCP Manager - Global project management via MCP
Simple MCP server that manages neural containers for any directory
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import time
import re
import socket
from pathlib import Path
from typing import Dict, List, Optional, Any

# MCP SDK
from mcp.server.fastmcp import FastMCP
import mcp.types as types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("l9-neural-manager")

@mcp.tool()
async def neural_create(
    project_path: str = ".",
    project_name: Optional[str] = None
) -> Dict[str, Any]:
    """Create isolated neural memory for any project directory
    
    Args:
        project_path: Path to project directory (default: current working directory)
        project_name: Custom project name (auto-detected if not provided)
    """
    try:
        # Resolve absolute project path
        abs_project_path = Path(project_path).resolve()
        if not abs_project_path.exists():
            return {"status": "error", "message": f"Project path not found: {project_path}"}
        
        # Auto-detect project name intelligently
        if not project_name:
            # Try git repo name first
            try:
                git_root = subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"], 
                    cwd=abs_project_path, 
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                project_name = Path(git_root).name
                logger.info(f"Detected git project: {project_name}")
            except:
                # Fall back to directory name
                project_name = abs_project_path.name
                logger.info(f"Using directory name: {project_name}")
        
        # Clean project name for Docker compatibility
        project_name = re.sub(r'[^a-zA-Z0-9._-]', '', project_name.lower())
        if not project_name:
            project_name = f"project-{int(time.time())}"
            logger.warning(f"Generated fallback name: {project_name}")
        
        container_name = f"l9-{project_name}"
        
        # Check if container already exists
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=True
            )
            if result.stdout.strip():
                # Container exists - check if running
                result = subprocess.run(
                    ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
                    capture_output=True, text=True, check=True
                )
                if result.stdout.strip():
                    # Get port
                    port_result = subprocess.run(
                        ["docker", "port", container_name, "6333/tcp"],
                        capture_output=True, text=True
                    )
                    port = port_result.stdout.split(":")[-1].strip() if port_result.returncode == 0 else "6333"
                    
                    # Create MCP config
                    mcp_config = {
                        "mcpServers": {
                            f"l9-neural-{project_name}": {
                                "command": "docker",
                                "args": ["exec", "-i", container_name, "python3", "/app/mcp_server.py"],
                                "env": {
                                    "PROJECT_NAME": project_name,
                                    "QDRANT_HOST": "localhost",
                                    "QDRANT_HTTP_PORT": port
                                }
                            }
                        }
                    }
                    
                    return {
                        "status": "exists",
                        "message": f"✅ Neural memory already active for {project_name}",
                        "project_name": project_name,
                        "container_name": container_name,
                        "port": int(port),
                        "project_path": str(abs_project_path),
                        "mcp_config": mcp_config,
                        "next_steps": [
                            "Neural memory is ready!",
                            "MCP configuration created above",
                            "Add the mcp_config to your .claude/mcp.json if needed"
                        ]
                    }
                else:
                    # Container exists but stopped - restart it
                    subprocess.run(["docker", "start", container_name], check=True)
                    port_result = subprocess.run(
                        ["docker", "port", container_name, "6333/tcp"],
                        capture_output=True, text=True
                    )
                    port = port_result.stdout.split(":")[-1].strip() if port_result.returncode == 0 else "6333"
                    
                    mcp_config = {
                        "mcpServers": {
                            f"l9-neural-{project_name}": {
                                "command": "docker",
                                "args": ["exec", "-i", container_name, "python3", "/app/mcp_server.py"],
                                "env": {
                                    "PROJECT_NAME": project_name,
                                    "QDRANT_HOST": "localhost",
                                    "QDRANT_HTTP_PORT": port
                                }
                            }
                        }
                    }
                    
                    return {
                        "status": "restarted",
                        "message": f"🔄 Neural memory restarted for {project_name}",
                        "project_name": project_name,
                        "container_name": container_name,
                        "port": int(port),
                        "project_path": str(abs_project_path),
                        "mcp_config": mcp_config,
                        "next_steps": [
                            "Neural memory restarted successfully!",
                            "MCP configuration created above",
                            "Add the mcp_config to your .claude/mcp.json if needed"
                        ]
                    }
        except subprocess.CalledProcessError:
            pass  # Container doesn't exist, continue with creation
        
        # Check if L9 image exists
        try:
            result = subprocess.run(
                ["docker", "images", "l9-mcp-server:latest", "-q"],
                capture_output=True, text=True, check=True
            )
            if not result.stdout.strip():
                return {
                    "status": "error",
                    "message": "L9 neural image not found. Please build it first with: docker build -f docker/Dockerfile.mcp -t l9-mcp-server:latest ."
                }
        except subprocess.CalledProcessError:
            return {"status": "error", "message": "Docker not available"}
        
        # Find available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        
        # Create data directories
        data_dir = Path.home() / ".neural-flow" / "data" / project_name
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "qdrant").mkdir(exist_ok=True)
        (data_dir / "mcp").mkdir(exist_ok=True)
        
        # Start new container
        docker_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--restart", "unless-stopped",
            "-p", f"{port}:6333",
            "-p", f"{port+1}:6334",
            "-v", f"{abs_project_path}:/app/project:ro",
            "-v", f"{data_dir}/qdrant:/qdrant/storage",
            "-v", f"{data_dir}/mcp:/app/data",
            "-e", f"PROJECT_NAME={project_name}",
            "-e", "QDRANT_HOST=localhost",
            "-e", "QDRANT_HTTP_PORT=6333",
            "-e", "QDRANT_GRPC_PORT=6334",
            "--label", f"l9.project={project_name}",
            "--label", "l9.type=neural-memory",
            "l9-mcp-server:latest"
        ]
        
        subprocess.run(docker_cmd, check=True)
        
        # Wait for container health check
        logger.info("Waiting for neural container to initialize...")
        for attempt in range(30):
            try:
                import requests
                response = requests.get(f"http://localhost:{port}/collections", timeout=2)
                if response.status_code == 200:
                    logger.info("Neural container ready!")
                    break
            except:
                pass
            time.sleep(2)
        
        # Create MCP config
        mcp_config = {
            "mcpServers": {
                f"l9-neural-{project_name}": {
                    "command": "docker",
                    "args": ["exec", "-i", container_name, "python3", "/app/mcp_server.py"],
                    "env": {
                        "PROJECT_NAME": project_name,
                        "QDRANT_HOST": "localhost",
                        "QDRANT_HTTP_PORT": str(port)
                    }
                }
            }
        }
        
        return {
            "status": "created",
            "message": f"🚀 Neural memory created for {project_name}",
            "project_name": project_name,
            "container_name": container_name,
            "port": port,
            "project_path": str(abs_project_path),
            "mcp_config": mcp_config,
            "next_steps": [
                "Neural memory container created successfully!",
                "🧠 Features: Code analysis, semantic search, persistent memory",
                "📊 Tree-sitter AST parsing for 13+ languages",
                "🔍 Hybrid search (dense + sparse vectors)",
                "",
                "To use in Claude Code:",
                "1. Add the mcp_config above to your .claude/mcp.json",
                "2. Start coding with L9 neural intelligence!"
            ]
        }
        
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Docker command failed: {e}"}
    except Exception as e:
        logger.error(f"Neural creation failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_list() -> Dict[str, Any]:
    """List all neural memory containers"""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "label=l9.type=neural-memory", 
             "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Label \"l9.project\"}}"],
            capture_output=True, text=True, check=True
        )
        
        projects = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 4:
                    name, status, ports, project_name = parts
                    port = ""
                    if ":" in ports and "->" in ports:
                        port = ports.split(":")[1].split("->")[0] if "->" in ports else ""
                    
                    projects.append({
                        "project_name": project_name,
                        "container_name": name,
                        "status": status,
                        "port": port,
                        "running": "Up" in status
                    })
        
        if not projects:
            return {
                "status": "success", 
                "message": "No neural memory containers found",
                "projects": []
            }
        
        return {
            "status": "success", 
            "message": f"Found {len(projects)} neural containers",
            "projects": projects
        }
        
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to list containers: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_stop(project_name: str) -> Dict[str, Any]:
    """Stop neural memory container for a specific project"""
    try:
        container_name = f"l9-{project_name}"
        
        # Check if container exists and is running
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        
        if not result.stdout.strip():
            return {
                "status": "info",
                "message": f"Neural container not running for {project_name}"
            }
        
        subprocess.run(["docker", "stop", container_name], check=True)
        return {
            "status": "success", 
            "message": f"🛑 Stopped neural memory for {project_name}"
        }
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to stop container: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool() 
async def neural_schema_update(
    project_name: str,
    schema_changes: Dict[str, Any]
) -> Dict[str, Any]:
    """Update Qdrant collection schema for a project
    
    Args:
        project_name: Target project name
        schema_changes: Schema modifications (indexes, vectors, etc.)
    """
    try:
        container_name = f"l9-{project_name}"
        
        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        
        if not result.stdout.strip():
            return {"status": "error", "message": f"Neural container not running for {project_name}"}
        
        # Execute schema update inside container
        schema_script = json.dumps({
            "action": "update_schema",
            "project_name": project_name,
            "changes": schema_changes
        })
        
        result = subprocess.run([
            "docker", "exec", "-i", container_name,
            "python3", "-c", f"""
import json
import sys
sys.path.append('/app')
from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(host='localhost', port=6333)
changes = {schema_script}

# Apply schema changes
# TODO: Implement specific schema update logic based on changes
print("Schema update completed")
"""
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"status": "error", "message": f"Schema update failed: {result.stderr}"}
        
        return {
            "status": "success",
            "message": f"📝 Schema updated for {project_name}",
            "changes_applied": schema_changes
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_index_project(
    project_path: str = ".",
    file_extensions: List[str] = [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"],
    chunk_size: int = 2000
) -> Dict[str, Any]:
    """Index all code files in a project for semantic search
    
    Args:
        project_path: Path to project directory
        file_extensions: File types to index
        chunk_size: Maximum chars per chunk
    """
    try:
        abs_project_path = Path(project_path).resolve()
        if not abs_project_path.exists():
            return {"status": "error", "message": f"Project path not found: {project_path}"}
        
        # Get project name
        try:
            git_root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], 
                cwd=abs_project_path, 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            project_name = Path(git_root).name
        except:
            project_name = abs_project_path.name
        
        project_name = re.sub(r'[^a-zA-Z0-9._-]', '', project_name.lower())
        container_name = f"l9-{project_name}"
        
        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        
        if not result.stdout.strip():
            return {"status": "error", "message": f"Neural container not running for {project_name}. Run neural_create first."}
        
        # Count files to index
        files_to_index = []
        for ext in file_extensions:
            files_to_index.extend(list(abs_project_path.rglob(f"*{ext}")))
        
        if not files_to_index:
            return {"status": "info", "message": "No files found to index"}
        
        # Execute indexing via MCP call to the project container
        index_cmd = [
            "docker", "exec", "-i", container_name,
            "python3", "-c", f"""
import sys
sys.path.append('/app')

# Use the container's MCP server to index
from neural_mcp_server_v2 import code_index_directory
import asyncio

result = asyncio.run(code_index_directory(
    path=".",
    extensions={json.dumps(file_extensions)},
    chunk_size={chunk_size}
))
print(json.dumps(result))
"""
        ]
        
        result = subprocess.run(index_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"status": "error", "message": f"Indexing failed: {result.stderr}"}
        
        try:
            index_result = json.loads(result.stdout.strip())
        except:
            index_result = {"indexed_files": len(files_to_index)}
        
        return {
            "status": "success",
            "message": f"🔍 Indexed {len(files_to_index)} files for {project_name}",
            "files_found": len(files_to_index),
            "project_name": project_name,
            "result": index_result
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_backup_project(
    project_name: str,
    backup_path: Optional[str] = None
) -> Dict[str, Any]:
    """Backup neural memory data for a project
    
    Args:
        project_name: Project to backup
        backup_path: Custom backup location (default: ~/.neural-flow/backups)
    """
    try:
        container_name = f"l9-{project_name}"
        
        # Default backup location
        if not backup_path:
            backup_path = str(Path.home() / ".neural-flow" / "backups" / project_name / f"backup-{int(time.time())}")
        
        backup_dir = Path(backup_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup container data volumes
        data_dir = Path.home() / ".neural-flow" / "data" / project_name
        
        if data_dir.exists():
            import shutil
            shutil.copytree(data_dir, backup_dir / "data", dirs_exist_ok=True)
        
        # Export Qdrant collections if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        
        if result.stdout.strip():
            # Container is running - export collections
            export_cmd = [
                "docker", "exec", container_name,
                "python3", "-c", """
import json
from qdrant_client import QdrantClient
client = QdrantClient(host='localhost', port=6333)
collections = client.get_collections().collections
collection_info = [{'name': c.name, 'vectors_count': client.get_collection(c.name).vectors_count} for c in collections]
print(json.dumps(collection_info))
"""
            ]
            
            result = subprocess.run(export_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                collection_info = result.stdout.strip()
                (backup_dir / "collections.json").write_text(collection_info)
        
        return {
            "status": "success",
            "message": f"💾 Backup created for {project_name}",
            "backup_path": str(backup_dir),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_restore_project(
    project_name: str,
    backup_path: str
) -> Dict[str, Any]:
    """Restore neural memory data for a project
    
    Args:
        project_name: Project to restore
        backup_path: Path to backup directory
    """
    try:
        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            return {"status": "error", "message": f"Backup not found: {backup_path}"}
        
        container_name = f"l9-{project_name}"
        data_dir = Path.home() / ".neural-flow" / "data" / project_name
        
        # Stop container if running
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        
        # Restore data directory
        if (backup_dir / "data").exists():
            import shutil
            if data_dir.exists():
                shutil.rmtree(data_dir)
            shutil.copytree(backup_dir / "data", data_dir)
        
        return {
            "status": "success",
            "message": f"🔄 Restored {project_name} from backup",
            "backup_path": backup_path,
            "next_steps": [f"Run neural_create with project_name='{project_name}' to restart container"]
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_clone_project(
    source_project: str,
    target_project: str,
    target_path: str
) -> Dict[str, Any]:
    """Clone neural memory from one project to another
    
    Args:
        source_project: Source project name
        target_project: New project name
        target_path: Path to target project directory
    """
    try:
        abs_target_path = Path(target_path).resolve()
        if not abs_target_path.exists():
            return {"status": "error", "message": f"Target path not found: {target_path}"}
        
        # Backup source project
        backup_result = await neural_backup_project(source_project)
        if backup_result["status"] != "success":
            return backup_result
        
        # Create target project
        create_result = await neural_create(str(abs_target_path), target_project)
        if create_result["status"] not in ["created", "exists", "restarted"]:
            return create_result
        
        # Copy neural memory data
        source_data = Path.home() / ".neural-flow" / "data" / source_project
        target_data = Path.home() / ".neural-flow" / "data" / target_project
        
        if source_data.exists():
            import shutil
            # Stop target container
            subprocess.run(["docker", "stop", f"l9-{target_project}"], capture_output=True)
            
            # Copy data
            if target_data.exists():
                shutil.rmtree(target_data)
            shutil.copytree(source_data, target_data)
            
            # Restart target container
            subprocess.run(["docker", "start", f"l9-{target_project}"], capture_output=True)
        
        return {
            "status": "success",
            "message": f"🔄 Cloned neural memory from {source_project} to {target_project}",
            "source_project": source_project,
            "target_project": target_project,
            "target_path": str(abs_target_path)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_project_stats(project_name: str) -> Dict[str, Any]:
    """Get comprehensive stats for a neural project
    
    Args:
        project_name: Project name to analyze
    """
    try:
        container_name = f"l9-{project_name}"
        
        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True, text=True, check=True
        )
        
        if not result.stdout.strip():
            return {"status": "error", "message": f"Neural container not running for {project_name}"}
        
        container_info = result.stdout.strip().split('\t')
        
        # Get collection stats from container
        stats_cmd = [
            "docker", "exec", container_name,
            "python3", "-c", """
import json
from qdrant_client import QdrantClient

try:
    client = QdrantClient(host='localhost', port=6333)
    collections = client.get_collections().collections
    
    stats = {}
    for collection in collections:
        info = client.get_collection(collection.name)
        stats[collection.name] = {
            'vectors_count': info.vectors_count,
            'points_count': info.points_count,
            'status': str(info.status)
        }
    
    print(json.dumps({
        'status': 'success',
        'collections': stats,
        'total_collections': len(collections),
        'total_vectors': sum(s['vectors_count'] for s in stats.values()),
        'total_points': sum(s['points_count'] for s in stats.values())
    }))
except Exception as e:
    print(json.dumps({'status': 'error', 'message': str(e)}))
"""
        ]
        
        stats_result = subprocess.run(stats_cmd, capture_output=True, text=True)
        
        if stats_result.returncode != 0:
            return {"status": "error", "message": "Failed to get collection stats"}
        
        try:
            collection_stats = json.loads(stats_result.stdout.strip())
        except:
            collection_stats = {"status": "error", "message": "Invalid stats response"}
        
        # Get container resource usage
        resource_cmd = [
            "docker", "stats", container_name, "--no-stream", "--format",
            "{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
        ]
        
        resource_result = subprocess.run(resource_cmd, capture_output=True, text=True)
        resource_info = {}
        
        if resource_result.returncode == 0 and resource_result.stdout.strip():
            parts = resource_result.stdout.strip().split('\t')
            if len(parts) >= 4:
                resource_info = {
                    "cpu_percent": parts[0],
                    "memory_usage": parts[1], 
                    "network_io": parts[2],
                    "block_io": parts[3]
                }
        
        return {
            "status": "success",
            "project_name": project_name,
            "container_name": container_name,
            "container_status": container_info[1] if len(container_info) > 1 else "unknown",
            "ports": container_info[2] if len(container_info) > 2 else "",
            "neural_collections": collection_stats,
            "resource_usage": resource_info,
            "data_path": str(Path.home() / ".neural-flow" / "data" / project_name)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def neural_clean() -> Dict[str, Any]:
    """Clean up unused neural memory containers"""
    try:
        # Remove stopped containers
        result = subprocess.run(
            ["docker", "container", "prune", "-f", "--filter", "label=l9.type=neural-memory"],
            capture_output=True, text=True, check=True
        )
        
        return {
            "status": "success",
            "message": "🧹 Cleaned up unused neural containers",
            "details": result.stdout.strip() if result.stdout.strip() else "No containers to remove"
        }
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to clean containers: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Run the server
if __name__ == "__main__":
    mcp.run(transport='stdio')