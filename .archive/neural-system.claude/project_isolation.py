#!/usr/bin/env python3
"""
Project Isolation Manager
Ensures each project gets its own Qdrant container with data isolation
while sharing embedding model resources
"""

import os
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple
import docker
import yaml

class ProjectIsolation:
    """
    Manages per-project Qdrant containers with shared model resources
    
    Architecture:
    - Each project gets its own Qdrant container (data isolation)
    - Container named: qdrant-{project_name}
    - Ports dynamically allocated based on project hash
    - Shared embedding model server (single instance for all projects)
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.base_port = 6500  # Start from 6500 to avoid conflicts
        self.max_port = 7000   # Maximum port range
        
    def get_project_context(self) -> Dict[str, str]:
        """Detect current project context from working directory"""
        cwd = Path.cwd()
        
        # Try to find project root (has .git, package.json, or .claude)
        project_root = cwd
        for parent in [cwd] + list(cwd.parents):
            if any((parent / marker).exists() for marker in ['.git', 'package.json', '.claude']):
                project_root = parent
                break
        
        project_name = project_root.name
        project_path = str(project_root)
        
        # Generate stable hash for port allocation
        project_hash = hashlib.md5(project_path.encode()).hexdigest()[:8]
        
        return {
            "name": project_name,
            "path": project_path,
            "hash": project_hash,
            "container_name": f"qdrant-{project_name}",
            "network_name": f"neural-network-{project_name}"
        }
    
    def calculate_project_ports(self, project_hash: str) -> Tuple[int, int]:
        """Calculate unique ports for project based on hash"""
        # Convert first 4 chars of hash to number
        hash_num = int(project_hash[:4], 16)
        
        # Map to port range (ensuring even number for REST)
        port_offset = (hash_num % 250) * 2  # 250 * 2 = 500 port range
        
        rest_port = self.base_port + port_offset
        grpc_port = rest_port + 1
        
        # Ensure within range
        if rest_port >= self.max_port:
            rest_port = self.base_port + (port_offset % 100) * 2
            grpc_port = rest_port + 1
            
        return rest_port, grpc_port
    
    def generate_project_compose(self, project: Dict[str, str]) -> str:
        """Generate project-specific docker-compose.yaml"""
        
        rest_port, grpc_port = self.calculate_project_ports(project['hash'])
        
        compose_config = {
            'version': '3.8',
            'services': {
                'qdrant': {
                    'image': 'qdrant/qdrant:v1.10.0',
                    'container_name': project['container_name'],
                    'ports': [
                        f"{rest_port}:6333",  # REST API
                        f"{grpc_port}:6334"   # gRPC
                    ],
                    'volumes': [
                        f"./.docker/qdrant/{project['name']}/storage:/qdrant/storage:z",
                        f"./.docker/qdrant/{project['name']}/snapshots:/qdrant/snapshots:z"
                    ],
                    'environment': [
                        'QDRANT__SERVICE__HTTP_PORT=6333',
                        'QDRANT__SERVICE__GRPC_PORT=6334',
                        'QDRANT__SERVICE__HOST=0.0.0.0',
                        'QDRANT__LOG_LEVEL=INFO',
                        f"QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage",
                        f"QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/snapshots",
                        # Performance optimizations
                        'QDRANT__STORAGE__WAL__WAL_CAPACITY_MB=32',
                        'QDRANT__STORAGE__WAL__WAL_SEGMENTS_AHEAD=2',
                        'QDRANT__STORAGE__PERFORMANCE__INDEXING_THRESHOLD_KB=20000',
                        'QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER=4'
                    ],
                    'labels': {
                        'project': project['name'],
                        'project.path': project['path'],
                        'project.hash': project['hash'],
                        'type': 'qdrant-db'
                    },
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', f"http://localhost:6333/health"],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '40s'
                    },
                    'networks': [project['network_name']],
                    'deploy': {
                        'resources': {
                            'limits': {
                                'cpus': '1',
                                'memory': '2G'
                            },
                            'reservations': {
                                'cpus': '0.5',
                                'memory': '1G'
                            }
                        }
                    }
                }
            },
            'networks': {
                project['network_name']: {
                    'driver': 'bridge',
                    'name': project['network_name']
                }
            }
        }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def ensure_project_container(self, project: Dict[str, str]) -> Dict[str, any]:
        """Ensure project-specific Qdrant container is running"""
        
        container_name = project['container_name']
        rest_port, grpc_port = self.calculate_project_ports(project['hash'])
        
        # Check if container exists
        try:
            container = self.docker_client.containers.get(container_name)
            if container.status == 'running':
                print(f"✅ Container {container_name} already running")
                return {
                    'status': 'existing',
                    'container': container_name,
                    'rest_port': rest_port,
                    'grpc_port': grpc_port
                }
            else:
                print(f"🔄 Starting existing container {container_name}")
                container.start()
                return {
                    'status': 'started',
                    'container': container_name,
                    'rest_port': rest_port,
                    'grpc_port': grpc_port
                }
        except docker.errors.NotFound:
            # Create new container
            print(f"🚀 Creating new container {container_name}")
            
            # Generate docker-compose
            compose_content = self.generate_project_compose(project)
            compose_path = Path(project['path']) / '.claude' / f'docker-compose.{project["name"]}.yaml'
            compose_path.parent.mkdir(parents=True, exist_ok=True)
            compose_path.write_text(compose_content)
            
            # Start with docker-compose
            subprocess.run([
                'docker-compose', '-f', str(compose_path), 'up', '-d'
            ], cwd=project['path'], check=True)
            
            return {
                'status': 'created',
                'container': container_name,
                'rest_port': rest_port,
                'grpc_port': grpc_port,
                'compose_file': str(compose_path)
            }
    
    def get_project_config(self) -> Dict[str, any]:
        """Get complete project configuration including ports"""
        project = self.get_project_context()
        rest_port, grpc_port = self.calculate_project_ports(project['hash'])
        
        return {
            'project_name': project['name'],
            'project_path': project['path'],
            'container_name': project['container_name'],
            'rest_port': rest_port,
            'grpc_port': grpc_port,
            'network_name': project['network_name'],
            'storage_path': f".docker/qdrant/{project['name']}/storage",
            'snapshots_path': f".docker/qdrant/{project['name']}/snapshots"
        }
    
    def get_container_name(self, project_name: Optional[str] = None) -> str:
        """Get container name for current or specified project"""
        if project_name is None:
            project = self.get_project_context()
            project_name = project['name']
        return f"qdrant-{project_name}"
    
    def list_all_project_containers(self):
        """List all project containers"""
        containers = self.docker_client.containers.list(
            all=True,
            filters={'label': 'type=qdrant-db'}
        )
        
        projects = []
        for container in containers:
            labels = container.labels
            projects.append({
                'name': labels.get('project', 'unknown'),
                'path': labels.get('project.path', 'unknown'),
                'container': container.name,
                'status': container.status,
                'ports': container.ports
            })
        
        return projects
    
    def cleanup_project_container(self, project_name: str):
        """Stop and remove a project container"""
        container_name = f"qdrant-{project_name}"
        
        try:
            container = self.docker_client.containers.get(container_name)
            container.stop()
            container.remove()
            print(f"🗑️ Removed container {container_name}")
            return True
        except docker.errors.NotFound:
            print(f"⚠️ Container {container_name} not found")
            return False


class SharedModelServer:
    """
    Shared embedding model server for all projects
    Single instance serves all projects to save memory
    """
    
    def __init__(self):
        self.model_port = 8090  # Fixed port for model server (avoiding conflict with neural-v36)
        self.container_name = "shared-model-server"
        self.docker_client = docker.from_env()
    
    def generate_model_server_compose(self) -> str:
        """Generate docker-compose for shared model server"""
        
        compose_config = {
            'version': '3.8',
            'services': {
                'model-server': {
                    'image': 'model-server:latest',
                    'container_name': self.container_name,
                    'ports': [
                        f"{self.model_port}:8090"  # Model serving API
                    ],
                    'environment': [
                        'MODEL_CACHE=/models',
                        'MODELS=Qodo/Qodo-Embed-1-1.5B,Qdrant/bm25',
                        'MAX_BATCH_SIZE=32',
                        'DEVICE=cpu'  # or cuda if available
                    ],
                    'volumes': [
                        'neural_model_cache:/models:z'  # Shared model cache
                    ],
                    'labels': {
                        'type': 'model-server',
                        'shared': 'true'
                    },
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', f"http://localhost:8090/health"],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'networks': ['shared-network'],
                    'deploy': {
                        'resources': {
                            'limits': {
                                'cpus': '2',
                                'memory': '4G'
                            }
                        }
                    }
                }
            },
            'volumes': {
                'neural_model_cache': {
                    'driver': 'local'
                }
            },
            'networks': {
                'shared-network': {
                    'driver': 'bridge',
                    'name': 'neural-shared-network'
                }
            }
        }
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    def ensure_model_server_running(self):
        """Ensure shared model server is running"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            if container.status == 'running':
                print(f"✅ Shared model server already running")
                return True
            else:
                container.start()
                print(f"🔄 Started shared model server")
                return True
        except docker.errors.NotFound:
            print(f"⚠️ Shared model server not found - needs setup")
            return False


def main():
    """Test project isolation manager"""
    
    print("🔍 PROJECT ISOLATION MANAGER")
    print("=" * 60)
    
    # Initialize manager
    manager = ProjectIsolation()
    
    # Get project context
    project = manager.get_project_context()
    print(f"\n📁 Current Project:")
    print(f"  Name: {project['name']}")
    print(f"  Path: {project['path']}")
    print(f"  Container: {project['container_name']}")
    
    # Calculate ports
    rest_port, grpc_port = manager.calculate_project_ports(project['hash'])
    print(f"\n🔌 Assigned Ports:")
    print(f"  REST: {rest_port}")
    print(f"  gRPC: {grpc_port}")
    
    # Get full config
    config = manager.get_project_config()
    print(f"\n📋 Project Configuration:")
    print(json.dumps(config, indent=2))
    
    # List all project containers
    print(f"\n📦 All Project Containers:")
    projects = manager.list_all_project_containers()
    for proj in projects:
        print(f"  • {proj['name']} ({proj['container']}): {proj['status']}")
    
    print("\n💡 Architecture Summary:")
    print("  • Each project gets isolated Qdrant container")
    print("  • Container named: qdrant-{project_name}")
    print("  • Ports derived from project path hash")
    print("  • Data stored in .docker/qdrant/{project}/")
    print("  • Models shared via central server (planned)")
    
    # Save config for project
    config_path = Path(project['path']) / '.claude' / 'project-config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\n💾 Saved config to: {config_path}")

if __name__ == "__main__":
    main()