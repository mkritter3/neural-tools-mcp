#!/usr/bin/env python3
"""
Configuration Manager
Automatically detects project context and configures connections
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Central configuration manager for Neural System
    Handles project detection and port allocation
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = self._detect_and_configure()
    
    def _detect_project_root(self) -> Path:
        """Find project root directory"""
        cwd = Path.cwd()
        
        # Look for project markers
        markers = ['.git', 'package.json', '.claude', 'pyproject.toml']
        
        for parent in [cwd] + list(cwd.parents):
            if any((parent / marker).exists() for marker in markers):
                return parent
        
        # Default to current directory
        return cwd
    
    def _calculate_project_ports(self, project_path: str) -> Tuple[int, int]:
        """Calculate unique ports based on project path"""
        # Create stable hash from project path
        path_hash = hashlib.md5(project_path.encode()).hexdigest()[:8]
        hash_num = int(path_hash[:4], 16)
        
        # Base port ranges (avoiding common ports)
        base_port = 6500  # Start from 6500
        
        # Calculate offset (max 250 projects * 2 ports each = 500 port range)
        port_offset = (hash_num % 250) * 2
        
        rest_port = base_port + port_offset
        grpc_port = rest_port + 1
        
        # Ensure within safe range (6500-7000)
        if rest_port >= 7000:
            rest_port = 6500 + (port_offset % 100) * 2
            grpc_port = rest_port + 1
        
        return rest_port, grpc_port
    
    def _detect_and_configure(self) -> Dict:
        """Detect project and configure connections"""
        
        # Find project root
        project_root = self._detect_project_root()
        project_name = project_root.name
        
        # Check for existing config
        config_path = project_root / '.claude' / 'project-config.json'
        if config_path.exists():
            logger.info(f"📋 Loading existing config from {config_path}")
            with open(config_path) as f:
                return json.load(f)
        
        # Generate new config
        project_path = str(project_root)
        rest_port, grpc_port = self._calculate_project_ports(project_path)
        
        config = {
            'project_name': project_name,
            'project_path': project_path,
            'container_name': f'qdrant-{project_name}',
            'rest_port': rest_port,
            'grpc_port': grpc_port,
            'network_name': f'neural-network-{project_name}',
            'storage_path': f'.docker/qdrant/{project_name}/storage',
            'snapshots_path': f'.docker/qdrant/{project_name}/snapshots',
            'model_server': {
                'host': 'localhost',
                'port': 8090,
                'shared': True
            }
        }
        
        # Save config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Created new config for project: {project_name}")
        logger.info(f"  REST Port: {rest_port}, gRPC Port: {grpc_port}")
        
        return config
    
    def get_qdrant_config(self, prefer_grpc: bool = True) -> Dict:
        """Get Qdrant connection configuration"""
        if prefer_grpc:
            return {
                'host': 'localhost',
                'port': self._config['grpc_port'],
                'prefer_grpc': True
            }
        else:
            return {
                'host': 'localhost',
                'port': self._config['rest_port'],
                'prefer_grpc': False
            }
    
    def get_project_name(self) -> str:
        """Get current project name"""
        return self._config['project_name']
    
    def get_container_name(self) -> str:
        """Get project-specific container name"""
        return self._config['container_name']
    
    def get_collection_name(self, base_name: str) -> str:
        """Get project-prefixed collection name"""
        project_name = self._config['project_name']
        return f"{project_name}_{base_name}"
    
    def get_model_server_config(self) -> Dict:
        """Get model server configuration"""
        return self._config.get('model_server', {
            'host': 'localhost',
            'port': 8080,
            'shared': True
        })
    
    def summary(self) -> str:
        """Get configuration summary"""
        return f"""
Configuration Summary
========================
Project: {self._config['project_name']}
Container: {self._config['container_name']}
REST Port: {self._config['rest_port']}
gRPC Port: {self._config['grpc_port']}
Storage: {self._config['storage_path']}
"""


# Singleton instance
def get_config() -> ConfigManager:
    """Get the singleton config manager instance"""
    return ConfigManager()


if __name__ == "__main__":
    # Test configuration detection
    config = get_config()
    print(config.summary())
    
    print("\n🔌 Qdrant Config (gRPC):")
    print(json.dumps(config.get_qdrant_config(prefer_grpc=True), indent=2))
    
    print("\n📦 Collection Names:")
    print(f"  Memory: {config.get_collection_name('memory')}")
    print(f"  Code: {config.get_collection_name('code_search')}")
    print(f"  Neural: {config.get_collection_name('neural_flow')}")