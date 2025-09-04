#!/usr/bin/env python3
"""
L9 Simplified Configuration Management
Single source of truth for all service configurations
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class L9Config:
    """Single configuration class for all Neural Tools services"""
    
    # Project
    project_name: str = os.environ.get('PROJECT_NAME', 'claude-l9-template')
    project_dir: str = os.environ.get('PROJECT_DIR', '/app/project')
    
    # Neo4j
    neo4j_host: str = os.environ.get('NEO4J_HOST', 'localhost')
    neo4j_port: int = int(os.environ.get('NEO4J_PORT', '7688'))
    neo4j_username: str = os.environ.get('NEO4J_USERNAME', 'neo4j')
    neo4j_password: str = os.environ.get('NEO4J_PASSWORD', 'neural-l9-2025')
    
    # Qdrant
    qdrant_host: str = os.environ.get('QDRANT_HOST', 'localhost')
    qdrant_http_port: int = int(os.environ.get('QDRANT_HTTP_PORT', '6681'))
    qdrant_grpc_port: int = int(os.environ.get('QDRANT_GRPC_PORT', '6681'))
    
    # Embeddings
    embedding_host: str = os.environ.get('EMBEDDING_SERVICE_HOST', 'localhost')
    embedding_port: int = int(os.environ.get('EMBEDDING_SERVICE_PORT', '8081'))
    
    # Container names (for Docker networking)
    neo4j_container: str = 'default-neo4j-graph'
    qdrant_container: str = 'default-neural-storage'
    embedding_container: str = 'default-neural-embeddings'
    neural_container: str = 'default-neural'
    
    @property
    def neo4j_uri(self) -> str:
        """Get Neo4j connection URI"""
        return f"bolt://{self.neo4j_host}:{self.neo4j_port}"
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant HTTP URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_http_port}"
    
    @property
    def embedding_url(self) -> str:
        """Get embedding service URL"""
        return f"http://{self.embedding_host}:{self.embedding_port}"
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return {
            'project': {
                'name': self.project_name,
                'dir': self.project_dir
            },
            'neo4j': {
                'host': self.neo4j_host,
                'port': self.neo4j_port,
                'uri': self.neo4j_uri,
                'username': self.neo4j_username
            },
            'qdrant': {
                'host': self.qdrant_host,
                'http_port': self.qdrant_http_port,
                'grpc_port': self.qdrant_grpc_port,
                'url': self.qdrant_url
            },
            'embeddings': {
                'host': self.embedding_host,
                'port': self.embedding_port,
                'url': self.embedding_url
            },
            'containers': {
                'neo4j': self.neo4j_container,
                'qdrant': self.qdrant_container,
                'embeddings': self.embedding_container,
                'neural': self.neural_container
            }
        }
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check ports are valid
        if not (1 <= self.neo4j_port <= 65535):
            errors.append(f"Invalid Neo4j port: {self.neo4j_port}")
        if not (1 <= self.qdrant_http_port <= 65535):
            errors.append(f"Invalid Qdrant HTTP port: {self.qdrant_http_port}")
        if not (1 <= self.embedding_port <= 65535):
            errors.append(f"Invalid embedding port: {self.embedding_port}")
        
        # Check required values are set
        if not self.neo4j_password:
            errors.append("Neo4j password not set")
        if not self.project_name:
            errors.append("Project name not set")
            
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def print_config(self):
        """Print current configuration for debugging"""
        print("L9 Neural Tools Configuration")
        print("=" * 40)
        print(f"Project: {self.project_name}")
        print(f"Project Dir: {self.project_dir}")
        print()
        print("Neo4j:")
        print(f"  URI: {self.neo4j_uri}")
        print(f"  Username: {self.neo4j_username}")
        print()
        print("Qdrant:")
        print(f"  URL: {self.qdrant_url}")
        print(f"  GRPC Port: {self.qdrant_grpc_port}")
        print()
        print("Embeddings:")
        print(f"  URL: {self.embedding_url}")
        print()
        print("Container Names:")
        print(f"  Neo4j: {self.neo4j_container}")
        print(f"  Qdrant: {self.qdrant_container}")
        print(f"  Embeddings: {self.embedding_container}")
        print(f"  Neural: {self.neural_container}")


# Global config instance
config = L9Config()

if __name__ == "__main__":
    # Test/debug configuration
    config.print_config()
    print()
    if config.validate():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
        exit(1)