#!/usr/bin/env python3
"""
ADR-0037 Configuration Validation Utility
Validates container configuration compliance with environment variable priority standard.

Usage:
    python3 scripts/validate-adr-0037.py [--container CONTAINER_NAME] [--fix]
    
Examples:
    python3 scripts/validate-adr-0037.py --container indexer-claude-l9-template
    python3 scripts/validate-adr-0037.py --all
    python3 scripts/validate-adr-0037.py --validate-code docker/scripts/
"""

import os
import sys
import json
import re
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ADR0037Validator:
    """Validates ADR-0037 configuration priority compliance"""
    
    REQUIRED_ENV_VARS = {
        'PROJECT_NAME': 'Project identifier (e.g., claude-l9-template)',
        'PROJECT_PATH': 'Project root directory (e.g., /workspace)',
    }
    
    # Services that need project context (others are infrastructure)
    PROJECT_AWARE_SERVICES = ['indexer']
    
    CONTAINER_ENV_VARS = {
        'NEO4J_URI': 'bolt://host.docker.internal:47687',
        'NEO4J_PASSWORD': 'graphrag-password',
        'QDRANT_HOST': 'host.docker.internal',
        'QDRANT_PORT': '46333',
        'EMBEDDING_SERVICE_HOST': 'host.docker.internal',
        'EMBEDDING_SERVICE_PORT': '48000',
        'REDIS_CACHE_HOST': 'host.docker.internal',
        'REDIS_CACHE_PORT': '46379',
        'REDIS_QUEUE_HOST': 'host.docker.internal',
        'REDIS_QUEUE_PORT': '46380',
    }
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = []
    
    def validate_container(self, container_name: str) -> bool:
        """Validate a running container's environment variables"""
        logger.info(f"🔍 Validating container: {container_name}")
        
        try:
            # Get container environment variables
            result = subprocess.run([
                'docker', 'inspect', container_name
            ], capture_output=True, text=True, check=True)
            
            container_info = json.loads(result.stdout)[0]
            env_vars = {}
            
            for env_var in container_info['Config']['Env']:
                if '=' in env_var:
                    key, value = env_var.split('=', 1)
                    env_vars[key] = value
            
            # Validate required environment variables
            self._validate_env_vars(container_name, env_vars)
            
            # Check container configuration
            self._validate_container_config(container_name, container_info)
            
            return len(self.errors) == 0
            
        except subprocess.CalledProcessError as e:
            self.errors.append(f"❌ Container '{container_name}' not found or not accessible")
            return False
        except Exception as e:
            self.errors.append(f"❌ Error validating container '{container_name}': {e}")
            return False
    
    def _validate_env_vars(self, container_name: str, env_vars: Dict[str, str]):
        """Validate environment variables against ADR-0037"""
        
        # Check required project variables (only for project-aware services)
        is_project_aware = any(service in container_name.lower() for service in self.PROJECT_AWARE_SERVICES)
        
        if is_project_aware:
            for var_name, description in self.REQUIRED_ENV_VARS.items():
                if var_name not in env_vars:
                    self.errors.append(
                        f"❌ {container_name}: Missing required environment variable '{var_name}' ({description})"
                    )
                elif not env_vars[var_name].strip():
                    self.errors.append(
                        f"❌ {container_name}: Environment variable '{var_name}' is empty"
                    )
                else:
                    self.passed.append(
                        f"✅ {container_name}: {var_name}='{env_vars[var_name]}'"
                    )
        else:
            self.passed.append(
                f"✅ {container_name}: Infrastructure service (PROJECT_NAME/PROJECT_PATH not required)"
            )
        
        # Check container service variables (warnings for missing, errors for localhost)
        for var_name, expected_value in self.CONTAINER_ENV_VARS.items():
            if var_name in env_vars:
                actual_value = env_vars[var_name]
                
                # Error: Using localhost instead of host.docker.internal
                if 'localhost' in actual_value:
                    self.errors.append(
                        f"❌ {container_name}: {var_name}='{actual_value}' uses 'localhost' "
                        f"(should use 'host.docker.internal' for container-to-host communication)"
                    )
                # Warning: Different from expected
                elif actual_value != expected_value:
                    self.warnings.append(
                        f"⚠️  {container_name}: {var_name}='{actual_value}' "
                        f"(expected '{expected_value}')"
                    )
                else:
                    self.passed.append(
                        f"✅ {container_name}: {var_name}='{actual_value}'"
                    )
            else:
                self.warnings.append(
                    f"⚠️  {container_name}: Missing service variable '{var_name}' "
                    f"(may use defaults)"
                )
    
    def _validate_container_config(self, container_name: str, container_info: Dict):
        """Validate container configuration"""
        
        # Check network mode
        network_mode = container_info['HostConfig']['NetworkMode']
        if 'l9-graphrag-network' not in network_mode:
            self.warnings.append(
                f"⚠️  {container_name}: Not using l9-graphrag-network "
                f"(current: {network_mode})"
            )
        else:
            self.passed.append(
                f"✅ {container_name}: Using correct network: {network_mode}"
            )
    
    def validate_code_patterns(self, directory: Path) -> bool:
        """Validate code follows ADR-0037 patterns"""
        logger.info(f"🔍 Validating code patterns in: {directory}")
        
        python_files = list(directory.rglob("*.py"))
        
        for file_path in python_files:
            self._validate_python_file(file_path)
        
        return len(self.errors) == 0
    
    def _validate_python_file(self, file_path: Path):
        """Validate Python file for ADR-0037 compliance"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for compliant patterns
            compliant_patterns = [
                r'os\.getenv\(["\']PROJECT_NAME["\']',
                r'os\.getenv\(["\']PROJECT_PATH["\']',
                r'ADR-0037',
                r'host\.docker\.internal'
            ]
            
            # Check for non-compliant patterns
            non_compliant_patterns = [
                (r'directories\[0\]', 'Using filesystem order instead of env vars'),
                (r'os\.path\.basename.*selected_dir', 'Deriving project name from directory'),
                (r'localhost:\d+', 'Using localhost instead of host.docker.internal'),
                (r'172\.18\.0\.\d+', 'Using Docker internal IP'),
            ]
            
            try:
                file_relative = file_path.relative_to(Path.cwd())
            except ValueError:
                file_relative = file_path
            
            # Check for compliant patterns
            for pattern in compliant_patterns:
                if re.search(pattern, content):
                    self.passed.append(f"✅ {file_relative}: Contains ADR-0037 compliant pattern")
                    break
            
            # Check for non-compliant patterns
            for pattern, issue in non_compliant_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    self.errors.append(
                        f"❌ {file_relative}:{line_num}: {issue} - '{match.group()}'"
                    )
        
        except Exception as e:
            self.warnings.append(f"⚠️  Could not validate {file_path}: {e}")
    
    def list_containers(self) -> List[str]:
        """List running containers that might need validation"""
        try:
            result = subprocess.run([
                'docker', 'ps', '--format', '{{.Names}}'
            ], capture_output=True, text=True, check=True)
            
            containers = result.stdout.strip().split('\n')
            
            # Filter for L9 containers
            l9_containers = [
                c for c in containers 
                if any(keyword in c.lower() for keyword in [
                    'indexer', 'neural', 'l9', 'claude', 'graphrag'
                ])
            ]
            
            return l9_containers
            
        except subprocess.CalledProcessError:
            logger.error("Failed to list Docker containers")
            return []
    
    def print_report(self):
        """Print validation report"""
        print("\n" + "="*60)
        print("ADR-0037 Configuration Validation Report")
        print("="*60)
        
        if self.passed:
            print(f"\n✅ PASSED ({len(self.passed)}):")
            for item in self.passed:
                print(f"   {item}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"   {item}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for item in self.errors:
                print(f"   {item}")
        
        print(f"\nSUMMARY: {len(self.passed)} passed, {len(self.warnings)} warnings, {len(self.errors)} errors")
        
        if self.errors:
            print("\n🚨 ADR-0037 COMPLIANCE: FAILED")
            print("Fix the errors above before deploying containers.")
        else:
            print("\n🎉 ADR-0037 COMPLIANCE: PASSED")
            print("All containers follow environment variable priority standard.")
        
        return len(self.errors) == 0


def main():
    parser = argparse.ArgumentParser(description='Validate ADR-0037 configuration compliance')
    parser.add_argument('--container', help='Validate specific container')
    parser.add_argument('--all', action='store_true', help='Validate all L9 containers')
    parser.add_argument('--validate-code', help='Validate code patterns in directory')
    parser.add_argument('--list', action='store_true', help='List L9 containers')
    
    args = parser.parse_args()
    
    validator = ADR0037Validator()
    success = True
    
    if args.list:
        containers = validator.list_containers()
        print("L9 Containers found:")
        for container in containers:
            print(f"  - {container}")
        return 0
    
    if args.container:
        success = validator.validate_container(args.container)
    
    if args.all:
        containers = validator.list_containers()
        if not containers:
            logger.warning("No L9 containers found")
        else:
            for container in containers:
                validator.validate_container(container)
    
    if args.validate_code:
        code_dir = Path(args.validate_code)
        if not code_dir.exists():
            logger.error(f"Directory not found: {code_dir}")
            return 1
        success = validator.validate_code_patterns(code_dir) and success
    
    if not any([args.container, args.all, args.validate_code]):
        # Default: validate all containers and common code directories
        containers = validator.list_containers()
        for container in containers:
            validator.validate_container(container)
        
        # Validate key code directories
        for code_dir in ['docker/scripts', 'neural-tools/src/servers']:
            dir_path = Path(code_dir)
            if dir_path.exists():
                validator.validate_code_patterns(dir_path)
    
    validator.print_report()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())