#!/usr/bin/env python3
"""
L9 MCP Launcher - Production-grade MCP server with proper lifecycle management
Ensures no Docker container sprawl and clean resource management
"""

import os
import sys
import subprocess
import signal
import atexit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class L9MCPLauncher:
    """L9-grade MCP launcher with proper lifecycle management"""
    
    def __init__(self):
        self.process = None
        self.container_id = None
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("🛑 Received shutdown signal, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up any running processes or containers"""
        # Kill any zombie Docker containers
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", "ancestor=neural-flow:production"],
                capture_output=True, text=True, check=False
            )
            if result.stdout.strip():
                containers = result.stdout.strip().split('\n')
                for container in containers:
                    subprocess.run(["docker", "rm", "-f", container], 
                                 capture_output=True, check=False)
                logger.info(f"✅ Cleaned up {len(containers)} Docker containers")
        except Exception:
            pass  # Docker might not be available
        
        # Terminate local process if running
        if self.process:
            self.process.terminate()
            self.process = None
    
    def run_local(self):
        """Run MCP server locally with Python"""
        logger.info("🚀 Starting L9 MCP server (local mode)...")
        
        # Set up environment
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': os.path.dirname(os.path.abspath(__file__)),
            'NEURAL_L9_MODE': '1',
            'USE_SINGLE_QODO_MODEL': '1',
            'ENABLE_AUTO_SAFETY': '1',
            'L9_PROTECTION_LEVEL': 'maximum'
        })
        
        # Start MCP server
        mcp_script = os.path.join(os.path.dirname(__file__), 'mcp_neural_server.py')
        self.process = subprocess.Popen(
            [sys.executable, mcp_script],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )
        
        # Wait for process to complete
        self.process.wait()
    
    def run_docker(self):
        """Run MCP server in Docker with proper lifecycle management"""
        logger.info("🐳 Starting L9 MCP server (Docker mode)...")
        
        # Always use --rm flag to auto-remove container
        cmd = [
            "docker", "run",
            "--rm",  # Auto-remove container when it stops
            "-i",
            "--name", "neural-flow-mcp-l9",  # Named container for easy management
            "--env", "NEURAL_L9_MODE=1",
            "--env", "USE_SINGLE_QODO_MODEL=1",
            "--env", "ENABLE_AUTO_SAFETY=1",
            "-v", f"{os.getcwd()}/.claude:/app/data",
            "neural-flow:l9-production",
            "python3", "/app/mcp_neural_server.py"
        ]
        
        # Check if container already exists and remove it
        existing = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "name=neural-flow-mcp-l9"],
            capture_output=True, text=True, check=False
        )
        if existing.stdout.strip():
            subprocess.run(["docker", "rm", "-f", "neural-flow-mcp-l9"],
                         capture_output=True, check=False)
            logger.info("✅ Removed existing container")
        
        # Start new container
        self.process = subprocess.Popen(cmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        self.process.wait()
    
    def run(self):
        """Determine best execution mode and run"""
        # Check if Docker is available and image exists
        docker_available = subprocess.run(
            ["docker", "images", "-q", "neural-flow:l9-production"],
            capture_output=True, text=True, check=False
        ).returncode == 0 and subprocess.run(
            ["docker", "images", "-q", "neural-flow:l9-production"],
            capture_output=True, text=True, check=False
        ).stdout.strip()
        
        if docker_available and os.getenv('USE_DOCKER_MCP', 'false').lower() == 'true':
            self.run_docker()
        else:
            self.run_local()

if __name__ == "__main__":
    launcher = L9MCPLauncher()
    launcher.run()