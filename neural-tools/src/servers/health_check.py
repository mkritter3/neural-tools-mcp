#!/usr/bin/env python3
"""
L9 Simple Health Check for Neural MCP Server
Checks service connectivity without complex orchestration
"""

import json
import subprocess
import sys
from typing import Dict, Any

def check_mcp_health() -> Dict[str, Any]:
    """Check if MCP server can initialize properly"""
    try:
        # Send initialize request to MCP server
        init_request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "HealthCheck", "version": "1.0.0"}
            }
        })
        
        # Test via docker exec with all required env vars
        cmd = [
            "docker", "exec", "-i",
            "-e", "QDRANT_HOST=localhost",
            "-e", "QDRANT_HTTP_PORT=6681",
            "-e", "QDRANT_GRPC_PORT=6681",
            "-e", "NEO4J_HOST=localhost",
            "-e", "NEO4J_PORT=7688",
            "-e", "NEO4J_USERNAME=neo4j",
            "-e", "NEO4J_PASSWORD=neural-l9-2025",
            "-e", "EMBEDDING_SERVICE_HOST=localhost",
            "-e", "EMBEDDING_SERVICE_PORT=8081",
            "-e", "PYTHONPATH=/app/project/neural-tools/src",
            "-e", "PROJECT_NAME=claude-l9-template",
            "-e", "PROJECT_DIR=/app/project",
            "-e", "PYTHONUNBUFFERED=1",
            "default-neural",
            "python3", "-u",
            "/app/project/neural-tools/src/servers/neural_server_stdio.py"
        ]
        
        result = subprocess.run(
            cmd,
            input=init_request.encode(),
            capture_output=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return {
                "status": "unhealthy",
                "error": f"MCP server returned non-zero exit code: {result.returncode}",
                "stderr": result.stderr.decode()[:500]
            }
        
        # Parse the response
        response_line = result.stdout.decode().split('\n')[0]
        response = json.loads(response_line)
        
        if "result" in response:
            return {
                "status": "healthy",
                "mcp_version": response["result"].get("protocolVersion"),
                "server_name": response["result"].get("serverInfo", {}).get("name"),
                "capabilities": list(response["result"].get("capabilities", {}).keys())
            }
        elif "error" in response:
            return {
                "status": "unhealthy", 
                "error": response["error"]
            }
        else:
            return {
                "status": "unhealthy",
                "error": "Invalid response format",
                "response": response
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "unhealthy",
            "error": "MCP server initialization timed out"
        }
    except json.JSONDecodeError as e:
        return {
            "status": "unhealthy",
            "error": f"Failed to parse MCP response: {e}"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": f"Health check failed: {e}"
        }

def main():
    """Run health check and report status"""
    health = check_mcp_health()
    
    # Print result
    print(json.dumps(health, indent=2))
    
    # Exit with appropriate code
    if health["status"] == "healthy":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()