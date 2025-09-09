#!/usr/bin/env python3
"""
Start the Neural Search API server
Handles dependency installation and server startup
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install FastAPI dependencies if needed"""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies already installed")
        return True
    except ImportError:
        print("📦 Installing FastAPI dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "fastapi>=0.104.0",
                "uvicorn[standard]>=0.24.0", 
                "pydantic>=2.4.0",
                "httpx>=0.25.0"
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

def start_server():
    """Start the FastAPI server"""
    api_dir = Path(__file__).parent
    main_file = api_dir / "main.py"
    
    if not main_file.exists():
        print(f"❌ Main API file not found: {main_file}")
        return False
    
    print("🚀 Starting Neural Search API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📋 API docs available at: http://localhost:8000/docs")
    print("🔄 Hot reload enabled for development")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Change to API directory
        os.chdir(api_dir)
        
        # Start uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        return False

def main():
    """Main entry point"""
    print("🧠 Neural Search API Server Startup")
    print("=" * 40)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Start server
    if not start_server():
        sys.exit(1)

if __name__ == "__main__":
    main()