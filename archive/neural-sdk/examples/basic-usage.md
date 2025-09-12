# Neural SDK - Basic Usage Examples

## 🚀 Quick Start Example

### 1. Install Neural SDK
```bash
npm install -g @neural-tools/sdk
```

### 2. Initialize in Any Project
```bash
cd my-awesome-project
neural init
```

**Expected Output:**
```
🧠 Neural Tools Installation
==================================

Detected node project: my-awesome-project
Initialize Neural Tools for "my-awesome-project"? (Y/n) y

[1/6] Checking Docker availability...
[2/6] Pulling Neural Tools container...  
[3/6] Creating project-specific services...
[4/6] Configuring MCP integration...
[5/6] Starting services...
[6/6] Validating setup...

🎉 Neural Tools ready!

✅ Ready to go!

📚 Available commands:
  neural status    - Check system status
  neural watch     - Start file watching  
  neural stop      - Stop services
  neural logs      - View logs

💡 Your project files are now automatically indexed!
```

## 📊 Check System Status

```bash
neural status
```

**Output:**
```
🔍 Neural Tools Status

Containers:
  🟢 neural-my-awesome-project - running

MCP Integration:  
  🟢 Configuration: Active

Project:
  📁 Name: my-awesome-project
  📊 Indexed files: 127  
  🕒 Last updated: 2025-01-15T14:22:10Z
```

## 👀 File Watching

```bash
neural watch
```

**Output:**
```
✅ File watcher started
👀 Watching for changes...
Press Ctrl+C to stop

📝 Indexing 3 new files...
✅ Processed 3 file changes

🔄 Re-indexing 1 changed files...  
✅ Processed 1 file changes
```

## 🏗️ Multi-Project Example

### Project A (React App)
```bash
cd ~/projects/ecommerce-frontend
neural init
# Creates: neural-ecommerce-frontend
```

### Project B (Python API)  
```bash
cd ~/projects/api-backend
neural init  
# Creates: neural-api-backend
```

### Project C (Go Service)
```bash
cd ~/projects/payment-service
neural init
# Creates: neural-payment-service  
```

**Result:** Each project gets isolated GraphRAG environment:
- Separate Neo4j databases
- Separate Qdrant collections
- Separate MCP configurations  
- Single shared container architecture

## 🔧 Advanced Configuration

### Custom Port
```bash
neural init --port 3001
```

### Global Installation
```bash
neural init --global
# Adds to ~/.claude/mcp_config.json instead of ./.mcp.json
```

### Skip Prompts
```bash
neural init -y
# Uses all defaults, perfect for CI/CD
```

## 🎯 Claude Code Integration

After `neural init`, these tools are automatically available in Claude Code:

### Search Your Codebase
```
Hey Claude, can you search for authentication logic in my codebase?
```
*Uses: `semantic_code_search` tool*

### Analyze Dependencies  
```
Show me the dependency graph for the user module
```
*Uses: `neo4j_code_dependencies` tool*

### Store Context
```
Remember that we're using JWT tokens for auth and store this context
```  
*Uses: `memory_store_enhanced` tool*

### System Health
```
What's the status of my Neural Tools system?
```
*Uses: `neural_system_status` tool*

## 🛠️ Management Commands

### View Logs
```bash
neural logs            # Last 100 lines
neural logs -f         # Follow logs
neural logs --lines 50 # Last 50 lines
```

### Stop Services
```bash
neural stop
```

### Complete Reset
```bash
neural reset
# ⚠️ This will delete all indexed data. Continue? (y/N)
```

## 🔍 Project Detection Examples

### Node.js Project
```
my-app/
├── package.json ✓
├── src/
│   ├── components/
│   └── utils/
└── tests/

Detected: Node.js + React + Jest
Package Manager: npm
```

### Python Project  
```
ml-service/
├── pyproject.toml ✓
├── src/
│   └── models/
├── tests/
└── requirements.txt

Detected: Python + FastAPI
Package Manager: poetry
```

### Multi-Language Project
```
full-stack/
├── package.json ✓
├── pyproject.toml ✓ 
├── frontend/
├── backend/
└── shared/

Detected: Mixed (Node.js + Python)
```

## 🚨 Error Handling

### Docker Not Running
```bash
neural init
# ❌ Docker is not running
# Please start Docker Desktop and try again
```

### Port Conflict  
```bash
neural init
# ❌ Port 3000 is already in use
# Try: neural init --port 3001
```

### Container Issues
```bash
neural status
# 🔴 neural-my-project - failed

neural logs
# Check container logs for issues

neural reset
# Nuclear option: clean slate
```

## 📈 Performance Tips

### Large Projects
- SDK automatically skips binary files
- Ignores `node_modules`, `dist`, `.git` 
- Processes files in batches
- Uses intelligent debouncing

### Resource Usage
- Single container shared across projects
- ~500MB RAM per active project
- ~100MB disk per 1000 files indexed
- Minimal CPU when idle

This SDK eliminates all the friction from the current manual setup process and provides a modern, zero-config experience that follows 2025 developer tool patterns!