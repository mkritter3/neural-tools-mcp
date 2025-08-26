# 🚀 Neural Flow Quick Start Guide

## One-Command Project Setup

Yes! Neural Flow provides an incredibly easy CLI/SDK for automatic project setup. Here's how simple it is:

### 🔧 Installation Options

#### Option 1: Global Installation (Recommended)
```bash
# Install Neural Flow globally
./install-global.sh

# Now available anywhere:
neural-init my-awesome-app python
cd my-awesome-app  
./neural-start.sh  # Starts neural flow + Claude Code automatically
```

#### Option 2: Local Usage
```bash  
# From neural-flow directory
./scripts/neural-init my-project python
cd my-project
./neural-start.sh
```

## 🎯 Available Templates

Neural Flow automatically configures projects for different tech stacks:

```bash
# Python project with proper structure
neural-init my-ai-app python

# React application  
neural-init my-frontend react

# Next.js application
neural-init my-webapp nextjs

# JavaScript/Node.js project
neural-init my-backend javascript

# Rust project with Cargo
neural-init my-systems-app rust

# Go module project
neural-init my-service go

# General purpose (any language)
neural-init my-project general
```

## ✨ What Each Template Creates

### 🐍 **Python Template**
```
my-ai-app/
├── src/main.py              # Sample Python code
├── requirements.txt         # Dependencies  
├── tests/                   # Test directory
├── .claude/                 # Neural Flow configuration
│   ├── settings.json        # Claude Code integration
│   ├── chroma/              # Vector database
│   └── memory/              # SQLite memory
├── .mcp.json                # MCP server registration
├── neural-start.sh          # One-command startup
├── neural-stop.sh           # Container management
└── README.md                # Documentation with neural examples
```

### ⚛️ **React Template**  
```
my-frontend/
├── src/
│   ├── App.jsx              # React component
│   └── components/          # Component directory
├── package.json             # React dependencies
├── .claude/                 # Neural configuration  
├── neural-start.sh          # Development startup
└── README.md                # React + Neural Flow guide
```

### 🦀 **Rust Template**
```
my-systems-app/
├── src/main.rs              # Rust source
├── Cargo.toml               # Rust manifest
├── .claude/                 # Neural intelligence
├── neural-start.sh          # Cargo + Neural startup
└── README.md                # Rust + AI development guide
```

## 🎬 Complete Development Workflow

### Step 1: Initialize Project
```bash
neural-init my-awesome-app python
```

**What this does automatically:**
- ✅ Creates proper project structure for Python
- ✅ Configures `.mcp.json` for Claude Code integration
- ✅ Sets up `.claude/settings.json` with optimal permissions
- ✅ Creates neural-start.sh for one-command development
- ✅ Configures Docker volume mounts and environment variables
- ✅ Generates comprehensive README with neural examples

### Step 2: Start Development
```bash
cd my-awesome-app
./neural-start.sh
```

**What happens:**
1. 🐋 Starts Docker container with neural intelligence
2. ⏱️ Waits for container health check (5-10 seconds)
3. 🧠 Loads Qodo-Embed model (first time: ~30 seconds)
4. 🔮 Launches Claude Code with neural capabilities active
5. 📊 Neural Flow MCP server automatically available

### Step 3: AI-Powered Development

Once Claude Code starts, you immediately have:

```
# Semantic code search
"Find all database connection logic"
"Show me authentication patterns"

# Conversation memory  
"Remember we decided to use FastAPI for the API"
"What was that performance issue we discussed?"

# Code understanding
"Explain how this function works"
"What would happen if I modify this parameter?"
"Help me refactor this class"
```

### Step 4: Stop When Done
```bash
./neural-stop.sh  # Stops container cleanly
```

## 🔄 Multi-Project Management

Switch between projects effortlessly:

```bash
# Start working on different project
neural-flow stop my-awesome-app
neural-flow start my-other-project

# Or use the project's local scripts
cd my-other-project
./neural-start.sh  # Auto-stops other containers
```

## 🎛️ Advanced Configuration

### Custom Environment Variables
Edit `.mcp.json` in your project:

```json
{
  "neural-flow": {
    "env": {
      "USE_QODO_EMBED": "true",           // Code-specific embeddings
      "ENABLE_AB_TESTING": "true",        // Compare models
      "ENABLE_PERFORMANCE_MONITORING": "true",
      "EMBEDDING_MODEL_PRIORITY": "qodo,openai,onnx"
    }
  }
}
```

### Claude Code Integration
Edit `.claude/settings.json`:

```json
{
  "enableAllProjectMcpServers": true,
  "permissions": {
    "allow": ["*"],
    "deny": [".env*", "secrets/**"]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command", 
            "command": "echo '🧠 Neural Flow: Code updated'"
          }
        ]
      }
    ]
  }
}
```

## 🚀 Project Templates in Detail

### **Python AI/ML Projects**
```bash
neural-init my-ml-project python
```
- Pre-configured for data science workflows
- Understands pandas, numpy, scikit-learn patterns  
- Neural Flow helps navigate complex ML codebases
- Example: "Find where we preprocess the training data"

### **React/Frontend Projects**
```bash
neural-init my-ui-project react  
```
- Component relationship understanding
- State management pattern recognition
- Hook usage analysis  
- Example: "Show me all components that use the user state"

### **Backend/API Projects**  
```bash
neural-init my-api-project javascript
```
- Route handler semantic search
- Database integration patterns
- Authentication flow understanding
- Example: "Find all endpoints that require admin auth"

## 🔍 Debugging and Monitoring

### Check Neural Flow Status
```bash
neural-flow status
# Shows:
# - Running containers 
# - Available projects
# - Resource usage
```

### Monitor Container Logs
```bash
neural-flow logs my-project
# Real-time neural system logs
# Model loading progress
# Embedding generation metrics
```

### Container Shell Access  
```bash
neural-flow shell my-project
# Direct access to neural systems
# Debug embedding models
# Check memory/CPU usage
```

## 🎯 Key Benefits Summary

| **Traditional Setup** | **Neural Flow Setup** |
|----------------------|---------------------|
| Hours configuring dependencies | ✅ `neural-init my-app python` (30 seconds) |
| Manual Docker/MCP configuration | ✅ Automatic `.mcp.json` generation |  
| Project environment conflicts | ✅ Perfect isolation per project |
| No AI code understanding | ✅ L9-grade semantic intelligence |
| Complex multi-project switching | ✅ `./neural-start.sh` in any project |

## 🤝 Integration Examples

### With Existing Projects
```bash  
# Add neural intelligence to existing project
cd my-existing-project
neural-init . python  # Initialize in current directory
./neural-start.sh     # Neural capabilities added!
```

### With Git Workflows
```bash
neural-init my-repo python
cd my-repo
git init
git add .
git commit -m "Initialize with Neural Flow intelligence"

# Neural Flow understands git history:
# "What files changed in the last commit?"
# "Show me recent authentication-related changes"
```

### With Team Development
```bash
# .mcp.json and .claude/ are committed to git
# Team members just need:
git clone repo
./neural-start.sh  # Everything works instantly
```

---

**🎉 That's it!** Neural Flow provides the easiest possible setup for AI-powered development. One command gets you from zero to L9-grade intelligent coding environment.