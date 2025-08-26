# Neural Flow Docker Integration Guide

## 🏗️ L9 Architecture Overview

This Docker containerization provides **seamless Claude Code integration** while solving dependency conflicts and enabling multi-project isolation. The architecture preserves all existing neural-flow capabilities while adding enterprise-grade deployment reliability.

## 🔧 Quick Setup

### Prerequisites
- Docker & Docker Compose
- Claude Code installed (`npm install -g @anthropic-ai/claude-code`)

### Installation

```bash
# 1. Build the neural flow container
docker-compose build neural-flow

# 2. Initialize your first project  
./scripts/neural-flow.sh init my-awesome-project

# 3. Start neural flow for the project
./scripts/neural-flow.sh start my-awesome-project

# 4. Navigate to project and start Claude Code
cd projects/my-awesome-project
claude
```

That's it! Neural Flow will be available as an MCP server within Claude Code.

## 🎯 Key Benefits

### ✅ **Dependency Isolation**
- **Problem Solved**: PyTorch conflicts, complex requirements, version mismatches
- **Solution**: All dependencies containerized, zero impact on host system
- **Result**: `pip install sentence-transformers` just works, every time

### ✅ **Multi-Project Data Isolation** 
- **Problem Solved**: Cross-project data contamination, resource conflicts
- **Solution**: Dedicated `.claude/` directories per project with volume mounts
- **Result**: Perfect project separation, no data bleeding

### ✅ **Resource Efficiency**
- **Problem Solved**: Duplicate model downloads, memory bloat
- **Solution**: Shared model cache, optimized container size (~1GB)
- **Result**: Qodo-Embed-1.5B downloaded once, shared across all projects

### ✅ **Seamless Claude Code Integration**
- **Problem Solved**: Complex MCP server registration and configuration
- **Solution**: Automatic `.mcp.json` generation, stdio transport compatibility
- **Result**: Neural Flow appears as native Claude Code capability

## 📁 Architecture Components

### **Container Structure**
```
Docker Container:
├── /app/neural-system/        # Neural Flow implementation
├── /app/mcp-server/           # MCP protocol server
├── /app/data/                 # Mounted project .claude/ directory
├── /app/models/               # Shared embedding models cache
├── /app/project/              # Mounted project source (read-only)
└── /app/docker-entrypoint.sh  # Container startup script
```

### **Volume Mounts**
```
Host → Container Mapping:
./projects/my-project/.claude/  → /app/data/           # Project data
./projects/my-project/src/      → /app/project/        # Source code
neural-flow-models:             → /app/models/         # Model cache
```

### **Network Architecture**
```
Claude Code ←→ Docker Container (stdio) ←→ Neural Systems
           ↑
        MCP Protocol
      (JSON-RPC 2.0)
```

## 🔄 Multi-Project Workflow

### **Project Initialization**
```bash
# Create new project with neural intelligence
./scripts/neural-flow.sh init new-project

# This creates:
# ├── projects/new-project/
# │   ├── src/                      # Your source code
# │   ├── .claude/
# │   │   ├── chroma/               # Vector database
# │   │   ├── memory/               # SQLite databases  
# │   │   └── settings.json         # Claude Code settings
# │   ├── .mcp.json                 # MCP server registration
# │   └── README.md                 # Project documentation
```

### **Project Switching**
```bash
# Switch between projects seamlessly
./scripts/neural-flow.sh stop old-project
./scripts/neural-flow.sh start new-project

# Each project maintains separate:
# - Vector embeddings
# - Conversation memory
# - Feature flag settings
# - Performance metrics
```

### **Development Workflow**
```bash
# Terminal 1: Start neural flow
./scripts/neural-flow.sh start my-project

# Terminal 2: Development
cd projects/my-project
claude  # Neural Flow automatically available

# Terminal 3: Monitor logs
./scripts/neural-flow.sh logs my-project
```

## 🧠 Neural Capabilities Available

### **Memory Management**
- **Tool**: `neural_memory_search` - Semantic conversation search
- **Tool**: `neural_memory_index` - Index new memories with priority
- **Resource**: `neural://memory/conversations` - Memory statistics

### **Code Intelligence** 
- **Tool**: `neural_project_search` - AST-aware code search
- **Resource**: `neural://project/index` - Project indexing status

### **Configuration Management**
- **Resource**: `neural://config/features` - Feature flags and A/B tests

## ⚙️ Configuration Options

### **Environment Variables** (via `.mcp.json`)
```json
{
  "neural-flow": {
    "env": {
      "PROJECT_NAME": "my-project",
      "USE_QODO_EMBED": "true",           // Enable code-specific embeddings
      "ENABLE_AB_TESTING": "true",        // A/B test different models
      "ENABLE_PERFORMANCE_MONITORING": "true",  // Track metrics
      "EMBEDDING_MODEL_PRIORITY": "qodo,openai,onnx"
    }
  }
}
```

### **Claude Code Integration** (via `.claude/settings.json`)
```json
{
  "enableAllProjectMcpServers": true,
  "env": {
    "PROJECT_NAME": "my-project",
    "USE_QODO_EMBED": "true"
  },
  "permissions": {
    "allow": ["*"],
    "deny": [".env", "secrets/**"]
  }
}
```

## 🚀 Performance Characteristics

### **Container Resource Usage**
- **Memory**: ~300MB baseline + model loading
- **CPU**: Native performance (no virtualization overhead)
- **Storage**: ~1GB container + shared model cache
- **Startup**: ~5-10 seconds (includes model loading)

### **Embedding Performance**
- **ONNX Baseline**: ~2-5ms per embedding
- **Qodo-Embed**: ~10-20ms per embedding (CPU inference)
- **Batch Processing**: ~100-500 embeddings/second
- **Model Cache**: Shared across projects (efficiency)

## 🔍 Monitoring & Debugging

### **Container Health**
```bash
# Check container status
./scripts/neural-flow.sh status

# View logs
./scripts/neural-flow.sh logs my-project

# Open container shell
./scripts/neural-flow.sh shell my-project
```

### **Neural System Health**
```bash
# Inside container or via logs
python3 -c "from neural_embeddings import get_neural_system; print(get_neural_system().vector_store.get_stats())"
```

### **MCP Protocol Debug**
```bash
# Test MCP communication
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | docker-compose exec -T neural-flow python3 -m mcp_neural_server
```

## 🛠️ Advanced Usage

### **Custom Model Configuration**
```bash
# Use different embedding model
export USE_CODESTRAL_EMBED=true
./scripts/neural-flow.sh start my-project
```

### **Development Mode**
```bash
# Mount development code for live editing
docker-compose -f docker-compose.dev.yml up neural-flow
```

### **Production Deployment**
```bash
# Optimized for production
docker-compose -f docker-compose.prod.yml up -d neural-flow
```

## 🔒 Security Considerations

### **Container Security**
- Non-root user execution
- Read-only project source mounts  
- Isolated network (no external access required)
- Resource limits (memory/CPU)

### **Data Privacy**
- All processing local (no external API calls for embeddings)
- Project data isolation (no cross-contamination)
- Optional telemetry (disabled by default)

## 🚨 Troubleshooting

### **Common Issues**

**"Container fails to start"**
```bash
# Check logs for specific error
./scripts/neural-flow.sh logs my-project

# Common fixes:
docker system prune -f  # Clean old containers
./scripts/neural-flow.sh clean  # Clean neural flow specifically
```

**"MCP server not found"**
```bash
# Verify .mcp.json exists and is valid
cat projects/my-project/.mcp.json

# Verify Claude Code can see MCP servers
claude /mcp list
```

**"Qodo-Embed fails to load"**
```bash
# Check if sentence-transformers installed properly
./scripts/neural-flow.sh shell my-project
pip list | grep sentence-transformers

# Fallback to ONNX embeddings
export USE_QODO_EMBED=false
```

### **Performance Issues**

**"Slow embedding generation"**
- Check CPU usage: Docker may need more CPU allocation
- Consider reducing batch sizes in feature flags
- Monitor memory usage - container may need more RAM

**"Model download timeouts"**
- Models download on first use (~1.5GB for Qodo-Embed)
- Check network connectivity from container
- Use model cache volume to persist across restarts

## 🎯 Next Steps

1. **Initialize your first project**: `./scripts/neural-flow.sh init my-project`
2. **Start neural flow**: `./scripts/neural-flow.sh start my-project` 
3. **Open Claude Code**: `cd projects/my-project && claude`
4. **Test neural capabilities**: Ask Claude to search your code or remember context
5. **Monitor performance**: Use evaluation harness to track improvements

---

**🎉 Congratulations!** You now have an L9-grade neural intelligence system integrated seamlessly with Claude Code, providing dependency-free deployment with multi-project isolation.