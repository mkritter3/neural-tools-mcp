# Neural SDK

**Zero-config GraphRAG integration for any project**

Transform any codebase into an intelligent, searchable knowledge graph with a single command. No manual Docker management, no complex configuration files, no project-specific setup.

## 🚀 Quick Start

```bash
# Install globally
npm install -g @neural-tools/sdk

# Initialize in any project (zero-config)
cd your-project
neural init

# That's it! Your project is now GraphRAG-enabled
```

## ✨ What You Get

- **🧠 GraphRAG Integration** - Hybrid vector + knowledge graph search across your codebase
- **🔍 Intelligent Indexing** - Auto-detects and indexes all project files with smart filtering
- **👀 Real-time Updates** - File watcher automatically re-indexes changes
- **🏗️ Multi-Project Support** - Isolated databases per project, single container
- **🔧 Claude Code MCP** - Seamless integration with Claude Code CLI
- **📊 Rich Analytics** - Project insights, dependency graphs, semantic search

## 🎯 Modern SDK Design

Following 2025 patterns for developer tools:

### Zero-Config Defaults
```bash
neural init  # Detects project type, language, frameworks automatically
```

### Container Abstraction
- No Docker commands
- No manual container management  
- No port conflicts
- Auto-pull required images

### Intelligent Discovery
- Auto-detects: Node.js, Python, Rust, Go, Java, C#, C++
- Recognizes frameworks: React, Vue, Django, FastAPI, Express, etc.
- Finds source directories, test files, documentation

### API-First Integration
```bash
neural status    # System health check
neural watch     # Start file monitoring  
neural logs -f   # Stream logs
neural reset     # Clean slate
```

## 📋 Commands

### `neural init`
Initialize Neural Tools in current project

```bash
neural init                    # Interactive setup
neural init -y                 # Skip prompts
neural init -g                 # Global installation
neural init --port 3001        # Custom port
```

### `neural status`
Check system health and project metrics

```bash
neural status
```
```
🔍 Neural Tools Status

Containers:
  🟢 neural-myproject - running
  
MCP Integration:
  🟢 Configuration: Active
  
Project:
  📁 Name: myproject  
  📊 Indexed files: 247
  🕒 Last updated: 2025-01-15T10:30:22Z
```

### `neural watch`
Intelligent file watching with auto-indexing

```bash
neural watch                                    # Default patterns
neural watch --patterns "**/*.{ts,py,md}"     # Custom patterns
neural watch --ignore "node_modules,dist"     # Custom ignore
```

### `neural stop / logs / reset`
```bash
neural stop           # Stop services
neural logs -f        # Follow logs  
neural logs --lines 50 # Last 50 lines
neural reset          # Complete reset (removes all data)
```

## 🏗️ Architecture

### Single Container, Multi-Project
- One container handles multiple projects
- Path-based project detection: `/workspace/project-name/`
- Isolated databases: Neo4j + Qdrant per project
- Zero container management

### MCP Auto-Configuration
```json
{
  "mcpServers": {
    "neural-tools-myproject": {
      "command": "docker",
      "args": ["exec", "-i", "neural-myproject", "python3", "/app/src/neural_mcp/neural_server_stdio.py"],
      "env": {
        "PROJECT_NAME": "myproject",
        "PROJECT_DIR": "/workspace/myproject"
      }
    }
  }
}
```

### File Watching Intelligence
- **Smart Filtering**: Skips binaries, large files, hidden files
- **Debounced Processing**: Batches rapid changes (500ms window)
- **Multi-language Support**: Handles all major programming languages
- **Real-time Updates**: Changes reflected immediately in search

## 🔧 Project Detection

Automatically detects:

| Project Type | Indicators | Package Manager |
|--------------|-----------|----------------|
| Node.js | `package.json` | npm, yarn, pnpm |
| Python | `pyproject.toml`, `requirements.txt` | pip, poetry |
| Rust | `Cargo.toml` | cargo |
| Go | `go.mod` | go modules |
| Java | `pom.xml`, `build.gradle` | maven, gradle |
| C# | `*.csproj`, `*.sln` | nuget |
| C++ | `CMakeLists.txt`, `Makefile` | cmake |

Plus frameworks: React, Vue, Django, FastAPI, Express, NestJS, and many more.

## 🎛️ Advanced Usage

### Custom Container Image
```bash
neural init --custom-image my-neural-tools:latest
```

### Environment Variables
```bash
export NEURAL_SDK_PORT=3001
export NEURAL_SDK_LOG_LEVEL=debug
neural init
```

### Programmatic API
```javascript
import { NeuralSDK } from '@neural-tools/sdk';

const sdk = new NeuralSDK();
const projectInfo = await sdk.detectProject('./my-app');

await sdk.initialize(projectInfo, {
  autoStart: true,
  port: 3000
});

const status = await sdk.getSystemStatus();
console.log(status);
```

## 📊 Integration Examples

### With Claude Code
```bash
# Initialize in your project
neural init

# Now available in Claude Code
# Use tools like:
# - neural_system_status
# - semantic_code_search  
# - neo4j_graph_query
# - memory_store_enhanced
```

### With CI/CD
```yaml
# .github/workflows/neural-index.yml
- name: Index codebase
  run: |
    npm install -g @neural-tools/sdk
    neural init -y
    neural watch &
    # Your tests run with intelligent code context
```

### Multi-Repository Setup
```bash
# Each repo gets isolated environment
cd project-a && neural init  # -> neural-project-a container
cd project-b && neural init  # -> neural-project-b container
cd project-c && neural init  # -> neural-project-c container

# All share base container image, isolated data
```

## 🛠️ Troubleshooting

### Docker Issues
```bash
# Check Docker
docker info

# Pull required images manually
docker pull neural-tools/graphrag:latest
docker pull qdrant/qdrant:v1.8.0  
docker pull neo4j:5.20.0
```

### Port Conflicts
```bash
neural init --port 3001  # Use different port
```

### Reset Everything
```bash
neural reset  # Removes all data and containers
```

### Debug Mode
```bash
DEBUG=neural:* neural init  # Verbose logging
neural logs -f             # Watch container logs
```

## 🚦 Requirements

- **Node.js** >= 18.0.0
- **Docker** Desktop or Engine
- **Available Ports**: 3000 (configurable)
- **Disk Space**: ~2GB for container images

## 📝 License

MIT - Use freely in any project

## 🤝 Contributing

We welcome contributions! The SDK is designed to be:
- **Extensible**: Easy to add new project types and frameworks
- **Testable**: Comprehensive test coverage 
- **Maintainable**: Clear separation of concerns
- **Documented**: Everything has examples

See `CONTRIBUTING.md` for development setup.

---

**Made with ❤️ by the Neural Tools team**

*Transform any codebase into intelligent, searchable knowledge*