# Neural Flow MCP Server Setup Guide (Preloaded Version)

## Prerequisites

- Python 3.8+ (supports 3.8, 3.9, 3.10, 3.11, 3.12, 3.13+)
- pip package manager
- At least 500MB free disk space for models and databases
- ~300MB RAM for pre-loaded models

## Installation

### 1. Clone or download this repository

```bash
git clone <repository-url>
cd claude-l9-template
```

### 2. Install Python dependencies (Automatic - Recommended)

Run the intelligent installer that auto-detects your Python version:

```bash
python install.py
```

The installer will:
- ✅ Detect your Python version
- ✅ Check existing package compatibility
- ✅ Install the correct dependencies for your environment
- ✅ Verify the installation

### 2. Alternative: Manual Installation

If you prefer manual installation, use the appropriate requirements file:

```bash
# For Python 3.8-3.9 (NumPy 1.x, ChromaDB 0.4.x)
pip install -r requirements/requirements-py38.txt

# For Python 3.10-3.12 (NumPy 1.x, ChromaDB 0.5.x)
pip install -r requirements/requirements-py310.txt

# For Python 3.13+ (NumPy 2.x, ChromaDB 1.0.20+)
pip install -r requirements/requirements-py313.txt
```

### 3. Verify installation

Test that all imports work correctly:

```bash
python3 -c "
import sys
sys.path.insert(0, '.claude/mcp-tools')
from mcp_neural_server import server, get_neural_systems
from neural_dynamic_memory_system import NeuralDynamicMemorySystem
from project_neural_indexer import ProjectNeuralIndexer
print('✅ All imports successful - neural-flow server ready!')
"
```

## Troubleshooting

### NumPy Compatibility Issues

If you encounter errors like `AttributeError: np.float_ was removed in NumPy 2.0`:

```bash
# Ensure ChromaDB is updated
pip3 install chromadb>=1.0.20 --upgrade
```

### Python Version Issues

This project requires Python 3.13+ due to NumPy 2.x requirements. Check your version:

```bash
python3 --version
```

### MCP Connection Issues

If the neural-flow MCP server fails to connect:

1. Check the `.mcp.json` configuration file exists
2. Verify the path to `mcp_neural_server.py` is correct
3. Test the server manually:

```bash
# Test the MCP server directly
python3 .claude/mcp-tools/mcp_neural_server.py
# Press Ctrl+C to exit

# Run the test suite
python3 test_mcp_server.py
```

### Dependency Conflicts

If you see pip dependency resolver warnings, they can usually be ignored if the core packages install successfully. The key requirements are:

- `chromadb >= 1.0.20` (for NumPy 2.x compatibility)  
- `numpy >= 2.1.0` (for Python 3.13)
- `mcp >= 1.13.1` (for MCP server framework)
- `onnxruntime >= 1.20.1` (for neural embeddings)
- `tiktoken >= 0.9.0` (for tokenization)

## MCP Server Configuration

The neural-flow MCP server is configured in `.mcp.json` to use the preloaded version:

```json
{
  "mcpServers": {
    "neural-flow": {
      "command": "python3",
      "args": [
        ".claude/mcp-tools/mcp_neural_server.py"
      ],
      "env": {}
    }
  }
}
```

### Performance Characteristics

- **Startup time**: 2-3 seconds (pre-loads all models)
- **First query**: <50ms (models already loaded)
- **Subsequent queries**: <50ms (consistent performance)
- **Memory usage**: ~300MB constant
- **Fallback**: Automatically falls back to lazy loading if pre-load fails

## Features

Once installed, the neural-flow MCP server provides:

- **Neural Memory System**: Store and query memories with semantic search
- **Project Indexing**: Index codebases with neural embeddings
- **Semantic Search**: Find code and documentation using natural language
- **Dynamic Relevance Scoring**: Combines neural and temporal scoring

## Usage

After installation, the MCP server will be available to Claude when restarted. Available tools:

- `memory_query`: Search memories semantically (<50ms)
- `memory_store`: Store new memories with embeddings
- `memory_stats`: Get system statistics
- `index_project_files`: Index project with neural embeddings
- `search_project_files`: Search project semantically
- `familiarize_with_project`: Comprehensive project analysis
- `preload_status`: Check pre-load status (preloaded server only)

## Directory Structure

```
claude-l9-template/
├── .mcp.json                    # MCP server configuration
├── requirements.txt             # Python dependencies
├── .claude/
│   ├── mcp-tools/
│   │   ├── mcp_neural_server.py            # Robust MCP server with smart pre-loading
│   │   ├── neural_flow_tools.py            # Core neural tools (used as fallback)
│   │   ├── neural_dynamic_memory_system.py
│   │   ├── project_neural_indexer.py
│   │   └── json_utils.py
│   └── neural-system/
│       ├── neural_embeddings.py    # Embedding system
│       ├── bert_tokenizer.py       # Tokenization
│       └── ast_analyzer.py         # Code analysis
└── memory/
    └── session-sticky.md       # Session state
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the error messages carefully - they often indicate missing dependencies
3. Ensure all paths in `.mcp.json` are correct for your system