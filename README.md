# Neural Flow MCP Server for Claude Code

A high-performance Model Context Protocol (MCP) server that provides neural-powered memory and project search capabilities to Claude Code.

## Features

### 🧠 Neural Memory System
- **Semantic Search**: Query memories using natural language
- **Dynamic Relevance Scoring**: Combines neural similarity with temporal relevance
- **<50ms Query Performance**: Pre-loaded models for instant responses
- **Persistent Storage**: SQLite with WAL mode for concurrent access

### 🔍 Project Intelligence
- **Neural Code Search**: Find code using semantic queries
- **Automatic Indexing**: Indexes your entire codebase with neural embeddings
- **Smart Chunking**: Intelligently splits code for optimal search results
- **Language-Aware**: Understands Python, JavaScript, TypeScript, and more

### ⚡ Performance
- **13.7x Faster**: First query in ~53ms vs ~733ms cold start
- **Background Pre-loading**: Models load while server stays responsive
- **Graceful Fallback**: Automatically falls back if pre-loading fails
- **300MB Memory Footprint**: Optimized for development machines

## Quick Start

### 1. Install Dependencies

```bash
# Automatic installation (recommended)
python install.py

# Or manual installation for your Python version:
# Python 3.8-3.9
pip install -r requirements/requirements-py38.txt

# Python 3.10-3.12
pip install -r requirements/requirements-py310.txt

# Python 3.13+
pip install -r requirements/requirements-py313.txt
```

### 2. Verify Installation

```bash
# Run the test suite
python3 test_mcp_server.py
```

### 3. Restart Claude Code

The neural-flow server will connect automatically and provide 6 tools:
- `memory_query` - Semantic search through memories
- `memory_store` - Store new memories with embeddings
- `memory_stats` - Get system statistics
- `index_project_files` - Index project with neural embeddings
- `search_project_files` - Search code semantically
- `familiarize_with_project` - Comprehensive project analysis

## Architecture

```
claude-l9-template/
├── .mcp.json                    # MCP server configuration
├── install.py                   # Smart installer (auto-detects Python version)
├── test_mcp_server.py          # Comprehensive test suite
├── requirements/               # Version-specific dependencies
│   ├── requirements-base.txt   # Common dependencies
│   ├── requirements-py38.txt   # Python 3.8-3.9
│   ├── requirements-py310.txt  # Python 3.10-3.12
│   └── requirements-py313.txt  # Python 3.13+
├── .claude/
│   ├── mcp-tools/
│   │   ├── mcp_neural_server.py            # Main MCP server
│   │   ├── neural_flow_tools.py            # Core neural tools
│   │   ├── neural_dynamic_memory_system.py # Memory system
│   │   ├── project_neural_indexer.py       # Project indexer
│   │   └── json_utils.py                   # JSON utilities
│   ├── neural-system/
│   │   ├── neural_embeddings.py            # ONNX embeddings
│   │   ├── bert_tokenizer.py               # BERT tokenizer
│   │   └── ast_analyzer.py                 # Code analysis
│   └── onnx_models/
│       └── all-MiniLM-L6-v2.onnx           # 384D neural model
└── memory/
    └── session-sticky.md                   # Session persistence
```

## How It Works

1. **MCP Connection**: Server connects immediately to Claude Code
2. **Background Pre-loading**: Models load in separate thread (~2 seconds)
3. **Smart Routing**: Uses pre-loaded models when ready, falls back otherwise
4. **Neural Processing**:
   - Text → BERT Tokenization → ONNX Embeddings → ChromaDB Vector Search
   - Combined with dynamic scoring for temporal relevance

## Performance Benchmarks

| Operation | Cold Start | Pre-loaded | Improvement |
|-----------|------------|------------|-------------|
| First Query | 733ms | 53ms | 13.7x faster |
| Subsequent | 48ms | 50ms | Consistent |
| Memory Store | 145ms | 66ms | 2.2x faster |
| Project Search | 89ms | 55ms | 1.6x faster |

## Troubleshooting

### Connection Issues
1. Restart Claude Code completely
2. Check `.mcp.json` configuration
3. Run `python3 test_mcp_server.py` to verify

### Dependency Issues
1. Run `python install.py` for automatic setup
2. Check Python version: `python3 --version` (3.8+ required)
3. For NumPy/ChromaDB conflicts, see version-specific requirements

### Performance Issues
1. Ensure ONNX model is downloaded (86.2MB)
2. Check available RAM (needs ~300MB)
3. Verify background pre-loading completes (~2 seconds)

## Support

For detailed setup instructions, see [SETUP.md](SETUP.md)

## License

MIT License - See LICENSE file for details