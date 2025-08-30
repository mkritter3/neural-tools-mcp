# 🚀 L9 Neural Flow - Quick Start Guide

Get up and running with L9 Neural Flow in 5 minutes.

## Prerequisites

- **Python 3.8+** (`python3 --version`)
- **Docker** (`docker --version`)
- **Claude Code** installed

## Installation

### Option 1: Clone Template (Recommended)

```bash
# Clone the template
git clone https://github.com/yourusername/l9-neural-template.git
cd l9-neural-template

# Run installer
./install.sh
```

### Option 2: Download Release

```bash
# Download latest release
curl -L https://github.com/yourusername/l9-neural-template/releases/latest/download/l9-neural.tar.gz | tar xz
cd l9-neural-template

# Run installer
./install.sh
```

## Adding L9 to Your Project

### Basic Setup (Current Project)

```bash
# Navigate to your project
cd /path/to/your-project

# Add L9 neural memory
mcp add
```

This creates:
- `.mcp.json` - MCP server configuration
- `.claude/` - Project-specific settings
- Docker container for your project's memory

### Global Setup (All Projects)

```bash
# Add L9 globally
mcp add --scope user
```

Now L9 is available in ALL your projects automatically.

### Team Setup (Shared Configuration)

```bash
# Add with project scope (commits to git)
mcp add --scope project

# Commit the configuration
git add .mcp.json
git commit -m "Add L9 neural memory for team"
git push
```

Team members just need to:
```bash
git pull
# L9 automatically activates via .mcp.json
```

## Using L9 Neural Memory

### Memory Operations

In Claude Code, try these commands:

```
# Store a memory
"Remember that we decided to use PostgreSQL for the main database"

# Recall memories
"What database did we decide to use?"
"What have we discussed about authentication?"

# Search code semantically
"Find all error handling logic"
"Show me database connection code"
"Where do we validate user input?"
```

### Project Context

L9 automatically:
- Indexes your code when files change
- Remembers conversation context
- Learns your coding patterns
- Maintains project-specific knowledge

## Verification

### Check Installation

```bash
# Verify MCP server is configured
cat .mcp.json

# Check Docker container
docker ps | grep l9-neural

# Test memory system
claude
> "Store test memory: L9 is working!"
> "Recall: What did I just store?"
```

### Troubleshooting

#### Python Dependencies

```bash
# If you see import errors
pip3 install qdrant-client chromadb sentence-transformers tiktoken
```

#### Docker Issues

```bash
# Start Docker if not running
sudo systemctl start docker  # Linux
open -a Docker              # macOS

# Check container logs
docker logs l9-neural-[project-name]
```

#### MCP Connection

```bash
# In Claude Code
/mcp  # Shows MCP server status

# Restart Claude Code if needed
```

## Configuration Options

### Custom Installation

```bash
# Without automation hooks
mcp add --no-hooks

# Without auto-indexing
mcp add --no-auto-indexing

# Minimal setup
mcp add --no-hooks --no-auto-indexing
```

### Environment Variables

Set in `.claude/settings.json`:

```json
{
  "env": {
    "USE_QODO_EMBED": "true",         // Code-specific embeddings
    "ENABLE_AB_TESTING": "false",     // A/B testing models
    "L9_PROTECTION_LEVEL": "maximum"  // Security level
  }
}
```

## What's Next?

- **Explore Features**: See [Architecture](ARCHITECTURE.md) for system design
- **API Reference**: Check [Function Reference](L9-FUNCTION-REFERENCE.md)
- **Examples**: Browse `examples/` directory
- **Templates**: Use `templates/` for new projects

## Common Use Cases

### 1. Remember Decisions
```
"Remember: We're using JWT for auth with 24-hour expiry"
"Remember: API rate limit is 100 requests per minute"
"What decisions have we made about authentication?"
```

### 2. Code Understanding
```
"How does the authentication flow work?"
"Find all places where we handle user permissions"
"What's the relationship between User and Profile models?"
```

### 3. Project Knowledge
```
"What's our testing strategy?"
"Which libraries are we using for state management?"
"What were the performance requirements we discussed?"
```

## Performance Tips

- **First Run**: Initial indexing may take 30-60 seconds for large projects
- **Memory Usage**: ~200MB RAM for embeddings, ~100MB for Qdrant
- **Response Time**: <100ms for memory recall, <500ms for semantic search

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/l9-neural-template/issues)
- **Docs**: Full documentation in `docs/`
- **Examples**: Working examples in `examples/`

---

**Ready to go!** Your project now has L9-grade neural memory. 🎉