# L9 Template Reorganization Plan

## Goal
Transform the L9 Neural Flow into a clean, copy-paste template while preserving all functionality and deprecated files for reference.

## Current State Analysis

### ✅ Core Working System (KEEP IN MAIN)
```
/
├── neural-memory-mcp.py          # Main MCP server - ACTIVE
├── mcp-add                        # MCP add command - ACTIVE
├── install-global.sh              # Global installer - ACTIVE
├── .mcp.json                      # MCP configuration - ACTIVE
├── .claude/
│   ├── neural-system/             # Core neural system - ACTIVE
│   │   ├── config_manager.py
│   │   ├── memory_system.py
│   │   ├── neural_embeddings.py
│   │   ├── project_isolation.py
│   │   ├── feature_flags.py
│   │   ├── feature_flags.json
│   │   ├── shared_model_client.py
│   │   └── bert_tokenizer.py
│   └── settings.json              # Claude settings - ACTIVE (hooks disabled)
├── scripts/
│   ├── neural-flow.sh             # Container management - ACTIVE
│   └── test-neural-flow.sh        # Testing - ACTIVE
└── docs/
    ├── L9-FUNCTION-REFERENCE.md   # New comprehensive docs - ACTIVE
    └── adr/
        └── 0004-*.md               # Architecture decisions - ACTIVE
```

### 🔄 Files to Deprecate (MOVE TO .deprecated/)
```
.deprecated/
├── initial-setup/                 # Old setup methods
│   ├── neural-setup.py            # Replaced by mcp-add
│   ├── scripts/neural-init        # Old bash init
│   ├── migrate-to-global.py
│   └── legacy/SETUP.md
├── hooks/                         # Currently broken hooks
│   ├── neural_auto_indexer.py
│   ├── context_injector.py
│   ├── project_context_loader.py
│   ├── session_memory_store.py
│   ├── safety_checker.py
│   └── style_preserver.py
├── mock-implementations/          # Test/mock files
│   ├── neural_dynamic_memory_system.py
│   ├── project_neural_indexer.py
│   └── mcp_neural_server.py (old mock)
├── docker-variants/               # Experimental Docker files
│   ├── Dockerfile.l9-minimal
│   ├── Dockerfile.l9-complete
│   ├── Dockerfile.test
│   └── Dockerfile.l9-production
├── experimental/                  # Experimental features
│   ├── shared-model-server.py
│   ├── neural-docker.sh
│   ├── start-neural-flow.sh
│   └── benchmark-performance.py
├── legacy-neural-system/          # Old neural system files
│   └── (contents of .claude/neural-system/legacy/)
└── old-docs/                      # Superseded documentation
    ├── DEPRECATION-STRATEGY.md
    ├── MIGRATION-GUIDE.md
    ├── MIGRATION-SUMMARY.md
    └── L9-SETUP.md
```

## New Clean Template Structure

```
l9-neural-template/
├── README.md                      # Quick start guide
├── install.sh                     # One-command installer
├── mcp-add                        # MCP add command
├── .mcp.json                      # MCP server config
│
├── core/                          # Core MCP server
│   ├── neural-memory-mcp.py      # Main MCP server
│   └── requirements.txt          # Python dependencies
│
├── neural-system/                 # Neural intelligence
│   ├── config_manager.py         # Configuration
│   ├── memory_system.py          # Memory wrapper
│   ├── neural_embeddings.py      # Embeddings
│   ├── project_isolation.py      # Docker management
│   ├── feature_flags.py          # Feature management
│   ├── feature_flags.json        # Feature config
│   ├── shared_model_client.py    # Model client
│   └── bert_tokenizer.py         # Tokenizer
│
├── scripts/                       # Management scripts
│   ├── neural-flow.sh            # Container management
│   └── test.sh                   # System testing
│
├── docker/                        # Docker configuration
│   ├── Dockerfile.l9             # Main L9 container
│   ├── docker-compose.yml        # Orchestration
│   └── model-server/             # Model server setup
│       └── Dockerfile
│
├── templates/                     # Project templates
│   ├── python/                   # Python template
│   ├── javascript/               # JS template
│   └── react/                    # React template
│
├── docs/                          # Documentation
│   ├── QUICK-START.md           # 5-minute guide
│   ├── ARCHITECTURE.md          # System architecture
│   ├── FUNCTION-REFERENCE.md    # Complete reference
│   └── adr/                      # Architecture decisions
│
├── examples/                      # Usage examples
│   ├── basic-setup/              # Basic project setup
│   ├── multi-project/            # Multi-project setup
│   └── advanced/                 # Advanced configurations
│
├── .deprecated/                   # Preserved deprecated files
│   └── [organized deprecated files as above]
│
└── .claude/                       # Claude-specific config
    └── settings.json              # Settings (hooks disabled)
```

## Benefits of New Structure

1. **Clean Root**: Only essential files at root level
2. **Logical Organization**: Core, neural-system, scripts clearly separated
3. **Easy Template Use**: Copy entire folder, run install.sh
4. **Preserved History**: All deprecated files in .deprecated for reference
5. **Clear Documentation**: Focused docs in main, old docs preserved
6. **Template Ready**: Project templates included for quick starts

## Migration Steps

1. Create new directory structure
2. Move deprecated files to .deprecated/
3. Reorganize active files into clean structure
4. Update all import paths and references
5. Create new simplified README.md
6. Test the clean template
7. Create install.sh for one-command setup

## Usage After Reorganization

```bash
# Copy template to new location
cp -r l9-neural-template/ my-project/

# Install and setup
cd my-project
./install.sh

# Add to any project
mcp add /path/to/project

# Or use globally
mcp add --scope user
```

## Files to Create

1. **install.sh** - Simplified installer replacing install-global.sh
2. **README.md** - Clean, focused quick start
3. **docs/QUICK-START.md** - 5-minute setup guide
4. **docs/ARCHITECTURE.md** - High-level architecture
5. **templates/** - Ready-to-use project templates

## Testing Checklist

- [ ] Clean template copies successfully
- [ ] install.sh sets up everything needed
- [ ] mcp add command works
- [ ] Neural system initializes properly
- [ ] Docker containers start correctly
- [ ] Memory operations work
- [ ] No references to deprecated paths
- [ ] Documentation is clear and complete