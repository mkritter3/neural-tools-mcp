# Migration Guide: L9 to Global Neural Memory System

## Quick Start Migration

For most users, the automatic migration script is the easiest option:

```bash
# 1. Backup your current system (recommended)
python3 migrate-to-global.py --backup-only

# 2. Run automatic migration
python3 migrate-to-global.py --auto

# 3. Verify the migration
python3 migrate-to-global.py --verify

# 4. Start using the new system
# Your Claude Code setup will automatically use the new global MCP server
```

**Migration time:** ~5-15 minutes depending on data size

## What's Changing

### Before (L9 System)
```
Project Structure:
├── .mcp.json (per-project MCP server)
├── .claude/neural-system/
│   ├── l9_config_manager.py
│   ├── l9_project_isolation.py
│   ├── mcp_l9_launcher.py
│   └── l9_qdrant_memory_v2.py
└── Docker containers per project
```

### After (Global System)
```
Project Structure:
├── .mcp.json (global MCP server)
├── .claude/neural-system/
│   ├── config_manager.py
│   ├── project_isolation.py
│   ├── memory_system.py (wrapper)
│   └── l9_qdrant_memory_v2.py (preserved)
├── neural-memory-mcp.py (global server)
└── Shared model server + per-project Qdrant
```

## Key Benefits After Migration

- **90%+ Memory Savings**: Shared model server instead of per-project models
- **Cross-Project Search**: Query memories across all your projects when needed
- **Zero Configuration**: Works automatically with any project
- **Better Performance**: <50ms response times with shared resources
- **Claude Code Integration**: Native MCP tools in Claude Code

## Step-by-Step Migration

### Step 1: Preparation

1. **Check Current System Status**
   ```bash
   ./neural-docker.sh status
   python3 .claude/neural-system/l9_config_manager.py
   ```

2. **Create Backup** (Essential!)
   ```bash
   python3 migrate-to-global.py --backup-only
   # Creates timestamped backup in ./backups/
   ```

### Step 2: Migration Options

Choose one migration approach:

#### Option A: Automatic Migration (Recommended)
```bash
# Full automatic migration with verification
python3 migrate-to-global.py --auto --verify
```

#### Option B: Step-by-Step Migration
```bash
# 1. Export existing data
python3 migrate-to-global.py --export-data

# 2. Deploy new system
python3 migrate-to-global.py --deploy-system

# 3. Import data to new system  
python3 migrate-to-global.py --import-data

# 4. Verify migration
python3 migrate-to-global.py --verify
```

#### Option C: Manual Migration
See [Manual Migration](#manual-migration) section below.

### Step 3: Verification

After migration, verify everything is working:

```bash
# Run comprehensive verification
python3 migrate-to-global.py --verify --detailed

# Test Claude Code integration
# Open Claude Code and try these commands:
# - Ask Claude to store something in memory
# - Ask Claude to recall recent project information
# - Test cross-project search if you have multiple projects
```

### Step 4: Clean Up (Optional)

Once you're satisfied with the migration:

```bash
# Remove legacy L9 containers (saves disk space)
python3 migrate-to-global.py --cleanup-legacy

# Archive backup (optional)
python3 migrate-to-global.py --archive-backup
```

## Manual Migration

If you prefer full control over the migration process:

### 1. Stop Current L9 System
```bash
./neural-docker.sh stop
docker container stop $(docker ps -q --filter "label=l9-system")
```

### 2. Export L9 Data
```bash
# Export memory data
python3 -c "
import sys
sys.path.append('.claude/neural-system')
from l9_qdrant_memory_v2 import L9QdrantMemoryV2
import asyncio
import json

async def export_data():
    memory = L9QdrantMemoryV2()
    await memory.initialize()
    
    # Export all project memories
    memories = await memory.search_project_memories('', limit=10000, include_other_projects=True)
    
    export_data = {
        'memories': [m.__dict__ for m in memories],
        'project': memory.project_name,
        'exported_at': '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
    }
    
    with open('l9-export.json', 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f'Exported {len(memories)} memories to l9-export.json')

asyncio.run(export_data())
"
```

### 3. Start New Global System
```bash
# Start unified docker system
./neural-docker.sh start

# Verify new system is running
./neural-docker.sh status
curl http://localhost:8080/health  # Check model server
```

### 4. Import Data to New System
```bash
# Import the exported data
python3 -c "
import sys
sys.path.append('.claude/neural-system')
from memory_system import MemorySystem
import asyncio
import json

async def import_data():
    with open('l9-export.json', 'r') as f:
        export_data = json.load(f)
    
    memory = MemorySystem()
    await memory.initialize()
    
    imported_count = 0
    for memory_data in export_data['memories']:
        try:
            await memory.store_memory(
                content=memory_data['content'],
                metadata={
                    'migrated_from': 'l9',
                    'original_timestamp': memory_data['timestamp'],
                    'original_project': memory_data['project']
                }
            )
            imported_count += 1
        except Exception as e:
            print(f'Failed to import memory: {e}')
    
    print(f'Successfully imported {imported_count} memories')

asyncio.run(import_data())
"
```

## Troubleshooting

### Common Issues

#### Migration Script Fails
```bash
# Check logs
python3 migrate-to-global.py --debug --log-file=migration.log

# Check system status
./neural-docker.sh status
docker ps -a | grep neural
```

#### Data Missing After Migration
```bash
# Restore from backup
python3 migrate-to-global.py --restore-backup --backup-name=YYYY-MM-DD_HH-MM-SS

# Re-run data import only
python3 migrate-to-global.py --import-data --force
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Restart shared model server
docker restart shared-model-server

# Check model server health
curl http://localhost:8080/health
```

#### Claude Code Not Finding MCP Server
```bash
# Check MCP configuration
cat .mcp.json

# Test MCP server directly
python3 neural-memory-mcp.py --test

# Restart Claude Code to reload MCP configuration
```

### Getting Help

1. **Check Migration Logs**
   ```bash
   cat migration.log
   ./neural-docker.sh logs
   ```

2. **Run Health Checks**
   ```bash
   python3 migrate-to-global.py --health-check
   ```

3. **Emergency Rollback**
   ```bash
   python3 migrate-to-global.py --rollback --backup-name=LATEST
   ```

## Post-Migration Validation

### Test Neural Memory Tools

Open Claude Code and try these commands:

1. **Test Memory Storage**
   > "Remember that we use PostgreSQL for our main database and Redis for caching"

2. **Test Memory Recall**
   > "What database systems do we use in this project?"

3. **Test Code Search**
   > "Find any functions related to user authentication"

4. **Test Project Context**
   > "What's the current project structure and memory status?"

### Expected Results

- ✅ Memory storage should work without errors
- ✅ Memory recall should find relevant information
- ✅ Code search should return relevant code snippets  
- ✅ Project context should show current project details
- ✅ Response times should be <100ms for typical queries

## Configuration Reference

### New .mcp.json Format
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python3",
      "args": ["/path/to/neural-memory-mcp.py"],
      "env": {
        "PYTHONPATH": "/path/to/.claude/neural-system"
      }
    }
  }
}
```

### Global System Environment Variables
```bash
# Model server configuration
MODEL_SERVER_HOST=localhost
MODEL_SERVER_PORT=8080

# Project configuration (auto-detected)
PROJECT_NAME=auto-detected
QDRANT_REST_PORT=auto-assigned
QDRANT_GRPC_PORT=auto-assigned

# Performance tuning
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
```

## Migration Checklist

Before declaring migration complete:

- [ ] All L9 data exported successfully
- [ ] New global system deployed and running
- [ ] Data imported to new system without errors
- [ ] Claude Code integration working (MCP tools available)
- [ ] Memory operations working (store, recall, search)
- [ ] Performance metrics acceptable (<100ms response times)
- [ ] Cross-project search working (if applicable)
- [ ] Original L9 backup created and verified
- [ ] Documentation updated for team members
- [ ] Legacy L9 system cleanly shut down

## Rollback Procedure

If you need to rollback to the L9 system:

```bash
# Quick rollback to last working state
python3 migrate-to-global.py --rollback

# Or restore from specific backup
python3 migrate-to-global.py --restore-backup --backup-name=2025-01-15_14-30-00

# Verify L9 system is working
./neural-docker.sh status
python3 .claude/neural-system/l9_config_manager.py
```

---

*This migration guide ensures a smooth transition to the global neural memory system. If you encounter any issues, refer to the troubleshooting section or create a backup restore point.*