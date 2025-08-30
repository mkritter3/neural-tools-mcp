# Deprecated Files - L9 Enhanced Transition

> **⚠️ IMPORTANT**: These files are **deprecated** as of the L9 Enhanced 2025 upgrade.  
> They remain in the repository for historical reference but should not be used for new deployments.

---

## 📋 Deprecated Root Files

### **Scripts & Installers**
- **`install.sh`** → Use `docker-compose -f docker-compose.l9-enhanced.yml up -d`
- **`install-global.sh`** → Container-only deployment, no global installation needed
- **`mcp-add`** → MCP configuration now handled via docker-compose

### **Legacy Configuration**
- **`requirements.txt`** → Use `docker/requirements-l9-enhanced.txt`
- **`l9-export.json`** → Enhanced system uses different schema

### **Documentation**
- **`TEMPLATE-REORGANIZATION-PLAN.md`** → Completed, see new architecture docs
- **`TEMPLATE-STRUCTURE.md`** → Replaced by L9-COMPLETE-ARCHITECTURE-2025.md

### **Legacy Docker Files** (in docker/ directory)
- **`docker-compose.yml`** → Use `docker-compose.l9-enhanced.yml`
- **`docker-compose.mcp.yml`** → Use `docker-compose.l9-enhanced.yml`
- **`docker-compose.unified.yml`** → Use `docker-compose.l9-enhanced.yml`
- **`Dockerfile`** → Use `Dockerfile.l9-enhanced`
- **`Dockerfile.mcp`** → Use `Dockerfile.l9-enhanced`
- **`Dockerfile.l9`** → Use `Dockerfile.l9-enhanced`

### **Legacy MCP Servers** (in docker/ directory)
- **`neural-mcp-server.py`** → Use `neural-mcp-server-enhanced.py`
- **`neural-mcp-server-v2.py`** → Use `neural-mcp-server-enhanced.py`
- **`embedding_client.py`** → Built-in to enhanced server
- **`embedding_server.py`** → Use `nomic_embed_server.py`

### **Legacy Requirements** (in docker/ directory)
- **`requirements-mcp.txt`** → Use `requirements-l9-enhanced.txt`

---

## 🔄 Migration Commands

### **Replace Old Commands:**
```bash
# OLD (deprecated)
./install.sh
./install-global.sh
mcp-add

# NEW (enhanced)
cd docker
docker-compose -f docker-compose.l9-enhanced.yml up -d
```

### **Replace Old Docker Commands:**
```bash
# OLD (deprecated)
docker-compose up -d
docker-compose -f docker-compose.mcp.yml up -d  
docker-compose -f docker-compose.unified.yml up -d

# NEW (enhanced)
docker-compose -f docker-compose.l9-enhanced.yml up -d
```

### **Replace Old MCP Config:**
```json
// OLD (deprecated)
{
  "servers": {
    "l9-neural": {
      "command": "docker",
      "args": ["exec", "-i", "mcp-server-default", "python", "neural-mcp-server.py"]
    }
  }
}

// NEW (enhanced)
{
  "servers": {
    "l9-enhanced": {
      "command": "docker", 
      "args": ["exec", "-i", "mcp-server-enhanced-default", "python", "neural-mcp-server-enhanced.py"]
    }
  }
}
```

---

## 🗂️ What to Use Instead

### **For Quick Start:**
- **Read**: `README.md` (completely rewritten for enhanced system)
- **Deploy**: `docker/docker-compose.l9-enhanced.yml`
- **Architecture**: `docs/L9-COMPLETE-ARCHITECTURE-2025.md`

### **For Development:**
- **MCP Server**: `docker/neural-mcp-server-enhanced.py`
- **Embedding**: `docker/nomic_embed_server.py` 
- **Requirements**: `docker/requirements-l9-enhanced.txt`
- **Container**: `docker/Dockerfile.l9-enhanced`

### **For Documentation:**
- **Architecture**: `docs/L9-ENHANCED-ARCHITECTURE.md`
- **Complete Reference**: `docs/L9-COMPLETE-ARCHITECTURE-2025.md`
- **Function Reference**: `docs/L9-FUNCTION-REFERENCE.md`
- **Quick Start**: `docs/QUICK-START.md`

---

## 🚨 Why These Files Are Deprecated

### **Performance Issues**
The deprecated files use **older technologies** that are significantly slower:
- FastEmbed instead of Nomic v2-MoE (30-40% slower embeddings)
- Python ast instead of Tree-sitter (limited language support)
- Basic vector search instead of RRF + MMR hybrid search

### **Architecture Problems**
- **Local dependencies** instead of container-pure architecture
- **Mixed local/container setup** causing deployment complexity
- **No project isolation** leading to data conflicts
- **Resource waste** due to inefficient model sharing

### **Missing Features**
- **No GraphRAG** - missing code relationship intelligence
- **No hybrid search** - basic semantic search only
- **No performance monitoring** - limited observability
- **No production readiness** - no health checks, scaling, etc.

---

## ⏰ Deprecation Timeline

### **Phase 1: Soft Deprecation** (Current)
- Files remain in repository with deprecation notice
- Documentation clearly states enhanced system as preferred
- New users guided to enhanced system only

### **Phase 2: Archive Preparation** (Next Release)
- Move deprecated files to `.deprecated/` directory
- Update all documentation to remove references
- Add warning messages to deprecated files

### **Phase 3: Full Removal** (Future Release)
- Remove deprecated files entirely
- Clean repository structure
- Maintain only enhanced system components

---

## 🔍 How to Identify Deprecated Usage

### **File Usage Check:**
```bash
# Check if you're using deprecated files
grep -r "neural-mcp-server.py" .mcp.json
grep -r "docker-compose.yml" scripts/
grep -r "requirements.txt" docker/

# If any matches found, migrate to enhanced system
```

### **Container Check:**
```bash
# Check running containers
docker ps | grep -E "(mcp-server-|qdrant-|neural-flow-)" | grep -v enhanced

# If any non-enhanced containers running, migrate them
```

### **MCP Config Check:**
```bash
# Check MCP configuration
cat .mcp.json | grep -E "(neural-mcp-server\.py|mcp-server-default)"

# If found, update to enhanced configuration
```

---

## 📞 Support

If you're having trouble migrating from deprecated files:

1. **Check Migration Guide**: See README.md "Migration from Standard L9" section
2. **Review Architecture**: Read `docs/L9-COMPLETE-ARCHITECTURE-2025.md`
3. **Follow Quick Start**: Step-by-step setup in README.md
4. **Parallel Testing**: Both systems can run simultaneously during migration

---

## 📝 Deprecation Log

| File | Deprecated Date | Reason | Replacement |
|------|----------------|--------|-------------|
| `install.sh` | 2025-08-29 | Local setup complexity | Docker compose |
| `neural-mcp-server.py` | 2025-08-29 | Performance/features | `neural-mcp-server-enhanced.py` |
| `docker-compose.yml` | 2025-08-29 | Legacy architecture | `docker-compose.l9-enhanced.yml` |
| `requirements.txt` | 2025-08-29 | Outdated dependencies | `requirements-l9-enhanced.txt` |

---

**✨ The future is enhanced - migrate today for 3-4x better performance!** 🚀