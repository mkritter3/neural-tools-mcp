# Docker Directory - L9 Enhanced System Only

> **🚀 Production System**: This directory contains **ONLY** the Enhanced L9 system files.  
> All legacy components have been archived to preserve history.

## 📋 Current Files (Enhanced System)

### **Core System**
- **`docker-compose.l9-enhanced.yml`** - Production docker-compose for enhanced system
- **`Dockerfile.l9-enhanced`** - Container definition for enhanced MCP server

### **Application Code**
- **`neural-mcp-server-enhanced.py`** - Enhanced MCP server with GraphRAG + Nomic v2-MoE
- **`nomic_embed_server.py`** - Dedicated Nomic Embed v2-MoE service
- **`tree_sitter_ast.py`** - Multi-language AST analysis engine

### **Configuration**
- **`requirements-l9-enhanced.txt`** - Python dependencies for enhanced system

### **Legacy (Archived)**
- **All legacy files** moved to `/.archive/docker-legacy/` for historical reference

## 🚀 Quick Start

```bash
# Deploy enhanced system
export PROJECT_NAME=my-project
docker-compose -f docker-compose.l9-enhanced.yml up -d

# Check status
docker-compose -f docker-compose.l9-enhanced.yml ps

# View logs
docker-compose -f docker-compose.l9-enhanced.yml logs -f
```

## 🔧 System Architecture

**3-Container Enhanced System:**
1. **MCP Server Enhanced** - Project-specific with GraphRAG + Tree-sitter
2. **Qdrant Vector Database** - Project-isolated with 2025 optimizations  
3. **Nomic Embed v2-MoE** - Shared embedding service with MoE architecture

**Performance Features:**
- Kuzu GraphRAG (3-10x faster than Neo4j)
- RRF hybrid search with MMR diversity
- INT8 quantization for 4x memory reduction
- Nomic v2-MoE (30-40% lower inference costs)

---

**Built for production. Optimized for performance. Zero legacy dependencies.** ✨