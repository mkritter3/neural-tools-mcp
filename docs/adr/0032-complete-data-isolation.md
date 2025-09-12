# ADR-0032: Complete Data Isolation for Multi-Project MCP Tools

**Status:** Accepted  
**Date:** September 12, 2025  
**Author:** L9 Engineering Team

## Context

During a data isolation audit, we discovered that several MCP tools are leaking cross-project data:

1. **`neural_system_status`** shows aggregate counts from ALL projects:
   - Neo4j node count includes all projects (12,092 total nodes)
   - Qdrant collections list shows all project collections
   - This misleads users about their specific project's indexing status

2. **`project_understanding`** returns static dummy data instead of real project information

3. **Status queries don't filter by project**, making it impossible to determine:
   - How many nodes belong to the current project
   - Whether the current project is actually indexed
   - What collections exist for the current project

This violates the multi-project isolation principle established in ADR-0029 and creates confusion when multiple projects share the same infrastructure.

## Problem Statement

When a Claude instance queries the system status in a specific project directory, it receives:
- Aggregate data from ALL projects
- No way to distinguish what belongs to the current project
- False positives about indexing status

Example of the problem:
```
Project A: 100 nodes indexed
Project B: 11,992 nodes indexed
---
User in Project A sees: "12,092 nodes" and thinks Project A is huge
User in Project C sees: "12,092 nodes" and thinks Project C is indexed (when it's not)
```

## Decision

We will implement **complete data isolation** across ALL MCP tools by:

1. **Adding project filtering to all database queries**
2. **Showing only project-specific collections and counts**
3. **Implementing real project understanding with actual data**
4. **Maintaining backward compatibility with proper defaults**

## Implementation Details

### 1. Fix `neural_system_status`

**Current (LEAKING):**
```python
# Shows ALL nodes across ALL projects
result = await container.neo4j.execute_query(
    "MATCH (n) RETURN count(n) as node_count"
)

# Shows ALL collections
collections = await container.qdrant.get_collections()
```

**Fixed (ISOLATED):**
```python
# Shows only nodes for current project
result = await container.neo4j.execute_query(
    "MATCH (n {project: $project}) RETURN count(n) as node_count",
    {"project": project_name}
)

# Shows only collections for current project
all_collections = await container.qdrant.get_collections()
project_collections = [c for c in all_collections if c.startswith(f"project_{project_name}_")]
```

### 2. Implement Real `project_understanding`

**Current (FAKE):**
```python
async def project_understanding_impl(scope: str = "full"):
    # Just returns static data
    return {"project": project_name, "scope": scope}
```

**Fixed (REAL):**
```python
async def project_understanding_impl(scope: str = "full"):
    # Query actual project data
    if scope == "full":
        file_count = await neo4j.query("MATCH (f:File {project: $project}) RETURN count(f)")
        class_count = await neo4j.query("MATCH (c:Class {project: $project}) RETURN count(c)")
        method_count = await neo4j.query("MATCH (m:Method {project: $project}) RETURN count(m)")
        vector_count = await qdrant.count_vectors(f"project_{project_name}_code")
        
        return {
            "project": project_name,
            "statistics": {
                "files": file_count,
                "classes": class_count,
                "methods": method_count,
                "vectors": vector_count
            },
            "indexed": file_count > 0,
            "last_indexed": last_index_time
        }
```

### 3. Add Project Context to All Status Tools

Every tool that queries data must:
1. Get the current project context
2. Filter all queries by project
3. Show only project-specific results

Tools requiring fixes:
- ✅ `semantic_code_search` - Already isolated
- ✅ `graphrag_hybrid_search` - Already isolated
- ❌ `neural_system_status` - Needs fixing
- ❌ `project_understanding` - Needs implementation
- ❌ `indexer_status` - Needs verification
- ✅ `canon_understanding` - Already isolated
- ✅ `backfill_metadata` - Already isolated

## Consequences

### Positive
- **True multi-project isolation** - Each project sees only its own data
- **Accurate status reporting** - Users know exactly what's indexed
- **No data leaks** - Projects remain completely separate
- **Better debugging** - Clear visibility into project-specific state

### Negative
- **Breaking change** for tools expecting aggregate data
- **Slightly more complex queries** with project filtering
- **Small performance overhead** from filtering

### Neutral
- Status tools will show smaller numbers (project-specific vs global)
- Some monitoring use cases may need a new "global_status" tool

## Migration Path

1. **Phase 1**: Add project filtering to all queries (backward compatible)
2. **Phase 2**: Update tool descriptions to clarify project scope
3. **Phase 3**: Add optional `show_all_projects` flag for admin use

## Success Criteria

- [ ] Each project sees only its own node count
- [ ] Each project sees only its own collections
- [ ] `project_understanding` returns real data
- [ ] No tool shows data from other projects
- [ ] All queries include project filtering

## References

- ADR-0029: Neo4j Multi-Project Isolation
- ADR-0031: Canonical Knowledge Management
- Issue: Cross-project data leakage in status tools

## Code Examples

### Proper Project Filtering Pattern

```python
# ALWAYS filter by project in Neo4j
async def get_project_files(project_name: str):
    query = """
    MATCH (f:File {project: $project})
    RETURN f.path, f.content_hash
    """
    return await neo4j.execute_query(query, {"project": project_name})

# ALWAYS use project-specific collections in Qdrant  
async def search_project_vectors(project_name: str, vector: List[float]):
    collection_name = f"project_{project_name}_code"
    return await qdrant.search(collection_name, vector)

# ALWAYS filter collection lists
async def get_project_collections(project_name: str):
    all_collections = await qdrant.get_collections()
    prefix = f"project_{project_name}_"
    return [c for c in all_collections if c.startswith(prefix)]
```

## Decision Outcome

**Accepted** - We must fix these data isolation issues immediately to maintain the integrity of the multi-project system. The current state creates confusion and false positives about indexing status.