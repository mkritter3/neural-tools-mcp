# ADR-0029: Neo4j Logical Partitioning for Multi-Project Isolation

**Status:** PROPOSED
**Date:** 2025-09-12
**Context:** Neural GraphRAG System, Multi-Project Architecture

## 1. Problem Statement

Following the implementation of a multi-project indexer (ADR-0027), a critical data isolation issue has been identified in our Neo4j graph database. While the Qdrant vector store correctly uses separate collections per project, the Neo4j database stores all data in a single, shared graph without any project-based separation.

This leads to severe problems:

1.  **Data Collisions:** The current schema enforces global uniqueness constraints (e.g., `f.path IS UNIQUE` for `File` nodes). This means if `project-a` has a file at `src/index.js`, `project-b` cannot be indexed if it also has a file at `src/index.js`.
2.  **Data Leakage:** Queries for graph context (e.g., dependencies, related functions) are not filtered by project. A search in `project-a` could incorrectly return nodes and relationships from `project-b`, breaking isolation and providing inaccurate results.
3.  **Inaccurate Analysis:** Any cross-project analysis is impossible, and single-project analysis is unreliable and prone to contamination from other projects' data.
4.  **Scalability Failure:** The current model does not scale beyond a single project and is fundamentally incompatible with our multi-project architecture goals.

## 2. Decision

We will implement a **Logical Partitioning** strategy within our single Neo4j database to enforce strict multi-project data isolation. This is the recommended and most robust approach for the open-source Community Edition of Neo4j.

The core of this decision involves three mandatory changes:

1.  **Project Property Enforcement:** Every project-specific node and relationship in the graph **must** have a `project` property (e.g., `{project: "claude-l9-template"}`).
2.  **Query Scoping:** Every Cypher query that reads, creates, or modifies data **must** be updated to filter by this `project` property.
3.  **Composite Uniqueness Constraints:** Global uniqueness constraints will be replaced with **composite constraints** that enforce uniqueness *within* a project's scope (e.g., on the combination of `(project, path)`).

This approach provides strong data isolation while working within the features available in Neo4j Community Edition and allowing for potential cross-project analysis in the future.

## 3. Architecture Design

### 3.1. Data Model with Project Property

All project-specific nodes will include a `project` property. This property will be indexed to ensure efficient filtering.

**Node Example:**
```json
// A File node for project 'claude-l9-template'
{
  "labels": ["File"],
  "properties": {
    "path": "src/index.js",
    "name": "index.js",
    "project": "claude-l9-template" // <-- Isolation Key
  }
}

// A File node for project 'eventfully-yours'
{
  "labels": ["File"],
  "properties": {
    "path": "src/index.js",
    "name": "index.js",
    "project": "eventfully-yours" // <-- Isolation Key
  }
}
```

### 3.2. Composite Uniqueness Constraints

We will replace all single-property uniqueness constraints with composite constraints that include the `project` property.

**Constraint Examples:**
```cypher
// Ensures a file path is unique *within* a given project
CREATE CONSTRAINT file_project_path_unique IF NOT EXISTS
FOR (f:File) REQUIRE (f.project, f.path) IS UNIQUE;

// Ensures a function signature is unique *within* a given project
CREATE CONSTRAINT function_project_signature_unique IF NOT EXISTS
FOR (fn:Function) REQUIRE (fn.project, fn.signature) IS UNIQUE;
```

### 3.3. Query Scoping Enforcement

All Cypher queries will be modified to use the `$project` parameter, ensuring they only operate on data belonging to the current project context.

**Query Example:**
```cypher
// Find all functions within a specific file *for a given project*
MATCH (f:File {path: $filePath, project: $project})<-[:DEFINED_IN]-(func:Function)
WHERE func.project = $project // Explicitly filter related nodes as well
RETURN func.name, func.signature
```

## 4. Implementation Plan

### Phase 1: Update `indexer_service.py`

The indexer service is responsible for writing data. We must ensure it adds the `project` property to every node and relationship it creates.

**File: `neural-tools/src/servers/services/indexer_service.py`**
```python
# In _index_graph method
# Ensure the MERGE statement for File nodes includes the project property
cypher = """
MERGE (f:File {path: $path, project: $project})
SET f.name = $name, f.type = $type, ...
"""
await self.container.neo4j.execute_cypher(cypher, {
    'path': str(relative_path),
    'project': self.project_name, # This is already being done correctly
    ...
})

# This must be repeated for ALL node and relationship creation queries
# (e.g., for CodeChunk, Function, Class, IMPORTS, etc.)
```

### Phase 2: Update `neo4j_service.py`

This service must be updated to manage constraints and enforce query scoping.

**File: `neural-tools/src/servers/services/neo4j_service.py`**

**1. Update Constraint Management:**
```python
# In _ensure_constraints method
async def _ensure_constraints(self):
    # Drop old, incorrect global constraints (run once during migration)
    # await self.client.execute_query("DROP CONSTRAINT ON (f:File) ASSERT f.path IS UNIQUE")

    # Create new, composite constraints
    constraints = [
        "CREATE CONSTRAINT file_project_path_unique IF NOT EXISTS FOR (f:File) REQUIRE (f.project, f.path) IS UNIQUE",
        "CREATE CONSTRAINT class_project_name_unique IF NOT EXISTS FOR (c:Class) REQUIRE (c.project, c.name, c.file_path) IS UNIQUE",
        "CREATE CONSTRAINT function_project_signature_unique IF NOT EXISTS FOR (m:Method) REQUIRE (m.project, m.name, m.class_name, m.file_path) IS UNIQUE",
        "CREATE INDEX project_property_index IF NOT EXISTS FOR (n) ON (n.project)" // CRITICAL for performance
    ]
    for constraint in constraints:
        await self.client.execute_query(constraint)
```

**2. Enforce Query Scoping:**
Modify `execute_cypher` to automatically inject the `project_name` into all queries.
```python
# In execute_cypher method
async def execute_cypher(self, cypher_query: str, parameters: Optional[Dict] = None) -> Dict[str, Any]:
    if not self.initialized or not self.client:
        return {"status": "error", "message": "Neo4j service not initialized"}

    # Automatically inject the project name for isolation
    if parameters is None:
        parameters = {}
    parameters['project'] = self.project_name

    # ... (rest of the method remains the same)
```

### Phase 3: Update `hybrid_retriever.py` and other read services

All services that read from Neo4j must be updated to include the `project` filter in their `MATCH` and `WHERE` clauses.

**File: `neural-tools/src/servers/services/hybrid_retriever.py`**
```cypher
# Example update in _fetch_graph_context
# BEFORE (INCORRECT):
# MATCH (c:CodeChunk {id: $chunk_id})

# AFTER (CORRECT):
MATCH (c:CodeChunk {id: $chunk_id, project: $project})
OPTIONAL MATCH (c)-[:PART_OF]->(f:File)
WHERE f.project = $project // Also filter the related nodes
OPTIONAL MATCH (f)-[:IMPORTS]->(m:Module)
// Note: Module nodes might be global or project-specific, needs decision
...
```

## 5. Migration Path

1.  **Update Code:** Implement the changes in `indexer_service.py`, `neo4j_service.py`, and `hybrid_retriever.py` as described above.
2.  **Drop Old Constraints:** Manually run the `DROP CONSTRAINT` commands in the Neo4j browser to remove the old global constraints.
3.  **Deploy Updated Code:** Deploy the new version of the `neural-tools` service.
4.  **Create New Constraints:** The `_ensure_constraints` method will automatically create the new composite constraints on startup.
5.  **Data Backfill:** Run a script to iterate through all nodes and relationships in the graph and add the correct `project` property based on the file path. This is a one-time data migration task.
6.  **Re-index (Optional but Recommended):** For a clean slate, delete all existing data and re-index all projects from scratch using the new multi-project indexer.

## 6. Benefits

- **True Data Isolation:** Prevents data collisions and leakage between projects.
- **Correctness:** Ensures that all graph queries return accurate, project-specific results.
- **Scalability:** The architecture can now support any number of projects within the same Neo4j instance.
- **Enables Future Features:** Allows for safe cross-project analysis by explicitly querying for multiple project names.

## 7. Risks and Mitigations

- **Risk:** Forgetting to add a `WHERE n.project = $project` clause in a new query.
  - **Mitigation:** The automatic injection of the `$project` parameter in `neo4j_service.py` reduces this risk. Code reviews must specifically check for this in all new Cypher queries.
- **Risk:** Performance degradation on a very large number of projects.
  - **Mitigation:** Creating an index on the `project` property is critical and is included in the implementation plan. This will ensure that filtering by project is highly efficient.

## 8. References

- **Neo4j Official Documentation on Constraints:** [https://neo4j.com/docs/cypher-manual/current/constraints/](https://neo4j.com/docs/cypher-manual/current/constraints/)
- **Community Guide on Multi-Tenancy:** [https://adamcowley.co.uk/neo4j/multi-tenancy-in-neo4j-a-perfect-match/](https://adamcowley.co.uk/neo4j/multi-tenancy-in-neo4j-a-perfect-match/)
- **Stack Overflow Discussion on Logical Partitioning:** [https://stackoverflow.com/questions/60985759/neo4j-multitenancy-database-design](https://stackoverflow.com/questions/60985759/neo4j-multitenancy-database-design)

## 9. Decision Outcome

**Approved.** We will implement the Logical Partitioning strategy to ensure robust multi-project data isolation in our Neo4j database. This is a critical fix that aligns our graph database with the multi-project architecture of the wider system.
