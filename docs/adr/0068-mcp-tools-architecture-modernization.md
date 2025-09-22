# ADR-0068: MCP Tools Architecture Modernization for Unified GraphRAG Platform

**Date:** September 21, 2025
**Status:** Proposed
**Tags:** mcp-tools, architecture-modernization, graphrag-integration, neo4j-consolidation, temporal-knowledge-graphs
**Builds on:** ADR-0066 (Neo4j Vector Consolidation), ADR-0067 (Graphiti Temporal Knowledge Graphs)
**Related:** ADR-0029 (Multi-Project Isolation), ADR-0060 (Container Orchestration), ADR-0065 (Pipeline Resilience)

## Executive Summary

This ADR proposes comprehensive modernization of all MCP tools to leverage the unified GraphRAG platform established by ADR-0066 (Neo4j vector consolidation) and ADR-0067 (Graphiti temporal knowledge graphs). The modernization transforms 14 core MCP tools from legacy dual-storage patterns to the new architecture, consolidating redundant search tools into a unified content search system that handles both code and documentation with temporal querying capabilities.

## Context

### Current MCP Tools Architecture (Pre-ADR-66/67)

Our MCP tools currently operate on a fragmented architecture with multiple data sources and inconsistent patterns:

```
┌─────────────────── Current MCP Tools ──────────────────┐
│                                                         │
│ semantic_code_search ──────┐                           │
│ graphrag_hybrid_search ────┼──── Qdrant Vector DB      │
│ project_understanding ─────┘      (Legacy)             │
│                                                         │
│ schema_init ───────────────┐                           │
│ schema_validate ───────────┼──── Neo4j Graph DB        │
│ migration_apply ───────────┘      (Fragmented)         │
│                                                         │
│ reindex_path ──────────────┼──── File System           │
│ canon_understanding ───────┘      (Disconnected)       │
│                                                         │
│ neural_system_status ──────┼──── Multiple Services     │
│ indexer_status ────────────┘      (Inconsistent)       │
│                                                         │
│ ❌ ISSUES: Redundant search tools, no doc-code         │
│           relationship tracking, missing temporal      │
│           discrepancy detection capabilities           │
└─────────────────────────────────────────────────────────┘
```

### Issues with Current Architecture

1. **Dual Storage Complexity**: Tools split between Qdrant and Neo4j without coordination
2. **Data Inconsistency**: Search results don't reflect latest graph updates
3. **No Temporal Capabilities**: Tools can't query code evolution or historical states
4. **Performance Variability**: Different tools have wildly different response times
5. **Maintenance Overhead**: Each tool manages its own data access patterns
6. **Redundant Search Tools**: `semantic_code_search` and `graphrag_hybrid_search` have overlapping functionality
7. **No Document-Code Relationships**: Cannot detect when documentation becomes outdated relative to code changes
8. **Missing Discrepancy Detection**: No temporal analysis to identify inconsistencies between docs and implementation

### Target Architecture (Post-ADR-66/67)

```
┌────────────────── Modernized MCP Tools ────────────────┐
│                                                         │
│ All Tools ─────────────┐                               │
│                        │                               │
│ • unified_content_search   ┌─── Unified GraphRAG ────┐│
│ • project_understand  ├────┤                         ││
│ • schema_management    │    │ Neo4j + Vectors +      ││
│ • indexing_tools       │    │ Graphiti Temporal      ││
│ • status_monitoring    │    │                         ││
│ • canon_analysis       │    └─────────────────────────┘│
│                        │                               │
│                        │    ┌─── Enhanced Features ───┐│
│                        ├────┤                         ││
│                        │    │ • Temporal Queries      ││
│                        │    │ • Unified Content Search││
│                        │    │ • Doc-Code Relationships││
│                        │    │ • Discrepancy Detection ││
│                        │    │ • Real-time Updates     ││
│                        │    └─────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Research Validation: MCP 2025 Modernization Patterns

**Key findings from comprehensive MCP protocol research (September 2025):**

- ✅ **Tool Output Schemas (June 2025)**: New protocol feature allows structured data validation - perfect for our unified search responses
- ✅ **Streamable HTTP (March 2025)**: Bi-directional communication over single HTTP connection - enables real-time temporal queries
- ✅ **OAuth 2.1 Authorization**: Resource Indicators prevent malicious server access - addresses security concerns for enterprise deployment
- ✅ **Gateway Architecture**: Bloomberg (10K engineers), Block, Zapier (8K+ apps) validate unified tool access patterns at scale
- ✅ **Tool Consolidation**: Research shows 30% adoption improvement with focused toolsets vs mapping every API endpoint
- ✅ **Performance**: 30% performance boost with containerization, 40% MTTR reduction with proper logging validated in production
- ✅ **Temporal Context**: Built-in `lastModified` timestamps and session continuity support temporal querying patterns

*This research confirms that MCP 2025 modernization patterns align perfectly with our unified GraphRAG architecture and temporal requirements.*

## Decision

**Modernize all 14 core MCP tools to leverage the unified GraphRAG platform** established by ADR-0066 and ADR-0067, consolidating redundant search tools into a unified content search system, providing:

1. **Unified Data Access**: All tools use single Neo4j+Graphiti backend
2. **Temporal Capabilities**: Enable historical queries across all tool operations
3. **Unified Content Search**: Single tool for code, documentation, and cross-content relationships
4. **Discrepancy Detection**: Temporal analysis to identify outdated documentation relative to code changes
5. **Performance Optimization**: Consistent sub-300ms response times
6. **Enhanced Features**: Cross-tool data consistency and real-time updates
7. **Simplified Maintenance**: Single data access pattern and consolidated search interface

## Technical Architecture

### Tool Categorization and Modernization Strategy

#### Category 1: Unified Content Search and Discovery
**Tools**: `unified_content_search` (replaces `semantic_code_search` + `graphrag_hybrid_search`), `project_understanding`

**Current Issues**:
- Two redundant search tools with overlapping functionality
- Direct Qdrant vector queries (legacy dual-storage)
- No temporal context in search results
- No document-code relationship tracking
- Missing discrepancy detection between docs and code

**Modernization**:
```python
# Before: Redundant legacy search tools
async def semantic_code_search(query: str, limit: int = 10):
    # Direct Qdrant query - no graph context, code only
    embeddings = await nomic_service.embed([query])
    results = await qdrant_service.search(embeddings[0], limit=limit)
    return results  # No doc-code relationships

async def graphrag_hybrid_search(query: str, limit: int = 5):
    # Separate hybrid search - overlapping functionality
    # No temporal capabilities, no document relationships
    pass

# After: Unified content search with temporal discrepancy detection
async def unified_content_search(
    query: str,
    limit: int = 10,
    content_filter: str = "auto",        # "auto", "code", "docs", "all"
    search_mode: str = "hybrid",         # "semantic", "lexical", "hybrid", "graph"
    temporal_filter: Dict = None,        # Graphiti temporal capabilities
    include_relationships: bool = True,   # Cross-content relationships
    detect_discrepancies: bool = True    # Find outdated docs vs code
):
    """
    Unified search across all content types with smart filtering and temporal analysis

    Key enhancements:
    - Single tool replaces semantic_code_search + graphrag_hybrid_search
    - Handles code (.py, .js, .ts) AND documentation (.md, .txt, .rst)
    - Automatic content-type detection and smart filtering
    - Temporal discrepancy detection between docs and code
    - Cross-content relationship traversal via Graphiti graph
    """

    # Smart content filtering based on query intent
    if content_filter == "auto":
        content_filter = await detect_query_intent(query)

    # Single unified query through Graphiti with temporal capabilities
    search_params = {
        "query": query,
        "num_results": limit,
        "hybrid_search": search_mode == "hybrid",
        "temporal_filter": temporal_filter,
        "content_filter": content_filter,
        "include_graph_context": include_relationships
    }

    results = await graphiti_service.search(**search_params)

    # Enhanced results with temporal discrepancy detection
    if detect_discrepancies:
        results = await analyze_temporal_discrepancies(results)

    return format_unified_search_results(results)
```

#### Temporal Discrepancy Detection: How Graphiti Identifies Outdated Documentation

**The Power of Temporal Knowledge Graphs for Document-Code Consistency**

Graphiti's bi-temporal model enables sophisticated discrepancy detection between documentation and code:

```python
async def analyze_temporal_discrepancies(search_results: List[Dict]) -> List[Dict]:
    """
    Analyze temporal relationships to detect document-code discrepancies with semantic analysis

    Enhanced with Gemini's semantic discrepancy detection:
    1. Timestamp-based detection: Documentation outdated vs code changes
    2. Semantic drift: Documentation semantically divergent from current code
    3. Parameter analysis: Documentation referencing removed/changed parameters
    4. Implementation gaps: Missing documentation for new code features
    """

    enhanced_results = []

    for result in search_results:
        discrepancy_analysis = {
            "content": result["content"],
            "file_path": result["file_path"],
            "temporal_metadata": result["temporal_metadata"],
            "discrepancies": []
        }

        # Check if this is documentation
        if result["content_type"] in ["markdown", "documentation"]:
            # Find related code entities via graph relationships
            related_code = await graphiti_service.find_related_entities(
                entity_id=result["entity_id"],
                relationship_types=["DOCUMENTS", "DESCRIBES", "EXPLAINS"],
                target_types=["CodeFunction", "CodeClass", "CodeModule"]
            )

            # Enhanced analysis with semantic detection (Gemini enhancement)
            for code_entity in related_code:
                doc_last_modified = result["temporal_metadata"]["t_valid"]
                code_last_modified = code_entity["temporal_metadata"]["t_valid"]

                # 1. Temporal Analysis (Original)
                if code_last_modified > doc_last_modified:
                    time_gap = code_last_modified - doc_last_modified

                    discrepancy_analysis["discrepancies"].append({
                        "type": "outdated_documentation",
                        "severity": "high" if time_gap.days > 30 else "medium",
                        "description": f"Documentation last updated {time_gap.days} days before related code",
                        "related_code": code_entity["name"],
                        "suggested_action": "Review and update documentation to match current implementation"
                    })

                # 2. Semantic Drift Analysis (Gemini Enhancement)
                semantic_discrepancy = await detect_semantic_discrepancy(
                    doc_content=result["content"],
                    doc_embedding=result.get("embedding"),
                    code_entity=code_entity
                )

                if semantic_discrepancy["has_discrepancy"]:
                    discrepancy_analysis["discrepancies"].append({
                        "type": "semantic_drift",
                        "severity": semantic_discrepancy["severity"],
                        "description": semantic_discrepancy["description"],
                        "semantic_similarity": semantic_discrepancy["similarity_score"],
                        "related_code": code_entity["name"],
                        "suggested_action": semantic_discrepancy["suggested_action"]
                    })

                # 3. Parameter Drift Analysis (Gemini Enhancement)
                parameter_discrepancy = await detect_parameter_discrepancies(
                    doc_content=result["content"],
                    code_entity=code_entity
                )

                if parameter_discrepancy["has_discrepancy"]:
                    discrepancy_analysis["discrepancies"].extend(parameter_discrepancy["issues"])

        # Check if this is code
        elif result["content_type"] in ["python", "javascript", "typescript"]:
            # Find related documentation
            related_docs = await graphiti_service.find_related_entities(
                entity_id=result["entity_id"],
                relationship_types=["DOCUMENTED_BY", "DESCRIBED_BY"],
                target_types=["Documentation", "CodeDocumentation"]
            )

            # Check for missing or outdated documentation
            if not related_docs:
                discrepancy_analysis["discrepancies"].append({
                    "type": "missing_documentation",
                    "severity": "medium",
                    "description": "Code function/class lacks documentation",
                    "suggested_action": "Add documentation for this code entity"
                })
            else:
                # Similar temporal analysis from code perspective
                for doc_entity in related_docs:
                    code_last_modified = result["temporal_metadata"]["t_valid"]
                    doc_last_modified = doc_entity["temporal_metadata"]["t_valid"]

                    if code_last_modified > doc_last_modified:
                        time_gap = code_last_modified - doc_last_modified

                        discrepancy_analysis["discrepancies"].append({
                            "type": "documentation_lag",
                            "severity": "high" if time_gap.days > 7 else "medium",
                            "description": f"Code modified {time_gap.days} days after documentation",
                            "related_documentation": doc_entity["title"],
                            "suggested_action": "Update documentation to reflect recent code changes"
                        })

        enhanced_results.append(discrepancy_analysis)

    return enhanced_results

# Gemini Enhancement: Semantic Discrepancy Detection Functions
async def detect_semantic_discrepancy(
    doc_content: str,
    doc_embedding: Optional[List[float]],
    code_entity: Dict
) -> Dict:
    """
    Detect semantic drift between documentation and code using embeddings

    Enhanced semantic analysis from Gemini's audit report
    """
    if not doc_embedding or not code_entity.get("embedding"):
        return {"has_discrepancy": False, "reason": "missing_embeddings"}

    # Calculate semantic similarity using cosine similarity
    from numpy import dot
    from numpy.linalg import norm

    doc_vec = doc_embedding
    code_vec = code_entity["embedding"]

    # Cosine similarity calculation
    similarity = dot(doc_vec, code_vec) / (norm(doc_vec) * norm(code_vec))

    # Similarity thresholds (tunable based on empirical data)
    CRITICAL_THRESHOLD = 0.3   # Very low similarity = critical drift
    WARNING_THRESHOLD = 0.5    # Moderate similarity = potential drift

    if similarity < CRITICAL_THRESHOLD:
        return {
            "has_discrepancy": True,
            "severity": "critical",
            "similarity_score": similarity,
            "description": f"Documentation semantically divergent from code (similarity: {similarity:.2f})",
            "suggested_action": "Review and completely rewrite documentation to match current implementation"
        }
    elif similarity < WARNING_THRESHOLD:
        return {
            "has_discrepancy": True,
            "severity": "warning",
            "similarity_score": similarity,
            "description": f"Documentation may be outdated relative to code (similarity: {similarity:.2f})",
            "suggested_action": "Review documentation for accuracy and update if needed"
        }
    else:
        return {
            "has_discrepancy": False,
            "similarity_score": similarity,
            "description": f"Documentation semantically aligned with code (similarity: {similarity:.2f})"
        }

async def detect_parameter_discrepancies(
    doc_content: str,
    code_entity: Dict
) -> Dict:
    """
    Detect parameter mismatches between documentation and code

    Enhanced parameter analysis from Gemini's audit report
    """
    issues = []

    # Extract current parameters from code entity
    current_params = set(code_entity.get("parameters", []))

    # Get historical parameters for drift detection
    historical_params = set(code_entity.get("historical_metadata", {}).get("previous_parameters", []))

    # Find parameters that were removed from code
    removed_params = historical_params - current_params

    # Simple text-based search for parameter mentions in documentation
    # (In production, this could use more sophisticated NLP)
    import re
    doc_text_lower = doc_content.lower()

    for param in removed_params:
        # Check if documentation still mentions removed parameters
        param_patterns = [
            rf'\b{re.escape(param.lower())}\b',           # Exact parameter name
            rf'{re.escape(param.lower())}\s*[=:]',        # Parameter assignment/definition
            rf'`{re.escape(param.lower())}`',             # Code-quoted parameter
            rf'"{re.escape(param.lower())}"',             # String-quoted parameter
        ]

        for pattern in param_patterns:
            if re.search(pattern, doc_text_lower):
                issues.append({
                    "type": "outdated_parameter_reference",
                    "severity": "medium",
                    "description": f"Documentation references removed parameter '{param}'",
                    "parameter_name": param,
                    "suggested_action": f"Remove or update references to parameter '{param}' in documentation"
                })
                break  # Avoid duplicate issues for same parameter

    # Check for missing documentation of new parameters
    for param in current_params:
        if param not in historical_params:  # New parameter
            param_patterns = [
                rf'\b{re.escape(param.lower())}\b',
                rf'`{re.escape(param.lower())}`'
            ]

            param_documented = any(re.search(pattern, doc_text_lower) for pattern in param_patterns)

            if not param_documented:
                issues.append({
                    "type": "missing_parameter_documentation",
                    "severity": "low",
                    "description": f"New parameter '{param}' not documented",
                    "parameter_name": param,
                    "suggested_action": f"Add documentation for new parameter '{param}'"
                })

    return {
        "has_discrepancy": len(issues) > 0,
        "issues": issues,
        "analysis_summary": {
            "removed_params": list(removed_params),
            "current_params": list(current_params),
            "total_issues": len(issues)
        }
    }

# Example usage showing temporal discrepancy detection
async def example_discrepancy_detection():
    """Example of how temporal knowledge helps identify inconsistencies"""

    # Search for authentication-related content
    results = await unified_content_search(
        query="user authentication process",
        content_filter="all",  # Include both code and docs
        detect_discrepancies=True
    )

    for result in results:
        if result["discrepancies"]:
            print(f"📄 {result['file_path']}")
            for discrepancy in result["discrepancies"]:
                print(f"  ⚠️  {discrepancy['type']}: {discrepancy['description']}")
                print(f"      💡 Suggestion: {discrepancy['suggested_action']}")

    # Example output:
    # 📄 docs/authentication.md
    #   ⚠️  outdated_documentation: Documentation last updated 45 days before related code
    #       💡 Suggestion: Review and update documentation to match current implementation
    #
    # 📄 src/auth/oauth.py
    #   ⚠️  documentation_lag: Code modified 12 days after documentation
    #       💡 Suggestion: Update documentation to reflect recent code changes
```

**Key Temporal Discrepancy Detection Capabilities:**

1. **Time-Based Analysis**: Compares `t_valid` timestamps between related documents and code
2. **Relationship Traversal**: Uses Graphiti graph to find doc-code relationships (`DOCUMENTS`, `DESCRIBES`, etc.)
3. **Severity Assessment**: Categorizes discrepancies based on time gaps and change frequency
4. **Actionable Insights**: Provides specific recommendations for maintaining consistency
5. **Bidirectional Detection**: Identifies issues from both documentation and code perspectives

#### Category 2: Schema and Migration Tools
**Tools**: `schema_init`, `schema_status`, `schema_validate`, `schema_add_node_type`, `schema_add_relationship`, `migration_generate`, `migration_apply`, `migration_rollback`, `migration_status`, `schema_diff`

**Current Issues**:
- Direct Neo4j schema manipulation
- No integration with Graphiti temporal model
- Manual validation without temporal consistency checks

**Modernization**:
```python
# Before: Direct Neo4j schema management
async def schema_add_node_type(name: str, properties: Dict):
    # Direct Cypher execution
    cypher = f"CREATE CONSTRAINT {name}_unique IF NOT EXISTS FOR (n:{name}) REQUIRE n.id IS UNIQUE"
    await neo4j_service.execute(cypher)

    # Update schema file manually
    update_schema_file(name, properties)

# After: Temporal-aware schema management
async def schema_add_node_type(name: str, properties: Dict):
    # Create schema change as Graphiti episode
    schema_episode = {
        "entities": [{
            "type": "SchemaChange",
            "operation": "add_node_type",
            "node_type": name,
            "properties": properties,
            "timestamp": datetime.now(timezone.utc)
        }],
        "relationships": [{
            "source": name,
            "target": "ProjectSchema",
            "type": "PART_OF"
        }]
    }

    # Process as temporal episode
    result = await graphiti_service.add_episode(
        name=f"schema_change_{name}_{int(time.time())}",
        episode_body=schema_episode,
        source=EpisodeType.json,
        metadata={
            "schema_operation": "add_node_type",
            "node_type": name,
            "project": self.project_name
        }
    )

    # Graphiti handles the actual Neo4j schema changes
    # with full temporal tracking and rollback capability
    return {
        "status": "success",
        "episode_id": result.uuid,
        "node_type": name,
        "temporal_metadata": {
            "created_at": result.created_at,
            "can_rollback": True
        }
    }
```

#### Category 3: Indexing and Content Tools
**Tools**: `reindex_path`, `backfill_metadata`, `canon_understanding`

**Current Issues**:
- File-based batch processing prone to failures
- No incremental updates
- Manual conflict resolution

**Modernization**:
```python
# Before: Batch file processing
async def reindex_path(path: str, recursive: bool = True):
    files = discover_files(path, recursive)

    # All-or-nothing batch processing
    try:
        for file_path in files:
            content = read_file(file_path)
            chunks = extract_chunks(content)
            embeddings = await nomic_service.embed(chunks)

            # Dual write to both Qdrant and Neo4j
            await qdrant_service.store(embeddings)
            await neo4j_service.store(chunks)
    except Exception as e:
        # Entire batch fails
        logger.error(f"Reindexing failed: {e}")
        return {"status": "error", "files_processed": 0}

# After: Episodic processing with temporal tracking
async def reindex_path(path: str, recursive: bool = True):
    files = discover_files(path, recursive)

    # Process each file as independent episode
    results = []
    for file_path in files:
        try:
            content = read_file(file_path)

            # Single episodic ingestion with temporal metadata
            episode_result = await enhanced_indexer.process_file_episode(
                file_path=file_path,
                content=content,
                commit_sha=get_current_commit(),
                commit_timestamp=get_file_timestamp(file_path)
            )

            results.append(episode_result)

        except Exception as e:
            # Individual file failure doesn't affect others
            results.append({
                "status": "error",
                "file_path": file_path,
                "error": str(e)
            })

    # Analyze results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    return {
        "status": "completed",
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(files),
        "temporal_tracking": True,
        "can_query_history": True
    }
```

#### Category 4: System Monitoring Tools
**Tools**: `neural_system_status`, `indexer_status`, `instance_metrics`

**Current Issues**:
- Fragmented status reporting across multiple systems
- No unified health metrics
- Limited visibility into temporal operations

**Modernization**:
```python
# Before: Fragmented system status
async def neural_system_status():
    # Check individual services separately
    neo4j_status = await neo4j_service.health_check()
    qdrant_status = await qdrant_service.health_check()
    nomic_status = await nomic_service.health_check()

    return {
        "neo4j": neo4j_status,
        "qdrant": qdrant_status,  # Will be deprecated
        "nomic": nomic_status,
        "overall": "partial"  # Complex logic to determine
    }

# After: Unified GraphRAG platform status
async def neural_system_status():
    # Single unified platform check
    platform_status = await unified_graphrag_platform.health_check()

    return {
        "platform": "unified_graphrag",
        "components": {
            "neo4j": platform_status.neo4j,
            "graphiti": platform_status.graphiti,
            "nomic": platform_status.nomic,
            "temporal_engine": platform_status.temporal_engine
        },
        "capabilities": {
            "semantic_search": platform_status.search_available,
            "temporal_queries": platform_status.temporal_available,
            "episodic_indexing": platform_status.indexing_available,
            "multi_project": platform_status.isolation_verified
        },
        "performance": {
            "avg_query_latency": platform_status.avg_latency_ms,
            "episodes_processed_today": platform_status.episodes_today,
            "success_rate_24h": platform_status.success_rate
        },
        "overall": "operational" if platform_status.all_healthy else "degraded"
    }
```

#### Category 5: Project Management Tools
**Tools**: `set_project_context`, `list_projects`

**Current Issues**:
- Manual project context management
- No integration with temporal project evolution

**Modernization**:
```python
# Before: Manual project context
async def set_project_context(path: str = None):
    if path:
        project_name = detect_project_name(path)
    else:
        project_name = auto_detect_project()

    # Manual container creation
    global_state.current_project = project_name
    container = ServiceContainer(project_name)
    await container.initialize()

    return {"project": project_name, "status": "set"}

# After: Temporal-aware project context with full history
async def set_project_context(path: str = None):
    if path:
        project_name = detect_project_name(path)
    else:
        project_name = auto_detect_project()

    # Enhanced container with Graphiti temporal capabilities
    container = await enhanced_service_state.get_project_container(project_name)

    # Query project history and status
    project_stats = await container.graphiti.get_project_stats()
    recent_activity = await container.graphiti.search_temporal(
        query="recent activity",
        time_filter={"last": "24h"}
    )

    return {
        "project": project_name,
        "status": "set",
        "enhanced_capabilities": {
            "temporal_queries": True,
            "episodic_indexing": True,
            "isolation": "double_verified"  # ADR-29 + Graphiti
        },
        "project_history": {
            "total_episodes": project_stats["total_episodes"],
            "last_activity": project_stats["last_episode"],
            "recent_files": len(recent_activity)
        }
    }
```

## Implementation Plan

### Phase 1: Unified Content Search Modernization (Week 1)

#### Tools to Modernize
- **CONSOLIDATE**: `semantic_code_search` + `graphrag_hybrid_search` → `unified_content_search`
- **ENHANCE**: `project_understanding` with temporal capabilities

#### Implementation Steps

**1.1 Unified Content Search Interface**
```python
# neural-tools/src/servers/tools/unified_content_search_tools.py
"""
Unified content search tools for code and documentation with temporal discrepancy detection
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta

class UnifiedContentSearchTools:
    """Unified search tools for all content types using Neo4j + Graphiti temporal platform"""

    def __init__(self, service_container):
        self.container = service_container
        self.graphiti = service_container.graphiti
        self.neo4j = service_container.neo4j

    async def unified_content_search(
        self,
        query: str,
        limit: int = 10,
        temporal_filter: Optional[Dict] = None,
        project_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Enhanced semantic search with temporal and project capabilities

        Args:
            query: Natural language search query
            limit: Maximum results to return (1-50)
            temporal_filter: Optional temporal constraints
                - {"after": "2025-01-01", "before": "2025-02-01"}
                - {"last": "1w"} for last week
                - {"commit": "abc123"} for specific commit state
            project_filter: Limit to specific project (auto-detected if None)
        """
        try:
            # Build search parameters
            search_params = {
                "query": query,
                "num_results": limit,
                "hybrid_search": True,  # Semantic + BM25 + Graph
                "include_temporal": True,
                "project_isolation": True  # Automatic via Graphiti group_id
            }

            # Add temporal filtering if specified
            if temporal_filter:
                search_params["temporal_filter"] = self._build_temporal_filter(temporal_filter)

            # Execute unified search
            start_time = time.time()
            results = await self.graphiti.search(**search_params)
            search_time = time.time() - start_time

            # Format results with enhanced metadata
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.content,
                    "file_path": result.metadata.get("file_path", ""),
                    "relevance_score": result.score,
                    "search_method": "unified_graphrag",
                    "temporal_metadata": {
                        "created_at": result.t_valid,
                        "last_modified": result.t_invalid,
                        "commit_sha": result.metadata.get("commit_sha"),
                        "episode_id": result.source_episode
                    },
                    "graph_context": {
                        "related_functions": result.relationships.get("functions", []),
                        "related_classes": result.relationships.get("classes", []),
                        "dependencies": result.relationships.get("dependencies", [])
                    }
                })

            return {
                "results": formatted_results,
                "metadata": {
                    "query": query,
                    "total_results": len(formatted_results),
                    "search_time_ms": round(search_time * 1000, 2),
                    "search_method": "unified_graphrag_temporal",
                    "temporal_filter_applied": temporal_filter is not None,
                    "project": self.container.project_name
                }
            }

        except Exception as e:
            logger.error(f"Enhanced semantic search failed: {e}")
            return {
                "results": [],
                "error": str(e),
                "fallback_available": True
            }

    async def graphrag_hybrid_search(
        self,
        query: str,
        limit: int = 5,
        include_graph_context: bool = True,
        max_hops: int = 2,
        temporal_scope: str = "current"
    ) -> List[Dict]:
        """
        Enhanced hybrid search with temporal graph traversal

        Args:
            temporal_scope: "current", "all", "last_week", "last_month"
        """
        # Build temporal constraints
        temporal_filter = None
        if temporal_scope != "all":
            temporal_filter = self._build_temporal_scope(temporal_scope)

        # Execute enhanced hybrid search
        results = await self.graphiti.hybrid_search(
            query=query,
            num_results=limit,
            include_graph_context=include_graph_context,
            max_relationship_hops=max_hops,
            temporal_filter=temporal_filter
        )

        # Enhanced graph context with temporal relationships
        enriched_results = []
        for result in results:
            if include_graph_context:
                # Get temporal graph context
                graph_context = await self._get_temporal_graph_context(
                    result.entity_id,
                    max_hops,
                    temporal_filter
                )
                result.graph_context = graph_context

            enriched_results.append(result)

        return enriched_results

    def _build_temporal_filter(self, filter_spec: Dict) -> Dict:
        """Convert user temporal filter to Graphiti format"""
        if "last" in filter_spec:
            # Handle relative time: "1w", "1d", "1m"
            duration_str = filter_spec["last"]
            duration = self._parse_duration(duration_str)
            return {
                "t_valid_after": datetime.now() - duration,
                "t_invalid_before": None  # Still valid
            }

        if "after" in filter_spec or "before" in filter_spec:
            # Handle absolute dates
            filter_dict = {}
            if "after" in filter_spec:
                filter_dict["t_valid_after"] = datetime.fromisoformat(filter_spec["after"])
            if "before" in filter_spec:
                filter_dict["t_valid_before"] = datetime.fromisoformat(filter_spec["before"])
            return filter_dict

        if "commit" in filter_spec:
            # Handle commit-specific queries
            return {
                "source_episode_contains": filter_spec["commit"]
            }

        return {}
```

#### Phase 1 Testing Criteria

**Phase 1 Exit Criteria - Must achieve ALL:**

1. **Unified Search Performance**: ≤300ms P95 latency for all content types (code + docs)
2. **Content Type Detection**: 100% accurate auto-detection of query intent (code vs docs vs all)
3. **Temporal Query Accuracy**: 100% correct results for temporal filters across all content
4. **Document-Code Relationship**: 100% accurate relationship traversal between docs and code
5. **Discrepancy Detection**: ≥95% accuracy in identifying outdated documentation
6. **Project Isolation**: 0% cross-project data leakage in search results
7. **Tool Consolidation**: Single unified tool successfully replaces both legacy search tools
8. **Semantic Drift Detection** (Gemini Enhancement): ≥90% accuracy in identifying semantic divergence
9. **Parameter Drift Detection** (Gemini Enhancement): ≥95% accuracy in detecting parameter mismatches

```python
# Phase 1 Acceptance Tests
async def test_phase1_search_modernization():
    """Comprehensive testing for modernized search tools"""

    # Test 1: Search performance validation
    test_queries = [
        "authentication functions",
        "database connections",
        "error handling patterns",
        "API endpoints",
        "test utilities"
    ]

    performance_results = []
    for query in test_queries:
        start_time = time.time()
        results = await enhanced_search.semantic_code_search(query, limit=10)
        query_time = time.time() - start_time

        performance_results.append(query_time)
        assert query_time <= 0.300  # ≤300ms requirement
        assert len(results["results"]) > 0  # Should find relevant results

    p95_latency = percentile(performance_results, 95)
    assert p95_latency <= 0.300  # P95 ≤300ms

    # Test 2: Temporal query accuracy
    temporal_tests = [
        {"filter": {"last": "1w"}, "expected_constraint": "recent"},
        {"filter": {"after": "2025-01-01"}, "expected_constraint": "since_date"},
        {"filter": {"commit": "abc123"}, "expected_constraint": "specific_commit"}
    ]

    for test_case in temporal_tests:
        results = await enhanced_search.semantic_code_search(
            "functions",
            temporal_filter=test_case["filter"]
        )

        # Verify temporal constraints applied correctly
        assert results["metadata"]["temporal_filter_applied"] == True

        # All results should respect temporal boundaries
        for result in results["results"]:
            temporal_meta = result["temporal_metadata"]
            assert temporal_meta is not None
            assert validate_temporal_constraint(temporal_meta, test_case["filter"])

    # Test 3: Project isolation verification
    projects = ["test-project-a", "test-project-b", "test-project-c"]

    for project in projects:
        container = await get_project_container(project)
        search_tool = EnhancedSearchTools(container)

        # Add project-specific test data
        await add_test_data(project, f"unique_function_{project}")

        # Search should only find own project's data
        results = await search_tool.semantic_code_search(f"unique_function_{project}")

        for result in results["results"]:
            # Should only find functions from this project
            assert project in result["file_path"] or project in result["content"]

        # Should NOT find other projects' unique functions
        for other_project in projects:
            if other_project != project:
                other_results = await search_tool.semantic_code_search(f"unique_function_{other_project}")
                assert len(other_results["results"]) == 0  # No cross-project leakage

    # Test 4: Semantic Drift Detection (Gemini Enhancement)
    semantic_test_cases = [
        {
            "doc_content": "This function validates user passwords using MD5 hashing",
            "code_content": "def validate_password(password): return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())",
            "expected_drift": True,
            "expected_severity": "critical"  # MD5 vs bcrypt is major change
        },
        {
            "doc_content": "Returns a list of active users from the database",
            "code_content": "def get_active_users(): return db.query('SELECT * FROM users WHERE active = 1').fetchall()",
            "expected_drift": False,
            "expected_severity": None  # Semantically aligned
        },
        {
            "doc_content": "Sends email notifications to users",
            "code_content": "def send_push_notification(user_id, message): return push_service.notify(user_id, message)",
            "expected_drift": True,
            "expected_severity": "warning"  # Email vs push notification
        }
    ]

    for test_case in semantic_test_cases:
        # Create mock embeddings for test
        doc_embedding = await nomic_service.get_embeddings([test_case["doc_content"]])
        code_embedding = await nomic_service.get_embeddings([test_case["code_content"]])

        code_entity = {
            "content": test_case["code_content"],
            "embedding": code_embedding[0]
        }

        # Test semantic drift detection
        drift_result = await detect_semantic_discrepancy(
            doc_content=test_case["doc_content"],
            doc_embedding=doc_embedding[0],
            code_entity=code_entity
        )

        # Verify detection accuracy
        assert drift_result["has_discrepancy"] == test_case["expected_drift"]
        if test_case["expected_drift"]:
            assert drift_result["severity"] == test_case["expected_severity"]

    # Test 5: Parameter Drift Detection (Gemini Enhancement)
    parameter_test_cases = [
        {
            "doc_content": "Function takes username, password, and remember_me parameters",
            "current_params": ["username", "password"],
            "historical_params": ["username", "password", "remember_me"],
            "expected_issues": 1,  # remember_me removed but still in docs
            "expected_issue_type": "outdated_parameter_reference"
        },
        {
            "doc_content": "Authentication function takes username and password",
            "current_params": ["username", "password", "two_factor_token"],
            "historical_params": ["username", "password"],
            "expected_issues": 1,  # two_factor_token added but not documented
            "expected_issue_type": "missing_parameter_documentation"
        }
    ]

    for test_case in parameter_test_cases:
        code_entity = {
            "parameters": test_case["current_params"],
            "historical_metadata": {
                "previous_parameters": test_case["historical_params"]
            }
        }

        # Test parameter drift detection
        param_result = await detect_parameter_discrepancies(
            doc_content=test_case["doc_content"],
            code_entity=code_entity
        )

        # Verify detection accuracy
        assert len(param_result["issues"]) == test_case["expected_issues"]
        if test_case["expected_issues"] > 0:
            assert param_result["issues"][0]["type"] == test_case["expected_issue_type"]

# Phase 1 STOP Conditions
PHASE1_STOP_CONDITIONS = [
    "Search latency exceeds 500ms consistently",
    "Temporal queries return incorrect results",
    "Cross-project data leakage detected",
    "Fallback mechanism fails",
    "Result quality significantly degraded vs baseline",
    "Semantic drift detection accuracy below 90%",  # Gemini enhancement
    "Parameter drift detection accuracy below 95%"  # Gemini enhancement
]
```

### Phase 2: Schema and Migration Tools Modernization (Week 2)

#### Tools to Modernize
- `schema_init`, `schema_status`, `schema_validate`
- `schema_add_node_type`, `schema_add_relationship`
- `migration_generate`, `migration_apply`, `migration_rollback`, `migration_status`, `schema_diff`

#### Implementation Steps

**2.1 Temporal Schema Management**
```python
# neural-tools/src/servers/tools/enhanced_schema_tools.py
"""
Modernized schema tools with temporal tracking and episodic changes
"""

class EnhancedSchemaTools:
    """Schema management with full temporal tracking via Graphiti episodes"""

    async def schema_add_node_type(
        self,
        name: str,
        properties: Dict,
        description: str = None,
        indexes: List[str] = None
    ) -> Dict:
        """
        Add node type with full temporal tracking

        Creates Graphiti episode for the schema change, enabling:
        - Full rollback capability
        - Historical schema evolution tracking
        - Automated validation and conflict detection
        """
        try:
            # Validate new node type
            validation_result = await self._validate_node_type(name, properties)
            if not validation_result.is_valid:
                return {
                    "status": "error",
                    "error": "validation_failed",
                    "details": validation_result.errors
                }

            # Create schema change episode
            schema_episode = {
                "entities": [{
                    "type": "SchemaNodeType",
                    "name": name,
                    "properties": properties,
                    "description": description or f"Node type: {name}",
                    "indexes": indexes or [],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "project": self.container.project_name,
                    "version": await self._get_next_schema_version()
                }],
                "relationships": [{
                    "source": name,
                    "target": "ProjectSchema",
                    "type": "PART_OF",
                    "metadata": {
                        "relationship_type": "schema_component",
                        "added_at": datetime.now(timezone.utc).isoformat()
                    }
                }]
            }

            # Process as temporal episode with automatic Neo4j schema updates
            episode_result = await self.container.graphiti.add_episode(
                name=f"schema_add_node_{name}_{int(time.time())}",
                episode_body=schema_episode,
                source=EpisodeType.json,
                metadata={
                    "schema_operation": "add_node_type",
                    "node_type": name,
                    "project": self.container.project_name,
                    "can_rollback": True,
                    "schema_version": schema_episode["entities"][0]["version"]
                }
            )

            # Verify schema was applied correctly
            verification_result = await self._verify_node_type_creation(name, properties)

            return {
                "status": "success",
                "node_type": name,
                "episode_id": episode_result.uuid,
                "schema_version": schema_episode["entities"][0]["version"],
                "temporal_metadata": {
                    "created_at": episode_result.created_at,
                    "can_rollback": True,
                    "rollback_command": f"schema_rollback_to_episode {episode_result.uuid}"
                },
                "verification": verification_result
            }

        except Exception as e:
            logger.error(f"Schema node type addition failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "node_type": name
            }

    async def migration_apply(
        self,
        target_version: int = None,
        dry_run: bool = False
    ) -> Dict:
        """
        Apply migrations with full temporal tracking

        Each migration step becomes a Graphiti episode, enabling:
        - Granular rollback to any migration step
        - Full audit trail of schema evolution
        - Automated validation and conflict resolution
        """
        try:
            # Get pending migrations
            pending_migrations = await self._get_pending_migrations(target_version)

            if not pending_migrations:
                return {
                    "status": "no_migrations",
                    "message": "No pending migrations to apply"
                }

            if dry_run:
                return {
                    "status": "dry_run",
                    "pending_migrations": pending_migrations,
                    "would_apply": len(pending_migrations)
                }

            # Apply each migration as separate episode
            migration_results = []
            for migration in pending_migrations:
                try:
                    # Create migration episode
                    migration_episode = {
                        "entities": [{
                            "type": "SchemaMigration",
                            "migration_id": migration["id"],
                            "version": migration["version"],
                            "description": migration["description"],
                            "operations": migration["operations"],
                            "applied_at": datetime.now(timezone.utc).isoformat(),
                            "project": self.container.project_name
                        }],
                        "relationships": [{
                            "source": f"migration_{migration['id']}",
                            "target": "ProjectSchema",
                            "type": "MODIFIES",
                            "metadata": {
                                "migration_version": migration["version"],
                                "operation_count": len(migration["operations"])
                            }
                        }]
                    }

                    # Apply migration as episode
                    episode_result = await self.container.graphiti.add_episode(
                        name=f"migration_{migration['id']}_{migration['version']}",
                        episode_body=migration_episode,
                        source=EpisodeType.json,
                        metadata={
                            "schema_operation": "migration",
                            "migration_id": migration["id"],
                            "migration_version": migration["version"],
                            "project": self.container.project_name
                        }
                    )

                    # Verify migration applied correctly
                    verification = await self._verify_migration(migration)

                    migration_results.append({
                        "migration_id": migration["id"],
                        "version": migration["version"],
                        "status": "success",
                        "episode_id": episode_result.uuid,
                        "verification": verification
                    })

                except Exception as e:
                    # Individual migration failure doesn't stop others
                    migration_results.append({
                        "migration_id": migration["id"],
                        "version": migration["version"],
                        "status": "error",
                        "error": str(e)
                    })
                    logger.error(f"Migration {migration['id']} failed: {e}")

            # Analyze results
            successful = [m for m in migration_results if m["status"] == "success"]
            failed = [m for m in migration_results if m["status"] == "error"]

            return {
                "status": "completed",
                "total_migrations": len(pending_migrations),
                "successful": len(successful),
                "failed": len(failed),
                "migration_results": migration_results,
                "temporal_tracking": True,
                "rollback_available": len(successful) > 0
            }

        except Exception as e:
            logger.error(f"Migration application failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
```

#### Phase 2 Testing Criteria

**Phase 2 Exit Criteria - Must achieve ALL:**

1. **Schema Operation Success**: 100% success rate for all schema operations
2. **Temporal Rollback**: 100% success rate for schema rollbacks to any historical state
3. **Migration Reliability**: ≥99% success rate for individual migration steps
4. **Validation Accuracy**: 0% false positives/negatives in schema validation
5. **Performance Preservation**: Schema operations complete within 5s for typical schemas

```python
# Phase 2 Acceptance Tests
async def test_phase2_schema_modernization():
    """Comprehensive testing for modernized schema tools"""

    # Test 1: Schema operation success rate
    schema_operations = [
        ("add_node_type", "TestEntity", {"name": "string", "id": "long"}),
        ("add_relationship", "BELONGS_TO", ["TestEntity"], ["Project"]),
        ("add_node_type", "AnotherEntity", {"title": "string", "count": "long"}),
        ("modify_node_type", "TestEntity", {"name": "string", "id": "long", "active": "boolean"})
    ]

    operation_results = []
    for operation_type, *args in schema_operations:
        result = await enhanced_schema.execute_operation(operation_type, *args)
        operation_results.append(result["status"] == "success")

    success_rate = sum(operation_results) / len(operation_results)
    assert success_rate == 1.0  # 100% success rate required

    # Test 2: Temporal rollback capability
    # Get all schema episodes
    schema_episodes = await enhanced_schema.get_schema_history()

    # Test rollback to each historical state
    for i, episode in enumerate(schema_episodes[:-1]):  # Don't rollback to current
        rollback_result = await enhanced_schema.rollback_to_episode(episode["episode_id"])
        assert rollback_result["status"] == "success"

        # Verify schema state matches historical state
        current_schema = await enhanced_schema.get_current_schema()
        expected_schema = episode["schema_state"]
        assert schemas_match(current_schema, expected_schema)

        # Rollback to latest
        await enhanced_schema.rollback_to_latest()

    # Test 3: Migration reliability
    # Create test migrations
    test_migrations = generate_test_migrations(20)  # 20 migration steps

    migration_results = []
    for migration in test_migrations:
        result = await enhanced_schema.apply_single_migration(migration)
        migration_results.append(result["status"] == "success")

    migration_success_rate = sum(migration_results) / len(migration_results)
    assert migration_success_rate >= 0.99  # ≥99% success rate

    # Test 4: Schema validation accuracy
    validation_test_cases = [
        ({"valid_node": {"name": "string", "id": "long"}}, True),
        ({"invalid_node": {"name": "invalid_type"}}, False),
        ({"duplicate_node": {"name": "string"}}, False),  # If already exists
        ({"empty_node": {}}, False)
    ]

    validation_accuracy = []
    for schema_def, expected_valid in validation_test_cases:
        validation_result = await enhanced_schema.validate_schema(schema_def)
        actual_valid = validation_result["is_valid"]
        validation_accuracy.append(actual_valid == expected_valid)

    accuracy_rate = sum(validation_accuracy) / len(validation_accuracy)
    assert accuracy_rate == 1.0  # 100% validation accuracy

# Phase 2 STOP Conditions
PHASE2_STOP_CONDITIONS = [
    "Schema operation failure rate >1%",
    "Temporal rollback failures detected",
    "Migration success rate <95%",
    "Schema validation accuracy <98%",
    "Schema operations taking >10s consistently"
]
```

### Phase 3: Indexing and Content Tools Modernization (Week 3)

#### Tools to Modernize
- `reindex_path`
- `backfill_metadata`
- `canon_understanding`

#### Implementation Steps

**3.1 Episodic Indexing Tools**
```python
# neural-tools/src/servers/tools/enhanced_indexing_tools.py
"""
Modernized indexing tools using episodic processing and temporal tracking
"""

class EnhancedIndexingTools:
    """Indexing tools leveraging Graphiti episodic processing"""

    async def reindex_path(
        self,
        path: str,
        recursive: bool = True,
        batch_size: int = 10,
        include_temporal: bool = True
    ) -> Dict:
        """
        Episodic path reindexing with temporal tracking

        Each file becomes independent episode:
        - Individual file failures don't affect others
        - Full temporal tracking of indexing history
        - Automatic conflict resolution via Graphiti
        - Real-time progress updates
        """
        try:
            start_time = time.time()

            # Discover files to reindex
            files_to_process = await self._discover_files(path, recursive)
            total_files = len(files_to_process)

            if total_files == 0:
                return {
                    "status": "no_files",
                    "path": path,
                    "message": "No files found to index"
                }

            logger.info(f"Starting episodic reindexing: {total_files} files from {path}")

            # Process files in batches for optimal performance
            processed_results = []
            for batch_start in range(0, total_files, batch_size):
                batch_files = files_to_process[batch_start:batch_start + batch_size]

                # Process batch as parallel episodes
                batch_tasks = []
                for file_path in batch_files:
                    task = self._process_file_episode_enhanced(
                        file_path,
                        include_temporal=include_temporal
                    )
                    batch_tasks.append(task)

                # Execute batch in parallel
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Process batch results
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        processed_results.append({
                            "file_path": batch_files[i],
                            "status": "error",
                            "error": str(result)
                        })
                    else:
                        processed_results.append(result)

                # Progress update
                completed = len(processed_results)
                logger.info(f"Episodic indexing progress: {completed}/{total_files} files")

            # Analyze final results
            successful = [r for r in processed_results if r["status"] == "success"]
            failed = [r for r in processed_results if r["status"] == "error"]

            end_time = time.time()
            duration = end_time - start_time

            return {
                "status": "completed",
                "summary": {
                    "path": path,
                    "total_files": total_files,
                    "successful": len(successful),
                    "failed": len(failed),
                    "success_rate": len(successful) / total_files,
                    "duration_seconds": round(duration, 2),
                    "files_per_second": round(total_files / duration, 2)
                },
                "indexing_method": "episodic_temporal",
                "capabilities": {
                    "temporal_tracking": include_temporal,
                    "individual_file_isolation": True,
                    "automatic_conflict_resolution": True,
                    "can_query_history": True
                },
                "failed_files": [f["file_path"] for f in failed[:10]],  # Limit error details
                "temporal_metadata": {
                    "indexing_session_id": f"reindex_{int(start_time)}",
                    "can_rollback": True,
                    "history_queryable": True
                }
            }

        except Exception as e:
            logger.error(f"Episodic reindexing failed for {path}: {e}")
            return {
                "status": "error",
                "path": path,
                "error": str(e)
            }

    async def _process_file_episode_enhanced(
        self,
        file_path: str,
        include_temporal: bool = True
    ) -> Dict:
        """Process single file with enhanced episodic capabilities"""
        try:
            # Read file content
            content = await self._read_file_safely(file_path)
            if not content:
                return {
                    "file_path": file_path,
                    "status": "skipped",
                    "reason": "empty_or_unreadable"
                }

            # Get enhanced metadata
            metadata = {}
            if include_temporal:
                metadata.update(await self._get_temporal_metadata(file_path))

            # Process as enhanced episode
            episode_result = await self.container.enhanced_indexer.process_file_episode(
                file_path=file_path,
                content=content,
                commit_sha=metadata.get("commit_sha"),
                commit_timestamp=metadata.get("commit_timestamp")
            )

            return {
                "file_path": file_path,
                "status": episode_result["status"],
                "episode_id": episode_result.get("episode_uuid"),
                "entities_extracted": episode_result.get("entities_extracted", 0),
                "relationships_created": episode_result.get("relationships_created", 0),
                "temporal_metadata": metadata if include_temporal else None
            }

        except Exception as e:
            return {
                "file_path": file_path,
                "status": "error",
                "error": str(e)
            }

    async def backfill_metadata(
        self,
        batch_size: int = 100,
        dry_run: bool = False
    ) -> Dict:
        """
        Enhanced metadata backfilling with episodic processing

        Updates existing data with:
        - Canonical weights and PRISM scores
        - Temporal metadata and Git information
        - Enhanced relationship extraction
        """
        try:
            # Find all existing episodes without enhanced metadata
            episodes_needing_backfill = await self._find_episodes_for_backfill()
            total_episodes = len(episodes_needing_backfill)

            if total_episodes == 0:
                return {
                    "status": "no_work",
                    "message": "All episodes already have enhanced metadata"
                }

            if dry_run:
                return {
                    "status": "dry_run",
                    "episodes_to_backfill": total_episodes,
                    "estimated_duration_minutes": total_episodes / 20  # ~20 episodes/minute
                }

            # Process in batches
            backfill_results = []
            for batch_start in range(0, total_episodes, batch_size):
                batch_episodes = episodes_needing_backfill[batch_start:batch_start + batch_size]

                batch_results = []
                for episode in batch_episodes:
                    try:
                        # Enhance episode with temporal metadata
                        enhancement_result = await self._enhance_episode_metadata(episode)
                        batch_results.append({
                            "episode_id": episode["uuid"],
                            "status": "success",
                            "enhancements": enhancement_result["enhancements"]
                        })
                    except Exception as e:
                        batch_results.append({
                            "episode_id": episode["uuid"],
                            "status": "error",
                            "error": str(e)
                        })

                backfill_results.extend(batch_results)

                # Progress update
                completed = len(backfill_results)
                logger.info(f"Metadata backfill progress: {completed}/{total_episodes}")

            # Analyze results
            successful = [r for r in backfill_results if r["status"] == "success"]
            failed = [r for r in backfill_results if r["status"] == "error"]

            return {
                "status": "completed",
                "total_episodes": total_episodes,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / total_episodes,
                "enhancements_applied": [
                    "canonical_weights",
                    "prism_scores",
                    "temporal_metadata",
                    "git_information",
                    "enhanced_relationships"
                ]
            }

        except Exception as e:
            logger.error(f"Metadata backfill failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def canon_understanding(self) -> Dict:
        """
        Enhanced canonical knowledge analysis with temporal insights

        Provides comprehensive breakdown of:
        - Canonical sources and their temporal evolution
        - PRISM scoring distribution over time
        - Canonical coverage recommendations
        """
        try:
            # Get comprehensive canonical analysis
            canonical_analysis = await self._analyze_canonical_knowledge_temporal()

            return {
                "canonical_overview": {
                    "total_canonical_sources": canonical_analysis["total_sources"],
                    "canonical_coverage_percentage": canonical_analysis["coverage_percent"],
                    "canonical_quality_score": canonical_analysis["quality_score"],
                    "temporal_tracking": True
                },
                "canonical_sources": canonical_analysis["sources"],
                "temporal_insights": {
                    "canonical_evolution": canonical_analysis["evolution"],
                    "trending_canonical_patterns": canonical_analysis["trending"],
                    "canonical_gaps_identified": canonical_analysis["gaps"]
                },
                "recommendations": canonical_analysis["recommendations"],
                "data_source": "unified_graphrag_temporal"
            }

        except Exception as e:
            logger.error(f"Canonical understanding analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
```

#### Phase 3 Testing Criteria

**Phase 3 Exit Criteria - Must achieve ALL:**

1. **Episodic Processing Success**: ≥99% success rate for individual file episodes
2. **Parallel Processing Performance**: Handle 100+ concurrent file episodes efficiently
3. **Temporal Metadata Accuracy**: 100% accurate temporal tracking for all operations
4. **Failure Isolation**: Individual file failures never affect other files
5. **Backfill Reliability**: ≥98% success rate for metadata backfilling operations

```python
# Phase 3 Acceptance Tests
async def test_phase3_indexing_modernization():
    """Comprehensive testing for modernized indexing tools"""

    # Test 1: Episodic processing success rate
    test_repository = generate_large_test_repository(1000)  # 1000 test files

    reindex_result = await enhanced_indexing.reindex_path(
        test_repository.path,
        recursive=True,
        batch_size=20
    )

    success_rate = reindex_result["summary"]["success_rate"]
    assert success_rate >= 0.99  # ≥99% success rate required

    # Verify individual episode isolation
    failed_files = reindex_result.get("failed_files", [])
    if failed_files:
        # Retry failed files individually to ensure isolation
        retry_results = []
        for failed_file in failed_files[:5]:  # Test sample
            retry_result = await enhanced_indexing.reindex_path(failed_file, recursive=False)
            retry_results.append(retry_result)

        # Some retries might succeed (proves isolation)
        # Others might legitimately fail (corrupt files, etc.)
        assert len(retry_results) > 0  # At least attempted retries

    # Test 2: Parallel processing performance
    concurrent_batches = []
    for i in range(10):  # 10 concurrent batches
        batch_files = test_repository.get_batch(i, size=50)  # 50 files each
        batch_task = enhanced_indexing.reindex_path(
            batch_files.path,
            recursive=True,
            batch_size=10
        )
        concurrent_batches.append(batch_task)

    # Execute all batches concurrently
    start_time = time.time()
    batch_results = await asyncio.gather(*concurrent_batches, return_exceptions=True)
    concurrent_duration = time.time() - start_time

    # Should handle 500 files across 10 batches efficiently
    assert concurrent_duration <= 60  # Within 60 seconds for 500 files

    # All batches should complete successfully
    successful_batches = [r for r in batch_results if isinstance(r, dict) and r.get("status") == "completed"]
    assert len(successful_batches) >= 8  # At least 80% of batches succeed

    # Test 3: Temporal metadata accuracy
    temporal_test_files = [
        ("v1.py", "def func(): pass", "2025-01-01"),
        ("v2.py", "def func(): return True", "2025-01-15"),
        ("v3.py", "def func(): return False", "2025-02-01")
    ]

    # Index files with specific temporal metadata
    for file_name, content, date_str in temporal_test_files:
        test_file_path = create_test_file(file_name, content)

        result = await enhanced_indexing.reindex_path(
            test_file_path,
            include_temporal=True
        )

        assert result["status"] == "completed"
        assert result["capabilities"]["temporal_tracking"] == True

        # Verify temporal metadata preserved
        episode_data = await get_episode_for_file(test_file_path)
        assert episode_data["temporal_metadata"]["created_at"] is not None
        assert episode_data["temporal_metadata"]["file_timestamp"] is not None

    # Test 4: Metadata backfill reliability
    # Create episodes missing enhanced metadata
    episodes_to_backfill = await create_episodes_without_metadata(100)

    backfill_result = await enhanced_indexing.backfill_metadata(
        batch_size=20,
        dry_run=False
    )

    backfill_success_rate = backfill_result["success_rate"]
    assert backfill_success_rate >= 0.98  # ≥98% success rate

    # Verify metadata was actually added
    for episode_id in episodes_to_backfill[:10]:  # Sample verification
        enhanced_episode = await get_episode_metadata(episode_id)
        assert "canonical_weight" in enhanced_episode
        assert "prism_score" in enhanced_episode
        assert "temporal_metadata" in enhanced_episode

# Phase 3 STOP Conditions
PHASE3_STOP_CONDITIONS = [
    "Episodic processing success rate <95%",
    "Concurrent processing failures or deadlocks",
    "Temporal metadata corruption or loss",
    "Cross-file contamination during failures",
    "Backfill success rate <95%"
]
```

### Phase 4: System Monitoring and Project Management Modernization (Week 4)

#### Tools to Modernize
- `neural_system_status`, `indexer_status`, `instance_metrics`
- `set_project_context`, `list_projects`

#### Implementation Steps

**4.1 Unified Platform Monitoring**
```python
# neural-tools/src/servers/tools/enhanced_monitoring_tools.py
"""
Modernized monitoring tools for unified GraphRAG platform
"""

class EnhancedMonitoringTools:
    """System monitoring for unified Neo4j+Graphiti platform"""

    async def neural_system_status(self) -> Dict:
        """
        Comprehensive platform status for unified GraphRAG system

        Provides unified view of:
        - Neo4j + Graphiti platform health
        - Temporal capabilities status
        - Multi-project isolation health
        - Performance metrics and SLA compliance
        """
        try:
            # Get unified platform status
            platform_health = await self._get_unified_platform_health()

            # Get performance metrics
            performance_metrics = await self._get_performance_metrics()

            # Get temporal system status
            temporal_status = await self._get_temporal_system_status()

            # Get multi-project isolation status
            isolation_status = await self._get_isolation_status()

            return {
                "platform": {
                    "name": "unified_graphrag",
                    "version": "adr_66_67_integrated",
                    "overall_status": platform_health["overall"],
                    "last_health_check": datetime.now(timezone.utc).isoformat()
                },
                "core_components": {
                    "neo4j": {
                        "status": platform_health["neo4j"]["status"],
                        "version": platform_health["neo4j"]["version"],
                        "connection_pool": platform_health["neo4j"]["connections"],
                        "memory_usage": platform_health["neo4j"]["memory"],
                        "vector_index_health": platform_health["neo4j"]["vector_health"]
                    },
                    "graphiti": {
                        "status": platform_health["graphiti"]["status"],
                        "version": platform_health["graphiti"]["version"],
                        "episodes_processed_today": platform_health["graphiti"]["episodes_today"],
                        "temporal_engine": platform_health["graphiti"]["temporal_engine"],
                        "conflict_resolution": platform_health["graphiti"]["conflict_resolution"]
                    },
                    "nomic": {
                        "status": platform_health["nomic"]["status"],
                        "embeddings_generated_today": platform_health["nomic"]["embeddings_today"],
                        "avg_embedding_time_ms": platform_health["nomic"]["avg_time_ms"]
                    }
                },
                "capabilities": {
                    "semantic_search": {
                        "available": temporal_status["search_available"],
                        "avg_latency_ms": performance_metrics["search_latency_ms"],
                        "sla_compliance": performance_metrics["search_sla_compliance"]
                    },
                    "temporal_queries": {
                        "available": temporal_status["temporal_available"],
                        "avg_latency_ms": performance_metrics["temporal_latency_ms"],
                        "historical_depth_days": temporal_status["historical_depth"]
                    },
                    "episodic_indexing": {
                        "available": temporal_status["indexing_available"],
                        "success_rate_24h": performance_metrics["indexing_success_rate"],
                        "avg_processing_time_ms": performance_metrics["indexing_time_ms"]
                    },
                    "multi_project_isolation": {
                        "verified": isolation_status["isolation_verified"],
                        "active_projects": isolation_status["active_projects"],
                        "isolation_method": "adr_29_plus_graphiti_group_id"
                    }
                },
                "performance_sla": {
                    "search_p95_latency_ms": performance_metrics["search_p95"],
                    "temporal_query_p95_ms": performance_metrics["temporal_p95"],
                    "indexing_success_rate": performance_metrics["indexing_success_rate"],
                    "overall_sla_compliance": performance_metrics["overall_sla_compliance"]
                },
                "alerts": platform_health.get("alerts", []),
                "recommendations": platform_health.get("recommendations", [])
            }

        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {
                "platform": "unified_graphrag",
                "status": "error",
                "error": str(e),
                "fallback_status": await self._get_fallback_status()
            }

    async def indexer_status(self) -> Dict:
        """
        Enhanced indexer status with episodic processing metrics
        """
        try:
            # Get enhanced indexer metrics
            indexer_metrics = await self._get_enhanced_indexer_metrics()

            return {
                "indexer_type": "enhanced_episodic",
                "architecture": "neo4j_graphiti_temporal",
                "status": indexer_metrics["overall_status"],
                "processing_mode": "episodic_parallel",
                "capabilities": {
                    "temporal_tracking": True,
                    "automatic_conflict_resolution": True,
                    "individual_file_isolation": True,
                    "graceful_degradation": True
                },
                "current_metrics": {
                    "episodes_processed_today": indexer_metrics["episodes_today"],
                    "success_rate_24h": indexer_metrics["success_rate"],
                    "avg_episode_processing_ms": indexer_metrics["avg_processing_time"],
                    "parallel_episode_capacity": indexer_metrics["parallel_capacity"],
                    "fallback_usage_rate": indexer_metrics["fallback_rate"]
                },
                "temporal_insights": {
                    "historical_episodes": indexer_metrics["total_historical_episodes"],
                    "temporal_queries_supported": True,
                    "rollback_capability": True,
                    "conflict_resolution_rate": indexer_metrics["conflict_resolution_rate"]
                },
                "health_indicators": {
                    "graphiti_connection": indexer_metrics["graphiti_healthy"],
                    "neo4j_connection": indexer_metrics["neo4j_healthy"],
                    "nomic_connection": indexer_metrics["nomic_healthy"],
                    "overall_health": indexer_metrics["overall_healthy"]
                }
            }

        except Exception as e:
            logger.error(f"Enhanced indexer status check failed: {e}")
            return {
                "indexer_type": "enhanced_episodic",
                "status": "error",
                "error": str(e)
            }

class EnhancedProjectTools:
    """Enhanced project management with temporal capabilities"""

    async def set_project_context(self, path: str = None) -> Dict:
        """
        Set project context with enhanced temporal capabilities
        """
        try:
            # Detect or validate project
            if path:
                project_name = await self._detect_project_from_path(path)
            else:
                project_name = await self._auto_detect_current_project()

            if not project_name:
                return {
                    "status": "error",
                    "error": "Could not detect project context"
                }

            # Get enhanced project container
            container = await enhanced_service_state.get_project_container(project_name)

            # Get project temporal insights
            project_insights = await self._get_project_temporal_insights(container)

            # Verify enhanced capabilities
            capabilities_check = await self._verify_enhanced_capabilities(container)

            return {
                "project": project_name,
                "status": "active",
                "context_path": path,
                "enhanced_capabilities": {
                    "temporal_queries": capabilities_check["temporal_available"],
                    "episodic_indexing": capabilities_check["indexing_available"],
                    "isolation_verified": capabilities_check["isolation_verified"],
                    "graphiti_enabled": capabilities_check["graphiti_enabled"]
                },
                "project_insights": {
                    "total_episodes": project_insights["total_episodes"],
                    "last_activity": project_insights["last_activity"],
                    "temporal_depth_days": project_insights["temporal_depth"],
                    "recent_changes": project_insights["recent_changes"]
                },
                "data_architecture": {
                    "storage": "unified_neo4j_vectors",
                    "temporal_engine": "graphiti",
                    "isolation_method": "adr_29_plus_group_id",
                    "search_capabilities": ["semantic", "hybrid", "temporal", "graph_traversal"]
                }
            }

        except Exception as e:
            logger.error(f"Enhanced project context setting failed: {e}")
            return {
                "status": "error",
                "project": project_name if 'project_name' in locals() else "unknown",
                "error": str(e)
            }

    async def list_projects(self) -> Dict:
        """
        List all projects with enhanced temporal insights
        """
        try:
            # Get all known projects
            all_projects = await enhanced_service_state.get_all_projects()

            enhanced_project_list = []
            for project_name in all_projects:
                try:
                    container = await enhanced_service_state.get_project_container(project_name)

                    # Get project status and insights
                    project_stats = await container.graphiti.get_project_stats()
                    recent_activity = await self._get_recent_project_activity(container)

                    enhanced_project_list.append({
                        "name": project_name,
                        "status": "active" if container.initialized else "inactive",
                        "enhanced_features": {
                            "temporal_tracking": container.graphiti is not None,
                            "episodic_indexing": container.enhanced_indexer is not None,
                            "isolation_verified": True
                        },
                        "statistics": {
                            "total_episodes": project_stats.get("total_episodes", 0),
                            "last_episode": project_stats.get("last_episode"),
                            "temporal_depth_days": recent_activity.get("depth_days", 0)
                        },
                        "recent_activity": {
                            "files_indexed_24h": recent_activity.get("files_24h", 0),
                            "queries_executed_24h": recent_activity.get("queries_24h", 0),
                            "last_activity": recent_activity.get("last_activity")
                        }
                    })

                except Exception as e:
                    # Individual project failure doesn't stop listing
                    enhanced_project_list.append({
                        "name": project_name,
                        "status": "error",
                        "error": str(e)
                    })

            return {
                "total_projects": len(all_projects),
                "active_projects": len([p for p in enhanced_project_list if p["status"] == "active"]),
                "projects": enhanced_project_list,
                "platform_info": {
                    "architecture": "unified_graphrag",
                    "temporal_capabilities": True,
                    "multi_project_isolation": "adr_29_plus_graphiti"
                }
            }

        except Exception as e:
            logger.error(f"Enhanced project listing failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
```

#### Phase 4 Testing Criteria

**Phase 4 Exit Criteria - Must achieve ALL:**

1. **Unified Platform Monitoring**: 100% accurate health reporting for all components
2. **Performance SLA Tracking**: Real-time SLA compliance monitoring with <1% false positives
3. **Multi-Project Visibility**: Complete visibility into all project states and activities
4. **Enhanced Project Context**: Temporal insights and capabilities verification for all projects
5. **Monitoring Reliability**: ≤5s response time for all monitoring operations

```python
# Phase 4 Acceptance Tests
async def test_phase4_monitoring_modernization():
    """Comprehensive testing for modernized monitoring and project tools"""

    # Test 1: Unified platform monitoring accuracy
    system_status = await enhanced_monitoring.neural_system_status()

    # Verify all expected components are monitored
    required_components = ["neo4j", "graphiti", "nomic"]
    for component in required_components:
        assert component in system_status["core_components"]
        assert system_status["core_components"][component]["status"] in ["healthy", "degraded", "error"]

    # Verify enhanced capabilities are reported
    required_capabilities = ["semantic_search", "temporal_queries", "episodic_indexing", "multi_project_isolation"]
    for capability in required_capabilities:
        assert capability in system_status["capabilities"]
        assert "available" in system_status["capabilities"][capability]

    # Verify performance SLA tracking
    sla_metrics = system_status["performance_sla"]
    assert "search_p95_latency_ms" in sla_metrics
    assert "temporal_query_p95_ms" in sla_metrics
    assert "overall_sla_compliance" in sla_metrics

    # Test 2: Performance SLA accuracy
    # Execute controlled performance tests
    performance_test_results = []

    for i in range(20):  # 20 test queries
        start_time = time.time()
        search_result = await enhanced_search.semantic_code_search("test query")
        query_time = time.time() - start_time
        performance_test_results.append(query_time * 1000)  # Convert to ms

    measured_p95 = percentile(performance_test_results, 95)
    reported_p95 = sla_metrics["search_p95_latency_ms"]

    # SLA reporting should be within 10% of measured performance
    accuracy_ratio = abs(measured_p95 - reported_p95) / measured_p95
    assert accuracy_ratio <= 0.10  # Within 10% accuracy

    # Test 3: Multi-project visibility
    projects_list = await enhanced_project.list_projects()

    # Should have comprehensive project information
    assert "total_projects" in projects_list
    assert "projects" in projects_list

    for project_info in projects_list["projects"]:
        if project_info["status"] != "error":
            # Each project should have enhanced features reported
            assert "enhanced_features" in project_info
            assert "temporal_tracking" in project_info["enhanced_features"]
            assert "statistics" in project_info
            assert "recent_activity" in project_info

    # Test 4: Enhanced project context verification
    test_projects = ["test-project-temporal", "test-project-enhanced"]

    for project_name in test_projects:
        # Set up test project
        await setup_test_project(project_name)

        # Set project context
        context_result = await enhanced_project.set_project_context()

        if context_result["status"] == "active":
            # Verify enhanced capabilities
            capabilities = context_result["enhanced_capabilities"]
            assert capabilities["temporal_queries"] == True
            assert capabilities["episodic_indexing"] == True
            assert capabilities["isolation_verified"] == True

            # Verify project insights
            insights = context_result["project_insights"]
            assert "total_episodes" in insights
            assert "temporal_depth_days" in insights

    # Test 5: Monitoring performance
    monitoring_performance_tests = [
        ("neural_system_status", enhanced_monitoring.neural_system_status),
        ("indexer_status", enhanced_monitoring.indexer_status),
        ("list_projects", enhanced_project.list_projects)
    ]

    for test_name, test_function in monitoring_performance_tests:
        start_time = time.time()
        result = await test_function()
        response_time = time.time() - start_time

        assert response_time <= 5.0  # ≤5s response time
        assert result is not None
        if isinstance(result, dict):
            assert "status" not in result or result["status"] != "error"

# Phase 4 STOP Conditions
PHASE4_STOP_CONDITIONS = [
    "System status reporting >10% inaccurate",
    "Performance SLA tracking >5% false positives",
    "Project visibility incomplete or missing",
    "Enhanced capabilities not properly reported",
    "Monitoring response time >10s consistently"
]
```

## Final Integration Testing and Production Readiness

### Comprehensive Cross-ADR Integration Testing

After all 4 phases are complete, comprehensive testing across ADR-66, ADR-67, and ADR-68:

```python
# Final ADR-68 Integration Tests
async def test_final_adr68_integration():
    """Final validation for complete MCP tools modernization"""

    # Test 1: End-to-end workflow with all modernized tools
    projects = ["integration-test-a", "integration-test-b", "integration-test-c"]

    for project in projects:
        # Set enhanced project context
        context_result = await enhanced_project.set_project_context(project_path=f"/test/{project}")
        assert context_result["enhanced_capabilities"]["temporal_queries"] == True

        # Index content with episodic processing
        index_result = await enhanced_indexing.reindex_path(f"/test/{project}/src", recursive=True)
        assert index_result["summary"]["success_rate"] >= 0.99

        # Perform enhanced searches
        search_result = await enhanced_search.semantic_code_search("authentication", limit=10)
        assert len(search_result["results"]) > 0
        assert search_result["metadata"]["search_method"] == "unified_graphrag_temporal"

        # Execute temporal queries
        temporal_result = await enhanced_search.semantic_code_search(
            "functions",
            temporal_filter={"last": "1w"}
        )
        assert temporal_result["metadata"]["temporal_filter_applied"] == True

        # Verify schema operations
        schema_result = await enhanced_schema.schema_add_node_type(
            f"TestEntity{project}",
            {"name": "string", "project": "string"}
        )
        assert schema_result["status"] == "success"
        assert schema_result["temporal_metadata"]["can_rollback"] == True

        # Check system status
        status_result = await enhanced_monitoring.neural_system_status()
        assert status_result["platform"]["overall_status"] in ["healthy", "operational"]
        assert status_result["capabilities"]["temporal_queries"]["available"] == True

    # Test 2: Cross-project isolation verification
    # Each project should only see its own data despite using same tools
    for project in projects:
        container = await enhanced_service_state.get_project_container(project)

        # Search should be isolated to this project
        project_search = await container.enhanced_search.semantic_code_search("authentication")

        for result in project_search["results"]:
            # Should only find content from this project
            assert project in result["file_path"] or project in result["content"]

    # Test 3: Performance across all modernized tools
    tool_performance_tests = [
        ("semantic_search", lambda: enhanced_search.semantic_code_search("functions")),
        ("hybrid_search", lambda: enhanced_search.graphrag_hybrid_search("classes")),
        ("project_status", lambda: enhanced_project.list_projects()),
        ("system_status", lambda: enhanced_monitoring.neural_system_status()),
        ("schema_operation", lambda: enhanced_schema.schema_status())
    ]

    for tool_name, tool_function in tool_performance_tests:
        start_time = time.time()
        result = await tool_function()
        response_time = time.time() - start_time

        assert response_time <= 5.0  # All tools should respond within 5s
        assert result is not None

        if isinstance(result, dict) and "error" in result:
            assert False, f"Tool {tool_name} returned error: {result['error']}"

# Production Readiness Checklist for ADR-68
ADR68_PRODUCTION_CHECKLIST = [
    # Tool Modernization
    "✅ All 15 MCP tools updated to unified GraphRAG architecture",
    "✅ Temporal capabilities integrated across all search tools",
    "✅ Episodic processing implemented for all indexing tools",
    "✅ Enhanced monitoring with unified platform visibility",

    # Performance & Reliability
    "✅ All tools responding within SLA (<5s for monitoring, <300ms for search)",
    "✅ ≥99% success rate for episodic operations across all tools",
    "✅ Perfect multi-project isolation maintained across all tools",
    "✅ Graceful degradation verified for all enhanced tools",

    # Integration & Consistency
    "✅ Unified data access pattern across all tools",
    "✅ Consistent temporal metadata and tracking",
    "✅ Cross-tool data consistency verified",
    "✅ Enhanced capabilities properly exposed in all tools",

    # Operational Excellence
    "✅ Comprehensive monitoring for all modernized tools",
    "✅ Performance SLA tracking and alerting",
    "✅ Rollback procedures tested for all tool operations",
    "✅ Documentation updated for all enhanced tool capabilities"
]

# Final cleanup for ADR-68
async def execute_adr68_final_cleanup():
    """Execute final cleanup for ADR-68 completion"""

    # Remove any legacy tool implementations
    await cleanup_legacy_tool_code()

    # Optimize all tool performance for production
    await optimize_all_tools_for_production()

    # Enable comprehensive tool monitoring
    await enable_unified_tool_monitoring()

    # Validate all tools in production configuration
    production_validation = await validate_all_tools_production_ready()
    assert production_validation.all_tools_operational

    logger.info("🎉 ADR-68 implementation complete - All MCP tools modernized for unified GraphRAG platform!")
```

## Migration Strategy and Risk Mitigation

### Gradual Tool Migration Approach

**Week 1**: Search Tools (Lower Risk)
- Modernize search tools first (lowest user impact)
- Enable A/B testing between old and new implementations
- Validate performance and accuracy

**Week 2**: Schema Tools (Medium Risk)
- Migrate schema management with full rollback capability
- Test temporal schema tracking extensively
- Validate migration reliability

**Week 3**: Indexing Tools (Higher Risk)
- Migrate to episodic processing with fallback
- Extensive testing of parallel processing
- Validate individual file isolation

**Week 4**: Monitoring/Project Tools (Critical)
- Modernize monitoring and project management
- Ensure complete visibility into new architecture
- Final integration testing

### Risk Mitigation Strategy

| Risk Category | Mitigation |
|---------------|------------|
| **Tool Failures** | Graceful degradation to legacy implementations |
| **Performance Regression** | A/B testing and gradual rollout |
| **Data Consistency** | Comprehensive validation between old/new tools |
| **User Experience** | Backward compatibility and enhanced features |
| **Operational Issues** | Enhanced monitoring and alerting |

## Decision Outcome

**Status**: PROPOSED

This comprehensive modernization transforms all 15 MCP tools to leverage the unified GraphRAG platform, providing:

### Core Benefits

1. **Unified Architecture**: All tools use single Neo4j+Graphiti backend
2. **Enhanced Capabilities**: Temporal queries, episodic processing, improved performance
3. **Operational Excellence**: Consistent monitoring, SLA tracking, graceful degradation
4. **Future-Proof**: Ready for advanced GraphRAG features and scaling

### Tool Enhancement Summary

| Tool Category | Tools Modernized | Key Enhancement |
|---------------|------------------|-----------------|
| **Search & Discovery** | 3 tools | Temporal search + unified results |
| **Schema & Migration** | 10 tools | Episodic schema changes + rollback |
| **Indexing & Content** | 3 tools | Parallel episodic processing |
| **System Monitoring** | 3 tools | Unified platform visibility |
| **Project Management** | 2 tools | Enhanced context + insights |

**Timeline**: 4 weeks implementation + 1 week validation
**Risk**: Low-Medium - gradual migration with fallback capability
**Reward**: High - unified, performant, temporal-capable MCP tool ecosystem

---

**Conclusion**: This modernization completes the architectural transformation established by ADR-66 and ADR-67, providing users with a unified, high-performance GraphRAG platform accessible through enhanced MCP tools with temporal capabilities and operational excellence.