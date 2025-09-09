# MCP API Versioning Strategy
**L9 Contract Management for Neural Tools MCP Server**

## Version Control Philosophy

### Semantic Versioning for MCP APIs
```
MAJOR.MINOR.PATCH
1.2.3
│ │ │
│ │ └── Patch: Bug fixes, no API changes
│ └──── Minor: New MCP tools, backward compatible
└────── Major: Breaking API changes, signature modifications
```

### Current API Version: `1.0.0`
- **Established**: P1 implementation (September 2025)
- **Baseline**: 9 core MCP tools with validated signatures
- **Compatibility**: All P1 fixes are backward compatible

## MCP Tool Contract Specifications

### Core Tools (`v1.0.0`)

#### `project_understanding`
```typescript
interface ProjectUnderstandingRequest {
  scope?: "full" | "summary" | "architecture" | "dependencies" | "core_logic"
  max_tokens?: string // Default: "2000"
}
```
**Contract**: Must return `Dict[str, Any]` with project analysis

#### `semantic_code_search` 
```typescript
interface SemanticSearchRequest {
  query: string                    // Required
  search_type?: "semantic" | "hybrid" | "vector"
  limit?: string                  // Default: "10"
  filters?: string               // JSON filter conditions
}
```
**Contract**: Must return search results with relevance scores
**P1 Impact**: Uses `search_vectors()` internally (non-breaking change)

#### `atomic_dependency_tracer`
```typescript  
interface DependencyTracerRequest {
  target: string                  // Required: function/class/module name
  trace_type?: "calls" | "imports" | "full"
  depth?: string                 // Default: "3"
}
```
**Contract**: Must return dependency graph structure

#### `vibe_preservation`
```typescript
interface VibePreservationRequest {
  action: string                  // Required: "analyze" | "generate" | "check"
  code_sample?: string           // Code to analyze
  context?: string               // Additional context
}
```
**Contract**: Must return style analysis and recommendations

#### `project_auto_index`
```typescript
interface AutoIndexRequest {
  scope?: "modified" | "all" | "incremental"
  since_minutes?: string         // Time window for indexing
  force_reindex?: string        // "true" | "false"
}
```
**Contract**: Must return indexing status and statistics

#### `graph_query`
```typescript
interface GraphQueryRequest {
  query: string                  // Required: Cypher query
}
```
**Contract**: Must return Neo4j query results or error

### Memory Tools (`v1.0.0`)

#### `memory_store_enhanced`
```typescript
interface MemoryStoreRequest {
  content: string                // Required: content to store
  category?: string             // Default: "general"  
  metadata?: string             // JSON metadata
}
```
**Contract**: Must return storage confirmation with ID

#### `memory_search_enhanced`
```typescript
interface MemorySearchRequest {
  query: string                  // Required: search query
  category?: string             // Filter by category
  limit?: string                // Default: "10"
}
```
**Contract**: Must return ranked search results

#### `schema_customization`
```typescript
interface SchemaCustomizationRequest {
  action: string                 // Required: "create" | "update" | "delete" | "list"
  collection_name?: string       // Target collection
  schema_data?: string          // JSON schema definition
}
```
**Contract**: Must return schema operation result

## Breaking Change Policy

### Major Version Changes (1.x → 2.x)
**Triggers:**
- Removing MCP tools
- Changing required parameters
- Modifying return value structure
- Renaming tool methods

**Process:**
1. **6-month deprecation notice** with clear migration path
2. **Dual version support** during transition
3. **Client impact assessment** before rollout
4. **Automated migration tools** where possible

### Minor Version Changes (1.0 → 1.1)
**Allowed:**
- Adding new MCP tools
- Adding optional parameters
- Enhancing return value data (additive only)
- Performance improvements

**Process:**
1. **Contract tests pass** on existing tools
2. **Backward compatibility validated** 
3. **Release notes** document new capabilities

### Patch Version Changes (1.0.0 → 1.0.1)
**Allowed:**
- Bug fixes in tool implementations
- Error message improvements
- Performance optimizations
- Internal refactoring (P1-style changes)

## P1 Integration Impact Assessment

### Non-Breaking Changes ✅
- **P1-A**: `search()` → `search_vectors()` (internal implementation)
- **P1-B**: Dimension validation (enhanced error handling)
- **P1-C**: `delete_points()` method (new capability, not exposed via MCP)
- **P1-D**: Centralized configuration (infrastructure improvement)

### API Compatibility Maintained
- All MCP tool signatures unchanged
- Response formats preserved  
- Error handling enhanced (not modified)
- Performance improved (no breaking changes)

**Result**: P1 implementation remains at `v1.0.0` - no version bump required

## Client Migration Guidance

### Version Detection
```bash
# Check MCP server API version
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | \
  mcp-client | jq '.result.tools[] | select(.name == "api_version")'
```

### Compatibility Testing
```bash
# Automated contract validation
./scripts/test-mcp-api-contract.sh --tool=semantic_code_search
./scripts/test-semantic-search-e2e.sh
```

### Migration Path Template
```typescript
// v1.0 → v1.1 migration example
// OLD: semantic_code_search(query: string)
// NEW: semantic_code_search(query: string, search_type?: string)

// Backward compatible - old calls still work
await mcp.semantic_code_search("test query");           // ✅ Works
await mcp.semantic_code_search("test query", "hybrid"); // ✅ Enhanced
```

## Testing Strategy

### Contract Test Suite
- **Signature Validation**: All tool parameter types and names
- **Response Schema**: Return value structure consistency  
- **Error Handling**: Graceful failure modes
- **Performance**: Regression detection for API calls

### Version Compatibility Matrix
```
Client Version    Server v1.0.x    Server v1.1.x    Server v2.0.x
v1.0.x           ✅ Full          ✅ Full          ❌ Breaking
v1.1.x           ✅ Full          ✅ Full          ❌ Breaking  
v2.0.x           ⚠️  Degraded     ⚠️  Degraded     ✅ Full
```

## Monitoring & Observability

### API Usage Metrics
- Tool call frequency by version
- Deprecated API usage tracking
- Client version distribution
- Error rates by API version

### Deployment Gates
```yaml
# CI/CD pipeline requirements
before_deploy:
  - contract_tests_pass: 100%
  - backward_compatibility: verified
  - performance_regression: none
  - client_impact_assessed: true
```

## Emergency Rollback

### Version Rollback Strategy
1. **Immediate**: Revert to previous container image
2. **Client Notification**: API version change announcements
3. **Data Consistency**: Ensure database schema compatibility
4. **Monitoring**: Track rollback impact metrics

### Rollback Testing
```bash
# Validate rollback compatibility
./scripts/test-api-rollback.sh --from=1.1.0 --to=1.0.0
```

## Future Roadmap

### Planned v1.1.0 Features
- Enhanced semantic search filters
- Batch processing capabilities for multiple tools
- Real-time indexing status notifications

### v2.0.0 Considerations
- GraphRAG integration improvements
- Multi-tenant collection support  
- Advanced query optimization

**Next Review Date**: January 2026