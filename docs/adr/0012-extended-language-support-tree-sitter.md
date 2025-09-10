# ADR-0012: Extended Language Support for Tree-Sitter Integration

**Status:** Proposed  
**Date:** 2025-09-09  
**Deciders:** Engineering Team  

## Context

We have successfully implemented tree-sitter extraction for Python, JavaScript, and TypeScript in our existing `TreeSitterExtractor` service. The system is operational and integrated into our neural indexer pipeline. To maximize value, we need to extend language support to cover more of our codebase.

## Decision

Extend the **existing** `TreeSitterExtractor` class in `neural-tools/src/servers/services/tree_sitter_extractor.py` to support:
- Go (.go)
- Rust (.rs) 
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- Ruby (.rb)
- Swift (.swift)

## Integration Approach

### 1. Modify Existing TreeSitterExtractor
```python
# EDIT neural-tools/src/servers/services/tree_sitter_extractor.py
# Add to __init__ method's self.languages dict:
'.go': Language(tsgo.language()),
'.rs': Language(tsrust.language()),
'.java': Language(tsjava.language()),
# etc.

# Add extraction methods:
def _extract_go_symbols(...)
def _extract_rust_symbols(...)
def _extract_java_symbols(...)
```

### 2. Update Dependencies
```txt
# EDIT neural-tools/config/requirements-indexer-lean.txt
# Add tree-sitter language packages:
tree-sitter-go>=0.21.0
tree-sitter-rust>=0.21.0
tree-sitter-java>=0.21.0
tree-sitter-c>=0.21.0
tree-sitter-cpp>=0.21.0
tree-sitter-ruby>=0.21.0
tree-sitter-swift>=0.21.0
```

### 3. Enhance Symbol Types
```python
# EDIT existing _extract_* methods to handle language-specific constructs:
- Go: interfaces, goroutines, channels, structs
- Rust: traits, impls, macros, enums
- Java: annotations, abstract classes, enums
- C/C++: structs, unions, templates, namespaces
```

## Implementation Details

### Phase 1: Core Languages (Week 1)
- Go and Rust (high priority for cloud-native codebases)
- Reuse existing extraction patterns
- Add to existing test suite

### Phase 2: Enterprise Languages (Week 2)
- Java and C/C++
- Handle complex inheritance hierarchies
- Support template/generic extraction

### Phase 3: Dynamic Languages (Week 3)
- Ruby and Swift
- Handle dynamic typing constructs
- Support protocol-oriented patterns

## Consequences

### Positive
- Broader codebase coverage using existing infrastructure
- No new services or containers needed
- Leverages existing Qdrant/Neo4j storage
- Incremental rollout possible via feature flags

### Negative
- Increased Docker image size (~50MB for parsers)
- Slightly longer initialization time
- More complex symbol type mappings

### Neutral
- Maintains existing architecture
- Uses same async batch processing
- Compatible with current timeout protections

## Testing Strategy

1. Extend `test_tree_sitter.py` with language-specific test cases
2. Add to existing integration tests
3. Verify symbol extraction accuracy per language
4. Performance benchmarks for large codebases

## Rollback Plan

- Feature flag per language: `EXTRACT_GO_ENABLED`, `EXTRACT_RUST_ENABLED`, etc.
- Graceful degradation if parser fails
- Existing Python/JS/TS extraction unaffected

## Metrics

- Symbol extraction rate per language
- Parse time per file type
- Memory usage with all parsers loaded
- Query performance impact

## References

- Current implementation: `neural-tools/src/servers/services/tree_sitter_extractor.py`
- Tree-sitter language support: https://tree-sitter.github.io/tree-sitter/
- Existing ADR-0011: Semantic Search Enablement