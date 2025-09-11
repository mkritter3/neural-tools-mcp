# ADR-0022: Semantic Search Exclusion and Re-ranking System for GraphRAG

## Status
Proposed

## Context

Our GraphRAG hybrid search is returning results from archived and deprecated code paths, making it difficult to distinguish between active and historical code. Additionally, all results are treated equally regardless of recency, relevance source, or archive status. Current search results mix:
- Active code in `neural-tools/`
- Archived code in `.archive/neural-tools/src_backup/`
- Deprecated code in `.archive/.deprecated/`
- Backup files and old implementations

This creates noise in semantic search results and confuses both Claude instances and developers trying to understand the current codebase structure.

### Research Findings (September 2025)

Based on current industry practices:

1. **GitHub Copilot** (2025) - Instant semantic indexing but no built-in archive exclusion
2. **CocoIndex** - Uses `.gitignore`-style patterns for RAG indexing with `excluded_patterns`
3. **Semgrep** - Implements `.semgrepignore` with `:include` directive for chaining ignore files
4. **VS Code** - Uses `search.exclude` and `files.exclude` settings with `.gitignore` integration
5. **No standard `.searchignore` or `.ragignore`** format exists yet in the industry

## Decision

Implement a **multi-layered exclusion and re-ranking system** for our GraphRAG semantic search and indexing, using a combination of:

1. **`.graphragignore` file** - Primary exclusion configuration
2. **Metadata tagging** - Mark archived/deprecated status in vector payloads
3. **Query-time filtering** - Optional exclusion of archived results
4. **Indexer configuration** - Prevent indexing of excluded paths
5. **Reciprocal Rank Fusion (RRF)** - Merge results from multiple search methods
6. **Dynamic re-ranking** - Apply recency boosts and archive penalties

## Implementation Design

### 1. `.graphragignore` File Format

Create a `.graphragignore` file in the project root with gitignore-compatible syntax:

```gitignore
# .graphragignore - Patterns to exclude from GraphRAG indexing

# Include .gitignore patterns
:include:.gitignore

# Archive and backup directories
.archive/
.deprecated/
*_backup/
*.backup
*.old
*.deprecated

# Build and dependency directories
dist/
build/
node_modules/
target/
__pycache__/
*.pyc

# Test fixtures and mocks
test/fixtures/
test/mocks/
**/__mocks__/

# Documentation archives
docs/archive/
docs/deprecated/

# Temporary and cache files
*.tmp
*.temp
.cache/
tmp/

# IDE and editor files
.idea/
.vscode/
*.swp
*.swo
*~

# Large binary files
*.bin
*.dat
*.db
*.sqlite

# Generated files
*.generated.*
*_gen.go
*.pb.go

# Version control metadata
.git/
.svn/
.hg/
```

### 2. Exclusion Pattern Parser

```python
# neural-tools/src/servers/services/exclusion_manager.py

import re
import fnmatch
from pathlib import Path
from typing import List, Set, Optional
from dataclasses import dataclass

@dataclass
class ExclusionPattern:
    """Represents a single exclusion pattern"""
    pattern: str
    is_negation: bool = False  # Patterns starting with !
    is_directory: bool = False  # Patterns ending with /
    source: str = ".graphragignore"

class ExclusionManager:
    """Manages file and directory exclusion patterns for GraphRAG"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.patterns: List[ExclusionPattern] = []
        self.compiled_patterns: List[re.Pattern] = []
        self._load_patterns()
    
    def _load_patterns(self):
        """Load patterns from .graphragignore and other sources"""
        ignore_file = self.project_root / ".graphragignore"
        
        if ignore_file.exists():
            self._parse_ignore_file(ignore_file)
        
        # Add default patterns if no ignore file exists
        if not self.patterns:
            self._add_default_patterns()
    
    def _parse_ignore_file(self, file_path: Path):
        """Parse a .graphragignore file"""
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Handle :include: directive
                if line.startswith(':include:'):
                    include_path = line[9:].strip()
                    included_file = self.project_root / include_path
                    if included_file.exists():
                        self._parse_ignore_file(included_file)
                    continue
                
                # Parse pattern
                is_negation = line.startswith('!')
                if is_negation:
                    line = line[1:]
                
                is_directory = line.endswith('/')
                
                pattern = ExclusionPattern(
                    pattern=line,
                    is_negation=is_negation,
                    is_directory=is_directory,
                    source=str(file_path.name)
                )
                
                self.patterns.append(pattern)
                self.compiled_patterns.append(
                    self._compile_pattern(pattern)
                )
    
    def _compile_pattern(self, pattern: ExclusionPattern) -> re.Pattern:
        """Convert gitignore pattern to regex"""
        # Convert gitignore pattern to regex
        regex_pattern = fnmatch.translate(pattern.pattern)
        
        # Handle ** for recursive matching
        regex_pattern = regex_pattern.replace('.*', '.**')
        
        return re.compile(regex_pattern)
    
    def _add_default_patterns(self):
        """Add sensible default exclusion patterns"""
        defaults = [
            '.archive/',
            '.deprecated/',
            '*_backup/',
            '*.backup',
            '*.old',
            'node_modules/',
            '__pycache__/',
            '.git/',
            'dist/',
            'build/',
            '*.pyc',
            '*.tmp',
            '*.temp'
        ]
        
        for pattern_str in defaults:
            pattern = ExclusionPattern(
                pattern=pattern_str,
                is_directory=pattern_str.endswith('/'),
                source="defaults"
            )
            self.patterns.append(pattern)
            self.compiled_patterns.append(
                self._compile_pattern(pattern)
            )
    
    def should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded from indexing"""
        relative_path = file_path.relative_to(self.project_root)
        path_str = str(relative_path)
        
        excluded = False
        
        for pattern, regex in zip(self.patterns, self.compiled_patterns):
            # Skip directory patterns if checking a file
            if pattern.is_directory and file_path.is_file():
                continue
            
            if regex.match(path_str):
                if pattern.is_negation:
                    excluded = False  # Negation patterns re-include
                else:
                    excluded = True
        
        return excluded
    
    def get_excluded_paths(self, base_path: Path) -> Set[Path]:
        """Get all excluded paths under a base path"""
        excluded = set()
        
        for path in base_path.rglob('*'):
            if self.should_exclude(path):
                excluded.add(path)
        
        return excluded
    
    def filter_search_results(self, results: List[dict]) -> List[dict]:
        """Filter search results to exclude archived/deprecated items"""
        filtered = []
        
        for result in results:
            file_path = result.get('file_path', '')
            if file_path:
                path = Path(file_path)
                if not self.should_exclude(path):
                    filtered.append(result)
            else:
                # Keep results without file paths
                filtered.append(result)
        
        return filtered
```

### 3. Integration with Indexer

```python
# Modify neural-tools/src/servers/services/neural_indexer.py

class NeuralIndexer:
    def __init__(self, ...):
        # ... existing init ...
        self.exclusion_manager = ExclusionManager(self.project_root)
    
    async def index_file(self, file_path: Path):
        """Index a single file if not excluded"""
        # Check exclusion before indexing
        if self.exclusion_manager.should_exclude(file_path):
            logger.debug(f"Skipping excluded file: {file_path}")
            return
        
        # Check if file is in archived/deprecated directory
        is_archived = self._is_archived_path(file_path)
        
        # Proceed with indexing
        chunks = await self._extract_chunks(file_path)
        
        for chunk in chunks:
            # Add metadata about archive status
            chunk['metadata']['is_archived'] = is_archived
            chunk['metadata']['archive_path'] = str(file_path) if is_archived else None
            
            await self._index_chunk(chunk)
    
    def _is_archived_path(self, file_path: Path) -> bool:
        """Check if path is in an archive directory"""
        path_str = str(file_path)
        archive_indicators = [
            '.archive/',
            '.deprecated/',
            '_backup/',
            '/archive/',
            '/deprecated/',
            '/backup/'
        ]
        
        return any(indicator in path_str for indicator in archive_indicators)
```

### 4. Query-Time Filtering with Dynamic Re-ranking

```python
# Modify graphrag_hybrid_search_impl in neural_server_stdio.py

async def graphrag_hybrid_search_impl(
    query: str, 
    limit: int = 5, 
    include_graph_context: bool = True,
    max_hops: int = 2,
    exclude_archived: bool = True,  # New parameter
    use_rrf: bool = True,  # Enable Reciprocal Rank Fusion
    rerank_with_recency: bool = True  # Boost recent files
) -> List[types.TextContent]:
    """Enhanced hybrid search with archive exclusion and re-ranking"""
    
    # Get results from multiple search methods
    search_results = []
    
    # 1. Vector search (semantic)
    vector_results = await container.qdrant.search(
        collection_name="code",
        query_vector=query_embedding,
        limit=limit * 3,  # Get more for re-ranking
        with_payload=True
    )
    search_results.append(('vector', vector_results))
    
    # 2. Keyword search (if available)
    if hasattr(container, 'keyword_search'):
        keyword_results = await container.keyword_search(
            query=query,
            limit=limit * 3
        )
        search_results.append(('keyword', keyword_results))
    
    # 3. Graph-based search (structural)
    if include_graph_context:
        graph_results = await container.neo4j.search_related(
            query=query,
            limit=limit * 3
        )
        search_results.append(('graph', graph_results))
    
    # Apply Reciprocal Rank Fusion (RRF)
    if use_rrf:
        merged_results = apply_rrf(search_results, k=60)
    else:
        merged_results = vector_results
    
    # Filter out archived results
    if exclude_archived:
        merged_results = filter_archived(merged_results, exclusion_manager)
    
    # Apply dynamic re-ranking
    if rerank_with_recency:
        merged_results = apply_recency_boost(merged_results)
    
    # Apply freshness penalty to archived items (if not excluded)
    merged_results = apply_archive_penalty(merged_results)
    
    return merged_results[:limit]

def apply_rrf(search_results: List[Tuple[str, List]], k: int = 60) -> List:
    """
    Apply Reciprocal Rank Fusion to merge multiple ranked lists.
    RRF Score = Σ(1 / (rank + k)) for each list where item appears
    """
    item_scores = {}
    
    for source, results in search_results:
        for rank, result in enumerate(results, 1):
            item_id = get_result_id(result)
            
            if item_id not in item_scores:
                item_scores[item_id] = {
                    'score': 0,
                    'sources': [],
                    'data': result
                }
            
            # Calculate RRF score: 1 / (rank + k)
            rrf_score = 1.0 / (rank + k)
            item_scores[item_id]['score'] += rrf_score
            item_scores[item_id]['sources'].append(source)
    
    # Sort by combined RRF score
    sorted_items = sorted(
        item_scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )
    
    return [item['data'] for item in sorted_items]

def apply_recency_boost(results: List) -> List:
    """
    Apply dynamic weighting based on file recency.
    Recent files get a boost, older files get penalized.
    """
    import time
    from datetime import datetime, timedelta
    
    current_time = time.time()
    
    for result in results:
        file_path = result.payload.get('file_path')
        if file_path and os.path.exists(file_path):
            # Get file modification time
            mtime = os.path.getmtime(file_path)
            age_days = (current_time - mtime) / 86400
            
            # Calculate recency weight (exponential decay)
            # Files modified today: 1.5x boost
            # Files modified this week: 1.2x boost
            # Files modified this month: 1.0x (neutral)
            # Older files: gradually decrease to 0.5x
            if age_days < 1:
                recency_weight = 1.5
            elif age_days < 7:
                recency_weight = 1.2
            elif age_days < 30:
                recency_weight = 1.0
            else:
                # Exponential decay for older files
                recency_weight = max(0.5, 1.0 * math.exp(-age_days / 365))
            
            # Apply weight to score
            result.score *= recency_weight
            result.payload['recency_weight'] = recency_weight
    
    # Re-sort by adjusted scores
    return sorted(results, key=lambda x: x.score, reverse=True)

def apply_archive_penalty(results: List) -> List:
    """
    Apply penalty to archived/deprecated items without excluding them.
    This allows archived code to still appear but ranked lower.
    """
    for result in results:
        file_path = result.payload.get('file_path', '')
        
        # Check if in archive directory
        if any(pattern in file_path for pattern in 
               ['.archive/', '.deprecated/', '_backup/']):
            # Apply 50% penalty to archived items
            result.score *= 0.5
            result.payload['archive_penalty'] = 0.5
        
        # Check if file has archive indicators
        if any(pattern in file_path for pattern in 
               ['.old', '.backup', '.deprecated']):
            # Apply 30% penalty for backup files
            result.score *= 0.7
            result.payload['backup_penalty'] = 0.7
    
    return sorted(results, key=lambda x: x.score, reverse=True)
```

### 5. MCP Tool Enhancement

Add new parameters to search tools:

```python
types.Tool(
    name="graphrag_hybrid_search",
    description="Hybrid search combining vector similarity and graph relationships",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 5
            },
            "include_graph_context": {
                "type": "boolean",
                "description": "Include related graph nodes",
                "default": True
            },
            "max_hops": {
                "type": "integer",
                "description": "Maximum graph traversal depth",
                "default": 2
            },
            "exclude_archived": {
                "type": "boolean",
                "description": "Exclude archived/deprecated code from results",
                "default": True
            },
            "include_patterns": {
                "type": "array",
                "description": "Additional patterns to include (overrides exclusions)",
                "items": {"type": "string"}
            }
        },
        "required": ["query"]
    }
)
```

### 6. Re-ranking Strategy

Based on 2025 research, we implement a multi-stage re-ranking approach:

#### Reciprocal Rank Fusion (RRF)
- **Formula**: `RRF Score = Σ(1 / (rank + k))` where k=60
- **Benefits**: No tuning required, robust across different data sources
- **Sources**: Combines vector search, keyword search, and graph relationships

#### Dynamic Weighting Factors

1. **Recency Boost** (Time-based decay):
   - Today: 1.5x boost
   - This week: 1.2x boost  
   - This month: 1.0x (neutral)
   - Older: Exponential decay to 0.5x

2. **Archive Penalty** (Path-based):
   - `.archive/` directories: 0.5x penalty
   - `.deprecated/` directories: 0.5x penalty
   - `.backup` files: 0.7x penalty

3. **Source Confidence** (Method-based):
   - Exact path match: 2.0x boost
   - High vector similarity (>0.9): 1.5x boost
   - Graph relationship: 1.2x boost
   - Keyword match: 1.0x (baseline)

#### Two-Stage Retrieval
1. **Stage 1**: Fast retrieval (get 3x limit from each source)
2. **Stage 2**: Intelligent re-ordering with RRF and dynamic weights

### 7. Configuration in .mcp.json

```json
{
  "mcpServers": {
    "neural-tools": {
      "env": {
        "GRAPHRAG_EXCLUDE_ARCHIVED": "true",
        "GRAPHRAG_IGNORE_FILE": ".graphragignore",
        "GRAPHRAG_DEFAULT_EXCLUSIONS": "true",
        "GRAPHRAG_ARCHIVE_PATTERNS": ".archive/,.deprecated/,_backup/"
      }
    }
  }
}
```

## Benefits

### Immediate Benefits
- **Cleaner search results** - No more mixing active and archived code
- **Better code understanding** - Claude instances see only relevant code
- **Faster indexing** - Skip unnecessary archived files
- **Reduced vector storage** - Don't store embeddings for excluded files

### Long-term Benefits
- **Gitignore compatibility** - Familiar pattern syntax
- **Flexible configuration** - Project-specific exclusions
- **Query-time control** - Can still search archives when needed
- **Metadata preservation** - Archive status tracked for audit

## Implementation Plan

### Phase 1: Core Exclusion System (Day 1)
1. Create `ExclusionManager` class
2. Implement `.graphragignore` parser
3. Add default exclusion patterns
4. Test pattern matching

### Phase 2: Indexer Integration (Day 2)
1. Integrate exclusion checks in indexer
2. Add archive metadata to vectors
3. Skip excluded files during indexing
4. Update indexing logs

### Phase 3: Search Enhancement (Day 3)
1. Add `exclude_archived` parameter to search tools
2. Implement query-time filtering
3. Update MCP tool definitions
4. Test with real queries

### Phase 4: Documentation & Testing (Day 4)
1. Create `.graphragignore` template
2. Document exclusion patterns
3. Add unit tests
4. Performance testing

## Consequences

### Positive
- Dramatically cleaner search results
- Better developer experience
- Reduced indexing time and storage
- Industry-standard pattern syntax
- Flexible per-project configuration

### Negative
- Additional configuration file to maintain
- Slight performance overhead for pattern matching
- Need to educate users about exclusion patterns
- Potential for over-exclusion if patterns too broad

### Neutral
- Follows gitignore conventions (familiar to developers)
- Can be disabled if needed
- Archives still searchable with flag

## Alternatives Considered

### 1. Hard-coded Exclusions
- **Pros**: Simple, no configuration needed
- **Cons**: Not flexible, requires code changes
- **Rejected**: Too rigid for diverse projects

### 2. Database-level Filtering
- **Pros**: No file parsing needed
- **Cons**: Archives still get indexed (waste resources)
- **Rejected**: Inefficient use of storage

### 3. Separate Archive Collections
- **Pros**: Complete separation
- **Cons**: Complex migration, duplicate schemas
- **Rejected**: Too complex for the problem

## Testing Strategy

```python
def test_exclusion_patterns():
    """Test exclusion pattern matching"""
    manager = ExclusionManager(Path("/project"))
    
    # Should exclude
    assert manager.should_exclude(Path(".archive/old_code.py"))
    assert manager.should_exclude(Path("src_backup/file.js"))
    assert manager.should_exclude(Path("node_modules/package/index.js"))
    
    # Should include
    assert not manager.should_exclude(Path("src/main.py"))
    assert not manager.should_exclude(Path("neural-tools/server.py"))
    
def test_search_filtering():
    """Test search result filtering"""
    results = [
        {"file_path": "src/main.py", "content": "..."},
        {"file_path": ".archive/old.py", "content": "..."},
        {"file_path": "neural-tools/active.py", "content": "..."}
    ]
    
    filtered = exclusion_manager.filter_search_results(results)
    assert len(filtered) == 2
    assert all(".archive" not in r["file_path"] for r in filtered)
```

## Monitoring & Metrics

Track effectiveness with:
- **Exclusion rate**: % of files excluded during indexing
- **Search quality**: User feedback on result relevance
- **Performance impact**: Time added by pattern matching
- **Storage savings**: Vectors not created for excluded files

## Migration Path

1. **Deploy with defaults** - Works immediately with built-in patterns
2. **Add `.graphragignore`** - Projects can customize as needed
3. **Re-index with exclusions** - Clean up existing vector stores
4. **Monitor and adjust** - Refine patterns based on usage

## Example `.graphragignore` for React Project

```gitignore
# React project .graphragignore

:include:.gitignore

# Build outputs
build/
dist/
.next/
out/

# Dependencies
node_modules/
.pnp/
.pnp.js

# Testing
coverage/
.nyc_output/

# Archives and backups
.archive/
.deprecated/
*_backup/
*.backup
*.old

# Storybook
storybook-static/
.storybook-out/

# IDE
.idea/
.vscode/
*.swp

# OS files
.DS_Store
Thumbs.db

# Don't exclude (re-include with !)
!src/
!components/
!pages/
!public/
```

## References

- GitHub Copilot Semantic Search (2025) - Instant indexing approach
- CocoIndex RAG patterns - Excluded patterns for code indexing
- Semgrep .semgrepignore - Include directive pattern
- VS Code search.exclude - IDE-level exclusion patterns
- Tree-sitter chunking - Syntax-aware code processing
- OpenSearch 2.19 (Feb 2025) - RRF implementation for hybrid search
- Azure AI Search - RRF for parallel query merging
- Elastic RRF - Default k=60 parameter recommendation
- NVIDIA Re-ranking Pipelines - Two-stage retrieval architecture
- Assembled.com - RRF outperforming complex weighting methods

## Decision Outcome

Implement the `.graphragignore` exclusion system with Reciprocal Rank Fusion (RRF) re-ranking and dynamic weighting. The system will:

1. **Exclude** archived/deprecated code by default using `.graphragignore` patterns
2. **Re-rank** results using RRF to combine multiple search sources (vector, keyword, graph)
3. **Boost** recent files and **penalize** archived content that isn't excluded
4. **Provide options** to include archived results when explicitly needed

This addresses both the noise problem (archived code in results) and the relevance problem (all results treated equally).

**Target: Complete implementation in 4 days, basic exclusion working in 1 day, RRF re-ranking in 2 days**