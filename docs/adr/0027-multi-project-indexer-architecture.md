# ADR-0022: Multi-Project Neural Indexer Architecture

**Status:** PROPOSED  
**Date:** 2025-09-11  
**Context:** Neural GraphRAG System  

## Problem Statement

The current indexer architecture has a critical limitation: it's single-project only. Each project requires its own indexer container with hardcoded PROJECT_NAME, leading to:

1. **Resource waste**: Multiple indexer containers for multiple projects
2. **Configuration complexity**: Each project needs separate Docker configuration
3. **Data isolation issues**: Data ending up in wrong collections (project_default_code vs project_claude-l9-template_code)
4. **Cross-project search impossible**: Can't search across multiple projects
5. **MCP integration broken**: Global neural-tools MCP expects multi-project support

## Current State (BROKEN)

```
┌─────────────────────────────────────┐
│         l9-neural-indexer           │
│    (PROJECT_NAME=claude-l9-template)│
│                                     │
│  Watches: /workspace/               │
│  Stores:  project_default_code      │ ← BUG: Should be project_claude-l9-template_code
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│          Qdrant Collections         │
│                                     │
│  project_default_code: 4291 points  │ ← All data here
│  project_claude-l9-template_code: 0 │ ← Should be here
│  project_neural-novelist_code: N/A  │ ← Doesn't exist!
└─────────────────────────────────────┘
```

## Proposed Solution: True Multi-Project Indexer

### Architecture

```
┌──────────────────────────────────────────────┐
│         Multi-Project Neural Indexer         │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │     Project Detection Engine        │    │
│  │  - Auto-detect from path structure  │    │
│  │  - /projects/PROJECT_NAME/...       │    │
│  │  - Git repo root detection          │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │     Concurrent Project Watchers     │    │
│  │  ┌──────────┐ ┌──────────┐ ┌─────┐ │    │
│  │  │Project A │ │Project B │ │ ... │ │    │
│  │  └──────────┘ └──────────┘ └─────┘ │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  ┌─────────────────────────────────────┐    │
│  │    Collection Routing Engine        │    │
│  │  File → Project → Collection Name   │    │
│  └─────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   Qdrant Collections   │
        │                        │
        │ project_claude-l9-template_code │
        │ project_neural-novelist_code    │
        │ project_my-app_code             │
        └───────────────────────┘
```

### Implementation Plan

#### Phase 1: Fix Collection Naming Bug (IMMEDIATE)
```python
# indexer_service.py
class IncrementalIndexer:
    def __init__(self, project_path: str, project_name: str = None):
        # CRITICAL FIX: Use provided project_name, not "default"
        if project_name and project_name != "default":
            self.project_name = project_name
        else:
            # Auto-detect from path
            self.project_name = self._detect_project_name(project_path)
        
        # Ensure collection name matches
        self.collection_name = f"project_{self.project_name}_code"
        logger.info(f"Indexer using collection: {self.collection_name}")
```

#### Phase 2: Multi-Project Watcher
```python
class MultiProjectIndexer:
    def __init__(self, base_path: str = "/workspace"):
        self.base_path = Path(base_path)
        self.project_indexers = {}  # project_name -> IncrementalIndexer
        self.project_watchers = {}  # project_name -> Observer
        
    async def discover_projects(self):
        """Auto-discover all projects under base_path"""
        for path in self.base_path.iterdir():
            if path.is_dir() and not path.name.startswith('.'):
                # Check if it's a valid project (has .git, package.json, etc.)
                if self._is_valid_project(path):
                    project_name = path.name
                    await self.add_project(project_name, str(path))
    
    async def add_project(self, project_name: str, project_path: str):
        """Add a new project to monitor"""
        if project_name not in self.project_indexers:
            logger.info(f"Adding project: {project_name} at {project_path}")
            
            # Create project-specific indexer
            indexer = IncrementalIndexer(
                project_path=project_path,
                project_name=project_name
            )
            self.project_indexers[project_name] = indexer
            
            # Start watching
            await indexer.start_watching()
            
            # Create collection if needed
            collection_name = f"project_{project_name}_code"
            await self._ensure_collection(collection_name)
```

#### Phase 3: Dynamic Project Detection
```python
class ProjectDetector:
    """Detect project from file path"""
    
    @staticmethod
    def detect_from_path(file_path: str) -> str:
        """Extract project name from path"""
        path = Path(file_path)
        
        # Strategy 1: Look for git root
        git_root = ProjectDetector._find_git_root(path)
        if git_root:
            return git_root.name
        
        # Strategy 2: Look for project markers
        for parent in path.parents:
            if (parent / "package.json").exists() or \
               (parent / "pyproject.toml").exists() or \
               (parent / "Cargo.toml").exists():
                return parent.name
        
        # Strategy 3: Use immediate parent of workspace
        if "/workspace/" in str(path):
            parts = Path(file_path).parts
            workspace_idx = parts.index("workspace")
            if workspace_idx + 1 < len(parts):
                return parts[workspace_idx + 1]
        
        return "default"
```

#### Phase 4: Collection Management
```python
class CollectionManager:
    def __init__(self):
        self.collections = {}  # project_name -> collection_config
        
    def get_collection_name(self, project_name: str, collection_type: str = "code"):
        """Get collection name for project"""
        # ENSURE CONSISTENCY
        return f"project_{project_name}_{collection_type}"
    
    async def ensure_collection_exists(self, project_name: str):
        """Create collection if it doesn't exist"""
        collection_name = self.get_collection_name(project_name)
        
        if collection_name not in await self.qdrant.get_collections():
            await self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance="Cosine")
            )
            logger.info(f"Created collection: {collection_name}")
```

### Configuration

#### Docker Environment
```yaml
# docker-compose.yml
services:
  l9-neural-indexer:
    environment:
      - INDEXER_MODE=multi-project  # New mode
      - BASE_WORKSPACE=/workspace   # Root for all projects
      - AUTO_DISCOVER=true          # Auto-find projects
      - PROJECT_PATTERNS=.git,package.json,pyproject.toml
```

#### MCP Integration
```python
# neural_server_stdio.py
async def get_project_context(arguments: Dict[str, Any]):
    # Auto-detect project from current working directory
    cwd = os.getcwd()
    project_name = Path(cwd).name
    
    # Ensure collection exists
    collection_name = f"project_{project_name}_code"
    
    # Search in correct collection
    return project_name, collection_name
```

## Benefits

1. **Single indexer for all projects**: Resource efficient
2. **Automatic project detection**: No configuration needed
3. **Correct collection routing**: Data goes where it should
4. **Cross-project search**: Can search across projects if needed
5. **MCP compatibility**: Works with global neural-tools

## Migration Path

1. **Immediate Fix**: Patch current indexer to use correct collection name
2. **Re-index**: Move data from project_default_code to project_claude-l9-template_code
3. **Deploy multi-project**: Replace single-project indexer
4. **Auto-discovery**: Let it find all projects automatically

## Success Metrics

- [ ] All project data in correct collections
- [ ] Multiple projects indexed simultaneously
- [ ] MCP can search any project's data
- [ ] No more "GraphRAG not available" errors
- [ ] Cross-project search capabilities

## Risks

- **Performance**: Watching multiple large projects
  - Mitigation: Implement file filters, ignore patterns
- **Memory**: Multiple indexers in one process
  - Mitigation: Shared connection pools, lazy loading
- **Conflicts**: Projects with same name
  - Mitigation: Use full path hash in collection name if needed

## Decision

We will implement a true multi-project indexer that:
1. Auto-discovers projects
2. Routes data to correct collections
3. Supports dynamic project addition
4. Integrates seamlessly with MCP

This fixes the current broken state where neural-novelist has no data and claude-l9-template data is in the wrong collection.

## References

- ADR-0016: MCP Container Connectivity
- ADR-0019: Instance Isolation  
- Current Bug: All data in project_default_code instead of project-specific collections