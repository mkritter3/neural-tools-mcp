# ADR-0033: Dynamic Workspace Detection for Multi-Project Support

**Status:** Accepted  
**Date:** 2025-09-12  
**Authors:** Claude L9 Engineering Team  
**Context:** MCP Protocol Limitations, Multi-Project Architecture

## Executive Summary

Replace static `cwd`-based project detection with a dynamic workspace detection system that actively discovers or receives explicit project context, enabling true multi-project support within MCP protocol constraints.

## Problem Statement

### Current State
The MCP server cannot reliably detect which project a user is working on because:

1. **Hardcoded Working Directory**: Both global (`~/.claude/mcp_config.json`) and local (`.mcp.json`) configs specify:
   ```json
   "cwd": "/Users/mkr/local-coding/claude-l9-template"
   ```

2. **MCP Protocol Limitation**: Claude launches MCP servers from its own location, not the user's project directory

3. **Broken Multi-Project Support**: All projects are incorrectly identified as "claude-l9-template", preventing:
   - ADR-0030's multi-container orchestration from working
   - Proper project isolation (ADR-0032)
   - Correct indexer container spawning

### Evidence
- Opening Claude from `/Users/mkr/local-coding/neural-novelist/` still detects project as "claude-l9-template"
- The IndexerOrchestrator never spawns project-specific containers
- Manual workarounds required (starting `l9-project-indexer` manually)

## Decision

Implement a **Dynamic Workspace Detection System** that:
1. Provides explicit `set_project_context` tool for users
2. Auto-detects project from usage patterns
3. Maintains project context across tool invocations
4. Falls back gracefully when detection fails

## Detailed Design

### Architecture Components

```
┌─────────────────────┐
│   Claude Desktop    │
│  (Any Directory)    │
└──────────┬──────────┘
           │ MCP Protocol
           ▼
┌─────────────────────────────────────┐
│         MCP Server                  │
│  ┌─────────────────────────────┐   │
│  │  Workspace Detection Layer  │   │
│  ├─────────────────────────────┤   │
│  │ • set_project_context tool  │   │
│  │ • Auto-detection heuristics │   │
│  │ • Context persistence        │   │
│  │ • Project registry          │   │
│  └─────────────────────────────┘   │
│           │                         │
│           ▼                         │
│  ┌─────────────────────────────┐   │
│  │    Project Container Map    │   │
│  │  neural-novelist → container│   │
│  │  claude-l9-template → cont. │   │
│  │  eventfully-yours → container│  │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Implementation Details

#### 1. Project Context Manager (New Component)

```python
# neural-tools/src/servers/services/project_context_manager.py

import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import asyncio

class ProjectContextManager:
    """
    Manages dynamic project context detection and switching.
    Implements ADR-0033 for multi-project support.
    """
    
    def __init__(self):
        self.current_project: Optional[str] = None
        self.current_project_path: Optional[Path] = None
        self.project_registry: Dict[str, Path] = {}
        self.detection_hints: List[str] = []  # Recent file accesses
        self.last_activity: Dict[str, datetime] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load known projects from persistent storage"""
        registry_path = Path.home() / ".claude" / "project_registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                data = json.load(f)
                self.project_registry = {
                    k: Path(v) for k, v in data.items()
                }
    
    def _save_registry(self):
        """Persist project registry"""
        registry_path = Path.home() / ".claude" / "project_registry.json"
        registry_path.parent.mkdir(exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump({
                k: str(v) for k, v in self.project_registry.items()
            }, f, indent=2)
    
    async def set_project(self, path: str) -> Dict:
        """
        Explicitly set the active project.
        
        Args:
            path: Absolute path to project directory
            
        Returns:
            Project info including name and detection method
        """
        project_path = Path(path).resolve()
        
        # Validate path
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {path}")
        if not project_path.is_dir():
            raise ValueError(f"Project path is not a directory: {path}")
        
        # Detect project name
        project_name = self._detect_project_name(project_path)
        
        # Update state
        self.current_project = project_name
        self.current_project_path = project_path
        self.project_registry[project_name] = project_path
        self.last_activity[project_name] = datetime.now()
        self._save_registry()
        
        return {
            "project": project_name,
            "path": str(project_path),
            "method": "explicit",
            "timestamp": datetime.now().isoformat()
        }
    
    def _detect_project_name(self, path: Path) -> str:
        """
        Detect project name from path using multiple strategies.
        
        Priority:
        1. .git directory name (from git config)
        2. package.json name field
        3. Directory name (sanitized)
        """
        # Try git config
        git_config = path / ".git" / "config"
        if git_config.exists():
            # Parse git remote URL for project name
            # This is simplified - real implementation would be more robust
            pass
        
        # Try package.json
        package_json = path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    if "name" in data:
                        return self._sanitize_name(data["name"])
            except:
                pass
        
        # Fall back to directory name
        return self._sanitize_name(path.name)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize project name for use as identifier"""
        import re
        # Replace non-alphanumeric with underscore, lowercase
        sanitized = re.sub(r'[^a-z0-9-]', '_', name.lower())
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or "default"
    
    async def detect_project(self) -> Dict:
        """
        Auto-detect active project using heuristics.
        
        Detection strategies:
        1. Most recently accessed project
        2. Project containing recently accessed files
        3. First project in registry
        4. Fall back to "default"
        """
        # Strategy 1: Most recent activity
        if self.last_activity:
            most_recent = max(
                self.last_activity.items(),
                key=lambda x: x[1]
            )
            project_name = most_recent[0]
            if project_name in self.project_registry:
                self.current_project = project_name
                self.current_project_path = self.project_registry[project_name]
                return {
                    "project": project_name,
                    "path": str(self.current_project_path),
                    "method": "recent_activity",
                    "confidence": 0.8
                }
        
        # Strategy 2: Check detection hints (files recently accessed)
        for hint_path in self.detection_hints[-10:]:  # Last 10 hints
            hint = Path(hint_path)
            for project_name, project_path in self.project_registry.items():
                try:
                    hint.relative_to(project_path)
                    # File is within this project
                    self.current_project = project_name
                    self.current_project_path = project_path
                    self.last_activity[project_name] = datetime.now()
                    return {
                        "project": project_name,
                        "path": str(project_path),
                        "method": "file_access_pattern",
                        "confidence": 0.9
                    }
                except ValueError:
                    continue
        
        # Strategy 3: Scan common directories
        common_dirs = [
            Path.home() / "local-coding",
            Path.home() / "projects",
            Path.home() / "code",
            Path.cwd()  # Current directory as last resort
        ]
        
        for base_dir in common_dirs:
            if base_dir.exists():
                for child in base_dir.iterdir():
                    if child.is_dir() and not child.name.startswith('.'):
                        # Check if it's a project (has git, package.json, etc.)
                        if (child / ".git").exists() or \
                           (child / "package.json").exists() or \
                           (child / "requirements.txt").exists():
                            project_name = self._detect_project_name(child)
                            self.project_registry[project_name] = child
                            self._save_registry()
        
        # Strategy 4: Use first registered project
        if self.project_registry:
            project_name = list(self.project_registry.keys())[0]
            self.current_project = project_name
            self.current_project_path = self.project_registry[project_name]
            return {
                "project": project_name,
                "path": str(self.current_project_path),
                "method": "first_registered",
                "confidence": 0.5
            }
        
        # Strategy 5: Fall back to default
        self.current_project = "default"
        self.current_project_path = Path.cwd()
        return {
            "project": "default",
            "path": str(self.current_project_path),
            "method": "fallback",
            "confidence": 0.1
        }
    
    def add_hint(self, file_path: str):
        """Add a file access hint for project detection"""
        self.detection_hints.append(file_path)
        # Keep only last 100 hints
        if len(self.detection_hints) > 100:
            self.detection_hints = self.detection_hints[-100:]
    
    async def get_current_project(self) -> Dict:
        """Get current project context, auto-detecting if needed"""
        if self.current_project:
            return {
                "project": self.current_project,
                "path": str(self.current_project_path),
                "method": "cached"
            }
        else:
            return await self.detect_project()
    
    async def list_projects(self) -> List[Dict]:
        """List all known projects"""
        projects = []
        for name, path in self.project_registry.items():
            projects.append({
                "name": name,
                "path": str(path),
                "last_activity": self.last_activity.get(name, "never"),
                "is_current": name == self.current_project
            })
        return sorted(projects, key=lambda x: x["name"])
```

#### 2. MCP Tool Integration

```python
# Add to neural_server_stdio.py

# Global project context manager
PROJECT_CONTEXT = ProjectContextManager()

@server.tool()
async def set_project_context(path: str = None) -> List[types.TextContent]:
    """
    Set or auto-detect the active project for this Claude session.
    
    Args:
        path: Optional absolute path to project directory.
              If not provided, will auto-detect from recent activity.
    
    Returns:
        Project information including name and detection method
    """
    global PROJECT_CONTEXT
    
    try:
        if path:
            # Explicit project setting
            result = await PROJECT_CONTEXT.set_project(path)
            
            # Trigger indexer for new project
            container = await get_project_container(result["project"])
            if hasattr(container, 'ensure_indexer_running'):
                await container.ensure_indexer_running(result["path"])
            
            return [types.TextContent(
                type="text",
                text=f"✅ Project context set to: {result['project']}\n"
                     f"Path: {result['path']}\n"
                     f"Indexer will start processing this project."
            )]
        else:
            # Auto-detection
            result = await PROJECT_CONTEXT.detect_project()
            
            confidence = result.get("confidence", 0)
            if confidence > 0.7:
                return [types.TextContent(
                    type="text",
                    text=f"🔍 Auto-detected project: {result['project']}\n"
                         f"Path: {result['path']}\n"
                         f"Detection method: {result['method']}\n"
                         f"Confidence: {confidence:.0%}"
                )]
            else:
                # Low confidence - ask user to confirm
                projects = await PROJECT_CONTEXT.list_projects()
                project_list = "\n".join([
                    f"  • {p['name']} ({p['path']})"
                    for p in projects
                ])
                
                return [types.TextContent(
                    type="text",
                    text=f"⚠️ Project detection uncertain.\n\n"
                         f"Detected: {result['project']} (confidence: {confidence:.0%})\n\n"
                         f"Known projects:\n{project_list}\n\n"
                         f"Please use set_project_context with explicit path to set project."
                )]
                
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Error setting project context: {str(e)}"
        )]

@server.tool()
async def list_projects() -> List[types.TextContent]:
    """
    List all known projects and their status.
    
    Returns:
        List of projects with paths and activity information
    """
    global PROJECT_CONTEXT
    
    projects = await PROJECT_CONTEXT.list_projects()
    
    if not projects:
        return [types.TextContent(
            type="text",
            text="No projects registered yet. Use set_project_context to add a project."
        )]
    
    output = "📁 Known Projects:\n\n"
    for proj in projects:
        marker = "➤" if proj["is_current"] else " "
        output += f"{marker} {proj['name']}\n"
        output += f"   Path: {proj['path']}\n"
        if proj["last_activity"] != "never":
            output += f"   Last active: {proj['last_activity']}\n"
        output += "\n"
    
    return [types.TextContent(type="text", text=output)]

# Update get_project_context to use dynamic detection
async def get_project_context(params: dict) -> tuple:
    """Get project context with dynamic detection"""
    global PROJECT_CONTEXT
    
    # Check if project explicitly provided in params
    if "project" in params:
        project_name = params["project"]
    else:
        # Use dynamic detection
        context = await PROJECT_CONTEXT.get_current_project()
        project_name = context["project"]
        
        # Add logging for debugging
        logger.info(f"🎯 Detected project: {project_name} via {context.get('method', 'unknown')}")
    
    # Get or create container for project
    container = await get_project_container(project_name)
    retriever = await get_project_retriever(project_name)
    
    return project_name, container, retriever
```

#### 3. Update All Existing Tools

```python
# Example: Update semantic_code_search
async def semantic_code_search_impl(query: str, limit: int = 10) -> List[types.TextContent]:
    try:
        # Use dynamic project detection
        project_name, container, retriever = await get_project_context({})
        
        # Add hint for project detection
        global PROJECT_CONTEXT
        # If we're searching for files, those paths become hints
        
        # ... rest of implementation remains the same
```

### Migration Strategy

#### Phase 1: Add Detection Layer (Non-Breaking)
1. Deploy ProjectContextManager
2. Add new tools: `set_project_context`, `list_projects`
3. Keep existing behavior as fallback

#### Phase 2: Integrate with Tools
1. Update `get_project_context` to use dynamic detection
2. Add hint collection to file-accessing tools
3. Test with multiple projects

#### Phase 3: Update Orchestrator
1. Connect IndexerOrchestrator to dynamic project detection
2. Spawn containers based on detected project
3. Remove manual workarounds

#### Phase 4: Clean Up
1. Remove hardcoded `cwd` from configs (optional)
2. Document new workflow
3. Update global MCP installation

### User Experience

#### Explicit Project Setting
```
User: "I'm working on Neural Novelist"
Claude: [Calls set_project_context("/Users/mkr/local-coding/neural-novelist")]
Result: "✅ Project context set to: neural-novelist"
```

#### Auto-Detection
```
User: "Search for the main character class"
Claude: [Auto-detects from recent activity or prompts if uncertain]
Result: Searches in the correct project
```

#### Project Switching
```
User: "Now let's work on the template project"
Claude: [Calls set_project_context("/Users/mkr/local-coding/claude-l9-template")]
Result: Seamlessly switches context
```

## Testing Strategy

### Unit Tests
- ProjectContextManager detection strategies
- Name sanitization
- Registry persistence

### Integration Tests
- Project switching with IndexerOrchestrator
- Tool execution with correct project context
- Multi-project data isolation

### End-to-End Tests
1. Open Claude from any directory
2. Set project context explicitly
3. Verify correct indexer spawns
4. Search returns project-specific results
5. Switch projects
6. Verify isolation maintained

## Consequences

### Positive
- **True multi-project support** without config changes
- **User control** over project context
- **Smart defaults** reduce friction
- **Works with existing MCP protocol** constraints
- **Enables ADR-0030** orchestration to function
- **Future-proof** for MCP enhancements

### Negative
- **Additional complexity** in project detection logic
- **First-use friction** until projects are registered
- **Potential for incorrect auto-detection** (mitigated by confidence scores)

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Wrong project detected | Confidence scoring, explicit confirmation for low confidence |
| Performance overhead | Cache current project, lazy detection |
| State persistence issues | JSON registry with atomic writes |
| User confusion | Clear feedback, list_projects tool |

## Alternatives Considered

### 1. Per-Project MCP Servers
- **Pros**: True isolation, simple project identification
- **Cons**: Config complexity, resource overhead, poor UX
- **Rejected**: Too much manual configuration burden

### 2. Environment Variable from Claude
- **Pros**: Simple if Claude supported it
- **Cons**: Requires Claude changes, not currently possible
- **Rejected**: Outside our control

### 3. Watching File System for .claude-project Files
- **Pros**: Explicit project markers
- **Cons**: Requires user action, file pollution
- **Rejected**: Poor developer experience

## Decision Outcome

Implement the Dynamic Workspace Detection system as specified. This provides the best balance of:
- User control (explicit setting)
- Convenience (auto-detection)
- Compatibility (works with current MCP)
- Future-proofing (ready for MCP improvements)

## Implementation Checklist

- [ ] Create ProjectContextManager class
- [ ] Add set_project_context tool
- [ ] Add list_projects tool
- [ ] Update get_project_context function
- [ ] Add hint collection to file tools
- [ ] Integrate with IndexerOrchestrator
- [ ] Add project registry persistence
- [ ] Update all tools to use dynamic context
- [ ] Test multi-project switching
- [ ] Update documentation
- [ ] Deploy to global MCP

## References

- ADR-0030: Multi-Container Indexer Orchestration
- ADR-0032: Complete Data Isolation
- MCP Protocol Specification 2025-06-18
- Issue: Project detection fails with hardcoded cwd

---

**Confidence: 95%**  
**Assumptions**: 
- MCP protocol will maintain backward compatibility
- Users willing to explicitly set project context when auto-detection fails
- File access patterns are good indicators of active project