#!/usr/bin/env python3
"""
Global Neural Memory MCP Server
Auto-detects Claude's current project and provides memory/search tools
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add neural-system to path
sys.path.append(str(Path(__file__).parent.parent / 'neural-system'))

from config_manager import get_config
from project_isolation import ProjectIsolation
from neural_embeddings import CodeSpecificEmbedder
from memory_system import MemorySystem

# MCP imports
try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
    import mcp.types as types
except ImportError:
    print("MCP not available, running in standalone mode", file=sys.stderr)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class ProjectContext:
    """Context information for current project"""
    name: str
    path: str
    container_name: str
    rest_port: int
    grpc_port: int
    collection_prefix: str

class GlobalNeuralMemoryMCP:
    """
    Global MCP server for neural memory across projects
    Auto-detects current project from Claude's working directory
    """
    
    def __init__(self):
        self.server = Server("neural-memory")
        self.embedder = None
        self.memory_systems: Dict[str, MemorySystem] = {}
        self.project_isolation = ProjectIsolation()
        self.current_project: Optional[ProjectContext] = None
        
        # Register MCP tools
        self._register_tools()
        
    async def initialize(self):
        """Initialize the MCP server and detect current project"""
        try:
            # Initialize embedder (shared across projects)
            self.embedder = CodeSpecificEmbedder()
            
            # Auto-detect current project
            await self._detect_current_project()
            
            logger.info(f"🚀 Global Neural Memory MCP initialized")
            if self.current_project:
                logger.info(f"📍 Current project: {self.current_project.name}")
                logger.info(f"🔌 Qdrant: {self.current_project.rest_port}/{self.current_project.grpc_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    async def _detect_current_project(self):
        """Detect current project from working directory"""
        try:
            # Get Claude's current working directory
            cwd = Path.cwd()
            
            # Use config manager to detect project
            config = get_config()
            project_name = config.get_project_name()
            
            # Create project context
            self.current_project = ProjectContext(
                name=project_name,
                path=str(cwd),
                container_name=config.get_container_name(),
                rest_port=config.get_qdrant_config()['port'],
                grpc_port=config.get_qdrant_config(prefer_grpc=True)['port'],
                collection_prefix=f"{project_name}_"
            )
            
            # Initialize memory system for current project
            await self._get_memory_system(project_name)
            
            logger.info(f"📋 Detected project: {project_name} at {cwd}")
            
        except Exception as e:
            logger.error(f"Failed to detect project: {e}")
            self.current_project = None
    
    async def _get_memory_system(self, project_name: str) -> MemorySystem:
        """Get or create memory system for a project"""
        if project_name not in self.memory_systems:
            try:
                # Initialize project-specific memory system
                config = get_config()
                qdrant_config = config.get_qdrant_config(prefer_grpc=True)
                
                memory_system = MemorySystem(
                    qdrant_host=qdrant_config['host'],
                    qdrant_port=qdrant_config['port'],
                    embedder=self.embedder,
                    collection_prefix=config.get_collection_name("")
                )
                
                await memory_system.initialize()
                self.memory_systems[project_name] = memory_system
                
                logger.info(f"💾 Initialized memory system for project: {project_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize memory system for {project_name}: {e}")
                raise
                
        return self.memory_systems[project_name]
    
    def _register_tools(self):
        """Register MCP tools for neural memory operations"""
        
        @self.server.call_tool()
        async def memory_store(content: str, metadata: Optional[Dict] = None) -> List[types.TextContent]:
            """Store content in current project's memory"""
            if not self.current_project:
                return [types.TextContent(type="text", text="❌ No project detected")]
            
            try:
                memory_system = await self._get_memory_system(self.current_project.name)
                
                # Prepare metadata
                if metadata is None:
                    metadata = {}
                metadata.update({
                    "project": self.current_project.name,
                    "timestamp": str(asyncio.get_event_loop().time()),
                    "source": "mcp_store"
                })
                
                # Store memory
                memory_id = await memory_system.store_memory(content, metadata)
                
                return [types.TextContent(
                    type="text", 
                    text=f"✅ Stored memory in project '{self.current_project.name}' with ID: {memory_id}"
                )]
                
            except Exception as e:
                logger.error(f"Failed to store memory: {e}")
                return [types.TextContent(type="text", text=f"❌ Error storing memory: {e}")]
        
        @self.server.call_tool()
        async def memory_recall(query: str, limit: int = 5, similarity_threshold: float = 0.7) -> List[types.TextContent]:
            """Recall memories from current project"""
            if not self.current_project:
                return [types.TextContent(type="text", text="❌ No project detected")]
            
            try:
                memory_system = await self._get_memory_system(self.current_project.name)
                
                # Search memories
                memories = await memory_system.search_memories(
                    query=query,
                    limit=limit,
                    similarity_threshold=similarity_threshold
                )
                
                if not memories:
                    return [types.TextContent(
                        type="text", 
                        text=f"🔍 No relevant memories found in project '{self.current_project.name}' for query: {query}"
                    )]
                
                # Format results
                results = [f"🧠 Found {len(memories)} relevant memories in project '{self.current_project.name}':\n"]
                
                for i, memory in enumerate(memories, 1):
                    score = memory.get('score', 0.0)
                    content = memory.get('payload', {}).get('content', 'No content')
                    metadata = memory.get('payload', {})
                    timestamp = metadata.get('timestamp', 'Unknown time')
                    
                    results.append(f"{i}. Score: {score:.3f} | Time: {timestamp}")
                    results.append(f"   Content: {content[:200]}{'...' if len(content) > 200 else ''}\n")
                
                return [types.TextContent(type="text", text="\n".join(results))]
                
            except Exception as e:
                logger.error(f"Failed to recall memory: {e}")
                return [types.TextContent(type="text", text=f"❌ Error recalling memory: {e}")]
        
        @self.server.call_tool()
        async def memory_recall_global(query: str, limit: int = 10, similarity_threshold: float = 0.6) -> List[types.TextContent]:
            """Recall memories across all projects (when needed for cross-project context)"""
            try:
                # Get all available projects
                available_projects = await self._discover_projects()
                
                all_memories = []
                
                for project_name in available_projects:
                    try:
                        memory_system = await self._get_memory_system(project_name)
                        memories = await memory_system.search_memories(
                            query=query,
                            limit=limit//len(available_projects) + 1,
                            similarity_threshold=similarity_threshold
                        )
                        
                        # Add project info to memories
                        for memory in memories:
                            memory['project_name'] = project_name
                        
                        all_memories.extend(memories)
                        
                    except Exception as e:
                        logger.warning(f"Failed to search project {project_name}: {e}")
                        continue
                
                # Sort by score and limit results
                all_memories.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                all_memories = all_memories[:limit]
                
                if not all_memories:
                    return [types.TextContent(
                        type="text", 
                        text=f"🔍 No relevant memories found across projects for query: {query}"
                    )]
                
                # Format results
                results = [f"🌐 Found {len(all_memories)} relevant memories across {len(available_projects)} projects:\n"]
                
                for i, memory in enumerate(all_memories, 1):
                    score = memory.get('score', 0.0)
                    project = memory.get('project_name', 'Unknown')
                    content = memory.get('payload', {}).get('content', 'No content')
                    metadata = memory.get('payload', {})
                    timestamp = metadata.get('timestamp', 'Unknown time')
                    
                    results.append(f"{i}. [{project}] Score: {score:.3f} | Time: {timestamp}")
                    results.append(f"   Content: {content[:200]}{'...' if len(content) > 200 else ''}\n")
                
                return [types.TextContent(type="text", text="\n".join(results))]
                
            except Exception as e:
                logger.error(f"Failed to recall global memory: {e}")
                return [types.TextContent(type="text", text=f"❌ Error recalling global memory: {e}")]
        
        @self.server.call_tool()
        async def code_search(query: str, file_types: Optional[List[str]] = None, limit: int = 10) -> List[types.TextContent]:
            """Search code in current project"""
            if not self.current_project:
                return [types.TextContent(type="text", text="❌ No project detected")]
            
            try:
                memory_system = await self._get_memory_system(self.current_project.name)
                
                # Search code with specific filters
                results = await memory_system.search_code(
                    query=query,
                    file_types=file_types,
                    limit=limit
                )
                
                if not results:
                    return [types.TextContent(
                        type="text", 
                        text=f"🔍 No code found in project '{self.current_project.name}' for query: {query}"
                    )]
                
                # Format results
                output = [f"💻 Found {len(results)} code snippets in project '{self.current_project.name}':\n"]
                
                for i, result in enumerate(results, 1):
                    file_path = result.get('payload', {}).get('file_path', 'Unknown file')
                    line_number = result.get('payload', {}).get('line_number', 'Unknown line')
                    code_snippet = result.get('payload', {}).get('content', 'No content')
                    score = result.get('score', 0.0)
                    
                    output.append(f"{i}. {file_path}:{line_number} (Score: {score:.3f})")
                    output.append(f"   {code_snippet[:300]}{'...' if len(code_snippet) > 300 else ''}\n")
                
                return [types.TextContent(type="text", text="\n".join(output))]
                
            except Exception as e:
                logger.error(f"Failed to search code: {e}")
                return [types.TextContent(type="text", text=f"❌ Error searching code: {e}")]
        
        @self.server.call_tool()
        async def project_context() -> List[types.TextContent]:
            """Get current project context and available operations"""
            if not self.current_project:
                return [types.TextContent(type="text", text="❌ No project detected")]
            
            try:
                # Get memory system stats
                memory_system = await self._get_memory_system(self.current_project.name)
                stats = await memory_system.get_stats()
                
                # Get available projects for global operations
                available_projects = await self._discover_projects()
                
                context_info = f"""
📍 **Current Project Context**

**Project Details:**
- Name: {self.current_project.name}
- Path: {self.current_project.path}
- Container: {self.current_project.container_name}
- Qdrant Ports: {self.current_project.rest_port} (REST), {self.current_project.grpc_port} (gRPC)

**Memory Statistics:**
- Total memories: {stats.get('total_memories', 0)}
- Code snippets: {stats.get('code_snippets', 0)}
- Last updated: {stats.get('last_updated', 'Unknown')}

**Available Projects for Global Search:**
{', '.join(available_projects)}

**Available MCP Tools:**
- `memory_store`: Store content in current project
- `memory_recall`: Search memories in current project
- `memory_recall_global`: Search across all projects
- `code_search`: Search code in current project
- `project_context`: Get this context information
"""
                
                return [types.TextContent(type="text", text=context_info)]
                
            except Exception as e:
                logger.error(f"Failed to get project context: {e}")
                return [types.TextContent(type="text", text=f"❌ Error getting project context: {e}")]
        
        @self.server.call_tool()
        async def setup_project(enable_hooks: bool = True, enable_auto_indexing: bool = True, 
                              project_path: Optional[str] = None) -> List[types.TextContent]:
            """Setup neural memory system for a new project with optional hooks"""
            
            try:
                # Use provided path or current working directory
                target_path = Path(project_path) if project_path else Path.cwd()
                project_name = target_path.name
                
                # Create project configuration
                claude_dir = target_path / '.claude'
                claude_dir.mkdir(exist_ok=True)
                
                # Generate project-specific configuration
                project_isolation = ProjectIsolation()
                config = project_isolation.get_project_config()
                
                # Save project configuration
                config_path = claude_dir / 'project-config.json'
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Create project-specific Qdrant container
                container_result = project_isolation.ensure_project_container(
                    project_isolation.get_project_context()
                )
                
                setup_results = [
                    f"✅ Project '{project_name}' initialized",
                    f"📁 Path: {target_path}",
                    f"🐳 Container: {config['container_name']}",
                    f"🔌 Ports: REST {config['rest_port']}, gRPC {config['grpc_port']}",
                    f"💾 Storage: {config['storage_path']}"
                ]
                
                # Setup hooks if requested
                if enable_hooks:
                    hooks_result = await self._setup_hooks(claude_dir, enable_auto_indexing)
                    setup_results.extend(hooks_result)
                
                # Setup .mcp.json if it doesn't exist
                mcp_json_path = target_path / '.mcp.json'
                if not mcp_json_path.exists():
                    mcp_config = {
                        "mcpServers": {
                            "neural-memory": {
                                "command": "python3",
                                "args": [str(Path(__file__).absolute())],
                                "env": {
                                    "PYTHONPATH": str(Path(__file__).parent / '.claude' / 'neural-system')
                                }
                            }
                        }
                    }
                    with open(mcp_json_path, 'w') as f:
                        json.dump(mcp_config, f, indent=2)
                    setup_results.append("📋 .mcp.json created for Claude Code integration")
                
                result_text = "🎉 PROJECT SETUP COMPLETE\\n\\n" + "\\n".join(setup_results)
                result_text += "\\n\\n🚀 Ready to use! Try:\\n- Ask Claude to store a memory\\n- Ask Claude to recall project information\\n- Use global search across all projects"
                
                return [types.TextContent(type="text", text=result_text)]
                
            except Exception as e:
                logger.error(f"Failed to setup project: {e}")
                return [types.TextContent(type="text", text=f"❌ Project setup failed: {e}")]
    
    async def _setup_hooks(self, claude_dir: Path, enable_auto_indexing: bool) -> List[str]:
        """Setup Claude Code hooks for automatic indexing and context injection"""
        
        hooks_dir = claude_dir / 'hooks'
        hooks_dir.mkdir(exist_ok=True)
        
        results = []
        
        try:
            # Get hook templates from current system
            template_hooks_dir = Path(__file__).parent / '.claude' / 'hooks'
            
            if template_hooks_dir.exists():
                # Copy hook files
                import shutil
                for hook_file in template_hooks_dir.glob('*.py'):
                    dest_file = hooks_dir / hook_file.name
                    shutil.copy2(hook_file, dest_file)
                    results.append(f"📎 Hook installed: {hook_file.name}")
                
                # Create or update Claude Code settings
                settings_path = claude_dir / 'settings.json'
                settings = {}
                if settings_path.exists():
                    with open(settings_path, 'r') as f:
                        settings = json.load(f)
                
                # Enable hooks in settings
                settings['hooks'] = {
                    'enabled': True,
                    'postToolUse': enable_auto_indexing,  # Auto-index after Edit/Write
                    'sessionStart': True,                 # Load context on session start
                    'userPromptSubmit': True,            # Inject context before prompts
                    'stop': True                         # Store session insights on stop
                }
                
                with open(settings_path, 'w') as f:
                    json.dump(settings, f, indent=2)
                
                results.append("⚙️ Claude Code settings updated for hooks")
                results.append(f"🔄 Auto-indexing: {'enabled' if enable_auto_indexing else 'disabled'}")
                
            else:
                results.append("⚠️ Hook templates not found - hooks skipped")
                
        except Exception as e:
            results.append(f"❌ Hook setup failed: {e}")
            logger.error(f"Hook setup error: {e}")
        
        return results

    async def _discover_projects(self) -> List[str]:
        """Discover available projects by scanning for running containers"""
        try:
            # Use project isolation to discover active projects
            active_projects = await self.project_isolation.list_active_projects()
            return active_projects
        except Exception as e:
            logger.error(f"Failed to discover projects: {e}")
            return [self.current_project.name] if self.current_project else []

async def main():
    """Main entry point for MCP server"""
    mcp_server = GlobalNeuralMemoryMCP()
    
    # Initialize server
    await mcp_server.initialize()
    
    # Run server
    async with mcp_server.server.run() as server:
        await server.wait_for_client()

if __name__ == "__main__":
    asyncio.run(main())