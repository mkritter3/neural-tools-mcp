"""
Multi-Project Neural Indexer
Implements ADR-0027: True multi-project indexing with auto-discovery
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Set, Optional, List
from dataclasses import dataclass
from watchdog.observers import Observer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from servers.services.indexer_service import IncrementalIndexer
from servers.services.service_container import ServiceContainer
from servers.services.collection_config import CollectionManager, CollectionType

logger = logging.getLogger(__name__)

@dataclass
class ProjectInfo:
    """Information about a discovered project"""
    name: str
    path: Path
    has_git: bool = False
    has_package_json: bool = False
    has_pyproject: bool = False
    has_cargo_toml: bool = False
    
    @property
    def is_valid(self) -> bool:
        """Check if this is a valid project to index"""
        return any([self.has_git, self.has_package_json, self.has_pyproject, self.has_cargo_toml])


class MultiProjectIndexer:
    """
    Manages multiple project indexers concurrently.
    Auto-discovers projects and routes data to correct collections.
    """
    
    def __init__(self, base_path: str = "/workspace", container: ServiceContainer = None):
        self.base_path = Path(base_path)
        self.container = container or ServiceContainer()
        self.project_indexers: Dict[str, IncrementalIndexer] = {}
        self.project_watchers: Dict[str, Observer] = {}
        self.discovered_projects: Set[str] = set()
        
        # Configuration
        self.auto_discover = os.getenv('AUTO_DISCOVER', 'true').lower() == 'true'
        self.project_patterns = os.getenv('PROJECT_PATTERNS', '.git,package.json,pyproject.toml,Cargo.toml').split(',')
        
        logger.info(f"MultiProjectIndexer initialized with base_path={base_path}, auto_discover={self.auto_discover}")
    
    async def start(self):
        """Start the multi-project indexer"""
        logger.info("Starting Multi-Project Indexer...")
        
        # Initialize shared container
        # Note: initialize() is sync and returns bool
        initialized = self.container.initialize()
        if not initialized:
            logger.warning("Service container initialization returned False")
        
        if self.auto_discover:
            await self.discover_projects()
        
        # Start monitoring discovered projects
        for project_name in self.discovered_projects:
            await self.start_project_indexer(project_name)
        
        logger.info(f"Multi-Project Indexer started with {len(self.discovered_projects)} projects")
    
    async def discover_projects(self) -> List[ProjectInfo]:
        """Auto-discover all projects under base_path"""
        logger.info(f"Discovering projects under {self.base_path}...")
        projects = []
        
        # Check if base_path itself is a project
        base_project = self._analyze_project(self.base_path)
        if base_project and base_project.is_valid:
            projects.append(base_project)
            self.discovered_projects.add(base_project.name)
            logger.info(f"Found project at base path: {base_project.name}")
        
        # Check subdirectories
        if self.base_path.is_dir():
            for path in self.base_path.iterdir():
                if path.is_dir() and not path.name.startswith('.'):
                    project_info = self._analyze_project(path)
                    if project_info and project_info.is_valid:
                        projects.append(project_info)
                        self.discovered_projects.add(project_info.name)
                        logger.info(f"Discovered project: {project_info.name} at {project_info.path}")
        
        logger.info(f"Discovered {len(projects)} valid projects")
        return projects
    
    def _analyze_project(self, path: Path) -> Optional[ProjectInfo]:
        """Analyze a directory to determine if it's a valid project"""
        if not path.is_dir():
            return None
        
        project = ProjectInfo(
            name=self._sanitize_project_name(path.name),
            path=path,
            has_git=(path / '.git').exists(),
            has_package_json=(path / 'package.json').exists(),
            has_pyproject=(path / 'pyproject.toml').exists(),
            has_cargo_toml=(path / 'Cargo.toml').exists()
        )
        
        return project if project.is_valid else None
    
    def _sanitize_project_name(self, name: str) -> str:
        """Sanitize project name for use in collection names"""
        # Keep hyphens and underscores, they're valid in collection names
        import re
        # Only replace invalid characters, keep alphanumeric, hyphens, and underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        return sanitized.lower()
    
    async def add_project(self, project_name: str, project_path: str):
        """Manually add a project to monitor"""
        project_path = Path(project_path)
        
        if not project_path.exists():
            logger.error(f"Project path does not exist: {project_path}")
            return
        
        sanitized_name = self._sanitize_project_name(project_name)
        
        if sanitized_name in self.project_indexers:
            logger.warning(f"Project {sanitized_name} already being indexed")
            return
        
        logger.info(f"Adding project: {sanitized_name} at {project_path}")
        
        self.discovered_projects.add(sanitized_name)
        await self.start_project_indexer(sanitized_name, str(project_path))
    
    async def start_project_indexer(self, project_name: str, project_path: str = None):
        """Start indexer for a specific project"""
        if project_name in self.project_indexers:
            logger.warning(f"Indexer for {project_name} already running")
            return
        
        # Determine project path
        if project_path is None:
            # Try to find project in discovered projects
            for path in self.base_path.iterdir():
                if self._sanitize_project_name(path.name) == project_name:
                    project_path = str(path)
                    break
        
        if project_path is None:
            # Default to subdirectory of base_path
            project_path = str(self.base_path / project_name)
        
        logger.info(f"Starting indexer for project {project_name} at {project_path}")
        
        try:
            # Create project-specific indexer
            indexer = IncrementalIndexer(
                project_path=project_path,
                project_name=project_name,
                container=self.container  # Share container across projects
            )
            
            # Ensure collection exists
            collection_name = f"project_{project_name}_code"
            await self._ensure_collection(collection_name)
            
            # Store indexer
            self.project_indexers[project_name] = indexer
            
            # Start initial indexing if needed
            if os.getenv('INITIAL_INDEX', 'true').lower() == 'true':
                logger.info(f"Starting initial index for {project_name}...")
                await indexer.initial_index()
            
            # Start watching for changes
            await indexer.start_watching()
            
            logger.info(f"Indexer started for project {project_name}")
            
        except Exception as e:
            logger.error(f"Failed to start indexer for {project_name}: {e}")
            # Remove from active indexers if failed
            if project_name in self.project_indexers:
                del self.project_indexers[project_name]
    
    async def _ensure_collection(self, collection_name: str):
        """Ensure Qdrant collection exists for project"""
        try:
            # Get collections list (returns list of collection names)
            collections = await self.container.qdrant.get_collections()
            
            if collection_name not in collections:
                logger.info(f"Creating collection: {collection_name}")
                
                # Create collection with proper vector configuration
                from qdrant_client.models import Distance, VectorParams
                
                # Use the qdrant_client synchronously
                self.container.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=768,  # Nomic embedding dimension
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"Collection {collection_name} created successfully")
            else:
                logger.info(f"Collection {collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection {collection_name}: {e}")
    
    async def stop(self):
        """Stop all project indexers"""
        logger.info("Stopping Multi-Project Indexer...")
        
        for project_name, indexer in self.project_indexers.items():
            try:
                logger.info(f"Stopping indexer for {project_name}")
                await indexer.stop()
            except Exception as e:
                logger.error(f"Error stopping indexer for {project_name}: {e}")
        
        self.project_indexers.clear()
        logger.info("Multi-Project Indexer stopped")
    
    async def get_status(self) -> Dict:
        """Get status of all project indexers"""
        status = {
            'base_path': str(self.base_path),
            'auto_discover': self.auto_discover,
            'projects': {}
        }
        
        for project_name, indexer in self.project_indexers.items():
            try:
                metrics = indexer.get_metrics()
                status['projects'][project_name] = {
                    'status': 'running',
                    'files_processed': metrics.get('files_processed', 0),
                    'files_queued': metrics.get('queue_depth', 0),
                    'embeddings_created': metrics.get('embeddings_created', 0)
                }
            except Exception as e:
                status['projects'][project_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return status


class ProjectDetector:
    """Detect project from file path - used by MCP to route queries"""
    
    @staticmethod
    def detect_from_path(file_path: str) -> str:
        """Extract project name from path"""
        path = Path(file_path)
        
        # Strategy 1: Look for git root
        git_root = ProjectDetector._find_git_root(path)
        if git_root:
            return ProjectDetector._sanitize_name(git_root.name)
        
        # Strategy 2: Look for project markers
        for parent in path.parents:
            if (parent / "package.json").exists() or \
               (parent / "pyproject.toml").exists() or \
               (parent / "Cargo.toml").exists():
                return ProjectDetector._sanitize_name(parent.name)
        
        # Strategy 3: Use immediate parent of workspace
        if "/workspace/" in str(path):
            parts = Path(file_path).parts
            try:
                workspace_idx = parts.index("workspace")
                if workspace_idx + 1 < len(parts):
                    return ProjectDetector._sanitize_name(parts[workspace_idx + 1])
            except ValueError:
                pass
        
        # Strategy 4: Use current working directory name
        return ProjectDetector._sanitize_name(Path.cwd().name)
    
    @staticmethod
    def _find_git_root(path: Path) -> Optional[Path]:
        """Find the git root directory"""
        current = path if path.is_dir() else path.parent
        
        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent
        
        return None
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize project name"""
        import re
        # Keep hyphens and underscores
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name).lower()


async def main():
    """Test the multi-project indexer"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with local paths
    indexer = MultiProjectIndexer(
        base_path=os.getenv('BASE_WORKSPACE', '/workspace')
    )
    
    await indexer.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(10)
            status = await indexer.get_status()
            logger.info(f"Status: {status}")
    except KeyboardInterrupt:
        await indexer.stop()


if __name__ == "__main__":
    asyncio.run(main())