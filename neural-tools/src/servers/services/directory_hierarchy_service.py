#!/usr/bin/env python3
"""
Directory Hierarchy Service - ADR-0047 Phase 2
Implements hierarchical organization with directory summaries for efficient million-file search.
Uses two-phase retrieval: directory summaries first, then file-level details.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DirectorySummary:
    """Summary information for a directory in the hierarchy"""
    path: str
    name: str
    level: int  # Depth in hierarchy (0 = root)
    file_count: int
    total_lines: int
    languages: List[str]
    frameworks: List[str]
    key_concepts: List[str]
    subdirectory_count: int
    last_modified: str
    size_bytes: int
    summary_embedding: Optional[List[float]] = None
    merkle_hash: Optional[str] = None  # For change detection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)

class DirectoryHierarchyService:
    """
    Manages hierarchical directory structures for efficient search across millions of files.
    Implements ADR-0047 Phase 2 optimizations.
    """

    def __init__(self, qdrant_service, neo4j_service, nomic_service):
        self.qdrant_service = qdrant_service
        self.neo4j_service = neo4j_service
        self.nomic_service = nomic_service
        self.directory_cache = {}
        self.merkle_tree = {}

    async def build_directory_hierarchy(self, root_path: str, max_depth: int = 5) -> Dict[str, DirectorySummary]:
        """
        Build hierarchical directory structure with summaries.

        Args:
            root_path: Root directory to analyze
            max_depth: Maximum depth to traverse

        Returns:
            Dictionary mapping paths to DirectorySummary objects
        """
        hierarchy = {}
        root = Path(root_path).resolve()

        async def process_directory(path: Path, level: int) -> DirectorySummary:
            """Process a single directory and create its summary"""
            if level > max_depth:
                return None

            try:
                # Collect directory statistics
                file_count = 0
                total_lines = 0
                languages = set()
                frameworks = set()
                key_concepts = set()
                subdirs = []
                size_bytes = 0
                last_modified = datetime.min

                # Scan directory contents
                for item in path.iterdir():
                    if item.is_file() and not item.name.startswith('.'):
                        file_count += 1
                        size_bytes += item.stat().st_size

                        # Detect language from extension
                        ext = item.suffix.lower()
                        if ext in {'.py', '.pyw'}:
                            languages.add('Python')
                        elif ext in {'.js', '.jsx', '.mjs'}:
                            languages.add('JavaScript')
                        elif ext in {'.ts', '.tsx'}:
                            languages.add('TypeScript')
                        elif ext in {'.java'}:
                            languages.add('Java')
                        elif ext in {'.go'}:
                            languages.add('Go')
                        elif ext in {'.rs'}:
                            languages.add('Rust')
                        elif ext in {'.cpp', '.cc', '.cxx', '.hpp'}:
                            languages.add('C++')

                        # Count lines (quick approximation)
                        if ext in {'.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp'}:
                            try:
                                with open(item, 'r', encoding='utf-8', errors='ignore') as f:
                                    total_lines += sum(1 for _ in f)
                            except:
                                pass

                        # Track modification time
                        mod_time = datetime.fromtimestamp(item.stat().st_mtime)
                        if mod_time > last_modified:
                            last_modified = mod_time

                    elif item.is_dir() and not item.name.startswith('.'):
                        subdirs.append(item)

                # Detect frameworks from special files
                framework_markers = {
                    'package.json': ['Node.js'],
                    'requirements.txt': ['Python'],
                    'pyproject.toml': ['Python'],
                    'Cargo.toml': ['Rust'],
                    'go.mod': ['Go'],
                    'pom.xml': ['Java', 'Maven'],
                    'build.gradle': ['Java', 'Gradle'],
                    'CMakeLists.txt': ['C++', 'CMake'],
                    'Dockerfile': ['Docker'],
                    'docker-compose.yml': ['Docker Compose'],
                    '.graphrag': ['GraphRAG'],
                }

                for marker, fw_list in framework_markers.items():
                    if (path / marker).exists():
                        frameworks.update(fw_list)

                # Extract key concepts from directory name and README
                dir_name = path.name.lower()
                concepts = []

                # Common patterns in directory names
                if 'test' in dir_name:
                    concepts.append('testing')
                if 'src' in dir_name or 'source' in dir_name:
                    concepts.append('source_code')
                if 'doc' in dir_name:
                    concepts.append('documentation')
                if 'config' in dir_name:
                    concepts.append('configuration')
                if 'service' in dir_name:
                    concepts.append('services')
                if 'model' in dir_name:
                    concepts.append('models')
                if 'util' in dir_name or 'helper' in dir_name:
                    concepts.append('utilities')
                if 'api' in dir_name:
                    concepts.append('api')
                if 'frontend' in dir_name or 'ui' in dir_name:
                    concepts.append('frontend')
                if 'backend' in dir_name:
                    concepts.append('backend')

                # Try to read README for more concepts
                readme_path = path / 'README.md'
                if readme_path.exists():
                    try:
                        with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                            readme_content = f.read(500)  # First 500 chars
                            # Extract key words from README (simple approach)
                            words = readme_content.lower().split()
                            tech_keywords = {'api', 'database', 'cache', 'queue', 'stream',
                                           'async', 'realtime', 'graphql', 'rest', 'websocket'}
                            concepts.extend([w for w in words if w in tech_keywords][:3])
                    except:
                        pass

                key_concepts.update(concepts[:5])  # Limit to 5 concepts

                # Create summary
                summary = DirectorySummary(
                    path=str(path),
                    name=path.name,
                    level=level,
                    file_count=file_count,
                    total_lines=total_lines,
                    languages=sorted(list(languages)),
                    frameworks=sorted(list(frameworks)),
                    key_concepts=sorted(list(key_concepts)),
                    subdirectory_count=len(subdirs),
                    last_modified=last_modified.isoformat() if last_modified != datetime.min else "",
                    size_bytes=size_bytes
                )

                # Generate embedding for directory summary
                if self.nomic_service:
                    summary_text = self._create_summary_text(summary)
                    try:
                        embeddings = await self.nomic_service.generate_embeddings([summary_text])
                        if embeddings and len(embeddings) > 0:
                            summary.summary_embedding = embeddings[0]
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for {path}: {e}")

                # Calculate Merkle hash for change detection
                summary.merkle_hash = self._calculate_merkle_hash(summary)

                # Store in hierarchy
                hierarchy[str(path)] = summary

                # Process subdirectories recursively
                for subdir in subdirs:
                    sub_summary = await process_directory(subdir, level + 1)
                    if sub_summary:
                        hierarchy[str(subdir)] = sub_summary

                return summary

            except Exception as e:
                logger.error(f"Error processing directory {path}: {e}")
                return None

        # Start processing from root
        await process_directory(root, 0)

        # Cache the hierarchy
        self.directory_cache[root_path] = hierarchy

        return hierarchy

    def _create_summary_text(self, summary: DirectorySummary) -> str:
        """Create text representation of directory summary for embedding"""
        parts = [
            f"Directory: {summary.name}",
            f"Contains {summary.file_count} files with {summary.total_lines} lines of code"
        ]

        if summary.languages:
            parts.append(f"Languages: {', '.join(summary.languages)}")
        if summary.frameworks:
            parts.append(f"Frameworks: {', '.join(summary.frameworks)}")
        if summary.key_concepts:
            parts.append(f"Concepts: {', '.join(summary.key_concepts)}")

        return " | ".join(parts)

    def _calculate_merkle_hash(self, summary: DirectorySummary) -> str:
        """Calculate Merkle hash for change detection"""
        # Create deterministic string representation
        hash_input = json.dumps({
            'path': summary.path,
            'file_count': summary.file_count,
            'total_lines': summary.total_lines,
            'languages': summary.languages,
            'last_modified': summary.last_modified,
            'size_bytes': summary.size_bytes
        }, sort_keys=True)

        return hashlib.sha256(hash_input.encode()).hexdigest()

    async def store_hierarchy_in_qdrant(self, hierarchy: Dict[str, DirectorySummary],
                                       collection_name: str = None) -> bool:
        """
        Store directory hierarchy in Qdrant for fast retrieval.

        Args:
            hierarchy: Directory hierarchy to store
            collection_name: Optional collection name override

        Returns:
            Success status
        """
        if not collection_name:
            collection_name = f"{self.qdrant_service.collection_prefix}-directories"

        try:
            # Ensure collection exists with proper configuration
            await self.qdrant_service.ensure_collection(
                collection_name=collection_name,
                vector_size=768,  # Nomic embedding size
                enable_hybrid=True,  # Enable BM25S
                quantization=True  # Enable scalar quantization
            )

            # Prepare points for insertion
            points = []
            for path, summary in hierarchy.items():
                if summary.summary_embedding:
                    # Create sparse vector for BM25S
                    summary_text = self._create_summary_text(summary)
                    sparse_vector = await self.qdrant_service._create_sparse_vector(summary_text)

                    point = PointStruct(
                        id=hashlib.sha256(path.encode()).hexdigest()[:16],
                        vector=summary.summary_embedding,
                        payload={
                            **summary.to_dict(),
                            'type': 'directory_summary'
                        }
                    )

                    # Add sparse vector if available
                    if sparse_vector:
                        point.vector = {
                            "dense": summary.summary_embedding,
                            "sparse": sparse_vector
                        }

                    points.append(point)

            # Use incremental upsert for efficiency
            if points:
                result = await self.qdrant_service.incremental_upsert(
                    collection_name=collection_name,
                    points=points,
                    batch_size=100
                )
                logger.info(f"Stored {len(points)} directory summaries in Qdrant")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to store hierarchy in Qdrant: {e}")
            return False

    async def store_hierarchy_in_neo4j(self, hierarchy: Dict[str, DirectorySummary]) -> bool:
        """
        Store directory hierarchy in Neo4j for graph traversal.

        Args:
            hierarchy: Directory hierarchy to store

        Returns:
            Success status
        """
        try:
            # Create directory nodes and relationships
            for path, summary in hierarchy.items():
                # Create or update directory node
                await self.neo4j_service.run_query(
                    """
                    MERGE (d:Directory {path: $path, project: $project})
                    SET d.name = $name,
                        d.level = $level,
                        d.file_count = $file_count,
                        d.total_lines = $total_lines,
                        d.languages = $languages,
                        d.frameworks = $frameworks,
                        d.key_concepts = $key_concepts,
                        d.subdirectory_count = $subdirectory_count,
                        d.last_modified = $last_modified,
                        d.size_bytes = $size_bytes,
                        d.merkle_hash = $merkle_hash
                    """,
                    parameters={
                        'path': summary.path,
                        'project': self.neo4j_service.project_name,
                        'name': summary.name,
                        'level': summary.level,
                        'file_count': summary.file_count,
                        'total_lines': summary.total_lines,
                        'languages': summary.languages,
                        'frameworks': summary.frameworks,
                        'key_concepts': summary.key_concepts,
                        'subdirectory_count': summary.subdirectory_count,
                        'last_modified': summary.last_modified,
                        'size_bytes': summary.size_bytes,
                        'merkle_hash': summary.merkle_hash
                    }
                )

                # Create parent-child relationships
                parent_path = str(Path(summary.path).parent)
                if parent_path in hierarchy and parent_path != summary.path:
                    await self.neo4j_service.run_query(
                        """
                        MATCH (parent:Directory {path: $parent_path, project: $project})
                        MATCH (child:Directory {path: $child_path, project: $project})
                        MERGE (parent)-[:CONTAINS_DIR]->(child)
                        """,
                        parameters={
                            'parent_path': parent_path,
                            'child_path': summary.path,
                            'project': self.neo4j_service.project_name
                        }
                    )

            logger.info(f"Stored {len(hierarchy)} directory nodes in Neo4j")
            return True

        except Exception as e:
            logger.error(f"Failed to store hierarchy in Neo4j: {e}")
            return False

    async def two_phase_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform two-phase hierarchical search:
        1. Search directory summaries to find relevant directories
        2. Search files within those directories

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of relevant file results
        """
        try:
            # Phase 1: Find relevant directories
            dir_collection = f"{self.qdrant_service.collection_prefix}-directories"

            # Generate query embedding
            query_embeddings = await self.nomic_service.generate_embeddings([query])
            if not query_embeddings:
                logger.warning("Failed to generate query embedding")
                return []

            # Search directory summaries
            dir_results = await self.qdrant_service.search(
                collection_name=dir_collection,
                query_vector=query_embeddings[0],
                limit=min(5, limit // 2),  # Get top 5 directories
                score_threshold=0.5  # Minimum similarity
            )

            if not dir_results or not dir_results.get('results'):
                logger.info("No relevant directories found")
                return []

            # Extract directory paths
            relevant_dirs = []
            for result in dir_results['results']:
                if 'payload' in result and 'path' in result['payload']:
                    relevant_dirs.append(result['payload']['path'])

            logger.info(f"Found {len(relevant_dirs)} relevant directories for query")

            # Phase 2: Search files within relevant directories
            file_results = []
            file_collection = f"{self.qdrant_service.collection_prefix}-code"

            for dir_path in relevant_dirs:
                # Create filter for files in this directory
                dir_filter = Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=dir_path, prefix=True)  # Prefix match
                        )
                    ]
                )

                # Search files in this directory
                results = await self.qdrant_service.search(
                    collection_name=file_collection,
                    query_vector=query_embeddings[0],
                    limit=limit // len(relevant_dirs) + 1,  # Distribute limit across dirs
                    filter=dir_filter
                )

                if results and 'results' in results:
                    file_results.extend(results['results'])

            # Sort by score and limit
            file_results.sort(key=lambda x: x.get('score', 0), reverse=True)

            return file_results[:limit]

        except Exception as e:
            logger.error(f"Two-phase search failed: {e}")
            return []

    async def detect_changes_with_merkle(self, root_path: str) -> Dict[str, List[str]]:
        """
        Detect changes in directory structure using Merkle tree comparison.

        Args:
            root_path: Root directory to check

        Returns:
            Dictionary with 'added', 'modified', 'deleted' directory lists
        """
        changes = {
            'added': [],
            'modified': [],
            'deleted': []
        }

        try:
            # Build current hierarchy
            current_hierarchy = await self.build_directory_hierarchy(root_path)

            # Get stored hierarchy from Neo4j
            stored_dirs = await self.neo4j_service.run_query(
                """
                MATCH (d:Directory {project: $project})
                WHERE d.path STARTS WITH $root_path
                RETURN d.path as path, d.merkle_hash as hash
                """,
                parameters={
                    'project': self.neo4j_service.project_name,
                    'root_path': root_path
                }
            )

            stored_hashes = {row['path']: row['hash'] for row in stored_dirs}

            # Compare current vs stored
            current_paths = set(current_hierarchy.keys())
            stored_paths = set(stored_hashes.keys())

            # Find added directories
            for path in current_paths - stored_paths:
                changes['added'].append(path)

            # Find deleted directories
            for path in stored_paths - current_paths:
                changes['deleted'].append(path)

            # Find modified directories (hash changed)
            for path in current_paths & stored_paths:
                if current_hierarchy[path].merkle_hash != stored_hashes[path]:
                    changes['modified'].append(path)

            total_changes = len(changes['added']) + len(changes['modified']) + len(changes['deleted'])
            logger.info(f"Detected {total_changes} directory changes using Merkle tree")

            return changes

        except Exception as e:
            logger.error(f"Merkle change detection failed: {e}")
            return changes

    async def update_changed_directories(self, changes: Dict[str, List[str]],
                                        hierarchy: Dict[str, DirectorySummary]) -> bool:
        """
        Update only changed directories in the databases.

        Args:
            changes: Dictionary of changes from detect_changes_with_merkle
            hierarchy: Current directory hierarchy

        Returns:
            Success status
        """
        try:
            # Handle deleted directories
            if changes['deleted']:
                # Remove from Neo4j
                await self.neo4j_service.run_query(
                    """
                    MATCH (d:Directory {project: $project})
                    WHERE d.path IN $paths
                    DETACH DELETE d
                    """,
                    parameters={
                        'project': self.neo4j_service.project_name,
                        'paths': changes['deleted']
                    }
                )

                # Remove from Qdrant
                dir_collection = f"{self.qdrant_service.collection_prefix}-directories"
                delete_ids = [hashlib.sha256(path.encode()).hexdigest()[:16]
                             for path in changes['deleted']]
                await self.qdrant_service.client.delete(
                    collection_name=dir_collection,
                    points_selector=delete_ids
                )

            # Update added and modified directories
            dirs_to_update = {}
            for path in changes['added'] + changes['modified']:
                if path in hierarchy:
                    dirs_to_update[path] = hierarchy[path]

            if dirs_to_update:
                # Update in both databases
                await self.store_hierarchy_in_qdrant(dirs_to_update)
                await self.store_hierarchy_in_neo4j(dirs_to_update)

            logger.info(f"Updated {len(dirs_to_update)} directories, deleted {len(changes['deleted'])}")
            return True

        except Exception as e:
            logger.error(f"Failed to update changed directories: {e}")
            return False