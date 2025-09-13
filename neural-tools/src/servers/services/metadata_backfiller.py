#!/usr/bin/env python3
"""
Metadata Backfiller for Existing Indexed Data
ADR-0031: Backfills canonical metadata and PRISM scores for already-indexed content
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add parent directories to path for imports
services_dir = Path(__file__).parent
sys.path.insert(0, str(services_dir))
# Add servers directory for absolute imports
sys.path.insert(0, str(services_dir.parent))
# Add infrastructure directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "infrastructure"))

from service_container import ServiceContainer
from collection_config import get_collection_manager, CollectionType
from pattern_extractor import PatternExtractor
from git_extractor import GitMetadataExtractor
from canon_manager import CanonManager

# Try to import PRISM scorer
try:
    from prism_scorer import PrismScorer
    PRISM_SCORING_ENABLED = True
except ImportError:
    PRISM_SCORING_ENABLED = False

logger = logging.getLogger(__name__)


class MetadataBackfiller:
    """Backfill metadata for already-indexed content"""
    
    def __init__(self, project_name: str, project_path: str = None, container: ServiceContainer = None):
        self.project_name = project_name
        # If no project path provided, use current directory
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.container = container
        self.collection_manager = get_collection_manager(project_name)
        
        # Initialize metadata extractors with the project path
        self.pattern_extractor = PatternExtractor()
        self.git_extractor = GitMetadataExtractor(str(self.project_path))
        self.canon_manager = CanonManager(self.project_name, str(self.project_path))
        
        if PRISM_SCORING_ENABLED:
            self.prism_scorer = PrismScorer(str(self.project_path))
        else:
            self.prism_scorer = None
            
        # Progress tracking
        self.processed_count = 0
        self.total_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.updated_neo4j = 0
        self.updated_qdrant = 0
        
    async def initialize(self):
        """Initialize service connections"""
        if not self.container:
            self.container = ServiceContainer(self.project_name)
            await self.container.initialize_all_services()
    
    async def _extract_metadata(self, file_path: str, content: str) -> dict:
        """Extract all metadata for a file (same as indexer)"""
        metadata = {}
        
        # 1. PRISM Scores
        if self.prism_scorer:
            try:
                prism_components = self.prism_scorer.get_score_components(file_path)
                metadata.update({
                    'complexity_score': prism_components.get('complexity', 0.0),
                    'dependencies_score': prism_components.get('dependencies', 0.0),
                    'recency_score': prism_components.get('recency', 0.0),
                    'contextual_score': prism_components.get('contextual', 0.0),
                    'prism_total': prism_components.get('total', 0.0)
                })
            except Exception as e:
                logger.debug(f"PRISM scoring failed for {file_path}: {e}")
        
        # 2. Git Metadata
        try:
            git_metadata = await self.git_extractor.extract(file_path)
            metadata.update({
                'last_modified': git_metadata.get('last_modified'),
                'change_frequency': git_metadata.get('change_frequency', 0),
                'author_count': git_metadata.get('author_count', 1),
                'last_commit': git_metadata.get('last_commit', 'unknown')
            })
        except Exception as e:
            logger.debug(f"Git metadata extraction failed for {file_path}: {e}")
        
        # 3. Pattern-based Extraction
        try:
            patterns = self.pattern_extractor.extract(content)
            metadata.update({
                'todo_count': patterns.get('todo_count', 0),
                'fixme_count': patterns.get('fixme_count', 0),
                'deprecated_markers': patterns.get('deprecated_count', 0),
                'test_markers': patterns.get('test_count', 0),
                'security_patterns': patterns.get('security_count', 0),
                'canon_markers': patterns.get('canon_markers', 0),
                'experimental_markers': patterns.get('experimental_markers', 0),
                'is_async': patterns.get('is_async', False),
                'has_type_hints': patterns.get('has_type_hints', False)
            })
        except Exception as e:
            logger.debug(f"Pattern extraction failed for {file_path}: {e}")
        
        # 4. Canon Configuration
        try:
            canon_data = await self.canon_manager.get_file_metadata(file_path)
            metadata.update({
                'canon_level': canon_data.get('level', 'none'),
                'canon_weight': canon_data.get('weight', 0.5),
                'canon_reason': canon_data.get('reason', ''),
                'is_canonical': canon_data.get('weight', 0.5) >= 0.7
            })
        except Exception as e:
            logger.debug(f"Canon metadata extraction failed for {file_path}: {e}")
        
        return metadata
    
    async def get_files_needing_backfill(self, check_field: str = 'complexity_score') -> List[Dict]:
        """Get list of files that need metadata backfill"""
        # Query Neo4j for files without metadata
        query = f"""
        MATCH (f:File {{project: $project}})
        WHERE f.{check_field} IS NULL
        RETURN f.path as path, f.content_hash as content_hash
        """
        
        result = await self.container.neo4j.execute_cypher(
            query, {'project': self.project_name}
        )
        
        if result.get('status') == 'success':
            return result.get('result', [])
        return []
    
    async def backfill_project(self, batch_size: int = 100, dry_run: bool = False):
        """Backfill metadata for entire project"""
        logger.info(f"Starting metadata backfill for project: {self.project_name}")
        
        # 1. Get all indexed files from Neo4j that need backfill
        files_to_backfill = await self.get_files_needing_backfill()
        self.total_count = len(files_to_backfill)
        
        if self.total_count == 0:
            logger.info("No files need metadata backfill")
            return
        
        logger.info(f"Found {self.total_count} files needing metadata backfill")
        
        if dry_run:
            logger.info("DRY RUN - No changes will be made")
            return
        
        # 2. Process in batches
        for i in range(0, len(files_to_backfill), batch_size):
            batch = files_to_backfill[i:i + batch_size]
            
            for file_info in batch:
                await self._process_file(file_info)
                
                # Progress logging every 10 files
                if self.processed_count % 10 == 0 and self.processed_count > 0:
                    progress = (self.processed_count / self.total_count) * 100
                    logger.info(
                        f"Backfill progress: {progress:.1f}% "
                        f"({self.processed_count}/{self.total_count}) "
                        f"Neo4j: {self.updated_neo4j}, Qdrant: {self.updated_qdrant}, "
                        f"Skipped: {self.skipped_count}, Errors: {self.error_count}"
                    )
            
            # Small delay between batches to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        # Final summary
        logger.info(
            f"Backfill complete! Processed: {self.processed_count}, "
            f"Updated Neo4j: {self.updated_neo4j}, Updated Qdrant: {self.updated_qdrant}, "
            f"Skipped: {self.skipped_count}, Errors: {self.error_count}"
        )
    
    async def _process_file(self, file_info: Dict):
        """Process a single file for metadata backfill"""
        self.processed_count += 1
        
        try:
            relative_path = file_info['path']
            file_path = self.project_path / relative_path
            
            # Check if file still exists
            if not file_path.exists():
                logger.debug(f"File no longer exists: {file_path}")
                self.skipped_count += 1
                return
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.debug(f"Could not read file {file_path}: {e}")
                self.error_count += 1
                return
            
            # Extract metadata
            metadata = await self._extract_metadata(str(file_path), content)
            
            # Update Neo4j
            await self._update_neo4j_metadata(relative_path, metadata)
            
            # Update Qdrant points
            await self._update_qdrant_metadata(relative_path, metadata)
            
        except Exception as e:
            logger.error(f"Error processing file {file_info.get('path')}: {e}")
            self.error_count += 1
    
    async def _update_neo4j_metadata(self, file_path: str, metadata: dict):
        """Update Neo4j node with metadata"""
        cypher = """
        MATCH (f:File {path: $path, project: $project})
        SET f.complexity_score = $complexity_score,
            f.dependencies_score = $dependencies_score,
            f.recency_score = $recency_score,
            f.contextual_score = $contextual_score,
            f.prism_total = $prism_total,
            f.canon_level = $canon_level,
            f.canon_weight = $canon_weight,
            f.canon_reason = $canon_reason,
            f.is_canonical = $is_canonical,
            f.last_modified = $last_modified,
            f.change_frequency = $change_frequency,
            f.author_count = $author_count,
            f.last_commit = $last_commit,
            f.todo_count = $todo_count,
            f.fixme_count = $fixme_count,
            f.deprecated_markers = $deprecated_markers,
            f.test_markers = $test_markers,
            f.security_patterns = $security_patterns,
            f.canon_markers = $canon_markers,
            f.experimental_markers = $experimental_markers,
            f.is_async = $is_async,
            f.has_type_hints = $has_type_hints,
            f.metadata_version = 'v1.0',
            f.metadata_backfilled_at = datetime()
        RETURN f.path
        """
        
        params = {
            'path': file_path,
            'project': self.project_name,
            'complexity_score': metadata.get('complexity_score', 0.0),
            'dependencies_score': metadata.get('dependencies_score', 0.0),
            'recency_score': metadata.get('recency_score', 0.0),
            'contextual_score': metadata.get('contextual_score', 0.0),
            'prism_total': metadata.get('prism_total', 0.0),
            'canon_level': metadata.get('canon_level', 'none'),
            'canon_weight': metadata.get('canon_weight', 0.5),
            'canon_reason': metadata.get('canon_reason', ''),
            'is_canonical': metadata.get('is_canonical', False),
            'last_modified': metadata.get('last_modified', datetime.now().isoformat()),
            'change_frequency': metadata.get('change_frequency', 0),
            'author_count': metadata.get('author_count', 1),
            'last_commit': metadata.get('last_commit', 'unknown'),
            'todo_count': metadata.get('todo_count', 0),
            'fixme_count': metadata.get('fixme_count', 0),
            'deprecated_markers': metadata.get('deprecated_markers', 0),
            'test_markers': metadata.get('test_markers', 0),
            'security_patterns': metadata.get('security_patterns', 0),
            'canon_markers': metadata.get('canon_markers', 0),
            'experimental_markers': metadata.get('experimental_markers', 0),
            'is_async': metadata.get('is_async', False),
            'has_type_hints': metadata.get('has_type_hints', False)
        }
        
        result = await self.container.neo4j.execute_cypher(cypher, params)
        if result.get('status') == 'success':
            self.updated_neo4j += 1
    
    async def _update_qdrant_metadata(self, file_path: str, metadata: dict):
        """Update Qdrant points with metadata"""
        try:
            # Get collection name
            collection_name = self.collection_manager.get_collection_name(CollectionType.CODE)
            
            # Search for points belonging to this file
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path)
                    ),
                    FieldCondition(
                        key="project",
                        match=MatchValue(value=self.project_name)
                    )
                ]
            )
            
            # Scroll through all points for this file
            scroll_result = await self.container.qdrant.client.scroll(
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=100,
                with_payload=False,
                with_vectors=False
            )
            
            points = scroll_result[0] if scroll_result else []
            
            if not points:
                logger.debug(f"No Qdrant points found for file: {file_path}")
                return
            
            # Update each point's payload with metadata
            point_ids = [point.id for point in points]
            
            # Batch update payloads
            await self.container.qdrant.client.set_payload(
                collection_name=collection_name,
                payload=metadata,
                points=point_ids
            )
            
            self.updated_qdrant += len(point_ids)
            logger.debug(f"Updated {len(point_ids)} Qdrant points for {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to update Qdrant metadata for {file_path}: {e}")


async def main():
    """Standalone script to run metadata backfill"""
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Backfill metadata for indexed content')
    parser.add_argument('--project', default='default', help='Project name')
    parser.add_argument('--path', default='.', help='Project path')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--dry-run', action='store_true', help='Dry run - only count files')
    
    args = parser.parse_args()
    
    # Create and run backfiller
    backfiller = MetadataBackfiller(args.project, args.path)
    await backfiller.initialize()
    
    if args.dry_run:
        files = await backfiller.get_files_needing_backfill()
        print(f"Files needing backfill: {len(files)}")
        if files and len(files) <= 10:
            print("Files:")
            for f in files:
                print(f"  - {f['path']}")
    else:
        await backfiller.backfill_project(batch_size=args.batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())