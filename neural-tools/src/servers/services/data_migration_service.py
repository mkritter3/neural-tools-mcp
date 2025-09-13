#!/usr/bin/env python3
"""
Data Migration Service - ADR-0034 Phase 2 Implementation
Provides data migration capabilities for project pipeline synchronization

This service implements the data migration functions identified in ADR-0034
to clear inconsistent collections and ensure clean project data storage.

Author: L9 Engineering Team
Date: 2025-09-12
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class DataMigrationService:
    """
    Data migration service for cleaning inconsistent project data
    Implements ADR-0034 Phase 2 data migration
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:47687", 
                 neo4j_username: str = "neo4j", 
                 neo4j_password: str = "graphrag-password",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 46333):
        """
        Initialize the migration service
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            qdrant_host: Qdrant host
            qdrant_port: Qdrant port
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        self.neo4j_driver = None
        self.qdrant_client = None
        self.migration_results: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize Neo4j driver
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            
            logger.info("DataMigrationService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DataMigrationService: {e}")
            raise
    
    async def discover_inconsistent_collections(self) -> Dict[str, List[str]]:
        """
        Discover collections that don't follow the new naming standard
        
        Returns:
            Dictionary with 'qdrant' and 'neo4j' keys containing inconsistent collection names
        """
        inconsistent = {
            'qdrant': [],
            'neo4j': []
        }
        
        # Check Qdrant collections
        try:
            qdrant_collections = self.qdrant_client.get_collections()
            for collection in qdrant_collections.collections:
                collection_name = collection.name
                
                # Check if collection follows new standard (project-{name})
                if not collection_name.startswith('project-') or '_code' in collection_name:
                    inconsistent['qdrant'].append(collection_name)
                    logger.info(f"🔍 Found inconsistent Qdrant collection: {collection_name}")
                    
        except Exception as e:
            logger.error(f"Error discovering Qdrant collections: {e}")
        
        # Check Neo4j data (look for nodes without project property or inconsistent naming)
        try:
            async with self.neo4j_driver.session() as session:
                # Find nodes without project property
                result = await session.run("""
                    MATCH (n)
                    WHERE n.project IS NULL
                    RETURN DISTINCT labels(n) as node_types, count(n) as count
                """)
                
                async for record in result:
                    node_types = record["node_types"]
                    count = record["count"]
                    if count > 0:
                        inconsistent['neo4j'].append(f"Nodes without project property: {node_types} ({count} nodes)")
                        logger.info(f"🔍 Found {count} Neo4j nodes without project property: {node_types}")
                
                # Find nodes with old project naming patterns
                result = await session.run("""
                    MATCH (n)
                    WHERE n.project IS NOT NULL AND (
                        n.project CONTAINS "_code" OR 
                        n.project = "default" OR
                        n.project = "neural-novelist" OR
                        n.project CONTAINS "shadow-conspiracy"
                    )
                    RETURN DISTINCT n.project as project_name, count(n) as count
                """)
                
                async for record in result:
                    project_name = record["project_name"]
                    count = record["count"]
                    inconsistent['neo4j'].append(f"Inconsistent project: {project_name} ({count} nodes)")
                    logger.info(f"🔍 Found {count} Neo4j nodes with inconsistent project: {project_name}")
                    
        except Exception as e:
            logger.error(f"Error discovering Neo4j inconsistencies: {e}")
        
        return inconsistent
    
    async def clear_qdrant_collection(self, collection_name: str) -> bool:
        """
        Clear a specific Qdrant collection
        
        Args:
            collection_name: Name of the collection to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🗑️ Clearing Qdrant collection: {collection_name}")
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist, skipping")
                return True
            
            # Delete the collection
            self.qdrant_client.delete_collection(collection_name)
            logger.info(f"✅ Successfully deleted Qdrant collection: {collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear Qdrant collection {collection_name}: {e}")
            return False
    
    async def clear_neo4j_inconsistent_data(self) -> bool:
        """
        Clear inconsistent Neo4j data (nodes without project property or with old naming)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("🗑️ Clearing inconsistent Neo4j data")
            
            async with self.neo4j_driver.session() as session:
                # Clear nodes without project property
                result = await session.run("""
                    MATCH (n)
                    WHERE n.project IS NULL
                    DETACH DELETE n
                    RETURN count(n) as deleted_count
                """)
                
                record = await result.single()
                deleted_without_project = record["deleted_count"] if record else 0
                
                if deleted_without_project > 0:
                    logger.info(f"✅ Deleted {deleted_without_project} Neo4j nodes without project property")
                
                # Clear nodes with inconsistent project naming
                result = await session.run("""
                    MATCH (n)
                    WHERE n.project IS NOT NULL AND (
                        n.project CONTAINS "_code" OR 
                        n.project = "default" OR
                        n.project = "neural-novelist" OR
                        n.project CONTAINS "shadow-conspiracy"
                    )
                    DETACH DELETE n
                    RETURN count(n) as deleted_count
                """)
                
                record = await result.single()
                deleted_inconsistent = record["deleted_count"] if record else 0
                
                if deleted_inconsistent > 0:
                    logger.info(f"✅ Deleted {deleted_inconsistent} Neo4j nodes with inconsistent project naming")
                
                total_deleted = deleted_without_project + deleted_inconsistent
                logger.info(f"✅ Total Neo4j cleanup: {total_deleted} nodes deleted")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to clear Neo4j inconsistent data: {e}")
            return False
    
    async def migrate_data(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute complete data migration (Phase 2 of ADR-0034)
        
        Args:
            dry_run: If True, only discover inconsistencies without deleting
            
        Returns:
            Migration result dictionary
        """
        migration_start = datetime.now()
        logger.info(f"🚀 Starting data migration (dry_run={dry_run})")
        
        # Step 1: Discover inconsistent collections
        inconsistent = await self.discover_inconsistent_collections()
        
        migration_result = {
            "migration_timestamp": migration_start.isoformat(),
            "dry_run": dry_run,
            "discovered_inconsistencies": inconsistent,
            "cleared_collections": {
                "qdrant": [],
                "neo4j": False
            },
            "success": True,
            "errors": []
        }
        
        if dry_run:
            logger.info("🔍 Dry run mode - discovered inconsistencies only")
            total_qdrant = len(inconsistent['qdrant'])
            total_neo4j = len(inconsistent['neo4j'])
            logger.info(f"📊 Found {total_qdrant} inconsistent Qdrant collections")
            logger.info(f"📊 Found {total_neo4j} Neo4j inconsistency types")
            return migration_result
        
        # Step 2: Clear inconsistent Qdrant collections
        for collection_name in inconsistent['qdrant']:
            success = await self.clear_qdrant_collection(collection_name)
            if success:
                migration_result["cleared_collections"]["qdrant"].append(collection_name)
            else:
                migration_result["success"] = False
                migration_result["errors"].append(f"Failed to clear Qdrant collection: {collection_name}")
        
        # Step 3: Clear inconsistent Neo4j data
        if inconsistent['neo4j']:
            neo4j_success = await self.clear_neo4j_inconsistent_data()
            migration_result["cleared_collections"]["neo4j"] = neo4j_success
            if not neo4j_success:
                migration_result["success"] = False
                migration_result["errors"].append("Failed to clear Neo4j inconsistent data")
        else:
            migration_result["cleared_collections"]["neo4j"] = True  # No inconsistent data found
        
        # Step 4: Record results
        self.migration_results.append(migration_result)
        
        duration = (datetime.now() - migration_start).total_seconds()
        
        if migration_result["success"]:
            logger.info(f"✅ Data migration completed successfully in {duration:.2f}s")
            logger.info(f"📊 Cleared {len(migration_result['cleared_collections']['qdrant'])} Qdrant collections")
            logger.info(f"📊 Neo4j cleanup: {'✅ Success' if migration_result['cleared_collections']['neo4j'] else '❌ Failed'}")
        else:
            logger.error(f"❌ Data migration completed with errors in {duration:.2f}s")
            for error in migration_result["errors"]:
                logger.error(f"❌ {error}")
        
        return migration_result
    
    async def validate_migration(self) -> bool:
        """
        Validate that migration was successful (Phase 2 exit condition)
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            logger.info("🔍 Validating data migration results")
            
            # Check for remaining inconsistent collections
            inconsistent = await self.discover_inconsistent_collections()
            
            qdrant_clean = len(inconsistent['qdrant']) == 0
            neo4j_clean = len(inconsistent['neo4j']) == 0
            
            if qdrant_clean:
                logger.info("✅ Qdrant validation: No inconsistent collections found")
            else:
                logger.error(f"❌ Qdrant validation: {len(inconsistent['qdrant'])} inconsistent collections remain")
                for collection in inconsistent['qdrant']:
                    logger.error(f"❌ Remaining inconsistent collection: {collection}")
            
            if neo4j_clean:
                logger.info("✅ Neo4j validation: No inconsistent data found")
            else:
                logger.error(f"❌ Neo4j validation: {len(inconsistent['neo4j'])} inconsistency types remain")
                for issue in inconsistent['neo4j']:
                    logger.error(f"❌ Remaining inconsistency: {issue}")
            
            validation_passed = qdrant_clean and neo4j_clean
            
            if validation_passed:
                logger.info("🎉 Phase 2 Migration Validation: ✅ ALL PASSED")
            else:
                logger.error("💥 Phase 2 Migration Validation: ❌ FAILED")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"❌ Migration validation failed with error: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        if self.qdrant_client:
            self.qdrant_client.close()
        logger.info("DataMigrationService cleanup completed")


# Global migration service instance
migration_service = DataMigrationService()


async def execute_phase2_migration(dry_run: bool = False) -> Dict[str, Any]:
    """
    Convenience function for Phase 2 data migration
    
    Args:
        dry_run: If True, only discover inconsistencies without deleting
        
    Returns:
        Migration result dictionary
    """
    await migration_service.initialize()
    try:
        result = await migration_service.migrate_data(dry_run=dry_run)
        return result
    finally:
        await migration_service.cleanup()


async def validate_phase2_migration() -> bool:
    """
    Convenience function for Phase 2 migration validation
    
    Returns:
        True if validation passes, False otherwise
    """
    await migration_service.initialize()
    try:
        return await migration_service.validate_migration()
    finally:
        await migration_service.cleanup()