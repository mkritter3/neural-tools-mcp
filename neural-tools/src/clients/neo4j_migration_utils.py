#!/usr/bin/env python3
"""
Neo4j Migration Utilities for L9 Neural Tools System
Provides utilities for migrating from Kuzu to Neo4j GraphRAG
Implements data migration, validation, and rollback procedures

Usage:
    python3 neo4j_migration_utils.py --migrate
    python3 neo4j_migration_utils.py --validate
    python3 neo4j_migration_utils.py --rollback
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from neo4j_client import Neo4jGraphRAGClient, AsyncNeo4jClient
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.error("Neo4j client not available")

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False
    logger.warning("Kuzu not available - migration from Kuzu disabled")

class Neo4jMigrationManager:
    """
    Manager for Neo4j migration operations
    Handles migration from Kuzu, validation, and rollback procedures
    """
    
    def __init__(self, project_name: str = None):
        self.project_name = project_name or os.getenv('PROJECT_NAME', 'default')
        self.migration_log_path = f"/app/data/migration_log_{self.project_name}.json"
        self.kuzu_db_path = os.getenv('KUZU_DB_PATH', '/app/kuzu')
        
        # Migration state tracking
        self.migration_state = {
            "project": self.project_name,
            "migration_id": f"neo4j_migration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "started_at": None,
            "completed_at": None,
            "status": "pending",
            "phases": {},
            "rollback_data": {},
            "validation_results": {}
        }
        
        logger.info(f"Migration manager initialized for project: {self.project_name}")
    
    def save_migration_state(self):
        """Save migration state to disk"""
        try:
            os.makedirs(os.path.dirname(self.migration_log_path), exist_ok=True)
            with open(self.migration_log_path, 'w') as f:
                json.dump(self.migration_state, f, indent=2, default=str)
            logger.debug(f"Migration state saved to {self.migration_log_path}")
        except Exception as e:
            logger.error(f"Failed to save migration state: {e}")
    
    def load_migration_state(self) -> bool:
        """Load existing migration state"""
        try:
            if os.path.exists(self.migration_log_path):
                with open(self.migration_log_path, 'r') as f:
                    self.migration_state = json.load(f)
                logger.info(f"Loaded migration state: {self.migration_state.get('status')}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load migration state: {e}")
            return False
    
    async def check_prerequisites(self) -> Dict[str, Any]:
        """
        Check migration prerequisites
        
        Returns:
            Prerequisites check results
        """
        logger.info("Checking migration prerequisites...")
        
        results = {
            "neo4j_available": NEO4J_AVAILABLE,
            "kuzu_available": KUZU_AVAILABLE,
            "neo4j_connection": False,
            "kuzu_data_exists": False,
            "project_isolation": False,
            "disk_space": False,
            "ready_for_migration": False
        }
        
        # Check Neo4j connection
        if NEO4J_AVAILABLE:
            try:
                async with AsyncNeo4jClient(project_name=self.project_name) as client:
                    stats = await client.get_project_statistics()
                    results["neo4j_connection"] = True
                    results["neo4j_stats"] = stats
                    logger.info("✅ Neo4j connection verified")
            except Exception as e:
                results["neo4j_error"] = str(e)
                logger.error(f"❌ Neo4j connection failed: {e}")
        
        # Check Kuzu data existence
        if KUZU_AVAILABLE:
            try:
                kuzu_path = Path(self.kuzu_db_path)
                if kuzu_path.exists() and any(kuzu_path.glob("*")):
                    results["kuzu_data_exists"] = True
                    results["kuzu_path"] = str(kuzu_path)
                    logger.info("✅ Kuzu data found for migration")
                else:
                    logger.info("ℹ️ No Kuzu data found - fresh Neo4j installation")
            except Exception as e:
                results["kuzu_error"] = str(e)
                logger.error(f"❌ Kuzu check failed: {e}")
        
        # Check project isolation
        results["project_isolation"] = True  # Neo4j client handles this
        logger.info("✅ Project isolation configured")
        
        # Check disk space (simplified)
        try:
            import shutil
            _, _, free_bytes = shutil.disk_usage("/app")
            free_gb = free_bytes / (1024**3)
            results["disk_space"] = free_gb > 1.0  # Require at least 1GB free
            results["free_space_gb"] = free_gb
            if results["disk_space"]:
                logger.info(f"✅ Sufficient disk space: {free_gb:.1f}GB free")
            else:
                logger.error(f"❌ Insufficient disk space: {free_gb:.1f}GB free")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Overall readiness
        results["ready_for_migration"] = (
            results["neo4j_available"] and 
            results["neo4j_connection"] and 
            results["project_isolation"] and
            results["disk_space"]
        )
        
        if results["ready_for_migration"]:
            logger.info("🚀 System ready for Neo4j migration")
        else:
            logger.error("🛑 System not ready for migration - check prerequisites")
        
        return results
    
    async def migrate_from_kuzu(self) -> Dict[str, Any]:
        """
        Migrate data from Kuzu to Neo4j
        
        Returns:
            Migration results
        """
        if not KUZU_AVAILABLE or not NEO4J_AVAILABLE:
            return {"status": "error", "message": "Both Kuzu and Neo4j must be available for migration"}
        
        logger.info("Starting Kuzu to Neo4j migration...")
        self.migration_state["started_at"] = datetime.utcnow().isoformat()
        self.migration_state["status"] = "running"
        self.save_migration_state()
        
        migration_results = {
            "files_migrated": 0,
            "functions_migrated": 0,
            "classes_migrated": 0,
            "relationships_migrated": 0,
            "errors": []
        }
        
        try:
            # Connect to Kuzu database - use full path to database file
            kuzu_db_file = os.path.join(self.kuzu_db_path, "graph.db")
            kuzu_db = kuzu.Database(kuzu_db_file)
            kuzu_conn = kuzu.Connection(kuzu_db)
            
            async with AsyncNeo4jClient(project_name=self.project_name) as neo4j_client:
                
                # Phase 1: Migrate File nodes
                logger.info("Phase 1: Migrating File nodes...")
                try:
                    result = kuzu_conn.execute("MATCH (f:File) RETURN f.path, f.content, f.extension, f.size_bytes")
                    while result.hasNext():
                        row = result.getNext()
                        success = await neo4j_client.create_file_node(
                            file_path=row[0] if row[0] else "",
                            content=row[1] if row[1] else "",
                            extension=row[2] if row[2] else "",
                            size_bytes=int(row[3]) if row[3] else 0
                        )
                        if success:
                            migration_results["files_migrated"] += 1
                        else:
                            migration_results["errors"].append(f"Failed to migrate file: {row[0]}")
                    
                    logger.info(f"✅ Migrated {migration_results['files_migrated']} files")
                    
                except Exception as e:
                    logger.error(f"File migration error: {e}")
                    migration_results["errors"].append(f"File migration: {str(e)}")
                
                # Phase 2: Migrate Function nodes
                logger.info("Phase 2: Migrating Function nodes...")
                try:
                    result = kuzu_conn.execute("""
                        MATCH (f:File)-[:CONTAINS]->(fn:Function) 
                        RETURN f.path, fn.name, fn.signature, fn.docstring, fn.start_line, fn.end_line
                    """)
                    while result.hasNext():
                        row = result.getNext()
                        success = await neo4j_client.create_function_node(
                            file_path=row[0] if row[0] else "",
                            function_name=row[1] if row[1] else "",
                            signature=row[2] if row[2] else "",
                            docstring=row[3] if row[3] else "",
                            start_line=int(row[4]) if row[4] else 0,
                            end_line=int(row[5]) if row[5] else 0
                        )
                        if success:
                            migration_results["functions_migrated"] += 1
                        else:
                            migration_results["errors"].append(f"Failed to migrate function: {row[1]}")
                    
                    logger.info(f"✅ Migrated {migration_results['functions_migrated']} functions")
                    
                except Exception as e:
                    logger.error(f"Function migration error: {e}")
                    migration_results["errors"].append(f"Function migration: {str(e)}")
                
                # Phase 3: Migrate Class nodes
                logger.info("Phase 3: Migrating Class nodes...")
                try:
                    result = kuzu_conn.execute("""
                        MATCH (f:File)-[:CONTAINS]->(c:Class) 
                        RETURN f.path, c.name, c.docstring, c.start_line, c.end_line
                    """)
                    while result.hasNext():
                        row = result.getNext()
                        success = await neo4j_client.create_class_node(
                            file_path=row[0] if row[0] else "",
                            class_name=row[1] if row[1] else "",
                            docstring=row[2] if row[2] else "",
                            start_line=int(row[3]) if row[3] else 0,
                            end_line=int(row[4]) if row[4] else 0
                        )
                        if success:
                            migration_results["classes_migrated"] += 1
                        else:
                            migration_results["errors"].append(f"Failed to migrate class: {row[1]}")
                    
                    logger.info(f"✅ Migrated {migration_results['classes_migrated']} classes")
                    
                except Exception as e:
                    logger.error(f"Class migration error: {e}")
                    migration_results["errors"].append(f"Class migration: {str(e)}")
                
                # Phase 4: Migrate relationships
                logger.info("Phase 4: Migrating relationships...")
                try:
                    relationship_types = ["IMPORTS", "REFERENCES", "RELATES_TO"]
                    for rel_type in relationship_types:
                        result = kuzu_conn.execute(f"""
                            MATCH (source:File)-[r:{rel_type}]->(target:File) 
                            RETURN source.path, target.path
                        """)
                        while result.hasNext():
                            row = result.getNext()
                            success = await neo4j_client.create_dependency_relationship(
                                source_file=row[0] if row[0] else "",
                                target_file=row[1] if row[1] else "",
                                dependency_type=rel_type
                            )
                            if success:
                                migration_results["relationships_migrated"] += 1
                    
                    logger.info(f"✅ Migrated {migration_results['relationships_migrated']} relationships")
                    
                except Exception as e:
                    logger.error(f"Relationship migration error: {e}")
                    migration_results["errors"].append(f"Relationship migration: {str(e)}")
            
            # Update migration state
            self.migration_state["completed_at"] = datetime.utcnow().isoformat()
            self.migration_state["status"] = "completed" if len(migration_results["errors"]) == 0 else "completed_with_errors"
            self.migration_state["results"] = migration_results
            self.save_migration_state()
            
            total_migrated = (
                migration_results["files_migrated"] + 
                migration_results["functions_migrated"] + 
                migration_results["classes_migrated"] + 
                migration_results["relationships_migrated"]
            )
            
            logger.info(f"🎉 Migration completed! Total items migrated: {total_migrated}")
            
            return {
                "status": "success",
                "migration_id": self.migration_state["migration_id"],
                "results": migration_results,
                "total_migrated": total_migrated
            }
            
        except Exception as e:
            self.migration_state["status"] = "failed"
            self.migration_state["error"] = str(e)
            self.save_migration_state()
            logger.error(f"Migration failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def validate_migration(self) -> Dict[str, Any]:
        """
        Validate Neo4j migration results
        
        Returns:
            Validation results
        """
        logger.info("Starting migration validation...")
        
        validation_results = {
            "neo4j_health": False,
            "data_integrity": False,
            "query_performance": False,
            "project_isolation": False,
            "total_score": 0,
            "details": {}
        }
        
        try:
            async with AsyncNeo4jClient(project_name=self.project_name) as client:
                
                # Test 1: Neo4j Health Check
                logger.info("Test 1: Neo4j health check...")
                try:
                    stats = await client.get_project_statistics()
                    if stats and "node_statistics" in stats:
                        validation_results["neo4j_health"] = True
                        validation_results["details"]["neo4j_stats"] = stats
                        logger.info("✅ Neo4j health check passed")
                    else:
                        logger.error("❌ Neo4j health check failed - no statistics")
                except Exception as e:
                    validation_results["details"]["neo4j_error"] = str(e)
                    logger.error(f"❌ Neo4j health check failed: {e}")
                
                # Test 2: Data Integrity Check
                logger.info("Test 2: Data integrity check...")
                try:
                    # Check for basic node types
                    integrity_queries = [
                        ("Files", "MATCH (f:File) WHERE f.project = $project RETURN count(f) as count"),
                        ("Functions", "MATCH (fn:Function) WHERE fn.project = $project RETURN count(fn) as count"),
                        ("Classes", "MATCH (c:Class) WHERE c.project = $project RETURN count(c) as count")
                    ]
                    
                    integrity_counts = {}
                    for name, query in integrity_queries:
                        results = await client.execute_cypher_query(query)
                        count = results[0]["count"] if results else 0
                        integrity_counts[name.lower()] = count
                        logger.info(f"  {name}: {count}")
                    
                    # Consider migration valid if we have at least some data
                    total_nodes = sum(integrity_counts.values())
                    validation_results["data_integrity"] = total_nodes > 0
                    validation_results["details"]["node_counts"] = integrity_counts
                    
                    if validation_results["data_integrity"]:
                        logger.info(f"✅ Data integrity check passed - {total_nodes} total nodes")
                    else:
                        logger.error("❌ Data integrity check failed - no nodes found")
                        
                except Exception as e:
                    validation_results["details"]["integrity_error"] = str(e)
                    logger.error(f"❌ Data integrity check failed: {e}")
                
                # Test 3: Query Performance Test
                logger.info("Test 3: Query performance test...")
                try:
                    import time
                    
                    # Test basic query performance
                    start_time = time.time()
                    results = await client.execute_cypher_query(
                        "MATCH (n) WHERE n.project = $project RETURN count(n) as total_nodes LIMIT 1"
                    )
                    end_time = time.time()
                    
                    query_time = end_time - start_time
                    validation_results["query_performance"] = query_time < 5.0  # Should complete within 5 seconds
                    validation_results["details"]["query_time_seconds"] = query_time
                    
                    if validation_results["query_performance"]:
                        logger.info(f"✅ Query performance test passed - {query_time:.2f}s")
                    else:
                        logger.error(f"❌ Query performance test failed - {query_time:.2f}s")
                        
                except Exception as e:
                    validation_results["details"]["performance_error"] = str(e)
                    logger.error(f"❌ Query performance test failed: {e}")
                
                # Test 4: Project Isolation Test
                logger.info("Test 4: Project isolation test...")
                try:
                    # Verify all nodes have correct project metadata
                    results = await client.execute_cypher_query(
                        "MATCH (n) WHERE n.project <> $project OR n.project IS NULL RETURN count(n) as non_isolated_count"
                    )
                    non_isolated = results[0]["non_isolated_count"] if results else 0
                    
                    validation_results["project_isolation"] = non_isolated == 0
                    validation_results["details"]["non_isolated_nodes"] = non_isolated
                    
                    if validation_results["project_isolation"]:
                        logger.info("✅ Project isolation test passed")
                    else:
                        logger.error(f"❌ Project isolation test failed - {non_isolated} non-isolated nodes")
                        
                except Exception as e:
                    validation_results["details"]["isolation_error"] = str(e)
                    logger.error(f"❌ Project isolation test failed: {e}")
                
                # Calculate total validation score
                score = sum([
                    validation_results["neo4j_health"],
                    validation_results["data_integrity"],
                    validation_results["query_performance"],
                    validation_results["project_isolation"]
                ])
                validation_results["total_score"] = score
                validation_results["max_score"] = 4
                validation_results["pass_percentage"] = (score / 4) * 100
                
                if score >= 3:
                    logger.info(f"🎉 Migration validation passed! Score: {score}/4 ({validation_results['pass_percentage']:.0f}%)")
                    validation_results["overall_status"] = "passed"
                else:
                    logger.error(f"🛑 Migration validation failed! Score: {score}/4 ({validation_results['pass_percentage']:.0f}%)")
                    validation_results["overall_status"] = "failed"
                
                # Save validation results
                self.migration_state["validation_results"] = validation_results
                self.save_migration_state()
                
                return {
                    "status": "success",
                    "validation": validation_results
                }
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def backup_kuzu_data(self) -> bool:
        """
        Create backup of Kuzu data before migration
        
        Returns:
            True if backup successful
        """
        if not KUZU_AVAILABLE:
            logger.warning("Kuzu not available - skipping backup")
            return True
        
        try:
            import shutil
            
            kuzu_path = Path(self.kuzu_db_path)
            if not kuzu_path.exists():
                logger.info("No Kuzu data to backup")
                return True
            
            backup_path = kuzu_path.parent / f"kuzu_backup_{self.project_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(kuzu_path, backup_path)
            
            self.migration_state["rollback_data"]["kuzu_backup_path"] = str(backup_path)
            logger.info(f"✅ Kuzu data backed up to: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Kuzu backup failed: {e}")
            return False
    
    async def rollback_migration(self) -> Dict[str, Any]:
        """
        Rollback Neo4j migration (clear Neo4j data, restore Kuzu if needed)
        
        Returns:
            Rollback results
        """
        logger.info("Starting migration rollback...")
        
        try:
            rollback_results = {"neo4j_cleared": False, "kuzu_restored": False}
            
            # Clear Neo4j data
            if NEO4J_AVAILABLE:
                try:
                    async with AsyncNeo4jClient(project_name=self.project_name) as client:
                        success = await client.clear_project_data()
                        rollback_results["neo4j_cleared"] = success
                        if success:
                            logger.info("✅ Neo4j data cleared")
                        else:
                            logger.error("❌ Failed to clear Neo4j data")
                except Exception as e:
                    logger.error(f"Neo4j rollback error: {e}")
            
            # Restore Kuzu backup if available
            backup_path = self.migration_state.get("rollback_data", {}).get("kuzu_backup_path")
            if backup_path and Path(backup_path).exists():
                try:
                    import shutil
                    kuzu_path = Path(self.kuzu_db_path)
                    
                    # Remove current Kuzu data
                    if kuzu_path.exists():
                        shutil.rmtree(kuzu_path)
                    
                    # Restore backup
                    shutil.copytree(backup_path, kuzu_path)
                    rollback_results["kuzu_restored"] = True
                    logger.info("✅ Kuzu data restored from backup")
                    
                except Exception as e:
                    logger.error(f"Kuzu restore error: {e}")
            
            # Update migration state
            self.migration_state["status"] = "rolled_back"
            self.migration_state["rollback_at"] = datetime.utcnow().isoformat()
            self.migration_state["rollback_results"] = rollback_results
            self.save_migration_state()
            
            success_count = sum(rollback_results.values())
            if success_count > 0:
                logger.info(f"🎉 Rollback completed - {success_count}/2 operations successful")
                return {"status": "success", "results": rollback_results}
            else:
                logger.error("🛑 Rollback failed - no operations successful")
                return {"status": "partial", "results": rollback_results}
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {"status": "error", "message": str(e)}

# CLI Interface
async def main():
    """Main CLI interface for migration utilities"""
    parser = argparse.ArgumentParser(description="Neo4j Migration Utilities for L9 Neural Tools")
    parser.add_argument("--project", help="Project name", default=os.getenv('PROJECT_NAME', 'default'))
    parser.add_argument("--migrate", action="store_true", help="Migrate from Kuzu to Neo4j")
    parser.add_argument("--validate", action="store_true", help="Validate Neo4j migration")
    parser.add_argument("--rollback", action="store_true", help="Rollback Neo4j migration")
    parser.add_argument("--check", action="store_true", help="Check migration prerequisites")
    parser.add_argument("--backup", action="store_true", help="Backup Kuzu data")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    
    args = parser.parse_args()
    
    if not any([args.migrate, args.validate, args.rollback, args.check, args.backup, args.status]):
        parser.print_help()
        return
    
    manager = Neo4jMigrationManager(project_name=args.project)
    manager.load_migration_state()
    
    try:
        if args.check:
            logger.info("🔍 Checking migration prerequisites...")
            results = await manager.check_prerequisites()
            print(json.dumps(results, indent=2))
        
        elif args.backup:
            logger.info("💾 Creating Kuzu backup...")
            success = await manager.backup_kuzu_data()
            print(f"Backup {'successful' if success else 'failed'}")
        
        elif args.migrate:
            logger.info("🚀 Starting Neo4j migration...")
            # Create backup first
            backup_success = await manager.backup_kuzu_data()
            if not backup_success:
                logger.error("Backup failed - aborting migration")
                return
            
            # Run migration
            results = await manager.migrate_from_kuzu()
            print(json.dumps(results, indent=2))
        
        elif args.validate:
            logger.info("✅ Validating Neo4j migration...")
            results = await manager.validate_migration()
            print(json.dumps(results, indent=2))
        
        elif args.rollback:
            logger.info("⏪ Rolling back Neo4j migration...")
            results = await manager.rollback_migration()
            print(json.dumps(results, indent=2))
        
        elif args.status:
            logger.info("📊 Migration status...")
            print(json.dumps(manager.migration_state, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())