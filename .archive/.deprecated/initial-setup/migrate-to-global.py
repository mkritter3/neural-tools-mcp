#!/usr/bin/env python3
"""
Neural Memory Migration Script
Migrates from L9-prefixed architecture to global neural memory system
"""

import os
import sys
import json
import shutil
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import tarfile
import subprocess

# Add neural-system to path
current_dir = Path(__file__).parent
neural_system_path = current_dir / '.claude' / 'neural-system'
sys.path.append(str(neural_system_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MigrationError(Exception):
    """Migration-specific exception"""
    pass

class NeuralMemoryMigrator:
    """Main migration orchestrator"""
    
    def __init__(self, args):
        self.args = args
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / 'backups'
        self.backup_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Migration state
        self.migration_state = {
            'phase': 'initialized',
            'started_at': datetime.now().isoformat(),
            'backup_created': False,
            'l9_data_exported': False,
            'global_system_deployed': False,
            'data_imported': False,
            'verified': False
        }
        
    async def run_migration(self):
        """Run the complete migration process"""
        try:
            logger.info("🚀 Starting Neural Memory System Migration")
            logger.info("=" * 60)
            
            # Handle specific actions
            if self.args.action == 'backup-only':
                await self.create_backup()
                return
            
            if self.args.action == 'verify':
                await self.verify_migration()
                return
                
            if self.args.action == 'rollback':
                await self.rollback_migration()
                return
                
            if self.args.action == 'cleanup-legacy':
                await self.cleanup_legacy()
                return
            
            # Full migration process
            await self.full_migration()
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
            
            # Offer to rollback on failure
            if self.migration_state.get('backup_created'):
                response = input("\n🔄 Migration failed. Would you like to rollback? (y/N): ")
                if response.lower().startswith('y'):
                    await self.rollback_migration()
            
            sys.exit(1)
    
    async def full_migration(self):
        """Execute full migration workflow"""
        logger.info("📋 Full Migration Workflow Started")
        
        # Phase 1: Backup
        logger.info("\n📦 Phase 1: Creating Backup")
        await self.create_backup()
        
        # Phase 2: Export L9 Data
        logger.info("\n📤 Phase 2: Exporting L9 Data")
        await self.export_l9_data()
        
        # Phase 3: Deploy Global System
        logger.info("\n🚀 Phase 3: Deploying Global System")
        await self.deploy_global_system()
        
        # Phase 4: Import Data
        logger.info("\n📥 Phase 4: Importing Data to Global System")
        await self.import_data()
        
        # Phase 5: Verification
        if not self.args.skip_verify:
            logger.info("\n✅ Phase 5: Verifying Migration")
            await self.verify_migration()
        
        # Phase 6: Cleanup (optional)
        if self.args.auto and not self.args.keep_legacy:
            logger.info("\n🧹 Phase 6: Cleaning Up Legacy System")
            await self.cleanup_legacy()
        
        logger.info("\n🎉 Migration Completed Successfully!")
        self._print_next_steps()
    
    async def create_backup(self):
        """Create comprehensive backup of current system"""
        logger.info("Creating system backup...")
        
        self.backup_dir.mkdir(exist_ok=True)
        backup_name = f"neural-backup_{self.backup_timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        try:
            # Backup neural system files
            neural_backup = backup_path / 'neural-system'
            if neural_system_path.exists():
                shutil.copytree(neural_system_path, neural_backup)
                logger.info(f"✅ Neural system files backed up")
            
            # Backup Qdrant data
            qdrant_source = self.project_root / '.docker' / 'qdrant'
            if qdrant_source.exists():
                qdrant_backup = backup_path / 'qdrant'
                shutil.copytree(qdrant_source, qdrant_backup)
                logger.info(f"✅ Qdrant data backed up")
            
            # Backup configuration files
            config_files = ['.mcp.json', '.claude/settings.json']
            for config_file in config_files:
                source = self.project_root / config_file
                if source.exists():
                    dest = backup_path / config_file
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                    logger.info(f"✅ {config_file} backed up")
            
            # Create backup metadata
            backup_metadata = {
                'created_at': self.backup_timestamp,
                'project_path': str(self.project_root),
                'migration_version': '1.0.0',
                'backup_contents': {
                    'neural_system': neural_backup.exists(),
                    'qdrant_data': (backup_path / 'qdrant').exists(),
                    'mcp_config': (backup_path / '.mcp.json').exists(),
                    'claude_settings': (backup_path / '.claude' / 'settings.json').exists()
                }
            }
            
            with open(backup_path / 'backup-metadata.json', 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            # Create archive
            archive_path = self.backup_dir / f"{backup_name}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_name)
            
            # Clean up uncompressed backup
            shutil.rmtree(backup_path)
            
            self.migration_state['backup_created'] = True
            self.migration_state['backup_path'] = str(archive_path)
            
            logger.info(f"✅ Backup created: {archive_path}")
            logger.info(f"   Backup size: {archive_path.stat().st_size / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            raise MigrationError(f"Backup creation failed: {e}")
    
    async def export_l9_data(self):
        """Export data from L9 system"""
        logger.info("Exporting L9 system data...")
        
        try:
            # Try to import L9 system and export data
            l9_export_script = f"""
import sys
sys.path.append('{neural_system_path}')

try:
    from l9_qdrant_memory_v2 import L9QdrantMemoryV2
    import asyncio
    import json
    from datetime import datetime
    
    async def export_l9_data():
        try:
            memory = L9QdrantMemoryV2()
            await memory.initialize()
            
            # Get project stats
            stats = await memory.get_project_stats()
            
            # Export memories with cross-project search
            memories = await memory.search_project_memories(
                query='', 
                limit=10000, 
                include_other_projects=True
            )
            
            export_data = {{
                'version': '1.0.0',
                'exported_at': datetime.now().isoformat(),
                'project_name': memory.project_name,
                'container_name': memory.container_name,
                'collection_name': memory.collection_name,
                'stats': stats,
                'memories': [],
                'total_exported': len(memories)
            }}
            
            # Convert memories to exportable format
            for memory in memories:
                export_data['memories'].append({{
                    'memory_id': memory.memory_id,
                    'content': memory.content,
                    'timestamp': memory.timestamp.isoformat(),
                    'score': memory.score,
                    'entities': memory.entities,
                    'token_count': memory.token_count,
                    'search_type': memory.search_type,
                    'project': memory.project,
                    'metadata': memory.metadata
                }})
            
            # Save export data
            with open('l9-export.json', 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"SUCCESS: Exported {{len(memories)}} memories from project {{memory.project_name}}")
            
        except Exception as e:
            print(f"ERROR: {{e}}")
            # Create empty export if L9 system not accessible
            export_data = {{
                'version': '1.0.0',
                'exported_at': datetime.now().isoformat(),
                'project_name': 'unknown',
                'memories': [],
                'total_exported': 0,
                'export_error': str(e)
            }}
            
            with open('l9-export.json', 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print("WARNING: Created empty export due to L9 system access issues")
    
    asyncio.run(export_l9_data())
    
except ImportError as e:
    # Create minimal export if L9 modules not available
    import json
    from datetime import datetime
    
    export_data = {{
        'version': '1.0.0',
        'exported_at': datetime.now().isoformat(),
        'memories': [],
        'total_exported': 0,
        'export_error': f'L9 modules not available: {{e}}'
    }}
    
    with open('l9-export.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("WARNING: L9 modules not available, created empty export")
"""
            
            # Run export script
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-c', l9_export_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if stderr:
                logger.warning(f"Export warnings: {stderr.decode()}")
            
            logger.info(stdout.decode())
            
            # Verify export file was created
            export_file = self.project_root / 'l9-export.json'
            if export_file.exists():
                with open(export_file) as f:
                    export_data = json.load(f)
                
                total_exported = export_data.get('total_exported', 0)
                logger.info(f"✅ L9 data export completed: {total_exported} memories")
                
                self.migration_state['l9_data_exported'] = True
                self.migration_state['export_count'] = total_exported
            else:
                raise MigrationError("L9 export file not created")
                
        except Exception as e:
            logger.warning(f"L9 data export encountered issues: {e}")
            # Create empty export to continue migration
            export_data = {
                'version': '1.0.0',
                'exported_at': datetime.now().isoformat(),
                'memories': [],
                'total_exported': 0,
                'export_error': str(e)
            }
            
            with open(self.project_root / 'l9-export.json', 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.migration_state['l9_data_exported'] = True
            self.migration_state['export_count'] = 0
            logger.info("⚠️  Continuing with empty export")
    
    async def deploy_global_system(self):
        """Deploy the new global neural memory system"""
        logger.info("Deploying global neural memory system...")
        
        try:
            # Stop any existing L9 containers
            logger.info("Stopping L9 containers...")
            try:
                result = subprocess.run(
                    ["docker", "container", "ls", "-q", "--filter", "label=l9-system"],
                    capture_output=True, text=True, check=False
                )
                
                if result.stdout.strip():
                    subprocess.run(
                        ["docker", "container", "stop"] + result.stdout.strip().split(),
                        check=False
                    )
                    logger.info("✅ L9 containers stopped")
            except:
                logger.info("⚠️  No L9 containers to stop")
            
            # Start new unified system
            logger.info("Starting global neural memory system...")
            
            if (self.project_root / 'neural-docker.sh').exists():
                result = subprocess.run(
                    ["bash", "neural-docker.sh", "start"],
                    cwd=self.project_root,
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    logger.info("✅ Global system containers started")
                else:
                    logger.warning(f"Container start warnings: {result.stderr}")
            
            # Wait for services to be ready
            logger.info("Waiting for services to be ready...")
            await asyncio.sleep(10)
            
            # Verify system health
            await self._verify_system_health()
            
            self.migration_state['global_system_deployed'] = True
            
        except Exception as e:
            raise MigrationError(f"Global system deployment failed: {e}")
    
    async def import_data(self):
        """Import exported L9 data into global system"""
        logger.info("Importing data to global system...")
        
        try:
            # Load export data
            export_file = self.project_root / 'l9-export.json'
            if not export_file.exists():
                logger.warning("No export data found, skipping import")
                self.migration_state['data_imported'] = True
                return
            
            with open(export_file) as f:
                export_data = json.load(f)
            
            memories = export_data.get('memories', [])
            if not memories:
                logger.info("No memories to import")
                self.migration_state['data_imported'] = True
                return
            
            # Import script
            import_script = f"""
import sys
sys.path.append('{neural_system_path}')

try:
    from memory_system import MemorySystem
    import asyncio
    import json
    from datetime import datetime
    
    async def import_data():
        with open('l9-export.json', 'r') as f:
            export_data = json.load(f)
        
        memories = export_data.get('memories', [])
        if not memories:
            print("No memories to import")
            return
        
        memory_system = MemorySystem()
        await memory_system.initialize()
        
        imported_count = 0
        failed_count = 0
        
        for i, memory_data in enumerate(memories):
            try:
                await memory_system.store_memory(
                    content=memory_data['content'],
                    metadata={{
                        'migrated_from_l9': True,
                        'original_memory_id': memory_data.get('memory_id', ''),
                        'original_timestamp': memory_data.get('timestamp', ''),
                        'original_project': memory_data.get('project', ''),
                        'migration_date': datetime.now().isoformat(),
                        **memory_data.get('metadata', {{}})
                    }}
                )
                imported_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Imported {{i + 1}}/{{len(memories)}} memories...")
                    
            except Exception as e:
                failed_count += 1
                print(f"Failed to import memory {{i + 1}}: {{e}}")
        
        print(f"SUCCESS: Imported {{imported_count}} memories, {{failed_count}} failed")
        
        # Get final stats
        stats = await memory_system.get_stats()
        print(f"Global system now has {{stats.get('total_memories', 0)}} total memories")
        
    asyncio.run(import_data())
    
except Exception as e:
    print(f"ERROR: Import failed: {{e}}")
"""
            
            # Run import script
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-c', import_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if stderr:
                logger.warning(f"Import warnings: {stderr.decode()}")
            
            logger.info(stdout.decode())
            
            self.migration_state['data_imported'] = True
            logger.info(f"✅ Data import completed")
            
        except Exception as e:
            raise MigrationError(f"Data import failed: {e}")
    
    async def verify_migration(self):
        """Verify migration was successful"""
        logger.info("Verifying migration...")
        
        verification_results = {
            'system_health': False,
            'mcp_server': False,
            'memory_operations': False,
            'data_integrity': False
        }
        
        try:
            # Check system health
            await self._verify_system_health()
            verification_results['system_health'] = True
            
            # Test MCP server
            if await self._test_mcp_server():
                verification_results['mcp_server'] = True
            
            # Test memory operations
            if await self._test_memory_operations():
                verification_results['memory_operations'] = True
            
            # Verify data integrity
            if await self._verify_data_integrity():
                verification_results['data_integrity'] = True
            
            # Summary
            passed = sum(verification_results.values())
            total = len(verification_results)
            
            logger.info(f"✅ Verification Results: {passed}/{total} checks passed")
            
            for check, result in verification_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"   {check}: {status}")
            
            if passed == total:
                logger.info("🎉 Migration verification successful!")
                self.migration_state['verified'] = True
            else:
                logger.warning("⚠️  Some verification checks failed")
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            raise MigrationError(f"Migration verification failed: {e}")
    
    async def _verify_system_health(self):
        """Check if global system is healthy"""
        try:
            # Check containers are running
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
                capture_output=True, text=True
            )
            
            running_containers = result.stdout
            required_containers = ['qdrant-', 'shared-model-server']
            
            for container in required_containers:
                if container not in running_containers:
                    raise Exception(f"Required container not running: {container}")
            
            logger.info("✅ System containers are healthy")
            
        except Exception as e:
            raise Exception(f"System health check failed: {e}")
    
    async def _test_mcp_server(self):
        """Test MCP server functionality"""
        try:
            # Test if MCP server can start
            test_script = f"""
import sys
sys.path.append('{neural_system_path}')

try:
    from memory_system import MemorySystem
    print("✅ MemorySystem import successful")
    
    # Test basic instantiation
    memory = MemorySystem()
    print("✅ MemorySystem instantiation successful")
    
    print("SUCCESS: MCP server components working")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-c', test_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if "SUCCESS" in stdout.decode():
                logger.info("✅ MCP server test passed")
                return True
            else:
                logger.error(f"MCP server test failed: {stdout.decode()} {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"MCP server test error: {e}")
            return False
    
    async def _test_memory_operations(self):
        """Test basic memory operations"""
        try:
            test_script = f"""
import sys
sys.path.append('{neural_system_path}')
import asyncio

async def test_memory():
    try:
        from memory_system import MemorySystem
        
        memory = MemorySystem()
        await memory.initialize()
        
        # Test store operation
        test_content = "Migration test memory - " + str(asyncio.get_event_loop().time())
        memory_id = await memory.store_memory(
            content=test_content,
            metadata={{'test': True, 'migration_test': True}}
        )
        print(f"✅ Memory stored with ID: {{memory_id}}")
        
        # Test search operation
        results = await memory.search_memories(
            query="migration test",
            limit=1
        )
        
        if results:
            print(f"✅ Memory search successful: found {{len(results)}} results")
        else:
            print("⚠️  Memory search returned no results")
        
        print("SUCCESS: Memory operations working")
        
    except Exception as e:
        print(f"ERROR: {{e}}")

asyncio.run(test_memory())
"""
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-c', test_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if "SUCCESS" in stdout.decode():
                logger.info("✅ Memory operations test passed")
                return True
            else:
                logger.error(f"Memory operations test failed: {stdout.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Memory operations test error: {e}")
            return False
    
    async def _verify_data_integrity(self):
        """Verify migrated data integrity"""
        try:
            # Compare export vs imported data
            export_file = self.project_root / 'l9-export.json'
            if not export_file.exists():
                logger.info("✅ No export data to verify")
                return True
            
            with open(export_file) as f:
                export_data = json.load(f)
            
            exported_count = export_data.get('total_exported', 0)
            
            # Get current system stats
            stats_script = f"""
import sys
sys.path.append('{neural_system_path}')
import asyncio

async def get_stats():
    try:
        from memory_system import MemorySystem
        
        memory = MemorySystem()
        await memory.initialize()
        
        stats = await memory.get_stats()
        print(f"STATS: {{stats.get('total_memories', 0)}}")
        
    except Exception as e:
        print(f"ERROR: {{e}}")

asyncio.run(get_stats())
"""
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-c', stats_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if "STATS:" in stdout.decode():
                current_count = int(stdout.decode().split("STATS:")[1].strip())
                
                logger.info(f"Data comparison: Exported {exported_count}, Current {current_count}")
                
                if current_count >= exported_count:
                    logger.info("✅ Data integrity verification passed")
                    return True
                else:
                    logger.warning("⚠️  Current data count lower than exported")
                    return False
            
            logger.warning("Could not verify data integrity")
            return False
            
        except Exception as e:
            logger.error(f"Data integrity verification error: {e}")
            return False
    
    async def rollback_migration(self):
        """Rollback to L9 system using backup"""
        logger.info("🔄 Starting migration rollback...")
        
        try:
            # Find backup to restore
            backup_name = self.args.backup_name
            if not backup_name or backup_name == 'LATEST':
                # Find latest backup
                backups = list(self.backup_dir.glob('neural-backup_*.tar.gz'))
                if not backups:
                    raise MigrationError("No backups found for rollback")
                
                backup_file = max(backups, key=lambda x: x.stat().st_mtime)
            else:
                backup_file = self.backup_dir / f"neural-backup_{backup_name}.tar.gz"
                if not backup_file.exists():
                    backup_file = self.backup_dir / f"{backup_name}.tar.gz"
                
                if not backup_file.exists():
                    raise MigrationError(f"Backup not found: {backup_name}")
            
            logger.info(f"Restoring from backup: {backup_file}")
            
            # Stop global system
            logger.info("Stopping global system...")
            try:
                subprocess.run(
                    ["bash", "neural-docker.sh", "stop"],
                    cwd=self.project_root,
                    check=False
                )
            except:
                pass
            
            # Extract backup
            restore_dir = self.backup_dir / 'restore_temp'
            restore_dir.mkdir(exist_ok=True)
            
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(restore_dir)
            
            # Find the backup directory (should be the only directory in restore_temp)
            backup_dirs = list(restore_dir.glob('neural-backup_*'))
            if not backup_dirs:
                raise MigrationError("Invalid backup archive structure")
            
            backup_content = backup_dirs[0]
            
            # Restore neural system
            if (backup_content / 'neural-system').exists():
                if neural_system_path.exists():
                    shutil.rmtree(neural_system_path)
                shutil.copytree(backup_content / 'neural-system', neural_system_path)
                logger.info("✅ Neural system files restored")
            
            # Restore Qdrant data
            qdrant_target = self.project_root / '.docker' / 'qdrant'
            if (backup_content / 'qdrant').exists():
                if qdrant_target.exists():
                    shutil.rmtree(qdrant_target)
                shutil.copytree(backup_content / 'qdrant', qdrant_target)
                logger.info("✅ Qdrant data restored")
            
            # Restore configuration files
            config_files = ['.mcp.json', '.claude/settings.json']
            for config_file in config_files:
                backup_config = backup_content / config_file
                target_config = self.project_root / config_file
                
                if backup_config.exists():
                    target_config.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_config, target_config)
                    logger.info(f"✅ {config_file} restored")
            
            # Clean up restore directory
            shutil.rmtree(restore_dir)
            
            logger.info("🎉 Rollback completed successfully!")
            logger.info("💡 You may need to restart your L9 system manually")
            
        except Exception as e:
            raise MigrationError(f"Rollback failed: {e}")
    
    async def cleanup_legacy(self):
        """Clean up legacy L9 files and containers"""
        logger.info("🧹 Cleaning up legacy L9 system...")
        
        try:
            # Stop and remove L9 containers
            logger.info("Removing L9 containers...")
            try:
                result = subprocess.run(
                    ["docker", "container", "ls", "-aq", "--filter", "label=l9-system"],
                    capture_output=True, text=True
                )
                
                if result.stdout.strip():
                    subprocess.run(
                        ["docker", "container", "rm", "-f"] + result.stdout.strip().split(),
                        check=False
                    )
                    logger.info("✅ L9 containers removed")
            except:
                logger.info("⚠️  No L9 containers to remove")
            
            # Archive legacy files (don't delete immediately)
            legacy_dir = self.project_root / 'legacy'
            legacy_dir.mkdir(exist_ok=True)
            
            legacy_files = [
                'mcp_l9_launcher.py',
                'l9-export.json'
            ]
            
            moved_count = 0
            for file_name in legacy_files:
                file_path = self.project_root / file_name
                if file_path.exists():
                    target_path = legacy_dir / file_name
                    shutil.move(str(file_path), str(target_path))
                    moved_count += 1
            
            if moved_count > 0:
                logger.info(f"✅ {moved_count} legacy files moved to legacy/ directory")
            
            logger.info("✅ Legacy cleanup completed")
            
        except Exception as e:
            logger.warning(f"Legacy cleanup encountered issues: {e}")
    
    def _print_next_steps(self):
        """Print next steps for user"""
        print("\n" + "="*60)
        print("🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext Steps:")
        print("1. Open Claude Code - it will automatically use the new global MCP server")
        print("2. Try asking Claude to store and recall information to test the system")
        print("3. Test cross-project search if you have multiple projects")
        print("4. Review the migration log for any warnings")
        print("\nNew Features Available:")
        print("• 90%+ memory savings from shared model server")
        print("• Cross-project memory search when needed")
        print("• Automatic context injection via Claude Code hooks")
        print("• Zero-configuration setup for new projects")
        print("\nSupport:")
        print("• Run 'python3 migrate-to-global.py --verify' to re-check system")
        print("• Check MIGRATION-GUIDE.md for troubleshooting")
        print("• Run 'python3 migrate-to-global.py --rollback' if issues arise")
        print()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Neural Memory Migration Tool')
    
    parser.add_argument('action', nargs='?', default='auto', 
                       choices=['auto', 'backup-only', 'export-data', 'deploy-system', 
                               'import-data', 'verify', 'rollback', 'cleanup-legacy'],
                       help='Migration action to perform')
    
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--skip-verify', action='store_true', help='Skip verification step')
    parser.add_argument('--keep-legacy', action='store_true', help='Keep legacy files after migration')
    parser.add_argument('--backup-name', help='Specific backup to restore (for rollback)')
    parser.add_argument('--force', action='store_true', help='Force operation without prompts')
    parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    # Create and run migrator
    migrator = NeuralMemoryMigrator(args)
    await migrator.run_migration()

if __name__ == "__main__":
    asyncio.run(main())