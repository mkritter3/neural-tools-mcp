#!/usr/bin/env python3
"""
Configuration Manager for Neural Search
Handles persistent storage of project paths and server settings
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
import time

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages persistent configuration for the neural search server
    Stores project paths, server settings, and last indexing timestamps
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Optional custom config directory. Defaults to ~/.neural_search
        """
        if config_dir:
            self.config_dir = Path(config_dir).expanduser().resolve()
        else:
            self.config_dir = Path.home() / ".neural_search"
            
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration structure
        self.default_config = {
            "version": "1.0.0",
            "projects": {},  # project_name -> {"path": str, "last_indexed": float, "auto_watch": bool}
            "server": {
                "host": "localhost",
                "port": 8000,
                "auto_start_watching": True,
                "debounce_interval": 1.5
            },
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        # Load existing config or create default
        self.config = self.load_config()
        logger.info(f"Configuration loaded from {self.config_file}")
    
    def load_config(self) -> Dict:
        """Load configuration from file, creating default if it doesn't exist"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                # Merge with defaults to handle version updates
                merged_config = self.default_config.copy()
                merged_config.update(config)
                
                # Ensure all projects have required fields
                for project_name, project_info in merged_config.get("projects", {}).items():
                    if isinstance(project_info, str):
                        # Legacy format: just a path string
                        merged_config["projects"][project_name] = {
                            "path": project_info,
                            "last_indexed": 0,
                            "auto_watch": True
                        }
                    else:
                        # Ensure all fields exist
                        project_info.setdefault("last_indexed", 0)
                        project_info.setdefault("auto_watch", True)
                
                logger.info(f"Loaded configuration for {len(merged_config.get('projects', {}))} projects")
                return merged_config
            else:
                logger.info("No existing configuration found, creating default")
                return self.default_config.copy()
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            return self.default_config.copy()
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            self.config["updated_at"] = time.time()
            
            # Create backup of existing config
            if self.config_file.exists():
                backup_file = self.config_file.with_suffix('.json.backup')
                backup_file.write_text(self.config_file.read_text())
            
            # Write new config
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
            
            logger.debug(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def add_project(self, project_name: str, project_path: str, auto_watch: bool = True) -> bool:
        """
        Add a project to the configuration
        
        Args:
            project_name: Unique project identifier
            project_path: Absolute path to project directory
            auto_watch: Whether to automatically start watching on server startup
            
        Returns:
            True if project was added successfully
        """
        try:
            project_path = str(Path(project_path).expanduser().resolve())
            
            if not Path(project_path).exists():
                logger.error(f"Project path does not exist: {project_path}")
                return False
            
            self.config["projects"][project_name] = {
                "path": project_path,
                "last_indexed": time.time(),
                "auto_watch": auto_watch
            }
            
            success = self.save_config()
            if success:
                logger.info(f"Added project '{project_name}' at {project_path}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to add project {project_name}: {e}")
            return False
    
    def remove_project(self, project_name: str) -> bool:
        """
        Remove a project from the configuration
        
        Args:
            project_name: Project identifier to remove
            
        Returns:
            True if project was removed successfully
        """
        try:
            if project_name in self.config["projects"]:
                removed_project = self.config["projects"].pop(project_name)
                success = self.save_config()
                if success:
                    logger.info(f"Removed project '{project_name}' (was at {removed_project['path']})")
                return success
            else:
                logger.warning(f"Project '{project_name}' not found in configuration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove project {project_name}: {e}")
            return False
    
    def get_project(self, project_name: str) -> Optional[Dict]:
        """Get project configuration by name"""
        return self.config["projects"].get(project_name)
    
    def get_all_projects(self) -> Dict[str, Dict]:
        """Get all project configurations"""
        return self.config["projects"].copy()
    
    def get_projects_for_auto_watch(self) -> Dict[str, str]:
        """
        Get projects that should be automatically watched on startup
        
        Returns:
            Dict mapping project_name -> project_path for auto-watch projects
        """
        auto_watch_projects = {}
        for project_name, project_info in self.config["projects"].items():
            if project_info.get("auto_watch", True):
                auto_watch_projects[project_name] = project_info["path"]
        
        logger.info(f"Found {len(auto_watch_projects)} projects configured for auto-watch")
        return auto_watch_projects
    
    def update_project_indexed_time(self, project_name: str) -> bool:
        """Update the last indexed timestamp for a project"""
        try:
            if project_name in self.config["projects"]:
                self.config["projects"][project_name]["last_indexed"] = time.time()
                return self.save_config()
            else:
                logger.warning(f"Cannot update indexed time: project '{project_name}' not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update indexed time for {project_name}: {e}")
            return False
    
    def get_server_config(self) -> Dict:
        """Get server configuration"""
        return self.config["server"].copy()
    
    def update_server_config(self, **kwargs) -> bool:
        """
        Update server configuration
        
        Args:
            **kwargs: Server config keys to update (host, port, etc.)
            
        Returns:
            True if configuration was updated successfully
        """
        try:
            self.config["server"].update(kwargs)
            success = self.save_config()
            if success:
                logger.info(f"Updated server configuration: {kwargs}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to update server configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict:
        """Get a summary of the current configuration"""
        return {
            "config_file": str(self.config_file),
            "version": self.config["version"],
            "total_projects": len(self.config["projects"]),
            "auto_watch_projects": len(self.get_projects_for_auto_watch()),
            "server": self.config["server"],
            "created_at": self.config["created_at"],
            "updated_at": self.config["updated_at"]
        }

# Example usage and testing
if __name__ == "__main__":
    def test_config_manager():
        """Test the configuration manager"""
        print("🔧 Testing Configuration Manager...")
        
        import tempfile
        
        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_manager = ConfigManager(tmp_dir)
            
            # Add a test project
            test_project_path = Path.cwd()  # Use current directory
            success = config_manager.add_project("test-project", str(test_project_path))
            print(f"   ✅ Added project: {success}")
            
            # Get projects for auto-watch
            auto_watch = config_manager.get_projects_for_auto_watch()
            print(f"   📊 Auto-watch projects: {list(auto_watch.keys())}")
            
            # Update indexed time
            config_manager.update_project_indexed_time("test-project")
            print("   ⏰ Updated indexed time")
            
            # Get summary
            summary = config_manager.get_config_summary()
            print(f"   📋 Config summary: {summary['total_projects']} projects")
            
            # Remove project
            removed = config_manager.remove_project("test-project")
            print(f"   🗑️ Removed project: {removed}")
            
        print("✅ Configuration manager test completed")
    
    test_config_manager()