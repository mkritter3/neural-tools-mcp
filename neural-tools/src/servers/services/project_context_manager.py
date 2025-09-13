#!/usr/bin/env python3
"""
Project Context Manager - Dynamic Workspace Detection
Implements ADR-0033 for true multi-project support within MCP constraints

This manager solves the fundamental problem that MCP servers cannot detect
which project a user is working on due to hardcoded working directories.
It provides both explicit project setting and smart auto-detection.

Author: L9 Engineering Team
Date: 2025-09-12
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class ProjectContextManager:
    """
    Manages dynamic project context detection and switching.
    Implements ADR-0033 for multi-project support.
    
    Key Features:
    - Explicit project setting via set_project_context tool
    - Smart auto-detection from usage patterns
    - Project registry persistence
    - Confidence-based detection
    - Integration with IndexerOrchestrator
    """
    
    def __init__(self):
        self.current_project: Optional[str] = None
        self.current_project_path: Optional[Path] = None
        self.project_registry: Dict[str, Path] = {}
        self.detection_hints: List[str] = []  # Recent file accesses
        self.last_activity: Dict[str, datetime] = {}
        self._instance_id = None  # Will be set by MCP server
        self._load_registry()
        logger.info("🎯 ProjectContextManager initialized")
    
    def set_instance_id(self, instance_id: str):
        """Set the MCP instance ID for logging"""
        self._instance_id = instance_id
    
    def _get_registry_path(self) -> Path:
        """Get path to project registry file"""
        return Path.home() / ".claude" / "project_registry.json"
    
    def _load_registry(self):
        """Load known projects from persistent storage"""
        registry_path = self._get_registry_path()
        if registry_path.exists():
            try:
                with open(registry_path) as f:
                    data = json.load(f)
                    self.project_registry = {
                        k: Path(v) for k, v in data.get("projects", {}).items()
                    }
                    # Load last activity timestamps
                    for k, v in data.get("last_activity", {}).items():
                        try:
                            self.last_activity[k] = datetime.fromisoformat(v)
                        except:
                            pass
                    logger.info(f"📚 Loaded {len(self.project_registry)} projects from registry")
            except Exception as e:
                logger.error(f"Failed to load project registry: {e}")
    
    def _save_registry(self):
        """Persist project registry"""
        registry_path = self._get_registry_path()
        try:
            registry_path.parent.mkdir(exist_ok=True, parents=True)
            
            data = {
                "projects": {k: str(v) for k, v in self.project_registry.items()},
                "last_activity": {
                    k: v.isoformat() for k, v in self.last_activity.items()
                },
                "version": "1.0",
                "updated": datetime.now().isoformat()
            }
            
            # Write atomically
            temp_path = registry_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            temp_path.replace(registry_path)
            
            logger.debug(f"💾 Saved project registry with {len(self.project_registry)} projects")
        except Exception as e:
            logger.error(f"Failed to save project registry: {e}")
    
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
        project_name = await self._detect_project_name(project_path)
        
        # Update state
        old_project = self.current_project
        self.current_project = project_name
        self.current_project_path = project_path
        self.project_registry[project_name] = project_path
        self.last_activity[project_name] = datetime.now()
        self._save_registry()
        
        if old_project and old_project != project_name:
            logger.info(f"🔄 [Instance {self._instance_id}] Switched from {old_project} to {project_name}")
        else:
            logger.info(f"📍 [Instance {self._instance_id}] Set project context to {project_name}")
        
        return {
            "project": project_name,
            "path": str(project_path),
            "method": "explicit",
            "confidence": 1.0,  # Explicit set is 100% confidence
            "timestamp": datetime.now().isoformat(),
            "previous": old_project
        }
    
    async def _detect_project_name(self, path: Path) -> str:
        """
        Detect project name from path using multiple strategies.
        
        Priority:
        1. package.json name field (Node projects)
        2. pyproject.toml name field (Python projects)
        3. .git config remote origin
        4. Directory name (sanitized)
        """
        logger.debug(f"🔍 [Detection] Starting project name detection for path: {path}")
        # Try package.json (Node/JavaScript projects)
        package_json = path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    if "name" in data:
                        name = self._sanitize_name(data["name"])
                        if name and name != "default":
                            logger.debug(f"✅ [Detection] Found from package.json: '{data['name']}' → sanitized: '{name}'")
                            return name
            except Exception as e:
                logger.debug(f"Failed to parse package.json: {e}")
        
        # Try pyproject.toml (Python projects)
        pyproject = path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, 'rb') as f:
                    data = tomllib.load(f)
                    if "project" in data and "name" in data["project"]:
                        name = self._sanitize_name(data["project"]["name"])
                        if name and name != "default":
                            logger.debug(f"✅ [Detection] Found from pyproject.toml: '{data['project']['name']}' → sanitized: '{name}'")
                            return name
            except Exception as e:
                logger.debug(f"Failed to parse pyproject.toml: {e}")
        
        # Try git remote (if it's a git repo)
        git_config = path / ".git" / "config"
        if git_config.exists():
            try:
                with open(git_config) as f:
                    content = f.read()
                    # Look for remote origin URL
                    import re
                    # Match patterns like: url = git@github.com:user/repo.git
                    # or: url = https://github.com/user/repo.git
                    pattern = r'url\s*=\s*(?:git@[^:]+:|https?://[^/]+/)([^/]+)/([^/\s]+?)(?:\.git)?$'
                    match = re.search(pattern, content, re.MULTILINE)
                    if match:
                        repo_name = match.group(2)
                        name = self._sanitize_name(repo_name)
                        if name and name != "default":
                            logger.debug(f"✅ [Detection] Found from git config: '{repo_name}' → sanitized: '{name}'")
                            return name
            except Exception as e:
                logger.debug(f"Failed to parse git config: {e}")
        
        # Fall back to directory name
        name = self._sanitize_name(path.name)
        logger.debug(f"🏠 [Detection] Using directory name as fallback: '{path.name}' → sanitized: '{name}'")
        return name
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize project name for use as identifier"""
        # Handle scoped packages like @company/package
        if name.startswith('@'):
            # Take the package name after the scope
            parts = name.split('/')
            if len(parts) > 1:
                name = parts[1]
        
        # Replace non-alphanumeric with hyphen (keep hyphens)
        sanitized = re.sub(r'[^a-z0-9-]', '-', name.lower())
        
        # Remove multiple consecutive hyphens
        sanitized = re.sub(r'-+', '-', sanitized)
        
        # Remove leading/trailing hyphens
        sanitized = sanitized.strip('-')
        
        return sanitized or "default"
    
    async def detect_project(self) -> Dict:
        """
        Auto-detect active project using heuristics.
        
        Detection strategies:
        1. Current working directory (highest priority for MCP)
        2. Most recently accessed project
        3. Project containing recently accessed files
        4. Scan common directories for projects
        5. Fall back to "default"
        """
        # Strategy 0: Always try current working directory first (critical for MCP)
        current_dir = Path.cwd()
        try:
            current_project_name = await self._detect_project_name(current_dir)
            if current_project_name and current_project_name != "default":
                # Update state and registry
                self.current_project = current_project_name
                self.current_project_path = current_dir
                self.project_registry[current_project_name] = current_dir
                self.last_activity[current_project_name] = datetime.now()
                self._save_registry()
                
                logger.info(f"🎯 Detected project from current directory: {current_project_name}")
                return {
                    "project": current_project_name,
                    "path": str(current_dir),
                    "method": "current_directory",
                    "confidence": 0.95
                }
        except Exception as e:
            logger.debug(f"Failed to detect from current directory: {e}")
        
        # Strategy 1: Already have a current project (but lower priority than cwd)
        if self.current_project and self.current_project_path:
            # Verify the path still exists
            if self.current_project_path.exists():
                logger.info(f"📌 Using cached project: {self.current_project}")
                return {
                    "project": self.current_project,
                    "path": str(self.current_project_path),
                    "method": "cached",
                    "confidence": 0.9
                }
            else:
                # Clear stale cache
                self.current_project = None
                self.current_project_path = None
        
        # Strategy 2: Most recent activity (only if not stale)
        if self.last_activity:
            most_recent = max(
                self.last_activity.items(),
                key=lambda x: x[1]
            )
            project_name = most_recent[0]
            # Only use if recent (within last hour) and path exists
            if (project_name in self.project_registry and 
                (datetime.now() - most_recent[1]).seconds < 3600 and
                self.project_registry[project_name].exists()):
                
                self.current_project = project_name
                self.current_project_path = self.project_registry[project_name]
                logger.info(f"🕐 Detected project from recent activity: {project_name}")
                return {
                    "project": project_name,
                    "path": str(self.current_project_path),
                    "method": "recent_activity",
                    "confidence": 0.7
                }
        
        # Strategy 3: Check detection hints (files recently accessed)
        for hint_path in reversed(self.detection_hints[-10:]):  # Last 10 hints, newest first
            hint = Path(hint_path)
            for project_name, project_path in self.project_registry.items():
                try:
                    if project_path.exists():
                        hint.relative_to(project_path)
                        # File is within this project
                        self.current_project = project_name
                        self.current_project_path = project_path
                        self.last_activity[project_name] = datetime.now()
                        logger.info(f"📁 Detected project from file access: {project_name}")
                        return {
                            "project": project_name,
                            "path": str(project_path),
                            "method": "file_access_pattern",
                            "confidence": 0.6
                        }
                except ValueError:
                    continue
        
        # Strategy 4: Scan common directories for new projects
        await self._scan_for_projects()
        
        # Strategy 5: Use first valid registered project
        if self.project_registry:
            for project_name, project_path in self.project_registry.items():
                if project_path.exists():
                    self.current_project = project_name
                    self.current_project_path = project_path
                    logger.info(f"📌 Using first valid registered project: {project_name}")
                    return {
                        "project": project_name,
                        "path": str(self.current_project_path),
                        "method": "first_registered",
                        "confidence": 0.4
                    }
        
        # Strategy 6: Fall back to current directory detection
        fallback_name = await self._detect_project_name(current_dir)
        self.current_project = fallback_name
        self.current_project_path = current_dir
        logger.warning(f"🏠 Falling back to current directory: {fallback_name}")
        return {
            "project": fallback_name,
            "path": str(self.current_project_path),
            "method": "fallback",
            "confidence": 0.2 if fallback_name == "default" else 0.3
        }
    
    async def _scan_for_projects(self):
        """Scan common directories for projects"""
        common_dirs = [
            Path.home() / "local-coding",
            Path.home() / "projects",
            Path.home() / "code",
            Path.home() / "Documents" / "code",
            Path.home() / "workspace",
            Path.cwd()  # Current directory as last resort
        ]
        
        new_projects = 0
        for base_dir in common_dirs:
            if base_dir.exists() and base_dir.is_dir():
                try:
                    # Only scan one level deep to avoid performance issues
                    for child in base_dir.iterdir():
                        if child.is_dir() and not child.name.startswith('.'):
                            # Check if it's a project (has git, package.json, requirements.txt, etc.)
                            if any([
                                (child / ".git").exists(),
                                (child / "package.json").exists(),
                                (child / "requirements.txt").exists(),
                                (child / "pyproject.toml").exists(),
                                (child / "Cargo.toml").exists(),
                                (child / "go.mod").exists(),
                            ]):
                                project_name = await self._detect_project_name(child)
                                if project_name not in self.project_registry:
                                    self.project_registry[project_name] = child
                                    new_projects += 1
                                    logger.debug(f"Discovered project: {project_name} at {child}")
                except PermissionError:
                    continue
        
        if new_projects > 0:
            self._save_registry()
            logger.info(f"🔍 Discovered {new_projects} new projects")
    
    def add_hint(self, file_path: str):
        """Add a file access hint for project detection"""
        self.detection_hints.append(file_path)
        # Keep only last 100 hints
        if len(self.detection_hints) > 100:
            self.detection_hints = self.detection_hints[-100:]
    
    async def get_current_project(self) -> Dict:
        """Get current project context, auto-detecting if needed"""
        if self.current_project and self.current_project_path:
            return {
                "project": self.current_project,
                "path": str(self.current_project_path),
                "method": "cached"
            }
        else:
            return await self.detect_project()
    
    async def list_projects(self) -> List[Dict]:
        """List all known projects"""
        # Scan for new projects first
        await self._scan_for_projects()
        
        projects = []
        for name, path in self.project_registry.items():
            last_activity = self.last_activity.get(name)
            projects.append({
                "name": name,
                "path": str(path),
                "last_activity": last_activity.isoformat() if last_activity else "never",
                "is_current": name == self.current_project,
                "exists": path.exists()  # Check if project still exists
            })
        
        # Sort by last activity (most recent first), then by name
        projects.sort(key=lambda x: (
            x["last_activity"] != "never",  # Active projects first
            x["last_activity"] if x["last_activity"] != "never" else "",
            x["name"]
        ), reverse=True)
        
        return projects
    
    async def switch_project(self, project_name: str) -> Dict:
        """
        Switch to a known project by name.
        
        Args:
            project_name: Name of a registered project
            
        Returns:
            Project info or error if not found
        """
        if project_name in self.project_registry:
            project_path = self.project_registry[project_name]
            if project_path.exists():
                return await self.set_project(str(project_path))
            else:
                raise ValueError(f"Project path no longer exists: {project_path}")
        else:
            # Try to find similar project names
            similar = [p for p in self.project_registry.keys() 
                      if project_name.lower() in p.lower()]
            if similar:
                raise ValueError(f"Project '{project_name}' not found. "
                               f"Did you mean: {', '.join(similar)}?")
            else:
                raise ValueError(f"Project '{project_name}' not found. "
                               f"Use list_projects to see available projects.")
    
    def clear_hints(self):
        """Clear detection hints (useful for testing)"""
        self.detection_hints.clear()
        logger.debug("Cleared detection hints")