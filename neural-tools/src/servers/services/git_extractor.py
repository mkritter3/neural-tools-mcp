#!/usr/bin/env python3
"""
Git metadata extraction service
Extracts commit history, author info, and change frequency
"""

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class GitMetadataExtractor:
    """Extract Git metadata for files"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self._cache = {}  # Cache to avoid repeated git calls
        self._git_available = self._check_git_available()
    
    def _check_git_available(self) -> bool:
        """Check if the project is a git repository"""
        try:
            git_dir = self.project_path / ".git"
            return git_dir.exists() and git_dir.is_dir()
        except:
            return False
    
    async def extract(self, file_path: str) -> Dict:
        """Extract git metadata for a file"""
        if not self._git_available:
            return self._default_metadata()
            
        if file_path in self._cache:
            return self._cache[file_path]
        
        try:
            # Convert to Path and make relative
            file_path_obj = Path(file_path)
            if file_path_obj.is_absolute():
                try:
                    relative_path = file_path_obj.relative_to(self.project_path)
                except ValueError:
                    # File is outside project path
                    return self._default_metadata()
            else:
                relative_path = file_path_obj
            
            # Get last modified date
            result = await self._run_git_command(
                ['git', 'log', '-1', '--format=%aI', str(relative_path)]
            )
            last_modified = result.strip() if result else datetime.now().isoformat()
            
            # Get change frequency (commits in last 30 days)
            thirty_days_ago = datetime.now().timestamp() - (30 * 24 * 3600)
            result = await self._run_git_command(
                ['git', 'rev-list', '--count', f'--since={int(thirty_days_ago)}', 
                 'HEAD', '--', str(relative_path)]
            )
            change_frequency = int(result.strip()) if result and result.strip().isdigit() else 0
            
            # Get unique author count
            result = await self._run_git_command(
                ['git', 'log', '--format=%ae', str(relative_path)]
            )
            authors = set(result.strip().split('\n')) if result and result.strip() else set()
            author_count = len([a for a in authors if a])  # Filter empty strings
            
            # Get last commit hash
            result = await self._run_git_command(
                ['git', 'log', '-1', '--format=%H', str(relative_path)]
            )
            last_commit = result.strip()[:8] if result else 'unknown'
            
            metadata = {
                'last_modified': last_modified,
                'change_frequency': change_frequency,
                'author_count': author_count,
                'last_commit': last_commit
            }
            
            self._cache[file_path] = metadata
            return metadata
            
        except Exception as e:
            logger.debug(f"Git metadata extraction failed for {file_path}: {e}")
            return self._default_metadata()
    
    def _default_metadata(self) -> Dict:
        """Return default metadata when git is unavailable"""
        return {
            'last_modified': datetime.now().isoformat(),
            'change_frequency': 0,
            'author_count': 1,
            'last_commit': 'unknown'
        }
    
    async def _run_git_command(self, cmd: list) -> Optional[str]:
        """Run git command asynchronously"""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode('utf-8')
        except asyncio.TimeoutError:
            logger.debug(f"Git command timed out: {' '.join(cmd)}")
            return None
        except Exception as e:
            logger.debug(f"Git command failed: {' '.join(cmd)} - {e}")
            return None