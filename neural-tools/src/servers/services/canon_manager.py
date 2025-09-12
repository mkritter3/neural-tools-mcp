#!/usr/bin/env python3
"""
Canonical Knowledge Manager
Manages .canon.yaml configuration and applies canonical weights
"""

import yaml
from pathlib import Path
from typing import Dict, Optional
import fnmatch
import logging

logger = logging.getLogger(__name__)

class CanonManager:
    """Manages canonical knowledge configuration and application"""
    
    def __init__(self, project_name: str, project_path: str = None):
        self.project_name = project_name
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.canon_config = None
        self.patterns_compiled = {}
        self._load_config()
        
    def _load_config(self):
        """Load .canon.yaml from project root"""
        canon_path = self.project_path / ".canon.yaml"
        if canon_path.exists():
            try:
                with open(canon_path) as f:
                    self.canon_config = yaml.safe_load(f)
                    self._compile_patterns()
                logger.info(f"Loaded .canon.yaml for project {self.project_name}")
            except Exception as e:
                logger.error(f"Failed to load .canon.yaml: {e}")
        else:
            logger.debug(f"No .canon.yaml found for project {self.project_name}")
    
    def _compile_patterns(self):
        """Compile glob patterns for efficient matching"""
        if not self.canon_config:
            return
            
        for level in ['primary', 'secondary', 'reference', 'deprecated', 'experimental']:
            if level in self.canon_config:
                for entry in self.canon_config[level]:
                    if 'pattern' in entry:
                        pattern = entry['pattern']
                        self.patterns_compiled[pattern] = {
                            'level': level,
                            'weight': entry.get('weight', self._default_weight(level)),
                            'description': entry.get('description', '')
                        }
    
    def _default_weight(self, level: str) -> float:
        """Get default weight for canon level"""
        weights = {
            'primary': 1.0,
            'secondary': 0.7,
            'reference': 0.4,
            'deprecated': 0.1,
            'experimental': 0.3
        }
        return weights.get(level, 0.5)
        
    async def get_file_metadata(self, file_path: str) -> Dict:
        """Get canonical metadata for a file"""
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
            
            # Check explicit paths first
            if self.canon_config:
                for level in ['primary', 'secondary', 'reference', 'deprecated', 'experimental']:
                    if level in self.canon_config:
                        for entry in self.canon_config[level]:
                            if 'path' in entry and entry['path'] == str(relative_path):
                                return {
                                    'level': level,
                                    'weight': entry.get('weight', self._default_weight(level)),
                                    'reason': entry.get('description', ''),
                                    'source': 'explicit_path'
                                }
            
            # Check patterns
            for pattern, metadata in self.patterns_compiled.items():
                if fnmatch.fnmatch(str(relative_path), pattern):
                    return {
                        'level': metadata['level'],
                        'weight': metadata['weight'],
                        'reason': metadata['description'],
                        'source': 'pattern_match'
                    }
            
            return self._default_metadata()
            
        except Exception as e:
            logger.error(f"Error getting canon metadata for {file_path}: {e}")
            return self._default_metadata()
    
    def _default_metadata(self) -> Dict:
        """Default metadata when no canon config matches"""
        return {
            'level': 'none',
            'weight': 0.5,
            'reason': 'Not in canonical configuration',
            'source': 'default'
        }
    
    def has_config(self) -> bool:
        """Check if canon configuration exists"""
        return self.canon_config is not None
    
    def get_config_summary(self) -> Dict:
        """Get summary of canon configuration"""
        if not self.canon_config:
            return {'configured': False}
        
        summary = {'configured': True, 'levels': {}}
        for level in ['primary', 'secondary', 'reference', 'deprecated', 'experimental']:
            if level in self.canon_config:
                count = len(self.canon_config[level])
                summary['levels'][level] = count
        
        return summary