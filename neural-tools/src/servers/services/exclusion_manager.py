"""
Exclusion Manager for GraphRAG Search
L9 2025 Architecture - ADR-0022 Implementation

Manages file exclusion patterns and dynamic re-ranking to prevent
archived/deprecated code from polluting search results.
"""

import fnmatch
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ExclusionRule:
    """Represents a single exclusion rule"""
    pattern: str
    rule_type: str  # 'path', 'glob', 'regex'
    action: str  # 'exclude', 'deprioritize'
    weight_penalty: float = 0.0  # For deprioritize action


class ExclusionManager:
    """
    Manages exclusion patterns and re-ranking for GraphRAG search
    Following ADR-0022 specifications
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.exclusion_rules: List[ExclusionRule] = []
        self.default_patterns = self._get_default_patterns()
        self.graphragignore_path = self.project_path / '.graphragignore'
        
        # Load exclusion patterns
        self._load_exclusion_patterns()
        
    def _get_default_patterns(self) -> List[ExclusionRule]:
        """Get default exclusion patterns"""
        return [
            # Archived and deprecated directories
            ExclusionRule('.archive/**', 'glob', 'exclude'),
            ExclusionRule('.deprecated/**', 'glob', 'exclude'),
            ExclusionRule('**/.archive/**', 'glob', 'exclude'),
            ExclusionRule('**/.deprecated/**', 'glob', 'exclude'),
            ExclusionRule('**/archived/**', 'glob', 'exclude'),
            ExclusionRule('**/deprecated/**', 'glob', 'exclude'),
            
            # Backup files
            ExclusionRule('**/*_backup.*', 'glob', 'deprioritize', 0.5),
            ExclusionRule('**/*.bak', 'glob', 'deprioritize', 0.5),
            ExclusionRule('**/*.old', 'glob', 'deprioritize', 0.5),
            ExclusionRule('**/backup/**', 'glob', 'deprioritize', 0.3),
            
            # Temporary and cache directories
            ExclusionRule('**/tmp/**', 'glob', 'exclude'),
            ExclusionRule('**/temp/**', 'glob', 'exclude'),
            ExclusionRule('**/__pycache__/**', 'glob', 'exclude'),
            ExclusionRule('**/.cache/**', 'glob', 'exclude'),
            
            # Build artifacts
            ExclusionRule('**/dist/**', 'glob', 'exclude'),
            ExclusionRule('**/build/**', 'glob', 'exclude'),
            ExclusionRule('**/target/**', 'glob', 'exclude'),
            ExclusionRule('**/.next/**', 'glob', 'exclude'),
            
            # Version control
            ExclusionRule('**/.git/**', 'glob', 'exclude'),
            ExclusionRule('**/.svn/**', 'glob', 'exclude'),
            
            # Dependencies
            ExclusionRule('**/node_modules/**', 'glob', 'exclude'),
            ExclusionRule('**/vendor/**', 'glob', 'exclude'),
            ExclusionRule('**/.venv/**', 'glob', 'exclude'),
            ExclusionRule('**/venv/**', 'glob', 'exclude'),
            
            # IDE and editor files
            ExclusionRule('**/.idea/**', 'glob', 'exclude'),
            ExclusionRule('**/.vscode/**', 'glob', 'exclude'),
            ExclusionRule('**/*.swp', 'glob', 'exclude'),
            ExclusionRule('**/.DS_Store', 'glob', 'exclude'),
            
            # Test fixtures that are old
            ExclusionRule('**/fixtures_old/**', 'glob', 'deprioritize', 0.7),
            ExclusionRule('**/test_data_old/**', 'glob', 'deprioritize', 0.7),
        ]
    
    def _load_exclusion_patterns(self):
        """Load exclusion patterns from .graphragignore file"""
        self.exclusion_rules = self.default_patterns.copy()
        
        if self.graphragignore_path.exists():
            logger.info(f"Loading exclusion patterns from {self.graphragignore_path}")
            try:
                with open(self.graphragignore_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        
                        # Skip comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        
                        # Parse the rule
                        rule = self._parse_rule_line(line, line_num)
                        if rule:
                            self.exclusion_rules.append(rule)
                            
                logger.info(f"Loaded {len(self.exclusion_rules)} exclusion rules")
                
            except Exception as e:
                logger.error(f"Error loading .graphragignore: {e}")
    
    def _parse_rule_line(self, line: str, line_num: int) -> Optional[ExclusionRule]:
        """Parse a single rule line from .graphragignore"""
        try:
            # Check for deprioritize directive
            if line.startswith('!deprioritize:'):
                parts = line[14:].split(',')
                if len(parts) >= 2:
                    pattern = parts[0].strip()
                    weight = float(parts[1].strip())
                    return ExclusionRule(pattern, 'glob', 'deprioritize', weight)
            
            # Check for regex pattern
            elif line.startswith('regex:'):
                pattern = line[6:].strip()
                return ExclusionRule(pattern, 'regex', 'exclude')
            
            # Default glob pattern
            else:
                return ExclusionRule(line, 'glob', 'exclude')
                
        except Exception as e:
            logger.warning(f"Error parsing line {line_num}: {line} - {e}")
            return None
    
    def should_exclude(self, file_path: str) -> bool:
        """Check if a file should be completely excluded from indexing"""
        rel_path = self._get_relative_path(file_path)
        
        for rule in self.exclusion_rules:
            if rule.action != 'exclude':
                continue
                
            if self._matches_rule(rel_path, rule):
                logger.debug(f"Excluding {rel_path} due to rule: {rule.pattern}")
                return True
        
        return False
    
    def get_weight_penalty(self, file_path: str) -> float:
        """
        Get weight penalty for a file (0.0 = no penalty, 1.0 = maximum penalty)
        Used for deprioritization in search results
        """
        rel_path = self._get_relative_path(file_path)
        max_penalty = 0.0
        
        for rule in self.exclusion_rules:
            if rule.action != 'deprioritize':
                continue
                
            if self._matches_rule(rel_path, rule):
                max_penalty = max(max_penalty, rule.weight_penalty)
        
        return max_penalty
    
    def _matches_rule(self, rel_path: str, rule: ExclusionRule) -> bool:
        """Check if a path matches a rule"""
        if rule.rule_type == 'glob':
            return fnmatch.fnmatch(rel_path, rule.pattern)
        elif rule.rule_type == 'regex':
            return bool(re.match(rule.pattern, rel_path))
        elif rule.rule_type == 'path':
            return rel_path == rule.pattern
        return False
    
    def _get_relative_path(self, file_path: str) -> str:
        """Get relative path from project root"""
        try:
            path = Path(file_path)
            if path.is_absolute():
                return str(path.relative_to(self.project_path))
            return str(path)
        except ValueError:
            # Path is not relative to project_path
            return str(file_path)
    
    def filter_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter search results to remove excluded files
        Used before re-ranking
        """
        filtered = []
        excluded_count = 0
        
        for result in results:
            file_path = result.get('file_path', result.get('path', ''))
            if not self.should_exclude(file_path):
                filtered.append(result)
            else:
                excluded_count += 1
        
        if excluded_count > 0:
            logger.info(f"Filtered out {excluded_count} excluded files from search results")
        
        return filtered
    
    def apply_dynamic_reranking(
        self, 
        results: List[Dict[str, Any]], 
        recency_boost: bool = True,
        archive_penalty: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Apply dynamic re-ranking to search results
        Following ADR-0022 specifications
        """
        now = datetime.now()
        
        for result in results:
            original_score = result.get('score', 0.0)
            file_path = result.get('file_path', result.get('path', ''))
            
            # Start with original score
            adjusted_score = original_score
            
            # Apply archive/deprecation penalty
            if archive_penalty:
                penalty = self.get_weight_penalty(file_path)
                if penalty > 0:
                    adjusted_score *= (1.0 - penalty)
                    result['penalty_applied'] = penalty
            
            # Apply recency boost
            if recency_boost and 'last_modified' in result:
                try:
                    last_modified = datetime.fromisoformat(result['last_modified'])
                    days_old = (now - last_modified).days
                    
                    if days_old < 7:
                        boost = 1.2  # 20% boost for very recent
                    elif days_old < 30:
                        boost = 1.1  # 10% boost for recent
                    elif days_old < 90:
                        boost = 1.0  # No change
                    elif days_old < 365:
                        boost = 0.95  # 5% penalty for older
                    else:
                        boost = 0.9  # 10% penalty for very old
                    
                    adjusted_score *= boost
                    result['recency_boost'] = boost
                    
                except Exception as e:
                    logger.debug(f"Could not parse last_modified for {file_path}: {e}")
            
            # Store both scores for transparency
            result['original_score'] = original_score
            result['adjusted_score'] = adjusted_score
        
        # Sort by adjusted score (descending)
        results.sort(key=lambda x: x.get('adjusted_score', 0), reverse=True)
        
        return results
    
    def create_default_graphragignore(self):
        """Create a default .graphragignore file if it doesn't exist"""
        if not self.graphragignore_path.exists():
            logger.info(f"Creating default .graphragignore at {self.graphragignore_path}")
            
            default_content = """# GraphRAG Exclusion Patterns
# Format: glob patterns (one per line)
# Special directives:
#   !deprioritize:pattern,weight - Deprioritize matching files (weight 0-1)
#   regex:pattern - Use regex instead of glob

# Archived and deprecated code
.archive/**
.deprecated/**
**/.archive/**
**/.deprecated/**
**/archived/**
**/deprecated/**

# Backup files (deprioritize but don't exclude)
!deprioritize:**/*_backup.*,0.5
!deprioritize:**/*.bak,0.5
!deprioritize:**/*.old,0.5
!deprioritize:**/backup/**,0.3

# Temporary directories
**/tmp/**
**/temp/**
**/__pycache__/**
**/.cache/**

# Build artifacts
**/dist/**
**/build/**
**/target/**
**/.next/**

# Version control
**/.git/**
**/.svn/**

# Dependencies
**/node_modules/**
**/vendor/**
**/.venv/**
**/venv/**

# IDE files
**/.idea/**
**/.vscode/**
**/*.swp
**/.DS_Store

# Old test fixtures (deprioritize)
!deprioritize:**/fixtures_old/**,0.7
!deprioritize:**/test_data_old/**,0.7

# Custom patterns (add your own below)
"""
            
            try:
                with open(self.graphragignore_path, 'w') as f:
                    f.write(default_content)
                logger.info("Default .graphragignore created successfully")
                # Reload patterns
                self._load_exclusion_patterns()
            except Exception as e:
                logger.error(f"Error creating .graphragignore: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about exclusion rules"""
        exclude_count = sum(1 for r in self.exclusion_rules if r.action == 'exclude')
        deprioritize_count = sum(1 for r in self.exclusion_rules if r.action == 'deprioritize')
        
        return {
            'total_rules': len(self.exclusion_rules),
            'exclude_rules': exclude_count,
            'deprioritize_rules': deprioritize_count,
            'graphragignore_exists': self.graphragignore_path.exists(),
            'graphragignore_path': str(self.graphragignore_path)
        }