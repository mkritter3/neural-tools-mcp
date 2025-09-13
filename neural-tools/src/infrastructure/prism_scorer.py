#!/usr/bin/env python3
"""
PRISM: Priority-Ranked Intelligence Scoring Model
Shared module for intelligent code importance scoring
Used by both SessionStart hooks and MCP search tools
"""

import ast
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict

class PrismScorer:
    """PRISM: Priority-Ranked Intelligence Scoring Model
    Advanced importance scoring for intelligent file selection and search ranking"""
    
    # Class-level weights for consistency across instances
    WEIGHTS = {
        'complexity': 0.30,      # Static analysis: complexity metrics
        'dependencies': 0.25,    # Dependency relationships
        'recency': 0.25,        # Dynamic: recent changes
        'contextual': 0.20      # Contextual: user patterns, business criticality
    }
    
    def __init__(self, project_root: str, cache_dir: Optional[str] = None):
        self.project_root = Path(project_root)
        
        # Allow custom cache directory (useful for Docker environments)
        if cache_dir:
            self.cache_path = Path(cache_dir) / "prism_cache.json"
        else:
            self.cache_path = self.project_root / ".claude" / "prism_cache.json"
            
        self.session_log_path = self.project_root / ".claude" / "session_activity.jsonl"
        self.cache = self._load_cache()
        self.dependency_graph = self._build_dependency_graph()
        
    def _load_cache(self) -> Dict[str, Any]:
        """Load cached scores"""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError):
            pass
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except IOError:
            pass
    
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build inverse dependency graph (who imports this file)"""
        graph = defaultdict(set)
        
        try:
            # Quick AST-based dependency analysis
            for py_file in self.project_root.rglob('*.py'):
                if any(skip in str(py_file) for skip in ['site-packages', '.venv', '__pycache__']):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read(), filename=str(py_file))
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            if isinstance(node, ast.ImportFrom) and node.module:
                                # Track which files this imports
                                module_path = node.module.replace('.', '/')
                                imported_file = self.project_root / f"{module_path}.py"
                                if imported_file.exists():
                                    graph[str(imported_file)].add(str(py_file))
                except:
                    continue
        except:
            pass
        
        return dict(graph)
    
    def get_cached_score(self, file_path: str, score_type: str) -> Optional[float]:
        """Get cached score if still valid"""
        file_info = self.cache.get(file_path)
        if not file_info:
            return None
        
        try:
            current_mtime = os.path.getmtime(file_path)
            if file_info.get("mtime") == current_mtime and score_type in file_info:
                return file_info[score_type]
        except FileNotFoundError:
            return None
        return None
    
    def update_cache(self, file_path: str, score_type: str, score: float):
        """Update cache with new score"""
        if file_path not in self.cache:
            self.cache[file_path] = {}
        try:
            self.cache[file_path]["mtime"] = os.path.getmtime(file_path)
            self.cache[file_path][score_type] = score
        except FileNotFoundError:
            pass
    
    def calculate_complexity_score(self, file_path: str) -> float:
        """Calculate complexity score using cyclomatic complexity approximation"""
        cached = self.get_cached_score(file_path, 'complexity')
        if cached is not None:
            return cached
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple complexity heuristics (without radon dependency)
            tree = ast.parse(content)
            
            # Count control flow elements
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
                elif isinstance(node, ast.FunctionDef):
                    complexity += 1
            
            # Count nesting depth (cognitive complexity)
            max_depth = self._calculate_max_nesting(tree)
            complexity += max_depth * 2
            
            # Normalize: complexity of 20+ is high
            score = 1 / (1 + math.exp(-0.1 * (complexity - 10)))
            
            self.update_cache(file_path, 'complexity', score)
            return score
        except Exception:
            return 0.0
    
    def _calculate_max_nesting(self, tree: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        max_depth = depth
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.FunctionDef, ast.ClassDef)):
                child_depth = self._calculate_max_nesting(node, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
    
    def calculate_dependency_score(self, file_path: str) -> float:
        """Calculate dependency score based on how many files import this one"""
        # In-degree: how many files depend on this file
        in_degree = len(self.dependency_graph.get(file_path, set()))
        
        # Normalize using logistic function
        # Files imported by 5+ others are important
        score = 1 / (1 + math.exp(-0.3 * (in_degree - 3)))
        return score
    
    def calculate_recency_score(self, file_path: str) -> float:
        """Calculate recency score based on modification time"""
        try:
            mtime = os.path.getmtime(file_path)
            age_seconds = time.time() - mtime
            age_days = age_seconds / (24 * 3600)
            
            # Score decays over time
            # Files modified today: ~1.0
            # Files modified week ago: ~0.5
            # Files modified month ago: ~0.1
            score = math.exp(-0.1 * age_days)
            return score
        except FileNotFoundError:
            return 0.0
    
    def calculate_contextual_score(self, file_path: str) -> float:
        """Calculate contextual score based on patterns and criticality"""
        path = Path(file_path)
        score = 0.0
        
        # Business criticality indicators
        critical_patterns = {
            'auth': 0.9,
            'payment': 0.9,
            'security': 0.9,
            'api': 0.8,
            'main': 0.8,
            'core': 0.7,
            'model': 0.7,
            'service': 0.6,
            'neural': 0.7,
            'mcp': 0.7,
            'hook': 0.6,
            'util': 0.4,
            'test': 0.3,
            'mock': 0.2
        }
        
        name_lower = path.stem.lower()
        for pattern, weight in critical_patterns.items():
            if pattern in name_lower:
                score = max(score, weight)
                break
        
        # Check for session interaction (simplified without log file)
        # In production, would read from session_activity.jsonl
        if self.session_log_path.exists():
            try:
                with open(self.session_log_path, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("path") == str(file_path):
                                score = max(score, 0.9)  # Recently accessed
                                break
                        except json.JSONDecodeError:
                            continue
            except IOError:
                pass
        
        return score
    
    def calculate_importance_score(self, file_path: str) -> float:
        """Calculate overall PRISM importance score"""
        scores = {
            'complexity': self.calculate_complexity_score(file_path),
            'dependencies': self.calculate_dependency_score(file_path),
            'recency': self.calculate_recency_score(file_path),
            'contextual': self.calculate_contextual_score(file_path)
        }
        
        # Weighted sum
        total_score = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        
        # Store composite score
        self.update_cache(file_path, 'prism_score', total_score)
        
        return total_score
    
    def get_score_components(self, file_path: str) -> Dict[str, float]:
        """Get individual score components for detailed analysis"""
        return {
            'complexity': self.calculate_complexity_score(file_path),
            'dependencies': self.calculate_dependency_score(file_path),
            'recency': self.calculate_recency_score(file_path),
            'contextual': self.calculate_contextual_score(file_path),
            'total': self.calculate_importance_score(file_path)
        }
    
    def boost_search_results(self, search_results: List[Dict], boost_factor: float = 0.3) -> List[Dict]:
        """Boost search results with PRISM scores
        
        Args:
            search_results: List of search results with 'file_path' and 'score' keys
            boost_factor: How much to blend PRISM score (0.0 = no boost, 1.0 = only PRISM)
        
        Returns:
            Re-ranked results with combined scores
        """
        boosted_results = []
        
        for result in search_results:
            file_path = result.get('file_path', result.get('path', ''))
            if file_path:
                # Get PRISM score
                prism_score = self.calculate_importance_score(file_path)
                
                # Combine with original score
                original_score = result.get('score', 0.5)
                combined_score = (1 - boost_factor) * original_score + boost_factor * prism_score
                
                # Add PRISM details to result
                result_copy = result.copy()
                result_copy['score'] = combined_score
                result_copy['original_score'] = original_score
                result_copy['prism_score'] = prism_score
                result_copy['prism_components'] = self.get_score_components(file_path)
                
                boosted_results.append(result_copy)
            else:
                boosted_results.append(result)
        
        # Re-sort by combined score
        boosted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return boosted_results
    
    def get_top_files(self, n: int = 10, file_pattern: str = '**/*.py') -> List[Tuple[str, float]]:
        """Get top N files by importance score"""
        file_scores = []
        
        for file_path in self.project_root.glob(file_pattern):
            if file_path.is_file():
                # Skip vendor/generated files
                if any(skip in str(file_path) for skip in ['site-packages', '.venv', '__pycache__', '.git']):
                    continue
                
                score = self.calculate_importance_score(str(file_path))
                file_scores.append((str(file_path), score))
        
        # Sort by score descending
        file_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Save cache after scoring
        self._save_cache()
        
        return file_scores[:n]
    
    def explain_score(self, file_path: str) -> str:
        """Generate human-readable explanation of a file's PRISM score"""
        components = self.get_score_components(file_path)
        
        explanation = f"PRISM Score Analysis for {Path(file_path).name}:\n"
        explanation += f"  Total Score: {components['total']:.3f}\n"
        explanation += f"  - Complexity: {components['complexity']:.3f} (30% weight)\n"
        explanation += f"  - Dependencies: {components['dependencies']:.3f} (25% weight)\n"
        explanation += f"  - Recency: {components['recency']:.3f} (25% weight)\n"
        explanation += f"  - Contextual: {components['contextual']:.3f} (20% weight)\n"
        
        # Add interpretation
        if components['total'] >= 0.8:
            explanation += "  Importance: CRITICAL - Essential file for the project"
        elif components['total'] >= 0.6:
            explanation += "  Importance: HIGH - Significant file in the codebase"
        elif components['total'] >= 0.4:
            explanation += "  Importance: MEDIUM - Moderately important file"
        else:
            explanation += "  Importance: LOW - Supporting or utility file"
        
        return explanation