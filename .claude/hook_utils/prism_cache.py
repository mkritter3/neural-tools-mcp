"""
L9 PRISM Cache - Intelligent caching for PRISM scores with TTL
Provides multi-tier fallback for graceful degradation
"""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging


@dataclass
class CachedPrismScore:
    """Cached PRISM score with metadata"""
    file_path: str
    prism_score: float
    prism_components: Dict[str, float]
    cached_at: float
    file_mtime: float
    importance_label: str  # HIGH, MEDIUM, LOW


class PrismCache:
    """Intelligent caching for PRISM scores with TTL and invalidation"""
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl_hours: int = 24):
        self.cache_dir = cache_dir or (Path(__file__).parent.parent / '.claude')
        self.cache_file = self.cache_dir / 'prism_scores.json'
        self.ttl_seconds = ttl_hours * 3600
        
        self.logger = logging.getLogger("l9_prism_cache")
        self._cache: Dict[str, CachedPrismScore] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert dict back to CachedPrismScore objects
                for file_path, score_data in data.items():
                    try:
                        self._cache[file_path] = CachedPrismScore(**score_data)
                    except (TypeError, ValueError) as e:
                        self.logger.warning(f"Invalid cache entry for {file_path}: {e}")
                        continue
                        
                self.logger.info(f"Loaded {len(self._cache)} cached PRISM scores")
        except (IOError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load cache: {e}")
            self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert CachedPrismScore objects to dict
            data = {}
            for file_path, score in self._cache.items():
                data[file_path] = asdict(score)
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except IOError as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def _is_valid(self, cached_score: CachedPrismScore, file_path: str) -> bool:
        """Check if cached score is still valid"""
        # Check TTL
        if time.time() - cached_score.cached_at > self.ttl_seconds:
            return False
        
        # Check if file was modified since caching
        try:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > cached_score.file_mtime:
                return False
        except (OSError, FileNotFoundError):
            # File doesn't exist anymore
            return False
        
        return True
    
    def get(self, file_path: str) -> Optional[CachedPrismScore]:
        """Get cached PRISM score if valid"""
        if file_path not in self._cache:
            return None
        
        cached_score = self._cache[file_path]
        if self._is_valid(cached_score, file_path):
            return cached_score
        else:
            # Remove invalid entry
            del self._cache[file_path]
            return None
    
    def put(self, file_path: str, prism_score: float, 
            prism_components: Dict[str, float]) -> CachedPrismScore:
        """Cache PRISM score with metadata"""
        try:
            file_mtime = os.path.getmtime(file_path)
        except (OSError, FileNotFoundError):
            file_mtime = time.time()
        
        # Determine importance label
        importance_label = self._score_to_label(prism_score)
        
        cached_score = CachedPrismScore(
            file_path=file_path,
            prism_score=prism_score,
            prism_components=prism_components,
            cached_at=time.time(),
            file_mtime=file_mtime,
            importance_label=importance_label
        )
        
        self._cache[file_path] = cached_score
        self._save_cache()
        
        return cached_score
    
    def _score_to_label(self, score: float) -> str:
        """Convert PRISM score to importance label"""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_top_files(self, limit: int = 5, 
                      min_score: float = 0.0) -> List[CachedPrismScore]:
        """Get top cached files by PRISM score"""
        valid_scores = []
        
        for file_path, cached_score in self._cache.items():
            if (self._is_valid(cached_score, file_path) and 
                cached_score.prism_score >= min_score):
                valid_scores.append(cached_score)
        
        # Sort by PRISM score descending
        valid_scores.sort(key=lambda x: x.prism_score, reverse=True)
        
        return valid_scores[:limit]
    
    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed"""
        expired_keys = []
        
        for file_path, cached_score in self._cache.items():
            if not self._is_valid(cached_score, file_path):
                expired_keys.append(file_path)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._save_cache()
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = time.time()
        valid_count = 0
        expired_count = 0
        
        for file_path, cached_score in self._cache.items():
            if self._is_valid(cached_score, file_path):
                valid_count += 1
            else:
                expired_count += 1
        
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_count,
            'expired_entries': expired_count,
            'cache_file': str(self.cache_file),
            'ttl_hours': self.ttl_seconds / 3600
        }
    
    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
        except OSError as e:
            self.logger.error(f"Failed to delete cache file: {e}")
        
        self.logger.info("PRISM cache cleared")


class ModificationTimeFallback:
    """Fallback scoring based on file modification time and patterns"""
    
    @staticmethod
    def get_fallback_scores(project_dir: Path, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate fallback scores based on modification time and patterns"""
        fallback_files = []
        
        try:
            # Get recently modified Python files
            for py_file in project_dir.rglob('*.py'):
                if any(skip in str(py_file) for skip in ['__pycache__', '.venv', 'site-packages']):
                    continue
                
                try:
                    mtime = py_file.stat().st_mtime
                    age_days = (time.time() - mtime) / (24 * 3600)
                    
                    # Simple scoring based on recency and name patterns
                    recency_score = max(0, 1 - (age_days / 30))  # Decay over 30 days
                    
                    # Pattern-based scoring
                    name_lower = py_file.stem.lower()
                    pattern_score = 0.3  # Base score
                    
                    critical_patterns = {
                        'main': 0.9, 'core': 0.8, 'api': 0.8,
                        'auth': 0.9, 'security': 0.9, 'payment': 0.9,
                        'neural': 0.7, 'mcp': 0.7, 'hook': 0.6,
                        'service': 0.6, 'model': 0.7
                    }
                    
                    for pattern, weight in critical_patterns.items():
                        if pattern in name_lower:
                            pattern_score = max(pattern_score, weight)
                            break
                    
                    # Combined score
                    combined_score = (recency_score * 0.6) + (pattern_score * 0.4)
                    
                    fallback_files.append({
                        'file_path': str(py_file),
                        'prism_score': combined_score,
                        'prism_components': {
                            'recency': recency_score,
                            'pattern': pattern_score,
                            'complexity': 0.5,  # Unknown
                            'dependencies': 0.5  # Unknown
                        },
                        'importance_label': PrismCache._score_to_label(None, combined_score),
                        'fallback': True
                    })
                    
                except (OSError, ValueError):
                    continue
            
            # Sort by score and return top N
            fallback_files.sort(key=lambda x: x['prism_score'], reverse=True)
            return fallback_files[:limit]
            
        except Exception as e:
            logging.getLogger("l9_prism_fallback").error(f"Fallback scoring failed: {e}")
            return []