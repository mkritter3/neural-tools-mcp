"""
L9 Fallback Classes - Consistent Fallback Implementations
Provides systematic fallback behavior when dependencies are unavailable
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any


class SimplePrismScorer:
    """
    Simplified PRISM scorer fallback when full implementation unavailable
    Provides basic scoring to maintain hook functionality
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def get_top_files(self, n: int = 10, file_pattern: str = '**/*.py') -> List[Tuple[str, float]]:
        """
        Simple file discovery with basic scoring
        Returns most recently modified files with default scores
        """
        try:
            files = []
            
            # Find matching files
            for file_path in self.project_root.glob(file_pattern):
                if not file_path.is_file():
                    continue
                
                # Skip common directories to exclude
                if any(skip in str(file_path) for skip in [
                    '__pycache__', '.git', '.venv', 'node_modules', 
                    'site-packages', '.pytest_cache'
                ]):
                    continue
                
                # Basic scoring based on file patterns
                score = self._calculate_basic_score(file_path)
                files.append((str(file_path), score))
            
            # Sort by score (descending) and return top n
            files.sort(key=lambda x: x[1], reverse=True)
            return files[:n]
            
        except Exception:
            # Fallback to empty list if any error
            return []
    
    def _calculate_basic_score(self, file_path: Path) -> float:
        """Calculate basic importance score based on simple heuristics"""
        score = 0.3  # Base score
        
        file_name = file_path.stem.lower()
        
        # Higher scores for common important patterns
        importance_patterns = {
            'main': 0.9,
            'app': 0.8,
            'server': 0.8,
            'api': 0.7,
            'core': 0.7,
            'config': 0.6,
            'utils': 0.4,
            'helper': 0.4,
            'test': 0.2,
        }
        
        for pattern, pattern_score in importance_patterns.items():
            if pattern in file_name:
                score = max(score, pattern_score)
        
        # Boost for recent modifications
        try:
            import time
            mtime = file_path.stat().st_mtime
            age_days = (time.time() - mtime) / (24 * 3600)
            
            if age_days < 1:
                score += 0.2  # Recent files are more important
            elif age_days < 7:
                score += 0.1  # This week
                
        except Exception:
            pass
        
        return min(score, 1.0)  # Cap at 1.0
    
    def boost_search_results(self, results: List[Dict[str, Any]], boost_factor: float = 0.3) -> List[Dict[str, Any]]:
        """
        Simple search result boosting fallback
        Provides basic reordering without complex scoring
        """
        try:
            # Add basic importance scores to results
            for result in results:
                file_path = result.get('file_path', '')
                if file_path:
                    basic_score = self._calculate_basic_score(Path(file_path))
                    original_score = result.get('score', 0.5)
                    
                    # Simple linear blend
                    combined_score = (1 - boost_factor) * original_score + boost_factor * basic_score
                    
                    result['score'] = combined_score
                    result['original_score'] = original_score
                    result['prism_score'] = basic_score
                    result['prism_components'] = {
                        'complexity': 0.0,
                        'dependencies': 0.0,
                        'recency': basic_score * 0.5,  # Approximation
                        'contextual': basic_score * 0.5  # Approximation
                    }
            
            # Re-sort by combined score
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            return results
            
        except Exception:
            # If anything fails, return original results
            return results


class SimpleMemoryExtractor:
    """
    Simple memory extraction fallback when full implementation unavailable
    Provides basic context extraction to maintain functionality
    """
    
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
    
    def extract_conversation_context(self, conversation_text: str) -> Dict[str, Any]:
        """Extract basic context from conversation"""
        try:
            from .utilities import estimate_tokens
            
            # Basic extraction - just summarize length and basic patterns
            token_count = estimate_tokens(conversation_text)
            
            # Look for common patterns
            has_code = any(pattern in conversation_text.lower() for pattern in [
                'def ', 'class ', 'import ', 'function', 'method'
            ])
            
            has_files = '.py' in conversation_text or '.js' in conversation_text
            
            return {
                "summary": f"Conversation with {token_count} tokens",
                "has_code": has_code,
                "has_files": has_files,
                "token_count": token_count,
                "extraction_method": "simple_fallback"
            }
            
        except Exception:
            return {"summary": "Basic context extraction", "extraction_method": "minimal_fallback"}