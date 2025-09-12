#!/usr/bin/env python3
"""
Pattern-based metadata extraction for code files
Fast regex-based extraction (<1ms per file)
"""

import re
from typing import Dict

class PatternExtractor:
    """Fast regex-based pattern extraction for code metadata"""
    
    def __init__(self):
        # Pre-compile patterns for performance
        self.patterns = {
            'todo': re.compile(r'#\s*TODO|//\s*TODO|\*\s*TODO', re.IGNORECASE),
            'fixme': re.compile(r'#\s*FIXME|//\s*FIXME|\*\s*FIXME', re.IGNORECASE),
            'deprecated': re.compile(r'@deprecated|#\s*deprecated|//\s*deprecated', re.IGNORECASE),
            'test': re.compile(r'def test_|class Test|@test|@pytest', re.IGNORECASE),
            'security': re.compile(r'password|secret|token|api_key|private_key|auth', re.IGNORECASE),
            'async': re.compile(r'async def|await |asyncio'),
            'type_hints': re.compile(r'->\s*\w+|:\s*\w+\['),
            'canon': re.compile(r'@canon|@authority|@source-of-truth', re.IGNORECASE),
            'experimental': re.compile(r'@experimental|@beta|@alpha', re.IGNORECASE),
        }
    
    def extract(self, content: str) -> Dict[str, int]:
        """Extract pattern counts from content"""
        return {
            'todo_count': len(self.patterns['todo'].findall(content)),
            'fixme_count': len(self.patterns['fixme'].findall(content)),
            'deprecated_count': len(self.patterns['deprecated'].findall(content)),
            'test_count': len(self.patterns['test'].findall(content)),
            'security_count': len(self.patterns['security'].findall(content)),
            'is_async': bool(self.patterns['async'].search(content)),
            'has_type_hints': bool(self.patterns['type_hints'].search(content)),
            'canon_markers': len(self.patterns['canon'].findall(content)),
            'experimental_markers': len(self.patterns['experimental'].findall(content)),
        }
    
    def extract_for_chunk(self, chunk_text: str) -> Dict[str, bool]:
        """Extract patterns for a specific chunk (lighter weight)"""
        return {
            'chunk_has_todo': self.patterns['todo'].search(chunk_text) is not None,
            'chunk_has_fixme': self.patterns['fixme'].search(chunk_text) is not None,
            'chunk_has_deprecated': self.patterns['deprecated'].search(chunk_text) is not None,
            'chunk_has_canon': self.patterns['canon'].search(chunk_text) is not None,
            'chunk_has_security': self.patterns['security'].search(chunk_text) is not None,
        }