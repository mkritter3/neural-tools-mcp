#!/usr/bin/env python3
"""
L9 Style Preserver - Post-tool execution style validation
Ensures AI maintains project coding patterns and conventions
"""

import sys
import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging  
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ProjectStyleAnalyzer:
    """Analyzes and preserves project coding style"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.style_patterns = {}
        
    def analyze_file_style(self, file_path: str) -> Dict[str, Any]:
        """Analyze the style of a specific file"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return {}
            
        file_ext = Path(file_path).suffix.lower()
        
        style_analysis = {
            'indentation': self._analyze_indentation(content),
            'line_endings': self._analyze_line_endings(content),
            'quotes': self._analyze_quote_style(content, file_ext),
            'naming_convention': self._analyze_naming_convention(content, file_ext),
            'imports': self._analyze_import_style(content, file_ext),
            'comments': self._analyze_comment_style(content, file_ext)
        }
        
        return style_analysis
        
    def _analyze_indentation(self, content: str) -> Dict[str, Any]:
        """Analyze indentation patterns"""
        lines = content.split('\n')
        
        space_indented = 0
        tab_indented = 0
        indent_sizes = []
        
        for line in lines:
            if line.strip():  # Non-empty line
                leading_spaces = len(line) - len(line.lstrip(' '))
                leading_tabs = len(line) - len(line.lstrip('\t'))
                
                if leading_tabs > 0:
                    tab_indented += 1
                elif leading_spaces > 0:
                    space_indented += 1
                    if leading_spaces % 2 == 0:
                        indent_sizes.append(leading_spaces)
                        
        # Determine most common indent size
        if indent_sizes:
            most_common_size = max(set(indent_sizes), key=indent_sizes.count)
        else:
            most_common_size = 4  # Default
            
        return {
            'type': 'tabs' if tab_indented > space_indented else 'spaces',
            'size': most_common_size if space_indented > tab_indented else 1,
            'consistency': (space_indented + tab_indented) > 0
        }
        
    def _analyze_line_endings(self, content: str) -> Dict[str, Any]:
        """Analyze line ending patterns"""
        crlf_count = content.count('\r\n')
        lf_count = content.count('\n') - crlf_count
        
        return {
            'type': 'CRLF' if crlf_count > lf_count else 'LF',
            'mixed': crlf_count > 0 and lf_count > 0
        }
        
    def _analyze_quote_style(self, content: str, file_ext: str) -> Dict[str, Any]:
        """Analyze quote style preferences"""
        if file_ext not in ['.py', '.js', '.ts', '.jsx', '.tsx']:
            return {}
            
        single_quotes = len(re.findall(r"'[^']*'", content))
        double_quotes = len(re.findall(r'"[^"]*"', content))
        
        return {
            'preference': 'single' if single_quotes > double_quotes else 'double',
            'single_count': single_quotes,
            'double_count': double_quotes
        }
        
    def _analyze_naming_convention(self, content: str, file_ext: str) -> Dict[str, Any]:
        """Analyze naming conventions"""
        conventions = {
            'functions': [],
            'variables': [],
            'classes': []
        }
        
        if file_ext == '.py':
            # Python naming patterns
            functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            variables = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', content)
            
            conventions['functions'] = functions
            conventions['classes'] = classes
            conventions['variables'] = variables
            
        elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
            # JavaScript/TypeScript naming patterns
            functions = re.findall(r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            functions.extend(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:function|\()', content))
            classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
            
            conventions['functions'] = functions
            conventions['classes'] = classes
            
        return conventions
        
    def _analyze_import_style(self, content: str, file_ext: str) -> Dict[str, Any]:
        """Analyze import/require patterns"""
        imports = {
            'style': 'unknown',
            'patterns': []
        }
        
        if file_ext == '.py':
            # Python imports
            import_lines = re.findall(r'^(import\s+.+|from\s+.+\s+import\s+.+)$', content, re.MULTILINE)
            imports['patterns'] = import_lines
            imports['style'] = 'python'
            
        elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
            # JavaScript imports
            import_lines = re.findall(r'^(import\s+.+|const\s+.+\s*=\s*require\(.+\))$', content, re.MULTILINE)
            imports['patterns'] = import_lines
            imports['style'] = 'javascript'
            
        return imports
        
    def _analyze_comment_style(self, content: str, file_ext: str) -> Dict[str, Any]:
        """Analyze comment patterns"""
        comments = {
            'single_line': [],
            'multi_line': [],
            'docstrings': []
        }
        
        if file_ext == '.py':
            # Python comments
            single_line = re.findall(r'#.*$', content, re.MULTILINE)
            docstrings = re.findall(r'""".*?"""', content, re.DOTALL)
            docstrings.extend(re.findall(r"'''.*?'''", content, re.DOTALL))
            
            comments['single_line'] = single_line
            comments['docstrings'] = docstrings
            
        elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
            # JavaScript comments
            single_line = re.findall(r'//.*$', content, re.MULTILINE)
            multi_line = re.findall(r'/\*.*?\*/', content, re.DOTALL)
            
            comments['single_line'] = single_line
            comments['multi_line'] = multi_line
            
        return comments

def validate_style_consistency(file_path: str, project_path: str) -> bool:
    """Validate that a file maintains project style consistency"""
    
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return True
        
    analyzer = ProjectStyleAnalyzer(project_path)
    
    try:
        # Analyze the modified file
        file_style = analyzer.analyze_file_style(file_path)
        
        # Basic style validations
        indentation = file_style.get('indentation', {})
        
        # Check for consistent indentation
        if indentation.get('type') and not indentation.get('consistency', True):
            logger.warning(f"🎨 STYLE: Inconsistent indentation in {file_path}")
            return False
            
        # Check for mixed line endings
        line_endings = file_style.get('line_endings', {})
        if line_endings.get('mixed', False):
            logger.warning(f"🎨 STYLE: Mixed line endings in {file_path}")
            return False
            
        logger.info(f"✅ STYLE: Style consistency validated for {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ STYLE: Error analyzing {file_path}: {e}")
        return True  # Don't block on analysis errors

def main():
    """Main style preserver entry point"""
    
    # Get the file that was just modified
    if len(sys.argv) > 1:
        # File path passed as argument
        file_path = sys.argv[1]
    else:
        # Check environment variables
        file_path = os.getenv('CLAUDE_MODIFIED_FILE', '')
        
    if not file_path:
        logger.info("✅ STYLE: No file modification detected")
        return 0
        
    # Get project root
    project_path = os.getenv('PWD', os.getcwd())
    
    # Validate style consistency
    is_consistent = validate_style_consistency(file_path, project_path)
    
    if not is_consistent:
        logger.warning(f"⚠️ STYLE: Style inconsistencies detected in {file_path}")
        logger.warning("Consider reviewing the changes to maintain project conventions")
        # Don't block - just warn
        return 0
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)