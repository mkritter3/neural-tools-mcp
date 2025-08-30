"""
L9 Hook Utilities - Shared Functions
Centralizes all common functionality used across hooks
"""

from pathlib import Path
from typing import Dict, List, Any
import json
import os


def estimate_tokens(text: str) -> int:
    """
    Consistent token estimation across all hooks
    Uses 1 token ≈ 4 characters heuristic
    """
    if not text:
        return 0
    return len(text) // 4


def format_context(data: Dict[str, Any], max_tokens: int = 1000) -> str:
    """
    Format context data with token limits
    Used by multiple hooks for consistent output
    """
    if not data:
        return ""
    
    # Convert to JSON with proper formatting
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    # Truncate if needed
    if estimate_tokens(json_str) > max_tokens:
        # Truncate and add indicator
        target_chars = max_tokens * 4
        truncated = json_str[:target_chars] + "...[truncated]"
        return truncated
    
    return json_str


def read_file_safely(file_path: Path, max_size: int = 100000) -> str:
    """
    Safely read files with size limits and encoding handling
    Used across hooks for consistent file reading
    """
    try:
        if not file_path.exists():
            return ""
        
        # Check file size
        if file_path.stat().st_size > max_size:
            return f"[File too large: {file_path.stat().st_size} bytes]"
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        return "[Unable to decode file]"
        
    except Exception as e:
        return f"[Error reading file: {e}]"


def find_files_by_pattern(root_dir: Path, pattern: str, exclude_dirs: List[str] = None) -> List[Path]:
    """
    Find files by glob pattern with consistent exclusions
    Used by hooks that need file discovery
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', '.venv', 'node_modules', 'site-packages']
    
    try:
        files = []
        for file_path in root_dir.rglob(pattern):
            # Skip if in excluded directory
            if any(excluded in str(file_path) for excluded in exclude_dirs):
                continue
            
            if file_path.is_file():
                files.append(file_path)
        
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        
    except Exception:
        return []


def get_project_metadata(project_dir: Path) -> Dict[str, Any]:
    """
    Extract consistent project metadata
    Used by hooks that need project understanding
    """
    metadata = {
        "name": project_dir.name,
        "path": str(project_dir),
        "total_files": 0,
        "main_languages": [],
        "has_docker": False,
        "has_package_json": False,
        "has_requirements": False
    }
    
    try:
        # Count files
        py_files = len(list(project_dir.rglob('*.py')))
        js_files = len(list(project_dir.rglob('*.js'))) + len(list(project_dir.rglob('*.ts')))
        
        metadata["total_files"] = py_files + js_files
        
        if py_files > 0:
            metadata["main_languages"].append("Python")
        if js_files > 0:
            metadata["main_languages"].append("JavaScript/TypeScript")
        
        # Check for key files
        metadata["has_docker"] = (project_dir / "Dockerfile").exists()
        metadata["has_package_json"] = (project_dir / "package.json").exists()
        metadata["has_requirements"] = (project_dir / "requirements.txt").exists()
        
    except Exception:
        pass
    
    return metadata


def validate_output_format(output: Any) -> Dict[str, Any]:
    """
    Validate and normalize hook output format
    Ensures all hooks return consistent structure
    """
    if isinstance(output, dict):
        # Ensure required fields
        output.setdefault("status", "success")
        output.setdefault("tokens_used", 0)
        output.setdefault("timestamp", None)
        return output
    
    elif isinstance(output, str):
        # Convert string output to dict
        return {
            "status": "success",
            "content": output,
            "tokens_used": estimate_tokens(output)
        }
    
    else:
        # Handle other types
        return {
            "status": "success", 
            "content": str(output),
            "tokens_used": estimate_tokens(str(output))
        }